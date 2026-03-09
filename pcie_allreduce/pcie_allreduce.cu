// PCIe-only custom allreduce extension.
// Optimized for CPU root complex topologies (no PCIe switch, no NVLink).
//
// Two fused kernel modes (eager double-buffer path):
//   1. Push+reduce: SM writes to all peers' IPC buffers (posted PCIe TLPs),
//      then reduces from own buffer (local HBM reads). Fastest when IPC
//      buffer is large enough for partitioned layout (msg_size * world_size).
//   2. Fused pull+reduce: SM copies input to own IPC buffer, barrier, then
//      reads peers. Saves one host API call vs legacy 2-step path.
//
// Legacy kernel retained for CUDA graph capture and register_buffer path.
// Self-contained: no sgl-kernel headers needed.

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/extension.h>

#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

#define CHECK_CUDA_SUCCESS(cmd)                                         \
  do {                                                                  \
    cudaError_t e = cmd;                                                \
    if (e != cudaSuccess) {                                             \
      std::stringstream _message;                                       \
      auto s = cudaGetErrorString(e);                                   \
      _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__; \
      throw std::runtime_error(_message.str());                         \
    }                                                                   \
  } while (0)

namespace pcie_allreduce {

// ---- Data structures ----

constexpr int kMaxBlocks = 36;
using FlagType = uint32_t;

// 128B stride per rank to avoid PCIe false sharing (one GPU L2 cache line).
constexpr int kFlagStride = 32;  // 32 * sizeof(uint32_t) = 128 bytes

struct Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][8];
  // 16 slots: 0..7 for start flags, 8..15 for end flags (streaming kernel).
  alignas(128) FlagType peer_counter[2][kMaxBlocks][16 * kFlagStride];
};

struct __align__(16) RankData {
  const void* __restrict__ ptrs[8];
};

struct __align__(16) RankSignals {
  Signal* signals[8];
};

template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <typename T>
struct packed_t {
  using P = array_t<T, 16 / sizeof(T)>;
  using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__

// ---- Scalar ops ----

DINLINE float upcast_s(half val) { return __half2float(val); }

template <typename T>
DINLINE T downcast_s(float val);
template <>
DINLINE half downcast_s(float val) { return __float2half(val); }

DINLINE half& assign_add(half& a, half b) { a = __hadd(a, b); return a; }
DINLINE float& assign_add(float& a, float b) { return a += b; }

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
DINLINE float upcast_s(nv_bfloat16 val) { return __bfloat162float(val); }
template <>
DINLINE nv_bfloat16 downcast_s(float val) { return __float2bfloat16(val); }
DINLINE nv_bfloat16& assign_add(nv_bfloat16& a, nv_bfloat16 b) { a = __hadd(a, b); return a; }
#endif

template <typename T, int N>
DINLINE array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; i++) assign_add(a.data[i], b.data[i]);
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; i++) out.data[i] = upcast_s(val.data[i]);
    return out;
  }
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return val;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; i++) out.data[i] = downcast_s<typename O::type>(val.data[i]);
    return out;
  }
}

// ---- Flag operations (system-scope relaxed for PCIe) ----

static DINLINE void st_flag_relaxed(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.relaxed.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_relaxed(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

// ---- System-scope barrier for PCIe ----
// Uses __threadfence_system() + relaxed atomics (DeepEP pattern).
// The fence flushes all prior writes system-wide, then relaxed ops
// handle signaling without per-operation ordering overhead.
template <int ngpus, bool is_start>
DINLINE void multi_gpu_barrier(const RankSignals& sg, Signal* self_sg, int rank) {
  if constexpr (!is_start) __syncthreads();
  if (threadIdx.x < ngpus) {
    __threadfence_system();
    auto val = self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->peer_counter[val % 2][blockIdx.x][rank * kFlagStride];
    auto self_counter_ptr = &self_sg->peer_counter[val % 2][blockIdx.x][threadIdx.x * kFlagStride];
    st_flag_relaxed(peer_counter_ptr, val);
    while (ld_flag_relaxed(self_counter_ptr) != val);
  }
  __syncthreads();
}

// ---- Reduction ----

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) packed_assign_add(tmp, upcast(ptrs[i][idx]));
  return downcast<P>(tmp);
}

// ---- Fused push+reduce kernel (optimal for PCIe) ----
//
// Each GPU WRITES its data to every peer's IPC buffer (posted PCIe TLPs,
// fire-and-forget) and then reduces from its OWN buffer (local HBM reads).
//
// The IPC buffer is partitioned: slot k holds rank k's data at offset
// k * packed_size. After barrier, each GPU reads slots 0..ngpus-1 from
// its own buffer — all local reads.
//
// Single block: for decode-sized messages (14KB), one block of 512 threads
// gives ~2 packed elements per thread which is plenty.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) fused_push_reduce_kernel(
    RankData* _dp, RankSignals sg, Signal* self_sg,
    const T* __restrict__ input, T* __restrict__ result,
    int rank, int packed_size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto dp = *_dp;

  // Phase 1: Push — write our input to slot[rank] of EVERY GPU's buffer.
  // Writes to peers go via BAR1 as posted PCIe TLPs (fire-and-forget).
  // Write to own buffer is a local HBM store.
  for (int idx = threadIdx.x; idx < packed_size; idx += blockDim.x) {
    P val = ((const P*)input)[idx];
#pragma unroll
    for (int g = 0; g < ngpus; g++) {
      ((P*)dp.ptrs[g])[rank * packed_size + idx] = val;
    }
  }

  // Ensure all threads finished pushing before entering barrier.
  __syncthreads();

  // Phase 2: Barrier — __threadfence_system() flushes all posted writes
  // system-wide, then flag exchange confirms all peers are done.
  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);

  // Phase 3: Local reduce — read all slots from OUR buffer.
  // All reads are local HBM (~3 TB/s), zero PCIe traffic.
  const P* my_buf = (const P*)dp.ptrs[rank];
  for (int idx = threadIdx.x; idx < packed_size; idx += blockDim.x) {
    A acc = upcast(my_buf[0 * packed_size + idx]);
#pragma unroll
    for (int k = 1; k < ngpus; k++) {
      packed_assign_add(acc, upcast(my_buf[k * packed_size + idx]));
    }
    ((P*)result)[idx] = downcast<P>(acc);
  }
}

// ---- Fused pull+reduce kernel ----
//
// Single kernel that copies input to IPC buffer (replacing cudaMemcpyAsync),
// does barrier, then reads peers and reduces. Saves one host API call (~5μs).
// Used when push mode can't fit (msg too large for partitioned buffer).
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) fused_pull_reduce_kernel(
    RankData* _dp, RankSignals sg, Signal* self_sg,
    const T* __restrict__ input, T* __restrict__ result,
    int rank, int packed_size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto dp = *_dp;

  // Phase 1: Copy input to our IPC buffer (local D2D, replaces cudaMemcpyAsync).
  P* my_buf = (P*)dp.ptrs[rank];
  for (int idx = threadIdx.x; idx < packed_size; idx += blockDim.x) {
    my_buf[idx] = ((const P*)input)[idx];
  }

  // Ensure copy complete before barrier.
  __syncthreads();

  // Phase 2: Barrier.
  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);

  // Phase 3: Reduce — read from all peers (staggered to spread PCIe traffic).
  const P* rotated[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    rotated[i] = (const P*)dp.ptrs[(rank + i) % ngpus];
  }
  for (int idx = threadIdx.x; idx < packed_size; idx += blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>(rotated, idx);
  }
}

// ---- Legacy PCIe kernel (multi-block, for graph capture and fallback) ----

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) pcie_allreduce_kernel(
    RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto dp = *_dp;
  const P* rotated[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    rotated[i] = (const P*)dp.ptrs[(rank + i) % ngpus];
  }
  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>(rotated, idx);
  }
}

// ---- IPC key ----

using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;

// ---- PCIeAllreduce class ----

class PCIeAllreduce {
 public:
  int rank_;
  int world_size_;

  RankSignals sg_;
  std::unordered_map<void*, RankData*> buffers_;
  Signal* self_sg_;

  RankData *d_rank_data_base_, *d_rank_data_end_;
  std::vector<void*> graph_unreg_buffers_;
  std::map<IPC_KEY, char*> ipc_handles_;

  // Double-buffer state (eliminates end barrier).
  bool dbuf_enabled_ = false;
  int dbuf_slot_ = 0;
  void* dbuf_raw_[2][8] = {};
  RankData* dbuf_rd_[2] = {};

  // Buffer capacity for push mode partitioning.
  size_t dbuf_capacity_ = 0;

  PCIeAllreduce(
      Signal** signals, void* rank_data, size_t rank_data_sz, int rank, int world_size)
      : rank_(rank),
        world_size_(world_size),
        self_sg_(signals[rank]),
        d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
        d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
    for (int i = 0; i < world_size_; i++) sg_.signals[i] = signals[i];
  }

  char* open_ipc_handle(const void* ipc_handle) {
    auto [it, new_handle] = ipc_handles_.insert({*((IPC_KEY*)ipc_handle), nullptr});
    if (new_handle) {
      char* ipc_ptr;
      CHECK_CUDA_SUCCESS(cudaIpcOpenMemHandle(
          (void**)&ipc_ptr, *((const cudaIpcMemHandle_t*)ipc_handle), cudaIpcMemLazyEnablePeerAccess));
      it->second = ipc_ptr;
    }
    return it->second;
  }

  std::pair<std::string, std::vector<int64_t>> get_graph_buffer_ipc_meta() {
    auto num_buffers = graph_unreg_buffers_.size();
    auto handle_sz = sizeof(cudaIpcMemHandle_t);
    std::string handles(handle_sz * num_buffers, static_cast<char>(0));
    std::vector<int64_t> offsets(num_buffers);
    for (size_t i = 0; i < num_buffers; i++) {
      auto ptr = graph_unreg_buffers_[i];
      void* base_ptr;
      if (cuPointerGetAttribute(&base_ptr, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, (CUdeviceptr)ptr) != CUDA_SUCCESS)
        throw std::runtime_error("failed to get pointer attr");
      CHECK_CUDA_SUCCESS(cudaIpcGetMemHandle((cudaIpcMemHandle_t*)&handles[i * handle_sz], base_ptr));
      offsets[i] = ((char*)ptr) - ((char*)base_ptr);
    }
    return std::make_pair(handles, offsets);
  }

  void check_rank_data_capacity(size_t num = 1) {
    if (d_rank_data_base_ + num > d_rank_data_end_)
      throw std::runtime_error("Rank data buffer overflow");
  }

  void register_pcie_buffers(void** ptrs0, void** ptrs1, size_t buf_capacity = 0) {
    check_rank_data_capacity(2);
    for (int s = 0; s < 2; s++) {
      void** ptrs = s == 0 ? ptrs0 : ptrs1;
      RankData data;
      for (int i = 0; i < world_size_; i++) {
        data.ptrs[i] = ptrs[i];
        dbuf_raw_[s][i] = ptrs[i];
      }
      dbuf_rd_[s] = d_rank_data_base_++;
      CHECK_CUDA_SUCCESS(cudaMemcpy(dbuf_rd_[s], &data, sizeof(RankData), cudaMemcpyHostToDevice));
    }
    buffers_[ptrs0[rank_]] = dbuf_rd_[0];
    buffers_[ptrs1[rank_]] = dbuf_rd_[1];
    dbuf_enabled_ = true;
    dbuf_slot_ = 0;
    dbuf_capacity_ = buf_capacity;
  }

  void register_buffer(void** ptrs) {
    check_rank_data_capacity();
    RankData data;
    for (int i = 0; i < world_size_; i++) data.ptrs[i] = ptrs[i];
    auto d_data = d_rank_data_base_++;
    CHECK_CUDA_SUCCESS(cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice));
    buffers_[ptrs[rank_]] = d_data;
  }

  void register_graph_buffers(
      const std::vector<std::string>& handles, const std::vector<std::vector<int64_t>>& offsets) {
    auto num_buffers = graph_unreg_buffers_.size();
    check_rank_data_capacity(num_buffers);
    std::vector<RankData> rank_data(num_buffers);
    for (size_t i = 0; i < num_buffers; i++) {
      auto self_ptr = graph_unreg_buffers_[i];
      auto& rd = rank_data[i];
      for (int j = 0; j < world_size_; j++) {
        if (j != rank_) {
          char* handle = open_ipc_handle(&handles[j][i * sizeof(cudaIpcMemHandle_t)]);
          handle += offsets[j][i];
          rd.ptrs[j] = handle;
        } else {
          rd.ptrs[j] = self_ptr;
        }
      }
    }
    CHECK_CUDA_SUCCESS(
        cudaMemcpy(d_rank_data_base_, rank_data.data(), sizeof(RankData) * num_buffers, cudaMemcpyHostToDevice));
    d_rank_data_base_ += num_buffers;
    graph_unreg_buffers_.clear();
  }

  template <typename T>
  void allreduce(cudaStream_t stream, T* input, T* output, int size, int threads = 512, int block_limit = 36) {
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error("allreduce requires input length to be multiple of " + std::to_string(d));
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("max supported block limit is " + std::to_string(kMaxBlocks));

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CHECK_CUDA_SUCCESS(cudaStreamIsCapturing(stream, &status));

    if (dbuf_enabled_ && status != cudaStreamCaptureStatusActive) {
      int slot = dbuf_slot_ % 2;
      dbuf_slot_++;
      auto input_size_bytes = (size_t)size * sizeof(T);
      int packed_size = size / d;
      ptrs = dbuf_rd_[slot];

      // Check if push model fits: need world_size slots of msg_size in buffer.
      bool can_push = (dbuf_capacity_ > 0) &&
                      (input_size_bytes * world_size_ <= dbuf_capacity_);

      if (can_push) {
        // Push+reduce: posted PCIe writes + local HBM reads. Fastest path.
#define PUSH_KL(ngpus) fused_push_reduce_kernel<T, ngpus><<<1, threads, 0, stream>>>( \
    ptrs, sg_, self_sg_, input, output, rank_, packed_size);
        switch (world_size_) {
          case 2: PUSH_KL(2); break;
          case 4: PUSH_KL(4); break;
          case 6: PUSH_KL(6); break;
          case 8: PUSH_KL(8); break;
          default: throw std::runtime_error("unsupported world size");
        }
#undef PUSH_KL
      } else {
        // Fused pull: SM copy + barrier + pull-reduce. Saves 1 host API call.
#define PULL_KL(ngpus) fused_pull_reduce_kernel<T, ngpus><<<1, threads, 0, stream>>>( \
    ptrs, sg_, self_sg_, input, output, rank_, packed_size);
        switch (world_size_) {
          case 2: PULL_KL(2); break;
          case 4: PULL_KL(4); break;
          case 6: PULL_KL(6); break;
          case 8: PULL_KL(8); break;
          default: throw std::runtime_error("unsupported world size");
        }
#undef PULL_KL
      }
      return;
    }

    // Graph capture path or non-dbuf path: use legacy kernel.
    if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input);
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error("buffer address " + std::to_string(reinterpret_cast<uint64_t>(input)) +
                                 " is not registered!");
      ptrs = it->second;
    }

    size /= d;
    int blocks = std::min(block_limit, (size + threads - 1) / threads);
    blocks = std::max(blocks, 1);

#define KL(ngpus) pcie_allreduce_kernel<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, size);
    switch (world_size_) {
      case 2: KL(2); break;
      case 4: KL(4); break;
      case 6: KL(6); break;
      case 8: KL(8); break;
      default:
        throw std::runtime_error("only supports (2,4,6,8) gpus, got " + std::to_string(world_size_));
    }
#undef KL
  }

  ~PCIeAllreduce() {
    for (auto [_, ptr] : ipc_handles_) CHECK_CUDA_SUCCESS(cudaIpcCloseMemHandle(ptr));
  }
};

}  // namespace pcie_allreduce

// ---- Python bindings ----

using fptr_t = int64_t;

static fptr_t
init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs, torch::Tensor& rank_data, int64_t rank) {
  int world_size = fake_ipc_ptrs.size();
  if (world_size > 8) throw std::invalid_argument("world size > 8 is not supported");
  if (world_size % 2 != 0) throw std::invalid_argument("Odd num gpus is not supported");
  if (rank < 0 || rank >= world_size) throw std::invalid_argument("invalid rank");

  pcie_allreduce::Signal* ipc_ptrs[8];
  for (int i = 0; i < world_size; i++)
    ipc_ptrs[i] = reinterpret_cast<pcie_allreduce::Signal*>(fake_ipc_ptrs[i]);
  return (fptr_t) new pcie_allreduce::PCIeAllreduce(
      ipc_ptrs, rank_data.data_ptr(), rank_data.numel(), rank, world_size);
}

static bool _is_weak_contiguous(torch::Tensor& t) {
  return t.is_contiguous() ||
         (t.storage().nbytes() - t.storage_offset() * t.element_size() == t.numel() * t.element_size());
}

static void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out, fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();

  // When double-buffer is active, allreduce handles the memcpy and slot
  // alternation internally — pass input directly.
  void* reg_buffer;
  if (fa->dbuf_enabled_) {
    reg_buffer = inp.data_ptr();
  } else if (_reg_buffer) {
    reg_buffer = reinterpret_cast<void*>(_reg_buffer);
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size, cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.data_ptr();
  }
  switch (out.scalar_type()) {
    case at::ScalarType::Float:
      fa->allreduce<float>(stream, (float*)reg_buffer, (float*)out.data_ptr(), out.numel());
      break;
    case at::ScalarType::Half:
      fa->allreduce<half>(stream, (half*)reg_buffer, (half*)out.data_ptr(), out.numel());
      break;
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16:
      fa->allreduce<nv_bfloat16>(stream, (nv_bfloat16*)reg_buffer, (nv_bfloat16*)out.data_ptr(), out.numel());
      break;
#endif
    default:
      throw std::runtime_error("only supports float32, float16 and bfloat16");
  }
}

static void dispose(fptr_t _fa) {
  delete reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
}

static int64_t meta_size() {
  return sizeof(pcie_allreduce::Signal);
}

static void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  TORCH_CHECK(fake_ipc_ptrs.size() == (size_t)fa->world_size_);
  void* ipc_ptrs[8];
  for (size_t i = 0; i < fake_ipc_ptrs.size(); i++)
    ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
  fa->register_buffer(ipc_ptrs);
}

static void register_pcie_buffers(
    fptr_t _fa, const std::vector<fptr_t>& ptrs0, const std::vector<fptr_t>& ptrs1, int64_t buf_capacity = 0) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  TORCH_CHECK(ptrs0.size() == (size_t)fa->world_size_);
  TORCH_CHECK(ptrs1.size() == (size_t)fa->world_size_);
  void* p0[8], *p1[8];
  for (size_t i = 0; i < ptrs0.size(); i++) {
    p0[i] = reinterpret_cast<void*>(ptrs0[i]);
    p1[i] = reinterpret_cast<void*>(ptrs1[i]);
  }
  fa->register_pcie_buffers(p0, p1, (size_t)buf_capacity);
}

static std::tuple<std::vector<int64_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(fptr_t _fa) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  auto [handle, offsets] = fa->get_graph_buffer_ipc_meta();
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offsets);
}

static void register_graph_buffers(
    fptr_t _fa, const std::vector<std::vector<int64_t>>& handles, const std::vector<std::vector<int64_t>>& offsets) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  std::vector<std::string> bytes;
  bytes.reserve(handles.size());
  for (size_t i = 0; i < handles.size(); i++)
    bytes.emplace_back(handles[i].begin(), handles[i].end());
  fa->register_graph_buffers(bytes, offsets);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_custom_ar", &init_custom_ar, "init PCIe allreduce");
  m.def("all_reduce", &all_reduce, "PCIe allreduce");
  m.def("dispose", &dispose, "dispose PCIe allreduce");
  m.def("meta_size", &meta_size, "signal metadata size");
  m.def("register_buffer", &register_buffer, "register IPC buffer");
  m.def("register_pcie_buffers", &register_pcie_buffers, "register double-buffered IPC buffers",
        pybind11::arg("fa"), pybind11::arg("ptrs0"), pybind11::arg("ptrs1"), pybind11::arg("buf_capacity") = 0);
  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta, "get graph buffer IPC meta");
  m.def("register_graph_buffers", &register_graph_buffers, "register graph buffers");
}
