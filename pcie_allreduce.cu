// PCIe-only custom allreduce extension.
// DMA-gather variant: uses Copy Engine to gather peer data to local scratch,
// then reduces from local HBM only. Avoids SM-initiated PCIe reads which
// are slow on CPU root complex (no PCIe switch) topologies.
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
constexpr int kMaxGpus = 8;
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

// ---- Barrier-only kernel (1 block) ----
// Launched between input memcpy and DMA gather to ensure all peers
// have finished writing their input data.
template <int ngpus>
__global__ void barrier_kernel(RankSignals sg, Signal* self_sg, int rank) {
  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);
}

// ---- Local reduce kernel ----
// Reads only from local GPU memory: own input buffer + DMA-gathered scratch copies.
// scratch_ptrs contains (ngpus-1) pointers to local scratch buffers holding peer data.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) local_reduce_kernel(
    const void* __restrict__ self_input,
    const void* const* __restrict__ scratch_ptrs,
    T* __restrict__ result,
    int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int n_peers = ngpus - 1;

  // Build local pointer array: self first, then scratch copies
  const P* local_ptrs[ngpus];
  local_ptrs[0] = (const P*)self_input;
#pragma unroll
  for (int i = 0; i < n_peers; i++) {
    local_ptrs[i + 1] = (const P*)scratch_ptrs[i];
  }

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>(local_ptrs, idx);
  }
}

// ---- Legacy PCIe kernel (kept for CUDA graph capture path) ----

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

  // DMA-gather scratch buffers (local GPU memory for peer data copies).
  // Double-buffered to match the IPC double-buffering.
  void* scratch_[2][kMaxGpus - 1] = {};
  const void** d_scratch_ptrs_[2] = {};  // Device-side pointer arrays
  RankData host_rd_[2];                  // Host-side copy of peer IPC pointers
  int ar_slot_ = 0;                      // Alternating slot for scratch buffers
  size_t scratch_buf_size_ = 0;          // Size of each scratch buffer

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

  void register_pcie_buffers(void** ptrs0, void** ptrs1) {
    check_rank_data_capacity(2);
    for (int s = 0; s < 2; s++) {
      void** ptrs = s == 0 ? ptrs0 : ptrs1;
      RankData data;
      for (int i = 0; i < world_size_; i++) {
        data.ptrs[i] = ptrs[i];
        dbuf_raw_[s][i] = ptrs[i];
      }
      // Save host-side copy of peer IPC pointers for DMA gather
      host_rd_[s] = data;
      dbuf_rd_[s] = d_rank_data_base_++;
      CHECK_CUDA_SUCCESS(cudaMemcpy(dbuf_rd_[s], &data, sizeof(RankData), cudaMemcpyHostToDevice));
    }
    buffers_[ptrs0[rank_]] = dbuf_rd_[0];
    buffers_[ptrs1[rank_]] = dbuf_rd_[1];
    dbuf_enabled_ = true;
    dbuf_slot_ = 0;
  }

  void allocate_scratch(size_t max_size) {
    if (scratch_buf_size_ >= max_size) return;  // Already large enough
    scratch_buf_size_ = max_size;

    int n_peers = world_size_ - 1;
    for (int s = 0; s < 2; s++) {
      // Free old scratch buffers if any
      for (int p = 0; p < n_peers; p++) {
        if (scratch_[s][p]) {
          cudaFree(scratch_[s][p]);
          scratch_[s][p] = nullptr;
        }
      }
      if (d_scratch_ptrs_[s]) {
        cudaFree((void*)d_scratch_ptrs_[s]);
        d_scratch_ptrs_[s] = nullptr;
      }

      // Allocate new scratch buffers
      for (int p = 0; p < n_peers; p++) {
        CHECK_CUDA_SUCCESS(cudaMalloc(&scratch_[s][p], max_size));
      }

      // Allocate and populate device-side pointer array
      const void** d_ptrs;
      CHECK_CUDA_SUCCESS(cudaMalloc(&d_ptrs, n_peers * sizeof(void*)));
      CHECK_CUDA_SUCCESS(cudaMemcpy(d_ptrs, scratch_[s], n_peers * sizeof(void*), cudaMemcpyHostToDevice));
      d_scratch_ptrs_[s] = d_ptrs;
    }
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

  template <int ngpus>
  void launch_barrier(cudaStream_t stream) {
    barrier_kernel<ngpus><<<1, ngpus, 0, stream>>>(sg_, self_sg_, rank_);
  }

  template <typename T, int ngpus>
  void dma_gather_allreduce(cudaStream_t stream, T* input, T* output, int size, int threads, int block_limit) {
    auto d = packed_t<T>::P::size;
    int packed_size = size / d;
    int blocks = std::min(block_limit, (packed_size + threads - 1) / threads);
    blocks = std::max(blocks, 1);
    auto input_size_bytes = size * sizeof(T);

    int slot = ar_slot_ % 2;
    ar_slot_++;

    // Step 1: Copy input to IPC double-buffer (peers can read it after barrier)
    int dbuf_slot = dbuf_slot_ - 1;  // dbuf_slot_ was already incremented by caller
    int dbuf_s = dbuf_slot % 2;

    // Step 2: Barrier — wait for all peers to finish writing their input
    switch (world_size_) {
      case 2: launch_barrier<2>(stream); break;
      case 4: launch_barrier<4>(stream); break;
      case 6: launch_barrier<6>(stream); break;
      case 8: launch_barrier<8>(stream); break;
    }

    // Step 3: DMA-gather peer data to local scratch buffers
    int peer_idx = 0;
    for (int i = 0; i < world_size_; i++) {
      if (i == rank_) continue;
      const void* peer_src = host_rd_[dbuf_s].ptrs[i];
      CHECK_CUDA_SUCCESS(cudaMemcpyAsync(
          scratch_[slot][peer_idx], peer_src, input_size_bytes, cudaMemcpyDeviceToDevice, stream));
      peer_idx++;
    }

    // Step 4: Local reduce kernel — reads only from local HBM
    // self_input is our own IPC buffer (already contains our data)
    const void* self_input = host_rd_[dbuf_s].ptrs[rank_];
    local_reduce_kernel<T, ngpus><<<blocks, threads, 0, stream>>>(
        self_input, d_scratch_ptrs_[slot], output, packed_size);
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
      auto input_size = size * sizeof(T);
      AT_CUDA_CHECK(cudaMemcpyAsync(dbuf_raw_[slot][rank_], input, input_size, cudaMemcpyDeviceToDevice, stream));

      // Use DMA-gather path: allocate scratch if needed, then gather+reduce
      allocate_scratch(input_size);

#define DMA_KL(ngpus) dma_gather_allreduce<T, ngpus>(stream, input, output, size, threads, block_limit);
      switch (world_size_) {
        case 2: DMA_KL(2); break;
        case 4: DMA_KL(4); break;
        case 6: DMA_KL(6); break;
        case 8: DMA_KL(8); break;
        default:
          throw std::runtime_error("only supports (2,4,6,8) gpus, got " + std::to_string(world_size_));
      }
#undef DMA_KL
      return;
    }

    // Graph capture path or non-dbuf path: use legacy SM-read kernel
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
    // Free scratch buffers
    int n_peers = world_size_ - 1;
    for (int s = 0; s < 2; s++) {
      for (int p = 0; p < n_peers; p++) {
        if (scratch_[s][p]) cudaFree(scratch_[s][p]);
      }
      if (d_scratch_ptrs_[s]) cudaFree((void*)d_scratch_ptrs_[s]);
    }
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

static void register_pcie_buffers(fptr_t _fa, const std::vector<fptr_t>& ptrs0, const std::vector<fptr_t>& ptrs1) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  TORCH_CHECK(ptrs0.size() == (size_t)fa->world_size_);
  TORCH_CHECK(ptrs1.size() == (size_t)fa->world_size_);
  void* p0[8], *p1[8];
  for (size_t i = 0; i < ptrs0.size(); i++) {
    p0[i] = reinterpret_cast<void*>(ptrs0[i]);
    p1[i] = reinterpret_cast<void*>(ptrs1[i]);
  }
  fa->register_pcie_buffers(p0, p1);
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
  m.def("register_pcie_buffers", &register_pcie_buffers, "register double-buffered IPC buffers");
  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta, "get graph buffer IPC meta");
  m.def("register_graph_buffers", &register_graph_buffers, "register graph buffers");
}
