# CUDA PCIe Allreduce

Custom CUDA allreduce kernel optimized for PCIe topologies (no NVLink). Two code paths:

- **DMA-gather path** (double-buffer mode): Uses Copy Engine to gather peer data to local scratch, then reduces locally from HBM. Targets CPU root complex topologies where SM-initiated PCIe reads are slow.
- **Legacy SM-read path** (single-buffer / CUDA graph): Direct SM reads from peer GPU memory with staggered access pattern.

Self-contained — JIT-compiled via `torch.utils.cpp_extension.load`, no external dependencies beyond PyTorch + CUDA.

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- 2+ NVIDIA GPUs (supports 2, 4, 6, 8)
- `ninja` build system (`pip install ninja`)

## Files

| File | Description |
|------|-------------|
| `pcie_allreduce.cu` | CUDA kernel — barrier, DMA-gather, local reduce, legacy SM-read |
| `__init__.py` | JIT compilation and Python bindings export |
| `cuda_ipc.py` | Minimal ctypes wrapper for CUDA IPC operations |
| `test_pcie_ar.py` | Basic correctness test (single-buffer path) |
| `stress_test.py` | Comprehensive stress test (double-buffer / DMA-gather path) |
| `bench_crossover.py` | Benchmark vs NCCL — finds crossover point |

## How to run

First clear the JIT cache (needed after code changes):

```bash
rm -rf ~/.cache/torch_extensions/
```

### Basic test (2 GPUs)

```bash
cd pcie_allreduce
CUDA_VISIBLE_DEVICES=0,1 python3 test_pcie_ar.py 2
```

### Stress test with DMA-gather (4 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 stress_test.py 4
```

### Benchmark vs NCCL (4 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 bench_crossover.py 4
```

### All 8 GPUs

```bash
python3 test_pcie_ar.py 8
python3 stress_test.py 8
python3 bench_crossover.py 8
```

## Architecture

### Allreduce flow (DMA-gather, double-buffer mode)

1. `cudaMemcpyAsync` input → registered IPC double-buffer
2. **Barrier kernel** (1 block) — ensures all peers wrote their data
3. `cudaMemcpyAsync × (world_size-1)` — DMA-gather each peer's IPC buffer → local scratch
4. **Local reduce kernel** — reads only from local GPU memory, writes output

All operations on the same CUDA stream → implicit ordering.

### CUDA graph compatibility

The legacy SM-read kernel is used during CUDA graph capture. `cudaMemcpyAsync` is graph-capturable, and the peer IPC addresses are fixed at registration time.

## Integration with SGLang

This is used as a drop-in replacement for `sgl_kernel.allreduce` in the PCIe communication path. The Python API matches the expected interface:

```python
import pcie_allreduce

fa = pcie_allreduce.init_custom_ar(meta_ptrs, rank_data, rank)
pcie_allreduce.register_pcie_buffers(fa, buf_ptrs_0, buf_ptrs_1)
pcie_allreduce.all_reduce(fa, input_tensor, output_tensor, 0, 0)
pcie_allreduce.dispose(fa)
```

## Notes

- The DMA-gather path benefits most on **CPU root complex topologies** (no PCIe switch) where SM-initiated reads have high latency (~2-3μs per TLP).
- On systems with **BAR1 P2P** or NVLink, NCCL may already be optimal for larger message sizes.
- The kernel wins consistently for **small messages** (< 4KB) due to lower launch overhead vs NCCL.
