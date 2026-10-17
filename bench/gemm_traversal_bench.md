# Pure GEMM traversal comparison

Run from the repository root on a B300 with CUDA-enabled PyTorch:

```bash
python bench/gemm_traversal_bench.py --build --json build/gemm_traversal.json
```

For B200, add `--arch sm_100a`. Override the toolkit with
`--cuda-home /path/to/cuda`. Building needs NVCC and CUDA libraries but no
PyTorch headers, GPU, networking libraries, distributed launch, or NCCL:

```bash
python3 bench/gemm_traversal_bench.py --build-only
python3 bench/gemm_traversal_bench.py --validate-only
```

The Python script allocates tensors with PyTorch, calls the compiled kernels
through ctypes, checks results against `cublasGemmEx`, and measures CUDA graphs
with CUDA events. The small C ABI library instantiates the kernel body in
`src/ag_gemm_warp_specialized.cu` twice. The only differing template argument
selects the tile-coordinate mapper in
`include/operators/ag_gemm/gemm_tile_traversal.h`. The benchmark no longer
compiles or calls `src/gemm_ar_blackwell.cu`.

## Workload and matched configuration

Every implementation computes `C = A @ B.T`, using identical contiguous BF16
`A[M,K]`, `B[N,K]`, and `C[M,N]` allocations and FP32 accumulation. The default
sweep is `M=N=K` at 2048, 4096, 8192, 16384, and 32768. For a rectangular sweep:

```bash
python bench/gemm_traversal_bench.py --n 4096 --k 8192
```

Both custom kernels use:

- Two MMA consumer warps, one producer warp, and one epilogue warpgroup: 224 threads.
- Two CTAs per cluster, 148 CTAs total, and a 512-by-256 cluster output region.
- A 128-by-256 accumulator per consumer per CTA and four input pipeline stages.
- Four 64-column epilogue chunks, a shared two-buffer staging ring, and a final
  store-completion wait. Neither benchmark path copies data or signals peers.
- The same supergroup width: 4 for M <= 4096, otherwise 8. Override both with
  `--supergroup-width 4` or `--supergroup-width 8`.

`device_Mlocal_N` applies the grouped snake within each logical device's rows
before advancing to the next device. `Mglobal_N` snakes over the full M axis.
The logical devices are local row shards on one physical GPU. `--logical-rank`
changes the first shard visited; it is independent of the physical `--device`.
No NVLink traffic is included.

Both variants use the same descriptor types and construction, allocation and
barrier setup, producer, two MMA consumers, epilogue, tile count, and launch
configuration. Only the mapping from persistent-loop tile ID to
`(device, local cluster row, column)` differs. The global mode maps its global
row back to the common device/local-row coordinates before the same loads and
stores. Compiler-generated instruction/register differences can still result
from the two mapping expressions; no other implementation is changed between
the variants.

## Small-shape padding

With the default eight logical devices, M=2048 gives 256 rows per shard. The
two-consumer cluster covers 512 rows, so **both modes** perform twice the useful
row work at this size. Both visit exactly the same padded tiles, using the same
3D tensor maps to zero-fill loads and discard stores outside each shard. The
global snake traverses the padded global cluster-row axis. The benchmark prints
`both traversal modes executed-row ratio=2.00x` and computes TFLOP/s from useful
`2*M*N*K` work. cuBLAS computes only the unpadded matrix. All other default
sizes have no padding.

For an unpadded comparison at every default size, use four logical devices:

```bash
python bench/gemm_traversal_bench.py --build --logical-devices 4
```

## Validation and timing

CPU validation models the producer/epilogue coordinates, checks unique complete
output coverage and equal padding, and compares both traversals for every
logical-rank rotation. A separate C++ test executes the actual host/device
coordinate mapper, checks the specified traversal orders, and exercises
partial supergroups and one-device equivalence:

```bash
make -f bench/Makefile.gemm_traversal check-traversal
```

Neither CPU test establishes GPU synchronization or numerical correctness.

On a GPU, correctness checks run before timing and after graph replay, with
NaN-poisoned output to detect unwritten tiles. An additional AG rank rotation is
checked. Comparisons cover the entire result in chunks to limit temporary
memory. A is scaled by `1/sqrt(K)` to keep output variance near one; default
tolerances are `rtol=0.02`, `atol=0.01`. A correctness failure aborts the run.

Timings report the median of 12 samples, each containing 10 GEMMs in a CUDA
graph, after 20 warmups per implementation. Order rotates and reverses across
samples. Allocation, descriptor construction, Python launch overhead,
correctness checks, and graph capture are outside the measured interval.
This measures repeated-buffer device throughput, not cold-cache or end-to-end
host latency. `--samples`, `--inner-iters`, and `--warmup` control the run.

JSON output includes every sample, useful TFLOP/s, cuBLAS speedup, padding
ratio, numerical error metrics, and configuration/GPU metadata. The largest
square case holds about 8 GiB in A, B, output, and reference tensors, plus
CUDA/cuBLAS workspaces and chunked correctness temporaries.

Earlier experimental changes to GEMM-AR remain in the branch, but that kernel
is not part of this traversal-only benchmark. Rebuild with `--build` after
updating from the earlier two-implementation comparison.
