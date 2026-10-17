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

## Running on Modal

Install Modal locally and authenticate once, then run the launcher from the
repository root:

```bash
pip install --upgrade modal
modal setup
modal run bench/gemm_traversal_modal.py
```

The launcher uploads the current benchmark sources and headers, builds the CUDA
library remotely, and calls `bench/gemm_traversal_bench.py` on one GPU using
Modal's `B200+` selection (B200 or B300). The image
provides CUDA 13.2, make, g++, and CUDA-enabled PyTorch. Benchmark output streams
to the terminal, and the JSON results are saved locally to
`build/gemm_traversal_modal.json` when the run succeeds.

Select a B200 or forward options to the benchmark with `--bench-args`:

```bash
modal run bench/gemm_traversal_modal.py --gpu B200 \
  --bench-args "--shapes 2048 4096 --logical-devices 4 --samples 3" \
  --output build/gemm_traversal_b200.json
```

`--gpu` accepts `B200`, `B300`, and `B200+` (the default). Explicit GPU types
select `sm_100a` and `sm_103a`, respectively. With `B200+`, the launcher detects
the allocated GPU's compute capability remotely and selects the matching
architecture before building.
The launcher manages `--arch`, `--cuda-home`, `--device`, and `--json`; use
`--output` for the local JSON destination. Other benchmark options, including
`--build-only` and `--validate-only`, can be passed through `--bench-args`; those
two modes finish without producing JSON. Each invocation requests one physical
GPU, regardless of `--logical-devices`, with a one-hour timeout.

To request Nsight Compute reports for the 32768 square case on the same GPU
type as the original run:

```bash
modal run bench/gemm_traversal_modal.py --gpu B200 --ncu \
  --ncu-output build/gemm_traversal_ncu_32768
```

NCU mode defaults to `--shapes 32768` and the `detailed` metric set, which
includes memory workload analysis. It captures separate `.ncu-rep` reports and
raw metric CSV files for `device_Mlocal_N`, `Mglobal_N`, and `cuBLAS`. Open the
reports with `ncu-ui`. Use `--ncu-set full` for additional counters or
`--bench-args` to change the workload.

Each capture validates the selected kernel, warms it up, captures a graph
containing one GEMM, warms the graph, and profiles one replay inside
`cudaProfilerStart/Stop`. Application replay repeats this setup for each
counter pass; NCU cache flushing and clock control are disabled to preserve the
application's warmup and default clock behavior. Profiling runs produce no
benchmark timing samples. Continue using ordinary benchmark runs for timings.

The launcher first probes hardware-counter access with a tiny kernel. If the
Modal runtime cannot initialize NCU, it stops before building or running the
large workload and downloads `probe.log` and `status.json` to the output
directory. Installing NCU alone does not guarantee that a container runtime
exposes the required GPU profiling interfaces.

The profiling entry point also works on a machine with NCU counter access:

```bash
python bench/gemm_traversal_bench.py --build-only --arch sm_100a
ncu --set detailed --profile-from-start off --replay-mode application \
  --cache-control none --clock-control none --graph-profiling node \
  -o build/device_Mlocal_N_32768 \
  python bench/gemm_traversal_bench.py --arch sm_100a --shapes 32768 \
    --profile-kernel device_Mlocal_N
```

Repeat with `--profile-kernel Mglobal_N` and a different output filename to
compare the traversals. Adjust `--cuda-home` if needed, and use `sm_103a` on B300.

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
