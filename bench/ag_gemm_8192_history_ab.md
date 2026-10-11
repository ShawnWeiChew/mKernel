# Historical AG-GEMM comparison at M=8192

Run from a checkout containing the history below, using Python 3.12+ with PyTorch
and an installed CUDA toolkit. The default target is eight local B300 GPUs:

```sh
python bench/ag_gemm_8192_history_ab.py
```

This command builds and benchmarks only **M=8192, N=3648, K=7168**, comparing:

- **before: `47dd7ea`** — immediately before the multicast/compile-time strategy
  work in `2ab4b93`, and before the cross-rank entry barrier. Uses the original
  flag reset and `cuStreamWriteValue32` completion signaling.
- **after: `f21b220`** — the current implementation at experiment creation.
  Uses the epoch entry barrier and four-byte copy completion signaling.

Both revisions select one consumer warp, 128x256 tiles, two CTAs per cluster,
and supergroup width 10. This compares their complete implementations; it does
not isolate the barrier from every intervening source change.

The runner extracts unmodified committed sources with `git archive` into a new
directory under `build/ag_gemm_8192_ab-*`, then compiles both with the same Python,
CUDA toolkit, architecture, and eight-rank setting. Your checkout, normal build
output, and kernel source are not modified. Full commit IDs and library hashes
are recorded in `manifest.json`. No SSH, remote jobs, downloads, or provisioning
are performed by this runner.

Each trial starts a fresh local eight-process torchrun job loading exactly one
revision's library, avoiding binding collisions and accidental selection of stale
shared libraries. The same harness and seeds are used for both revisions. Only
the API adapter differs: the old single entry point versus current prepare+launch.

The default two rounds run **before/after/after/before**, then
**after/before/before/after**. Each trial has five timing samples; each sample has
100 warmup replays and 200 timed replays. One graph replay is one full AG-GEMM
invocation, including each revision's own flag preparation, transfers, and joins.
CUDA events measure average replay latency, reduced to the slowest rank per sample.
Trial medians and paired round ratios are reported; a before/after ratio above 1
means the current revision is faster.

Inputs remain fixed throughout timed batches. All ranks synchronize before source
writes and after completion, preserving the baseline's input lifetime requirements.
The runner checks eager output, changes inputs after graph capture, checks repeated
graph output, and checks output again after timing. It uses the existing benchmark's
BF16 error criteria. There are no extra per-invocation cross-rank barriers in the
baseline graph.

The output directory contains build logs, exact command arguments, trial logs,
per-rank samples, correctness results in logs, GPU clock/power/utilization snapshots,
and `results.json`. Each revision uses its original allocation implementation;
different processes may receive different addresses. This experiment reproduces
the historical implementation comparison, rather than controlling memory placement.

Options:

```sh
# Compile and validate the two builds without executing any GPU work.
python bench/ag_gemm_8192_history_ab.py --build-only

# B200 instead of B300, or a different toolkit installation.
python bench/ag_gemm_8192_history_ab.py --arch sm_100a --cuda-home /usr/local/cuda

# More samples and an explicitly named new output directory.
python bench/ag_gemm_8192_history_ab.py --rounds 4 --samples 10 \
    --warmup 100 --iters 200 --output-dir build/8192-history-run1
```

Every invocation creates fresh snapshots and builds. An explicit output directory
must not already exist, so prior results are not overwritten. Run on an otherwise
idle node, inspect per-trial variability, and repeat before interpreting small gains.
