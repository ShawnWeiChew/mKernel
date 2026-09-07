#!/usr/bin/env python3
"""Measure an already-ready pass through ag_gemm_kda_mla's acquire wait.

Build and run on one GPU (no torchrun or peer GPUs required):

    make GPU=blackwell acquire-load-pass-bench
    python bench/acquire_load_pass_bench.py

The tested condition is copied from src/ag_gemm_kda_mla.cu. The ready flag is
initialized to the requested epoch, so the first acquire load succeeds and the
__nanosleep(64) loop body is never executed.
"""

import argparse
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402


def percentile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.to(torch.float64), q / 100.0))


def summary(values: torch.Tensor) -> str:
    return (
        f"min {int(values.min()):4d} ns   "
        f"p50 {percentile(values, 50):6.1f} ns   "
        f"p90 {percentile(values, 90):6.1f} ns   "
        f"p99 {percentile(values, 99):6.1f} ns"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if not 0 <= args.epoch <= 0x7FFF_FFFF:
        parser.error("--epoch must fit in a positive torch.int32")

    torch.cuda.set_device(args.device)
    module = load_module.load("acquire_load_pass_bench")

    # ready == epoch makes the while condition false on its first acquire load.
    ready = torch.full(
        (1,), args.epoch, dtype=torch.int32, device=f"cuda:{args.device}"
    )
    measured = module.measure_ready_pass(ready, args.epoch, args.samples).cpu()
    timer_ns = measured[:, 0]
    pass_ns = measured[:, 1]

    timer_p50 = percentile(timer_ns, 50)
    pass_p50 = percentile(pass_ns, 50)
    print(f"device:          {torch.cuda.get_device_name(args.device)}")
    print(f"samples:         {args.samples:,}")
    print(f"timer baseline:  {summary(timer_ns)}")
    print(f"ready pass:      {summary(pass_ns)}")
    print(f"p50 difference:  {pass_p50 - timer_p50:.1f} ns "
          f"({(pass_p50 - timer_p50) / 1000.0:.4f} us)")
    print("\nThe p50 difference is the useful empty-pass estimate. The raw ready-pass")
    print("number includes the two %globaltimer reads used to observe it.")


if __name__ == "__main__":
    main()
