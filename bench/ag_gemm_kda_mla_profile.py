"""Run one instrumented ag_gemm_kda_mla launch and dump its in-kernel trace.

Needs the profile build, which is a separate .so from the shipping one:

    make -j 10 GPU=blackwell ag-gemm-kda-mla-profile
    python bench/ag_gemm_kda_mla_profile.py --shape 8192

No torchrun needed: with no RANK in the environment this script re-execs itself
under torch.distributed.run, one process per visible GPU. Running it under
torchrun directly still works -- it just skips the bootstrap.

Exactly one iteration is profiled. Warmup launches pass a null ring so their
records never land in the buffer: a second iteration would start at whatever
index each CTA's head had reached, which is neither overlapping nor plottable.

Unlike gemm_ar_blackwell there are no comm CTAs -- every block computes, and the
all-gather rides inside the producer's A load, which TMAs out of a peer's
buffer. The question the trace answers is whether that remote fetch hides behind
the compute, which is why the consumer's input wait is split into
"mma: wait A (local)" and "mma: wait A (remote)".

Re-render any time without a GPU:

    python python/timings.py traces/ag_gemm_kda_mla.npz --collapse
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))

WARMUP = 5

# Matches bench/ag_gemm_kda_mla_bench.py: the 8-way tensor-parallel KDA
# projection width before the kernel's column padding.
LOGICAL_N = 6284
KERNEL_NAME = "ag_gemm_kda_mla"


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shape", type=int, default=8192,
                    help="global M. Local rows are M // world_size.")
    ap.add_argument("--out", default="traces/ag_gemm_kda_mla.npz")
    ap.add_argument("--pdf", default="", help="default: --out with a .pdf suffix")
    ap.add_argument("--dump-rank", type=int, default=0)
    ap.add_argument("--nproc", type=int, default=0,
                    help="ranks to spawn (default: every visible GPU)")
    ap.add_argument("--rows-per-role", type=int, default=8,
                    help="rows per band in the PDF; 0 for every CTA")
    ap.add_argument("--collapse", action="store_true",
                    help="render the two-colour wait/work view")
    return ap


def bootstrap(args):
    """Re-exec under torch.distributed.run, one rank per GPU."""
    import torch

    nproc = args.nproc or torch.cuda.device_count()
    if nproc <= 0:
        raise SystemExit("no CUDA devices visible")

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc-per-node={nproc}",
        str(Path(__file__).resolve()), *sys.argv[1:],
    ]
    print(f"spawning {nproc} ranks: {' '.join(cmd)}\n", flush=True)
    return subprocess.call(cmd)


def round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def run(args):
    import torch
    import torch.distributed as dist

    import load_module
    import timings as tt

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"]))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))

    mod = load_module.load("ag_gemm_kda_mla_profile")
    if not hasattr(mod, "EVENTS_PER_BLOCK"):
        raise RuntimeError(
            "loaded .so is not a profile build (no EVENTS_PER_BLOCK). "
            "Run `make -j 10 ag-gemm-kda-mla-profile`."
        )
    if world_size != mod.NUM_DEVICES:
        raise SystemExit(
            f"world size {world_size} != INTRA_NUM_DEVICES={mod.NUM_DEVICES} the .so "
            f"was built with. Rebuild with INTRA_NUM_DEVICES={world_size}, or run "
            f"with --nproc {mod.NUM_DEVICES}."
        )

    M = args.shape
    if M % world_size != 0:
        raise SystemExit(f"global M={M} is not divisible by world_size={world_size}")
    local_m = M // world_size
    K = mod.K
    # The kernel picks COL_BLOCK from M, and N has to be a whole number of them.
    col_block = mod.col_block_for_m(M)
    padded_n = round_up(LOGICAL_N, col_block)

    num_blocks = mod.NUM_BLOCKS
    events_per_block = mod.EVENTS_PER_BLOCK
    if rank == 0:
        ring_mb = num_blocks * events_per_block * mod.TIMING_RECORD_SIZE / 1e6
        print(
            f"M={M} (local {local_m}) N={padded_n} K={K} world={world_size} | "
            f"COL_BLOCK={col_block} | {num_blocks} CTAs (all compute) | "
            f"ring {num_blocks}x{events_per_block} = {ring_mb:.0f} MB/rank",
            flush=True,
        )

    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)
    A_local = torch.randn((local_m, K), device="cuda", dtype=torch.bfloat16) / (K**0.25)

    A_kernel = mod.DistBuffer(
        (local_m, K), dtype=torch.bfloat16, local_rank=local_rank,
        local_world_size=world_size, multicast=True,
    )
    A_kernel.data_.copy_(A_local)
    # Landing buffer for the copy-engine all-gather: every device's A shard,
    # stacked, so shape is [global_M, K] -- not the local shard's [local_m, K].
    A_local_buf = torch.empty((M, K), device="cuda", dtype=torch.bfloat16)
    B = torch.zeros((K, padded_n), device="cuda", dtype=torch.bfloat16)
    B[:, :LOGICAL_N].copy_(
        torch.randn((K, LOGICAL_N), device="cuda", dtype=torch.bfloat16) / (K**0.25)
    )
    C = torch.zeros((M, padded_n), device="cuda", dtype=torch.bfloat16)
    dist.barrier()

    # Warmup with a null ring: pointer 0 short-circuits the store in
    # emit_timing_impl, so these launches leave the buffer untouched.
    for _ in range(WARMUP):
        mod.ag_gemm_kda_mla(A_kernel, A_local_buf, B, C, 0)
    torch.cuda.synchronize()
    dist.barrier()

    # Two int64s per 16-byte record. Zero init is what makes timestamp==0 the
    # "unwritten" sentinel the host unpacker uses to find each block's head.
    ring = torch.zeros(num_blocks * events_per_block * 2, dtype=torch.int64, device="cuda")

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    dist.barrier()

    start.record()
    mod.ag_gemm_kda_mla(A_kernel, A_local_buf, B, C, ring.data_ptr())
    end.record()
    torch.cuda.synchronize()
    kernel_ms = start.elapsed_time(end)
    dist.barrier()

    if rank == 0:
        print(f"profiled launch: {kernel_ms:.3f} ms wall (includes emit overhead)", flush=True)

    if rank != args.dump_rank:
        dist.barrier()
        dist.destroy_process_group()
        return 0

    records, heads = tt.unpack(ring.cpu(), num_blocks, events_per_block)
    overflowed = int((heads >= events_per_block).sum())
    print(f"\n{records.shape[0]} records across {int((heads > 0).sum())} CTAs "
          f"(max {int(heads.max())}/{events_per_block} per CTA)")
    if overflowed:
        print(f"  WARNING: {overflowed} CTAs hit the cap and dropped their tail. "
              f"Rebuild with EVENTS_PER_BLOCK={events_per_block * 2}.")

    warp_layout = {str(k): int(v) for k, v in mod.WARP_LAYOUT.items()}
    name_to_id = {str(k): int(v) for k, v in mod.TIMING_EVENTS.items()}
    phases = tt.phases_for(KERNEL_NAME)

    out = tt.save(
        args.out,
        records,
        name_to_id,
        warp_layout=warp_layout,
        kernel=KERNEL_NAME,
        # Every CTA computes, so there is no comm half of the grid; telling the
        # renderer the split is the whole grid is what keeps it from inventing
        # an AR band.
        num_comp_sm=num_blocks,
        num_blocks=num_blocks,
        events_per_block=events_per_block,
        M=M, N=padded_n, K=K,
        local_m=local_m,
        col_block=col_block,
        rank=rank,
        world_size=world_size,
        kernel_ms=kernel_ms,
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    all_spans = tt.spans_for_all_roles(records, name_to_id, num_blocks, warp_layout, phases)
    spans, shown = tt.limit_rows_per_role(
        all_spans, num_blocks, warp_layout, args.rows_per_role
    )
    trace_us = tt.wall_span_us(records)
    print(f"\n{len(all_spans)} spans | trace wall span {trace_us:.1f} us "
          f"vs {kernel_ms * 1000:.1f} us measured "
          f"({100.0 * trace_us / (kernel_ms * 1000):.0f}% instrumented)")
    if shown:
        detail = "  ".join(
            f"{tt.ROLE_DISPLAY.get(r, r)} {n}/{avail}"
            for r, (n, avail) in sorted(
                shown.items(), key=lambda kv: tt.ROLE_ORDER.index(kv[0])
            )
        )
        print(f"  rows shown: {detail}")
    print(tt.summarize(spans))

    pdf = Path(args.pdf) if args.pdf else Path(args.out).with_suffix(".pdf")
    title = (f"ag_gemm_kda_mla  M={M} N={padded_n} K={K}  rank {rank}/{world_size}  "
             f"{num_blocks} CTAs  ({kernel_ms:.3f} ms)")
    written = tt.plot(records, pdf, name_to_id, num_comp_sm=num_blocks,
                      layout=warp_layout, title=title, collapse=args.collapse,
                      rows_per_role=args.rows_per_role, phases_by_role=phases,
                      marker_events=tt.MARKER_EVENTS.get(KERNEL_NAME, ()))
    print(f"wrote {written}" if written else "nothing to plot")

    dist.barrier()
    dist.destroy_process_group()
    return 0


def main():
    args = build_argparser().parse_args()
    # torchrun sets RANK in the children; its absence means we are the launcher.
    if "RANK" not in os.environ:
        return bootstrap(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
