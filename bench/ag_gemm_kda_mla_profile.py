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

Nsight Compute (--ncu) is a separate mode and deliberately shares nothing with
the trace path:

    make -j 10 GPU=blackwell run-ag-gemm-kda-mla-ncu

It loads the *shipping* .so, not the profile one, and passes a null ring, so the
cubin ncu measures carries no emit instructions, no extra registers and no ring
traffic -- the counters describe the kernel that actually ships. Only one rank
runs under ncu: the rank re-execs itself under `ncu`, the peers run natively and
sit in a barrier keeping their buffers alive, and cudaProfilerStart/Stop brackets
just the profiled launches so nothing else (NCCL, warmup) lands in the report.

Kernel replay is the only workable mode here: application replay would re-run the
torchrun child from scratch. Replay is safe because the kernel only *reads* the
copy-engine ready flags (a monotone counter the host stream writes), so a second
pass sees them already set instead of hanging.
"""

import argparse
import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))

WARMUP = 5

# Set on the rank that re-execs itself under ncu, so the new process runs the
# kernel instead of wrapping itself again.
NCU_ACTIVE_ENV = "MKERNEL_NCU_ACTIVE"

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

    ncu = ap.add_argument_group(
        "nsight compute",
        "--ncu switches modes entirely: shipping .so, null ring, no trace, no PDF.",
    )
    ncu.add_argument("--ncu", action="store_true",
                     help="profile one rank under Nsight Compute instead of tracing")
    ncu.add_argument("--ncu-rank", type=int, default=0,
                     help="the only rank that runs under ncu; the rest run natively")
    ncu.add_argument("--ncu-out", default="traces/ag_gemm_kda_mla_ncu",
                     help="report path, without the .ncu-rep suffix ncu appends")
    ncu.add_argument("--ncu-set", default="full",
                     help="ncu --set (full, detailed, basic, roofline, ...)")
    ncu.add_argument("--ncu-iters", type=int, default=1,
                     help="launches inside the profiled region")
    ncu.add_argument("--ncu-bin", default=os.environ.get("NCU", "ncu"),
                     help="ncu executable")
    # Values that start with a dash need the = form, or argparse eats them as
    # options of ours: --ncu-arg=--replay-mode --ncu-arg=application.
    ncu.add_argument("--ncu-arg", action="append", default=[], metavar="ARG",
                     help="extra flag passed through to ncu; repeatable, "
                          "use --ncu-arg=--flag for dashed values")
    ncu.add_argument("--ncu-timeout-min", type=int, default=60,
                     help="collective timeout while the peers wait out the replay")
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


def exec_under_ncu(args, rank):
    """Re-run this rank's process under ncu, leaving the peers untouched.

    torchrun has no per-rank wrapper hook, so the child does it to itself: the
    inherited RANK/MASTER_ADDR environment survives the exec, which is what
    keeps the re-launched process the same rank of the same job.
    """
    ncu_bin = shutil.which(args.ncu_bin) or args.ncu_bin
    out = Path(args.ncu_out)
    if out.suffix == ".ncu-rep":
        out = out.with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ncu_bin,
        # Only this process; the peer ranks must run at full speed, both to
        # feed the copy engine and to keep the report free of their kernels.
        "--target-processes", "application-only",
        # cudaProfilerStart/Stop in run() brackets the launches, so NCCL setup,
        # the barriers and the warmup never reach the profiler.
        "--profile-from-start", "off",
        "--set", args.ncu_set,
        "--force-overwrite",
        "--export", str(out),
        *args.ncu_arg,
        sys.executable, str(SELF), *sys.argv[1:],
    ]
    env = dict(os.environ, **{NCU_ACTIVE_ENV: "1"})
    print(f"rank {rank} under ncu: {' '.join(cmd)}\n", flush=True)
    code = subprocess.call(cmd, env=env)
    if code == 0:
        print(f"wrote {out}.ncu-rep  (open with: ncu-ui {out}.ncu-rep)", flush=True)
    else:
        print(
            f"ncu exited {code}. ERR_NVGPUCTRPERM means counters are locked to "
            f"root: run as root, or `sudo sh -c 'echo options nvidia "
            f"NVreg_RestrictProfilingToAdminUsers=0 > "
            f"/etc/modprobe.d/nvidia-profile.conf'` and reboot.",
            file=sys.stderr, flush=True,
        )
    return code


def round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def run(args):
    import torch
    import torch.distributed as dist

    import numpy as np

    import load_module
    import timings as tt

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"]))
    torch.cuda.set_device(local_rank)
    # Under ncu the peers sit in the closing barrier for as long as the replay
    # takes, which is minutes, not the 10 the default timeout allows.
    dist.init_process_group(
        "nccl", device_id=torch.device(f"cuda:{local_rank}"),
        timeout=datetime.timedelta(minutes=args.ncu_timeout_min) if args.ncu else None,
    )

    # ncu measures the shipping cubin, never the instrumented one: the emits,
    # the registers they cost and the ring's stores are work the real kernel
    # does not do, and every counter in the report would carry them.
    mod_name = KERNEL_NAME if args.ncu else f"{KERNEL_NAME}_profile"
    make_target = mod_name.replace("_", "-")
    mod = load_module.load(mod_name)
    if args.ncu and hasattr(mod, "EVENTS_PER_BLOCK"):
        raise RuntimeError(
            f"lib{mod_name}.so was built with -DPROFILE_TIMINGS. ncu must see a "
            f"cubin with no timing instrumentation in it -- rebuild with "
            f"`make -j 10 {make_target}`."
        )
    if not args.ncu and not hasattr(mod, "EVENTS_PER_BLOCK"):
        raise RuntimeError(
            f"lib{mod_name}.so is not a profile build (no EVENTS_PER_BLOCK). "
            f"Run `make -j 10 {make_target}`."
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
    events_per_block = 0 if args.ncu else mod.EVENTS_PER_BLOCK
    if rank == 0:
        if args.ncu:
            tail = f"ncu on rank {args.ncu_rank}, no ring (uninstrumented build)"
        else:
            ring_mb = num_blocks * events_per_block * mod.TIMING_RECORD_SIZE / 1e6
            tail = (f"ring {num_blocks}x{events_per_block} = "
                    f"{ring_mb:.0f} MB/rank")
        print(
            f"M={M} (local {local_m}) N={padded_n} K={K} world={world_size} | "
            f"COL_BLOCK={col_block} | {num_blocks} CTAs (all compute) | {tail}",
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

    if args.ncu:
        return run_ncu(args, mod, rank, A_kernel, A_local_buf, B, C)

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

    # Copy-engine spans. Widths come from CUDA events (exact, and independent of
    # any clock base); their position comes from the kernel's own ACOPY_READY
    # records. Anchoring uses the peer whose flag a CTA was actually caught
    # spinning on -- that observation brackets the real completion tightly, where
    # a peer nobody had to wait for only gives a loose upper bound.
    copy_rows = []
    try:
        raw = mod.copy_times(local_rank)
    except Exception as exc:                      # profile build without the export
        print(f"  (copy_times unavailable: {exc})")
        raw = []
    if raw:
        ev_ready = name_to_id["ACOPY_READY"]
        ev_begin = name_to_id["ACOPY_WAIT_BEGIN"]
        seq = records[:, 3] & ((1 << 28) - 1)
        # copy_end[p] <= earliest ACOPY_READY[p] holds for EVERY peer, because a
        # CTA cannot observe a flag before the copy that sets it finished. Each
        # peer therefore caps the anchor; only the tightest cap satisfies all of
        # them, so take the minimum rather than trusting any single peer.
        #
        # Picking the peer with the *smallest* observed spin is exactly wrong: a
        # short spin means the flag was already set, i.e. the copy finished at
        # some unknown earlier time -- the loosest bound available. The binding
        # constraint comes from whichever peer actually blocked someone.
        best_peer, anchor = None, None
        for peer, t_begin_ms, t_end_ms in raw:
            m = (records[:, 2] == ev_ready) & (seq == peer)
            if not m.any():
                continue
            cand = int(records[m, 1].min()) - int(t_end_ms * 1e6)
            if anchor is None or cand < anchor:
                best_peer, anchor = peer, cand
        best_slack = 0
        if anchor is not None:
            for peer, t_begin_ms, t_end_ms in raw:
                copy_rows.append((int(peer),
                                  anchor + int(t_begin_ms * 1e6),
                                  anchor + int(t_end_ms * 1e6)))
            t0 = int(records[:, 1].min())
            print(f"\ncopy engine (anchor bound by peer {best_peer} -- the one that "
                  f"actually gated a CTA):")
            for peer, b_ns, e_ns in copy_rows:
                print(f"  peer {peer}: {(b_ns-t0)/1000.0:8.1f} -> {(e_ns-t0)/1000.0:8.1f} us "
                      f"({(e_ns-b_ns)/1000.0:7.1f} us, "
                      f"{local_m*K*2/1e9/((e_ns-b_ns)/1e9):.0f} GB/s)")

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
        # (peer, begin_ns, end_ns) on the same clock as the records.
        copy_spans=np.array(copy_rows, dtype=np.int64).reshape(-1, 3),
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


def run_ncu(args, mod, rank, A_kernel, A_local_buf, B, C):
    """Profiled launches only: null ring, no records, no plot.

    Every rank runs this; only args.ncu_rank has an ncu attached, and the others
    are here to serve its remote A reads and then wait in the closing barrier.
    Leaving early would free the buffers a replay pass is still reading.
    """
    import torch
    import torch.distributed as dist

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    dist.barrier()

    profiled = rank == args.ncu_rank
    if profiled:
        torch.cuda.profiler.start()
    start.record()
    for _ in range(args.ncu_iters):
        mod.ag_gemm_kda_mla(A_kernel, A_local_buf, B, C, 0)
    end.record()
    torch.cuda.synchronize()
    if profiled:
        torch.cuda.profiler.stop()
    per_launch_ms = start.elapsed_time(end) / max(args.ncu_iters, 1)

    if profiled:
        # Serialized and replayed, so this is wall time, not the kernel's time.
        # The honest number is in the report; this only shows it ran.
        print(f"\n{args.ncu_iters} launch(es) profiled, {per_launch_ms:.3f} ms each "
              f"under the profiler (not a benchmark -- read the report)", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


def main():
    args = build_argparser().parse_args()
    # torchrun sets RANK in the children; its absence means we are the launcher.
    if "RANK" not in os.environ:
        return bootstrap(args)
    rank = int(os.environ["RANK"])
    if args.ncu and rank == args.ncu_rank and not os.environ.get(NCU_ACTIVE_ENV):
        # Before torch touches the GPU: ncu has to own the context from birth.
        return exec_under_ncu(args, rank)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
