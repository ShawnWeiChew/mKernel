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

Replay mode is the whole story here. Kernel replay -- ncu's default -- snapshots
every allocation reachable from the context before each pass, and a DistBuffer's
multicast/peer-imported mappings are not copyable, so it dies before the first
pass with:

    cuda_context_state>> Failed to copy memory
    executeInternal returned an error: ContextSaveFailed

There is no ncu flag to exclude an allocation from that snapshot, so the default
here is application replay, which re-runs instead of saving. That relaunches the
whole job once per metric pass, which means ncu has to wrap *torchrun*: relaunch
one rank alone and its peers are gone. So this mode profiles from the launcher,
attaches to every rank (--target-processes all), and lets cudaProfilerStart pick
which one actually records -- the peers are attached but collect nothing.

Each pass is a full job restart, so keep the pass count down: --ncu-set detailed,
or a targeted --ncu-arg=--metrics=... beats `full` by minutes.
"""

import argparse
import datetime
import json
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
    ncu.add_argument("--ncu-out", default=None,
                     help="report path, without the .ncu-rep suffix ncu appends. "
                          "Defaults per --impl, so the two never clobber")
    ncu.add_argument("--ncu-replay", default="application",
                     choices=("application", "kernel", "app-range", "range"),
                     help="application (default): relaunch the job per pass, ncu "
                          "wraps torchrun. kernel: ncu wraps one rank, but this "
                          "kernel's multicast buffers make ncu's context save "
                          "fail. app-range: profile the whole profiled region as "
                          "one unit, no per-kernel serialization -- the escape "
                          "hatch when a device-side collective deadlocks")
    ncu.add_argument("--ncu-ranks", default="auto", choices=("auto", "one", "all"),
                     help="which ranks call cudaProfilerStart. auto: one for the "
                          "kernel, all for CUTLASS, whose GEMM rendezvouses "
                          "device-side and hangs if its peers are not profiled too")
    ncu.add_argument("--ncu-kernel", default=None,
                     help="ncu --kernel-name filter. Defaults per --impl; keeps "
                          "NCCL and tensor-init kernels out of the report")
    # Every extra pass is another full job restart under application replay,
    # so `detailed` rather than `full` is the default.
    ncu.add_argument("--ncu-set", default="detailed",
                     help="ncu --set (detailed, basic, roofline, full, ...)")
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

    cut = ap.add_argument_group(
        "cutlass baseline",
        "Profile CUTLASS's distributed all-gather GEMM at the same shape, for a "
        "report you can baseline the kernel's against.",
    )
    cut.add_argument("--impl", default="mkernel", choices=("mkernel", "cutlass"),
                     help="what to profile. cutlass needs --ncu or "
                          "--cutlass-tune-only; the in-kernel trace is mkernel-only")
    cut.add_argument("--cutlass-tiler", default="auto",
                     help="mma_tiler_mn as MxN (e.g. 256x256), or auto to take the "
                          "winner from the bench script's autotune")
    cut.add_argument("--cutlass-tune-only", action="store_true",
                     help="autotune, cache the winner and exit; run this once "
                          "before profiling so no replay pass pays for tuning")
    cut.add_argument("--cutlass-retune", action="store_true",
                     help="ignore the cached winner")
    cut.add_argument("--cutlass-cache", default="traces/.cutlass_tiler",
                     help="cache prefix; one JSON per rank")
    cut.add_argument("--cutlass-warmup", type=int, default=0,
                     help="warmup launches inside the profiled call. 0 keeps the "
                          "region to the timed launches; the JIT and caches are "
                          "already warm from the call before it")
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
        str(SELF), *sys.argv[1:],
    ]
    env = dict(os.environ)
    out = None
    if args.ncu and args.ncu_replay == "application":
        # Application replay relaunches the profiled application once per metric
        # pass. That application has to be the whole job: relaunch a single rank
        # and it comes back to peers that have long since exited.
        out = ncu_report_path(args)
        cmd = ncu_prefix(args, "all", out) + cmd
        # The children are already inside ncu; stop the profiled rank from
        # wrapping itself a second time.
        env[NCU_ACTIVE_ENV] = "1"
        print(f"application replay: the whole {nproc}-rank job reruns once per "
              f"metric pass, so prefer --ncu-set detailed over full\n", flush=True)

    print(f"spawning {nproc} ranks: {' '.join(cmd)}\n", flush=True)
    code = subprocess.call(cmd, env=env)
    return report_ncu_exit(code, out) if out is not None else code


def ncu_report_path(args):
    out = Path(args.ncu_out)
    if out.suffix == ".ncu-rep":     # ncu appends it; don't end up with two
        out = out.with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def ncu_prefix(args, target_processes, out):
    """The ncu invocation both wrapping strategies share."""
    return [
        shutil.which(args.ncu_bin) or args.ncu_bin,
        "--target-processes", target_processes,
        # cudaProfilerStart/Stop in run_ncu() brackets the launches, so NCCL
        # setup, the barriers and the warmup never reach the profiler -- and
        # under --target-processes all it is also what keeps the peer ranks,
        # which never call it, out of the report.
        "--profile-from-start", "off",
        "--replay-mode", args.ncu_replay,
        "--set", args.ncu_set,
        "--force-overwrite",
        "--export", str(out),
        # Without this the profiler region also collects the NCCL barrier and
        # the elementwise kernels upstream's run() launches while it allocates.
        # Profiling a collective barrier is both wasted passes and a deadlock
        # risk, since ncu serializes the launch it is profiling.
        *(("--kernel-name", args.ncu_kernel) if args.ncu_kernel else ()),
        *args.ncu_arg,
    ]


def report_ncu_exit(code, out):
    if code == 0:
        print(f"\nwrote {out}.ncu-rep  (open with: ncu-ui {out}.ncu-rep)", flush=True)
        return code
    print(
        f"\nncu exited {code}.\n"
        f"  ContextSaveFailed / 'Failed to copy memory': kernel replay cannot "
        f"snapshot the multicast DistBuffer mappings. Use the default "
        f"--ncu-replay application.\n"
        f"  ERR_NVGPUCTRPERM: counters are locked to root -- run as root, or "
        f"`sudo sh -c 'echo options nvidia "
        f"NVreg_RestrictProfilingToAdminUsers=0 > "
        f"/etc/modprobe.d/nvidia-profile.conf'` and reboot.",
        file=sys.stderr, flush=True,
    )
    return code


def exec_under_ncu(args, rank):
    """Kernel replay: wrap this one rank, leave the peers untouched.

    torchrun has no per-rank wrapper hook, so the child does it to itself: the
    inherited RANK/MASTER_ADDR environment survives the exec, which is what
    keeps the re-launched process the same rank of the same job. Only valid for
    kernel replay -- application replay would relaunch this rank alone.
    """
    out = ncu_report_path(args)
    cmd = [
        # application-only: the peers must run at full speed, both to serve the
        # remote A reads and to stay out of the report.
        *ncu_prefix(args, "application-only", out),
        sys.executable, str(SELF), *sys.argv[1:],
    ]
    env = dict(os.environ, **{NCU_ACTIVE_ENV: "1"})
    print(f"rank {rank} under ncu: {' '.join(cmd)}\n", flush=True)
    return report_ncu_exit(subprocess.call(cmd, env=env), out)


def round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def parse_tiler(text):
    parts = text.replace(",", "x").split("x")
    if len(parts) != 2:
        raise SystemExit(f"--cutlass-tiler wants MxN (e.g. 256x256), got {text!r}")
    return tuple(int(p) for p in parts)


def cutlass_tiler_cache(args, rank):
    # Per rank: every rank tunes (run() is collective) and they would otherwise
    # race on one file. They all see the same max-rank timings, so the winner
    # agrees across ranks anyway.
    return Path(f"{args.cutlass_cache}_rank{rank}.json")


def resolve_cutlass_tiler(args, bench, rank, M, N, K):
    """The bench script's autotuned winner, cached across processes.

    Application replay restarts the whole job once per metric pass, and CUTLASS
    is JIT-compiled per config -- tuning inside the profiled run would pay that
    on every pass. The cache turns it into a one-off, which is also what keeps
    the launch sequence identical across passes.
    """
    if args.cutlass_tiler != "auto":
        return parse_tiler(args.cutlass_tiler)

    key = f"{M}x{N}x{K}"
    path = cutlass_tiler_cache(args, rank)
    cache = {}
    if path.exists():
        try:
            cache = json.loads(path.read_text())
        except ValueError:                        # truncated by an interrupted run
            cache = {}
    if not args.cutlass_retune and key in cache:
        return tuple(cache[key])

    # cutlass_benchmark is the bench script's own autotune, honouring
    # CUTLASS_AUTOTUNE: with it off, this is just the mKernel-matched tile.
    ms, tiler, log = bench.cutlass_benchmark(m=M, n=N, k=K, warmup=2, iterations=5)
    if rank == 0:
        for cand, cand_ms in sorted(log, key=lambda kv: kv[1]):
            mark = " <- best" if tuple(cand) == tuple(tiler) else ""
            print(f"  [autotune] mma_tiler_mn={tuple(cand)}: {cand_ms:8.3f} ms{mark}",
                  flush=True)
    cache[key] = list(tiler)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2))
    return tuple(tiler)


def run_cutlass(args, rank, M, padded_n, K):
    """Profile the CUTLASS baseline at the same shape as the kernel.

    Nothing here touches an mKernel .so: this path exists to produce a second
    report at the same M/N/K, so the two can be compared as baselines.
    """
    import torch
    import torch.distributed as dist

    import ag_gemm_kda_mla_bench as bench

    tiler = resolve_cutlass_tiler(args, bench, rank, M, padded_n, K)
    if rank == 0:
        print(f"CUTLASS mma_tiler_mn={tiler} "
              f"cluster_shape_mn={bench._CUTLASS_CLUSTER_SHAPE}", flush=True)
    if args.cutlass_tune_only:
        dist.barrier()
        dist.destroy_process_group()
        return 0

    # Warm the JIT, the allocator and the caches outside the profiled region.
    # Upstream run() builds its own CUDA graph, so the profiled call below is a
    # graph launch -- ncu's default --graph-profiling node profiles the kernels
    # inside it individually, which is what makes the report comparable.
    bench._run_cutlass_once(m=M, n=padded_n, k=K, mma_tiler_mn=tiler,
                            warmup=WARMUP, iterations=1)
    torch.cuda.synchronize()
    dist.barrier()

    profiled = ncu_records_here(args, rank)
    if profiled:
        torch.cuda.profiler.start()
    ms = bench._run_cutlass_once(m=M, n=padded_n, k=K, mma_tiler_mn=tiler,
                                 warmup=args.cutlass_warmup,
                                 iterations=args.ncu_iters)
    if profiled:
        torch.cuda.profiler.stop()

    if rank == args.ncu_rank:
        print(f"\nCUTLASS profiled: {ms:.3f} ms/iter under the profiler "
              f"(not a benchmark -- read the report)", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


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

    if args.impl == "cutlass":
        # No mKernel .so is involved. N still has to match the kernel run
        # exactly for the reports to be comparable, and padded_n_for_m applies
        # the same 128/256 column-block rule col_block_for_m does.
        import ag_gemm_kda_mla_bench as bench

        M = args.shape
        if M % world_size != 0:
            raise SystemExit(
                f"global M={M} is not divisible by world_size={world_size}")
        padded_n = bench.padded_n_for_m(M, LOGICAL_N)
        if rank == 0:
            print(f"CUTLASS all-gather GEMM | M={M} (local {M // world_size}) "
                  f"N={padded_n} K={bench.K} world={world_size}", flush=True)
        return run_cutlass(args, rank, M, padded_n, bench.K)

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


def ncu_records_here(args, rank):
    """Whether this rank calls cudaProfilerStart.

    One rank is right for a kernel whose peers only supply memory: it keeps the
    report small and lets the peers run at full speed. It is wrong for a kernel
    that rendezvouses device-side -- ncu serializes the launch it profiles while
    the unprofiled peers run ahead and exit their kernel, so the rendezvous
    never completes and the launch hangs.
    """
    return args.ncu_ranks == "all" or rank == args.ncu_rank


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

    profiled = ncu_records_here(args, rank)
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

    if rank == args.ncu_rank:
        # Serialized and replayed, so this is wall time, not the kernel's time.
        # The honest number is in the report; this only shows it ran.
        print(f"\n{args.ncu_iters} launch(es) profiled, {per_launch_ms:.3f} ms each "
              f"under the profiler (not a benchmark -- read the report)", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


def resolve_defaults(args):
    """Fill in what depends on --impl, before any wrapping decision is made."""
    cutlass = args.impl == "cutlass"
    if args.ncu_out is None:
        args.ncu_out = ("traces/cutlass_ag_gemm_ncu" if cutlass
                        else "traces/ag_gemm_kda_mla_ncu")
    if args.ncu_ranks == "auto":
        # CUTLASS's GEMM signals between ranks from inside the kernel, so every
        # rank has to be in the same profiled phase. The kernel's peers only
        # serve memory, so one rank there keeps the report small.
        args.ncu_ranks = "all" if cutlass else "one"
    if args.ncu_kernel is None:
        # upstream names them kernel_cutlass_*; leave the kernel's own run
        # unfiltered, its profiled region holds nothing else.
        args.ncu_kernel = "regex:cutlass" if cutlass else ""
    if args.ncu_replay in ("range", "app-range"):
        # Range modes profile the whole region as one result, so there is no
        # per-launch identity left for a kernel filter to select. Everything
        # launched inside the region lands in the aggregate instead.
        args.ncu_kernel = ""
    if args.impl == "cutlass" and not (args.ncu or args.cutlass_tune_only):
        raise SystemExit(
            "--impl cutlass only profiles under ncu: add --ncu, or "
            "--cutlass-tune-only to just pick and cache the config."
        )
    if args.cutlass_tune_only:
        if args.impl != "cutlass":
            raise SystemExit("--cutlass-tune-only needs --impl cutlass")
        # Tuning is the step that exists so ncu never has to pay for it.
        args.ncu = False


def main():
    args = build_argparser().parse_args()
    resolve_defaults(args)
    # torchrun sets RANK in the children; its absence means we are the launcher.
    if "RANK" not in os.environ:
        return bootstrap(args)
    rank = int(os.environ["RANK"])
    if args.ncu and not os.environ.get(NCU_ACTIVE_ENV):
        if args.ncu_replay == "application":
            # Someone ran this under their own torchrun, so the launcher ncu has
            # to wrap is out of reach.
            raise SystemExit(
                "--ncu-replay application needs to wrap the launcher: run "
                "`python bench/ag_gemm_kda_mla_profile.py --ncu ...` without "
                "torchrun and let it spawn the ranks itself."
            )
        if rank == args.ncu_rank:
            # Before torch touches the GPU: ncu has to own the context from birth.
            return exec_under_ncu(args, rank)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
