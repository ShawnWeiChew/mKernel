"""Run gemm_ar_blackwell under Nsight Compute.

One command, no arguments needed:

    python bench/gemm_ar_blackwell_ncu.py

That launches `ncu` around `torchrun --nproc-per-node=<gpus>` running this same
file, warms the kernel up, and profiles exactly ONE launch of
gemm_ar_fused_kernel_stub per rank. Reports land in traces/ncu/ (one per
process; each rank prints its own pid so report <-> rank is unambiguous).

Why this instead of pointing ncu at gemm_ar_blackwell_bench.py: that script
runs the correctness sweep, the NCCL baseline, the cutlass reference and every
compiled (strategy, split, unroll, depth) combination, so ncu would profile
hundreds of launches of dozens of instantiations. Here the config is pinned,
the profiled region is bracketed with cudaProfilerStart/Stop, and warmup runs
outside it.

Examples:
    python bench/gemm_ar_blackwell_ncu.py --shape 8192
    python bench/gemm_ar_blackwell_ncu.py --strategy push --comp-sm 128 --unroll 64
    python bench/gemm_ar_blackwell_ncu.py --set full --ranks 1   # see the note below
    python bench/gemm_ar_blackwell_ncu.py --source            # keep SASS/source correlation
    python bench/gemm_ar_blackwell_ncu.py --dry-run           # just print the ncu command
    python bench/gemm_ar_blackwell_ncu.py --no-ncu            # workload only, no profiler

Replay and multi-rank (the one thing to be careful about):
    The kernel is a collective -- the comm SMs spin on peer barrier flags over
    multimem. If ncu needs more than one pass to collect the requested metrics
    it REPLAYS the kernel on each rank independently, and ranks then wait on
    peers that are in a different pass: a hang, not a slow run. So the default
    metric list is deliberately small. `--set full` (or any wide metric set) is
    safe on --ranks 1, and otherwise needs --replay-mode application at your
    own risk. If a multi-rank run hangs with the GPUs pinned, that is what
    happened -- kill it and narrow the metrics.

    --ranks 1 also requires a module built with INTRA_NUM_DEVICES=1
    (make -j 10 GPU=blackwell INTRA_NUM_DEVICES=1 gemm_ar_blackwell); the
    default build hardcodes 8 peers.

Permissions: ncu needs GPU performance counter access. If it exits with
ERR_NVGPUCTRPERM, the host needs
`sudo sh -c 'echo options nvidia NVreg_RestrictProfilingToAdminUsers=0 > /etc/modprobe.d/nvidia-profiler.conf'`
plus a reload/reboot, or run as root.
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# The templated kernel stub in src/gemm_ar_blackwell.cu. Every (strategy,
# split, unroll, depth) instantiation shares the name, so this regex plus
# --launch-count is what pins the profile to our one launch.
KERNEL_REGEX = "regex:gemm_ar_fused_kernel_stub"

STRATEGIES = {"push": 0, "pull": 1}

# Deliberately narrow: enough to see where the fused kernel spends its time
# without pushing ncu into multi-pass replay, which deadlocks a collective
# (see the module docstring). Override wholesale with --metrics / --set.
DEFAULT_METRICS = ",".join([
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sectors.avg.pct_of_peak_sustained_elapsed",
])


# ----------------------------------------------------------------------
# child: the workload ncu actually profiles
# ----------------------------------------------------------------------

def run_child(args) -> int:
    import torch
    import torch.distributed as dist

    sys.path.insert(0, str(ROOT / "python"))
    import load_module  # noqa: E402

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"]))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    is_chief = rank == 0

    mod = load_module.load("gemm_ar_blackwell")

    # ncu names each report by pid (%p), so print the mapping here -- it is the
    # only place both numbers exist.
    print(f"[rank {rank}] pid={os.getpid()} device=cuda:{local_rank}", flush=True)

    # Ask the module what it was built with instead of trusting a default: a
    # pinned build (e.g. -D'GEMM_AR_FOR_EACH_UNROLL(F)=F(16)') would otherwise
    # TORCH_CHECK inside the profiled launch, after ncu has already attached.
    splits = list(mod.compiled_comp_sm_splits())
    unrolls = list(mod.compiled_ar_unrolls())
    depths = list(mod.compiled_signal_depths())
    strategies = set(mod.compiled_strategies())

    def pick(name, want, avail):
        if want in avail:
            return want
        fallback = sorted(avail)[0]
        if is_chief:
            print(f"  [warn] {name}={want} not compiled in "
                  f"(have {sorted(avail)}); using {fallback}", flush=True)
        return fallback

    strategy = pick("strategy", STRATEGIES[args.strategy], strategies)
    comp_sm = pick("comp_sm", args.comp_sm, splits)
    depth = pick("signal_depth", args.depth, depths)
    # unroll 0 == AR_UNROLL_BY_SHAPE: let the kernel's shape heuristic choose,
    # but only if the value it will choose was compiled in.
    unroll = args.unroll
    if unroll != 0:
        unroll = pick("ar_unroll", unroll, unrolls)
    elif (32 if args.shape <= 2048 else 64) not in unrolls:
        unroll = pick("ar_unroll", 32 if args.shape <= 2048 else 64, unrolls)

    M = N = args.shape
    K = M // world_size
    if is_chief:
        strategy_name = next(k for k, v in STRATEGIES.items() if v == strategy)
        print(f"M={M} K={K} N={N}  world_size={world_size}\n"
              f"config: strategy={strategy_name.upper()} "
              f"comp_sm={comp_sm}:{mod.num_blocks() - comp_sm} "
              f"unroll={'by-shape' if unroll == 0 else unroll} depth={depth}",
              flush=True)

    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)
    A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) / (K ** 0.25)
    B = torch.randn((K, N), device="cuda", dtype=torch.bfloat16) / (K ** 0.25)

    C_dbuf = mod.DistBuffer((M, N), dtype=torch.bfloat16, local_rank=local_rank,
                            local_world_size=world_size, multicast=True)
    C_dbuf.data_.zero_()
    C_final = mod.DistBuffer((M, N), dtype=torch.bfloat16, local_rank=local_rank,
                             local_world_size=world_size, multicast=True)
    C_final.data_.zero_()
    # The barrier is never cleared between launches, so the epoch has to keep
    # rising -- same contract as the bench (see make_barrier there).
    barrier = mod.DistBuffer((2, 1024, 1024), dtype=torch.int, local_rank=local_rank,
                             local_world_size=world_size, multicast=True)
    barrier.data_.zero_()

    epoch = 0

    def launch():
        nonlocal epoch
        epoch += 1
        mod.gemm_ar_intranode_blackwell(A, B, C_dbuf, barrier, C_final, epoch,
                                        strategy, comp_sm, unroll, depth)

    # Warmup is outside the profiled region: it pays module load, the
    # cudaFuncSetAttribute for dynamic smem, and clock ramp.
    for _ in range(args.warmup):
        torch.cuda.synchronize()
        dist.barrier()
        launch()
    torch.cuda.synchronize()
    dist.barrier()

    # cudaProfilerStart/Stop pairs with ncu's --profile-from-start off, so the
    # warmup launches above are invisible to the profiler no matter how many
    # there are.
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    for _ in range(args.launches):
        launch()
    torch.cuda.synchronize()
    cudart.cudaProfilerStop()

    dist.barrier()
    if is_chief:
        print(f"[rank {rank}] profiled {args.launches} launch(es)", flush=True)
    dist.destroy_process_group()
    return 0


# ----------------------------------------------------------------------
# parent: build the ncu + torchrun command line
# ----------------------------------------------------------------------

def find_ncu() -> str:
    if os.environ.get("NCU"):
        return os.environ["NCU"]
    found = shutil.which("ncu")
    if found:
        return found
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    candidate = Path(cuda_home) / "bin" / "ncu"
    if candidate.exists():
        return str(candidate)
    sys.exit("ncu not found. Set NCU=/path/to/ncu or add it to PATH "
             "(it ships in $CUDA_HOME/bin).")


def default_ranks() -> int:
    env = os.environ.get("INTRA_NUM_DEVICES")
    if env:
        return int(env)
    try:
        import torch
        n = torch.cuda.device_count()
        if n:
            return n
    except Exception:
        pass
    return 8


def child_argv(args) -> list[str]:
    return [
        "--child",
        "--shape", str(args.shape),
        "--strategy", args.strategy,
        "--comp-sm", str(args.comp_sm),
        "--unroll", str(args.unroll),
        "--depth", str(args.depth),
        "--warmup", str(args.warmup),
        "--launches", str(args.launches),
    ]


def run_parent(args) -> int:
    so = ROOT / "build" / "libgemm_ar_blackwell.so"
    if not so.exists():
        sys.exit(f"{so} does not exist. Build it first:\n"
                 f"  make -j 10 GPU=blackwell gemm_ar_blackwell")

    torchrun = [sys.executable, "-m", "torch.distributed.run",
                "--standalone", f"--nproc-per-node={args.ranks}",
                str(HERE / "gemm_ar_blackwell_ncu.py")] + child_argv(args)

    if args.no_ncu:
        cmd = torchrun
    else:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)

        ncu = [
            find_ncu(),
            # torchrun's launcher makes no CUDA calls; its children are the
            # ranks, so every profiled process comes from "all".
            "--target-processes", "all",
            "--profile-from-start", "off",
            "--kernel-name", args.kernel_name,
            "--launch-count", str(args.launches),
            "--replay-mode", args.replay_mode,
            "-f", "-o", f"{out}_%p",
        ]
        if args.set:
            ncu += ["--set", args.set]
        else:
            ncu += ["--metrics", args.metrics]
        if not args.source:
            # Source correlation makes the report much larger and needs the
            # .cu around to be useful; the build already passes -lineinfo, so
            # this is purely opt-in.
            ncu += ["--import-source", "no"]
        ncu += shlex.split(args.ncu_args)
        cmd = ncu + torchrun

        if args.ranks > 1 and (args.set or args.metrics != DEFAULT_METRICS):
            print("[warn] custom metrics on >1 rank: if ncu needs multiple "
                  "passes it will replay the kernel per rank and the collective "
                  "will hang. See the docstring.", flush=True)

    print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if args.dry_run:
        return 0

    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc == 0 and not args.no_ncu:
        print(f"\nreports: {Path(args.out).name}_<pid>.ncu-rep under "
              f"{(ROOT / args.out).parent}\n"
              f"open with: ncu-ui <report>.ncu-rep\n"
              f"or dump:   ncu --import <report>.ncu-rep --page details")
    return rc


def main() -> int:
    p = argparse.ArgumentParser(
        description="Profile gemm_ar_blackwell under Nsight Compute.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--shape", type=int, default=4096,
                   help="square problem size; M=N=shape, K=shape/world_size "
                        "(default: 4096)")
    p.add_argument("--strategy", choices=sorted(STRATEGIES), default="pull",
                   help="comp->comm signalling strategy (default: pull)")
    p.add_argument("--comp-sm", type=int, default=128,
                   help="compute SMs; the rest run the all-reduce (default: 128)")
    p.add_argument("--unroll", type=int, default=0,
                   help="AR unroll factor; 0 = the kernel's shape heuristic "
                        "(default: 0)")
    p.add_argument("--depth", type=int, default=0,
                   help="signal pipeline depth (default: 0)")
    p.add_argument("--warmup", type=int, default=5,
                   help="unprofiled launches before the profiled region "
                        "(default: 5)")
    p.add_argument("--launches", type=int, default=1,
                   help="launches inside the profiled region; also ncu's "
                        "--launch-count (default: 1)")
    p.add_argument("--ranks", type=int, default=None,
                   help="GPUs to run on (default: all visible)")

    p.add_argument("--metrics", default=DEFAULT_METRICS,
                   help="ncu --metrics list (default: a small single-pass set)")
    p.add_argument("--set", default=None,
                   help="ncu --set (e.g. full); replaces --metrics. Multi-pass "
                        "-- safe on --ranks 1, may hang a collective otherwise")
    p.add_argument("--replay-mode", default="kernel",
                   choices=["kernel", "application", "range", "app-range"],
                   help="ncu --replay-mode (default: kernel)")
    p.add_argument("--kernel-name", default=KERNEL_REGEX,
                   help=f"ncu --kernel-name filter (default: {KERNEL_REGEX})")
    p.add_argument("--source", action="store_true",
                   help="keep source correlation in the report (bigger file)")
    p.add_argument("--out", default="traces/ncu/gemm_ar_blackwell",
                   help="report basename; _<pid>.ncu-rep is appended "
                        "(default: traces/ncu/gemm_ar_blackwell)")
    p.add_argument("--ncu-args", default="",
                   help="extra flags passed through to ncu verbatim")
    p.add_argument("--no-ncu", action="store_true",
                   help="run the workload without the profiler (sanity check)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the command and exit")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)

    args = p.parse_args()

    # Being inside torchrun is what makes this a rank, with or without --child.
    if args.child or "LOCAL_RANK" in os.environ:
        return run_child(args)

    if args.ranks is None:
        args.ranks = default_ranks()
    return run_parent(args)


if __name__ == "__main__":
    sys.exit(main())
