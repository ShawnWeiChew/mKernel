import os
import sys
import torch
import torch.distributed as dist
import time
from pathlib import Path
from enum import Enum

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402
import cutlass_dgemm_ar  # noqa: E402
from common import check_close

SHAPES= [2048, 4096, 8192, 16384, 32768]
# Split the sweep is compared against; must be in GEMM_AR_FOR_EACH_COMP_SM.
DEFAULT_COMP_SM = 128
DEFAULT_SIGNAL_DEPTH = 0


def default_ar_unroll(M):
    """Mirror of default_ar_unroll() in gemm_ar_blackwell.cuh -- the unroll the
    shape heuristic picks when ar_unroll is left at AR_UNROLL_BY_SHAPE. Used
    only to label which sweep entry is the pre-sweep default."""
    return 32 if M <= 2048 else 64
WARMUP = 30
# Target sample count per configuration. The timed loop rounds it to a whole
# number of Williams orders (i.e. a multiple of the condition count) so the
# position balancing stays exact, so the effective count can differ slightly --
# it is printed per shape.
BENCH_ITER = 60

class GemmToArSignal(Enum):
    PUSH = 0
    PULL = 1

# Both signalling strategies are benchmarked (and correctness-checked) on every
# shape, so the runtime knob can be compared head to head.
ALL_STRATEGIES = (GemmToArSignal.PUSH, GemmToArSignal.PULL)


def make_barrier(mod, local_rank, world_size):
    """Allocate a zeroed comp->comm barrier buffer.

    Each strategy needs its OWN barrier: the two leave the counters on
    incompatible scales. PUSH has all NUM_DEVICES peers release_add 1 into the
    destination's slot, so a tile is ready when the counter reaches
    epoch * NUM_DEVICES. PULL has each device release_store its own epoch into
    its own slot, and the reader multimem MIN-reduces across peers, so a tile is
    ready when the reduced value reaches epoch. Sharing one buffer would let a
    PULL store (epoch) knock the accumulated PUSH counter (epoch * NUM_DEVICES)
    backwards, and a following PUSH would then never reach its gate -- a hang,
    not a wrong answer.
    """
    barrier = mod.DistBuffer((2, 1024, 1024), dtype=torch.int,
        local_rank=local_rank, local_world_size=world_size, multicast=True)
    barrier.data_.zero_()
    return barrier


def williams_orders(items):
    """Balanced Latin square (Williams design) over `items`.

    Returns len(items) orderings in which every condition occupies every
    position exactly once and -- for an even count -- every ordered pair is
    adjacent exactly once. Full permutations balance both as well, but
    factorial(n) is unusable past about four conditions and the SM sweep pushes
    the count into the twenties.

    Construction: first row alternates from the ends (0, 1, n-1, 2, n-2, ...);
    every later row shifts it by one modulo n. Because all rows are the same
    row shifted, the order is identical on every rank, which it has to be --
    every condition is a collective.
    """
    n = len(items)
    first, lo, hi = [], 0, n - 1
    while lo <= hi:
        first.append(lo)
        if lo != hi:
            first.append(hi)
        lo, hi = lo + 1, hi - 1
    return [tuple(items[(v + r) % n] for v in first) for r in range(n)]


def elapsed_ms(samples):
    """Drain (start, end) cuda event pairs into per-iter wall times (ms)."""
    return [s.elapsed_time(e) for s, e in samples]


def sync_ranks():
    """Drain the local stream, then line every rank up on the host.

    Both timed loops call this before recording, so each iteration starts from
    an idle stream on every rank. Without it the two paths are not comparable:
    a back-to-back loop hides launch overhead behind the queue and lets ranks
    self-synchronize, while a loop that resets state between iters does not.
    """
    torch.cuda.synchronize()
    dist.barrier()


def median_then_max_cuda(samples, label=""):
    """Local median over iters, then max over ranks (the slowest rank sets the
    collective's cost). Also prints the per-rank medians so a straggler is
    visible instead of being hidden behind the max."""
    sorted_samples = sorted(float(x) for x in samples)
    median = sorted_samples[len(sorted_samples) // 2]

    t = torch.tensor([median], dtype=torch.float64, device="cuda")
    gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    if label and dist.get_rank() == 0:
        per_rank = " ".join(
            f"r{i}={float(x.item()):.3f}" for i, x in enumerate(gathered)
        )
        print(f"  [rank-ms] {label}: {per_rank}", flush=True)

    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())

def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"]))
    torch.cuda.set_device(local_rank)

    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    is_chief = local_rank == 0
    mod = load_module.load("gemm_ar_blackwell")

    NUM_DEVICES = 8
    if dist.is_initialized():
        NUM_DEVICES = dist.get_world_size()

    # Sweep exactly the comp/comm SM splits this module was built with. A
    # default build compiles one; `make COMP_SM_SWEEP=1 ...` compiles the list
    # in GEMM_AR_FOR_EACH_COMP_SM. Reading it from the module keeps the bench
    # from asking for a split that would TORCH_CHECK at launch.
    COMP_SM_SPLITS = list(mod.compiled_comp_sm_splits())
    AR_UNROLLS = list(mod.compiled_ar_unrolls())
    SIGNAL_DEPTHS = list(mod.compiled_signal_depths())
    # Only sweep strategies that were actually instantiated -- a build can drop
    # one to halve the kernel count once it has been settled.
    _enabled = set(mod.compiled_strategies())
    STRATEGIES = tuple(st for st in ALL_STRATEGIES if st.value in _enabled)
    NUM_BLOCKS = mod.num_blocks()
    n_fused = (len(STRATEGIES) * len(COMP_SM_SPLITS) * len(AR_UNROLLS)
              * len(SIGNAL_DEPTHS))
    if is_chief:
        print(f"comp/comm SM splits compiled in (of {NUM_BLOCKS} blocks): "
              + ", ".join(f"{sm}/{NUM_BLOCKS - sm}" for sm in COMP_SM_SPLITS),
              flush=True)
        print("signal strategies compiled in: "
              + ", ".join(st.name for st in STRATEGIES), flush=True)
        print(f"AR unroll factors compiled in: "
              + ", ".join(str(u) for u in AR_UNROLLS), flush=True)
        print(f"signal pipeline depths compiled in: "
              + ", ".join(str(d) for d in SIGNAL_DEPTHS), flush=True)
        print(f"{n_fused} fused configurations "
              f"({len(STRATEGIES)} strategies x {len(COMP_SM_SPLITS)} splits "
              f"x {len(AR_UNROLLS)} unrolls x {len(SIGNAL_DEPTHS)} depths)",
              flush=True)

    for n in SHAPES:
        M, K, N = n, n // NUM_DEVICES, n

        torch.manual_seed(42 + rank); torch.cuda.manual_seed(42 + rank)
        A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) / (K ** 0.25)
        B = torch.randn((K, N), device="cuda", dtype=torch.bfloat16) / (K ** 0.25)

        C_dbuf = mod.DistBuffer((M, N), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        C_dbuf.data_.zero_()

        barriers = {s: make_barrier(mod, local_rank, world_size) for s in STRATEGIES}

        C_final = mod.DistBuffer((M, N), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        C_final.data_.zero_()

        dist.barrier()

        # collect a run first
        C_ref_cpu = torch.matmul(A, B).detach().float()
        local_ref_cpu = C_ref_cpu.clone()
        dist.all_reduce(C_ref_cpu, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        # Every (strategy, split) is a separate kernel instantiation, so each is
        # validated at least once -- a mis-specialised split would otherwise
        # surface only as a suspiciously fast wrong answer in the sweep table.
        # Full cross-product only at the smallest shape; the per-check host
        # comparison of an MxN tensor is far too slow to repeat at M=32768.
        if n == SHAPES[0]:
            configs_to_check = [(sm, un, sd) for sm in COMP_SM_SPLITS
                                for un in AR_UNROLLS for sd in SIGNAL_DEPTHS]
        else:
            configs_to_check = [(DEFAULT_COMP_SM, default_ar_unroll(n),
                                 DEFAULT_SIGNAL_DEPTH)]
        check_epochs = {st: 0 for st in STRATEGIES}
        for strategy, (comp_sm, unroll, depth) in (
                (st, cfg) for st in STRATEGIES for cfg in configs_to_check):
            # Unlike the timed loop, the outputs ARE cleared between strategies.
            # Both write every element, so leaving the previous strategy's result
            # in place would let a strategy that writes nothing still pass the
            # check. The memset cost is irrelevant outside the timed loop.
            #
            # sync_ranks() brackets the clear: rank r must not zero its C_dbuf
            # while a peer is still multimem.ld_reduce-ing the previous run out
            # of it, and no rank may launch until every peer has finished
            # zeroing.
            sync_ranks()
            C_dbuf.data_.zero_()
            C_final.data_.zero_()
            sync_ranks()

            # do our own run. The barrier is shared across splits for a given
            # strategy and never cleared, so the epoch has to keep rising.
            check_epochs[strategy] += 1
            mod.gemm_ar_intranode_blackwell(
                A, B, C_dbuf, barriers[strategy], C_final,
                check_epochs[strategy], strategy.value, comp_sm, unroll, depth)
            torch.cuda.synchronize()

            tag = f"{strategy.name}/{comp_sm}/u{unroll}/d{depth}"
            gemm_correctness_check = check_close(
                f"gemm M={M} [{tag}]", C_dbuf.data_, local_ref_cpu)

            # C_final is the all-reduced output, so it is checked against the
            # all-reduced reference. The C_dbuf check above is the local GEMM slice,
            # which isolates a comp-side bug from a comm-side one.
            correctness_ok = check_close(
                f"gemm_ar_blackwell M={M} [{tag}]", C_final.data_, C_ref_cpu,
                atol=0.55, rtol=0.12
            )

            if not gemm_correctness_check:
                if is_chief:
                    print(f"{M=} [{tag}] GEMM error :(")
                dist.destroy_process_group()
                return 1
            elif not correctness_ok:
                if is_chief:
                    print(f"{M=} [{tag}] AR Error :(((")
                dist.destroy_process_group()
                return 1

            if is_chief:
                print(f"{M=} [{tag}] correct :)")

        del C_dbuf, C_final, barriers, A, B, C_ref_cpu, local_ref_cpu
        dist.barrier()

    if is_chief:
        print("Correctness checks passed, benchmarking now...")

    for n in SHAPES:
        M, K, N = n, n // NUM_DEVICES, n

        torch.manual_seed(42 + rank); torch.cuda.manual_seed(42 + rank)
        A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) / (K ** 0.25)
        B = torch.randn((K, N), device="cuda", dtype=torch.bfloat16) / (K ** 0.25)

        C_dbuf = mod.DistBuffer((M, N), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        C_dbuf.data_.zero_()

        barriers = {s: make_barrier(mod, local_rank, world_size) for s in STRATEGIES}

        C_final = mod.DistBuffer((M, N), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        C_final.data_.zero_()

        # NVIDIA's CuTeDSL GEMM+AR example, as a second external reference next
        # to cuBLAS+NCCL. It brings its own symmetric-memory buffers, so it is
        # timed but not correctness-checked here -- the upstream example checks
        # itself against the same dist.all_reduce reference we use.
        #
        # Every rank must agree on whether it is in, or the interleave would put
        # one rank in a collective the others are not running. availability() is
        # local, so the decision is all-reduced to a unanimous answer.
        cutlass_ok, cutlass_why = cutlass_dgemm_ar.availability()
        if cutlass_ok:
            try:
                cutlass_run = cutlass_dgemm_ar.build(
                    M=M, N=N, K=K, rank=rank, world_size=world_size,
                    device=local_rank)
            except Exception as exc:
                cutlass_ok, cutlass_why = False, f"{type(exc).__name__}: {exc}"
        vote = torch.tensor([1 if cutlass_ok else 0], device="cuda")
        dist.all_reduce(vote, op=dist.ReduceOp.MIN)
        if not vote.item():
            if cutlass_ok:
                del cutlass_run
            cutlass_ok = False
            if is_chief:
                print(f"  [skip] cutlass CuTeDSL GEMM+AR: {cutlass_why or 'peer opted out'}",
                      flush=True)

        # Drain, line the ranks up, then soak. The sleep only resets temperature
        # if the GPU is already idle when it starts, so it has to come after the
        # synchronize -- otherwise the queue is still draining through it.
        torch.cuda.synchronize()
        dist.barrier()
        time.sleep(5)
        # Every condition is warmed before ANY of them is timed, so none pays
        # another's cold-start cost once the measured loop begins. epochs is
        # carried across warmup and timing because the barriers are never
        # cleared -- see make_barrier.
        epochs = {s: 0 for s in STRATEGIES}

        for _ in range(WARMUP):
            C_tmp = torch.matmul(A, B)
            dist.all_reduce(C_tmp)
            torch.cuda.synchronize()
            del C_tmp

        # Each (strategy, split) is a distinct kernel instantiation, so each
        # needs its own warmup -- otherwise the first timed sample of a split
        # pays module load and cudaFuncSetAttribute.
        for strategy in STRATEGIES:
            for comp_sm in COMP_SM_SPLITS:
                for unroll in AR_UNROLLS:
                    for depth in SIGNAL_DEPTHS:
                        for _ in range(WARMUP):
                            sync_ranks()
                            epochs[strategy] += 1
                            mod.gemm_ar_intranode_blackwell(
                                A, B, C_dbuf, barriers[strategy], C_final,
                                epochs[strategy], strategy.value, comp_sm,
                                unroll, depth)

        if cutlass_ok:
            for _ in range(WARMUP):
                sync_ranks()
                cutlass_run()

        torch.cuda.synchronize()
        dist.barrier()
        time.sleep(5)

        # Interleave the conditions rather than running each to completion.
        # Clock and thermal state drift monotonically across a run, so a fixed
        # order (all baseline, then all PUSH, ...) systematically favours
        # whichever condition runs while the part is coolest -- a gap measured
        # that way cannot be distinguished from drift. Rotating the order within
        # each iteration spreads the drift evenly over every condition instead.
        #
        # Counterbalanced rather than randomised: cycling through all
        # factorial(#conditions) permutations puts every condition in every slot
        # exactly BENCH_ITER / len(ORDERS) times, and balances which condition
        # immediately precedes which. A shuffle only achieves that in
        # expectation, and at these sample counts still leaves visible skew.
        #
        # The order is identical on every rank by construction, which it must
        # be: every condition is a collective, so a rank running PUSH while a
        # peer runs the NCCL baseline would deadlock.
        BASELINE = ("baseline",)
        CUTLASS = ("cutlass",)
        # One condition per (signal strategy, comp/comm SM split). The split is
        # only a launch argument -- the barrier protocol depends on the strategy
        # alone -- so splits share a strategy's barrier and epoch counter, which
        # keeps rising monotonically across all of them.
        conditions = ([BASELINE]
                      + [("fused", st, sm, un, sd)
                         for st in STRATEGIES
                         for sm in COMP_SM_SPLITS
                         for un in AR_UNROLLS
                         for sd in SIGNAL_DEPTHS]
                      + ([CUTLASS] if cutlass_ok else []))
        ORDERS = williams_orders(conditions)
        # Round the target iteration count to a whole number of orders so the
        # balancing is exact rather than approximate.
        iterations = max(1, round(BENCH_ITER / len(ORDERS))) * len(ORDERS)

        samples = {c: [] for c in conditions}

        for it in range(iterations):
            for cond in ORDERS[it % len(ORDERS)]:
                sync_ranks()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                if cond == BASELINE:
                    s.record()
                    C_tmp = torch.matmul(A, B)
                    dist.all_reduce(C_tmp)
                    e.record()
                    # Freed here rather than at loop end so the baseline's
                    # output does not sit on the device through the two fused
                    # runs -- at M=N=32768 that is another 2 GB of headroom.
                    # The caching allocator is stream-ordered, so releasing it
                    # before the recorded work completes is safe.
                    del C_tmp
                elif cond == CUTLASS:
                    s.record()
                    cutlass_run()
                    e.record()
                else:
                    _, strategy, comp_sm, unroll, depth = cond
                    epochs[strategy] += 1
                    s.record()
                    mod.gemm_ar_intranode_blackwell(
                        A, B, C_dbuf, barriers[strategy], C_final,
                        epochs[strategy], strategy.value, comp_sm, unroll, depth)
                    e.record()
                samples[cond].append((s, e))

        torch.cuda.synchronize()
        dist.barrier()
        time.sleep(5)

        # events are only readable once the stream has drained
        if is_chief:
            print(f"M={M} K={K} N={N}", flush=True)

        baseline_ms = median_then_max_cuda(
            elapsed_ms(samples[BASELINE]), label="cublas+nccl")
        fused_ms = {
            (st, sm, un, sd): median_then_max_cuda(
                elapsed_ms(samples[("fused", st, sm, un, sd)]),
                label=f"fused[{st.name}/{sm}/u{un}/d{sd}]")
            for st in STRATEGIES for sm in COMP_SM_SPLITS
            for un in AR_UNROLLS for sd in SIGNAL_DEPTHS
        }
        cutlass_ms = (
            median_then_max_cuda(elapsed_ms(samples[CUTLASS]), label="cutlass")
            if cutlass_ok else None
        )

        # 2*M*K*N per rank for the local GEMM slice
        flops = 2.0 * M * K * N
        if is_chief:
            def tflops(ms):
                return flops / (ms * 1e9) if ms > 0 else float("nan")

            print(f"  {'cublas+nccl':<20}: {baseline_ms:8.3f} ms  "
                  f"({tflops(baseline_ms):7.1f} TFLOP/s)", flush=True)
            if cutlass_ms is not None:
                print(f"  {'cutlass':<20}: {cutlass_ms:8.3f} ms  "
                      f"({tflops(cutlass_ms):7.1f} TFLOP/s)  "
                      f"{baseline_ms / cutlass_ms:6.3f}x vs cublas+nccl",
                      flush=True)

            # Sweep table, sorted fastest first, so the best split is obvious
            # and the shape of the curve is visible next to it.
            print(f"  -- sweep: strategy / comp:comm of {NUM_BLOCKS} / "
                  f"AR unroll / signal depth --",
                  flush=True)
            for (st, sm, un, sd), ms in sorted(fused_ms.items(), key=lambda kv: kv[1]):
                tag = f"{st.name}/{sm}:{NUM_BLOCKS - sm}/u{un}/d{sd}"
                line = (f"  {tag:<24}: {ms:8.3f} ms  ({tflops(ms):7.1f} TFLOP/s)  "
                        f"{baseline_ms / ms:6.3f}x vs cublas+nccl")
                if cutlass_ms is not None and ms > 0:
                    line += f"  {cutlass_ms / ms:6.3f}x vs cutlass"
                print(line, flush=True)

            best = min(fused_ms, key=fused_ms.get)
            best_ms = fused_ms[best]
            baseline_cfg = (best[0], DEFAULT_COMP_SM, default_ar_unroll(M),
                            DEFAULT_SIGNAL_DEPTH)
            msg = (f"  best: {best[0].name} @ {best[1]}:{NUM_BLOCKS - best[1]} "
                   f"unroll={best[2]} depth={best[3]} = {best_ms:.3f} ms")
            if baseline_cfg in fused_ms and fused_ms[baseline_cfg] > 0:
                msg += (f"  ({fused_ms[baseline_cfg] / best_ms:.3f}x vs the "
                        f"{DEFAULT_COMP_SM}/u{baseline_cfg[2]}/d{DEFAULT_SIGNAL_DEPTH} "
                        f"default)")
            if cutlass_ms is not None and best_ms > 0:
                verdict = "BEATS" if best_ms < cutlass_ms else "behind"
                msg += f"  [{verdict} cutlass by {abs(1 - cutlass_ms / best_ms) * 100:.1f}%]"
            print(msg, flush=True)

            if len(STRATEGIES) == 2:
                push_best = min(v for k, v in fused_ms.items()
                                if k[0] is GemmToArSignal.PUSH)
                pull_best = min(v for k, v in fused_ms.items()
                                if k[0] is GemmToArSignal.PULL)
                if pull_best > 0:
                    print(f"  push vs pull  : {push_best / pull_best:8.3f}x "
                          f"(>1 means PULL is faster; best config of each)",
                          flush=True)

        del C_dbuf, barriers, C_final, A, B
        if cutlass_ok:
            del cutlass_run
        dist.barrier()

    dist.destroy_process_group()
    return 0
        

if __name__ == "__main__":
    sys.exit(main())
