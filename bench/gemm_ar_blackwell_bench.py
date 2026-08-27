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


def variant_tag(variant):
    """(ar_unroll, try_vec) -> the label used in every printed row."""
    unroll, try_vec = variant
    return f"u{unroll}v" if try_vec else f"u{unroll}"
WARMUP = 30
# Target sample count per configuration. The timed loop rounds it to a whole
# number of Williams orders (i.e. a multiple of the condition count) so the
# position balancing stays exact, so the effective count can differ slightly --
# it is printed per shape.
BENCH_ITER = 60
# Cap on how many Williams rotations a run uses. Each iteration runs every
# condition, so using all n rotations costs O(n^2) launches; 8 keeps a wide
# sweep to ~BENCH_ITER iterations while still placing every condition in 8
# distinct positions. Small sweeps are unaffected -- with n <= 8 every rotation
# is used and the balancing is exactly as before.
MAX_ORDERS = 8

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


def reference_config(M, splits, variants, depths):
    """The 'default' configuration, clamped to what this build compiled.

    A build that pins an axis -- e.g. -D'GEMM_AR_FOR_EACH_UNROLL(F)=F(16)' --
    will not contain the shape heuristic's pick, so asking for it
    unconditionally trips the kernel's TORCH_CHECK. Used both to choose which
    config the non-smallest shapes validate, and to label which sweep row was
    the pre-sweep default.
    """
    def pick(want, avail):
        return want if want in avail else avail[0]

    return (pick(DEFAULT_COMP_SM, splits),
            pick((default_ar_unroll(M), 0), variants),
            pick(DEFAULT_SIGNAL_DEPTH, depths))


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
    return stats_then_max_cuda(samples, label)[0]


def stats_then_max_cuda(samples, label=""):
    """As median_then_max_cuda, but also carries that rank's dispersion out.

    Returns (median_ms, iqr_pct) for the rank that set the max, where iqr_pct
    is (p75 - p25) / median * 100. The spread has to come from the *same* rank
    as the median -- mixing a fast rank's spread with a slow rank's centre
    would describe a distribution nobody measured -- so this all_gathers the
    triple and indexes the argmax rather than reducing each field separately.

    Dispersion is not decoration here. The sweep ranks configurations whose
    true separation is around 1%, and picking the argmin over dozens of noisy
    conditions is a max-of-noise estimator: it selects whichever condition got
    lucky, and gets more optimistic the more conditions you add. Printing the
    spread next to the median is what makes that visible instead of inferred.
    """
    sorted_samples = sorted(float(x) for x in samples)
    n = len(sorted_samples)
    median = sorted_samples[n // 2]
    p25 = sorted_samples[max(0, int(0.25 * n))]
    p75 = sorted_samples[min(n - 1, int(0.75 * n))]

    t = torch.tensor([median, p25, p75], dtype=torch.float64, device="cuda")
    gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)

    rows = [g.tolist() for g in gathered]
    slowest = max(range(len(rows)), key=lambda i: rows[i][0])
    med, lo, hi = rows[slowest]

    if label and dist.get_rank() == 0:
        per_rank = " ".join(f"r{i}={r[0]:.3f}" for i, r in enumerate(rows))
        print(f"  [rank-ms] {label}: {per_rank}", flush=True)

    iqr_pct = (hi - lo) / med * 100.0 if med > 0 else float("nan")
    # Standard error of the MEDIAN, which is the statistic actually compared --
    # not the spread of single iterations. For roughly normal samples
    # sigma ~ IQR/1.349 and SE_median ~ 1.2533*sigma/sqrt(n), so
    # SE_median ~ 0.929*IQR/sqrt(n). At IQR 5.8% over 64 samples that is 0.67%,
    # an order of magnitude tighter than the raw IQR -- which is why ranking on
    # the IQR declares everything tied when the medians are in fact separable.
    # Corroborated by cross-run reproducibility: the same cutlass config lands
    # within ~0.7% across independent runs.
    se_pct = 0.929 * iqr_pct / (n ** 0.5) if n > 0 else float("nan")
    return med, iqr_pct, se_pct

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
    # (ar_unroll, try_vec) pairs. try_vec=1 is the vectorised AR inner loop,
    # whose useful unroll range is 8x lower than the scalar path's -- the two
    # lists are paired rather than crossed, so this is one axis, not two.
    AR_VARIANTS = [tuple(v) for v in mod.compiled_ar_variants()]
    SIGNAL_DEPTHS = list(mod.compiled_signal_depths())
    # Only sweep strategies that were actually instantiated -- a build can drop
    # one to halve the kernel count once it has been settled.
    _enabled = set(mod.compiled_strategies())
    STRATEGIES = tuple(st for st in ALL_STRATEGIES if st.value in _enabled)
    NUM_BLOCKS = mod.num_blocks()
    n_fused = (len(STRATEGIES) * len(COMP_SM_SPLITS) * len(AR_VARIANTS)
              * len(SIGNAL_DEPTHS))
    if is_chief:
        print(f"comp/comm SM splits compiled in (of {NUM_BLOCKS} blocks): "
              + ", ".join(f"{sm}/{NUM_BLOCKS - sm}" for sm in COMP_SM_SPLITS),
              flush=True)
        print("signal strategies compiled in: "
              + ", ".join(st.name for st in STRATEGIES), flush=True)
        print(f"AR unroll variants compiled in (v = vectorised): "
              + ", ".join(variant_tag(v) for v in AR_VARIANTS), flush=True)
        print(f"signal pipeline depths compiled in: "
              + ", ".join(str(d) for d in SIGNAL_DEPTHS), flush=True)
        print(f"{n_fused} fused configurations "
              f"({len(STRATEGIES)} strategies x {len(COMP_SM_SPLITS)} splits "
              f"x {len(AR_VARIANTS)} unroll variants x {len(SIGNAL_DEPTHS)} depths)",
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
            configs_to_check = [(sm, var, sd) for sm in COMP_SM_SPLITS
                                for var in AR_VARIANTS for sd in SIGNAL_DEPTHS]
        else:
            configs_to_check = [
                reference_config(n, COMP_SM_SPLITS, AR_VARIANTS, SIGNAL_DEPTHS)]
        check_epochs = {st: 0 for st in STRATEGIES}
        for strategy, (comp_sm, variant, depth) in (
                (st, cfg) for st in STRATEGIES for cfg in configs_to_check):
            unroll, try_vec = variant
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
                check_epochs[strategy], strategy.value, comp_sm, unroll, depth,
                try_vec)
            torch.cuda.synchronize()

            tag = f"{strategy.name}/{comp_sm}/{variant_tag(variant)}/d{depth}"
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
        # Each variant is admitted independently: lamport lives in a different
        # upstream file that a given checkout may not have, and there is no
        # reason a missing one should cost us the other.
        cutlass_runs = {}
        for variant in cutlass_dgemm_ar.VARIANTS:
            ok, why = cutlass_dgemm_ar.availability(variant)
            run = None
            if ok:
                try:
                    run = cutlass_dgemm_ar.build(
                        M=M, N=N, K=K, rank=rank, world_size=world_size,
                        device=local_rank, variant=variant)
                except Exception as exc:
                    ok, why = False, f"{type(exc).__name__}: {exc}"
            # Every rank must agree on whether a variant is in, or the
            # interleave would put one rank in a collective the others are not
            # running. availability() is local, so vote it to unanimous.
            vote = torch.tensor([1 if ok else 0], device="cuda")
            dist.all_reduce(vote, op=dist.ReduceOp.MIN)
            if not vote.item():
                del run
                if is_chief:
                    print(f"  [skip] cutlass[{variant}] CuTeDSL GEMM+AR: "
                          f"{why or 'peer opted out'}", flush=True)
                continue
            cutlass_runs[variant] = run
            if is_chief:
                swz, raster = run.config
                note = ("autotuned" if hasattr(run, "autotune_log")
                        else "matched to SUPERGROUP_WIDTH")
                print(f"  cutlass[{variant}] config: swizzle_size={swz} "
                      f"raster_order={raster} ({note})", flush=True)
                for cswz, craster, cms in getattr(run, "autotune_log", []):
                    print(f"    [autotune] swizzle={cswz} raster={craster}: "
                          f"{cms:8.3f} ms", flush=True)
        # The "vs cutlass" ratio columns stay pinned to the LDMC variant: it is
        # built on the same primitives as ours (multimem ld_reduce/st,
        # reduce-scatter shaped), so that ratio isolates our implementation
        # rather than our choice of algorithm. Lamport is reported alongside and
        # judged separately in the verdict line.

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
                for unroll, try_vec in AR_VARIANTS:
                    for depth in SIGNAL_DEPTHS:
                        for _ in range(WARMUP):
                            sync_ranks()
                            epochs[strategy] += 1
                            mod.gemm_ar_intranode_blackwell(
                                A, B, C_dbuf, barriers[strategy], C_final,
                                epochs[strategy], strategy.value, comp_sm,
                                unroll, depth, try_vec)

        for run in cutlass_runs.values():
            for _ in range(WARMUP):
                sync_ranks()
                run()

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
        CUTLASS_CONDS = {v: ("cutlass", v) for v in cutlass_runs}
        # One condition per (signal strategy, comp/comm SM split). The split is
        # only a launch argument -- the barrier protocol depends on the strategy
        # alone -- so splits share a strategy's barrier and epoch counter, which
        # keeps rising monotonically across all of them.
        conditions = ([BASELINE]
                      + [("fused", st, sm, var, sd)
                         for st in STRATEGIES
                         for sm in COMP_SM_SPLITS
                         for var in AR_VARIANTS
                         for sd in SIGNAL_DEPTHS]
                      + [CUTLASS_CONDS[v] for v in sorted(cutlass_runs)])
        # Use at most MAX_ORDERS of the Williams rotations. Using all n of them
        # makes the balancing perfect -- every condition sits in every position
        # exactly once -- but it forces iterations to be a multiple of n, and
        # since each iteration runs all n conditions the run costs O(n^2)
        # launches. At n=227 that is ~51,500 launches per shape, several minutes
        # of unbroken GPU load, and the thermal drift it induces is far larger
        # than the 1% effects the sweep is trying to resolve: a sweep that heats
        # the part until nothing is separable has not measured anything.
        #
        # k rotations still put every condition in k distinct positions, which
        # removes position bias to the accuracy that matters here, and lets the
        # sample count be chosen for statistics instead of for combinatorics.
        all_orders = williams_orders(conditions)
        ORDERS = all_orders[:MAX_ORDERS]
        # Round up to a whole number of rotations so the balancing stays exact
        # over the rotations actually used.
        iterations = max(1, -(-BENCH_ITER // len(ORDERS))) * len(ORDERS)
        if is_chief and len(ORDERS) < len(all_orders):
            full_iters = max(1, -(-BENCH_ITER // len(all_orders))) * len(all_orders)
            print(f"  [orders] {len(conditions)} conditions, using "
                  f"{len(ORDERS)}/{len(all_orders)} rotations, "
                  f"{iterations} iters -> {iterations * len(conditions)} launches "
                  f"(all {len(all_orders)} rotations: {full_iters} iters -> "
                  f"{full_iters * len(conditions)} launches)", flush=True)

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
                elif cond[0] == "cutlass":
                    run = cutlass_runs[cond[1]]
                    s.record()
                    run()
                    e.record()
                else:
                    _, strategy, comp_sm, (unroll, try_vec), depth = cond
                    epochs[strategy] += 1
                    s.record()
                    mod.gemm_ar_intranode_blackwell(
                        A, B, C_dbuf, barriers[strategy], C_final,
                        epochs[strategy], strategy.value, comp_sm, unroll,
                        depth, try_vec)
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
        fused_stats = {
            (st, sm, var, sd): stats_then_max_cuda(
                elapsed_ms(samples[("fused", st, sm, var, sd)]),
                label=f"fused[{st.name}/{sm}/{variant_tag(var)}/d{sd}]")
            for st in STRATEGIES for sm in COMP_SM_SPLITS
            for var in AR_VARIANTS for sd in SIGNAL_DEPTHS
        }
        fused_ms = {k: v[0] for k, v in fused_stats.items()}
        cutlass_all = {
            v: median_then_max_cuda(elapsed_ms(samples[CUTLASS_CONDS[v]]),
                                    label=f"cutlass[{v}]")
            for v in sorted(cutlass_runs)
        }
        # Ratio columns are against LDMC -- see the note where it is built.
        cutlass_ms = cutlass_all.get("ldmc")

        # 2*M*K*N per rank for the local GEMM slice
        flops = 2.0 * M * K * N
        if is_chief:
            def tflops(ms):
                return flops / (ms * 1e9) if ms > 0 else float("nan")

            print(f"  {'cublas+nccl':<20}: {baseline_ms:8.3f} ms  "
                  f"({tflops(baseline_ms):7.1f} TFLOP/s)", flush=True)
            for v, vms in sorted(cutlass_all.items(), key=lambda kv: kv[1]):
                line = (f"  {'cutlass[' + v + ']':<20}: {vms:8.3f} ms  "
                        f"({tflops(vms):7.1f} TFLOP/s)  "
                        f"{baseline_ms / vms:6.3f}x vs cublas+nccl")
                if cutlass_ms is not None and v != "ldmc":
                    line += f"  {cutlass_ms / vms:6.3f}x vs cutlass[ldmc]"
                print(line, flush=True)

            # Sweep table, sorted fastest first, so the best split is obvious
            # and the shape of the curve is visible next to it.
            print(f"  -- sweep: strategy / comp:comm of {NUM_BLOCKS} / "
                  f"AR unroll / signal depth --",
                  flush=True)
            for (st, sm, var, sd), ms in sorted(fused_ms.items(), key=lambda kv: kv[1]):
                tag = f"{st.name}/{sm}:{NUM_BLOCKS - sm}/{variant_tag(var)}/d{sd}"
                line = (f"  {tag:<24}: {ms:8.3f} ms  ({tflops(ms):7.1f} TFLOP/s)  "
                        f"{baseline_ms / ms:6.3f}x vs cublas+nccl")
                if cutlass_ms is not None and ms > 0:
                    line += f"  {cutlass_ms / ms:6.3f}x vs cutlass"
                line += f"  +/-{fused_stats[(st, sm, var, sd)][1]:4.1f}%"
                print(line, flush=True)

            best = min(fused_ms, key=fused_ms.get)
            best_ms = fused_ms[best]

            # Noise floor and tie band. The argmin above is only meaningful if
            # the gap to the runners-up exceeds the per-condition spread; when
            # it does not, every config inside the band is statistically tied
            # and the "winner" is whichever one got lucky this run. Reporting
            # the band stops a 0.5% reshuffle from being read as a result.
            # Resolution is 2 standard errors of the median (~95%), not the raw
            # per-iteration IQR: the IQR describes single samples, the medians
            # are what get ranked. Using the IQR here declared 13 configs tied
            # when their medians reproduce to well under 1% across runs.
            ses = sorted(v[2] for v in fused_stats.values())
            iqrs = sorted(v[1] for v in fused_stats.values())
            noise_pct = 2.0 * ses[len(ses) // 2]
            tied = sorted(
                (k for k, ms in fused_ms.items()
                 if ms > 0 and (ms - best_ms) / best_ms * 100.0 <= noise_pct),
                key=lambda k: fused_ms[k])
            print(f"  resolution    : +/-{noise_pct:.2f}% (2 SE of the median) "
                  f"over {len(fused_stats)} configs; "
                  f"per-iteration IQR {iqrs[len(iqrs) // 2]:.1f}%; "
                  f"{len(tied)} within resolution of the best", flush=True)
            if len(tied) > 1:
                names = ", ".join(
                    f"{k[0].name}/{k[1]}:{NUM_BLOCKS - k[1]}/{variant_tag(k[2])}/d{k[3]}"
                    for k in tied[:6])
                more = f" (+{len(tied) - 6} more)" if len(tied) > 6 else ""
                print(f"  tied for best : {names}{more}", flush=True)
            ref_sm, ref_var, ref_sd = reference_config(
                M, COMP_SM_SPLITS, AR_VARIANTS, SIGNAL_DEPTHS)
            baseline_cfg = (best[0], ref_sm, ref_var, ref_sd)
            msg = (f"  best: {best[0].name} @ {best[1]}:{NUM_BLOCKS - best[1]} "
                   f"unroll={best[2][0]}{' vec' if best[2][1] else ''} "
                   f"depth={best[3]} = {best_ms:.3f} ms")
            if baseline_cfg in fused_ms and fused_ms[baseline_cfg] > 0:
                msg += (f"  ({fused_ms[baseline_cfg] / best_ms:.3f}x vs the "
                        f"{ref_sm}/{variant_tag(ref_var)}/d{ref_sd} reference)")
            if cutlass_all and best_ms > 0:
                # Judge against the strongest cutlass variant at this shape, not
                # just LDMC -- lamport is a real alternative a reader would run,
                # so beating LDMC while losing to lamport is not a win.
                tough = min(cutlass_all, key=cutlass_all.get)
                tough_ms = cutlass_all[tough]
                verdict = "BEATS" if best_ms < tough_ms else "behind"
                msg += (f"  [{verdict} cutlass[{tough}] by "
                        f"{abs(1 - tough_ms / best_ms) * 100:.1f}%]")
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

            # The question the vec path was added to answer, as one number.
            scalar = [v for k, v in fused_ms.items() if not k[2][1]]
            vec = [v for k, v in fused_ms.items() if k[2][1]]
            if scalar and vec and min(vec) > 0:
                print(f"  scalar vs vec : {min(scalar) / min(vec):8.3f}x "
                      f"(>1 means the vectorised AR is faster; best config "
                      f"of each)", flush=True)

        del C_dbuf, barriers, C_final, A, B
        cutlass_runs.clear()
        cutlass_dgemm_ar.release()
        dist.barrier()

    dist.destroy_process_group()
    return 0
        

if __name__ == "__main__":
    sys.exit(main())
