import os
import sys
import torch
import torch.distributed as dist
import time
from pathlib import Path
from enum import Enum
from itertools import permutations

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402
from common import check_close

SHAPES= [2048, 4096, 8192, 16384, 32768]
WARMUP = 10
BENCH_ITER = 30

class GemmToArSignal(Enum):
    PUSH = 0
    PULL = 1

# Both signalling strategies are benchmarked (and correctness-checked) on every
# shape, so the runtime knob can be compared head to head.
STRATEGIES = (GemmToArSignal.PUSH, GemmToArSignal.PULL)


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

        for strategy in STRATEGIES:
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

            # do our own run
            mod.gemm_ar_intranode_blackwell(
                A, B, C_dbuf, barriers[strategy], C_final, 1, strategy.value)
            torch.cuda.synchronize()

            tag = strategy.name
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

        for strategy in STRATEGIES:
            for _ in range(WARMUP):
                sync_ranks()
                epochs[strategy] += 1
                mod.gemm_ar_intranode_blackwell(
                    A, B, C_dbuf, barriers[strategy], C_final,
                    epochs[strategy], strategy.value)

        torch.cuda.synchronize()
        dist.barrier()
        time.sleep(5)

        # Interleave the conditions rather than running each to completion.
        # Clock and thermal state drift monotonically across a run, so a fixed
        # order (all baseline, then all PUSH, then all PULL) systematically
        # favours whichever condition runs while the part is coolest. A
        # PUSH-vs-PULL gap measured that way cannot be distinguished from drift.
        # Shuffling the order within each iteration spreads the drift evenly
        # over all three instead of aligning it with one.
        #
        # Counterbalanced rather than randomised: cycling through all 3! = 6
        # permutations puts every condition in every slot exactly the same
        # number of times (BENCH_ITER / 6 each), and also balances which
        # condition immediately precedes which. A shuffle only achieves that in
        # expectation -- at BENCH_ITER=30 it still leaves visible skew, which is
        # the very thing being controlled for.
        #
        # The order is identical on every rank by construction, which it must
        # be: all three conditions are collectives, so a rank running PUSH while
        # a peer runs the NCCL baseline would deadlock.
        BASELINE = "baseline"
        ORDERS = list(permutations([BASELINE, *STRATEGIES]))
        if is_chief and BENCH_ITER % len(ORDERS):
            print(f"  [warn] BENCH_ITER={BENCH_ITER} is not a multiple of "
                  f"{len(ORDERS)}; condition order is only partly balanced",
                  flush=True)

        samples = {BASELINE: []}
        samples.update({s: [] for s in STRATEGIES})

        for it in range(BENCH_ITER):
            for cond in ORDERS[it % len(ORDERS)]:
                sync_ranks()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                if cond is BASELINE:
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
                else:
                    epochs[cond] += 1
                    s.record()
                    mod.gemm_ar_intranode_blackwell(
                        A, B, C_dbuf, barriers[cond], C_final,
                        epochs[cond], cond.value)
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
            s: median_then_max_cuda(
                elapsed_ms(samples[s]), label=f"fused[{s.name}]")
            for s in STRATEGIES
        }

        # 2*M*K*N per rank for the local GEMM slice
        flops = 2.0 * M * K * N
        if is_chief:
            print(
                f"  cublas+nccl   : {baseline_ms:8.3f} ms  "
                f"({flops / (baseline_ms * 1e9):7.1f} TFLOP/s)",
                flush=True,
            )
            for strategy in STRATEGIES:
                ms = fused_ms[strategy]
                speedup = baseline_ms / ms if ms > 0 else float("nan")
                print(
                    f"  fused[{strategy.name:<4}]   : {ms:8.3f} ms  "
                    f"({flops / (ms * 1e9):7.1f} TFLOP/s)  "
                    f"{speedup:6.3f}x vs cublas+nccl",
                    flush=True,
                )
            push_ms = fused_ms[GemmToArSignal.PUSH]
            pull_ms = fused_ms[GemmToArSignal.PULL]
            if pull_ms > 0:
                print(
                    f"  push vs pull  : {push_ms / pull_ms:8.3f}x "
                    f"(>1 means PULL is faster)",
                    flush=True,
                )

        del C_dbuf, barriers, C_final, A, B
        dist.barrier()

    dist.destroy_process_group()
    return 0
        

if __name__ == "__main__":
    sys.exit(main())
