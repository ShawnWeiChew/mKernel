import os
import sys
import torch
import torch.distributed as dist
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402
from common import check_close

SHAPES= [2048, 4096, 8192, 16384, 32768]
WARMUP = 20
BENCH_ITER = 10

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

        barrier = mod.DistBuffer((2, 1024, 1024), dtype=torch.int,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        barrier.data_.zero_()

        C_final = mod.DistBuffer((M, N), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        C_final.data_.zero_()

        dist.barrier()

        # collect a run first
        C_ref_cpu = torch.matmul(A, B).detach().float()
        local_ref_cpu = C_ref_cpu.clone()
        dist.all_reduce(C_ref_cpu, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        # do our own run
        mod.gemm_ar_intranode_blackwell(A, B, C_dbuf, barrier, C_final, 1)
        torch.cuda.synchronize()

        gemm_correctness_check = check_close(f"gemm M={M}", C_dbuf.data_, local_ref_cpu)

        # C_final is the all-reduced output, so it is checked against the
        # all-reduced reference. The C_dbuf check above is the local GEMM slice,
        # which isolates a comp-side bug from a comm-side one.
        correctness_ok = check_close(
            f"gemm_ar_blackwell M={M}", C_final.data_, C_ref_cpu, atol=0.55, rtol=0.12
        )

        if not gemm_correctness_check:
            if is_chief:
                print(f"{M=} GEMM error :(")
            dist.destroy_process_group()
            return 1
        elif not correctness_ok:
            if is_chief:
                print(f"{M=} AR Error :(((")
            dist.destroy_process_group()
            return 1

        if is_chief:
            print(f"{M=} correct :)")

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

        barrier = mod.DistBuffer((2, 1024, 1024), dtype=torch.int,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        barrier.data_.zero_()

        C_final = mod.DistBuffer((M, N), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        C_final.data_.zero_()

        # Drain, line the ranks up, then soak. The sleep only resets temperature
        # if the GPU is already idle when it starts, so it has to come after the
        # synchronize -- otherwise the queue is still draining through it.
        torch.cuda.synchronize()
        dist.barrier()
        time.sleep(5)
        # warmup cublas + NCCL
        for _ in range(WARMUP):
            C_tmp = torch.matmul(A, B)
            dist.all_reduce(C_tmp)
            torch.cuda.synchronize()
            del C_tmp

        torch.cuda.synchronize()
        dist.barrier()
        time.sleep(5)

        baseline_samples = []
        for _ in range(BENCH_ITER):
            sync_ranks()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            C_tmp = torch.matmul(A, B)
            dist.all_reduce(C_tmp)
            e.record()
            baseline_samples.append((s, e))

        torch.cuda.synchronize()
        dist.barrier()
        time.sleep(5)

        # No per-iteration clearing. The kernel gates each tile on
        # epoch * NUM_DEVICES and never clears the counters on the device, so the
        # barrier only has to be zeroed once at allocation — the rising threshold
        # does the rest. C_dbuf and C_final do not need clearing either: the
        # epilogue TMA-stores every 128x256 tile of C_dbuf and the AR multimem.st
        # writes every element of C_final, so both are fully overwritten each
        # launch. At M=N=32768 that removes ~2 GB of memset per iteration.
        #
        # epoch must keep rising across BOTH loops, since the barrier is not
        # cleared between them: restarting it at 1 for the timed loop would gate
        # on NUM_DEVICES while the counters already sit at WARMUP * NUM_DEVICES,
        # so every wait would fall through instantly and the AR would read tiles
        # the GEMM had not written yet.
        #
        # sync_ranks() stays. It is not about the barrier any more — it is what
        # stops rank r from starting launch k+1 and overwriting C_dist[r] while a
        # peer is still multimem.ld_reduce-ing epoch k out of it.
        epoch = 0

        # warmup fused kernel
        for _ in range(WARMUP):
            sync_ranks()
            epoch += 1
            mod.gemm_ar_intranode_blackwell(A, B, C_dbuf, barrier, C_final, epoch)
        torch.cuda.synchronize()
        dist.barrier()
        time.sleep(5)


        fused_kernel_samples = []
        for _ in range(BENCH_ITER):
            sync_ranks()
            epoch += 1
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            mod.gemm_ar_intranode_blackwell(A, B, C_dbuf, barrier, C_final, epoch)
            e.record()
            fused_kernel_samples.append((s, e))

        torch.cuda.synchronize()
        dist.barrier()
        time.sleep(5)

        # events are only readable once the stream has drained
        if is_chief:
            print(f"M={M} K={K} N={N}", flush=True)

        baseline_ms = median_then_max_cuda(
            elapsed_ms(baseline_samples), label="cublas+nccl")
        fused_ms = median_then_max_cuda(
            elapsed_ms(fused_kernel_samples), label="fused")

        # 2*M*K*N per rank for the local GEMM slice
        flops = 2.0 * M * K * N
        if is_chief:
            speedup = baseline_ms / fused_ms if fused_ms > 0 else float("nan")
            print(
                f"  cublas+nccl : {baseline_ms:8.3f} ms  "
                f"({flops / (baseline_ms * 1e9):7.1f} TFLOP/s)",
                flush=True,
            )
            print(
                f"  fused       : {fused_ms:8.3f} ms  "
                f"({flops / (fused_ms * 1e9):7.1f} TFLOP/s)",
                flush=True,
            )
            print(f"  speedup     : {speedup:8.3f}x", flush=True)

        del C_dbuf, barrier, C_final, A, B
        dist.barrier()

    dist.destroy_process_group()
    return 0
        

if __name__ == "__main__":
    sys.exit(main())
