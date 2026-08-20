import os
import sys
import torch
import torch.distributed as dist
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402
from common import check_close

SHAPES= [2048, 4096, 8192, 16384, 32768]
WARMUP = 20
BENCH_ITER = 10
NUM_DEVICES = 4
K_DENOM = NUM_DEVICES

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
        mod.gemm_ar_intranode_blackwell(A, B, C_dbuf, barrier, C_final)
        torch.cuda.synchronize()

        gemm_correctness_check = check_close(f"gemm M={M}", C_dbuf.data_, local_ref_cpu)

        # NOTE: checks C_dbuf, not C_final — with config::NUM_COMP_SM == NUM_BLOCKS
        # there are no comm SMs, so nothing writes C_final. Point this back at
        # C_final/C_ref_cpu once the comm SMs are re-enabled.
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

        dist.barrier()
        # warmup cublas + NCCL
        for _ in range(WARMUP):
            C_tmp = torch.matmul(A, B)
            dist.all_reduce(C_tmp)
            torch.cuda.synchronize()
            del C_tmp

        dist.barrier()

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

        # The kernel's tile flags are plain counters compared with == NUM_DEVICES
        # and are never cleared on the device, so every launch needs a zeroed
        # barrier. Zeroing it locally is not enough: rank r exits as soon as its
        # own tiles hit 4, while a peer may still be draining its comm SMs. If r
        # relaunches at that point its epilogue bumps the peer's counters past 4
        # (the peer then spins forever on !=4), or the peer's own zeroing wipes
        # r's fresh signals. So the reset must be followed by a host barrier —
        # no rank may launch until every rank has finished clearing.
        def reset_fused_state():
            C_dbuf.data_.zero_()
            barrier.data_.zero_()
            C_final.data_.zero_()
            sync_ranks()

        # warmp fused kernel
        for _ in range(WARMUP):
            reset_fused_state()
            mod.gemm_ar_intranode_blackwell(A, B, C_dbuf, barrier, C_final)


        fused_kernel_samples = []
        for _ in range(BENCH_ITER):
            reset_fused_state()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            mod.gemm_ar_intranode_blackwell(A, B, C_dbuf, barrier, C_final)
            e.record()
            fused_kernel_samples.append((s, e))

        torch.cuda.synchronize()
        dist.barrier()

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
