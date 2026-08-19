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

base_n = 4096
K_denom = 16

def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"]))
    torch.cuda.set_device(local_rank)

    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    is_chief = local_rank == 0
    mod = load_module.load("gemm_ar_blackwell")

    M, K, N = base_n, base_n // K_denom, base_n

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

    correctness_ok = check_close(
        f"gemm_ar_blackwell M={M}", C_final.data_, C_ref_cpu, atol=0.55, rtol=0.12
    ) 

    if is_chief:
        if not gemm_correctness_check:
            print("GEMM error :(")
        elif not correctness_ok:
            print("AR Error :(")
        else:
            print("Correctness checks passed")

    dist.destroy_process_group()
    if not correctness_ok:
        return 1

    return 0
        

if __name__ == "__main__":
    main()
