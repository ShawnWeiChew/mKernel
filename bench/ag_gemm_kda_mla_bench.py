import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402
from common import check_close  # noqa: E402


GLOBAL_M = [2048, 3072, 3584, 4096, 8192, 16384, 32768]
K = 7168

# Eight-way tensor parallel KDA projection width before kernel padding:
#   (4 * 12288 + 96) / 8 + 128 = 6284.
LOGICAL_N = 6284


def round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def padded_n_for_m(m: int) -> int:
    col_block = 128 if m < 4096 else 256
    return round_up(LOGICAL_N, col_block)


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(
        os.environ.get("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"])
    )
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        "nccl", device_id=torch.device(f"cuda:{local_rank}")
    )
    world_size = dist.get_world_size()
    is_chief = rank == 0

    if world_size != 8:
        raise RuntimeError(
            f"This correctness test fixes the logical projection width at "
            f"{LOGICAL_N}, which assumes 8 ranks; got {world_size}."
        )
    if local_world_size != world_size:
        raise RuntimeError(
            "ag_gemm_kda_mla is an intra-node test and requires "
            "LOCAL_WORLD_SIZE == WORLD_SIZE"
        )

    mod = load_module.load("ag_gemm_kda_mla")
    all_correct = True

    for m in GLOBAL_M:
        if m % world_size != 0:
            raise ValueError(f"global M={m} is not divisible by {world_size=}")

        local_m = m // world_size
        padded_n = padded_n_for_m(m)

        # Reference tensors retain the original, unpadded problem shapes.
        torch.manual_seed(42 + rank)
        torch.cuda.manual_seed(42 + rank)
        A_ref_local = torch.randn(
            (local_m, K), device="cuda", dtype=torch.bfloat16
        ) / (K**0.25)
        A_ref = torch.empty((m, K), device="cuda", dtype=torch.bfloat16)
        B_ref = torch.randn(
            (K, LOGICAL_N), device="cuda", dtype=torch.bfloat16
        ) / (K**0.25)
        C_ref = torch.empty(
            (m, LOGICAL_N), device="cuda", dtype=torch.bfloat16
        )

        dist.all_gather_into_tensor(A_ref, A_ref_local)
        torch.mm(A_ref, B_ref, out=C_ref)

        # The modified implementation gets its own tensors. A already meets
        # the kernel's K/row requirements, so only B and C need N padding.
        A_kernel = mod.DistBuffer(
            (local_m, K),
            dtype=torch.bfloat16,
            local_rank=local_rank,
            local_world_size=local_world_size,
            multicast=True,
        )
        A_kernel.data_.copy_(A_ref_local)

        B_kernel = torch.zeros(
            (K, padded_n), device="cuda", dtype=torch.bfloat16
        )
        B_kernel[:, :LOGICAL_N].copy_(B_ref)
        C_kernel = torch.zeros(
            (m, padded_n), device="cuda", dtype=torch.bfloat16
        )

        dist.barrier()
        mod.ag_gemm_kda_mla(A_kernel, B_kernel, C_kernel)
        torch.cuda.synchronize()

        # Ignore the padded output columns and compare the original 6284-wide
        # result against the unpadded PyTorch reference.
        is_correct = check_close(
            f"ag-gemm-kda-mla M={m} N={LOGICAL_N} padded_n={padded_n}",
            C_kernel[:, :LOGICAL_N],
            C_ref,
        )
        all_correct = all_correct and is_correct

        if is_chief:
            status = "passed :)" if is_correct else "FAILED :("
            print(
                f"M={m} local_m={local_m} N={LOGICAL_N} "
                f"padded_n={padded_n}: {status}",
                flush=True,
            )

        del A_ref_local, A_ref, B_ref, C_ref
        del A_kernel, B_kernel, C_kernel
        dist.barrier()

    dist.destroy_process_group()
    return 0 if all_correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
