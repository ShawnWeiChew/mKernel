"""Benchmark mKernel GEMM+RS against cuBLAS+NCCL, CUTLASS and TK on eight Modal B200+ GPUs.

Run from the repo with the Modal CLI installed and authenticated:
    modal run bench/modal_gemm_rs_blackwell_bench.py
    modal run bench/modal_gemm_rs_blackwell_bench.py --shapes 8192 --iters 100

Runs bench/gemm_rs_blackwell_bench.py (correctness checks, then timings and the
per-shape speedups) for M=8192,16384 by default. B200+ may land on B200
(sm_100a) or B300 (sm_103a), so mKernel is built for both and TK is built once
per arch. TK and CUTLASS live in cached image layers; only the mKernel layer
rebuilds when local sources change. The full log is written to --output-dir.
"""

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/opt/mkernel"
TK_ROOT = "/opt/ThunderKittens"
TK_BUILDS = "/opt/tk"
CUTLASS_ROOT = "/opt/cutlass"
# ThunderKittens checkout the local runs were made against.
TK_COMMIT = "be0e7e57e90858dfa2bbeab7296ff252755f8a37"
# Matches the CUTLASS example pinned in gemm_rs_blackwell_bench.py.
CUTLASS_COMMIT = "dcf215af68a2d08d305076c152a06f201728cd53"
# The Makefile's GPU=blackwell targets sm_103a only; that cubin will not load on B200.
FAT_ARCH = "-gencode arch=compute_100a,code=sm_100a -gencode arch=compute_103a,code=sm_103a"
# The CPU build uses the CUDA driver stub only for linking. At runtime,
# Modal supplies the real GPU driver; never put stubs in LD_LIBRARY_PATH.
STUBS = "LIBRARY_PATH=/usr/local/cuda/lib64/stubs"


def fetch_commit(url: str, commit: str, dest: str) -> str:
    return (
        f"git init -q {dest} && git -C {dest} fetch -q --depth 1 {url} {commit} "
        f"&& git -C {dest} checkout -q FETCH_HEAD"
    )


# TK's gemm_rs Makefile only accepts SM100, so drive its common.mk directly to
# also get an SM103 build. A failed build only makes the bench skip TK.
TK_BUILD = " && ".join(
    [
        f"cd {TK_ROOT}/kernels/parallel/gemm_rs",
        "SUFFIX=$(python3 -c \"import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))\")",
        "for arch in SM100 SM103; do "
        f"mkdir -p {TK_BUILDS}/$arch && "
        f"({STUBS} make -f ../../common.mk ARCH=$arch SRC=gemm_rs_b200.cu OUT=_C$SUFFIX "
        "CMD=true CONFIG=pytorch "
        f"&& mv _C$SUFFIX {TK_BUILDS}/$arch/ || echo \"[warn] TK $arch build failed\"); "
        "done",
    ]
)

app = modal.App("mkernel-gemm-rs-blackwell-bench")
image = (
    modal.Image.from_registry("nvidia/cuda:13.2.0-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("build-essential", "git")
    .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cu132")
    .pip_install("numpy", "pybind11", "nvidia-cutlass-dsl[cu13]==4.7.0")
    .env({"CUDA_HOME": "/usr/local/cuda", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1"})
    .run_commands(
        fetch_commit("https://github.com/NVIDIA/cutlass.git", CUTLASS_COMMIT, CUTLASS_ROOT),
        fetch_commit("https://github.com/HazyResearch/ThunderKittens.git", TK_COMMIT, TK_ROOT),
        TK_BUILD,
    )
    .add_local_file(ROOT / "Makefile", f"{REMOTE_ROOT}/Makefile", copy=True)
    .add_local_dir(ROOT / "include", f"{REMOTE_ROOT}/include", copy=True)
    .add_local_file(
        ROOT / "src/gemm_rs_blackwell.cu",
        f"{REMOTE_ROOT}/src/gemm_rs_blackwell.cu", copy=True,
    )
    .run_commands(
        f"{STUBS} make -C {REMOTE_ROOT} "
        "GPU=blackwell CUDA_HOME=/usr/local/cuda INTRA_NUM_DEVICES=8 "
        f'ARCH="{FAT_ARCH}" PYTHON=python3 '
        "BACKEND_LIBS= BACKEND_DEFINES= gemm-rs-blackwell"
    )
)
for relative_path in (
    "bench/gemm_rs_blackwell_bench.py",
    "bench/common.py",
    "python/load_module.py",
    "python/load_thirdparty.py",
):
    image = image.add_local_file(ROOT / relative_path, f"{REMOTE_ROOT}/{relative_path}")


@app.function(image=image, gpu="B200+:8", cpu=16, memory=65536, timeout=3600)
def bench(shapes: str, warmup: int, iters: int) -> tuple[int, str]:
    """Returns (torchrun exit code, combined log)."""
    import os

    import torch

    major, minor = torch.cuda.get_device_capability(0)
    tk_arch = {(10, 0): "SM100", (10, 3): "SM103"}.get((major, minor))
    gpu_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,compute_cap", "--format=csv"],
        capture_output=True, text=True,
    ).stdout
    print(gpu_info, flush=True)

    env = dict(os.environ, CUTLASS_PATH=CUTLASS_ROOT)
    if tk_arch is not None:
        # load_thirdparty falls back to the root itself when kernels/parallel/ is absent.
        env["THUNDERKITTENS_PATH"] = f"{TK_BUILDS}/{tk_arch}"
    command = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", "--nnodes=1", "--nproc-per-node=8", "--max-restarts=0",
        "bench/gemm_rs_blackwell_bench.py",
        "--shapes", shapes, "--warmup", str(warmup), "--iters", str(iters),
    ]
    print("Running:", " ".join(command), flush=True)
    # Stream to the Modal console while keeping a copy for the local log.
    log = [gpu_info]
    with subprocess.Popen(
        command, cwd=REMOTE_ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.append(line)
    return process.returncode, "".join(log)


@app.local_entrypoint()
def main(
    shapes: str = "8192,16384",
    warmup: int = 20,
    iters: int = 50,
    output_dir: str = str(ROOT / "bench/results"),
):
    returncode, log = bench.remote(shapes, warmup, iters)

    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = dest_dir / f"gemm_rs_blackwell_{stamp}.log"
    log_path.write_text(log)
    print(f"wrote {log_path}", flush=True)
    if returncode:
        raise SystemExit(f"remote benchmark exited {returncode}; see {log_path}")
