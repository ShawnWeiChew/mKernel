"""Thin importlib loader for prebuilt .so kernels.

Replaces torch.utils.cpp_extension.load() — no JIT, no env-var lookups,
no ninja. Assumes the Makefile has produced build/lib<name>.so.
"""
from __future__ import annotations
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"


def load(name: str):
    """Load build/lib<name>.so as a Python module.

    Args:
        name: kernel base name (e.g. "dispatch_gemm").

    Returns:
        Python module exposing the kernel's pybind functions.

    Raises:
        RuntimeError if the .so doesn't exist (run `make` first).
    """
    torch_tag = "torch" + torch.__version__.split("+", 1)[0].split("a", 1)[0].replace(".", "_")
    tagged_so = BUILD / f"lib{name}_{torch_tag}.so"
    plain_so = BUILD / f"lib{name}.so"
    candidates = [p for p in (tagged_so, plain_so) if p.exists()]
    so = max(candidates, key=lambda p: p.stat().st_mtime_ns) if candidates else plain_so
    if not so.exists():
        raise RuntimeError(
            f"{so} does not exist. Run `make build/lib{name}.so_sw` (or `make all`) "
            f"from {ROOT} first."
        )
    load_so = so
    if os.environ.get("MKERNEL_STAGE_SO_LOCAL", "1") != "0":
        for attempt in range(10):
            try:
                stat = so.stat()
                break
            except OSError:
                if attempt == 9:
                    raise
                time.sleep(0.2)
        local_dir = Path(os.environ.get("MKERNEL_LOCAL_SO_DIR", "/tmp/mkernel_so"))
        local_dir.mkdir(parents=True, exist_ok=True)
        local_so = local_dir / f"{so.stem}-{stat.st_mtime_ns}-{stat.st_size}{so.suffix}"
        for attempt in range(10):
            try:
                if not local_so.exists() or local_so.stat().st_size != stat.st_size:
                    tmp_so = local_so.with_suffix(local_so.suffix + f".tmp.{os.getpid()}")
                    shutil.copy2(so, tmp_so)
                    os.replace(tmp_so, local_so)
                break
            except OSError:
                if attempt == 9:
                    raise
                time.sleep(0.2)
        load_so = local_so

    mod_name = f"mkernel_release_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, load_so)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create import spec for {load_so}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod
