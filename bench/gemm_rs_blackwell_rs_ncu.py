"""Nsight Compute comparison of the standalone RS lane layouts (32B vs 16B).

    make GPU=blackwell INTRA_NUM_DEVICES=8 gemm-rs-blackwell
    python bench/gemm_rs_blackwell_rs_ncu.py --nproc 8

    # Other points of the RS tuning matrix:
    python bench/gemm_rs_blackwell_rs_ncu.py --tile-rows 128 256 --num-warps 1 8

Same RS setup as gemm_rs_blackwell_rs_bench.py: a multicast BF16 [M,N] DistBuffer
input on every rank and a separate compact [M/world_size,N] output. torchrun
starts all ranks; rank 0 re-execs itself under ncu, so only rank 0 is profiled.
Each variant is launched once on rank 0 while the other ranks idle at a barrier
with their inputs alive, so counters show one rank's traffic without peer
contention (gpu__time is therefore optimistic versus the timed bench). After
profiling, every variant runs on all ranks and is checked exactly against NCCL.

NCU cannot save/restore multicast (NVLS) memory, so kernel replay fails when a
profile needs more than one pass. Metrics are therefore split into small
single-unit groups; each group is profiled in its own torchrun run (one pass,
no replay) to --output_g<i>.ncu-rep, and the reports are merged and summarized:
load instructions/requests/sectors, L2 peer-aperture sectors, and NVLink bytes,
normalized per output byte, with the 32B/16B ratio per (tile, warps) point.
A group that still needs several passes fails alone; its metrics show n/a.

Counter access needs admin rights or NVreg_RestrictProfilingToAdminUsers=0.
"""

import argparse
import csv
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CHILD_ENV = "MKERNEL_RS_NCU_CHILD"
KERNEL_REGEX = "regex:rs_only_kernel"

# Verified with `ncu --query-metrics --chip gb100/gb110` (B200 / B300). Each
# group stays within one hardware unit so it should collect in a single pass.
METRIC_GROUPS = [
    ["gpu__time_duration.sum"],
    ["smsp__sass_inst_executed_op_global_ld.sum",
     "smsp__sass_inst_executed_op_global_red.sum",
     "smsp__sass_inst_executed_op_global_st.sum"],
    ["l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
     "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"],
    ["l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
     "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum"],
    # Together these need two passes (-> memory save, which fails on multicast).
    ["l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"],
    ["l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"],
    ["lts__t_requests_srcunit_tex_op_read.sum",
     "lts__t_sectors_srcunit_tex_op_read.sum",
     "lts__t_sectors_srcunit_tex_op_write.sum"],
    ["syslts__t_requests_aperture_peer.sum",
     "syslts__t_sectors_aperture_peer.sum"],
    ["nvltx__bytes.sum", "nvlrx__bytes.sum"],
    ["nvltx__bytes_data_user.sum", "nvlrx__bytes_data_user.sum"],
    ["dram__bytes_read.sum", "dram__bytes_write.sum"],
]
METRICS = [metric for group in METRIC_GROUPS for metric in group]

# (label, metric, normalize per output byte?)
SUMMARY = [
    ("time (us)", "gpu__time_duration.sum", False),
    ("ld inst", "smsp__sass_inst_executed_op_global_ld.sum", False),
    ("red inst", "smsp__sass_inst_executed_op_global_red.sum", False),
    ("st inst", "smsp__sass_inst_executed_op_global_st.sum", False),
    ("L1 ld req", "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", False),
    ("L1 ld sectors", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", False),
    ("L1 red sectors", "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum", False),
    ("L1 st sectors", "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", False),
    ("L2 read sectors", "lts__t_sectors_srcunit_tex_op_read.sum", False),
    ("peer requests", "syslts__t_requests_aperture_peer.sum", False),
    ("peer sectors", "syslts__t_sectors_aperture_peer.sum", False),
    ("peer B / out B", "syslts__t_sectors_aperture_peer.sum", True),
    ("NVL tx B / out B", "nvltx__bytes.sum", True),
    ("NVL rx B / out B", "nvlrx__bytes.sum", True),
    ("NVL rx user / out B", "nvlrx__bytes_data_user.sum", True),
    ("DRAM rd B / out B", "dram__bytes_read.sum", True),
]


def find_ncu():
    for candidate in (shutil.which("ncu"), Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/ncu"):
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise SystemExit("ncu not found; add it to PATH or set CUDA_HOME")


def variants(args):
    return [(t, w, b) for t in args.tile_rows for w in args.num_warps for b in args.lane_bytes]


def group_report(args, group):
    return args.output.with_name(f"{args.output.name}_g{group}")


def ncu_command(args, ncu):
    # Only explicit metrics (no sections): one pass means no memory save/restore.
    return [
        ncu, "--target-processes", "all", "--kernel-name", KERNEL_REGEX,
        "--launch-count", str(len(variants(args))), "--replay-mode", "kernel",
        "--metrics", ",".join(METRIC_GROUPS[args.metric_group]),
        "-o", str(group_report(args, args.metric_group)), "-f",
    ]


# ----------------------------------------------------------------------------- worker


def worker(args):
    rank = int(os.environ["LOCAL_RANK"])
    if rank == 0 and os.environ.get(CHILD_ENV) != "1":
        # Re-exec rank 0 under ncu before CUDA initializes; same PID for torchrun.
        os.environ[CHILD_ENV] = "1"
        command = ncu_command(args, find_ncu()) + [sys.executable, *sys.argv]
        os.execvp(command[0], command)

    import torch
    import torch.distributed as dist

    sys.path.insert(0, str(ROOT / "python"))
    sys.path.insert(0, str(ROOT / "bench"))
    import load_module
    from gemm_rs_blackwell_rs_bench import check_output, sync_ranks

    world = int(os.environ["WORLD_SIZE"])
    m, n = args.m, args.n
    if m % (world * max(args.tile_rows)) or n % 256:
        raise SystemExit(f"M must divide by {world * max(args.tile_rows)}; N by 256")
    torch.cuda.set_device(rank)
    module = load_module.load("gemm_rs_blackwell")
    if module.gemm_rs_blackwell_world_size != world:
        raise SystemExit(f"Rebuild: make GPU=blackwell INTRA_NUM_DEVICES={world} gemm-rs-blackwell")
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    try:
        source = module.DistBuffer((m, n), dtype=torch.bfloat16, local_rank=rank,
                                   local_world_size=world, multicast=True)
        epochs = module.DistBuffer((1,), dtype=torch.int32, local_rank=rank,
                                   local_world_size=world, multicast=False)
        epochs.data_.zero_()
        output = torch.empty((m // world, n), dtype=torch.bfloat16, device="cuda")
        torch.manual_seed(42 + rank)
        # Small integers sum exactly in BF16, as in the timed bench.
        source.data_.copy_(torch.randint(-4, 5, (m, n), device="cuda"))
        sync_ranks()

        # Profiled launches: rank 0 only, one per variant, in variants() order.
        for tile_rows, warps, lane_bytes in variants(args):
            if rank == 0:
                module.rs_blackwell_launch(source, output, epochs, False, tile_rows, warps, lane_bytes)
            sync_ranks()

        if args.metric_group != 0:
            return
        # Correctness on all ranks (past --launch-count, so not profiled).
        expected = torch.empty_like(output, dtype=torch.float32)
        dist.reduce_scatter_tensor(expected, source.data_.float(), op=dist.ReduceOp.SUM)
        for tile_rows, warps, lane_bytes in variants(args):
            output.fill_(float("nan"))
            module.rs_blackwell_launch(source, output, epochs, False, tile_rows, warps, lane_bytes)
            check_output(output, expected, f"tile={tile_rows} warps={warps} lane={lane_bytes}B",
                         exact=True)
        if rank == 0:
            print(f"M={m} N={n}: all {len(variants(args))} variants match NCCL exactly", flush=True)
        sync_ranks()
    finally:
        dist.destroy_process_group()


# ----------------------------------------------------------------------------- report


def parse_kernel(name):
    """rs_only_kernel<256, 1, 16>(...) -> (256, 1, 16)."""
    inner = name[name.index("<") + 1:name.index(">")]
    return tuple(int(value.strip().rstrip("u")) for value in inner.split(","))


def load_report(ncu, report, metrics):
    text = subprocess.run(
        [ncu, "--import", str(report), "--csv", "--page", "raw", "--print-units", "base",
         "--metrics", ",".join(metrics)],
        check=True, capture_output=True, text=True,
    ).stdout
    rows = list(csv.reader(io.StringIO(text)))
    header = rows[0]
    results = {}
    for row in rows[2:]:  # rows[1] holds units
        record = dict(zip(header, row))
        values = {}
        for metric in metrics:
            raw = record.get(metric, "").replace(",", "")
            try:
                values[metric] = float(raw)
            except ValueError:
                values[metric] = float("nan")
        results[parse_kernel(record["Kernel Name"])] = values
    return results


def fmt(value):
    if value != value:
        return "n/a"
    if abs(value) >= 1e4:
        return f"{value:,.0f}"
    return f"{value:.3f}"


def merged_results(args, ncu):
    nan = {metric: float("nan") for metric in METRICS}
    results = {key: dict(nan) for key in variants(args)}
    for group, metrics in enumerate(METRIC_GROUPS):
        path = group_report(args, group).with_suffix(".ncu-rep")
        if not path.exists():
            print(f"[missing] group {group} report {path}: {', '.join(metrics)} -> n/a")
            continue
        for key, values in load_report(ncu, path, metrics).items():
            results.setdefault(key, dict(nan)).update(values)
    return results


def report(args, ncu):
    results = merged_results(args, ncu)
    out_bytes = args.m // args.nproc * args.n * 2
    print(f"\nNCU rank 0, M={args.m} N={args.n} world={args.nproc}; "
          f"output/rank = {out_bytes / 2**20:.1f} MiB; reports {args.output}_g*.ncu-rep")
    for tile_rows in args.tile_rows:
        for warps in args.num_warps:
            lanes = [b for b in args.lane_bytes if (tile_rows, warps, b) in results]
            if not lanes:
                continue
            print(f"\ntile={tile_rows} warps={warps}")
            columns = [f"lane={b}B" for b in lanes]
            has_ratio = set(lanes) == {16, 32}
            print(f"  {'metric':22s}" + "".join(f"{c:>16s}" for c in columns)
                  + (f"{'32B / 16B':>12s}" if has_ratio else ""))
            for label, metric, per_byte in SUMMARY:
                vals = {}
                for b in lanes:
                    value = results[(tile_rows, warps, b)][metric]
                    if metric == "gpu__time_duration.sum":
                        value /= 1e3
                    if per_byte:
                        value = value * (32 if metric.startswith("syslts__t_sectors") else 1) / out_bytes
                    vals[b] = value
                line = f"  {label:22s}" + "".join(f"{fmt(vals[b]):>16s}" for b in lanes)
                if has_ratio:
                    ratio = vals[32] / vals[16] if vals[16] else float("nan")
                    line += f"{fmt(ratio):>12s}"
                print(line)


# ----------------------------------------------------------------------------- driver


def worker_args(args):
    return [
        "--nproc", str(args.nproc), "--m", str(args.m), "--n", str(args.n),
        "--tile-rows", *map(str, args.tile_rows), "--num-warps", *map(str, args.num_warps),
        "--lane-bytes", *map(str, args.lane_bytes), "--output", str(args.output),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nproc", type=int, default=8, choices=(2, 4, 8), help="GPUs (torchrun ranks)")
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=7168)
    parser.add_argument("--tile-rows", nargs="+", type=int, choices=(128, 256, 512), default=[256])
    parser.add_argument("--num-warps", nargs="+", type=int, choices=(1, 4, 8), default=[1])
    parser.add_argument("--lane-bytes", nargs="+", type=int, choices=(16, 32), default=[32, 16])
    parser.add_argument("--output", type=Path, default=ROOT / "results/rs-ncu/rs_lanes")
    parser.add_argument("--report-only", action="store_true", help="Summarize an existing --output report")
    parser.add_argument("--groups", nargs="+", type=int,
                        help=f"Metric groups to profile (default all 0..{len(METRIC_GROUPS) - 1})")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--metric-group", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("tile_rows", "num_warps", "lane_bytes"):
        setattr(args, name, list(dict.fromkeys(getattr(args, name))))
    args.output = args.output.with_suffix("").resolve()

    if args.worker:
        worker(args)
        return
    ncu = find_ncu()
    if not args.report_only:
        if args.m <= 0 or args.m % (args.nproc * max(args.tile_rows)) or args.n <= 0 or args.n % 256:
            parser.error(f"M must be a positive multiple of {args.nproc * max(args.tile_rows)}; "
                         "N a positive multiple of 256")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        forwarded = worker_args(args)
        env = dict(os.environ)
        env.pop(CHILD_ENV, None)
        failed = []
        groups = args.groups if args.groups is not None else range(len(METRIC_GROUPS))
        for group in groups:
            group_report(args, group).with_suffix(".ncu-rep").unlink(missing_ok=True)
            command = [
                sys.executable, "-m", "torch.distributed.run", "--standalone",
                f"--nproc-per-node={args.nproc}", "--max-restarts=0", "--",
                str(Path(__file__).resolve()), "--worker", "--metric-group", str(group), *forwarded,
            ]
            print(f"\n[group {group}] {', '.join(METRIC_GROUPS[group])}", flush=True)
            if subprocess.run(command, cwd=ROOT, env=env).returncode:
                failed.append(group)
        if failed:
            print(f"\nGroups {failed} failed (still multi-pass or another error); see logs above.")
    report(args, ncu)


if __name__ == "__main__":
    main()
