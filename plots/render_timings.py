#!/usr/bin/env python3
"""Render an in-kernel timing trace as a per-CTA Gantt chart.

    python3 plots/render_timings.py plots/ag_gemm_kda_mla_trace_rank0.npz

Traces are self-describing (they carry their own event/role name maps), so an
old .npz still renders after the kernel's enum has grown, and you can iterate
on colors and labels without re-running the kernel.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# A phase is a labelled span between two events, paired by payload. `lane`
# names the sub-row a phase draws on: the three warp roles run concurrently
# inside one CTA, so each gets its own row rather than overlapping bars.
#
#   (lane, begin_event, end_event, color, label, pair_by)
TILE_PHASES = [
    ("PROD", "PROD_TILE_BEGIN", "PROD_TILE_DONE", "#3b82f6", "prod: load tile", "payload"),
    ("MMA", "MMA_TILE_BEGIN", "MMA_TMEM_READY", "#ef4444", "mma: wait tmem", "payload"),
    ("MMA", "MMA_TMEM_READY", "MMA_TILE_DONE", "#10b981", "mma: issue mma", "payload"),
    ("EPI", "EPI_TILE_BEGIN", "EPI_MMA_READY", "#f59e0b", "epi: wait mma", "payload"),
    ("EPI", "EPI_MMA_READY", "EPI_TMEM_LOADED", "#8b5cf6", "epi: tmem -> reg", "payload"),
    ("EPI", "EPI_TMEM_LOADED", "EPI_TILE_DONE", "#0891b2", "epi: store C", "payload"),
    ("EPI", "EPI_DRAIN_BEGIN", "EPI_DRAIN_DONE", "#64748b", "epi: drain", "payload"),
]

# Only present when the .so was built with PROFILE_FINE=1.
FINE_PHASES = [
    ("PROD/k", "PROD_K_BEGIN", "PROD_K_STAGE_READY", "#fca5a5", "prod: wait stage", "payload"),
    ("PROD/k", "PROD_K_STAGE_READY", "PROD_K_ISSUED", "#93c5fd", "prod: issue tma", "payload"),
    ("MMA/k", "MMA_K_BEGIN", "MMA_K_INPUT_READY", "#fdba74", "mma: wait tma", "payload"),
    ("MMA/k", "MMA_K_INPUT_READY", "MMA_K_ISSUED", "#6ee7b7", "mma: issue mma-k", "payload"),
]

CTA_PHASE = [("CTA", "CTA_BEGIN", "CTA_END", "#cbd5e1", "cta: alive", "payload")]

LANE_ORDER = ["CTA", "PROD", "PROD/k", "MMA", "MMA/k", "EPI"]

ROLE_SHIFT = 28  # payload layout is (role << 28) | sequence


def parse_block_filter(spec: str | None, num_blocks: int) -> set[int]:
    """"0-15,20" -> {0..15, 20}. None means every block."""
    if not spec:
        return set(range(num_blocks))
    blocks: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            blocks.update(range(int(lo), int(hi) + 1))
        else:
            blocks.add(int(part))
    return blocks


def build_spans(records, name_to_id, phases, block_filter):
    """Pair begin/end events into (lane, block, start_ns, dur_ns, color, label).

    Fully vectorized per phase: no Python loop over records.
    """
    if records.shape[0] == 0:
        return []
    rec = records[np.isin(records[:, 0], np.fromiter(block_filter, np.int64))]
    if rec.shape[0] == 0:
        return []
    blk = rec[:, 0].astype(np.int64)
    ts = rec[:, 1].astype(np.int64)
    eid = rec[:, 2].astype(np.int64)
    pld = rec[:, 3].astype(np.int64) & 0xFFFFFFFF

    spans = []
    for lane, start_name, end_name, color, label, mode in phases:
        sid = name_to_id.get(start_name)
        eid_v = name_to_id.get(end_name)
        if sid is None or eid_v is None:
            continue
        s_mask = eid == sid
        e_mask = eid == eid_v
        if not s_mask.any() or not e_mask.any():
            continue

        if mode == "payload":
            # 64-bit key = (block << 32) | payload. The payload already carries
            # a 4-bit role tag, so spans from warps sharing a CTA head never
            # pair against each other.
            s_key = (blk[s_mask] << 32) | pld[s_mask]
            e_key = (blk[e_mask] << 32) | pld[e_mask]
            s_ts, e_ts = ts[s_mask], ts[e_mask]
            order = np.argsort(e_key, kind="stable")
            e_key_s, e_ts_s = e_key[order], e_ts[order]
            idx = np.searchsorted(e_key_s, s_key)
            in_range = idx < e_key_s.size
            ok = np.zeros(s_key.size, dtype=bool)
            ok[in_range] = e_key_s[idx[in_range]] == s_key[in_range]
            t0 = s_ts[ok]
            t1 = e_ts_s[idx[ok]]
            b_out = blk[s_mask][ok]
        elif mode == "preceding":
            # End carries no matching payload: pair each end with the most
            # recent begin in the same CTA.
            b_l, t0_l, t1_l = [], [], []
            for b in np.unique(np.concatenate([blk[s_mask], blk[e_mask]])):
                sts = np.sort(ts[s_mask & (blk == b)])
                ets = np.sort(ts[e_mask & (blk == b)])
                if sts.size == 0 or ets.size == 0:
                    continue
                j = np.searchsorted(sts, ets, side="left") - 1
                valid = j >= 0
                b_l.append(np.full(int(valid.sum()), b, dtype=np.int64))
                t0_l.append(sts[j[valid]])
                t1_l.append(ets[valid])
            if not b_l:
                continue
            b_out = np.concatenate(b_l)
            t0 = np.concatenate(t0_l)
            t1 = np.concatenate(t1_l)
        else:
            raise ValueError(f"unknown pair_by mode {mode!r}")

        durs = t1 - t0
        # Drop zero/negative spans: a begin and end inside one globaltimer tick,
        # or a stray end with no begin. Negatives render as inside-out bars.
        keep = durs > 0
        n = int(keep.sum())
        if n == 0:
            continue
        spans.append(
            (lane, b_out[keep], t0[keep], durs[keep], color, label)
        )
    return spans


def summarize(spans, kernel_ms):
    """Per-phase n / mean / p50 / max, the first thing worth checking."""
    lines = [
        f"{'phase':<20}{'n':>9}{'mean us':>10}{'p50 us':>10}{'max us':>10}"
        f"{'total us':>11}"
    ]
    span_min = None
    span_max = None
    for _, _, t0, durs, _, label in spans:
        us = durs / 1000.0
        lines.append(
            f"{label:<20}{us.size:>9}{us.mean():>10.3f}"
            f"{np.median(us):>10.3f}{us.max():>10.3f}{us.sum():>11.1f}"
        )
        lo, hi = t0.min(), (t0 + durs).max()
        span_min = lo if span_min is None else min(span_min, lo)
        span_max = hi if span_max is None else max(span_max, hi)
    if span_min is not None:
        traced_ms = (span_max - span_min) / 1e6
        lines.append("")
        lines.append(f"traced wall time   {traced_ms:8.3f} ms")
        if kernel_ms is not None:
            gap = 100.0 * (kernel_ms - traced_ms) / kernel_ms
            lines.append(
                f"launch wall time   {kernel_ms:8.3f} ms  "
                f"({gap:+.1f}% outside instrumented spans)"
            )
    return "\n".join(lines)


def plot(spans, out_path, title, group_by):
    """One PolyCollection per (color, label) keeps 100k+ bars fast to draw."""
    if not spans:
        print("no spans to draw")
        return

    # Row assignment. Rows are keyed by (lane, block); group_by decides whether
    # lanes or CTAs are the outer sort key.
    keys = set()
    for lane, blocks, _, _, _, _ in spans:
        keys.update((lane, int(b)) for b in np.unique(blocks))

    def lane_rank(lane):
        return LANE_ORDER.index(lane) if lane in LANE_ORDER else len(LANE_ORDER)

    if group_by == "role":
        ordered = sorted(keys, key=lambda k: (lane_rank(k[0]), k[1]))
    else:
        ordered = sorted(keys, key=lambda k: (k[1], lane_rank(k[0])))

    row = {}
    y = 0
    prev_group = None
    for key in ordered:
        group = key[0] if group_by == "role" else key[1]
        if prev_group is not None and group != prev_group:
            y += 2  # blank rows between groups
        row[key] = y
        y += 1
        prev_group = group

    fig, ax = plt.subplots(figsize=(18, max(4, y * 0.11)))

    t_zero = min(t0.min() for _, _, t0, _, _, _ in spans)
    by_label = {}
    for lane, blocks, t0, durs, color, label in spans:
        ys = np.array([row[(lane, int(b))] for b in blocks], dtype=np.float64)
        xs = (t0 - t_zero) / 1000.0
        ws = durs / 1000.0
        entry = by_label.setdefault(label, [color, [], [], []])
        entry[1].append(ys)
        entry[2].append(xs)
        entry[3].append(ws)

    h = 0.8
    for label, (color, ys_l, xs_l, ws_l) in by_label.items():
        ys = np.concatenate(ys_l)
        xs = np.concatenate(xs_l)
        ws = np.concatenate(ws_l)
        verts = np.stack(
            [
                np.stack([xs, ys - h / 2], axis=1),
                np.stack([xs + ws, ys - h / 2], axis=1),
                np.stack([xs + ws, ys + h / 2], axis=1),
                np.stack([xs, ys + h / 2], axis=1),
            ],
            axis=1,
        )
        # Rasterized: a vector PDF of 500k bars is either 100 MB or unscrollable.
        ax.add_collection(
            PolyCollection(
                verts,
                facecolors=color,
                edgecolors="none",
                linewidths=0,
                rasterized=True,
            )
        )

    if group_by == "role":
        for lane in LANE_ORDER:
            ys = [v for (ln, _), v in row.items() if ln == lane]
            if not ys:
                continue
            ax.text(
                -0.015,
                (min(ys) + max(ys)) / 2,
                lane,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontweight="bold",
                fontsize=9,
            )

    counts, totals = {}, {}
    for _, _, _, durs, _, label in spans:
        counts[label] = counts.get(label, 0) + durs.size
        totals[label] = totals.get(label, 0.0) + durs.sum() / 1000.0
    handles = [
        Patch(
            facecolor=by_label[label][0],
            label=f"{label} (n={counts[label]} "
            f"avg={totals[label] / counts[label]:.2f}us)",
        )
        for label in by_label
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)

    x_max = max(
        float(((t0 - t_zero) + durs).max()) for _, _, t0, durs, _, _ in spans
    )
    ax.set_xlim(0, x_max / 1000.0)
    ax.set_ylim(y, -1)  # inverted: first CTA on top
    ax.set_xlabel("time (us)")
    # Lane labels already sit in the left margin; pad the axis label past them.
    ax.set_ylabel(
        "CTA (block id)" if group_by == "role" else "block / lane",
        labelpad=34 if group_by == "role" else 4,
    )
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path, help="path to a .npz trace")
    ap.add_argument("--out", type=Path, default=None, help="output PDF/PNG")
    ap.add_argument(
        "--blocks", default=None, help='block filter, e.g. "0-15,20"'
    )
    ap.add_argument(
        "--level",
        choices=["tile", "fine", "both"],
        default="both",
        help=(
            "which spans to draw: tile-level phases, the per-reduction-step "
            "phases, or both on separate rows (default: %(default)s)"
        ),
    )
    ap.add_argument(
        "--group-by",
        choices=["role", "block"],
        default="role",
        help=(
            "row order: all rows of a role together, or a CTA's roles "
            "adjacent (default: %(default)s)"
        ),
    )
    ap.add_argument(
        "--no-cta-span",
        action="store_true",
        help="omit the whole-CTA lifetime bar",
    )
    args = ap.parse_args()

    data = np.load(args.trace, allow_pickle=False)
    records = data["records"]
    name_to_id = {
        str(n): int(i)
        for n, i in zip(data["event_names"], data["event_ids"])
    }
    num_blocks = int(data["num_blocks"]) if "num_blocks" in data else (
        int(records[:, 0].max()) + 1 if records.size else 0
    )
    kernel_ms = float(data["kernel_ms"]) if "kernel_ms" in data else None

    phases = []
    if not args.no_cta_span:
        phases += CTA_PHASE
    if args.level in ("tile", "both"):
        phases += TILE_PHASES
    if args.level in ("fine", "both"):
        phases += FINE_PHASES

    blocks = parse_block_filter(args.blocks, num_blocks)
    spans = build_spans(records, name_to_id, phases, blocks)

    print(summarize(spans, kernel_ms))

    meta = []
    for key, fmt in (
        ("problem_m", "M={}"),
        ("problem_n", "N={}"),
        ("problem_k", "K={}"),
        ("rank", "rank={}"),
    ):
        if key in data:
            meta.append(fmt.format(data[key]))
    title = f"{args.trace.name}  " + "  ".join(meta)

    out = args.out or args.trace.with_suffix(".pdf")
    plot(spans, out, title, args.group_by)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
