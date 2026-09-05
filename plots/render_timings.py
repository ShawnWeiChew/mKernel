#!/usr/bin/env python3
"""Render an in-kernel timing trace as a per-CTA Gantt chart.

    python3 plots/render_timings.py plots/ag_gemm_kda_mla_trace_rank0.npz

The output is two stacked panels, because one linear time axis cannot show
both scales in this kernel: the launch is milliseconds long while the pipeline
structure is microseconds wide.

  overview  every CTA, the whole launch. Load imbalance and the tail.
  detail    a few CTAs over an auto-selected window. Who is waiting on whom.

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
# Leaf phases: each brackets exactly one call, so within a role they partition
# the warp's time into wait vs work with nothing unaccounted for.
PROD_LEAVES = [
    ("PROD/k", "PROD_K_BEGIN", "PROD_K_STAGE_READY", "#dc2626", "prod: wait stage", "payload"),
    ("PROD/k", "PROD_K_STAGE_READY", "PROD_K_ISSUED", "#60a5fa", "prod: issue tma", "payload"),
]
MMA_LEAVES = [
    ("MMA", "MMA_TILE_BEGIN", "MMA_TMEM_READY", "#ef4444", "mma: wait tmem", "payload"),
    ("MMA/k", "MMA_K_BEGIN", "MMA_K_INPUT_READY", "#f97316", "mma: wait tma", "payload"),
    ("MMA/k", "MMA_K_INPUT_READY", "MMA_K_ISSUED", "#059669", "mma: issue mma-k", "payload"),
]
EPI_LEAVES = [
    ("EPI", "EPI_TILE_BEGIN", "EPI_MMA_READY", "#f59e0b", "epi: wait mma", "payload"),
    ("EPI", "EPI_MMA_READY", "EPI_TMEM_LOADED", "#8b5cf6", "epi: tmem -> reg", "payload"),
    ("EPI", "EPI_TMEM_LOADED", "EPI_TILE_DONE", "#0891b2", "epi: store C", "payload"),
    ("EPI", "EPI_DRAIN_BEGIN", "EPI_DRAIN_DONE", "#64748b", "epi: drain", "payload"),
]
LEAF_PHASES = PROD_LEAVES + MMA_LEAVES + EPI_LEAVES

# Containers wrap a whole K reduction, so they cover ~100% of the timeline with
# no gaps and cannot show waiting. Kept for the per-tile statistics in the
# summary table; never drawn.
CONTAINER_PHASES = [
    ("CTA", "CTA_BEGIN", "CTA_END", "#cbd5e1", "cta: alive", "payload"),
    ("PROD", "PROD_TILE_BEGIN", "PROD_TILE_DONE", "#3b82f6", "prod: load tile", "payload"),
    ("MMA", "MMA_TMEM_READY", "MMA_TILE_DONE", "#10b981", "mma: issue mma", "payload"),
]

# (title, phases) for the occupancy panels, one per warp role.
ROLE_GROUPS = [
    ("producer warp (TMA loads)", PROD_LEAVES),
    ("mma warp (tcgen05 issue)", MMA_LEAVES),
    ("epilogue warpgroup", EPI_LEAVES),
]

LANE_ORDER = ["PROD/k", "MMA", "MMA/k", "EPI"]

# The build flag that supplies each phase group, named in the missing-event
# warning so a blank lane points at its own fix.
FINE_EVENT_HINT = (
    "built with PROFILE_COARSE=1 -- rebuild with `make PROFILE=1` for these"
)

FIG_WIDTH_IN = 18.0
DPI = 130
PIXELS = FIG_WIDTH_IN * DPI

DEFAULT_DETAIL_BLOCKS = 8


def parse_block_filter(spec, num_blocks):
    """"0-15,20" -> {0..15, 20}. None means every block."""
    if not spec:
        return set(range(num_blocks))
    blocks = set()
    for part in str(spec).split(","):
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
    """Pair begin/end events into per-phase span arrays.

    Returns (spans, missing) where each span entry is
    (lane, blocks, starts_ns, durs_ns, color, label) and `missing` names the
    phases whose events are absent from the trace entirely.
    """
    missing = []
    if records.shape[0] == 0:
        return [], [p[4] for p in phases]
    rec = records[np.isin(records[:, 0], np.fromiter(block_filter, np.int64))]
    if rec.shape[0] == 0:
        return [], [p[4] for p in phases]
    blk = rec[:, 0].astype(np.int64)
    ts = rec[:, 1].astype(np.int64)
    eid = rec[:, 2].astype(np.int64)
    pld = rec[:, 3].astype(np.int64) & 0xFFFFFFFF

    spans = []
    for lane, start_name, end_name, color, label, mode in phases:
        sid = name_to_id.get(start_name)
        eid_v = name_to_id.get(end_name)
        if sid is None or eid_v is None:
            missing.append(label)
            continue
        s_mask = eid == sid
        e_mask = eid == eid_v
        if not s_mask.any() or not e_mask.any():
            missing.append(label)
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
                missing.append(label)
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
        if not keep.any():
            missing.append(label)
            continue
        spans.append((lane, b_out[keep], t0[keep], durs[keep], color, label))
    return spans, missing


def clip_spans(spans, lo_ns, hi_ns):
    """Clip spans to [lo, hi]. A span crossing the window is truncated, not
    dropped -- otherwise a long bar spanning the whole window disappears."""
    out = []
    for lane, b, t0, dur, color, label in spans:
        t1 = t0 + dur
        keep = (t1 > lo_ns) & (t0 < hi_ns)
        if not keep.any():
            continue
        cs = np.clip(t0[keep], lo_ns, hi_ns)
        ce = np.clip(t1[keep], lo_ns, hi_ns)
        d = ce - cs
        nz = d > 0
        if not nz.any():
            continue
        out.append((lane, b[keep][nz], cs[nz], d[nz], color, label))
    return out


def auto_window(spans, t_lo, t_hi, target_bars=45.0):
    """Pick a detail window wide enough to hold ~`target_bars` of the finest
    recurring phase, centred midway through the launch.

    The finest phase is what sets readability: a window sized off the coarse
    phases leaves the interesting ones sub-pixel.
    """
    medians = [
        float(np.median(dur))
        for _, _, _, dur, _, _ in spans
        if dur.size >= 8 and float(np.median(dur)) >= 500.0  # ignore <0.5us
    ]
    total = t_hi - t_lo
    if not medians:
        width = total
    else:
        width = min(total, max(min(medians) * target_bars, 20_000.0))
    centre = t_lo + total * 0.5
    lo = max(t_lo, centre - width / 2)
    return lo, min(t_hi, lo + width)


def pick_detail_blocks(spans, n):
    """Prefer contiguous blocks that carry MMA spans -- the MMA warp only runs
    on cta_rank 0, so an arbitrary slice can miss it entirely."""
    all_blocks = set()
    mma_blocks = set()
    for lane, b, _, _, _, _ in spans:
        u = set(int(x) for x in np.unique(b))
        all_blocks |= u
        if lane.startswith("MMA"):
            mma_blocks |= u
    if not all_blocks:
        return set()
    start = min(mma_blocks) if mma_blocks else min(all_blocks)
    ordered = sorted(x for x in all_blocks if x >= start)
    return set(ordered[:n])


def summarize(spans, kernel_ms, missing):
    lines = [
        f"{'phase':<20}{'n':>9}{'mean us':>10}{'p50 us':>10}{'max us':>10}"
        f"{'total us':>11}"
    ]
    span_min = span_max = None
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
    if missing:
        lines.append("")
        lines.append(
            f"WARNING: no events in this trace for: {', '.join(missing)}"
        )
        if any(m.endswith(("wait stage", "issue tma", "wait tma", "issue mma-k"))
               for m in missing):
            lines.append(f"         the K-step phases are {FINE_EVENT_HINT}")
    return "\n".join(lines)


def subpixel_report(spans, window_ns, label):
    """Name the phases that cannot be seen at this zoom, with the fix."""
    us_per_px = (window_ns / 1000.0) / PIXELS
    invisible = []
    for _, _, _, dur, _, lab in spans:
        med_us = float(np.median(dur)) / 1000.0
        if med_us < us_per_px:
            invisible.append((lab, med_us))
    if not invisible:
        return None
    body = ", ".join(f"{lab} (p50 {v:.3f}us)" for lab, v in invisible)
    return (
        f"NOTE: on the {label} panel 1 px = {us_per_px:.3f} us, so these are "
        f"sub-pixel and will not render: {body}\n"
        f"      zoom in with --tmin/--tmax (us, relative to trace start)."
    )


def coverage(a, b, edges):
    """Exact time covered by spans [a,b) inside each bin of `edges`.

    O(n log n + bins) instead of bins x spans: S(x) = total span-time before x
    is closed-form from sorted starts/ends, and each bin is a difference of S.
    Exactness matters -- snapping short spans to a grid would erase precisely
    the sub-microsecond phases this panel exists to show.
    """
    a_s = np.sort(a)
    order_b = np.argsort(b)
    b_s = b[order_b]
    a_by_b = a[order_b]
    z = np.zeros(1)
    cum_a = np.concatenate([z, np.cumsum(a_s)])
    cum_dur_b = np.concatenate([z, np.cumsum(b_s - a_by_b)])
    cum_a_by_b = np.concatenate([z, np.cumsum(a_by_b)])
    ia = np.searchsorted(a_s, edges, side="left")   # #{a < x}
    ib = np.searchsorted(b_s, edges, side="right")  # #{b <= x}
    S = cum_dur_b[ib] + (ia - ib) * edges - (cum_a[ia] - cum_a_by_b[ib])
    return np.diff(S)


def occupancy_stack(spans, edges, origin):
    """Per-phase mean number of CTAs inside that phase, per time bin.

    Everything is shifted to `origin` first: %globaltimer values are ~1e15 ns,
    and cumsum-ing 350k of them overflows float64's significant digits, which
    silently turns the whole panel into noise.
    """
    rel_edges = edges - origin
    width = np.diff(rel_edges)
    out = []
    for _, _, t0, dur, color, label in spans:
        a = t0.astype(np.float64) - origin
        out.append((label, color, coverage(a, a + dur, rel_edges) / width))
    return out


def assign_rows(spans, group_by):
    keys = set()
    for lane, blocks, _, _, _, _ in spans:
        keys.update((lane, int(b)) for b in np.unique(blocks))
    if not keys:
        return {}, 0

    def lane_rank(lane):
        return LANE_ORDER.index(lane) if lane in LANE_ORDER else len(LANE_ORDER)

    if group_by == "role":
        ordered = sorted(keys, key=lambda k: (lane_rank(k[0]), k[1]))
    else:
        ordered = sorted(keys, key=lambda k: (k[1], lane_rank(k[0])))

    row, y, prev = {}, 0, None
    for key in ordered:
        group = key[0] if group_by == "role" else key[1]
        if prev is not None and group != prev:
            y += 2  # blank rows between groups
        row[key] = y
        y += 1
        prev = group
    return row, y


def draw_panel(ax, spans, group_by, t_lo, t_hi, title, show_lane_labels=True):
    """One PolyCollection per label keeps 100k+ bars fast to draw."""
    row, nrows = assign_rows(spans, group_by)
    by_label = {}
    for lane, blocks, t0, durs, color, label in spans:
        ys = np.array([row[(lane, int(b))] for b in blocks], dtype=np.float64)
        xs = (t0 - t_lo) / 1000.0
        ws = durs / 1000.0
        e = by_label.setdefault(label, [color, [], [], []])
        e[1].append(ys)
        e[2].append(xs)
        e[3].append(ws)

    h = 0.82
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
        # Rasterized: a vector PDF of 100k+ bars is either huge or unscrollable.
        ax.add_collection(
            PolyCollection(
                verts,
                facecolors=color,
                edgecolors="none",
                linewidths=0,
                rasterized=True,
            )
        )

    if show_lane_labels and group_by == "role":
        for lane in LANE_ORDER:
            ys = [v for (ln, _), v in row.items() if ln == lane]
            if not ys:
                continue
            ax.text(
                -0.012,
                (min(ys) + max(ys)) / 2,
                lane,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontweight="bold",
                fontsize=9,
            )

    ax.set_xlim(0, (t_hi - t_lo) / 1000.0)
    ax.set_ylim(nrows, -1)  # inverted: first CTA on top
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(title, fontsize=10, loc="left")
    return by_label


def legend_handles(spans, by_label):
    counts, totals = {}, {}
    for _, _, _, durs, _, label in spans:
        counts[label] = counts.get(label, 0) + durs.size
        totals[label] = totals.get(label, 0.0) + durs.sum() / 1000.0
    return [
        Patch(
            facecolor=by_label[label][0],
            label=f"{label} (n={counts[label]} "
            f"avg={totals[label] / counts[label]:.2f}us)",
        )
        for label in by_label
    ]


def render_trace(
    trace_path,
    out_path=None,
    blocks=None,
    detail_blocks=None,
    group_by="role",
    tmin=None,
    tmax=None,
    bins=240,
    quiet=False,
    **_ignored,
):
    """Render a .npz trace to a PDF. Returns the output path.

    Two views, because a Gantt only works at one scale. Over a multi-millisecond
    launch every span is either ~100% of the timeline (a solid band) or
    sub-pixel (invisible), so the macro view is an occupancy stack instead:
    how many CTAs are sitting in each phase, over time. The Gantt is kept for
    an auto-selected window where individual bars are actually resolvable.
    """
    trace_path = Path(trace_path)
    data = np.load(trace_path, allow_pickle=False)
    records = data["records"]
    name_to_id = {
        str(n): int(i) for n, i in zip(data["event_names"], data["event_ids"])
    }
    num_blocks = (
        int(data["num_blocks"])
        if "num_blocks" in data
        else (int(records[:, 0].max()) + 1 if records.size else 0)
    )
    kernel_ms = float(data["kernel_ms"]) if "kernel_ms" in data else None

    keep_blocks = parse_block_filter(blocks, num_blocks)
    leaves, missing = build_spans(records, name_to_id, LEAF_PHASES, keep_blocks)
    containers, _ = build_spans(
        records, name_to_id, CONTAINER_PHASES, keep_blocks
    )

    log = [summarize(leaves + containers, kernel_ms, missing)]
    if not leaves:
        log.append("")
        log.append(
            "no leaf spans to draw -- only the tile containers are present, "
            "and those cover the whole timeline with no gaps."
        )
        if not quiet:
            print("\n".join(log))
        return None

    all_spans = leaves + containers
    t_lo = min(int(t0.min()) for _, _, t0, _, _, _ in all_spans)
    t_hi = max(int((t0 + d).max()) for _, _, t0, d, _, _ in all_spans)

    if tmin is not None or tmax is not None:
        d_lo = t_lo + int((tmin or 0.0) * 1000)
        d_hi = t_lo + int(tmax * 1000) if tmax is not None else t_hi
    else:
        d_lo, d_hi = auto_window(leaves, t_lo, t_hi)

    d_blocks = (
        parse_block_filter(detail_blocks, num_blocks)
        if detail_blocks
        else pick_detail_blocks(leaves, DEFAULT_DETAIL_BLOCKS)
    )
    detail = clip_spans(leaves, d_lo, d_hi)
    if d_blocks:
        keep = np.fromiter(d_blocks, np.int64)
        detail = [
            (lane, b[m], t0[m], d[m], c, lab)
            for lane, b, t0, d, c, lab in detail
            for m in [np.isin(b, keep)]
            if m.any()
        ]

    if detail:
        note = subpixel_report(detail, d_hi - d_lo, "detail")
        if note:
            log.append("")
            log.append(note)

    # --- figure ----------------------------------------------------------
    groups = [
        (title, [s for s in leaves if s[5] in {p[4] for p in phases}])
        for title, phases in ROLE_GROUPS
    ]
    groups = [g for g in groups if g[1]]
    _, n_det = assign_rows(detail, group_by) if detail else ({}, 0)
    h_det = float(np.clip(n_det * 0.16, 3.0, 7.0)) if detail else 0.0
    heights = [1.45] * len(groups) + ([h_det] if detail else [])

    fig, axes = plt.subplots(
        len(heights),
        1,
        figsize=(FIG_WIDTH_IN, sum(heights) + 1.8),
        gridspec_kw={"height_ratios": heights},
    )
    axes = np.atleast_1d(axes)

    meta = []
    for key, fmt in (
        ("problem_m", "M={}"),
        ("problem_n", "N={}"),
        ("problem_k", "K={}"),
        ("rank", "rank={}"),
    ):
        if key in data:
            meta.append(fmt.format(data[key]))
    fig.suptitle(f"{trace_path.name}   " + "   ".join(meta), fontsize=11)

    edges = np.linspace(float(t_lo), float(t_hi), bins + 1)
    x_us = (edges[:-1] + np.diff(edges) / 2 - t_lo) / 1000.0
    for ax, (title, gspans) in zip(axes, groups):
        stack = occupancy_stack(gspans, edges, float(t_lo))
        ax.stackplot(
            x_us,
            *[v for _, _, v in stack],
            colors=[c for _, c, _ in stack],
            labels=[lab for lab, _, _ in stack],
            edgecolor="none",
        )
        ax.set_xlim(0, (t_hi - t_lo) / 1000.0)
        peak = max((v.max() for _, _, v in stack), default=1.0)
        ax.set_ylim(0, peak * 1.45)
        ax.set_ylabel("CTAs", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=9, loc="left")
        ax.legend(loc="upper right", fontsize=7, ncol=len(stack), framealpha=0.9)
        ax.axvspan(
            (d_lo - t_lo) / 1000.0,
            (d_hi - t_lo) / 1000.0,
            color="black",
            alpha=0.10,
            lw=0,
        )
    axes[len(groups) - 1].set_xlabel(
        f"time (us) — shaded band is the detail window below", fontsize=9
    )

    if detail:
        ax = axes[-1]
        by_label = draw_panel(
            ax,
            detail,
            group_by,
            d_lo,
            d_hi,
            f"detail — CTAs {min(d_blocks)}..{max(d_blocks)}, "
            f"t = {(d_lo - t_lo) / 1000.0:.1f}..{(d_hi - t_lo) / 1000.0:.1f} us "
            f"(--tmin/--tmax to move)",
        )
        ax.set_xlabel(f"time (us), offset {(d_lo - t_lo) / 1000.0:.1f} us")
        ax.legend(
            handles=legend_handles(detail, by_label),
            loc="lower right",
            bbox_to_anchor=(1.0, 1.005),
            fontsize=8,
            ncol=5,
            frameon=False,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path = Path(out_path) if out_path else trace_path.with_suffix(".pdf")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    log.append("")
    log.append(f"wrote {out_path}")
    if not quiet:
        print("\n".join(log))
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path, help="path to a .npz trace")
    ap.add_argument("--out", type=Path, default=None, help="output PDF/PNG")
    ap.add_argument("--blocks", default=None, help='overview filter, e.g. "0-15,20"')
    ap.add_argument(
        "--detail-blocks",
        default=None,
        help="blocks for the detail panel (default: first 8 with MMA activity)",
    )
    ap.add_argument(
        "--tmin", type=float, default=None,
        help="detail window start, us relative to trace start",
    )
    ap.add_argument(
        "--tmax", type=float, default=None,
        help="detail window end, us relative to trace start",
    )
    ap.add_argument("--group-by", choices=["role", "block"], default="role")
    ap.add_argument(
        "--bins", type=int, default=240, help="time bins in the occupancy panels"
    )
    args = ap.parse_args()
    render_trace(
        args.trace,
        out_path=args.out,
        blocks=args.blocks,
        detail_blocks=args.detail_blocks,
        group_by=args.group_by,
        tmin=args.tmin,
        tmax=args.tmax,
        bins=args.bins,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
