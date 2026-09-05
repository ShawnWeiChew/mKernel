"""Dump and render the in-kernel timing ring.

Serves every kernel instrumented with include/common/timings.cuh -- currently
gemm_ar_blackwell and ag_gemm_kda_mla. The .npz records which one it came from,
so the right phase table is picked automatically and an old trace keeps decoding
against the events it was actually emitted with.

Three concerns, matching the device side in include/common/timings.cuh:

  unpack()      ring buffer -> flat (block, ts, event_id, payload) table
  save()/load() table       -> self-describing .npz
  plot()        table       -> per-role Gantt PDF

Everything downstream of the flat table is ordinary numpy, so the phase tables
below can be recoloured, relabelled or extended without touching the pairing.

The .npz carries its own event name -> id map and its own warp layout, so a
saved trace re-renders without a GPU or the compiled .so, and still decodes
correctly after the event enum has grown:

    python python/timings.py traces/ag_gemm_kda_mla.npz
    python python/timings.py traces/ag_gemm_kda_mla.npz --collapse
    python python/timings.py traces/ag_gemm_kda_mla.npz --rows-per-role 0
    python python/timings.py traces/ag_gemm_kda_mla.npz --blocks 0-7

Cross-rank note: %globaltimer is device-wide monotonic but is *not*
synchronised across GPUs. Each rank's trace is its own time axis; do not
overlay two ranks without anchoring a shared event to t=0 on both.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Mirrors mkernel_timings::WARP_ID_SHIFT. The ring is partitioned per CTA, not
# per warp, so all warps of a block interleave into one slot; what separates
# them again is the warp id the kernel packs into every payload's top bits,
# which is also what keeps the pairing key unique when two warps are on the
# same sequence number.
WARP_ID_SHIFT = 28
SEQ_MASK = (1 << WARP_ID_SHIFT) - 1

# Fallback warp layout, used only when a trace predates the WARP_LAYOUT export.
# The live values come from the module and are saved into the .npz.
DEFAULT_WARP_LAYOUT = {
    "NUM_WARPS": 12,
    "EPILOGUE_WARPS": 8,
    "PRODUCER_WARP_ID": 8,
    "FIRST_CONSUMER_WARP_ID": 9,
    "CONSUMER_WARPS": 2,
    "WARPGROUP_WARPS": 4,
    "NUM_CLUSTERS": 2,
}

# Row order in the plot, top to bottom. Producer/mma/epilogue are the comp CTAs'
# warp roles; comm is a whole CTA of the all-reduce half of the grid.
ROLE_ORDER = ["producer", "mma", "epilogue", "comm"]

# Band labels, in the kernel's own vocabulary rather than the renderer's:
# the producer warp is the one issuing TMA loads, the comm CTAs are the
# all-reduce.
ROLE_DISPLAY = {
    "producer": "TMA",
    "mma": "MMA",
    "epilogue": "EPILOGUE",
    "comm": "AR",
}


def row_role(block: int, warp: int, num_comp_sm: int, layout: dict) -> str | None:
    """(block, warp) -> role, or None for a row that is not plotted.

    Mirrors the branch at the bottom of fused_comp_sm plus the comp/comm split
    in fused_kernel. Warps that do work but do not stamp their own spans (the
    non-leader epilogue warps, the padding warp) return None: they still emit a
    SETUP record, and dropping them here is what keeps those from becoming rows
    with a single bar.
    """
    if block >= num_comp_sm:
        # Comm CTAs stamp from thread 0 only, so warp is always 0.
        return "comm" if warp == 0 else None
    if warp == layout["PRODUCER_WARP_ID"]:
        return "producer"
    first_consumer = layout["FIRST_CONSUMER_WARP_ID"]
    if first_consumer <= warp < first_consumer + layout["CONSUMER_WARPS"]:
        return "mma"
    # One row per epilogue warpgroup, stamped by its leader.
    if warp < layout["EPILOGUE_WARPS"] and warp % layout["WARPGROUP_WARPS"] == 0:
        return "epilogue"
    return None


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
#
# The split is carried by warmth *and* brightness, not brightness alone:
#
#   work = hot and saturated  (red / orange / amber / yellow)
#   wait = dark and cool      (navy / indigo / purple / forest / teal)
#
# Two channels rather than one is what makes it readable at this bar density --
# a pastel-vs-dark scheme washes out, because pale bars carry no hue signal.
# Hue within each group still separates the individual phases for the legend.
# assert_palette() below enforces both channels so a new phase cannot quietly
# land on the wrong side.
WAIT = {
    "ring": "#1e3a8a",       # producer  -- dark blue
    "tmem": "#4c1d95",       # mma       -- dark purple
    "tma": "#312e81",        # mma       -- dark indigo
    "mainloop": "#14532d",   # epilogue  -- dark green
    "store": "#134e4a",      # epilogue  -- dark teal
    "signal": "#581c87",     # comm      -- purple; the headline stall
    "a_remote": "#0c4a6e",   # ag_gemm   -- dark sky; A from a peer's shard
}
WORK = {
    "tma": "#fdba74",        # producer  -- light orange
    "mma": "#ef4444",        # mma       -- red
    "tmem": "#fbbf24",       # epilogue  -- amber
    "smem": "#facc15",       # epilogue  -- yellow
    "signal": "#fb7185",     # epilogue  -- rose
    "reduce": "#f97316",     # comm      -- orange; the all-reduce actually running
}
# Neither: a one-off prologue bar, kept neutral so it reads as neither a stall
# nor real work.
SETUP_COLOR = "#9ca3af"

# White: dark cool bars and hot saturated bars both read against it, and gaps --
# time in no instrumented phase at all -- stay as plain white space, which is
# where the interesting stalls hide.
CANVAS = "#ffffff"

# Collapsed mode (--collapse) throws hue away entirely and draws exactly two
# colors, which is the shortest path to "how much of this kernel is waiting".
COLLAPSED = {"wait": "#1e3a8a", "work": "#f97316", "setup": SETUP_COLOR}
# Copy-engine waits stay their own colour even when collapsed: they are a
# different kind of stall from a compute-pipeline one, and they are usually so
# short that merging them into the generic wait navy hides them completely.
COLLAPSED_COPY = "#06b6d4"

# A span shorter than this fraction of the x-range is widened to it *for
# drawing only* -- the legend and the summary always report true durations. A
# 391 ns copy wait in a 405 us trace is 0.002 of a pixel, so without a floor a
# real event renders as nothing at all and reads as "it never happened".
MIN_BAR_FRAC = 0.0012

# Instants worth a full-height rule rather than a bar, per kernel. The copy
# engine cannot stamp itself, so the moment a shard lands is the one thing the
# trace can say about it precisely -- and it matters far more than the width of
# the sub-microsecond spin that observed it.
MARKER_EVENTS = {
    "ag_gemm_kda_mla": [("ACOPY_READY", "#0891b2", "A shard from peer {seq} ready")],
}


# ---------------------------------------------------------------------------
# Phase tables
# ---------------------------------------------------------------------------
#
# (begin_event, end_event, color, label, pair_by). Events are named, not
# numbered, so a trace saved before the enum grew still renders.
#
# The kernel emits chained milestones rather than begin/end pairs, so the end of
# one phase is the begin of the next and consecutive rows share an event. Every
# phase pairs by payload: the kernel gives all milestones of one loop iteration
# the same sequence number precisely so that works.
SETUP_PHASES = [
    ("SETUP_BEGIN", "SETUP_DONE", SETUP_COLOR, "setup", "payload"),
]

PRODUCER_PHASES = SETUP_PHASES + [
    ("LOAD_STEP_BEGIN", "LOAD_MMA_FREE", WAIT["ring"], "prod: wait ring slot", "payload"),
    ("LOAD_MMA_FREE", "LOAD_TMA_ISSUED", WORK["tma"], "prod: issue tma", "payload"),
]

MMA_PHASES = SETUP_PHASES + [
    ("MMA_TILE_BEGIN", "MMA_TMEM_FREE", WAIT["tmem"], "mma: wait tmem", "payload"),
    ("MMA_STEP_BEGIN", "MMA_INPUTS_READY", WAIT["tma"], "mma: wait tma", "payload"),
    ("MMA_INPUTS_READY", "MMA_ISSUED", WORK["mma"], "mma: issue mma", "payload"),
]

EPILOGUE_PHASES = SETUP_PHASES + [
    ("EPI_TILE_BEGIN", "EPI_MMA_DONE", WAIT["mainloop"], "epi: wait mainloop", "payload"),
    ("EPI_MMA_DONE", "EPI_TMEM_READ", WORK["tmem"], "epi: tmem->reg", "payload"),
    ("EPI_TMEM_READ", "EPI_SMEM_WRITTEN", WORK["smem"], "epi: reg->smem", "payload"),
    # store_async_wait(): the stores are in flight, but this warp is blocked.
    ("EPI_SMEM_WRITTEN", "EPI_STORE_DONE", WAIT["store"], "epi: drain tma store", "payload"),
    ("EPI_STORE_DONE", "EPI_SIGNALLED", WORK["signal"], "epi: signal comm", "payload"),
]

COMM_PHASES = SETUP_PHASES + [
    # The one to read first: dark bars here are the all-reduce sitting idle
    # waiting for the GEMM to publish a tile.
    ("AR_TILE_BEGIN", "AR_SIGNAL_SEEN", WAIT["signal"], "ar: wait gemm signal", "payload"),
    ("AR_SIGNAL_SEEN", "AR_TILE_DONE", WORK["reduce"], "ar: multimem reduce", "payload"),
]

PHASES_GEMM_AR = {
    "producer": PRODUCER_PHASES,
    "mma": MMA_PHASES,
    "epilogue": EPILOGUE_PHASES,
    "comm": COMM_PHASES,
}

# ag_gemm_kda_mla has no comm CTAs: every block computes. The all-gather is
# staged by the copy engine into A_local_buf, so both A sources are local HBM
# reads by the time the MMA sees them -- the split below says which *shard* a
# tile came from, not which device it was read from. The event names still say
# LOCAL/REMOTE because renaming them would stop older traces decoding; the
# labels are what carry the current meaning.
#
# The two bars being about equal is the copy engine keeping up. A peer-shard bar
# much longer than an own-shard one means the compute has outrun the staging.
# The copy engine cannot stamp itself, so this span is the producer's spin on the
# flag the copy stream writes: it starts when the kernel first needs a peer's
# shard and ends when that shard lands. A short bar means the copy was already
# hidden behind earlier work; a long one means the kernel outran it.
AG_COPY_PHASES = [
    ("ACOPY_WAIT_BEGIN", "ACOPY_READY", WAIT["store"], "copy: wait A shard", "payload"),
]

AG_PRODUCER_PHASES = SETUP_PHASES + AG_COPY_PHASES + [
    ("LOAD_STEP_BEGIN", "LOAD_MMA_FREE", WAIT["ring"], "prod: wait ring slot", "payload"),
    ("LOAD_MMA_FREE", "LOAD_TMA_ISSUED", WORK["tma"], "prod: issue tma", "payload"),
]

AG_MMA_PHASES = SETUP_PHASES + [
    ("MMA_TILE_BEGIN", "MMA_TMEM_FREE", WAIT["tmem"], "mma: wait tmem", "payload"),
    ("MMA_STEP_BEGIN", "MMA_INPUTS_LOCAL", WAIT["tma"], "mma: wait A (own shard)", "payload"),
    ("MMA_STEP_BEGIN", "MMA_INPUTS_REMOTE", WAIT["a_remote"], "mma: wait A (peer shard)", "payload"),
    # Same colour and label from either input event, so the two merge into one
    # legend entry and one draw call.
    ("MMA_INPUTS_LOCAL", "MMA_ISSUED", WORK["mma"], "mma: issue mma", "payload"),
    ("MMA_INPUTS_REMOTE", "MMA_ISSUED", WORK["mma"], "mma: issue mma", "payload"),
]

# No per-tile store drain here: this kernel drains its TMA stores once, after
# the whole tile loop, so there is no span to pair per tile.
AG_EPILOGUE_PHASES = SETUP_PHASES + [
    ("EPI_TILE_BEGIN", "EPI_MMA_DONE", WAIT["mainloop"], "epi: wait mainloop", "payload"),
    ("EPI_MMA_DONE", "EPI_TMEM_READ", WORK["tmem"], "epi: tmem->reg", "payload"),
    ("EPI_TMEM_READ", "EPI_SMEM_WRITTEN", WORK["smem"], "epi: reg->smem", "payload"),
]

PHASES_AG_GEMM = {
    "producer": AG_PRODUCER_PHASES,
    "mma": AG_MMA_PHASES,
    "epilogue": AG_EPILOGUE_PHASES,
}

# The .npz records which kernel it came from, so one renderer serves both and an
# old trace still picks the table its events were emitted against.
KERNEL_PHASES = {
    "gemm_ar_blackwell": PHASES_GEMM_AR,
    "ag_gemm_kda_mla": PHASES_AG_GEMM,
}
DEFAULT_KERNEL = "gemm_ar_blackwell"


def phases_for(kernel):
    if kernel not in KERNEL_PHASES:
        raise SystemExit(
            f"unknown kernel {kernel!r} in trace; known: {sorted(KERNEL_PHASES)}"
        )
    return KERNEL_PHASES[kernel]


# Backwards-compatible alias.
PHASES_BY_ROLE = PHASES_GEMM_AR

# label -> "wait" | "work" | "setup". Drives legend order, the summary's kind
# column, and --collapse. Derived from the tables above so a new phase cannot be
# added without landing on one side of the split.
PHASE_KIND = {}
for _tables in KERNEL_PHASES.values():
  for _table in _tables.values():
    for _s, _e, _c, _label, _m in _table:
        PHASE_KIND[_label] = (
            "setup" if _c == SETUP_COLOR else "wait" if _c in WAIT.values() else "work"
        )


def _luminance(hex_color):
    """Relative luminance, per WCAG. Only used by the self-check below."""

    def chan(v):
        v /= 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * chan(r) + 0.7152 * chan(g) + 0.0722 * chan(b)


def _warmth(hex_color):
    """Red minus blue, in [-1, 1]. Positive is warm (fire), negative is cool."""
    r, _, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return (r - b) / 255.0


def assert_palette(gap=0.15, warmth_gap=0.25):
    """Both channels must separate the two groups, on every colour.

    A hand-picked hex is easy to get wrong when adding a phase, and getting it
    wrong is invisible until someone squints at a 5 ms trace -- so check it
    rather than trusting the eye.
    """
    waits, works = list(WAIT.values()), list(WORK.values())
    lum_w, lum_k = [_luminance(c) for c in waits], [_luminance(c) for c in works]
    if max(lum_w) + gap >= min(lum_k):
        raise AssertionError(
            f"brightness overlap: darkest work {min(lum_k):.3f} is not {gap} "
            f"above the lightest wait {max(lum_w):.3f}"
        )
    warm_w, warm_k = [_warmth(c) for c in waits], [_warmth(c) for c in works]
    if max(warm_w) + warmth_gap >= min(warm_k):
        raise AssertionError(
            f"warmth overlap: coolest work {min(warm_k):+.3f} is not "
            f"{warmth_gap} above the warmest wait {max(warm_w):+.3f}"
        )
    return {
        "lightest wait": max(lum_w), "darkest work": min(lum_k),
        "warmest wait": max(warm_w), "coolest work": min(warm_k),
    }


# ---------------------------------------------------------------------------
# Dump
# ---------------------------------------------------------------------------
def unpack(raw_i64, num_blocks: int, events_per_block: int):
    """Ring buffer -> ((N, 4) int64 table, per-block head array).

    `raw_i64` is the int64 tensor/array the kernel wrote: two int64s per 16-byte
    record. Columns of the returned table are (block, timestamp_ns, event_id,
    payload).
    """
    raw = np.asarray(raw_i64).ravel().view(np.uint64)
    raw = raw.reshape(num_blocks, events_per_block, 2)

    ts = raw[..., 0]
    packed = raw[..., 1]
    event_ids = (packed >> np.uint64(32)).astype(np.uint32)
    payloads = (packed & np.uint64(0xFFFFFFFF)).astype(np.uint32)

    # The buffer is zeroed by the host and %globaltimer has been counting since
    # GPU boot, so a zero timestamp can only mean "never written". The first
    # such index in a block's slot is that block's event count -- no separate
    # head array has to come back from the device.
    valid = ts != 0
    heads = valid.argmin(axis=1)
    heads = np.where(valid.all(axis=1), events_per_block, heads)

    rows = []
    for b in range(num_blocks):
        n = int(heads[b])
        if n == 0:
            continue
        rows.append(
            np.stack(
                [
                    np.full(n, b, dtype=np.int64),
                    ts[b, :n].astype(np.int64),
                    event_ids[b, :n].astype(np.int64),
                    payloads[b, :n].astype(np.int64),
                ],
                axis=1,
            )
        )

    records = np.concatenate(rows, axis=0) if rows else np.zeros((0, 4), np.int64)
    return records, heads


def save(path, records, name_to_id, warp_layout=None, kernel=DEFAULT_KERNEL, **meta):
    """Write the flat table plus enough metadata to render offline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layout = dict(warp_layout or DEFAULT_WARP_LAYOUT)
    np.savez_compressed(
        path,
        records=records,
        kernel=np.array(kernel),
        event_ids=np.array(list(name_to_id.values()), dtype=np.int64),
        event_names=np.array(list(name_to_id.keys())),
        layout_keys=np.array(list(layout.keys())),
        layout_values=np.array(list(layout.values()), dtype=np.int64),
        **{k: np.asarray(v) for k, v in meta.items()},
    )
    return path


def load(path):
    """.npz -> (records, name_to_id, warp_layout, meta dict)."""
    z = np.load(path, allow_pickle=False)
    records = z["records"]
    name_to_id = {str(n): int(i) for n, i in zip(z["event_names"], z["event_ids"])}
    if "layout_keys" in z.files:
        layout = {str(k): int(v) for k, v in zip(z["layout_keys"], z["layout_values"])}
    else:
        layout = dict(DEFAULT_WARP_LAYOUT)
    skip = {"records", "event_ids", "event_names", "layout_keys", "layout_values"}
    meta = {k: z[k] for k in z.files if k not in skip}
    # Traces written before the field existed are all gemm_ar_blackwell.
    meta.setdefault("kernel", np.array(DEFAULT_KERNEL))
    return records, name_to_id, layout, meta


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------
def build_spans(records, name_to_id, phases, row_mask=None):
    """Flat table -> [(block, warp, start_ns, dur_ns, color, label)].

    Pairing uses (block << 32) | payload, which is already unique per
    (block, warp, seq) because the kernel packs the warp id into the payload's
    top bits. The warp is pulled back out so each role gets its own row.

    Vectorized per phase: no Python loop over records, which matters at the ~1e6
    events a large shape produces.
    """
    if records.shape[0] == 0:
        return []

    blk = records[:, 0].astype(np.int64)
    ts = records[:, 1].astype(np.int64)
    ev = records[:, 2].astype(np.int64)
    pld = records[:, 3].astype(np.int64) & 0xFFFFFFFF
    warp = pld >> WARP_ID_SHIFT

    if row_mask is not None:
        blk, ts, ev, pld, warp = (a[row_mask] for a in (blk, ts, ev, pld, warp))
        if blk.size == 0:
            return []

    key = (blk << 32) | pld

    spans = []
    for start_name, end_name, color, label, mode in phases:
        sid = name_to_id.get(start_name)
        eid = name_to_id.get(end_name)
        if sid is None or eid is None:
            continue
        s_mask = ev == sid
        e_mask = ev == eid
        if not s_mask.any() or not e_mask.any():
            continue

        if mode == "payload":
            s_key, e_key = key[s_mask], key[e_mask]
            s_ts, e_ts = ts[s_mask], ts[e_mask]
            # stable sort: insurance against two records ever sharing a key.
            order = np.argsort(e_key, kind="stable")
            e_key_s, e_ts_s = e_key[order], e_ts[order]
            idx = np.searchsorted(e_key_s, s_key)
            in_range = idx < e_key_s.size
            ok = np.zeros(s_key.size, dtype=bool)
            ok[in_range] = e_key_s[idx[in_range]] == s_key[in_range]
            t0 = s_ts[ok]
            durs = e_ts_s[idx[ok]] - t0
            b_out = blk[s_mask][ok]
            w_out = warp[s_mask][ok]

        elif mode == "preceding":
            # End carries no matching payload: pair each end with the most
            # recent begin on the same (block, warp) row.
            row = (blk << 8) | warp
            parts = []
            for r in np.unique(row):
                on_row = row == r
                sts = np.sort(ts[s_mask & on_row])
                ets = np.sort(ts[e_mask & on_row])
                if sts.size == 0 or ets.size == 0:
                    continue
                j = np.searchsorted(sts, ets, side="left") - 1
                valid = j >= 0
                t0v = sts[j[valid]]
                parts.append((int(r) >> 8, int(r) & 0xFF, t0v, ets[valid] - t0v))
            if not parts:
                continue
            b_out = np.concatenate([np.full(t.size, b, np.int64) for b, _, t, _ in parts])
            w_out = np.concatenate([np.full(t.size, w, np.int64) for _, w, t, _ in parts])
            t0 = np.concatenate([t for _, _, t, _ in parts])
            durs = np.concatenate([d for _, _, _, d in parts])
        else:
            raise ValueError(f"unknown pair_by mode {mode!r}")

        # Drop zero- and negative-duration spans: a begin and end landing in the
        # same globaltimer tick, or a stray end with no begin. Letting negatives
        # through draws inside-out bars instead of flagging the bug.
        keep = durs > 0
        n = int(keep.sum())
        if n == 0:
            continue
        spans += list(
            zip(
                b_out[keep].tolist(),
                w_out[keep].tolist(),
                t0[keep].tolist(),
                durs[keep].tolist(),
                [color] * n,
                [label] * n,
            )
        )
    return spans


def spans_for_all_roles(records, name_to_id, num_comp_sm, layout, phases_by_role=None):
    """Every role's phase table, applied to the rows that play that role."""
    phases_by_role = phases_by_role or PHASES_GEMM_AR
    if records.shape[0] == 0:
        return []
    blk = records[:, 0].astype(np.int64)
    warp = (records[:, 3].astype(np.int64) & 0xFFFFFFFF) >> WARP_ID_SHIFT

    # Vectorized role lookup: the (block-is-comp, warp) pairs are few, so
    # resolve each distinct warp id once rather than per record.
    is_comp = blk < num_comp_sm
    spans = []
    for role in ROLE_ORDER:
        mask = np.zeros(blk.size, dtype=bool)
        for w in np.unique(warp):
            w = int(w)
            if row_role(0, w, num_comp_sm, layout) == role:
                mask |= is_comp & (warp == w)
            if row_role(num_comp_sm, w, num_comp_sm, layout) == role:
                mask |= (~is_comp) & (warp == w)
        if not mask.any() or role not in phases_by_role:
            continue
        spans += build_spans(records, name_to_id, phases_by_role[role], row_mask=mask)
    return spans


def limit_rows_per_role(spans, num_comp_sm, layout, n):
    """Keep only the first n (block, warp) rows of each role band.

    148 CTAs is ~500 rows, which is unreadable however it is coloured -- and
    the rows within a band are near-identical, so a handful of them tells the
    same story. Rows are taken in block order rather than sampled: adjacent
    blocks are the ones that share a cluster, so the first few show the
    cta_rank 0/1 asymmetry (only rank 0 runs the MMA warps).

    Rows whose only span is the prologue are skipped. Every warp leader stamps
    SETUP, including the consumer warps of a cta_rank 1 CTA that then never call
    consume() -- so half the MMA rows exist but carry nothing, and picking
    blindly in block order would spend 4 of 8 rows on blank ones.

    Returns (kept spans, {role: (shown, available)}), where available counts
    only rows that have something to show.
    """
    if not n or n <= 0:
        return spans, {}

    active = {}
    for b, w, _st, _d, _c, label in spans:
        role = row_role(b, w, num_comp_sm, layout)
        if role is not None and label != "setup":
            active.setdefault(role, set()).add((b, w))

    keep, shown = set(), {}
    for role, rows in active.items():
        chosen = sorted(rows)[:n]
        keep.update(chosen)
        shown[role] = (len(chosen), len(rows))
    return [s for s in spans if (s[0], s[1]) in keep], shown


def summarize(spans):
    """Per-phase count and mean duration -- the first thing to check. A count
    that is not the expected tiles * steps is a guard mismatch or a missing warp
    discriminator, not a slow kernel.

    Grouped by kind, with a wait/work rollup: that ratio is the overlap number
    the Gantt is showing you qualitatively."""
    agg = {}
    for _, _, _, dur, _, label in spans:
        tot, n = agg.get(label, (0, 0))
        agg[label] = (tot + dur, n + 1)

    lines = [f"  {'phase':<24} {'kind':>6} {'count':>9} {'total us':>12} {'mean ns':>10}"]
    by_kind = {}
    for kind in ("wait", "work", "setup"):
        rows = [(k, v) for k, v in agg.items() if PHASE_KIND.get(k) == kind]
        if not rows:
            continue
        for label, (total_ns, n) in sorted(rows, key=lambda kv: -kv[1][0]):
            lines.append(
                f"  {label:<24} {kind:>6} {n:>9} "
                f"{total_ns / 1000.0:>12.1f} {total_ns / n:>10.1f}"
            )
        by_kind[kind] = sum(t for _, (t, _) in rows)

    span_total = sum(by_kind.values())
    if span_total:
        lines.append("")
        for kind, total_ns in by_kind.items():
            lines.append(
                f"  {kind:<24} {'':>6} {'':>9} {total_ns / 1000.0:>12.1f} "
                f"{100.0 * total_ns / span_total:>9.1f}%"
            )
    return "\n".join(lines)


def wall_span_us(records):
    """Wall time covered by the trace. Should agree with the benchmark's kernel
    time to within a few percent; a large gap is time in uninstrumented phases."""
    if records.shape[0] == 0:
        return 0.0
    return float(records[:, 1].max() - records[:, 1].min()) / 1000.0


def parse_blocks(spec: str):
    """'0-3,108,120-122' -> {0,1,2,3,108,120,121,122}. None for the empty spec."""
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def plot(records, out_path, name_to_id, *, num_comp_sm, layout, title="",
         max_height=60.0, collapse=False, rows_per_role=None, phases_by_role=None,
         marker_events=()):
    """One Y-row per active (block, warp), grouped by role, as PolyCollections.

    collapse=True throws hue away and draws exactly two colours -- dark navy for
    every wait, orange for every unit of work -- which is the shortest path to
    seeing how much of the kernel is stalled.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    if records.shape[0] == 0:
        return None

    records = records.copy()
    records[:, 1] -= records[:, 1].min()  # zero the time axis

    spans = spans_for_all_roles(records, name_to_id, num_comp_sm, layout, phases_by_role)
    spans, _shown = limit_rows_per_role(spans, num_comp_sm, layout, rows_per_role)
    if not spans:
        return None

    # Row assignment: role bands in ROLE_ORDER, blank rows between them.
    active = {(b, w) for b, w, *_ in spans}
    row_to_y, y = {}, 0
    role_rows = {}
    for role in ROLE_ORDER:
        members = sorted(k for k in active if row_role(k[0], k[1], num_comp_sm, layout) == role)
        if not members:
            continue
        role_rows[role] = (y, y + len(members) - 1)
        for k in members:
            row_to_y[k] = y
            y += 1
        y += 2  # gap between role bands

    # Give rows real height once there are few enough to see individually.
    row_h = 0.11 if y > 120 else 0.30
    height = min(max(4.0, y * row_h), max_height)
    fig, ax = plt.subplots(figsize=(18, height))

    # One PolyCollection per (color, label): batches every bar of a phase into a
    # single draw call, which is the difference between seconds and minutes at
    # 1e5+ bars.
    by_color = {}
    for b, w, start, dur, color, label in spans:
        if collapse:
            if label.startswith("copy:"):
                color, label = COLLAPSED_COPY, "copy wait"
            else:
                kind = PHASE_KIND.get(label, "work")
                color, label = COLLAPSED[kind], kind
        by_color.setdefault((color, label), []).append(
            (row_to_y[(b, w)], start / 1000.0, dur / 1000.0)
        )
    xmax_us = max((s[2] + s[3]) for s in spans) / 1000.0
    min_w = xmax_us * MIN_BAR_FRAC
    for (color, label), bars in by_color.items():
        arr = np.asarray(bars, dtype=np.float64)
        ys, xs, ws = arr[:, 0], arr[:, 1], arr[:, 2]
        ws = np.maximum(ws, min_w)   # drawing floor only; stats use true widths
        h = 0.85
        verts = np.stack(
            [
                np.stack([xs, ys - h / 2], axis=1),
                np.stack([xs + ws, ys - h / 2], axis=1),
                np.stack([xs + ws, ys + h / 2], axis=1),
                np.stack([xs, ys + h / 2], axis=1),
            ],
            axis=1,
        )
        # Rasterized: a vector PDF of this many bars is either 100 MB or
        # unscrollable. 130 dpi is plenty for reading the shape.
        ax.add_collection(
            PolyCollection(
                verts,
                facecolors=color,
                edgecolors="none",
                linewidths=0,
                rasterized=True,
                label=label,
            )
        )

    for role, (lo, hi) in role_rows.items():
        ax.text(
            -0.008,
            (lo + hi) / 2,
            ROLE_DISPLAY.get(role, role),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontweight="bold",
            fontsize=10,
        )
        ax.axhline(hi + 1, color="#d4d4d8", lw=0.6)

    # Full-height rules at the marker instants, earliest observation per payload.
    for ev_name, mcolor, fmt in marker_events:
        mid = name_to_id.get(ev_name)
        if mid is None:
            continue
        sel = records[:, 2] == mid
        if not sel.any():
            continue
        mts = records[sel, 1]
        mseq = records[sel, 3].astype(np.int64) & SEQ_MASK
        for sq in np.unique(mseq):
            t_us = mts[mseq == sq].min() / 1000.0
            ax.axvline(t_us, color=mcolor, lw=1.1, ls="--", alpha=0.9, zorder=5)
            ax.text(t_us, -0.5, " " + fmt.format(seq=int(sq)), color=mcolor,
                    fontsize=7, rotation=90, va="top", ha="left", zorder=6)

    counts, totals = {}, {}
    for _, _, _, dur, _, label in spans:
        key = ("copy wait" if label.startswith("copy:") else PHASE_KIND.get(label, "work")) \
            if collapse else label
        totals[key] = totals.get(key, 0) + dur / 1000.0
        counts[key] = counts.get(key, 0) + 1

    # Legend grouped waits-first, then work: the ordering is what teaches the
    # dark/light scheme without a caption.
    kind_rank = {"wait": 0, "work": 1, "setup": 2}
    ordered = sorted(
        by_color,
        key=lambda cl: (
            kind_rank.get("wait" if cl[1] == "copy wait"
                          else cl[1] if collapse else PHASE_KIND.get(cl[1], "work"), 3),
            -totals[cl[1]],
        ),
    )
    handles = [
        Patch(
            facecolor=c,
            edgecolor="#52525b" if _luminance(c) > 0.7 else "none",
            linewidth=0.4,
            label=f"{l} (n={counts[l]} avg={totals[l] / counts[l]:.2f}us)",
        )
        for (c, l) in ordered
    ]
    # Above the axes, not inside them: at this bar density an inset legend
    # always lands on top of data worth reading.
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0),
              fontsize=8, framealpha=0.95, borderaxespad=0.4,
              ncol=1 if collapse else 3,
              title="dark + cool = waiting      hot = working", title_fontsize=8)

    ax.set_xlim(0, xmax_us)
    ax.set_ylim(y, -1)  # inverted: block 0 on top
    ax.set_xlabel("time (us)")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_facecolor(CANVAS)
    ax.grid(axis="x", alpha=0.25, color="#52525b", lw=0.6)
    if title:
        # Left-aligned so it clears the legend now sitting above the axes.
        ax.set_title(title, fontsize=11, loc="left")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI: re-render a saved trace
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace", help="path to a .npz written by save()")
    ap.add_argument("--out", default="", help="output PDF (default: alongside the .npz)")
    ap.add_argument("--blocks", default="", help="block filter, e.g. '0-3,140-147'")
    ap.add_argument("--title", default="", help="override the plot title")
    ap.add_argument("--collapse", action="store_true",
                    help="draw exactly two colours: dark=any wait, hot=any work")
    ap.add_argument("--rows-per-role", type=int, default=8, metavar="N",
                    help="rows per band (TMA/MMA/EPILOGUE/AR); 0 for every CTA "
                         "(default: 8)")
    args = ap.parse_args(argv)

    records, name_to_id, layout, meta = load(args.trace)
    kernel = str(meta["kernel"])
    phases = phases_for(kernel)
    num_comp_sm = int(meta.get("num_comp_sm", 0))
    if num_comp_sm <= 0:
        print("trace has no num_comp_sm; cannot tell comp CTAs from comm CTAs", file=sys.stderr)
        return 1

    keep = parse_blocks(args.blocks)
    if keep is not None:
        records = records[np.isin(records[:, 0], sorted(keep))]
        if records.shape[0] == 0:
            print(f"no records for blocks {args.blocks}", file=sys.stderr)
            return 1

    all_spans = spans_for_all_roles(records, name_to_id, num_comp_sm, layout, phases)
    spans, shown = limit_rows_per_role(
        all_spans, num_comp_sm, layout, args.rows_per_role
    )
    print(f"{args.trace}: [{kernel}] {records.shape[0]} records, {len(all_spans)} spans")
    print(f"  wall span: {wall_span_us(records):.1f} us")
    if shown:
        detail = "  ".join(
            f"{ROLE_DISPLAY.get(r, r)} {n}/{avail}"
            for r, (n, avail) in sorted(
                shown.items(), key=lambda kv: ROLE_ORDER.index(kv[0])
            )
        )
        print(f"  rows shown: {detail}   ({len(spans)} of {len(all_spans)} spans)")
    print(summarize(spans))

    def scalar(key, default=None):
        v = meta.get(key)
        return default if v is None else v.item()

    num_blocks = int(meta["num_blocks"].item()) if "num_blocks" in meta else 0
    split = (
        f"  comp SMs 0..{num_comp_sm - 1}, comm SMs {num_comp_sm}..{num_blocks - 1}"
        if num_blocks and num_comp_sm < num_blocks
        else f"  {num_blocks} CTAs" if num_blocks else ""
    )
    title = args.title or (
        f"{kernel}  M={scalar('M', '?')} N={scalar('N', '?')} K={scalar('K', '?')}  "
        f"rank {scalar('rank', '?')}/{scalar('world_size', '?')}{split}"
    )

    out = Path(args.out) if args.out else Path(args.trace).with_suffix(".pdf")
    written = plot(
        records, out, name_to_id, num_comp_sm=num_comp_sm, layout=layout,
        title=title, collapse=args.collapse, rows_per_role=args.rows_per_role,
        phases_by_role=phases, marker_events=MARKER_EVENTS.get(kernel, ()),
    )
    print(f"wrote {written}" if written else "nothing to plot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
