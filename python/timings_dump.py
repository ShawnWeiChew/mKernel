"""Host side of the in-kernel timing ring (see include/common/timings.cuh).

Allocates the device ring, materializes it into a flat
(block, timestamp, event_id, payload) table, and writes a self-describing
.npz that plots/render_timings.py can render without the compiled .so.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


def allocate_ring(num_blocks: int, events_per_block: int, device) -> torch.Tensor:
    """Zero-filled ring for one profile run.

    Zero init is load-bearing: timestamp == 0 is the "unwritten" sentinel used
    to recover each CTA's head, so no separate head array is needed. %globaltimer
    has been running since GPU boot, so no CTA ever records a legitimate 0.
    """
    return torch.zeros(
        num_blocks * events_per_block * 2, dtype=torch.int64, device=device
    )


def unpack_ring(
    timings: torch.Tensor, num_blocks: int, events_per_block: int
) -> np.ndarray:
    """Ring tensor -> (n, 4) int64 table of (block, ts_ns, event_id, payload)."""
    raw = (
        timings.cpu()
        .numpy()
        .view(np.uint64)
        .reshape(num_blocks, events_per_block, 2)
    )
    ts = raw[..., 0]
    packed = raw[..., 1]
    event_ids = (packed >> np.uint64(32)).astype(np.uint32)
    payloads = (packed & np.uint64(0xFFFFFFFF)).astype(np.uint32)

    # Per-block head = first zero-timestamp index. argmin on the boolean mask
    # finds it; a slot with no zero was filled to the cap (or overflowed past
    # it, in which case the tail was dropped by the bounds check in EMIT).
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
    records = (
        np.concatenate(rows, axis=0) if rows else np.zeros((0, 4), np.int64)
    )
    return records, heads.astype(np.int64)


def save_trace(
    out_path,
    records: np.ndarray,
    heads: np.ndarray,
    events: Mapping[str, int],
    roles: Mapping[str, int],
    events_per_block: int,
    **metadata: Any,
) -> None:
    """Persist the table plus enough metadata to render offline.

    The event and role name<->id maps travel inside the file, so a trace stays
    renderable after the kernel's enum has grown.
    """
    np.savez_compressed(
        out_path,
        records=records,
        block_heads=heads,
        event_ids=np.array(list(events.values()), dtype=np.int64),
        event_names=np.array(list(events.keys())),
        role_ids=np.array(list(roles.values()), dtype=np.int64),
        role_names=np.array(list(roles.keys())),
        events_per_block=np.int64(events_per_block),
        **{k: np.asarray(v) for k, v in metadata.items()},
    )
