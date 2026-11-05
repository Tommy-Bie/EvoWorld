"""Pure-metadata selection for Stage-2 one-hop event-to-revisit pairs."""
from __future__ import annotations

import random

from cw_dataset import TARGET_LAT


def select_pairs(index, max_samples: int, seed: int):
    by_key = {(s["clip_id"], int(s["win"])): s for s in index.samples}
    candidates = []
    for sample in index.samples:
        predecessor = by_key.get((sample["clip_id"], int(sample["win"]) - TARGET_LAT))
        if not sample.get("is_revisit") or predecessor is None or predecessor.get("is_revisit"):
            continue
        if int(predecessor["seg"]) != int(sample["seg"]) - 1:
            continue
        candidates.append((sample, predecessor))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected, used_clips = [], set()
    for pair in candidates:
        clip_id = pair[0]["clip_id"]
        if clip_id in used_clips:
            continue
        used_clips.add(clip_id)
        selected.append(pair)
        if max_samples > 0 and len(selected) >= max_samples:
            break
    return selected

