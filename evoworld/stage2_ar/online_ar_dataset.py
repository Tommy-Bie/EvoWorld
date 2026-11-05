#!/usr/bin/env python
"""Dataset inputs for Stage-2 online autoregressive training."""
from __future__ import annotations

import random

from torch.utils.data import Dataset

from cw_dataset import OoSWindowDataset, PREFIX_LAT, TARGET_LAT
from selection import select_pairs


class OnlineARPrefixDataset(Dataset):
    """Combine Stage-1 targets with deterministic online and GT entries."""

    def __init__(
        self,
        *,
        max_samples: int = 2048,
        gt_mix_ratio: float = 0.25,
        selection_seed: int = 42,
        **base_kwargs,
    ) -> None:
        if not 0.0 <= gt_mix_ratio < 1.0:
            raise ValueError(f"gt_mix_ratio must be in [0, 1), got {gt_mix_ratio}")

        base_kwargs["oversample_revisit"] = 1
        base_kwargs["prefix_noise"] = 0.0
        base_kwargs["oversample_i2v_start"] = 1
        base_kwargs["i2v_start"] = False
        self.base = OoSWindowDataset(**base_kwargs)

        key_to_index = {
            (sample["clip_id"], int(sample["win"])): i
            for i, sample in enumerate(self.base.index.samples)
        }
        pairs = select_pairs(self.base.index, max_samples, selection_seed)
        if not pairs:
            raise ValueError("online Stage-2 selection contains no event-to-revisit pairs")

        pair_indices: list[tuple[int, int]] = []
        for target, predecessor in pairs:
            target_key = (target["clip_id"], int(target["win"]))
            predecessor_key = (predecessor["clip_id"], int(predecessor["win"]))
            pair_indices.append((key_to_index[target_key], key_to_index[predecessor_key]))

        self.entries: list[tuple[int, int, int, bool]] = [
            (target_i, target_i, predecessor_i, True)
            for target_i, predecessor_i in pair_indices
        ]

        n_gt = round(len(pairs) * gt_mix_ratio / max(1e-12, 1.0 - gt_mix_ratio))
        rng = random.Random(selection_seed)
        gt_indices = list(range(len(self.base)))
        rng.shuffle(gt_indices)
        for j in range(n_gt):
            rollout_target_i, rollout_predecessor_i = pair_indices[j % len(pair_indices)]
            self.entries.append(
                (gt_indices[j % len(gt_indices)], rollout_target_i, rollout_predecessor_i, False)
            )

        self.num_online = len(pairs)
        self.num_gt = n_gt

    @property
    def index(self):
        return self.base.index

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        target_i, rollout_target_i, predecessor_i, use_online = self.entries[i]
        item = self.base[target_i]
        target_meta = item["meta"]
        item["online_ar_generated"] = int(use_online)
        item["meta"] = dict(target_meta, online_ar_generated=bool(use_online))
        rollout_target = self.base[rollout_target_i]
        predecessor = self.base[predecessor_i]
        rollout_target_meta = rollout_target["meta"]
        predecessor_meta = predecessor["meta"]
        expected_prev_win = int(rollout_target_meta["win"]) - TARGET_LAT
        if predecessor_meta["clip_id"] != rollout_target_meta["clip_id"]:
            raise RuntimeError("online predecessor belongs to a different clip")
        if int(predecessor_meta["win"]) != expected_prev_win:
            raise RuntimeError(
                f"online predecessor window mismatch: {predecessor_meta['win']} != {expected_prev_win}"
            )
        if int(predecessor_meta["seg"]) != int(rollout_target_meta["seg"]) - 1:
            raise RuntimeError("online predecessor is not the immediately preceding segment")
        if bool(predecessor_meta["is_revisit"]) or not bool(rollout_target_meta["is_revisit"]):
            raise RuntimeError("online pair must be non-revisit predecessor -> revisit target")
        if use_online and target_i != rollout_target_i:
            raise RuntimeError("generated-prefix entry must roll out its own predecessor")

        predecessor_window = predecessor["video_latents"]["latents"]
        item["online_ar_seed"] = {
            "latents": predecessor_window[:, :PREFIX_LAT].clone(),
            "num_frames": PREFIX_LAT,
            "height": predecessor["video_latents"]["height"],
            "width": predecessor["video_latents"]["width"],
            "fps": predecessor["video_latents"]["fps"],
        }
        item["online_ar_conditions"] = predecessor["conditions"]
        item["online_ar_reference"] = rollout_target[self.base.ref_key]
        item["online_ar_meta"] = {
            "clip_id": predecessor_meta["clip_id"],
            "win": int(predecessor_meta["win"]),
            "seg": int(predecessor_meta["seg"]),
            "target_win": int(rollout_target_meta["win"]),
            "target_seg": int(rollout_target_meta["seg"]),
        }
        return item
