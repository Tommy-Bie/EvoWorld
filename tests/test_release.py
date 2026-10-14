"""Release wiring tests with synthetic metadata; no torch, private data or CUDA."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "seerworld"),
               str(ROOT / "seerworld/stage2_ar")]

import yaml
from configure import expand
from cw_dataset import OoSWindowIndex, PREFIX_LAT, TARGET_LAT
from make_stage1_smoke_config import derive_smoke_config
from selection import select_pairs


class ReleaseTest(unittest.TestCase):
    def test_portable_config_resolution(self) -> None:
        names = ("CW_CKPT", "CW_GEMMA", "CW_VIDEO_VAE", "CW_AUDIO_VAE",
                 "CW_DATA", "CW_OUTPUT_ROOT")
        with patch.dict(os.environ, {name: "/example/" + name.lower() for name in names}, clear=True):
            cfg = expand(yaml.safe_load((ROOT / "configs/stage1.yaml").read_text()))
        self.assertNotIn("${", yaml.safe_dump(cfg))
        self.assertEqual(cfg["optimization"]["steps"], 8000)
        self.assertEqual(cfg["lora"]["rank"], cfg["lora"]["alpha"])
        self.assertEqual(cfg["training_strategy"]["video"]["conditions"][1]["temporal_boundary"], PREFIX_LAT)
        self.assertIn("_iaafix", cfg["output_dir"])
        self.assertIsNone(cfg["validation"]["interval"])

    def test_missing_environment_is_rejected(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "CW_CKPT"):
                expand("${CW_CKPT}")

    def test_online_pairs_from_synthetic_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            caps, manifests = [], []
            for number in range(3):
                clip = f"synthetic_{number}"
                caps.append({"clip_id": clip, "segment_prompts": ["initial", "event", "revisit", "hold"],
                             "nseg": 4, "seg_s": 3.75, "subject_on_screen": [True] * 4})
                manifests.append({"clip_id": clip, "split": "train", "n_frames": 361,
                                  "duration_s": 15, "phase": {"arrival_s": 8.0}})
            for name, rows in (("captions.jsonl", caps), ("manifest.jsonl", manifests)):
                (work / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
            index = OoSWindowIndex(work, split="train")
            pairs = select_pairs(index, max_samples=2, seed=42)
            self.assertEqual(len(pairs), 2)
            self.assertEqual(len({target["clip_id"] for target, _ in pairs}), 2)
            for target, predecessor in pairs:
                self.assertTrue(target["is_revisit"])
                self.assertFalse(predecessor["is_revisit"])
                self.assertEqual(predecessor["seg"], target["seg"] - 1)
                self.assertEqual(predecessor["win"] + TARGET_LAT, target["win"])

    def test_stage1_smoke_config_is_short_and_isolated(self) -> None:
        names = ("CW_CKPT", "CW_GEMMA", "CW_VIDEO_VAE", "CW_AUDIO_VAE",
                 "CW_DATA", "CW_OUTPUT_ROOT")
        with patch.dict(os.environ, {name: "/example/" + name.lower() for name in names}, clear=True):
            config = expand(yaml.safe_load((ROOT / "configs/stage1.yaml").read_text()))
        output_dir = Path("/tmp/seerworld_stage1_smoke_test_iaafix")
        smoke = derive_smoke_config(config, output_dir, steps=2)
        self.assertEqual(smoke["optimization"]["steps"], 2)
        self.assertEqual(smoke["optimization"]["gradient_accumulation_steps"], 1)
        self.assertEqual(smoke["data"]["num_dataloader_workers"], 0)
        self.assertFalse(smoke["wandb"]["enabled"])
        self.assertEqual(smoke["output_dir"], str(output_dir))


if __name__ == "__main__":
    unittest.main()
