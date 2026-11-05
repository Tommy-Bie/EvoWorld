#!/usr/bin/env python
"""Derive an isolated online-AR Stage-2 config from Stage 1."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-config", required=True)
    ap.add_argument("--stage1-lora", required=True)
    ap.add_argument("--resume-from", default="")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--validate-every", type=int, default=0)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--gt-mix-ratio", type=float, default=0.25)
    ap.add_argument("--iaa-beta", type=float, default=3.0)
    ap.add_argument("--save-training-state", choices=("full", "minimal", "off"), default="full")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.stage1_config).read_text())
    cfg["model"]["load_checkpoint"] = str(Path(args.resume_from or args.stage1_lora).resolve())
    cfg["output_dir"] = str(Path(args.output_dir))
    cfg["optimization"]["steps"] = args.steps
    cfg["optimization"]["learning_rate"] = args.lr
    cfg["optimization"]["gradient_accumulation_steps"] = args.grad_accum
    cfg["checkpoints"]["interval"] = args.save_every
    cfg["checkpoints"]["no_resume"] = not bool(args.resume_from)
    cfg["checkpoints"]["save_training_state"] = args.save_training_state
    cfg["validation"]["interval"] = args.validate_every if args.validate_every > 0 else None
    cfg["validation"]["skip_initial_validation"] = True
    if args.validate_every <= 0:
        # Disable validation assets when validation is not scheduled.
        cfg["validation"]["samples"] = []
        cfg["validation"]["prompts"] = []
        cfg["validation"]["images"] = None
        cfg["validation"]["reference_videos"] = None
        cfg["validation"]["generate_video"] = False
        cfg["validation"]["generate_audio"] = False
    tags = list(cfg.get("wandb", {}).get("tags") or [])
    cfg["wandb"]["tags"] = tags + [
        "stage2",
        "online-ar-prefix-sft",
        "on-policy-conditioning",
        "one-hop",
        "event-to-revisit",
        "distilled8-rollout-only",
        f"gtmix{args.gt_mix_ratio:g}",
        f"iaa-beta{args.iaa_beta:g}",
        f"lr{args.lr:g}",
        f"s{args.steps}",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(out)


if __name__ == "__main__":
    main()
