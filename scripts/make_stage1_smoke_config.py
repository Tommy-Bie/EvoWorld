"""Derive a short, isolated Stage-1 smoke config from the resolved recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def derive_smoke_config(config: dict, output_dir: Path, steps: int) -> dict:
    """Convert a resolved Stage-1 recipe into a short isolated smoke run."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if "_iaafix" not in output_dir.name:
        raise ValueError("smoke output directory name must contain _iaafix")

    config["optimization"]["steps"] = steps
    config["optimization"]["gradient_accumulation_steps"] = 1
    config["data"]["num_dataloader_workers"] = 0
    config["validation"]["interval"] = None
    config["validation"]["samples"] = []
    config["checkpoints"]["interval"] = None
    config["checkpoints"]["keep_last_n"] = 1
    config["checkpoints"]["save_training_state"] = "minimal"
    config["wandb"]["enabled"] = False
    config["hub"]["push_to_hub"] = False
    config["output_dir"] = str(output_dir)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    try:
        config = derive_smoke_config(yaml.safe_load(args.input.read_text()), args.output_dir, args.steps)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    print(args.output)


if __name__ == "__main__":
    main()
