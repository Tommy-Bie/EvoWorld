"""Resolve portable YAML paths without importing torch or changing source configs."""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def expand(value: object) -> object:
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if not os.environ.get(name):
                raise ValueError(f"Missing environment variable: {name}. See scripts/env.example.sh")
            return os.environ[name]
        return re.sub(r"\$\{(CW_[A-Z0-9_]+)\}", replace, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=ROOT / "configs/stage1.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "configs/generated/stage1.yaml")
    args = parser.parse_args()
    config = expand(yaml.safe_load(args.template.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(config, sort_keys=False))
    print(args.out)


if __name__ == "__main__":
    main()
