#!/usr/bin/env python3
"""Validate EvoWorld's frozen-prompt examples and prepare an image manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def index_unique(rows: list[dict], key: str, source: Path) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        value = str(row.get(key) or "").strip()
        if not value:
            raise ValueError(f"{source}: missing {key}")
        if value in indexed:
            raise ValueError(f"{source}: duplicate {key}={value!r}")
        indexed[value] = row
    return indexed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--case", dest="case_filter")
    args = parser.parse_args()

    cases_path = args.cases.resolve()
    prompts_path = args.prompts.resolve()
    image_root = args.image_root.resolve()
    work_dir = args.work_dir.resolve()
    cases = read_jsonl(cases_path)
    prompts = read_jsonl(prompts_path)
    case_index = index_unique(cases, "clip_id", cases_path)
    prompt_index = index_unique(prompts, "clip_id", prompts_path)
    if set(case_index) != set(prompt_index):
        missing = sorted(set(case_index) - set(prompt_index))
        extra = sorted(set(prompt_index) - set(case_index))
        raise ValueError(f"case/prompt mismatch: missing={missing}, extra={extra}")
    if args.case_filter and args.case_filter not in case_index:
        raise ValueError(f"example case not found: {args.case_filter}")

    image_rows = []
    for clip_id, row in case_index.items():
        if args.case_filter and clip_id != args.case_filter:
            continue
        for field in ("initial_prompt", "event_prompt", "first_frame"):
            if not str(row.get(field) or "").strip():
                raise ValueError(f"{clip_id}: empty {field}")
        prompt = str(prompt_index[clip_id].get("reasoning_prompt") or "").strip()
        if len(prompt) < 40:
            raise ValueError(f"{clip_id}: reasoning prompt is too short")
        relative_image = Path(row["first_frame"])
        if relative_image.is_absolute() or ".." in relative_image.parts:
            raise ValueError(f"{clip_id}: unsafe first_frame path {relative_image}")
        source_image = image_root / relative_image.name
        if not source_image.is_file():
            raise FileNotFoundError(f"{clip_id}: missing source image {source_image}")
        if relative_image.with_suffix("").as_posix() != clip_id:
            raise ValueError(f"{clip_id}: first_frame path does not match clip_id")
        image_rows.append({"image": relative_image.as_posix()})

    work_dir.mkdir(parents=True, exist_ok=True)
    media_link = work_dir / "media"
    if media_link.is_symlink():
        if media_link.resolve() != image_root:
            media_link.unlink()
            media_link.symlink_to(image_root, target_is_directory=True)
    elif media_link.exists():
        raise ValueError(f"{media_link} exists but is not a symlink")
    else:
        media_link.symlink_to(image_root, target_is_directory=True)
    manifest = work_dir / "images.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in image_rows),
        encoding="utf-8",
    )
    print(f"PASS: {len(image_rows)} EvoWorld inference example(s)")
    print(f"image manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
