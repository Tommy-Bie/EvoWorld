#!/usr/bin/env python3
"""Download, verify, and extract the EvoWorld training cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "Tommy-Bie/EvoWorld-Data"
DEFAULT_DATASET_DIR = ROOT / "data" / "EvoWorld-Data"
EXPECTED_SPLITS = {"train": 11817, "additional": 301}
EXPECTED_FILES = {
    "latents": 12118,
    "reference_latents": 12118,
    "conditions": 42059,
}
METADATA_PATTERNS = [
    "training_cache/ltx-2.5/README.md",
    "training_cache/ltx-2.5/SHA256SUMS",
    "training_cache/ltx-2.5/core.json",
    "training_cache/ltx-2.5/conditions.json",
    "training_cache/ltx-2.5/training_metadata/*.jsonl",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def read_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: invalid checksum entry")
        digest, relative = fields[0].lower(), fields[1].lstrip("*")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{path}:{line_number}: invalid SHA-256 digest")
        if relative in checksums:
            raise ValueError(f"{path}:{line_number}: duplicate path {relative!r}")
        checksums[relative] = digest
    return checksums


def component_archives(checksums: dict[str, str], components: str) -> list[tuple[str, str]]:
    prefixes = ["core/"] if components == "core" else ["core/", "conditions/"]
    selected = [
        (relative, digest)
        for relative, digest in checksums.items()
        if any(relative.startswith(prefix) for prefix in prefixes) and relative.endswith(".tar")
    ]
    selected.sort()
    expected = 11 if components == "core" else 110
    if len(selected) != expected:
        raise ValueError(
            f"expected {expected} {components} archives, found {len(selected)} in SHA256SUMS"
        )
    return selected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def checked_members(archive: tarfile.TarFile, components: str) -> list[tarfile.TarInfo]:
    allowed = {"preprocessed/latents", "preprocessed/reference_latents"}
    if components == "all":
        allowed.add("preprocessed/conditions")
    members = archive.getmembers()
    names: set[str] = set()
    for member in members:
        pure = PurePosixPath(member.name)
        parent = pure.parent.as_posix()
        if (
            not member.isfile()
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.suffix != ".pt"
            or parent not in allowed
        ):
            raise ValueError(f"unsafe or unexpected TAR member: {member.name!r}")
        if member.name in names:
            raise ValueError(f"duplicate TAR member: {member.name}")
        names.add(member.name)
    return members


def marker_path(cache_root: Path, relative_archive: str) -> Path:
    safe_name = relative_archive.replace("/", "__") + ".json"
    return cache_root / ".extract_state" / safe_name


def marker_is_valid(marker: Path, destination: Path, digest: str) -> bool:
    if not marker.is_file():
        return False
    try:
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state.get("sha256") != digest:
            return False
        members = state.get("members")
        if not isinstance(members, list) or not members:
            return False
        for item in members:
            path = destination / item["path"]
            if not path.is_file() or path.stat().st_size != item["size"]:
                return False
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def extract_archive(
    archive_path: Path,
    destination: Path,
    marker: Path,
    digest: str,
    components: str,
) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:") as archive:
        members = checked_members(archive, components)
        archive.extractall(destination, members=members)
    state = {
        "sha256": digest,
        "members": [{"path": member.name, "size": member.size} for member in members],
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(json.dumps(state, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(marker)
    return len(members)


def validate_metadata(cache_root: Path) -> None:
    metadata = cache_root / "training_metadata"
    captions = read_jsonl(metadata / "captions.jsonl")
    manifests = read_jsonl(metadata / "manifest.jsonl")
    expected_total = sum(EXPECTED_SPLITS.values())
    if len(captions) != expected_total or len(manifests) != expected_total:
        raise ValueError(
            f"training metadata must contain {expected_total:,} rows; "
            f"captions={len(captions):,}, manifest={len(manifests):,}"
        )
    caption_ids = {row.get("clip_id") for row in captions}
    manifest_ids = {row.get("clip_id") for row in manifests}
    if len(caption_ids) != expected_total or caption_ids != manifest_ids:
        raise ValueError("caption and manifest public IDs do not match exactly")
    splits = Counter(row.get("split") for row in manifests)
    if splits != Counter(EXPECTED_SPLITS):
        raise ValueError(f"unexpected training splits: {dict(splits)}")


def validate_extracted(cache_root: Path, components: str) -> None:
    preprocessed = cache_root / "extracted" / "preprocessed"
    expected = {
        "latents": EXPECTED_FILES["latents"],
        "reference_latents": EXPECTED_FILES["reference_latents"],
    }
    if components == "all":
        expected["conditions"] = EXPECTED_FILES["conditions"]
    for name, count in expected.items():
        files = list((preprocessed / name).glob("*.pt"))
        if len(files) != count:
            raise ValueError(f"{name}: expected {count:,} files, found {len(files):,}")
    conditions = preprocessed / "conditions"
    if conditions.is_dir() and any("__global" in path.name for path in conditions.glob("*.pt")):
        raise ValueError("global-prompt conditions must not be present")


def write_cache_env(path: Path, cache_root: Path, *, force: bool) -> None:
    preprocessed = cache_root / "extracted" / "preprocessed"
    values = {
        "EW_WORK": cache_root / "training_metadata",
        "EW_DATA": preprocessed,
        "EW_LATENTS": preprocessed / "latents",
        "EW_CONDITIONS": preprocessed / "conditions",
        "EW_REFERENCE": preprocessed / "reference_latents",
    }
    content = "# Generated by scripts/download_training_cache.py\n" + "".join(
        f"export {name}={shlex.quote(str(value.resolve()))}\n"
        for name, value in values.items()
    )
    if path.exists() and path.read_text(encoding="utf-8") != content and not force:
        raise FileExistsError(f"refusing to replace existing environment file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def download_metadata(repo_id: str, dataset_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is missing; run `uv sync --locked --extra natten`") from exc
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=dataset_dir,
        allow_patterns=METADATA_PATTERNS,
    )


def download_archive(repo_id: str, dataset_dir: Path, relative: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is missing; run `uv sync --locked --extra natten`") from exc
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"training_cache/ltx-2.5/{relative}",
            local_dir=dataset_dir,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--components",
        choices=("core", "all"),
        default="all",
        help="all is the no-preprocessing training path; core omits text conditions",
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.cache")
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--force-env", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="download, verify, and extract only the first shard of each selected component",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="download and validate manifests without downloading TAR shards",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    cache_root = dataset_dir / "training_cache" / "ltx-2.5"
    print(f"Dataset: {args.repo_id}")
    print(f"Local directory: {dataset_dir}")
    download_metadata(args.repo_id, dataset_dir)
    validate_metadata(cache_root)
    checksums = read_checksums(cache_root / "SHA256SUMS")
    archives = component_archives(checksums, args.components)
    print(f"PASS metadata; selected {len(archives)} archive shards")
    if args.metadata_only:
        return 0
    if args.smoke:
        core = next(item for item in archives if item[0].startswith("core/"))
        smoke_archives = [core]
        if args.components == "all":
            smoke_archives.append(
                next(item for item in archives if item[0].startswith("conditions/"))
            )
        archives = smoke_archives
        print(f"Smoke mode: processing {len(archives)} representative shards")

    destination = cache_root / "extracted"
    for index, (relative, digest) in enumerate(archives, 1):
        marker = marker_path(cache_root, relative)
        if marker_is_valid(marker, destination, digest):
            print(f"[{index}/{len(archives)}] already extracted: {relative}")
            continue
        print(f"[{index}/{len(archives)}] download: {relative}", flush=True)
        archive_path = download_archive(args.repo_id, dataset_dir, relative)
        actual = sha256_file(archive_path)
        if actual != digest:
            archive_path.unlink()
            raise ValueError(
                f"checksum mismatch for {relative}: expected {digest}, found {actual}"
            )
        count = extract_archive(
            archive_path, destination, marker, digest, args.components
        )
        print(f"[{index}/{len(archives)}] verified and extracted {count:,} files")
        if not args.keep_archives:
            archive_path.unlink()

    if args.smoke:
        print("PASS: remote cache smoke download, checksum, and extraction")
        print("Rerun without --smoke to complete the cache; finished shards will be reused")
        return 0

    validate_extracted(cache_root, args.components)
    if args.components == "all":
        write_cache_env(args.env_file, cache_root, force=args.force_env)
        print("PASS: complete cache is ready for preprocessing-free training")
        print(f"Environment file: {args.env_file}")
        print(f"Next: source {args.env_file}")
        print("Then: bash scripts/smoke_stage1.sh")
    else:
        print("PASS: core cache is ready; text conditions must still be computed or downloaded")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
