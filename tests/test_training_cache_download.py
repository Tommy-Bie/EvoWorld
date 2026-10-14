from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from download_training_cache import (  # noqa: E402
    component_archives,
    extract_archive,
    marker_is_valid,
    marker_path,
    read_checksums,
    validate_extracted,
    validate_metadata,
    write_cache_env,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class TrainingCacheDownloadTests(unittest.TestCase):
    def test_checksum_inventory_and_component_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checksum = Path(temporary) / "SHA256SUMS"
            lines = []
            for index in range(11):
                lines.append(f"{'a' * 64}  core/core-{index:05d}-of-00011.tar\n")
            for index in range(99):
                lines.append(f"{'b' * 64}  conditions/conditions-{index:05d}-of-00099.tar\n")
            checksum.write_text("".join(lines), encoding="ascii")
            inventory = read_checksums(checksum)
            self.assertEqual(len(component_archives(inventory, "core")), 11)
            self.assertEqual(len(component_archives(inventory, "all")), 110)

    def test_safe_extract_and_resumable_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "core.tar"
            payload = b"latent"
            with tarfile.open(archive_path, "w") as archive:
                info = tarfile.TarInfo("preprocessed/latents/seerworld_000000.pt")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            destination = root / "extracted"
            marker = marker_path(root, "core/core.tar")
            count = extract_archive(archive_path, destination, marker, digest, "core")
            self.assertEqual(count, 1)
            self.assertTrue(marker_is_valid(marker, destination, digest))
            (destination / "preprocessed/latents/seerworld_000000.pt").unlink()
            self.assertFalse(marker_is_valid(marker, destination, digest))

    def test_rejects_unsafe_tar_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "unsafe.tar"
            with tarfile.open(archive_path, "w") as archive:
                info = tarfile.TarInfo("../escape.pt")
                info.size = 1
                archive.addfile(info, io.BytesIO(b"x"))
            with self.assertRaisesRegex(ValueError, "unsafe"):
                extract_archive(
                    archive_path,
                    root / "extracted",
                    root / "marker.json",
                    "0" * 64,
                    "core",
                )

    def test_metadata_extracted_counts_and_environment_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "training_cache/ltx-2.5"
            metadata = cache / "training_metadata"
            rows = [
                {"clip_id": "seerworld_000000", "split": "train"},
                {"clip_id": "seerworld_000001", "split": "additional"},
            ]
            write_jsonl(metadata / "captions.jsonl", [
                {"clip_id": "seerworld_000000"},
                {"clip_id": "seerworld_000001"},
            ])
            write_jsonl(metadata / "manifest.jsonl", rows)
            preprocessed = cache / "extracted/preprocessed"
            for folder, names in {
                "latents": ["a.pt", "b.pt"],
                "reference_latents": ["a.pt", "b.pt"],
                "conditions": ["a__seg0.pt", "b__seg0.pt", "b__seg1.pt"],
            }.items():
                directory = preprocessed / folder
                directory.mkdir(parents=True, exist_ok=True)
                for name in names:
                    (directory / name).write_bytes(b"x")
            with patch("download_training_cache.EXPECTED_SPLITS", {"train": 1, "additional": 1}), patch(
                "download_training_cache.EXPECTED_FILES",
                {"latents": 2, "reference_latents": 2, "conditions": 3},
            ):
                validate_metadata(cache)
                validate_extracted(cache, "all")
            env = root / ".env.cache"
            write_cache_env(env, cache, force=False)
            text = env.read_text()
            self.assertIn("CW_WORK=", text)
            self.assertIn("CW_CONDITIONS=", text)


if __name__ == "__main__":
    unittest.main()
