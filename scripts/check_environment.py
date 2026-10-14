"""Check the core versions used by the original SeerWorld training run."""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version

import torch


EXPECTED = {
    "torch": {"2.13.0+cu132"},
    # The original environment's metadata reported 0.28.0, while a fresh
    # installation from the cu132 index reports the equivalent local version
    # 0.28.0+cu132.
    "torchvision": {"0.28.0", "0.28.0+cu132"},
    "torchaudio": {"2.11.0+cu132"},
    "accelerate": {"1.14.0"},
    "transformers": {"5.14.1"},
    "peft": {"0.20.0"},
    "natten": {"0.21.7+torch2130cu132"},
    "safetensors": {"0.8.0"},
}


def main() -> None:
    failures: list[str] = []
    print(f"Python: {platform.python_version()}")
    for package, accepted in EXPECTED.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "not installed"
        expected = " or ".join(sorted(accepted))
        print(f"{package}: {installed} (expected {expected})")
        if installed not in accepted:
            failures.append(f"{package}: expected {expected}, found {installed}")

    print(f"CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.version.cuda != "13.2":
        failures.append(f"CUDA build: expected 13.2, found {torch.version.cuda}")
    if not torch.cuda.is_available():
        failures.append("CUDA is not available")

    if failures:
        raise SystemExit("Environment mismatch:\n- " + "\n- ".join(failures))
    print("PASS: core training environment matches the recorded versions.")


if __name__ == "__main__":
    main()
