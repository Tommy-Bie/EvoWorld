"""Dependency-free structural/syntax checks; does not load weights or run CUDA."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    errors: list[str] = []
    files = list(ROOT.rglob("*"))
    python_files = [path for path in files if path.suffix == ".py"]
    for path in python_files:
        try:
            ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
        except (SyntaxError, UnicodeError) as error:
            errors.append(str(error))
    for path in files:
        if not path.is_file():
            continue
        if path.suffix in {".pt", ".safetensors", ".mp4"} or path.name == ".env":
            errors.append(f"Private/runtime artifact bundled: {path.relative_to(ROOT)}")
        if path.suffix in {".py", ".sh", ".yaml", ".toml"}:
            content = path.read_text()
            # Construct strings so the checker does not flag itself.
            for marker in ("/" + "efs/", "/" + "Users/", "/" + "root/"):
                if marker in content:
                    errors.append(f"Machine-specific path: {path.relative_to(ROOT)}")
    for relative in ("seerworld/train_stage1.py", "seerworld/iaa_utils.py",
                     "seerworld/stage2_ar/train_stage2_online_ar.py",
                     "scripts/download_training_cache.py",
                     "licenses/LTX-LICENSE.md", "LICENSE", "NOTICE"):
        if not (ROOT / relative).is_file():
            errors.append(f"Missing: {relative}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"PASS: {len(python_files)} Python files parse; structural checks passed.")
    print("GPU execution, installation, licenses, and comprehensive secret scan not verified.")


if __name__ == "__main__":
    main()
