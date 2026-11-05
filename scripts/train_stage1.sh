#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${EW_PYTHON:-$ROOT/.venv/bin/python}"
ACCELERATE="${EW_ACCELERATE:-$ROOT/.venv/bin/accelerate}"
[[ -x "$PY" ]] || { echo "Missing Python: $PY (run uv sync or set EW_PYTHON)" >&2; exit 1; }
[[ -x "$ACCELERATE" ]] || { echo "Missing accelerate: $ACCELERATE (run uv sync or set EW_ACCELERATE)" >&2; exit 1; }
for name in EW_WORK EW_LATENTS EW_CONDITIONS EW_REFERENCE; do
  if [[ -z "${!name:-}" || ! -d "${!name}" ]]; then
    echo "Missing directory: $name (source your .env first)" >&2
    exit 1
  fi
done
export EW_USE_IAA=1 EW_IAA_BETA=3.0 EW_IAA_ENTITY_REFS=2
export EW_I2V_START=0 EW_I2V_OVERSAMPLE=1 EW_PREFIX_NOISE=0
export EW_REVISIT_OVERSAMPLE=3
"$PY" scripts/configure.py
"$PY" evoworld/verify_iaa_runtime.py
"$ACCELERATE" launch --num_processes "${EW_NPROC:-8}" --mixed_precision bf16 \
  evoworld/train_stage1.py configs/generated/stage1.yaml \
  --work "$EW_WORK" --latents "$EW_LATENTS" \
  --conditions "$EW_CONDITIONS" --reference "$EW_REFERENCE" "$@"
