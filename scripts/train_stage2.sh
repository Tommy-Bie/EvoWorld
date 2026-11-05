#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${EW_PYTHON:-$ROOT/.venv/bin/python}"
ACCELERATE="${EW_ACCELERATE:-$ROOT/.venv/bin/accelerate}"
[[ -x "$PY" ]] || { echo "Missing Python: $PY (run uv sync or set EW_PYTHON)" >&2; exit 1; }
[[ -x "$ACCELERATE" ]] || { echo "Missing accelerate: $ACCELERATE (run uv sync or set EW_ACCELERATE)" >&2; exit 1; }
source "$ROOT/scripts/setup_cuda_libraries.sh"
setup_cuda_libraries "$PY"
: "${EW_STAGE1_LORA:?Set EW_STAGE1_LORA to your Stage-1 checkpoint}"
: "${EW_DISTILLED_LORA:?Set EW_DISTILLED_LORA to the official rollout adapter}"
: "${EW_OUTPUT_ROOT:?Source your .env first}"
for name in EW_WORK EW_LATENTS EW_CONDITIONS EW_REFERENCE; do
  if [[ -z "${!name:-}" || ! -d "${!name}" ]]; then
    echo "Missing directory: $name" >&2
    exit 1
  fi
done
export EW_USE_IAA=1 EW_IAA_BETA=3.0 EW_IAA_ENTITY_REFS=2
export EW_I2V_START=0 EW_I2V_OVERSAMPLE=1 EW_PREFIX_NOISE=0
export EW_REVISIT_OVERSAMPLE=1
"$PY" scripts/configure.py
"$PY" evoworld/stage2_ar/make_stage2_online_config.py \
  --stage1-config configs/generated/stage1.yaml --stage1-lora "$EW_STAGE1_LORA" \
  --out configs/generated/stage2.yaml \
  --output-dir "$EW_OUTPUT_ROOT/evoworld_ar2online_iaafix" \
  --steps 400 --lr 2e-5 --save-every 50 --gt-mix-ratio 0.25 --iaa-beta 3.0
args=(configs/generated/stage2.yaml --stage1-lora "$EW_STAGE1_LORA"
  --distilled-lora "$EW_DISTILLED_LORA" --work "$EW_WORK"
  --latents "$EW_LATENTS" --conditions "$EW_CONDITIONS" --reference "$EW_REFERENCE"
  --max-samples 2048 --gt-mix-ratio 0.25)
"$PY" evoworld/verify_iaa_runtime.py
"$PY" evoworld/stage2_ar/train_stage2_online_ar.py "${args[@]}" --preflight-only
"$ACCELERATE" launch --num_processes "${EW_NPROC:-8}" --mixed_precision bf16 \
  evoworld/stage2_ar/train_stage2_online_ar.py "${args[@]}" "$@"
