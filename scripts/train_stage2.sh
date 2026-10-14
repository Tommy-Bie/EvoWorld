#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CW_STAGE1_LORA:?Set CW_STAGE1_LORA to your Stage-1 checkpoint}"
: "${CW_DISTILLED_LORA:?Set CW_DISTILLED_LORA to the official rollout adapter}"
: "${CW_OUTPUT_ROOT:?Source your .env first}"
for name in CW_WORK CW_LATENTS CW_CONDITIONS CW_REFERENCE; do
  if [[ -z "${!name:-}" || ! -d "${!name}" ]]; then
    echo "Missing directory: $name" >&2
    exit 1
  fi
done
export CW_USE_IAA=1 CW_IAA_BETA=3.0 CW_IAA_ENTITY_REFS=2
export CW_I2V_START=0 CW_I2V_OVERSAMPLE=1 CW_PREFIX_NOISE=0
export CW_REVISIT_OVERSAMPLE=1
python scripts/configure.py
python seerworld/stage2_ar/make_stage2_online_config.py \
  --stage1-config configs/generated/stage1.yaml --stage1-lora "$CW_STAGE1_LORA" \
  --out configs/generated/stage2.yaml \
  --output-dir "$CW_OUTPUT_ROOT/seerworld_ar2online_iaafix" \
  --steps 400 --lr 2e-5 --save-every 50 --gt-mix-ratio 0.25 --iaa-beta 3.0
args=(configs/generated/stage2.yaml --stage1-lora "$CW_STAGE1_LORA"
  --distilled-lora "$CW_DISTILLED_LORA" --work "$CW_WORK"
  --latents "$CW_LATENTS" --conditions "$CW_CONDITIONS" --reference "$CW_REFERENCE"
  --max-samples 2048 --gt-mix-ratio 0.25)
python seerworld/verify_iaa_runtime.py
python seerworld/stage2_ar/train_stage2_online_ar.py "${args[@]}" --preflight-only
accelerate launch --num_processes "${CW_NPROC:-8}" --mixed_precision bf16 \
  seerworld/stage2_ar/train_stage2_online_ar.py "${args[@]}" "$@"
