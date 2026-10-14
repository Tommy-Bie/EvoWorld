#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for name in CW_WORK CW_LATENTS CW_CONDITIONS CW_REFERENCE; do
  if [[ -z "${!name:-}" || ! -d "${!name}" ]]; then
    echo "Missing directory: $name (source your .env first)" >&2
    exit 1
  fi
done
export CW_USE_IAA=1 CW_IAA_BETA=3.0 CW_IAA_ENTITY_REFS=2
export CW_I2V_START=0 CW_I2V_OVERSAMPLE=1 CW_PREFIX_NOISE=0
export CW_REVISIT_OVERSAMPLE=3
python scripts/configure.py
python seerworld/verify_iaa_runtime.py
accelerate launch --num_processes "${CW_NPROC:-8}" --mixed_precision bf16 \
  seerworld/train_stage1.py configs/generated/stage1.yaml \
  --work "$CW_WORK" --latents "$CW_LATENTS" \
  --conditions "$CW_CONDITIONS" --reference "$CW_REFERENCE" "$@"
