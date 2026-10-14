#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

for name in CW_CKPT CW_GEMMA CW_VIDEO_VAE CW_AUDIO_VAE; do
  value="${!name:-}"
  if [[ -z "$value" || ! -f "$value" ]]; then
    echo "Missing model file: $name=$value (source .env first)" >&2
    exit 1
  fi
done
for name in CW_WORK CW_LATENTS CW_CONDITIONS CW_REFERENCE CW_OUTPUT_ROOT; do
  value="${!name:-}"
  if [[ -z "$value" ]]; then
    echo "Missing environment variable: $name (source .env first)" >&2
    exit 1
  fi
  if [[ "$name" != "CW_OUTPUT_ROOT" && ! -d "$value" ]]; then
    echo "Missing directory: $name=$value" >&2
    exit 1
  fi
done

SMOKE_STEPS="${CW_SMOKE_STEPS:-2}"
SMOKE_GPU="${CW_SMOKE_GPU:-0}"
SMOKE_TAG="${CW_SMOKE_TAG:-$(date +%Y%m%d_%H%M%S)}"
SMOKE_OUTPUT="${CW_SMOKE_OUTPUT:-$CW_OUTPUT_ROOT/seerworld_stage1_smoke_${SMOKE_TAG}_iaafix}"
SMOKE_CONFIG="${TMPDIR:-/tmp}/seerworld_stage1_smoke_${UID}_${SMOKE_TAG}.yaml"

if [[ -e "$SMOKE_OUTPUT" ]]; then
  echo "Refusing to reuse smoke output directory: $SMOKE_OUTPUT" >&2
  echo "Set a new CW_SMOKE_TAG or CW_SMOKE_OUTPUT." >&2
  exit 1
fi

mkdir -p "$CW_OUTPUT_ROOT"

export CW_USE_IAA=1
export CW_IAA_BETA=3.0
export CW_IAA_ENTITY_REFS=2
export CW_I2V_START=0
export CW_I2V_OVERSAMPLE=1
export CW_PREFIX_NOISE=0
export CW_REVISIT_OVERSAMPLE=3
export WANDB_MODE=disabled

uv run python scripts/configure.py
uv run python scripts/make_stage1_smoke_config.py \
  --input configs/generated/stage1.yaml \
  --output "$SMOKE_CONFIG" \
  --output-dir "$SMOKE_OUTPUT" \
  --steps "$SMOKE_STEPS"
uv run python seerworld/verify_iaa_runtime.py

echo "==========================================================="
echo "SeerWorld Stage-1 smoke run"
echo "GPU:        $SMOKE_GPU"
echo "Steps:      $SMOKE_STEPS"
echo "Config:     $SMOKE_CONFIG"
echo "Output:     $SMOKE_OUTPUT"
echo "==========================================================="

CUDA_VISIBLE_DEVICES="$SMOKE_GPU" uv run python -m accelerate.commands.launch \
  --num_processes 1 --mixed_precision bf16 \
  seerworld/train_stage1.py "$SMOKE_CONFIG" \
  --work "$CW_WORK" --latents "$CW_LATENTS" \
  --conditions "$CW_CONDITIONS" --reference "$CW_REFERENCE" \
  --disable-progress-bars

CHECKPOINT="$SMOKE_OUTPUT/checkpoints/lora_weights_step_$(printf '%05d' "$SMOKE_STEPS").safetensors"
STATE="$SMOKE_OUTPUT/checkpoints/training_state_step_$(printf '%05d' "$SMOKE_STEPS").pt"
if [[ ! -s "$CHECKPOINT" || ! -s "$STATE" ]]; then
  echo "Smoke training finished but expected checkpoint artifacts are missing:" >&2
  echo "  $CHECKPOINT" >&2
  echo "  $STATE" >&2
  exit 1
fi

echo "PASS: Stage-1 smoke training completed and saved:"
echo "  $CHECKPOINT"
echo "  $STATE"
