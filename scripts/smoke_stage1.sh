#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

for name in EW_CKPT EW_GEMMA EW_VIDEO_VAE EW_AUDIO_VAE; do
  value="${!name:-}"
  if [[ -z "$value" || ! -f "$value" ]]; then
    echo "Missing model file: $name=$value (source .env first)" >&2
    exit 1
  fi
done
for name in EW_WORK EW_LATENTS EW_CONDITIONS EW_REFERENCE EW_OUTPUT_ROOT; do
  value="${!name:-}"
  if [[ -z "$value" ]]; then
    echo "Missing environment variable: $name (source .env first)" >&2
    exit 1
  fi
  if [[ "$name" != "EW_OUTPUT_ROOT" && ! -d "$value" ]]; then
    echo "Missing directory: $name=$value" >&2
    exit 1
  fi
done

SMOKE_STEPS="${EW_SMOKE_STEPS:-2}"
SMOKE_GPU="${EW_SMOKE_GPU:-0}"
SMOKE_TAG="${EW_SMOKE_TAG:-$(date +%Y%m%d_%H%M%S)}"
SMOKE_OUTPUT="${EW_SMOKE_OUTPUT:-$EW_OUTPUT_ROOT/evoworld_stage1_smoke_${SMOKE_TAG}_iaafix}"
SMOKE_CONFIG="${TMPDIR:-/tmp}/evoworld_stage1_smoke_${UID}_${SMOKE_TAG}.yaml"

if [[ -e "$SMOKE_OUTPUT" ]]; then
  echo "Refusing to reuse smoke output directory: $SMOKE_OUTPUT" >&2
  echo "Set a new EW_SMOKE_TAG or EW_SMOKE_OUTPUT." >&2
  exit 1
fi

mkdir -p "$EW_OUTPUT_ROOT"

export EW_USE_IAA=1
export EW_IAA_BETA=3.0
export EW_IAA_ENTITY_REFS=2
export EW_I2V_START=0
export EW_I2V_OVERSAMPLE=1
export EW_PREFIX_NOISE=0
export EW_REVISIT_OVERSAMPLE=3
export WANDB_MODE=disabled

uv run python scripts/configure.py
uv run python scripts/make_stage1_smoke_config.py \
  --input configs/generated/stage1.yaml \
  --output "$SMOKE_CONFIG" \
  --output-dir "$SMOKE_OUTPUT" \
  --steps "$SMOKE_STEPS"
uv run python evoworld/verify_iaa_runtime.py

echo "==========================================================="
echo "EvoWorld Stage-1 smoke run"
echo "GPU:        $SMOKE_GPU"
echo "Steps:      $SMOKE_STEPS"
echo "Config:     $SMOKE_CONFIG"
echo "Output:     $SMOKE_OUTPUT"
echo "==========================================================="

CUDA_VISIBLE_DEVICES="$SMOKE_GPU" uv run python -m accelerate.commands.launch \
  --num_processes 1 --mixed_precision bf16 \
  evoworld/train_stage1.py "$SMOKE_CONFIG" \
  --work "$EW_WORK" --latents "$EW_LATENTS" \
  --conditions "$EW_CONDITIONS" --reference "$EW_REFERENCE" \
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
