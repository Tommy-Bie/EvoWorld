#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${CW_STAGE1_LORA:?Set CW_STAGE1_LORA to the checkpoint produced by smoke_stage1.sh}"
: "${CW_DISTILLED_LORA:?Source .env first}"
: "${CW_OUTPUT_ROOT:?Source .env first}"

for name in CW_STAGE1_LORA CW_DISTILLED_LORA; do
  value="${!name}"
  if [[ ! -f "$value" ]]; then
    echo "Missing checkpoint: $name=$value" >&2
    exit 1
  fi
done
for name in CW_WORK CW_LATENTS CW_CONDITIONS CW_REFERENCE; do
  value="${!name:-}"
  if [[ -z "$value" || ! -d "$value" ]]; then
    echo "Missing directory: $name=$value (source .env first)" >&2
    exit 1
  fi
done

SMOKE_STEPS="${CW_STAGE2_SMOKE_STEPS:-1}"
SMOKE_GPU="${CW_SMOKE_GPU:-0}"
SMOKE_SAMPLES="${CW_STAGE2_SMOKE_SAMPLES:-8}"
SMOKE_TAG="${CW_SMOKE_TAG:-$(date +%Y%m%d_%H%M%S)}"
SMOKE_OUTPUT="${CW_STAGE2_SMOKE_OUTPUT:-$CW_OUTPUT_ROOT/seerworld_ar2online_smoke_${SMOKE_TAG}_iaafix}"
SMOKE_CONFIG="${TMPDIR:-/tmp}/seerworld_stage2_smoke_${UID}_${SMOKE_TAG}.yaml"

if [[ -e "$SMOKE_OUTPUT" ]]; then
  echo "Refusing to reuse smoke output directory: $SMOKE_OUTPUT" >&2
  echo "Set a new CW_SMOKE_TAG or CW_STAGE2_SMOKE_OUTPUT." >&2
  exit 1
fi

mkdir -p "$CW_OUTPUT_ROOT"

export CW_USE_IAA=1
export CW_IAA_BETA=3.0
export CW_IAA_ENTITY_REFS=2
export CW_I2V_START=0
export CW_I2V_OVERSAMPLE=1
export CW_PREFIX_NOISE=0
export CW_REVISIT_OVERSAMPLE=1
export WANDB_MODE=disabled

uv run python scripts/configure.py
uv run python seerworld/stage2_ar/make_stage2_online_config.py \
  --stage1-config configs/generated/stage1.yaml \
  --stage1-lora "$CW_STAGE1_LORA" \
  --out "$SMOKE_CONFIG" \
  --output-dir "$SMOKE_OUTPUT" \
  --steps "$SMOKE_STEPS" \
  --save-every "$SMOKE_STEPS" \
  --lr 2e-5 \
  --grad-accum 1 \
  --gt-mix-ratio 0.25 \
  --iaa-beta 3.0 \
  --save-training-state minimal

args=(
  "$SMOKE_CONFIG"
  --stage1-lora "$CW_STAGE1_LORA"
  --distilled-lora "$CW_DISTILLED_LORA"
  --work "$CW_WORK"
  --latents "$CW_LATENTS"
  --conditions "$CW_CONDITIONS"
  --reference "$CW_REFERENCE"
  --max-samples "$SMOKE_SAMPLES"
  --gt-mix-ratio 0.25
)

uv run python seerworld/verify_iaa_runtime.py
uv run python seerworld/stage2_ar/train_stage2_online_ar.py "${args[@]}" --preflight-only

echo "==========================================================="
echo "SeerWorld Stage-2 online-AR smoke run"
echo "GPU:          $SMOKE_GPU"
echo "Steps:        $SMOKE_STEPS"
echo "Pairs:        $SMOKE_SAMPLES"
echo "Stage-1 LoRA: $CW_STAGE1_LORA"
echo "Config:       $SMOKE_CONFIG"
echo "Output:       $SMOKE_OUTPUT"
echo "==========================================================="

CUDA_VISIBLE_DEVICES="$SMOKE_GPU" uv run python -m accelerate.commands.launch \
  --num_processes 1 --mixed_precision bf16 \
  seerworld/stage2_ar/train_stage2_online_ar.py "${args[@]}" \
  --disable-progress-bars

STEP_PADDED="$(printf '%05d' "$SMOKE_STEPS")"
CHECKPOINT="$SMOKE_OUTPUT/checkpoints/lora_weights_step_${STEP_PADDED}.safetensors"
STATE="$SMOKE_OUTPUT/checkpoints/training_state_step_${STEP_PADDED}.pt"
if [[ ! -s "$CHECKPOINT" || ! -s "$STATE" ]]; then
  echo "Stage-2 smoke training finished but expected artifacts are missing:" >&2
  echo "  $CHECKPOINT" >&2
  echo "  $STATE" >&2
  exit 1
fi

echo "PASS: Stage-2 smoke training completed and saved a clean current adapter:"
echo "  $CHECKPOINT"
echo "  $STATE"
