#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${EW_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || { echo "Missing Python: $PY (run uv sync or set EW_PYTHON)" >&2; exit 1; }
source "$ROOT/scripts/setup_cuda_libraries.sh"
setup_cuda_libraries "$PY"

: "${EW_STAGE1_LORA:?Set EW_STAGE1_LORA to the checkpoint produced by smoke_stage1.sh}"
: "${EW_DISTILLED_LORA:?Source .env first}"
: "${EW_OUTPUT_ROOT:?Source .env first}"

for name in EW_STAGE1_LORA EW_DISTILLED_LORA; do
  value="${!name}"
  if [[ ! -f "$value" ]]; then
    echo "Missing checkpoint: $name=$value" >&2
    exit 1
  fi
done
for name in EW_WORK EW_LATENTS EW_CONDITIONS EW_REFERENCE; do
  value="${!name:-}"
  if [[ -z "$value" || ! -d "$value" ]]; then
    echo "Missing directory: $name=$value (source .env first)" >&2
    exit 1
  fi
done

SMOKE_STEPS="${EW_STAGE2_SMOKE_STEPS:-1}"
SMOKE_GPU="${EW_SMOKE_GPU:-0}"
SMOKE_SAMPLES="${EW_STAGE2_SMOKE_SAMPLES:-8}"
SMOKE_TAG="${EW_SMOKE_TAG:-$(date +%Y%m%d_%H%M%S)}"
SMOKE_OUTPUT="${EW_STAGE2_SMOKE_OUTPUT:-$EW_OUTPUT_ROOT/evoworld_ar2online_smoke_${SMOKE_TAG}_iaafix}"
SMOKE_CONFIG="${TMPDIR:-/tmp}/evoworld_stage2_smoke_${UID}_${SMOKE_TAG}.yaml"

if [[ -e "$SMOKE_OUTPUT" ]]; then
  echo "Refusing to reuse smoke output directory: $SMOKE_OUTPUT" >&2
  echo "Set a new EW_SMOKE_TAG or EW_STAGE2_SMOKE_OUTPUT." >&2
  exit 1
fi

mkdir -p "$EW_OUTPUT_ROOT"

export EW_USE_IAA=1
export EW_IAA_BETA=3.0
export EW_IAA_ENTITY_REFS=2
export EW_I2V_START=0
export EW_I2V_OVERSAMPLE=1
export EW_PREFIX_NOISE=0
export EW_REVISIT_OVERSAMPLE=1
export WANDB_MODE=disabled

uv run python scripts/configure.py
uv run python evoworld/stage2_ar/make_stage2_online_config.py \
  --stage1-config configs/generated/stage1.yaml \
  --stage1-lora "$EW_STAGE1_LORA" \
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
  --stage1-lora "$EW_STAGE1_LORA"
  --distilled-lora "$EW_DISTILLED_LORA"
  --work "$EW_WORK"
  --latents "$EW_LATENTS"
  --conditions "$EW_CONDITIONS"
  --reference "$EW_REFERENCE"
  --max-samples "$SMOKE_SAMPLES"
  --gt-mix-ratio 0.25
)

uv run python evoworld/verify_iaa_runtime.py
uv run python evoworld/stage2_ar/train_stage2_online_ar.py "${args[@]}" --preflight-only

echo "==========================================================="
echo "EvoWorld Stage-2 online-AR smoke run"
echo "GPU:          $SMOKE_GPU"
echo "Steps:        $SMOKE_STEPS"
echo "Pairs:        $SMOKE_SAMPLES"
echo "Stage-1 LoRA: $EW_STAGE1_LORA"
echo "Config:       $SMOKE_CONFIG"
echo "Output:       $SMOKE_OUTPUT"
echo "==========================================================="

CUDA_VISIBLE_DEVICES="$SMOKE_GPU" uv run python -m accelerate.commands.launch \
  --num_processes 1 --mixed_precision bf16 \
  evoworld/stage2_ar/train_stage2_online_ar.py "${args[@]}" \
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
