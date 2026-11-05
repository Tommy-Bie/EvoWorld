#!/usr/bin/env bash
# Run the six bundled EvoWorld examples with frozen or live VLM reasoning.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEC="$ROOT/examples/prompts"
CASES="${EW_DEMO_CASES:-$SPEC/cases.jsonl}"
PROMPTS="${EW_DEMO_PROMPTS:-$SPEC/reasoning_prompts.jsonl}"
IMAGE_ROOT="${EW_DEMO_IMAGE_ROOT:-$ROOT/examples/media}"
WORK_DIR="${EW_DEMO_WORK_DIR:-$ROOT/outputs/example_inputs}"
LATENTS="$WORK_DIR/latents"
USE_LIVE_VLM="${EW_USE_LIVE_VLM:-0}"
[ "$USE_LIVE_VLM" = "0" ] || [ "$USE_LIVE_VLM" = "1" ] || \
  { echo "ERROR: EW_USE_LIVE_VLM must be 0 or 1" >&2; exit 1; }
if [ "$USE_LIVE_VLM" = "1" ]; then
  OUT_ROOT="${EW_DEMO_OUTPUT:-$ROOT/outputs/example_results_live_vlm}"
else
  OUT_ROOT="${EW_DEMO_OUTPUT:-$ROOT/outputs/example_results}"
fi
PY="${EW_PYTHON:-$ROOT/.venv/bin/python}"
GPU_LIST="${EW_DEMO_GPUS:-0 1 2 3 4 5}"

fail() { echo "ERROR: $*" >&2; exit 1; }
[ -x "$PY" ] || fail "Python not found: $PY (run uv sync or set EW_PYTHON)"

export EW_LTX_ROOT="${EW_LTX_ROOT:-$ROOT/models/ltx-2.5}"
export EW_CKPT="${EW_CKPT:-$EW_LTX_ROOT/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors}"
export EW_GEMMA="${EW_GEMMA:-$EW_LTX_ROOT/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors}"
export EW_VIDEO_VAE="${EW_VIDEO_VAE:-$EW_LTX_ROOT/vae/ltx-2.5-video-vae-bf16.safetensors}"
export EW_DISTILLED_LORA="${EW_DISTILLED_LORA:-$EW_LTX_ROOT/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors}"
export EW_INFERENCE_LORA="${EW_INFERENCE_LORA:-$ROOT/models/EvoWorld/weights/evoworld_lora.safetensors}"

read -r -a GPUS <<< "$GPU_LIST"
[ "${#GPUS[@]}" -gt 0 ] || fail "EW_DEMO_GPUS is empty"
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || fail "invalid GPU index in EW_DEMO_GPUS: $gpu"
done

mkdir -p "$WORK_DIR" "$LATENTS" "$OUT_ROOT"
"$PY" "$ROOT/scripts/prepare_inference_examples.py" \
  --cases "$CASES" --prompts "$PROMPTS" \
  --image-root "$IMAGE_ROOT" --work-dir "$WORK_DIR"

if [ "${1:-}" = "--preflight-only" ]; then
  echo "PASS: example specification is valid; GPU inference was not started"
  exit 0
fi

PY_SITE="$($PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
[ -d "$PY_SITE" ] || fail "cannot resolve site-packages for $PY: $PY_SITE"
NVIDIA_LIBRARY_PATHS="$(find "$PY_SITE/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | sort | tr '\n' ':')"
[ -n "$NVIDIA_LIBRARY_PATHS" ] || fail "no NVIDIA runtime libraries found under $PY_SITE/nvidia"
[ -d "$PY_SITE/torch/lib" ] || fail "PyTorch library directory is missing: $PY_SITE/torch/lib"
export LD_LIBRARY_PATH="${NVIDIA_LIBRARY_PATHS}${PY_SITE}/torch/lib:${LD_LIBRARY_PATH:-}"

for name in EW_CKPT EW_GEMMA EW_VIDEO_VAE EW_DISTILLED_LORA EW_INFERENCE_LORA; do
  [ -n "${!name:-}" ] || fail "$name is not set"
done
for path in "$EW_CKPT" "$EW_GEMMA" "$EW_VIDEO_VAE" \
            "$EW_DISTILLED_LORA" "$EW_INFERENCE_LORA"; do
  [ -f "$path" ] || fail "required weight is missing: $path"
done

if [ "$USE_LIVE_VLM" = "1" ]; then
  : "${EW_VLM_URL:=http://127.0.0.1:8001/v1}"
  : "${EW_VLM_MODEL:=Qwen3.8-27B}"
  export EW_VLM_URL EW_VLM_MODEL
  "$PY" - "$EW_VLM_URL" <<'PY'
import sys
import requests
url = sys.argv[1].rstrip("/") + "/models"
response = requests.get(url, timeout=10)
response.raise_for_status()
print(f"[examples] live VLM ready: {url}")
PY
fi

if [ "${1:-}" = "--check-only" ]; then
  echo "PASS: all example inputs, weights, and requested services are available"
  exit 0
fi

missing=0
while IFS= read -r clip; do
  [ -s "$LATENTS/$clip.pt" ] || missing=$((missing + 1))
done < <("$PY" -c 'import json,sys; [print(json.loads(x)["clip_id"]) for x in open(sys.argv[1], encoding="utf-8") if x.strip()]' "$CASES")
if [ "$missing" -gt 0 ]; then
  echo "[examples] encoding $missing missing first-frame latent(s) on GPU ${GPUS[0]}"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PY" packages/ltx-trainer/scripts/process_videos.py \
      "$WORK_DIR/images.jsonl" --resolution-buckets "896x512x1" \
      --video-column image --output-dir "$LATENTS" \
      --model-path "$EW_VIDEO_VAE" --video-vae-path "$EW_VIDEO_VAE" \
      --batch-size 1 --vae-tiling --overwrite
  )
fi

export EW_ENGINE=i2v EW_LATENTS="$LATENTS" EW_WORK="$WORK_DIR"
export EW_USE_IAA=1 EW_IAA_STRICT=1 EW_IAA_BETA="${EW_IAA_BETA:-3.0}" EW_IAA_ENTITY_REFS=1
export EW_VLM_RETRIEVE=0 EW_FORCE_CAPTIONS_PROMPT=0
unset EW_REFERENCE EW_SCENE_REFERENCE

CLIPS=()
while IFS= read -r clip; do CLIPS+=("$clip"); done < <("$PY" -c 'import json,sys; [print(json.loads(x)["clip_id"]) for x in open(sys.argv[1], encoding="utf-8") if x.strip()]' "$CASES")
[ "${#GPUS[@]}" -ge "${#CLIPS[@]}" ] || fail "need ${#CLIPS[@]} GPU indices; got ${#GPUS[@]}"
LOG_DIR="$OUT_ROOT/_worker_logs"
mkdir -p "$LOG_DIR"

run_case() {
  local gpu="$1" clip="$2" safe="${2//\//__}"
  local args=(
    --testset "$CASES" --pick "$clip"
    --lora "$EW_INFERENCE_LORA" --lora-yaml "$ROOT/configs/stage1.yaml"
    --lora-strength "${EW_OUR_LORA_STRENGTH:-1.0}"
    --distilled-lora "$EW_DISTILLED_LORA"
    --distilled-lora-strength "${EW_DISTILLED_LORA_STRENGTH:-1.0}"
    --steps 8 --seed "${EW_SEED:-42}" --out-root "$OUT_ROOT"
  )
  if [ "$USE_LIVE_VLM" != "1" ]; then
    args+=(--fixed-seg2-prompts "$PROMPTS" --fixed-seg2-mode precomputed-reasoning)
  fi
  export EW_SEG2_SUMMARY_PATH="$LOG_DIR/${safe}.seg2_prompt.jsonl"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" evoworld/inference/deploy_pipeline.py "${args[@]}"
}

mode="frozen VLM prompts"
[ "$USE_LIVE_VLM" = "1" ] && mode="live VLM reasoning"
echo "[examples] rendering ${#CLIPS[@]} cases with $mode; one case per GPU"
PIDS=()
for i in "${!CLIPS[@]}"; do
  gpu="${GPUS[$i]}"; clip="${CLIPS[$i]}"; safe="${clip//\//__}"
  echo "  GPU $gpu -> $clip"
  run_case "$gpu" "$clip" >"$LOG_DIR/${safe}.log" 2>&1 &
  PIDS+=("$!")
done

cleanup_workers() { for pid in "${PIDS[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done; }
trap cleanup_workers INT TERM
failed=0
for i in "${!PIDS[@]}"; do
  clip="${CLIPS[$i]}"; safe="${clip//\//__}"
  if wait "${PIDS[$i]}"; then
    echo "[PASS] GPU ${GPUS[$i]}: $clip"
  else
    echo "[FAIL] GPU ${GPUS[$i]}: $clip; log=$LOG_DIR/${safe}.log" >&2
    tail -n 80 "$LOG_DIR/${safe}.log" >&2 || true
    failed=$((failed + 1))
  fi
done
trap - INT TERM
[ "$failed" -eq 0 ] || fail "$failed inference worker(s) failed"

"$PY" -c 'import json,pathlib,sys; root=pathlib.Path(sys.argv[1]); cases=[json.loads(x) for x in open(sys.argv[2], encoding="utf-8") if x.strip()]; missing=[r["clip_id"] for r in cases if not (root / r["clip_id"].replace("/", "__") / "output.mp4").is_file()]; raise SystemExit(f"missing outputs: {missing}" if missing else 0)' "$OUT_ROOT" "$CASES"
echo "PASS: all ${#CLIPS[@]} EvoWorld example videos were generated"
