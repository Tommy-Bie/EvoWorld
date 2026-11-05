# Copy to .env from the repository root, then: source .env
# scripts/download_training_cache.py writes cache paths to the .env.cache file
# selected with --env-file; source it after this file.
# Model files use the repository-local download layout by default.
export EW_LTX_ROOT="${EW_LTX_ROOT:-$PWD/models/ltx-2.5}"
export EW_CKPT="$EW_LTX_ROOT/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
export EW_GEMMA="$EW_LTX_ROOT/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
export EW_VIDEO_VAE="$EW_LTX_ROOT/vae/ltx-2.5-video-vae-bf16.safetensors"
export EW_AUDIO_VAE="$EW_LTX_ROOT/vae/ltx-2.5-audio-vae-bf16.safetensors"
export EW_DISTILLED_LORA="$EW_LTX_ROOT/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
export EW_INFERENCE_LORA="${EW_INFERENCE_LORA:-$PWD/models/EvoWorld/weights/evoworld_lora.safetensors}"
# Training-cache paths are supplied by EvoWorld-Data/.env.cache.
export EW_DATA="${EW_DATA:-}"
export EW_WORK="${EW_WORK:-}"
export EW_LATENTS="${EW_LATENTS:-}"
export EW_CONDITIONS="${EW_CONDITIONS:-}"
export EW_REFERENCE="${EW_REFERENCE:-}"
export EW_OUTPUT_ROOT="$PWD/outputs"
export EW_NPROC=8
# Set after finishing Stage 1.
# export EW_STAGE1_LORA="$EW_OUTPUT_ROOT/evoworld_stage1_iaafix/checkpoints/lora_weights_step_08000.safetensors"

# Optional live VLM inference settings used by scripts/infer_examples.sh.
# export EW_VLM_URL=http://127.0.0.1:8001/v1
# export EW_VLM_MODEL=Qwen3.8-27B
# export EW_VLM_THINKING=0
