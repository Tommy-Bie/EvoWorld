# Copy to .env, edit model paths, then: source .env
# scripts/download_training_cache.py writes cache paths to .env.cache; source
# that file after this one instead of editing the CW_DATA/CW_WORK values below.
# No automatic search of a user's filesystem or private directories.
export CW_CKPT=/path/to/ltx-2.5-22b-dev-transformer-bf16.safetensors
export CW_GEMMA=/path/to/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
export CW_VIDEO_VAE=/path/to/ltx-2.5-video-vae-bf16.safetensors
export CW_AUDIO_VAE=/path/to/ltx-2.5-audio-vae-bf16.safetensors
export CW_DISTILLED_LORA=/path/to/ltx-2.5-22b-distilled-lora-450-bf16.safetensors
export CW_DATA="${CW_DATA:-/path/to/preprocessed}"
export CW_WORK="${CW_WORK:-/path/to/training_metadata}"
export CW_LATENTS="$CW_DATA/latents"
export CW_CONDITIONS="$CW_DATA/conditions"
export CW_REFERENCE="$CW_DATA/reference_latents"
export CW_OUTPUT_ROOT="$PWD/outputs"
export CW_NPROC=8
# Set after finishing Stage 1.
# export CW_STAGE1_LORA="$CW_OUTPUT_ROOT/seerworld_stage1_iaafix/checkpoints/lora_weights_step_08000.safetensors"
