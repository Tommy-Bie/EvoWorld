# SeerWorld

Towards Out-of-Sight World Evolution in Generative Video Models

This repository contains the release implementation of SeerWorld. Model
weights and training data are distributed separately; private ID mappings,
service credentials, logs, and internal evaluation artifacts are not bundled.

## Code layout

- `seerworld/train_stage1.py`: Stage-1 LoRA training with identity-anchored attention (IAA).
- `seerworld/stage2_ar/`: Stage-2 **online** one-hop event-to-revisit rollout training.
- `seerworld/iaa_utils.py`: attention token layout and inference-side IAA.
- `seerworld/inference/`: VLM brain, autoregressive renderer, and deployment pipeline.
- `packages/`: local LTX core, pipelines, trainer, and preprocessing CLIs.
- `configs/`, `scripts/`: portable recipes and launchers without machine-specific paths.
- `docs/`: data contract and release-readiness notes.

The local LTX packages include changes required by SeerWorld. Installing only
an unrelated upstream LTX version is not a substitute for these sources.

## Setup

The original training environment used Python 3.10, PyTorch 2.13.0+cu132, and
NATTEN 0.21.7+torch2130cu132. Install the pinned Linux/CUDA environment with
[uv](https://docs.astral.sh/uv/) from the repository root:

```bash
uv sync --extra natten --python 3.10.20
uv run python scripts/check_environment.py
cp scripts/env.example.sh .env
# Edit .env with your model-weight paths.
source .env
```

The local LTX workspace packages include the SeerWorld changes and are installed
by the same command. Do not install Diffusers for this code path. See
`docs/TRAINING_ENVIRONMENT.md` for the verified versions and
`docs/RELEASE_STATUS.md` for the remaining end-to-end validation status.

### LTX-2.5 model weights

SeerWorld is trained from LTX-2.5. Accept the model terms on Hugging Face, log
in with a token that can read gated repositories, and download the official
components used by the training recipes:

```bash
hf auth login
hf download Lightricks/LTX-2.5 \
  diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  vae/ltx-2.5-video-vae-bf16.safetensors \
  vae/ltx-2.5-audio-vae-bf16.safetensors \
  loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors \
  --local-dir models/ltx-2.5
```

Then point `CW_CKPT`, `CW_GEMMA`, `CW_VIDEO_VAE`, `CW_AUDIO_VAE`, and
`CW_DISTILLED_LORA` in `.env` to the downloaded files. The model paths and
current filenames are documented by the official
[LTX-2 repository](https://github.com/Lightricks/LTX-2) and hosted in the
[LTX-2.5 model repository](https://huggingface.co/Lightricks/LTX-2.5).

## Training

The simplest training path uses the released LTX-2.5 cache and does not require
downloading or preprocessing the raw MP4 videos. SeerWorld-Data is gated, so
request access and authenticate once before preparing the cache:

```bash
hf auth login
export HF_XET_HIGH_PERFORMANCE=1

# Optional ~10 GiB remote-path smoke check (one core + one condition shard).
uv run python scripts/download_training_cache.py \
  --dataset-dir /path/to/SeerWorld-Data --smoke

# Downloads each shard, verifies SHA-256, extracts it, and removes the TAR by
# default to avoid requiring space for both archive and extracted copies.
uv run python scripts/download_training_cache.py \
  --dataset-dir /path/to/SeerWorld-Data

# Keep model paths in .env and generated cache paths in .env.cache.
source .env
source .env.cache
```

The command is resumable: completed shard records are checked against the
extracted files before a shard is skipped. Use `--keep-archives` only when
there is enough space for both forms. See `docs/DATA.md` for manual download,
preprocessing, and cache-layout details.

```bash
# First validate model loading, real data, IAA, backward/optimizer, and saving
# with a single-GPU two-step run (source .env first).
bash scripts/smoke_stage1.sh

# Point this to the checkpoint printed by the command above, then validate one
# online Stage-2 rollout/training step and adapter-only checkpoint saving.
export CW_STAGE1_LORA=/path/to/lora_weights_step_00002.safetensors
bash scripts/smoke_stage2.sh

# Full Stage 1 and Stage 2 recipes.
bash scripts/train_stage1.sh
export CW_STAGE1_LORA="$CW_OUTPUT_ROOT/seerworld_stage1_iaafix/checkpoints/lora_weights_step_08000.safetensors"
bash scripts/train_stage2.sh
```

Stage 1: rank/alpha 128, learning rate 1e-4, 8,000 steps, batch 1 per GPU,
gradient accumulation 8, revisit oversampling 3, IAA beta 3.0.
Stage 2: online distilled 8-step predecessor rollout, detached four-latent
prefix, learning rate 2e-5, GT-prefix mixture 0.25, up to 2,048 selected pairs,
batch 1 and gradient accumulation 1. The launcher runs 400 steps and saves every
50; the reported full-method evaluation used the **step-200 checkpoint**.

Training uses two temporal entity-reference latents; I2V benchmark inference
uses a single-frame reference. The source also retains research options beyond
the default recipe. Original private validation cases are omitted and scheduled
validation is disabled in this portable configuration.

## Inference and evaluation

`seerworld/inference/deploy_pipeline.py --help` exposes the VLM-brain + renderer
pipeline. It currently uses the internal caption/cache interface; this is not
yet a polished public image-and-prompt CLI. The VLM server is separate and must
be configured through environment variables, without committing credentials.
The existing benchmark repository remains the evaluation source of truth;
benchmark adapters and a public-data inference example are pending integration.

## Checks

```bash
uv run python scripts/check_release.py
uv run python seerworld/test_iaa_layout.py
uv run python -m unittest discover -s tests
# Requires the GPU training environment:
uv run python seerworld/verify_iaa_runtime.py
```

## Licensing

Original SeerWorld code is released under the Apache License 2.0; see
`LICENSE`. The vendored and modified LTX-2 components under `packages/`, and
SeerWorld components that constitute derivatives of LTX-2.x, remain subject to
the LTX-2.x Community License Agreement in `licenses/LTX-LICENSE.md`. See
`NOTICE` for attribution and modification details. Dataset and model-weight
licenses are separate and must be accepted at their respective repositories.
