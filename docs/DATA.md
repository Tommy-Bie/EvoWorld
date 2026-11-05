# EvoWorld Data

[EvoWorld-Data](https://huggingface.co/datasets/Tommy-Bie/EvoWorld-Data)
provides both the original MP4 videos and a preprocessed LTX-2.5 training
cache. Use the videos for custom data processing, or use the cache to start
training directly.

## Download the training cache

```bash
uv run hf auth login
export HF_XET_HIGH_PERFORMANCE=1
export EVOWORLD_DATA_DIR="$PWD/data/EvoWorld-Data"

# Alternatively, choose another location.
# export EVOWORLD_DATA_DIR="/path/to/EvoWorld-Data"

uv run python scripts/download_training_cache.py \
  --dataset-dir "$EVOWORLD_DATA_DIR" \
  --env-file "$EVOWORLD_DATA_DIR/.env.cache"
```

The command downloads the cache, verifies its checksums, extracts it, and
writes the resolved paths to `.env.cache`. Load the paths together with the
model configuration before training:

```bash
source .env
source "$EVOWORLD_DATA_DIR/.env.cache"
```

You can then start training:

```bash
bash scripts/train_stage1.sh
```

## Use the original videos

Download the raw videos and metadata directly from the
[dataset repository](https://huggingface.co/datasets/Tommy-Bie/EvoWorld-Data).
The metadata provides the prompt, duration, and coarse subject, event, and
reaction types for each video.
