# Training data contract

Public SeerWorld-Data contains 12,118 MP4 videos: 11,817 train and 301 additional.
The additional samples were not used to train or validate the paper model, but
may be merged into training for downstream use.
Release metadata includes prompts, duration, and coarse subject/event/reaction
types. It is deliberately distinct from the internal training cache format.

The release may additionally provide an optional LTX-2.5 training cache:

```
training_cache/ltx-2.5/
  training_metadata/
    captions.jsonl
    manifest.jsonl
  preprocessed/
    latents/                    # 12,118 full-video VAE caches
    reference_latents/          # 12,118 entity-reference caches
    conditions/                 # 42,059 per-segment text-feature caches
```

On the dataset Hub, these files are distributed as separate uncompressed TAR
shards for `core` (video latents plus reference latents) and `conditions`
(per-segment text features). Extract the selected shards into one directory;
the archive member paths reconstruct the `preprocessed/` tree shown above.
Verify `SHA256SUMS` before training. The dataset repository includes the exact
download, verification, and extraction commands.

The latent and reference files form the approximately 53 GiB core cache. The
approximately 493 GiB condition cache is a separate optional download for
users who want to reproduce training without running the Gemma text encoder.
Each condition contains `video_prompt_embeds`, `audio_prompt_embeds`, and
`prompt_attention_mask`; both embedding branches are required by the released
LTX-2.5 connectors. Global-prompt condition files are not used by the released
training recipes and are therefore not distributed.

The existing dataset reader expects:

```
training_metadata/                 # CW_WORK
  captions.jsonl
  manifest.jsonl
  camera_phases.jsonl              # optional if valid phase is embedded in manifest
preprocessed/                      # CW_DATA
  latents/                         # CW_LATENTS: full-video VAE caches
  conditions/                      # CW_CONDITIONS: per-segment text caches
  reference_latents/               # CW_REFERENCE: first two temporal latents
```

`captions.jsonl` uses `clip_id`, `segment_prompts`, and segment timing
(`nseg`, `seg_s`). `manifest.jsonl` provides matching clip IDs, split (`train`
or `additional`), frame metadata, and camera phases, or phases are loaded from
`camera_phases.jsonl`. See `OoSWindowIndex` in `seerworld/cw_dataset.py` for the
authoritative schema and filtering rules. Missing phases can discard examples;
they must not be silently replaced with nominal prompt boundaries.

Relevant preprocessing tools retained here:

- `packages/ltx-trainer/scripts/process_videos.py`: video VAE encoding.
- `seerworld/data_process/build_caption_rows.py`: per-segment caption rows.
- `packages/ltx-trainer/scripts/process_captions.py`: text feature caching.
- `seerworld/make_reference_latents.py`: entity-reference cache slicing.

Run each tool with `--help` in the installed environment. Keep cache naming
consistent with `clip_id`; renaming videos alone does not rename caches.
The reference directory must be named `reference_latents`, matching the config.
The dataset uses a 15-latent window: four clean prefix latents and eleven target
latents, with two entity-reference latents supplied separately.

When the optional cache is installed, configure the training paths as follows:

```
CW_WORK=<dataset>/training_cache/ltx-2.5/training_metadata
CW_DATA=<dataset>/training_cache/ltx-2.5/extracted/preprocessed
CW_LATENTS=$CW_DATA/latents
CW_CONDITIONS=$CW_DATA/conditions
CW_REFERENCE=$CW_DATA/reference_latents
```

For the complete preprocessing-free path, the release code automates download,
SHA-256 verification, extraction, resume state, and environment generation:

```bash
hf auth login
export HF_XET_HIGH_PERFORMANCE=1
# Optional representative remote download/check/extraction test:
uv run python scripts/download_training_cache.py \
  --dataset-dir /path/to/SeerWorld-Data --smoke
# Complete cache:
uv run python scripts/download_training_cache.py \
  --dataset-dir /path/to/SeerWorld-Data
source .env.cache
```

The default `--components all` downloads both `core` and `conditions`. Choosing
`--components core` saves download space but is not immediately trainable: text
conditions must then be produced locally. Archives are deleted after successful
verification and extraction unless `--keep-archives` is specified.

The public training metadata includes the exact camera-arrival and visibility
fields needed to reconstruct training windows. Private-to-public ID mappings
are author-side release inputs and must never be distributed.
