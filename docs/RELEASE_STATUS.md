# Release readiness

Included: Stage-1 + IAA, online Stage-2, local LTX runtime sources, selected
preprocessing tools, VLM/renderer modules, portable paths and launchers.

Excluded: historical launch variants, offline Stage-2 experiments, private
datasets/mappings, logs, caches, checkpoint files, service credentials, and
benchmark results. Original source is untouched.

Validated in the original GPU training environment on 2026-10-13:

- `uv sync --locked --extra natten --python 3.10.20` created a fresh environment
  matching the recorded core package and CUDA versions.
- The structural, unit, environment, and IAA runtime checks passed.
- A single-GPU two-step Stage-1 smoke run completed on the real preprocessed
  training cache and saved its LoRA and training state.
- A single-GPU one-step online Stage-2 smoke run completed the distilled
  predecessor rollout, detached-prefix conditioning, IAA/FM backward pass,
  optimizer update, and clean adapter-only checkpoint save.
- The public-ID training metadata and optional LTX-2.5 cache passed a real
  dataset-loader smoke test. Release packaging contains 24,236 core tensors and
  42,059 per-segment condition tensors with no global-prompt conditions.
- All 11 core and 99 condition TAR shards passed SHA-256 verification; the
  conditions verification completed on 2026-10-14.
- The public cache download path now performs authenticated resumable download,
  per-shard SHA-256 verification, safe extraction, exact file-count validation,
  and cache environment generation without requiring the raw MP4 videos.
- On 2026-10-14, a fresh authenticated download from the gated Hub repository
  passed metadata validation, all 110 shard checksum/extraction checks, and the
  Stage-1 and Stage-2 smoke runs using only the newly downloaded public cache.

Known scope:

1. Provide a small public-ID inference example and benchmark evaluation wrapper.
   The smoke runs validate execution, not reproduction of the reported metrics.
2. Model weights and data are distributed separately under their respective
   licenses; this repository does not redistribute either artifact.
3. Audit any additional files before copying them; the included structural
   checker is not a comprehensive secret or legal compliance audit.

