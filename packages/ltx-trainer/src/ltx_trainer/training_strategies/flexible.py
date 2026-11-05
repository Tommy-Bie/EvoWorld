"""Flexible conditioning strategy used by EvoWorld training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_core.types import SpatioTemporalScaleFactors
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)

_IAA_LAYOUT_LOGGED = False

# =============================================================================
# Configuration Classes
# =============================================================================


class IntrinsicConditionBase(BaseModel):

    model_config = ConfigDict(extra="forbid")

    probability: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Probability of applying this condition",
    )


class FirstFrameConditionConfig(IntrinsicConditionBase):

    type: Literal["first_frame"] = "first_frame"


class PrefixConditionConfig(IntrinsicConditionBase):

    type: Literal["prefix"] = "prefix"
    temporal_boundary: int = Field(
        ...,
        ge=1,
        description="Number of temporal units for prefix region. "
        "For video: number of latent frames. For audio: number of audio latent timesteps.",
    )


class SuffixConditionConfig(IntrinsicConditionBase):

    type: Literal["suffix"] = "suffix"
    temporal_boundary: int = Field(
        ...,
        ge=1,
        description="Number of temporal units for suffix region. "
        "For video: number of latent frames. For audio: number of audio latent timesteps.",
    )


class SpatialCropConditionConfig(IntrinsicConditionBase):

    type: Literal["spatial_crop"] = "spatial_crop"
    spatial_region: tuple[int, int, int, int] = Field(
        ...,
        description="Spatial crop region as (y1, x1, y2, x2) in pixel coordinates",
    )


class MaskConditionConfig(IntrinsicConditionBase):

    type: Literal["mask"] = "mask"
    mask_dir: str = Field(
        ...,
        description="Directory containing per-sample masks",
    )


class ReferenceConditionConfig(BaseModel):

    model_config = ConfigDict(extra="forbid")

    type: Literal["reference"] = "reference"
    latents_dir: str = Field(..., description="Directory for reference latents")
    probability: float = Field(default=1.0, ge=0.0, le=1.0, description="Probability of applying this condition")


# Discriminated union for condition configs
ConditionConfig = Annotated[
    Union[
        FirstFrameConditionConfig,
        PrefixConditionConfig,
        SuffixConditionConfig,
        SpatialCropConditionConfig,
        MaskConditionConfig,
        ReferenceConditionConfig,
    ],
    Field(discriminator="type"),
]


class ModalityConfig(BaseModel):

    model_config = ConfigDict(extra="forbid")

    is_generated: bool = Field(
        ...,
        description="True = generated modality (denoised, contributes to loss), False = conditioning-only modality",
    )

    latents_dir: str = Field(
        ...,
        description="Directory for latents",
    )

    conditions: list[ConditionConfig] = Field(
        default_factory=list,
        description="List of conditions (e.g. first_frame, prefix, reference). Text conditioning is always applied.",
    )


class FlexibleStrategyConfig(TrainingStrategyConfigBase):

    name: Literal["flexible"] = "flexible"

    video: ModalityConfig | None = Field(
        default=None,
        description="Video modality configuration",
    )

    audio: ModalityConfig | None = Field(
        default=None,
        description="Audio modality configuration",
    )

    @model_validator(mode="after")
    def validate_at_least_one_generated(self) -> "FlexibleStrategyConfig":
        has_video_target = self.video is not None and self.video.is_generated
        has_audio_target = self.audio is not None and self.audio.is_generated
        if not has_video_target and not has_audio_target:
            raise ValueError("At least one modality must have is_generated=true")
        return self

    @model_validator(mode="after")
    def validate_audio_intrinsic_regions(self) -> "FlexibleStrategyConfig":
        if self.audio is None:
            return self
        for cond in self.audio.conditions:
            if isinstance(cond, (FirstFrameConditionConfig, SpatialCropConditionConfig)):
                raise ValueError(
                    f"Intrinsic condition '{cond.type}' is not supported for audio. "
                    f"Audio supports: prefix, suffix, mask."
                )
        return self

    def get_data_sources(self) -> dict[str, str]:
        sources: dict[str, str] = {"conditions": "conditions"}

        if self.video is not None:
            sources[self.video.latents_dir] = "video_latents"
        if self.audio is not None:
            sources[self.audio.latents_dir] = "audio_latents"

        for modality_config in (self.video, self.audio):
            if modality_config is None:
                continue
            for cond in modality_config.conditions:
                if isinstance(cond, ReferenceConditionConfig):
                    sources[cond.latents_dir] = cond.latents_dir
                elif isinstance(cond, MaskConditionConfig):
                    sources[cond.mask_dir] = cond.mask_dir

        return sources


# =============================================================================
# Helper Data Structures
# =============================================================================


@dataclass
class ModalityProcessingResult:

    modality: Modality
    targets: Tensor | None
    loss_mask: Tensor | None


@dataclass
class LatentData:

    latents: Tensor  # [B, seq_len, C]
    num_frames: int
    height: int
    width: int
    fps: float


# =============================================================================
# FlexibleStrategy Implementation
# =============================================================================


class FlexibleStrategy(TrainingStrategy):

    config: FlexibleStrategyConfig

    def __init__(self, config: FlexibleStrategyConfig):
        super().__init__(config)
        self.config = config
        self.reference_spatial_scale_factor, self.reference_temporal_scale_factor = (
            self._infer_reference_scale_factors_from_config()
        )


        import os as _os
        self._i2v_start = int(_os.environ.get("EW_I2V_START", "0") or 0)
        self._i2v_oversample = int(_os.environ.get("EW_I2V_OVERSAMPLE", "1") or 1)

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        video_result = self._process_modality(self.config.video, batch, "video", timestep_sampler)
        audio_result = self._process_modality(self.config.audio, batch, "audio", timestep_sampler)

        return ModelInputs(
            video=video_result.modality if video_result else None,
            audio=audio_result.modality if audio_result else None,
            video_targets=video_result.targets if video_result else None,
            audio_targets=audio_result.targets if audio_result else None,
            video_loss_mask=video_result.loss_mask if video_result else None,
            audio_loss_mask=audio_result.loss_mask if audio_result else None,
        )

    def compute_loss(
        self,
        video_pred: Tensor | None,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        total_loss = None

        if video_pred is not None and inputs.video_targets is not None:
            video_loss = self._compute_modality_loss(
                pred=video_pred,
                targets=inputs.video_targets,
                loss_mask=inputs.video_loss_mask,
            )
            total_loss = video_loss

        if audio_pred is not None and inputs.audio_targets is not None:
            audio_loss = self._compute_modality_loss(
                pred=audio_pred,
                targets=inputs.audio_targets,
                loss_mask=inputs.audio_loss_mask,
            )
            total_loss = audio_loss if total_loss is None else total_loss + audio_loss

        if total_loss is None:
            raise ValueError("No valid predictions and targets provided for loss computation")

        return total_loss

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        spatial = self.reference_spatial_scale_factor
        temporal = self.reference_temporal_scale_factor
        if spatial is not None and spatial != 1:
            metadata["reference_spatial_scale_factor"] = spatial
            metadata["reference_downscale_factor"] = spatial  # backward compat
        if temporal is not None and temporal != 1:
            metadata["reference_temporal_scale_factor"] = temporal
        return metadata

    def _infer_reference_scale_factors_from_config(self) -> tuple[int | None, int | None]:
        if self.config.video is None:
            return None, None
        for cond in self.config.video.conditions:
            if not isinstance(cond, ReferenceConditionConfig):
                continue
            target_dir = Path(self.config.video.latents_dir)
            ref_dir = Path(cond.latents_dir)
            for sample_file in target_dir.rglob("*.pt"):
                ref_file = ref_dir / sample_file.relative_to(target_dir)
                if not ref_file.exists():
                    continue
                target_data = torch.load(sample_file, map_location="cpu", weights_only=True)
                ref_data = torch.load(ref_file, map_location="cpu", weights_only=True)
                if "height" not in ref_data or "height" not in target_data:
                    continue
                spatial = self._infer_scale_factor(
                    ref_data["height"],
                    ref_data["width"],
                    target_data["height"],
                    target_data["width"],
                )
                temporal = self._infer_temporal_scale_factor(
                    ref_data["num_frames"],
                    target_data["num_frames"],
                )
                return spatial, temporal
        return None, None

    def _process_modality(
        self,
        modality_config: ModalityConfig | None,
        batch: dict[str, Any],
        modality_key: str,
        timestep_sampler: TimestepSampler,
    ) -> ModalityProcessingResult | None:
        if modality_config is None:
            return None

        # Step 1: Load and patchify latents
        data = self._patchify_latent_data(batch[f"{modality_key}_latents"], modality_key)
        latents = data.latents

        batch_size, seq_len, _ = latents.shape
        device = latents.device
        dtype = latents.dtype

        # Step 2: Get text embeddings
        conditions = batch["conditions"]
        prompt_embeds = conditions[f"{modality_key}_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        # Step 3: Initialize noise, timesteps, and loss mask based on is_generated flag
        if modality_config.is_generated:
            noisy_latents, targets, timesteps, loss_mask, sigmas = self._initialize_noisy_target(
                latents, timestep_sampler
            )
        else:
            # Conditioning modality: keep clean (sigma=0), no loss
            noisy_latents = latents
            targets = None
            timesteps = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
            loss_mask = None
            sigmas = torch.zeros(batch_size, device=device, dtype=dtype)

        # Step 4: Generate positions
        if modality_key == "video":
            positions = self._get_video_positions(
                num_frames=data.num_frames,
                height=data.height,
                width=data.width,
                batch_size=batch_size,
                fps=data.fps,
                device=device,
            )
        else:
            positions = self._get_audio_positions(
                num_time_steps=seq_len,
                batch_size=batch_size,
                device=device,
            )

        # Step 5: Apply conditions (intrinsic first, then extrinsic)
        for cond in modality_config.conditions:
            if isinstance(cond, IntrinsicConditionBase) and modality_config.is_generated:
                noisy_latents, timesteps, loss_mask = self._apply_intrinsic_condition(
                    noisy_latents=noisy_latents,
                    clean_latents=latents,
                    timesteps=timesteps,
                    loss_mask=loss_mask,
                    config=cond,
                    height=data.height,
                    width=data.width,
                    batch=batch,
                )

        for cond in modality_config.conditions:
            if isinstance(cond, ReferenceConditionConfig):
                noisy_latents, positions, timesteps, loss_mask, targets = self._apply_reference_condition(
                    noisy_latents=noisy_latents,
                    positions=positions,
                    timesteps=timesteps,
                    loss_mask=loss_mask,
                    targets=targets,
                    batch=batch,
                    config=cond,
                    modality_key=modality_key,
                )


        # Modality.attention_mask


        self_attn_mask = None
        if modality_key == "video":
            import os as _os_iaa
            if int(_os_iaa.environ.get("EW_USE_IAA", "0") or 0) != 0:
                self_attn_mask = self._build_iaa_mask(
                    batch=batch,
                    seq_len=noisy_latents.shape[1],
                    target_window_seq_len=seq_len,
                    loss_mask=loss_mask,
                    # Height and width are already expressed on the latent grid.
                    h_pat=data.height,
                    w_pat=data.width,
                    device=noisy_latents.device,
                    dtype=noisy_latents.dtype,
                )

        # Step 6: Build Modality
        modality = Modality(
            enabled=True,
            latent=noisy_latents,
            sigma=sigmas,
            timesteps=timesteps,
            positions=positions,
            context=prompt_embeds,
            context_mask=prompt_attention_mask,
            attention_mask=self_attn_mask,
        )

        return ModalityProcessingResult(
            modality=modality,
            targets=targets,
            loss_mask=loss_mask,
        )

    @staticmethod
    def _build_iaa_mask(
        batch: dict[str, Any],
        seq_len: int,
        target_window_seq_len: int,
        loss_mask: Tensor | None,
        h_pat: int,
        w_pat: int,
        device,
        dtype,
    ) -> Tensor | None:
        global _IAA_LAYOUT_LOGGED
        import math as _math
        import os as _os_iaa
        is_revisit = batch.get("is_revisit")
        if is_revisit is None:
            raise RuntimeError("IAA is enabled but batch has no top-level is_revisit field")
        if not isinstance(is_revisit, torch.Tensor):
            is_revisit = torch.as_tensor(is_revisit, device=device)
        if is_revisit.sum().item() == 0:
            return None

        beta = float(_os_iaa.environ.get("EW_IAA_BETA", "3.0") or 3.0)
        n_entity_refs = int(_os_iaa.environ.get("EW_IAA_ENTITY_REFS", "2") or 2)

        n_sink = n_entity_refs * h_pat * w_pat
        if n_sink <= 0 or n_sink >= seq_len:
            raise RuntimeError(
                f"invalid IAA entity sink: refs={n_entity_refs}, tokens={n_sink}, seq={seq_len}")

        actual_reference_tokens = seq_len - target_window_seq_len
        if actual_reference_tokens != n_sink:
            raise RuntimeError(
                "IAA reference length mismatch: "
                f"combined={seq_len}, target_window={target_window_seq_len}, "
                f"actual_ref_tokens={actual_reference_tokens}, configured_sink={n_sink} "
                f"(EW_IAA_ENTITY_REFS={n_entity_refs})"
            )

        if loss_mask is None or loss_mask.shape != (is_revisit.shape[0], seq_len):
            raise RuntimeError(
                f"IAA requires loss_mask [B,T]={is_revisit.shape[0], seq_len}, "
                f"got {None if loss_mask is None else tuple(loss_mask.shape)}"
            )
        from iaa_utils import training_iaa_ranges
        (window_start, window_end), (ks, ke) = training_iaa_ranges(seq_len, n_sink)

        b = is_revisit.shape[0]
        mask = torch.ones(b, seq_len, seq_len, device=device, dtype=dtype)
        boost = _math.exp(beta)
        idx = is_revisit.nonzero(as_tuple=False).flatten().tolist()
        for b_i in idx:
            q_idx = loss_mask[b_i].nonzero(as_tuple=False).flatten()
            if q_idx.numel() == 0:
                raise RuntimeError("IAA revisit sample has no generated/loss-bearing target queries")
            qs, qe = int(q_idx[0].item()), int(q_idx[-1].item()) + 1
            if q_idx.numel() != qe - qs or qs < window_start or qe > window_end:
                raise RuntimeError(
                    f"IAA target queries must be one contiguous span inside "
                    f"[{window_start},{window_end}), got [{qs},{qe}) count={q_idx.numel()}"
                )
            mask[b_i, qs:qe, ks:ke] = mask[b_i, qs:qe, ks:ke] * boost
            if not _IAA_LAYOUT_LOGGED and _os_iaa.environ.get("LOCAL_RANK", "0") == "0":
                print(
                    f"[IAA/train] layout=[entity|temporal|target] q=[{qs},{qe}) "
                    f"entity_k=[{ks},{ke}) beta={beta} seq={seq_len}",
                    flush=True,
                )
                _IAA_LAYOUT_LOGGED = True
        return mask

    @staticmethod
    def _initialize_noisy_target(
        latents: Tensor,
        timestep_sampler: TimestepSampler,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, seq_len, _ = latents.shape
        sigmas = timestep_sampler.sample_for(latents)
        noise = torch.randn_like(latents)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_latents = (1 - sigmas_expanded) * latents + sigmas_expanded * noise
        targets = noise - latents  # velocity prediction
        timesteps = sigmas.view(-1, 1).expand(batch_size, seq_len).clone()
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=latents.device)
        return noisy_latents, targets, timesteps, loss_mask, sigmas

    def _apply_intrinsic_condition(
        self,
        noisy_latents: Tensor,
        clean_latents: Tensor,
        timesteps: Tensor,
        loss_mask: Tensor,
        config: IntrinsicConditionBase,
        height: int,
        width: int,
        batch: dict[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, seq_len, _ = noisy_latents.shape
        device = noisy_latents.device

        # Per-sample Bernoulli draw -- each element is independently conditioned.
        apply_per_sample = torch.rand(batch_size, device=device) < config.probability
        if not apply_per_sample.any():
            return noisy_latents, timesteps, loss_mask

        if isinstance(config, FirstFrameConditionConfig):
            mask = self._compute_temporal_mask(batch_size, seq_len, height, width, 1, False, device)
        elif isinstance(config, PrefixConditionConfig):


            if self._i2v_start:
                fps = self._i2v_start_frames(batch, batch_size, config.temporal_boundary)
                if fps is not None:
                    mask = self._compute_temporal_mask(
                        batch_size, seq_len, height, width, fps, False, device)
                else:
                    mask = self._compute_temporal_mask(
                        batch_size, seq_len, height, width, config.temporal_boundary, False, device)
            else:
                mask = self._compute_temporal_mask(
                    batch_size, seq_len, height, width, config.temporal_boundary, False, device)
        elif isinstance(config, SuffixConditionConfig):
            mask = self._compute_temporal_mask(
                batch_size, seq_len, height, width, config.temporal_boundary, True, device
            )
        elif isinstance(config, SpatialCropConditionConfig):
            mask = self._compute_spatial_crop_mask(
                batch_size, seq_len, height, width, config.spatial_region, device, self.video_scale_factors
            )
        elif isinstance(config, MaskConditionConfig):
            # Binarize to match inference, which thresholds masks at load time
            # (validation_runner._load_and_downsample_mask / _load_audio_mask).
            mask = (batch[config.mask_dir]["mask"].reshape(batch_size, seq_len) > 0.5).float()
        else:
            raise ValueError(f"Unknown intrinsic condition type: {type(config).__name__}")

        # Zero the mask for samples the per-sample draw did not select.
        mask = mask * apply_per_sample.view(-1, 1).to(mask.dtype)

        # Apply binary mask: clean conditioned tokens, noisy generated tokens.
        m = mask.unsqueeze(-1)
        noisy_latents = m * clean_latents + (1 - m) * noisy_latents
        timesteps = (1 - mask) * timesteps
        loss_mask = loss_mask & (mask == 0)

        return noisy_latents, timesteps, loss_mask

    def _i2v_start_frames(self, batch, batch_size, default_frames):
        metas = batch.get("meta")
        if isinstance(metas, dict):
            metas = [metas]
        if not isinstance(metas, (list, tuple)):
            return None
        per_first = []
        for m in metas:
            if isinstance(m, dict) and "is_first" in m:
                per_first.append(bool(m["is_first"]))
            elif isinstance(m, dict) and "win" in m:
                per_first.append(int(m["win"]) == 0)
            else:
                return None
        if len(per_first) != batch_size:
            return None
        if not any(per_first):
            return None
        return [1 if f else default_frames for f in per_first]

    @staticmethod
    def _compute_temporal_mask(
        batch_size: int,
        seq_len: int,
        height: int,
        width: int,
        num_frames,
        from_end: bool,
        device: torch.device,
    ) -> Tensor:
        tokens_per_frame = height * width
        mask = torch.zeros(batch_size, seq_len, device=device)
        if isinstance(num_frames, (list, tuple)):

            if len(set(num_frames)) == 1:
                nt = num_frames[0] * tokens_per_frame
                if from_end:
                    mask[:, -nt:] = 1.0
                else:
                    mask[:, :nt] = 1.0
            else:
                for bi, fr in enumerate(num_frames):
                    nt = fr * tokens_per_frame
                    if from_end:
                        mask[bi, -nt:] = 1.0
                    else:
                        mask[bi, :nt] = 1.0
            return mask
        num_tokens = num_frames * tokens_per_frame
        if from_end:
            mask[:, -num_tokens:] = 1.0
        else:
            mask[:, :num_tokens] = 1.0
        return mask

    @staticmethod
    def _compute_spatial_crop_mask(
        batch_size: int,
        seq_len: int,
        height: int,
        width: int,
        region: tuple[int, int, int, int],
        device: torch.device,
        scale_factors: SpatioTemporalScaleFactors,
    ) -> Tensor:
        y1, x1, y2, x2 = region
        num_frames = seq_len // (height * width)

        # Convert pixel to latent coordinates and clamp (per-axis VAE scale factor).
        def to_latent(v: int, scale: int, max_v: int) -> int:
            return max(0, min(v // scale, max_v))

        ly1 = to_latent(y1, scale_factors.height, height)
        ly2 = to_latent(y2, scale_factors.height, height)
        lx1 = to_latent(x1, scale_factors.width, width)
        lx2 = to_latent(x2, scale_factors.width, width)

        # Create spatial mask and tile across frames
        spatial_mask = torch.zeros(height, width, device=device)
        spatial_mask[ly1:ly2, lx1:lx2] = 1.0
        full_mask = spatial_mask.flatten().repeat(num_frames)

        return full_mask.unsqueeze(0).expand(batch_size, -1)

    def _patchify_latent_data(self, latent_data: dict[str, Any], modality_key: str) -> LatentData:
        latents = latent_data["latents"]

        if modality_key == "video":
            num_frames = latent_data["num_frames"][0].item()
            height = latent_data["height"][0].item()
            width = latent_data["width"][0].item()
            fps = latent_data.get("fps")
            fps = fps[0].item() if fps is not None else DEFAULT_FPS
            latents = self._video_patchifier.patchify(latents)
        else:
            num_frames = latent_data.get("num_frames", [latents.shape[2]])[0]
            if isinstance(num_frames, Tensor):
                num_frames = num_frames.item()
            height = 1
            width = 1
            fps = 1.0
            latents = self._audio_patchifier.patchify(latents)

        return LatentData(latents=latents, num_frames=num_frames, height=height, width=width, fps=fps)

    def _apply_reference_condition(
        self,
        noisy_latents: Tensor,
        positions: Tensor,
        timesteps: Tensor,
        loss_mask: Tensor | None,
        targets: Tensor | None,
        batch: dict[str, Any],
        config: ReferenceConditionConfig,
        modality_key: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        if torch.rand((), device=noisy_latents.device).item() >= config.probability:
            return noisy_latents, positions, timesteps, loss_mask, targets

        # Load and patchify condition latents
        cond = self._patchify_latent_data(batch[config.latents_dir], modality_key)
        cond_latents = cond.latents

        batch_size, cond_seq_len, _ = cond_latents.shape
        device = cond_latents.device
        dtype = cond_latents.dtype

        # Generate condition positions
        if modality_key == "video":
            cond_positions = self._get_video_positions(
                num_frames=cond.num_frames,
                height=cond.height,
                width=cond.width,
                batch_size=batch_size,
                fps=cond.fps,
                device=device,
            )
        else:
            cond_positions = self._get_audio_positions(
                num_time_steps=cond_seq_len,
                batch_size=batch_size,
                device=device,
            )

        # Translate / rescale ref positions into the target's frame (video only).
        if modality_key == "video":
            spatial_sf = self.reference_spatial_scale_factor or 1
            temporal_sf = self.reference_temporal_scale_factor or 1
            if spatial_sf != 1 or temporal_sf != 1:
                cond_positions = cond_positions.clone()
                if temporal_sf != 1:
                    # Ref positions are already at the ref's effective fps (source_fps / S,
                    # stored by process_videos.py). Shift by (S - 1) / target_fps so ref's
                    # last patch aligns with target's last; clamp the causal patch at 0.
                    t_target = positions[:, 0, 0:1, 1:2]  # = 1 / target_fps
                    cond_positions[:, 0, ...] = torch.clamp(
                        cond_positions[:, 0, ...] - (temporal_sf - 1) * t_target, min=0
                    )
                if spatial_sf != 1:
                    cond_positions[:, 1, ...] *= spatial_sf
                    cond_positions[:, 2, ...] *= spatial_sf

        # Condition tokens: clean, timestep=0, no loss
        cond_timesteps = torch.zeros(batch_size, cond_seq_len, device=device, dtype=dtype)
        cond_loss_mask = torch.zeros(batch_size, cond_seq_len, dtype=torch.bool, device=device)

        # Concatenate condition and target sequences (condition first, then target)
        combined_latents = torch.cat([cond_latents, noisy_latents], dim=1)
        combined_positions = torch.cat([cond_positions, positions], dim=2)
        combined_timesteps = torch.cat([cond_timesteps, timesteps], dim=1)

        combined_loss_mask = torch.cat([cond_loss_mask, loss_mask], dim=1) if loss_mask is not None else None

        # Targets remain unchanged (only for target portion, not condition portion)

        return combined_latents, combined_positions, combined_timesteps, combined_loss_mask, targets

    @staticmethod
    def _compute_modality_loss(pred: Tensor, targets: Tensor, loss_mask: Tensor) -> Tensor:
        # Slice prediction to match targets length (removes any prepended condition tokens)
        target_len = targets.shape[1]
        pred = pred[:, -target_len:, :]
        mask = loss_mask[:, -target_len:]

        # Compute masked MSE loss, reduce per-element [B,] over (seq, channels)
        mask_expanded = mask.unsqueeze(-1).float()
        squared_error = (pred - targets).pow(2)
        masked_loss = squared_error * mask_expanded
        return masked_loss.mean(dim=[-2, -1]) / mask_expanded.mean(dim=[-2, -1]).clamp(min=1e-8)

    @staticmethod
    def _infer_scale_factor(cond_height: int, cond_width: int, target_height: int, target_width: int) -> int:
        if target_height == cond_height and target_width == cond_width:
            return 1
        scale_h = target_height // cond_height if cond_height > 0 else 1
        scale_w = target_width // cond_width if cond_width > 0 else 1
        if scale_h != scale_w:
            raise ValueError(
                f"Non-uniform scale factors between condition and target: height={scale_h}, width={scale_w}. "
                "Condition and target resolutions must scale uniformly."
            )
        return scale_h

    @staticmethod
    def _infer_temporal_scale_factor(cond_num_frames: int, target_num_frames: int) -> int:
        if target_num_frames == cond_num_frames:
            return 1
        target_groups = target_num_frames - 1
        cond_groups = cond_num_frames - 1
        if cond_groups <= 0 or target_groups <= 0:
            return 1
        if target_groups % cond_groups != 0:
            raise ValueError(
                f"Target temporal groups ({target_groups}) is not evenly divisible by "
                f"condition temporal groups ({cond_groups})."
            )
        return target_groups // cond_groups
