#!/usr/bin/env python
"""Stage-2 online autoregressive training for EvoWorld."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
import typer
from accelerate.utils import DistributedType
from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
EVOWORLD_SRC = HERE.parent
sys.path[:0] = [str(EVOWORLD_SRC), str(HERE)]

from ltx_core.text_encoders.gemma import convert_to_additive_mask  # noqa: E402
from ltx_trainer.config import LtxTrainerConfig  # noqa: E402
from ltx_trainer.trainer import LtxvTrainer  # noqa: E402
from cw_dataset import PREFIX_LAT, TARGET_LAT, WINDOW_LAT  # noqa: E402
from online_ar_dataset import OnlineARPrefixDataset  # noqa: E402
from train_stage1 import attach_extra_metrics, load_yaml_strict, log0, patch_ddp_num_blocks  # noqa: E402

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)

DEFAULT_ADAPTER = "default"
DISTILLED_ADAPTER = "distilled_rollout"


def _canonical_lora_state(path: str | Path, strength: float = 1.0) -> dict[str, torch.Tensor]:
    """Load official/Comfy LoRA tensors into PEFT's adapter-neutral key form."""
    if not math.isfinite(strength):
        raise ValueError(f"invalid distilled LoRA strength: {strength}")
    raw = load_file(str(path), device="cpu")
    state: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        key = key.replace("diffusion_model.", "", 1)
        key = key.replace(".lora_A.default.", ".lora_A.")
        key = key.replace(".lora_B.default.", ".lora_B.")
        if key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight"):
            if key.endswith(".lora_B.weight") and strength != 1.0:
                value = value * strength
            state[key] = value
    if not state:
        raise ValueError(f"no LoRA A/B tensors found in {path}")
    return state


def _lora_layout(state: dict[str, torch.Tensor]) -> tuple[list[str], dict[str, int]]:
    """Validate A/B pairs and return exact module names plus their ranks."""
    suffix_a = ".lora_A.weight"
    suffix_b = ".lora_B.weight"
    prefixes = sorted(key[: -len(suffix_a)] for key in state if key.endswith(suffix_a))
    if not prefixes:
        raise ValueError("distilled LoRA contains no A matrices")
    ranks: dict[str, int] = {}
    for prefix in prefixes:
        a_key, b_key = prefix + suffix_a, prefix + suffix_b
        if b_key not in state:
            raise ValueError(f"distilled LoRA is missing B matrix for {prefix}")
        a, b = state[a_key], state[b_key]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(f"invalid LoRA pair {prefix}: A={tuple(a.shape)} B={tuple(b.shape)}")
        ranks[prefix] = int(a.shape[0])
    extra_b = sorted(
        key[: -len(suffix_b)] for key in state if key.endswith(suffix_b)
        and key[: -len(suffix_b)] not in ranks
    )
    if extra_b:
        raise ValueError(f"distilled LoRA has unpaired B matrices: {extra_b[:4]}")
    return prefixes, ranks


class OnlineARTrainer(LtxvTrainer):
    """Generic LTX trainer with a rollout-only frozen distilled adapter."""

    def __init__(
        self,
        cfg: LtxTrainerConfig,
        *,
        distilled_lora: str,
        distilled_strength: float,
        width: int,
        height: int,
        rollout_seed: int,
    ) -> None:
        self._online_distilled_lora = str(Path(distilled_lora).resolve())
        self._online_distilled_strength = float(distilled_strength)
        self._online_width = int(width)
        self._online_height = int(height)
        self._online_rollout_seed = int(rollout_seed)
        self._online_rollout_calls = 0
        self._online_tools = None
        self._online_default_state_keys: set[str] = set()
        super().__init__(cfg)

    def _collect_trainable_params(self) -> None:
        supported = {DistributedType.NO, DistributedType.MULTI_GPU}
        if self._accelerator.distributed_type not in supported:
            raise RuntimeError(
                "online Stage-2 currently requires plain DDP/single-GPU, got "
                f"{self._accelerator.distributed_type.value}: "
                "runtime adapter switching and default-only checkpoint export have not "
                "been validated with FSDP/DeepSpeed-style parameter wrappers"
            )
        super()._collect_trainable_params()
        self._install_frozen_distilled_adapter()
        self._activate_adapters([DEFAULT_ADAPTER], training=True)
        self._trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]
        if not self._trainable_params:
            raise RuntimeError("online Stage-2 has no trainable default-adapter parameters")
        if any(DISTILLED_ADAPTER in name and p.requires_grad
               for name, p in self._transformer.named_parameters()):
            raise RuntimeError("distilled rollout adapter unexpectedly remains trainable")

    def _install_frozen_distilled_adapter(self) -> None:
        state = _canonical_lora_state(
            self._online_distilled_lora,
            strength=self._online_distilled_strength,
        )
        targets, ranks = _lora_layout(state)
        modules = dict(self._transformer.get_base_model().named_modules())
        absent = [name for name in targets if name not in modules]
        if absent:
            raise RuntimeError(
                "distilled LoRA targets are absent from the LTX transformer: "
                + ", ".join(absent[:8])
            )
        default_rank = ranks[targets[0]]
        config = LoraConfig(
            r=default_rank,
            lora_alpha=default_rank,  # alpha/r=1, matching official B@A fusion
            target_modules=targets,
            rank_pattern=ranks,
            alpha_pattern=ranks,
            lora_dropout=0.0,
            init_lora_weights=True,
        )
        self._transformer.add_adapter(DISTILLED_ADAPTER, config)
        base_model = self._transformer.get_base_model()
        incompatible = set_peft_model_state_dict(
            base_model,
            state,
            adapter_name=DISTILLED_ADAPTER,
        )
        unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
        if unexpected:
            raise RuntimeError(f"unexpected distilled LoRA keys: {unexpected[:8]}")

        modules = dict(self._transformer.get_base_model().named_modules())
        actual_targets = {
            name for name, module in modules.items()
            if hasattr(module, "lora_A") and DISTILLED_ADAPTER in module.lora_A
        }
        if actual_targets != set(targets):
            raise RuntimeError(
                "distilled adapter target mismatch: "
                f"missing={sorted(set(targets)-actual_targets)[:8]}, "
                f"extra={sorted(actual_targets-set(targets))[:8]}"
            )
        for name in targets:
            module = modules[name]
            if DISTILLED_ADAPTER not in module.lora_A or DISTILLED_ADAPTER not in module.lora_B:
                raise RuntimeError(f"distilled adapter was not injected into {name}")
            if tuple(module.lora_A[DISTILLED_ADAPTER].weight.shape) != tuple(
                state[name + ".lora_A.weight"].shape
            ):
                raise RuntimeError(f"distilled A shape mismatch at {name}")
            if tuple(module.lora_B[DISTILLED_ADAPTER].weight.shape) != tuple(
                state[name + ".lora_B.weight"].shape
            ):
                raise RuntimeError(f"distilled B shape mismatch at {name}")
            for side, layer in (("A", module.lora_A), ("B", module.lora_B)):
                loaded = layer[DISTILLED_ADAPTER].weight.detach().cpu()
                expected = state[f"{name}.lora_{side}.weight"].to(dtype=loaded.dtype)
                if not torch.equal(loaded, expected):
                    raise RuntimeError(f"distilled {side} values failed exact load audit at {name}")
            if not math.isclose(float(module.scaling[DISTILLED_ADAPTER]), 1.0, rel_tol=0, abs_tol=1e-12):
                raise RuntimeError(
                    f"distilled adapter scaling at {name} is "
                    f"{module.scaling[DISTILLED_ADAPTER]}, expected 1.0"
                )
        log0(
            f"[AR2-online/model] loaded frozen distilled adapter: "
            f"{len(targets)} modules, ranks={sorted(set(ranks.values()))}"
        )

    def _load_lora_checkpoint(self, checkpoint_path: Path) -> None:
        """Load through the parent, then audit the current/default adapter exactly."""
        super()._load_lora_checkpoint(checkpoint_path)
        state = _canonical_lora_state(checkpoint_path)
        targets, _ = _lora_layout(state)
        modules = dict(self._transformer.get_base_model().named_modules())
        actual_targets = {
            name for name, module in modules.items()
            if hasattr(module, "lora_A") and DEFAULT_ADAPTER in module.lora_A
        }
        if actual_targets != set(targets):
            raise RuntimeError(
                "current/default adapter target mismatch after Stage-1/resume load: "
                f"missing={sorted(set(targets)-actual_targets)[:8]}, "
                f"extra={sorted(actual_targets-set(targets))[:8]}"
            )
        for name in targets:
            module = modules[name]
            for side, layer in (("A", module.lora_A), ("B", module.lora_B)):
                loaded = layer[DEFAULT_ADAPTER].weight.detach().cpu()
                expected = state[f"{name}.lora_{side}.weight"].to(dtype=loaded.dtype)
                if tuple(loaded.shape) != tuple(expected.shape) or not torch.equal(loaded, expected):
                    raise RuntimeError(f"current/default {side} failed exact load audit at {name}")
            if not math.isclose(float(module.scaling[DEFAULT_ADAPTER]), 1.0, rel_tol=0, abs_tol=1e-12):
                raise RuntimeError(
                    f"current/default adapter scaling at {name} is "
                    f"{module.scaling[DEFAULT_ADAPTER]}, expected 1.0"
                )
        self._online_default_state_keys = set(state)
        log0(f"[AR2-online/model] exact current-adapter load audit PASS: {len(targets)} modules")

    def _peft_model(self):
        if hasattr(self, "_accelerator"):
            return self._accelerator.unwrap_model(self._transformer)
        return self._transformer

    def _activate_adapters(self, adapters: list[str], *, training: bool) -> None:
        model = self._peft_model()
        unknown = sorted(set(adapters) - set(model.peft_config))
        if unknown:
            raise RuntimeError(f"cannot activate unknown PEFT adapters: {unknown}")

        tuner = getattr(model, "base_model", None)
        if tuner is None or not hasattr(tuner, "set_adapter"):
            raise RuntimeError("PEFT model has no BaseTuner.set_adapter API")
        tuner.set_adapter(adapters if len(adapters) > 1 else adapters[0])
        if set(model.active_adapters) != set(adapters):
            raise RuntimeError(
                f"PEFT adapter activation mismatch: requested={adapters}, "
                f"active={model.active_adapters}"
            )

        activated_counts = {name: 0 for name in adapters}
        for module_name, module in model.named_modules():
            if not hasattr(module, "lora_A"):
                continue
            available = set(module.lora_A.keys())
            expected = set(adapters) & available
            active = getattr(module, "active_adapters", [])
            if isinstance(active, str):
                active = [active]
            observed = set(active) & available
            if observed != expected:
                raise RuntimeError(
                    f"LoRA layer activation mismatch at {module_name}: "
                    f"available={sorted(available)}, requested={adapters}, "
                    f"active={list(active)}"
                )
            for name in expected:
                activated_counts[name] += 1
        missing_active = [name for name, count in activated_counts.items() if count == 0]
        if missing_active:
            raise RuntimeError(f"PEFT adapters active in no LoRA layers: {missing_active}")

        for name, parameter in model.named_parameters():
            if "lora_" not in name:
                continue
            parameter.requires_grad_(f".{DEFAULT_ADAPTER}." in name)
        self._transformer.train(training)

    def _prompt_context(self, conditions: dict[str, torch.Tensor]) -> torch.Tensor:
        if "video_prompt_embeds" in conditions:
            video_features = conditions["video_prompt_embeds"]
            audio_features = conditions.get("audio_prompt_embeds")
        else:
            video_features = conditions["prompt_embeds"]
            audio_features = conditions["prompt_embeds"]
        mask = conditions["prompt_attention_mask"]
        additive_mask = convert_to_additive_mask(mask, video_features.dtype)
        video_context, _, _ = self._embeddings_processor.create_embeddings(
            video_features,
            audio_features,
            additive_mask,
        )
        return video_context

    @torch.inference_mode()
    def _rollout_predecessor(self, batch: dict) -> torch.Tensor:
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
        from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
        from ltx_core.model.transformer import X0Model
        from ltx_pipelines.utils.constants import DISTILLED_SIGMAS
        from ltx_pipelines.utils.denoisers import SimpleDenoiser
        from ltx_pipelines.utils.samplers import _step_state

        seed = batch["online_ar_seed"]["latents"]
        reference = batch["online_ar_reference"]["latents"]
        if seed.shape[0] != 1 or seed.shape[2] != PREFIX_LAT:
            raise RuntimeError(f"online rollout seed must be [1,C,{PREFIX_LAT},H,W], got {tuple(seed.shape)}")
        if reference.shape[0] != 1 or reference.shape[2] != 2:
            raise RuntimeError(f"online entity reference must contain head-2 latents, got {tuple(reference.shape)}")
        if seed.shape[-2:] != (self._online_height // 32, self._online_width // 32):
            raise RuntimeError(
                f"online rollout spatial mismatch: seed={tuple(seed.shape[-2:])}, "
                f"expected={(self._online_height // 32, self._online_width // 32)}"
            )

        if self._online_tools is None:
            pixel_frames = 1 + 8 * (WINDOW_LAT - 1)
            self._online_tools = self._validation_runner._create_video_tools(  # noqa: SLF001
                self._online_width,
                self._online_height,
                pixel_frames,
            )
        tools = self._online_tools
        device = seed.device
        dtype = seed.dtype
        state = tools.create_initial_state(device=device, dtype=dtype)
        state = VideoConditionByReferenceLatent(
            latent=reference,
            downscale_factor=1,
            temporal_scale_factor=1,
            strength=1.0,
        ).apply_to(state, tools)
        state = VideoConditionByLatentIndex(
            latent=seed,
            strength=1.0,
            latent_idx=0,
        ).apply_to(state, tools)

        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        rollout_seed = self._online_rollout_seed + rank * 10_000_000 + self._online_rollout_calls
        self._online_rollout_calls += 1
        generator = torch.Generator(device=device).manual_seed(rollout_seed)
        state = GaussianNoiser(generator=generator)(state, noise_scale=1.0)
        context = self._prompt_context(batch["online_ar_conditions"])

        self._activate_adapters([DEFAULT_ADAPTER, DISTILLED_ADAPTER], training=False)
        try:
            sigmas = DISTILLED_SIGMAS.to(device=device, dtype=torch.float32)
            stepper = EulerDiffusionStep()
            transformer = X0Model(self._transformer)
            denoiser = SimpleDenoiser(v_context=context, a_context=None)
            output = state
            for step_idx in range(len(sigmas) - 1):
                video_result, _ = denoiser(transformer, output, None, sigmas, step_idx)
                output = _step_state(output, video_result.denoised, stepper, sigmas, step_idx)
            generated = tools.unpatchify(tools.clear_conditioning(output)).latent
            prefix = generated[:, :, -PREFIX_LAT:].detach().contiguous()
        finally:
            self._activate_adapters([DEFAULT_ADAPTER], training=True)
        if prefix.requires_grad or prefix.grad_fn is not None:
            raise RuntimeError("online AR prefix must be detached")
        if not torch.isfinite(prefix).all():
            raise RuntimeError("online AR prefix contains NaN/Inf")
        return prefix

    def _training_step(self, batch):
        use_online = batch.get("online_ar_generated")
        if use_online is None:
            raise RuntimeError("online Stage-2 batch lacks online_ar_generated")
        use_online = bool(int(use_online.flatten()[0].item()))
        if "online_ar_seed" not in batch or "online_ar_conditions" not in batch:
            raise RuntimeError("online Stage-2 entry lacks synchronized rollout inputs")

        prefix = self._rollout_predecessor(batch).clone()
        if use_online:
            window = batch["video_latents"]["latents"]
            target_before = window[:, :, PREFIX_LAT:].clone()
            expected_prefix_shape = tuple(window[:, :, :PREFIX_LAT].shape)
            if tuple(prefix.shape) != expected_prefix_shape:
                raise RuntimeError(
                    f"generated prefix shape {tuple(prefix.shape)} != {expected_prefix_shape}"
                )
            if torch.equal(prefix.to(window.dtype), window[:, :, :PREFIX_LAT]):
                raise RuntimeError(
                    "generated online prefix is bit-identical to the GT target prefix; "
                    "refusing a possible teacher-forcing/data-leakage path"
                )
            replacement = window.clone()
            replacement[:, :, :PREFIX_LAT] = prefix.to(window.dtype)
            if not torch.equal(replacement[:, :, PREFIX_LAT:], target_before):
                raise RuntimeError("online prefix replacement modified GT target latents")
            batch["video_latents"] = dict(batch["video_latents"])
            batch["video_latents"]["latents"] = replacement
        self._activate_adapters([DEFAULT_ADAPTER], training=True)
        return super()._training_step(batch)

    def train(self, *args, **kwargs):
        initial_step, _ = self._resume_state
        self._online_rollout_calls = (
            int(initial_step) * self._config.optimization.gradient_accumulation_steps
        )
        return super().train(*args, **kwargs)

    def _save_checkpoint(self):
        self._activate_adapters([DEFAULT_ADAPTER], training=True)
        saved = super()._save_checkpoint()
        if saved is not None:
            raw_keys = set(load_file(str(saved), device="cpu"))
            leaked = [key for key in raw_keys if DISTILLED_ADAPTER in key]
            if leaked:
                raise RuntimeError(f"checkpoint leaked frozen distilled adapter: {leaked[:4]}")
            saved_state = _canonical_lora_state(saved)
            if set(saved_state) != self._online_default_state_keys:
                raise RuntimeError(
                    "saved checkpoint adapter layout changed: "
                    f"missing={sorted(self._online_default_state_keys-set(saved_state))[:8]}, "
                    f"extra={sorted(set(saved_state)-self._online_default_state_keys)[:8]}"
                )
        return saved


@app.command()
def main(
    config_path: str = typer.Argument(...),
    stage1_lora: str = typer.Option(...),
    distilled_lora: str = typer.Option(...),
    distilled_strength: float = typer.Option(1.0),
    work: str = typer.Option(lambda: os.environ.get("EW_WORK", "")),
    latents: str = typer.Option(lambda: os.environ.get("EW_LATENTS", "")),
    conditions: str = typer.Option(lambda: os.environ.get("EW_CONDITIONS", "")),
    reference: str = typer.Option(lambda: os.environ.get("EW_REFERENCE", "")),
    max_samples: int = typer.Option(2048),
    gt_mix_ratio: float = typer.Option(0.25),
    rollout_seed: int = typer.Option(42000),
    width: int = typer.Option(896),
    height: int = typer.Option(512),
    preflight_only: bool = typer.Option(False, "--preflight-only"),
    runtime_probe_only: bool = typer.Option(False, "--runtime-probe-only"),
    disable_progress_bars: bool = typer.Option(False, "--disable-progress-bars"),
) -> None:
    for name, value in (("work", work), ("latents", latents), ("conditions", conditions),
                        ("reference", reference)):
        if not value:
            raise typer.BadParameter(f"--{name} is empty")
    for path in (stage1_lora, distilled_lora):
        if not Path(path).is_file():
            raise typer.BadParameter(f"missing checkpoint: {path}")

    raw = load_yaml_strict(Path(config_path))
    cfg = LtxTrainerConfig(**raw)
    out = Path(cfg.output_dir)
    if "_ar2online_" not in out.name:
        raise typer.BadParameter(f"online Stage-2 output_dir must contain '_ar2online_': {out}")
    if cfg.optimization.steps <= 0 or cfg.checkpoints.interval <= 0:
        raise typer.BadParameter(
            "online Stage-2 requires positive steps/checkpoint interval, got "
            f"{cfg.optimization.steps}/{cfg.checkpoints.interval}"
        )
    if cfg.optimization.steps % cfg.checkpoints.interval != 0:
        raise typer.BadParameter(
            "online Stage-2 requires steps divisible by checkpoint interval so the final "
            f"checkpoint is saved, got {cfg.optimization.steps}/{cfg.checkpoints.interval}"
        )
    if cfg.model.training_mode != "lora" or not cfg.model.load_checkpoint:
        raise typer.BadParameter("online Stage-2 requires LoRA mode and a load checkpoint")
    load_checkpoint = Path(cfg.model.load_checkpoint).resolve()
    stage1_checkpoint = Path(stage1_lora).resolve()
    if cfg.checkpoints.no_resume:
        if load_checkpoint != stage1_checkpoint:
            raise typer.BadParameter("fresh online Stage-2 must initialize from Stage-1 LoRA")
    elif load_checkpoint.parent != (out / "checkpoints").resolve():
        raise typer.BadParameter("online Stage-2 resume must use its own output/checkpoints")
    if cfg.optimization.batch_size != 1:
        raise typer.BadParameter("online Stage-2 requires batch_size=1")
    if cfg.acceleration.quantization is not None or cfg.training_strategy.audio is not None:
        raise typer.BadParameter("online Stage-2 requires video-only bf16/no-quantization training")
    if int(os.environ.get("EW_USE_IAA", "0") or 0) != 1:
        raise typer.BadParameter("online Stage-2 requires EW_USE_IAA=1")
    if int(os.environ.get("EW_IAA_ENTITY_REFS", "2") or 2) != 2:
        raise typer.BadParameter("online Stage-2 requires head-2 entity references")
    beta = float(os.environ.get("EW_IAA_BETA", "3.0") or 3.0)
    if not math.isfinite(beta) or beta < 0:
        raise typer.BadParameter(f"invalid EW_IAA_BETA={beta}")

    lora_cfg = raw.get("lora") or {}
    if int(lora_cfg.get("rank", -1)) != 128 or int(lora_cfg.get("alpha", -2)) != 128:
        raise typer.BadParameter(
            f"online Stage-2 requires rank=alpha=128, got "
            f"{lora_cfg.get('rank')}/{lora_cfg.get('alpha')}"
        )
    expected_targets = {
        "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
        "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
    }
    actual_targets = set(lora_cfg.get("target_modules") or [])
    if actual_targets != expected_targets:
        raise typer.BadParameter(
            f"online Stage-2 LoRA target mismatch: "
            f"missing={sorted(expected_targets-actual_targets)}, "
            f"extra={sorted(actual_targets-expected_targets)}"
        )

    prefix_cfgs = [c for c in cfg.training_strategy.video.conditions
                   if getattr(c, "type", None) == "prefix"]
    ref_cfgs = [c for c in cfg.training_strategy.video.conditions
                if getattr(c, "type", None) == "reference"]
    if len(prefix_cfgs) != 1 or prefix_cfgs[0].temporal_boundary != PREFIX_LAT:
        raise typer.BadParameter("online Stage-2 requires exactly one four-latent prefix")
    if float(prefix_cfgs[0].probability) != 1.0:
        raise typer.BadParameter("online Stage-2 temporal prefix probability must be 1.0")
    if len(ref_cfgs) != 1 or ref_cfgs[0].latents_dir != Path(reference).name:
        raise typer.BadParameter("online Stage-2 requires exactly one declared entity reference")
    if float(ref_cfgs[0].probability) != 1.0:
        raise typer.BadParameter("online Stage-2 entity reference probability must be 1.0")

    ds = OnlineARPrefixDataset(
        max_samples=max_samples,
        gt_mix_ratio=gt_mix_ratio,
        selection_seed=42,
        work=work,
        latents_dir=latents,
        conditions_dir=conditions,
        reference_dir=reference,
        split="train",
        verbose=True,
    )
    online_probe = ds[0]
    if not online_probe["meta"]["is_revisit"] or not online_probe["online_ar_generated"]:
        raise typer.BadParameter("first online Stage-2 probe is not a generated-prefix revisit")
    target = online_probe["video_latents"]["latents"]
    seed = online_probe["online_ar_seed"]["latents"]
    ref = online_probe["online_ar_reference"]["latents"]
    if target.shape[1] != WINDOW_LAT or seed.shape[1] != PREFIX_LAT or ref.shape[1] != 2:
        raise typer.BadParameter(
            f"layout mismatch: target={tuple(target.shape)}, seed={tuple(seed.shape)}, ref={tuple(ref.shape)}"
        )
    if online_probe["online_ar_meta"]["seg"] != online_probe["online_ar_meta"]["target_seg"] - 1:
        raise typer.BadParameter("predecessor prompt is not from target_seg-1")

    audit_positions = sorted(set(
        round(i * (len(ds.entries) - 1) / 7)
        for i in range(min(8, len(ds.entries)))
    ))
    for pos in audit_positions:
        target_i, rollout_target_i, predecessor_i, use_online = ds.entries[pos]
        staged = ds[pos]
        original = ds.base[target_i]
        predecessor = ds.base[predecessor_i]
        rollout_target = ds.base[rollout_target_i]
        if not torch.equal(staged["video_latents"]["latents"], original["video_latents"]["latents"]):
            raise typer.BadParameter("online dataset modified GT FM window before rollout")
        if not torch.equal(
            staged[Path(reference).name]["latents"], original[Path(reference).name]["latents"]
        ):
            raise typer.BadParameter("online dataset modified the FM target entity reference")
        for key, value in original["conditions"].items():
            if isinstance(value, torch.Tensor) and not torch.equal(staged["conditions"][key], value):
                raise typer.BadParameter(f"online dataset modified target prompt tensor {key}")
        expected_seed = predecessor["video_latents"]["latents"][:, :PREFIX_LAT]
        if not torch.equal(staged["online_ar_seed"]["latents"], expected_seed):
            raise typer.BadParameter("rollout seed is not predecessor head-4 latents")
        for key, value in predecessor["conditions"].items():
            if isinstance(value, torch.Tensor) and not torch.equal(
                staged["online_ar_conditions"][key], value
            ):
                raise typer.BadParameter(f"rollout prompt is not predecessor prompt tensor {key}")
        if not torch.equal(
            staged["online_ar_reference"]["latents"],
            rollout_target[Path(reference).name]["latents"],
        ):
            raise typer.BadParameter("rollout entity reference belongs to the wrong clip")
        if use_online and target_i != rollout_target_i:
            raise typer.BadParameter("online row rolls out a predecessor from another target")

    state = _canonical_lora_state(distilled_lora, distilled_strength)
    targets, ranks = _lora_layout(state)
    log0(
        f"[AR2-online/preflight] samples={len(ds):,}, online={ds.num_online:,}, GT={ds.num_gt:,}, "
        f"layout=[entity 2 | generated prefix {PREFIX_LAT} | GT target {TARGET_LAT}], "
        f"distilled_modules={len(targets)}, distilled_ranks={sorted(set(ranks.values()))}"
    )
    if preflight_only:
        log0("[AR2-online/preflight] PASS — metadata/tensors/LoRA layout verified; no model loaded")
        return

    patch_ddp_num_blocks()
    trainer = OnlineARTrainer(
        cfg,
        distilled_lora=distilled_lora,
        distilled_strength=distilled_strength,
        width=width,
        height=height,
        rollout_seed=rollout_seed,
    )
    if runtime_probe_only:
        ds.entries = ds.entries[:ds.num_online]
    trainer._dataset = ds  # noqa: SLF001
    effective_batch = cfg.optimization.gradient_accumulation_steps * int(os.environ.get("WORLD_SIZE", "1"))
    attach_extra_metrics(trainer, len(ds), effective_batch)
    if runtime_probe_only:
        trainer._init_dataloader()  # noqa: SLF001
        trainer._init_timestep_sampler()  # noqa: SLF001
        batch = next(iter(trainer._dataloader))  # noqa: SLF001
        output = trainer._training_step(batch)  # noqa: SLF001
        loss = output.loss.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"online runtime probe produced non-finite loss: {loss}")
        trainer._accelerator.backward(loss)  # noqa: SLF001
        model = trainer._peft_model()  # noqa: SLF001
        current_grads = []
        distilled_grads = []
        for name, parameter in model.named_parameters():
            if f".{DEFAULT_ADAPTER}." in name and parameter.grad is not None:
                current_grads.append(parameter.grad)
            if f".{DISTILLED_ADAPTER}." in name and parameter.grad is not None:
                distilled_grads.append(parameter.grad)
        if not current_grads:
            raise RuntimeError("online runtime probe found no gradients on current/default adapter")
        if any(not torch.isfinite(grad).all() for grad in current_grads):
            raise RuntimeError("online runtime probe found NaN/Inf current-adapter gradients")
        if not any(torch.count_nonzero(grad).item() > 0 for grad in current_grads):
            raise RuntimeError("online runtime probe found only zero current-adapter gradients")
        if distilled_grads:
            raise RuntimeError("frozen distilled rollout adapter received gradients")
        trainer._accelerator.wait_for_everyone()  # noqa: SLF001
        log0(
            f"[AR2-online/runtime-probe] PASS — fresh distilled8 rollout + detached prefix + "
            f"revisit IAA + GT FM backward; loss={float(loss.detach()):.6f}"
        )
        trainer._accelerator.end_training()  # noqa: SLF001
        return
    trainer.train(disable_progress_bars=disable_progress_bars)


if __name__ == "__main__":
    app()
