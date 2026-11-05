#!/usr/bin/env python
"""Stage-1 training entry point for EvoWorld."""
import os
import sys
import math
from pathlib import Path

import typer
import yaml

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.trainer import LtxvTrainer

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


IS_RANK0 = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0"


def log0(*a, **k):
    if IS_RANK0:
        print(*a, **k)


def load_yaml_strict(path: Path) -> dict:
    text = path.read_text()
    root = yaml.compose(text)

    def walk(node) -> None:
        if isinstance(node, yaml.MappingNode):
            seen: dict[str, object] = {}
            for key_node, value_node in node.value:
                key = str(getattr(key_node, "value", key_node))
                if key in seen:
                    line = key_node.start_mark.line + 1
                    raise typer.BadParameter(f"duplicate YAML key {key!r} at {path}:{line}")
                seen[key] = key_node
                walk(value_node)
        elif isinstance(node, yaml.SequenceNode):
            for item in node.value:
                walk(item)

    if root is not None:
        walk(root)
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise typer.BadParameter(f"training config root must be a mapping: {path}")
    return loaded


def patch_ddp_num_blocks() -> None:
    try:
        from torch.nn.parallel import DistributedDataParallel as DDP
    except Exception:
        return
    if not hasattr(DDP, "num_blocks"):
        DDP.num_blocks = property(lambda self: self.module.num_blocks)


def patch_dataloader_collate(target_ds, collate_fn) -> bool:
    try:
        import ltx_trainer.trainer as T
    except Exception:
        return False
    orig = T.DataLoader

    class _DL(orig):
        def __init__(self, dataset, *a, **k):
            if dataset is target_ds and k.get("collate_fn") is None:
                k["collate_fn"] = collate_fn
            super().__init__(dataset, *a, **k)

    T.DataLoader = _DL
    return True


_GRAD = {}


def attach_extra_metrics(trainer, n_samples: int, eff_batch: int):
    acc = trainer._accelerator                     # noqa: SLF001
    orig_clip = acc.clip_grad_norm_

    def clip(*a, **k):
        n = orig_clip(*a, **k)
        try:
            _GRAD["v"] = float(n)
        except (TypeError, ValueError):
            pass
        return n

    acc.clip_grad_norm_ = clip
    orig_log = trainer._log_metrics                # noqa: SLF001
    state = {"n": 0, "warned": False}

    def log(metrics):
        if "train/loss" in metrics:
            state["n"] += 1
            if "v" in _GRAD:
                metrics["train/grad_norm"] = _GRAD["v"]
            elif state["n"] > 20 and not state["warned"]:
                state["warned"] = True
                log0("Warning: grad_norm was not observed for 20 optimizer steps.", flush=True)
            step = metrics.get("train/global_step", 0)


            if n_samples:
                metrics["train/epoch"] = step * eff_batch / n_samples
        orig_log(metrics)

    trainer._log_metrics = log                     # noqa: SLF001


@app.command()
def main(
    config_path: str = typer.Argument(..., help="YAML configuration path"),


    work: str = typer.Option(lambda: os.environ.get("EW_WORK", ""), help="Training metadata directory"),
    latents: str = typer.Option(lambda: os.environ.get("EW_LATENTS", ""), help="Full-video latent directory"),
    conditions: str = typer.Option(lambda: os.environ.get("EW_CONDITIONS", ""), help="Text embedding directory"),
    reference: str = typer.Option(lambda: os.environ.get("EW_REFERENCE", ""), help="Reference latent directory"),
    scene_reference: str = typer.Option(lambda: os.environ.get("EW_SCENE_REFERENCE", ""),
                                        help="Optional scene-reference latent directory"),
    split: str = typer.Option("train"),
    cw_dataset_path: str = typer.Option("", help="Directory containing cw_dataset.py"),
    disable_progress_bars: bool = typer.Option(False, "--disable-progress-bars"),
) -> None:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise typer.Exit(f"Configuration not found: {cfg_path}")
    for name, val in (("--work", work), ("--latents", latents),
                      ("--conditions", conditions), ("--reference", reference)):
        if not val:
            raise typer.Exit(f"{name} is required; set the corresponding EW_* variable or CLI option.")
    raw_config = load_yaml_strict(cfg_path)
    config = LtxTrainerConfig(**raw_config)

    sys.path.insert(0, cw_dataset_path or str(Path(__file__).resolve().parent))
    from cw_dataset import (
        OoSWindowDataset, PREFIX_LAT, TARGET_LAT, REF_LAT, WINDOW_LAT,
        TEMPORAL_FACTOR, collate,
    )

    if config.validation.interval and config.validation.samples:
        validation_frames = int(config.validation.video_dims[2])
        validation_latents = (validation_frames - 1) // TEMPORAL_FACTOR + 1
        if validation_latents != WINDOW_LAT:
            raise typer.Exit(
                f"validation length mismatch: {validation_frames} frames -> {validation_latents} latents, "
                f"but training window is {WINDOW_LAT} ({PREFIX_LAT} prefix + {TARGET_LAT} target). "
                f"Use {1 + TEMPORAL_FACTOR * (WINDOW_LAT - 1)} frames.")

    iaa_on = int(os.environ.get("EW_USE_IAA", "0") or 0) != 0
    i2v_on = int(os.environ.get("EW_I2V_START", "0") or 0) != 0
    if iaa_on:
        beta = float(os.environ.get("EW_IAA_BETA", "3.0") or 3.0)
        entity_refs = int(os.environ.get("EW_IAA_ENTITY_REFS", "2") or 2)
        expected_refs = 1 if i2v_on else REF_LAT
        if not math.isfinite(beta) or beta < 0:
            raise typer.Exit(f"EW_IAA_BETA must be finite and non-negative, got {beta}")
        if entity_refs != expected_refs:
            raise typer.Exit(
                f"IAA reference count mismatch: {'i2v' if i2v_on else 'v2v'} requires "
                f"EW_IAA_ENTITY_REFS={expected_refs}, got {entity_refs}")
        out_dir = str(raw_config.get("output_dir", ""))
        if "_iaafix" not in Path(out_dir).name:
            raise typer.Exit(
                f"refusing IAA run without _iaafix output tag: {out_dir}. "
                "Regenerate configs with `bash run.sh configs`.")
        if i2v_on != ("_i2v" in Path(out_dir).name):
            raise typer.Exit(
                f"I2V mode/output mismatch: EW_I2V_START={int(i2v_on)}, output_dir={out_dir}")
        lora_cfg = raw_config.get("lora") or {}
        if int(lora_cfg.get("alpha", -1)) != int(lora_cfg.get("rank", -2)):
            raise typer.Exit(
                f"IAA LoRA requires alpha=rank for scale 1.0, got "
                f"rank={lora_cfg.get('rank')} alpha={lora_cfg.get('alpha')}")
        expected_targets = {
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
        }
        actual_targets = set(lora_cfg.get("target_modules") or [])
        if actual_targets != expected_targets:
            raise typer.Exit(
                f"IAA LoRA target_modules mismatch: missing={sorted(expected_targets-actual_targets)}, "
                f"extra={sorted(actual_targets-expected_targets)}")
        if config.training_strategy.audio is not None:
            raise typer.Exit("OoS/IAA is video-only, but the config enables an audio training branch")


    prefix_cfgs = [c for c in config.training_strategy.video.conditions
                   if getattr(c, "type", None) == "prefix"]
    if len(prefix_cfgs) != 1:
        raise typer.Exit(f"OoS training requires exactly one prefix condition, got {len(prefix_cfgs)}")
    if float(prefix_cfgs[0].probability) != 1.0:
        raise typer.Exit(
            f"OoS prefix probability must be 1.0, got {prefix_cfgs[0].probability}; "
            "otherwise clean-prefix and target loss roles vary silently.")
    tb = prefix_cfgs[0].temporal_boundary
    if tb != PREFIX_LAT:
        raise typer.Exit(
            f"Prefix mismatch: temporal_boundary={tb}, PREFIX_LAT={PREFIX_LAT}.")


    ref_cfgs = [c for c in config.training_strategy.video.conditions
                if getattr(c, "type", None) == "reference"]
    if iaa_on:
        if len(ref_cfgs) != 1:
            raise typer.Exit(f"IAA requires exactly one entity reference condition, got {len(ref_cfgs)}")
        if float(ref_cfgs[0].probability) != 1.0:
            raise typer.Exit(
                f"IAA reference probability must be 1.0, got {ref_cfgs[0].probability}; "
                "otherwise some batches have no entity tokens while the mask assumes they do.")
    yaml_ref_dirs = [c.latents_dir for c in ref_cfgs]
    cli_ref_dirs = [Path(reference).name] + ([Path(scene_reference).name] if scene_reference else [])

    if sorted(yaml_ref_dirs) != sorted(cli_ref_dirs):
        raise typer.Exit(
            f"Reference mismatch: config={yaml_ref_dirs}, CLI directories={cli_ref_dirs}.")

    ds = OoSWindowDataset(work=work, latents_dir=latents, conditions_dir=conditions,
                          reference_dir=reference,
                          scene_reference_dir=(scene_reference or None),
                          split=split,
                          oversample_revisit=int(os.environ.get("EW_REVISIT_OVERSAMPLE", "1")),
                          prefix_noise=float(os.environ.get("EW_PREFIX_NOISE", "0")),
                          oversample_i2v_start=int(os.environ.get("EW_I2V_OVERSAMPLE", "1")),
                          i2v_start=(int(os.environ.get("EW_I2V_START", "0") or 0) != 0))
    if float(os.environ.get("EW_PREFIX_NOISE", "0")) > 0:
        log0(f"[oos] prefix_noise={os.environ['EW_PREFIX_NOISE']}")
    if int(os.environ.get("EW_I2V_START", "0") or 0) != 0:
        log0(f"[oos] I2V-start enabled; oversample={os.environ.get('EW_I2V_OVERSAMPLE','1')}")
    idx = ds.index
    log0(f"[oos] split={split} samples={len(ds):,} revisit={sum(1 for s in idx.samples if s['is_revisit']):,}"
          f" prefix={PREFIX_LAT} target={TARGET_LAT} window={idx.window_lat}")
    if idx.dropped:
        log0(f"[oos] dropped={dict(sorted(idx.dropped.items(), key=lambda kv: -kv[1]))}")
    if idx.n_seg_s_fallback:
        log0(f"[oos] seg_s fallback rows={idx.n_seg_s_fallback:,}")
    if len(ds) == 0:
        raise typer.Exit("Dataset is empty; check metadata paths and split values.")
    if iaa_on:
        n_revisit = sum(1 for s in idx.samples if s["is_revisit"])
        if n_revisit == 0:
            raise typer.Exit("IAA dataset contains zero revisit windows; the attention branch would never train")
        probe = ds[0]
        ref_key = Path(reference).name
        actual_ref_latents = int(probe[ref_key]["latents"].shape[1])
        expected_ref_latents = int(os.environ["EW_IAA_ENTITY_REFS"])
        actual_window_latents = int(probe["video_latents"]["latents"].shape[1])
        video_tensor = probe["video_latents"]["latents"]
        meta_h = int(probe["video_latents"]["height"])
        meta_w = int(probe["video_latents"]["width"])
        tensor_h, tensor_w = map(int, video_tensor.shape[-2:])
        if (meta_h, meta_w) != (tensor_h, tensor_w) or meta_h <= 0 or meta_w <= 0:
            raise typer.Exit(
                f"latent spatial metadata mismatch: metadata={meta_h}x{meta_w}, "
                f"tensor={tensor_h}x{tensor_w}. IAA token ranges would be wrong.")
        if actual_ref_latents != expected_ref_latents:
            raise typer.Exit(
                f"actual reference latent length={actual_ref_latents}, but IAA expects "
                f"{expected_ref_latents} ({ref_key})")
        if actual_window_latents != WINDOW_LAT:
            raise typer.Exit(
                f"actual target window length={actual_window_latents}, expected WINDOW_LAT={WINDOW_LAT}")
        log0(
            f"[IAA/preflight] beta={beta} mode={'i2v' if i2v_on else 'v2v'} "
            f"entity_ref={actual_ref_latents} window={actual_window_latents} grid={meta_h}x{meta_w} "
            f"revisit={n_revisit:,} output={Path(out_dir).name} ✓")


    want = set(config.training_strategy.get_data_sources().values())
    got = set(ds[0].keys()) - {"meta", "idx"}
    if not want.issubset(got):
        raise typer.Exit(
            f"Batch keys mismatch: required={sorted(want)}, got={sorted(got)}, missing={sorted(want - got)}.")
    log0(f"[oos] batch_keys={sorted(got)}")


    patch_ddp_num_blocks()


    if config.optimization.batch_size > 1:
        if not patch_dataloader_collate(ds, collate):
            raise typer.Exit("Failed to install the collate function required for batch_size > 1.")
        log0(f"[oos] using custom collate for batch_size={config.optimization.batch_size}")

    trainer = LtxvTrainer(config)
    trainer._dataset = ds

    if config.wandb.enabled and not os.environ.get("WANDB_API_KEY"):
        raise typer.Exit("wandb.enabled=true requires WANDB_API_KEY.")
    eff = (config.optimization.batch_size
           * config.optimization.gradient_accumulation_steps
           * int(os.environ.get("WORLD_SIZE", "1")))
    attach_extra_metrics(trainer, len(ds), eff)
    log0(f"[oos] wandb={'on' if config.wandb.enabled else 'off'}"
          f" effective_batch={eff} steps_per_epoch≈{len(ds)/max(eff,1):.0f}"
          f" requested_epochs≈{config.optimization.steps*eff/max(len(ds),1):.2f}")
    trainer.train(disable_progress_bars=disable_progress_bars)


if __name__ == "__main__":
    app()
