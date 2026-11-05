"""Shared utilities for EvoWorld inference."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

import torch

PREFIX_LAT = 4
TARGET_LAT = 11
WINDOW_LAT = PREFIX_LAT + TARGET_LAT


def resolve_pt(root: str | Path, clip: str) -> Path:
    """Resolve a cached tensor while supporting an optional videos/ level."""
    root = Path(root)
    for candidate in (root / f"{clip}.pt", root / "videos" / f"{clip}.pt"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No tensor cache found for {clip!r} under {root}")


def segment_prompts(work: str | Path, clip: str) -> list[str]:
    """Load segment prompts for one clip from captions.jsonl."""
    with (Path(work) / "captions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("clip_id") == clip and row.get("segment_prompts"):
                return row["segment_prompts"]
    raise KeyError(f"No segment prompts found for {clip!r}")


def _load_lora_config(yaml_path: str | Path):
    import yaml
    from peft import LoraConfig

    with Path(yaml_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)["lora"]
    return LoraConfig(
        r=config["rank"],
        lora_alpha=config["alpha"],
        target_modules=list(config["target_modules"]),
        lora_dropout=config.get("dropout", 0.0),
        init_lora_weights=True,
    )


def load_transformer_with_lora(ckpt, lora_ckpt, lora_yaml, device):
    """Load the base transformer and an optional PEFT LoRA."""
    from ltx_trainer.model_loader import load_transformer

    transformer = load_transformer(
        checkpoint_path=ckpt,
        device="cpu",
        dtype=torch.bfloat16,
    )
    if lora_ckpt:
        from peft import get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file

        transformer = get_peft_model(transformer, _load_lora_config(lora_yaml))
        state = {
            key.replace("diffusion_model.", "", 1): value
            for key, value in load_file(lora_ckpt).items()
        }
        set_peft_model_state_dict(transformer.get_base_model(), state)
    transformer = transformer.to(device).eval()
    transformer.requires_grad_(False)
    return transformer


def load_transformer_with_fused_loras(
    ckpt: str | Path,
    loras: Iterable[tuple[str | Path, float]],
    device: torch.device,
) -> torch.nn.Module:
    """Load the base transformer and fuse Comfy-format LoRA adapters."""
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator

    resolved = []
    for path, strength in loras:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"LoRA not found: {path}")
        resolved.append(
            LoraPathStrengthAndSDOps(
                path=str(path),
                strength=float(strength),
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        )
    if not resolved:
        raise ValueError("At least one LoRA adapter is required")

    transformer = SingleGPUModelBuilder(
        model_path=str(ckpt),
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        loras=tuple(resolved),
    ).build(device=device, dtype=torch.bfloat16)
    transformer = transformer.eval()
    transformer.requires_grad_(False)
    return transformer


class SamplingContext:
    """Minimal sampling context required by the LTX denoising runner."""

    def advance_step(self, *_args, **_kwargs) -> None:
        return None


def save_mp4(pixels: torch.Tensor, path: Path, fps: float = 24.0) -> None:
    """Encode a [C,F,H,W] RGB tensor in [0,1] as H.264 MP4."""
    array = (pixels.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8)
    array = array.permute(1, 2, 3, 0).contiguous().numpy()
    frames, height, width, channels = array.shape
    if channels != 3 or frames < 1:
        raise ValueError(f"Expected non-empty RGB video, got {array.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(int(fps)), "-i", "-", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    process.stdin.write(array.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed to encode {path}")
