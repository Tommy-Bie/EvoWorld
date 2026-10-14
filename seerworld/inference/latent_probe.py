"""最小验证版：latent 空间单窗口探针，隔离 reference 机制。

复刻【一个训练窗口】的生成（prefix 4 + target 11 = 15 latent），用 ground truth prefix +
reference latent + 训练时的 ValidationRunner 机制，只生成 revisit 那一段。去掉自回归交接和
exposure bias，纯粹回答：训练机制的 reference 在推理端能不能正确工作（不"人变熊"、帮身份保持）。

一次跑出三个对照（同 clip 同 seed 同 prefix）：
    probe_gt.mp4     真值窗口直接 decode（参照，无模型）
    probe_ref.mp4    lora + reference（我们方法的正确形态）
    probe_noref.mp4  lora，无 reference（隔离 reference 净作用）

用法（GPU 环境）：
    CW_GPU=0 python latent_probe.py --clip seedance2_10s/09873 \
        --lora <lora.safetensors> --lora-yaml <oos_lora_r128.yaml> --out <dir>

需要在具备 GPU、权重和依赖的环境中运行。
"""
from __future__ import annotations
import argparse, os, sys
from collections.abc import Iterable
from pathlib import Path

import torch


def resolve_pt(root: str, clip: str) -> Path:
    """latents/ 和 reference_latents/ 可能多一层 videos/（process_videos 镜像源路径）。"""
    for sub in ("", "videos"):
        p = (Path(root) / sub / f"{clip}.pt") if sub else (Path(root) / f"{clip}.pt")
        if p.is_file():
            return p
    raise FileNotFoundError(f"{root} 和 {root}/videos 下都没有 {clip}.pt")


def load_lora_cfg(yaml_path: str):
    import yaml
    from peft import LoraConfig
    l = yaml.safe_load(open(yaml_path))["lora"]
    return LoraConfig(r=l["rank"], lora_alpha=l["alpha"],
                      target_modules=list(l["target_modules"]),
                      lora_dropout=l.get("dropout", 0.0), init_lora_weights=True)


def load_transformer_with_lora(ckpt, lora_ckpt, lora_yaml, device):
    from ltx_trainer.model_loader import load_transformer
    t = load_transformer(checkpoint_path=ckpt, device="cpu", dtype=torch.bfloat16)
    if lora_ckpt:
        from peft import get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file
        t = get_peft_model(t, load_lora_cfg(lora_yaml))
        # 训练存的是 ComfyUI 格式（trainer.py:960-966 加了 "diffusion_model." 前缀），
        # resume 时脱掉（trainer.py:517）。照抄。
        sd = {k.replace("diffusion_model.", "", 1): v for k, v in load_file(lora_ckpt).items()}
        set_peft_model_state_dict(t.get_base_model(), sd)
    t = t.to(device).eval()
    t.requires_grad_(False)
    return t


def load_transformer_with_fused_loras(
    ckpt: str | Path,
    loras: Iterable[tuple[str | Path, float]],
    device: torch.device,
) -> torch.nn.Module:
    """Load the dev transformer and fuse multiple Comfy-format LoRAs officially.

    ``loras`` is an iterable of ``(path, strength)`` pairs.  This deliberately
    uses ltx-core's native multi-LoRA builder instead of stacking PEFT adapters:
    the released distilled LoRA and trainer checkpoints both use the official
    Comfy key layout (including the optional ``diffusion_model.`` prefix).
    """
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator

    resolved = []
    for path, strength in loras:
        path = str(Path(path).expanduser().resolve())
        if not Path(path).is_file():
            raise FileNotFoundError(f"LoRA 不存在: {path}")
        resolved.append(
            LoraPathStrengthAndSDOps(
                path=path,
                strength=float(strength),
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        )
    if not resolved:
        raise ValueError("load_transformer_with_fused_loras 至少需要一个 LoRA")

    print(
        "[model] official multi-LoRA fusion: "
        + " + ".join(f"{Path(item.path).name}@{item.strength:g}" for item in resolved),
        flush=True,
    )
    t = SingleGPUModelBuilder(
        model_path=str(ckpt),
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        loras=tuple(resolved),
    ).build(device=device, dtype=torch.bfloat16)
    t = t.eval()
    t.requires_grad_(False)
    return t


class _StubCtx:
    """sampling_ctx 只被 _run_denoising 用来 advance_step()（validation_runner.py:980）。"""
    def advance_step(self, *a, **k):
        pass


def revisit_prompt(work: str, clip: str) -> str:
    import json
    for line in open(Path(work) / "captions.jsonl", encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("clip_id") == clip and r.get("segment_prompts"):
            return r["segment_prompts"][-1]      # 最后一段 = revisit
    raise KeyError(f"captions.jsonl 里找不到 {clip} 的 caption")


def save_mp4(pixels: torch.Tensor, path: Path, fps: float = 24.0):
    """pixels [C,F,H,W] in [0,1] → mp4。用 ffmpeg（torchvision 这个 build 没带 write_video）。"""
    import subprocess
    arr = (pixels.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8)
    arr = arr.permute(1, 2, 3, 0).contiguous().numpy()          # [F,H,W,C] uint8
    F, H, W, C = arr.shape
    assert C == 3, f"期望 RGB，得到 C={C}"
    path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(int(fps)), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE)
    p.stdin.write(arr.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg 编码失败: {path}")


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--lora", default="", help="lora .safetensors；空=base")
    ap.add_argument("--lora-yaml", default="", help="取 LoraConfig（rank/alpha/target_modules）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=896)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--window", type=int, default=15, help="prefix4 + target11")
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()

    if a.lora and not a.lora_yaml:
        return int(bool(print("!! 给了 --lora 就要给 --lora-yaml（取 LoraConfig）")))
    E = os.environ
    for k in ("CW_CKPT", "CW_GEMMA", "CW_VIDEO_VAE", "CW_LATENTS", "CW_REFERENCE", "CW_WORK"):
        if not E.get(k):
            return int(bool(print(f"!! 环境变量 {k} 未设置（请先 source .env）")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # 帧数：window 个 latent → (window-1)*8+1 像素帧。15 → 113。
    nframes = (a.window - 1) * 8 + 1
    assert (nframes - 1) % 8 == 0, nframes
    print(f"[probe] {a.clip}  window={a.window} latent = {nframes} 帧  {a.width}x{a.height}")

    # ---- 现成 latent ----
    full = torch.load(resolve_pt(E["CW_LATENTS"], a.clip), map_location="cpu",
                      weights_only=True)["latents"]           # [128, F, 16, 28]
    if full.dim() == 5:
        full = full[0]
    if full.shape[1] < a.window:
        return int(bool(print(f"!! 整片只有 {full.shape[1]} 个 latent，不足 window {a.window}")))
    win_gt = full[:, -a.window:].unsqueeze(0).to(device)      # [1,128,15,16,28] 真值窗口
    prefix = full[:, -a.window:-(a.window - 4)].unsqueeze(0).to(device)  # 前 4 = ground truth prefix
    ref = torch.load(resolve_pt(E["CW_REFERENCE"], a.clip), map_location="cpu",
                     weights_only=True)["latents"]
    if ref.dim() == 4:
        ref = ref.unsqueeze(0)
    ref = ref.to(device)
    prompt = revisit_prompt(E["CW_WORK"], a.clip)
    print(f"[probe] revisit prompt: {prompt[:100]}")

    # ---- ValidationRunner（空 conditions → 媒体编码 no-op；只编码 prompt + 装 decoder）----
    from ltx_trainer.config import ValidationConfig, ValidationSample
    from ltx_trainer.validation_runner import ValidationRunner
    cfg = ValidationConfig(samples=[ValidationSample(prompt=prompt)],
                           video_dims=(a.width, a.height, nframes),
                           generate_video=True, generate_audio=False,
                           inference_steps=a.steps, seed=a.seed)
    runner = ValidationRunner(config=cfg, model_path=E["CW_CKPT"],
                              text_encoder_path=E["CW_GEMMA"], video_vae_path=E["CW_VIDEO_VAE"])
    if getattr(runner, "_vae_decoder", None) is not None:
        runner._vae_decoder = runner._vae_decoder.to(device)   # runner 默认放 CPU

    def decode(latent_state_or_tensor, tools=None):
        if tools is not None:
            st = tools.clear_conditioning(latent_state_or_tensor)
            st = tools.unpatchify(st)
        else:
            st = latent_state_or_tensor
        return runner._decode_video(st, device)                 # [C,F,H,W] in [0,1] cpu

    # ---- 1) 真值窗口直接 decode ----
    tools0 = runner._create_video_tools(a.width, a.height, nframes)
    # create_initial_state(initial_latent=...) 要求形状 == target_shape（15 latent），win_gt 正好。
    st_gt = tools0.create_initial_state(device=device, dtype=torch.bfloat16,
                                        initial_latent=win_gt.to(torch.bfloat16))
    save_mp4(decode(tools0.unpatchify(st_gt)), out / "probe_gt.mp4")
    print("[probe] probe_gt.mp4 ✓（真值）")

    # ---- 2/3) lora ref / noref ----
    t = load_transformer_with_lora(E["CW_CKPT"], a.lora, a.lora_yaml, device)
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
    from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent

    def gen(use_reference: bool, name: str):
        tools = runner._create_video_tools(a.width, a.height, nframes)
        state = tools.create_initial_state(device=device, dtype=torch.bfloat16)
        if use_reference:
            state = VideoConditionByReferenceLatent(
                latent=ref, downscale_factor=1, temporal_scale_factor=1, strength=1.0
            ).apply_to(state, tools)
        state = VideoConditionByLatentIndex(latent=prefix, strength=1.0, latent_idx=0).apply_to(state, tools)
        video_clean = state
        gen_ = torch.Generator(device=device).manual_seed(a.seed)
        state = GaussianNoiser(generator=gen_)(state, noise_scale=1.0)
        v_pos, _, v_neg, _ = runner._get_prompt_embeddings(runner._cached_embeddings[0], device)
        out_state, _ = runner._run_denoising(
            transformer=t, video_state=state, audio_state=None,
            video_clean=video_clean, audio_clean=None,
            video_frozen=False, audio_frozen=False,
            v_ctx_pos=v_pos, a_ctx_pos=None, v_ctx_neg=v_neg, a_ctx_neg=None,
            device=device, sampling_ctx=_StubCtx())
        save_mp4(decode(out_state, tools), out / f"{name}.mp4")
        print(f"[probe] {name}.mp4 ✓")

    gen(use_reference=True, name="probe_ref")
    gen(use_reference=False, name="probe_noref")
    print(f"\n[probe] 完成 → {out}")
    print("  对比 probe_ref vs probe_noref：reference 有没有帮身份保持、有没有'人变熊'。")
    print("  probe_gt 是真值参照。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
