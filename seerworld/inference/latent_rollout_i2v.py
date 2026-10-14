"""I2V 版 latent 空间自回归 rollout（方案 A2）——第一段仅首帧起跳，后续段照旧 AR。

与 latent_rollout.py 的唯一区别在【第一段的起跳条件】：
  - V2V（latent_rollout.py）：seed_prefix = 前4 latent（25帧视频），reference = 前2 latent（9帧，GT 文件）
  - I2V（本文件）：       seed_prefix = 首帧（1 latent），reference = 同一张首帧（1 latent）

后续段完全一致（prefix = 上一段生成的末 4 latent，AR 交接；reference 全程用首帧锚身份）。
这正是训练侧 `CW_I2V_START=1` 对应的推理路径（首帧既起跳又当 reference）。

⚠️ 刻意独立成文件，不动 latent_rollout.py —— 你要求 I2V 不污染现有 V2V 推理，
效果不好能立刻退回。现有 latent_rollout.py / run_batchroll.sh / eval 全部照旧。

用法（配置所需环境变量后）：
  CW_GPU=0 python latent_rollout_i2v.py --clips <id> --lora <ckpt> --lora-yaml <yaml> --out-root <dir>
参考 run.sh latentroll 的 env 约定。多卡分片 / 断点续跑同 latent_rollout.py。
"""
from __future__ import annotations
import argparse, os, sys, time, traceback
from pathlib import Path

import torch

# 复用 latent_rollout.py 里已测通的所有工具（seg_prompts、load_transformer_with_lora、save_mp4、_StubCtx）
from latent_rollout import ARM_USE_REF, seg_prompts                       # noqa: E402
from latent_probe import resolve_pt, load_transformer_with_lora, _StubCtx, save_mp4  # noqa: E402
from latent_rollout import PREFIX_LAT, TARGET_LAT, WINDOW_LAT             # noqa: E402


def run_one_clip_i2v(t, clip: str, out: Path, device, seed, width, height, steps, arms):
    from ltx_trainer.config import ValidationConfig, ValidationSample
    from ltx_trainer.validation_runner import ValidationRunner
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
    from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent

    E = os.environ
    win_frames = (WINDOW_LAT - 1) * 8 + 1
    prompts = seg_prompts(E["CW_WORK"], clip)
    nseg = len(prompts)

    full = torch.load(resolve_pt(E["CW_LATENTS"], clip), map_location="cpu", weights_only=True)["latents"]
    if full.dim() == 5:
        full = full[0]
    if full.shape[1] < WINDOW_LAT:
        raise ValueError(f"整片只有 {full.shape[1]} latent")

    # ★ I2V 第一段条件：只有首帧。latent[0] 同时充当 reference（身份记忆）+ 起跳 prefix。
    first_frame = full[:, :1].unsqueeze(0).to(device).to(torch.bfloat16)   # [1,C,1,H,W]
    # ★ CW_I2V_REF：后续段的 reference 用几帧。
    #   first（默认）: 全程首帧（1 latent）—— 严格 A2 推理，与训练侧 CW_I2V_START=1 对齐
    #   multi        : 第一段仍首帧，但后续段改用完整 V2V reference（前2 latent=9帧，CW_REFERENCE 文件）
    #                  —— 现有 V2V 模型是拿 9 帧 reference 训的，喂 9 帧更贴近它的分布，
    #                     这样能公平测"首帧起跳 + 已有模型"的上限。
    ref_mode = os.environ.get("CW_I2V_REF", "first")
    if ref_mode not in ("first", "multi"):
        raise ValueError(f"!! CW_I2V_REF 只能是 first|multi，收到 {ref_mode!r}")
    v2v_ref = None
    if ref_mode == "multi":
        v2v_ref = torch.load(resolve_pt(E["CW_REFERENCE"], clip), map_location="cpu",
                             weights_only=True)["latents"]
        if v2v_ref.dim() == 4:
            v2v_ref = v2v_ref.unsqueeze(0)
        v2v_ref = v2v_ref.to(device).to(torch.bfloat16)

    def gen_window(prefix, seg_idx, use_reference):
        tools = runner._create_video_tools(width, height, win_frames)
        state = tools.create_initial_state(device=device, dtype=torch.bfloat16)
        if use_reference:
            # seg0 用首帧；seg>=1 按 CW_I2V_REF 用首帧或完整 V2V reference。
            ref_lat = first_frame if seg_idx == 0 or v2v_ref is None else v2v_ref
            state = VideoConditionByReferenceLatent(
                latent=ref_lat, downscale_factor=1, temporal_scale_factor=1, strength=1.0).apply_to(state, tools)
        state = VideoConditionByLatentIndex(latent=prefix, strength=1.0, latent_idx=0).apply_to(state, tools)
        clean = state
        g = torch.Generator(device=device).manual_seed(seed + seg_idx)
        state = GaussianNoiser(generator=g)(state, noise_scale=1.0)
        v_pos, _, v_neg, _ = runner._get_prompt_embeddings(runner._cached_embeddings[seg_idx], device)
        out_state, _ = runner._run_denoising(
            transformer=t, video_state=state, audio_state=None, video_clean=clean, audio_clean=None,
            video_frozen=False, audio_frozen=False,
            v_ctx_pos=v_pos, a_ctx_pos=None, v_ctx_neg=v_neg, a_ctx_neg=None,
            device=device, sampling_ctx=_StubCtx())
        return tools.unpatchify(tools.clear_conditioning(out_state)).latent

    cfg = ValidationConfig(samples=[ValidationSample(prompt=p) for p in prompts],
                           video_dims=(width, height, win_frames),
                           generate_video=True, generate_audio=False, inference_steps=steps, seed=seed)
    runner = ValidationRunner(config=cfg, model_path=E["CW_CKPT"],
                              text_encoder_path=E["CW_GEMMA"], video_vae_path=E["CW_VIDEO_VAE"])
    if getattr(runner, "_vae_decoder", None) is not None:
        runner._vae_decoder = runner._vae_decoder.to(device)

    def rollout(use_reference, name):
        # 第一段：prefix = 首帧（1 latent），而非前4 latent。
        prefix, acc = first_frame, None
        for i in range(nseg):
            ts = time.time()
            win = gen_window(prefix, i, use_reference)
            acc = win if acc is None else torch.cat([acc, win[:, :, PREFIX_LAT:]], dim=2)
            prefix = acc[:, :, -PREFIX_LAT:]
            print(f"    [{name}] seg {i+1}/{nseg}  {time.time()-ts:.0f}s", flush=True)
        fr = (acc.shape[2] - 1) * 8 + 1
        tools_full = runner._create_video_tools(width, height, fr)
        st = tools_full.create_initial_state(device=device, dtype=torch.bfloat16,
                                             initial_latent=acc.to(torch.bfloat16))
        save_mp4(runner._decode_video(tools_full.unpatchify(st), device), out / f"{name}.mp4")

    out.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        name = f"joined_{arm}"
        if (out / f"{name}.mp4").is_file():        # 单臂断点续跑：已有的臂不重做
            continue
        rollout(ARM_USE_REF[arm], name)
    del runner


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="", help="逗号分隔 clip_id")
    ap.add_argument("--clip-file", default="", help="每行一个 clip_id（批量）")
    ap.add_argument("--out-root", required=True, help="输出根；每 clip 建 <clip_safe>/ 子目录")
    ap.add_argument("--lora", default="")
    ap.add_argument("--lora-yaml", default="")
    ap.add_argument("--arms", default="ref",
                    help="逗号分隔，子集 of {ref,noref,base}。默认只 ref（快速看效果）；base 那遍传 --arms base --lora ''")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=896)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    if a.lora and not a.lora_yaml:
        return int(bool(print("!! --lora 要配 --lora-yaml")))
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in arms if x not in ARM_USE_REF]
    if bad:
        return int(bool(print(f"!! 未知臂 {bad}，只能是 {list(ARM_USE_REF)}")))
    if "ref" in arms and not a.lora:
        return int(bool(print("!! --arms 含 ref 但没给 --lora（ref 必须带 LoRA）")))
    if "base" in arms and a.lora:
        return int(bool(print("!! --arms 含 base 但给了 --lora —— base 是原生 LTX 基线，必须 --lora '' 单独跑")))
    for k in ("CW_CKPT", "CW_GEMMA", "CW_VIDEO_VAE", "CW_LATENTS", "CW_WORK"):
        if not os.environ.get(k):
            return int(bool(print(f"!! {k} 未设置（请先 source .env）")))

    clips = [c.strip() for c in a.clips.split(",") if c.strip()]
    if a.clip_file:
        clips += [l.strip() for l in open(a.clip_file, encoding="utf-8") if l.strip()]
    if not clips:
        return int(bool(print("!! 没给 clip（--clips 或 --clip-file）")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(a.out_root)

    print(f"[i2vroll] {len(clips)} 个 clip，臂={arms}，加载 transformer（lora={a.lora or '无=base'}）…", flush=True)
    t = load_transformer_with_lora(os.environ["CW_CKPT"], a.lora, a.lora_yaml, device)

    ok = fail = skip = 0
    for n, clip in enumerate(clips, 1):
        out = out_root / clip.replace("/", "__")
        if all((out / f"joined_{arm}.mp4").is_file() for arm in arms):
            skip += 1
            continue
        t0 = time.time()
        try:
            run_one_clip_i2v(t, clip, out, device, a.seed, a.width, a.height, a.steps, arms)
            ok += 1
            print(f"[{n}/{len(clips)}] ✓ {clip}  {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            fail += 1
            out.mkdir(parents=True, exist_ok=True)
            (out / "ERROR.txt").write_text(f"{e}\n\n{traceback.format_exc()}")
            print(f"[{n}/{len(clips)}] ✗ {clip}  {e}", flush=True)
    print(f"\n[i2vroll] 完成 ok={ok} 跳过={skip} 失败={fail} / 共 {len(clips)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
