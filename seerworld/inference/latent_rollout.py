"""完整版：latent 空间自回归 rollout，对齐训练。支持批量（一次加载模型、循环多个 clip）。

复用 latent_probe.py 里已验证通的机制。和 probe 的唯一区别：prefix 用【上一段生成的】latent
逐段交接（暴露 exposure bias）。按 --arms 产出 joined_<臂>.mp4，臂 ∈ {ref,noref,base}：
  ref   我们的 LoRA + reference（完整方法；需 --lora）
  noref 我们的 LoRA + 不给 reference（消融：reference 通道的净贡献；需 --lora）
  base  原生 LTX-2.5（无 LoRA）+ 不给 reference（头条基线；必须 --lora "" 单独跑）

单个：   CW_GPU=0 python latent_rollout.py --clips <id> --lora <ckpt> --lora-yaml <yaml> --out-root <dir>
批量：   python latent_rollout.py --clip-file clips.txt --lora ... --arms ref,noref --out-root ...
base：   python latent_rollout.py --clip-file clips.txt --lora "" --arms base --out-root <同一目录>
过夜批量健壮：transformer 只加载一次；每 clip 独立 try/except（失败跳过记录）；断点续跑
（请求的每个臂 mp4 都在才 skip，只补缺的臂）。
本地无法验证。
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latent_probe import resolve_pt, load_transformer_with_lora, _StubCtx, save_mp4  # noqa: E402

PREFIX_LAT, TARGET_LAT = 4, 11
WINDOW_LAT = PREFIX_LAT + TARGET_LAT      # 15


def seg_prompts(work: str, clip: str) -> list[str]:
    for line in open(Path(work) / "captions.jsonl", encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("clip_id") == clip and r.get("segment_prompts"):
            return r["segment_prompts"]
    raise KeyError(f"captions.jsonl 里找不到 {clip}")


# 三个评测臂 → 是否把 reference latent 拼进窗口。
#   ref   我们的 LoRA + reference（完整方法）
#   noref 我们的 LoRA + 不给 reference（消融：隔离 reference 通道的净贡献）
#   base  原生 LTX-2.5（无 LoRA）+ 不给 reference（头条基线：整套方法 vs vanilla LTX）
# ref/noref 与 base 的差别是【加载的 transformer 不同】（base 那遍用 --lora "" 加载），
# 与 reference 无关；这个表只管 reference 拼不拼。base 跑在 base transformer 上、不拼 reference。
ARM_USE_REF = {"ref": True, "noref": False, "base": False}


def run_one_clip(t, clip: str, out: Path, device, seed, width, height, steps, arms):
    """处理一个 clip：建 runner（每 clip 的 prompt 不同）→ 按 arms 逐个 rollout。transformer t 复用。"""
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
    seed_prefix = full[:, :PREFIX_LAT].unsqueeze(0).to(device).to(torch.bfloat16)
    # reference latent 只有 ref 臂用得到；base/noref-only 的遍不必加载（也不必存在）。
    ref = None
    if "ref" in arms:
        ref = torch.load(resolve_pt(E["CW_REFERENCE"], clip), map_location="cpu", weights_only=True)["latents"]
        if ref.dim() == 4:
            ref = ref.unsqueeze(0)
        ref = ref.to(device)

    cfg = ValidationConfig(samples=[ValidationSample(prompt=p) for p in prompts],
                           video_dims=(width, height, win_frames),
                           generate_video=True, generate_audio=False, inference_steps=steps, seed=seed)
    runner = ValidationRunner(config=cfg, model_path=E["CW_CKPT"],
                              text_encoder_path=E["CW_GEMMA"], video_vae_path=E["CW_VIDEO_VAE"])
    if getattr(runner, "_vae_decoder", None) is not None:
        runner._vae_decoder = runner._vae_decoder.to(device)

    def gen_window(prefix, seg_idx, use_reference):
        tools = runner._create_video_tools(width, height, win_frames)
        state = tools.create_initial_state(device=device, dtype=torch.bfloat16)
        if use_reference:
            state = VideoConditionByReferenceLatent(
                latent=ref, downscale_factor=1, temporal_scale_factor=1, strength=1.0).apply_to(state, tools)
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

    def rollout(use_reference, name):
        prefix, acc = seed_prefix, None
        for i in range(nseg):
            win = gen_window(prefix, i, use_reference)
            acc = win if acc is None else torch.cat([acc, win[:, :, PREFIX_LAT:]], dim=2)
            prefix = acc[:, :, -PREFIX_LAT:]
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
    ap.add_argument("--arms", default="ref,noref",
                    help="逗号分隔，子集 of {ref,noref,base}。带 lora 默认 ref,noref；base 那遍传 --arms base --lora ''")
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
    # ref 臂需要 LoRA；base 臂必须【不】带 LoRA（带了就不是 base 了）。挡在启动处，别跑一半才发现。
    if "ref" in arms and not a.lora:
        return int(bool(print("!! --arms 含 ref 但没给 --lora（ref 是我们方法的完整形态，必须带 LoRA）")))
    if "base" in arms and a.lora:
        return int(bool(print("!! --arms 含 base 但给了 --lora —— base 是原生 LTX 基线，必须 --lora '' 单独跑")))
    for k in ("CW_CKPT", "CW_GEMMA", "CW_VIDEO_VAE", "CW_LATENTS", "CW_REFERENCE", "CW_WORK"):
        if not os.environ.get(k):
            return int(bool(print(f"!! {k} 未设置（请先 source .env）")))

    clips = [c.strip() for c in a.clips.split(",") if c.strip()]
    if a.clip_file:
        clips += [l.strip() for l in open(a.clip_file, encoding="utf-8") if l.strip()]
    if not clips:
        return int(bool(print("!! 没给 clip（--clips 或 --clip-file）")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(a.out_root)

    print(f"[rollout] {len(clips)} 个 clip，臂={arms}，加载 transformer（lora={a.lora or '无=base'}）…", flush=True)
    t = load_transformer_with_lora(os.environ["CW_CKPT"], a.lora, a.lora_yaml, device)

    ok = fail = skip = 0
    for n, clip in enumerate(clips, 1):
        out = out_root / clip.replace("/", "__")
        # 断点续跑：请求的每个臂的 mp4 都在才跳过（只补缺的臂）。
        if all((out / f"joined_{arm}.mp4").is_file() for arm in arms):
            skip += 1
            continue
        t0 = time.time()
        try:
            run_one_clip(t, clip, out, device, a.seed, a.width, a.height, a.steps, arms)
            ok += 1
            print(f"[{n}/{len(clips)}] ✓ {clip}  {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            fail += 1
            (out).mkdir(parents=True, exist_ok=True)
            (out / "ERROR.txt").write_text(f"{e}\n\n{traceback.format_exc()}")
            print(f"[{n}/{len(clips)}] ✗ {clip}  {e}", flush=True)
    print(f"\n[rollout] 完成 ok={ok} 跳过={skip} 失败={fail} / 共 {len(clips)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
