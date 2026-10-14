"""端到端部署管线：VLM brain + video renderer。

真正想部署的形态：给一张首帧图（i2v）或起始视频（v2v）+ 两个预定义 prompt
（seg0 initial / seg1 event），系统自回归生成 3 段视频，其中：
  - seg0：用预定义 initial prompt 生成 → VLM 看，多轮对话记录【主体+环境】
  - seg1：用预定义 event prompt 生成   → VLM 看，记录【画外事件】
  - VLM 综合前两段推理出【主体的 reaction】，写出 seg2 的 revisit prompt（风格同训练）
  - seg2：用 VLM 生成的 prompt 生成      → VLM 再看一次做闭环自校验
输出：拼接后的完整 mp4 + 一份 brain_log.json（VLM 每步的记录/推理/校验）。

renderer 在 cuda（CW_GPU 指定，默认 0）；VLM 是独立的 vLLM 服务在另一张卡（见 CW_VLM_URL）。
renderer 用哪个起跳由 CW_ENGINE=v2v|i2v 决定（复用现有两套逻辑，不新造）。

预定义 seg0/seg1 prompt：复用 captions.jsonl 里该 clip 的 segment_prompts 前两段
（评估阶段方便和 GT 对照）；seg2 的 GT prompt 只用于旁边打印对比，不喂给生成。

用法（配置环境变量并启动 vLLM 服务后）：
  CW_GPU=0 CW_ENGINE=i2v python deploy_pipeline.py --clips <id> --lora <ckpt> --lora-yaml <yaml> --out-root <dir>
先起 vLLM 服务（另一张卡，示例）：
  CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
      --model /path/to/Qwen3.8-27B \
      --served-model-name Qwen3.8-27B --port 8001 --trust-remote-code
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

import torch

from latent_rollout import seg_prompts, PREFIX_LAT, WINDOW_LAT              # noqa: E402
from latent_probe import (  # noqa: E402
    _StubCtx,
    load_transformer_with_fused_loras,
    load_transformer_with_lora,
    resolve_pt,
    save_mp4,
)
from vlm_brain import VlmBrain                                             # noqa: E402


def latent_to_pil_frames(pixels, n_frames: int, return_idx: bool = False):
    """[C,F,H,W] in [0,1] cpu → 均匀抽 n_frames 张 PIL.Image。
    return_idx=True 时同时返回选到的 pixel-frame 索引数组（供 VLM retrieval 反推 latent index）。"""
    from PIL import Image
    F = pixels.shape[1]
    if F <= 0:
        return ([], []) if return_idx else []
    idx = [round(i * (F - 1) / max(1, n_frames - 1)) for i in range(min(n_frames, F))]
    idx = sorted(set(idx))
    out = []
    for i in idx:
        arr = (pixels[:, i].clamp(0, 1) * 255).round().to(torch.uint8)     # [C,H,W]
        arr = arr.permute(1, 2, 0).contiguous().numpy()                    # [H,W,C]
        out.append(Image.fromarray(arr))
    return (out, idx) if return_idx else out


def load_precomputed_embeds(embeds_dir, clip):
    """从预编码目录读该 clip 的 .pt，拼成 {seg_idx: CachedPromptEmbeddings}。
    找不到文件 → 返回 None（上层回退到现场编码）。三段分开存，seg 为空则该段 None。"""
    from ltx_trainer.validation_runner import CachedPromptEmbeddings
    p = Path(embeds_dir) / (clip.replace("/", "__") + ".pt")
    if not p.is_file():
        return None
    rec = torch.load(p, map_location="cpu", weights_only=False)
    neg = rec["neg"]
    out = {}
    for idx, seg in enumerate(("seg0", "seg1", "seg2")):
        s = rec.get(seg)
        if s is None:
            out[idx] = None
            continue
        out[idx] = CachedPromptEmbeddings(
            video_context_positive=s["video"],
            audio_context_positive=s["audio"],
            video_context_negative=neg["video"],
            audio_context_negative=neg["audio"],
        )
    return out


def make_prompt_encoder(device):
    """独立的 prompt 编码器：编码 VLM 运行时生成的 seg2 prompt。
    ★ 为什么不用 runner 内部：ValidationRunner 在 init 缓存完预定义 prompt 后会【卸载 text
    encoder】（validation_runner.py:366 del），运行时无法再编码动态 prompt。所以 deploy 管线
    自己持有一份 text encoder + embeddings processor，专门给 VLM 生成的 seg2 prompt 编码。
    返回一个函数：prompt_text → CachedPromptEmbeddings（与 runner._cached_embeddings 元素同构）。"""
    from ltx_trainer.model_loader import load_text_encoder, load_embeddings_processor, embedding_weight_paths
    from ltx_trainer.validation_runner import CachedPromptEmbeddings
    E = os.environ
    te = load_text_encoder(gemma_model_path=E["CW_GEMMA"], device=device, dtype=torch.bfloat16, load_in_8bit=False)
    ep = load_embeddings_processor(
        checkpoint_path=embedding_weight_paths(E["CW_CKPT"], E["CW_GEMMA"]),
        gemma_model_path=E["CW_GEMMA"], device=device, dtype=torch.bfloat16)
    neg_hs, neg_mask = te.encode([""])[0]                 # 负向用空串（与 runner 默认一致）
    neg_out = ep.process_hidden_states(neg_hs, neg_mask)

    @torch.no_grad()
    def encode(prompt_text: str):
        pos_hs, pos_mask = te.encode([prompt_text])[0]
        pos_out = ep.process_hidden_states(pos_hs, pos_mask)
        return CachedPromptEmbeddings(
            video_context_positive=pos_out.video_encoding.cpu(),
            audio_context_positive=pos_out.audio_encoding.cpu(),
            video_context_negative=neg_out.video_encoding.cpu(),
            audio_context_negative=(neg_out.audio_encoding.cpu() if neg_out.audio_encoding is not None else None),
        )
    return encode


def run_one_clip(t, clip: str, out: Path, device, seed, width, height, steps,
                 engine: str, n_vlm_frames: int, prompt_encoder,
                 initial_prompt=None, event_prompt=None, gt_revisit=None,
                 gt_seg2_mode: bool = False, fixed_seg2_prompt=None,
                 fixed_seg2_source=None, fixed_seg2_mode="precomputed-reasoning",
                 precomputed_embeds=None,
                 distilled_8step: bool = False):
    """initial_prompt/event_prompt：seg0/seg1 的预定义 prompt。
    - 显式传入（benchmark 走 testset.jsonl 自带的 initial_prompt/event_prompt）时用传入的；
    - 都不传时回退到 captions.jsonl 的前两段（早期单 clip 调试用）。
    gt_revisit：GT 的 revisit prompt，只用于 GT 模式或日志/评测对照。
    fixed_seg2_prompt：外部冻结的 realized reasoning prompt。给定时直接用于
      seg2，完全不调用 VLM；它不会覆盖或伪装成 benchmark GT。
    - gt_seg2_mode=False（默认）：gt_revisit 仅写进 log 作对照，seg2 由 VLM brain 推理。
    - gt_seg2_mode=True：直接把 gt_revisit 当 seg2 prompt 生成，不调 VLM brain；
      结果完全可复现（不依赖 VLM 服务），适合 benchmark 受控实验。
    precomputed_embeds：{seg_idx: CachedPromptEmbeddings|None} 或 None。
    - 非 None：用预编码 embedding，runner 用空 samples 构建（跳过 gemma 加载，省时）。数值等价。
    - None：现有路径（runner init 内部编码 seg0/seg1；VLM 模式 seg2 用 prompt_encoder）。"""
    from ltx_trainer.config import ValidationConfig, ValidationSample
    from ltx_trainer.validation_runner import ValidationRunner
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
    from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent

    E = os.environ
    disable_entity_anchor = int(E.get("CW_DISABLE_ENTITY_ANCHOR", "0") or 0) != 0
    if disable_entity_anchor and int(E.get("CW_USE_IAA", "0") or 0) != 0:
        raise ValueError(
            "CW_DISABLE_ENTITY_ANCHOR=1 is incompatible with CW_USE_IAA=1: "
            "IAA has no entity keys to bias. Disable IAA for the no-entity ablation."
        )
    win_frames = (WINDOW_LAT - 1) * 8 + 1
    # ★ prompt 来源：优先用传入的（benchmark 的 testset 自带）；否则回退 captions 前两段。
    # CW_FORCE_CAPTIONS_PROMPT=1 时强制走 captions.jsonl 而忽略 testset 的 initial/event ——
    # 用于 testset 里 initial/event prompt 本身是"整段故事"（未三段化）导致 seg 0/1 生成时
    # 混入后续段内容的情况；captions.jsonl 是 step6_caption 用规则严格三段化过的。
    force_captions = int(E.get("CW_FORCE_CAPTIONS_PROMPT", "0") or 0) != 0
    if (not force_captions) and initial_prompt and event_prompt:
        prompts = [initial_prompt, event_prompt]
        gt_seg2 = gt_revisit or ""
    else:
        cap = seg_prompts(E["CW_WORK"], clip)
        if len(cap) < 3:
            raise ValueError(f"{clip} captions 只有 {len(cap)} 段，且未显式传入 initial/event prompt")
        prompts = [cap[0], cap[1]]
        gt_seg2 = cap[2] if not gt_revisit else gt_revisit    # 若 testset 有 gt_revisit 用作对照，仍读它
        if force_captions:
            print(f"    [{clip}] CW_FORCE_CAPTIONS_PROMPT=1 —— seg0/1 强制从 captions.jsonl 读", flush=True)

    full = torch.load(resolve_pt(E["CW_LATENTS"], clip), map_location="cpu", weights_only=True)["latents"]
    if full.dim() == 5:
        full = full[0]
    # i2v 部署实际只从输入读首帧 latent，后续全部是自回归生成；
    # 因此允许 process_videos.py 直接编码的单张 PNG（1 latent）作为输入。
    # v2v 则确实需要前 PREFIX_LAT 个 latent 作为起始视频。
    required_input_latents = 1 if engine == "i2v" else PREFIX_LAT
    if full.shape[1] < required_input_latents:
        raise ValueError(
            f"{engine} 输入至少需要 {required_input_latents} 个 latent，"
            f"当前只有 {full.shape[1]} 个"
        )

    # 起跳与 reference：i2v = 首帧（1 latent）；v2v = 前4 latent + 9帧 reference（读 CW_REFERENCE）。
    first_frame = full[:, :1].unsqueeze(0).to(device).to(torch.bfloat16)
    if engine == "i2v":
        seed_prefix = first_frame
        ref_latent = first_frame
    else:  # v2v
        seed_prefix = full[:, :PREFIX_LAT].unsqueeze(0).to(device).to(torch.bfloat16)
        rf = torch.load(resolve_pt(E["CW_REFERENCE"], clip), map_location="cpu", weights_only=True)["latents"]
        if rf.dim() == 4:
            rf = rf.unsqueeze(0)
        ref_latent = rf.to(device).to(torch.bfloat16)

    # ── scene anchor（可选，SEER-World 训练时启用）─────────────────────────────
    # 训练侧 yaml 里如果加了第二条 `type: reference / latents_dir: scene_reference_latents`
    # （见 _oos_lora_scene_template.yaml），LoRA 就是【双 ref 训出来的】。推理时必须也拼进
    # 相同的两个 ref，否则 renderer 看到的 latent 前缀分布和训练不一致，LoRA 效果打折扣。
    # 通过 CW_SCENE_REFERENCE 环境变量传入。空则退化为单 ref（base 训练路径）。
    #
    # 数据来源：make_scene_reference_latents.py 从整片 latent 里按 refs.jsonl 的
    # scene_anchor_latent（pan-peak）切 1 个 latent 存到 scene_reference_latents/。
    # 推理侧我们从整片 latent 里【当场切】—— 效果等价，避免额外磁盘依赖。
    scene_ref_latent = None
    if E.get("CW_SCENE_REFERENCE"):
        # 优先从磁盘缓存目录读；缺则从整片 latent 现切（走 pan-peak 索引，需 refs.jsonl）。
        sr_path = None
        try:
            sr_path = resolve_pt(E["CW_SCENE_REFERENCE"], clip)
        except Exception:
            sr_path = None
        if sr_path is not None:
            sr = torch.load(sr_path, map_location="cpu", weights_only=True)["latents"]
            if sr.dim() == 4:
                sr = sr.unsqueeze(0)
            scene_ref_latent = sr.to(device).to(torch.bfloat16)

    direct_seg2_mode = gt_seg2_mode or fixed_seg2_prompt is not None
    seg2_render_prompt = fixed_seg2_prompt if fixed_seg2_prompt is not None else gt_seg2
    seg2_mode_label = fixed_seg2_mode if fixed_seg2_prompt is not None else (
        "gt" if gt_seg2_mode else "vlm"
    )
    if fixed_seg2_prompt is not None and not str(fixed_seg2_prompt).strip():
        raise ValueError(f"{clip} fixed seg2 reasoning prompt is empty")

    # runner 构建：
    # - precomputed_embeds 非 None：用【空 samples】建 runner → _cache_prompt_embeddings 早返回 []，
    #   完全不加载 gemma（validation_runner.py:328）。embedding 从预编码取。省掉每 clip 重加载 gemma。
    # - 否则走原路径：runner init 内部编码 seg0/seg1（gt_seg2 时连 GT 一起编码缓存）。
    if precomputed_embeds is not None:
        cfg = ValidationConfig(samples=[],
                               video_dims=(width, height, win_frames),
                               generate_video=True, generate_audio=False, inference_steps=steps, seed=seed)
    else:
        runner_prompts = prompts if not direct_seg2_mode else prompts + [seg2_render_prompt]
        cfg = ValidationConfig(samples=[ValidationSample(prompt=p) for p in runner_prompts],
                               video_dims=(width, height, win_frames),
                               generate_video=True, generate_audio=False, inference_steps=steps, seed=seed)
    runner = ValidationRunner(config=cfg, model_path=E["CW_CKPT"],
                              text_encoder_path=E["CW_GEMMA"], video_vae_path=E["CW_VIDEO_VAE"])
    if getattr(runner, "_vae_decoder", None) is not None:
        runner._vae_decoder = runner._vae_decoder.to(device)

    def embeds_for_prompt(prompt_text, seg_idx):
        """seg0/seg1 用缓存好的预定义嵌入；seg2 视模式而定。
        precomputed_embeds 非 None 时，全部从预编码取（gt 模式含 seg2；vlm 模式 seg2 仍走 prompt_encoder）。"""
        if precomputed_embeds is not None:
            if seg_idx < 2 or direct_seg2_mode:
                ce = precomputed_embeds.get(seg_idx)
                if ce is None:
                    raise ValueError(f"{clip} 预编码缺 seg{seg_idx}（该段 prompt 为空？）")
                return runner._get_prompt_embeddings(ce, device)
            # vlm 模式 seg2：预编码没有（现场生成），用 prompt_encoder。
            return runner._get_prompt_embeddings(prompt_encoder(prompt_text), device)
        # 原路径：runner 内部缓存的嵌入。
        if seg_idx < 2 or direct_seg2_mode:
            return runner._get_prompt_embeddings(runner._cached_embeddings[seg_idx], device)
        # 动态：VLM 生成的 seg2 prompt 用独立 prompt_encoder 现编码（runner 的 text encoder 已卸载）。
        return runner._get_prompt_embeddings(prompt_encoder(prompt_text), device)

    def gen_window(prefix, seg_idx, prompt_text,
                   entity_override=None, scene_override=None):
        """override 只在 seg=2 且 VLM retrieve 开时非 None —— 用 VLM 从 self-generated latent
        里挑的 slice 替代 default entity/scene ref。"""
        tools = runner._create_video_tools(width, height, win_frames)
        state = tools.create_initial_state(device=device, dtype=torch.bfloat16)
        entity_ref_for_this = entity_override if entity_override is not None else ref_latent
        # Controlled inference-only ablation.  The first frame is still used to
        # seed the i2v rollout (seg0), but it is not re-appended as a persistent
        # reference in any segment.  Later segments therefore receive only the
        # previous AR prefix.  Default remains the historical entity-anchor path.
        if not disable_entity_anchor:
            state = VideoConditionByReferenceLatent(
                latent=entity_ref_for_this, downscale_factor=1, temporal_scale_factor=1, strength=1.0).apply_to(state, tools)
        # SEER 双 ref：把 scene anchor 作为第二条 reference append 进去（幂等 apply）。
        # 训练时 yaml 里就是两条 `type: reference` —— flexible.py 逐条 apply；这里对齐。
        scene_ref_for_this = scene_override if scene_override is not None else scene_ref_latent
        if scene_ref_for_this is not None:
            state = VideoConditionByReferenceLatent(
                latent=scene_ref_for_this, downscale_factor=1, temporal_scale_factor=1, strength=1.0).apply_to(state, tools)
        state = VideoConditionByLatentIndex(latent=prefix, strength=1.0, latent_idx=0).apply_to(state, tools)
        # ── IAA 推理侧 hook（可选）─────────────────────────────────────────────
        # CW_USE_IAA=1 + seg==2 (revisit chunk) 时，对 state.attention_mask 里
        # (target rows, last N_sink cols) 位置乘 e^β，让 revisit 生成时更强 attend
        # 到 entity anchor。timestep schedule 未在推理侧实现（去噪 loop 在 runner
        # 内部，改动大），当前实现是 bias 全程加，建议推理时用较小的 β（如 2.0）。
        if int(E.get("CW_USE_IAA", "0") or 0) != 0 and seg_idx == 2:
            try:
                # 延迟 import，避免非 IAA 场景下的开销
                import sys as _sys
                _here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if _here not in _sys.path:
                    _sys.path.insert(0, _here)
                from iaa_utils import apply_iaa_to_state
                # h_pat/w_pat = 像素 H/W / 32（LTX VAE 空间下采样），假设 patch_size=1
                h_pat = height // 32
                w_pat = width // 32
                state = apply_iaa_to_state(
                    state, revisit_this_seg=True, timestep_ratio=0.0,
                    n_entity_ref_latents=int(E.get("CW_IAA_ENTITY_REFS", "2") or 2),
                    h_pat=h_pat, w_pat=w_pat)
                if state.attention_mask is None or not bool((state.attention_mask > 1).any().item()):
                    raise RuntimeError("IAA returned without any positive attention boost")
                print(f"    [IAA] seg2 attention bias 已注入 (beta={E.get('CW_IAA_BETA','3.0')})", flush=True)
            except Exception as _ie:
                print(f"    [IAA] warn: attention bias 注入失败，本 seg 走 no-bias: {_ie}", flush=True)
                # IAA experiments fail closed by default. Set 0 only for an intentional
                # diagnostic fallback to the base attention path.
                if int(E.get("CW_IAA_STRICT", "1") or 1) != 0:
                    raise RuntimeError("IAA attention bias injection failed in strict mode") from _ie
        clean = state
        g = torch.Generator(device=device).manual_seed(seed + seg_idx)
        state = GaussianNoiser(generator=g)(state, noise_scale=1.0)
        v_pos, _, v_neg, _ = embeds_for_prompt(prompt_text, seg_idx)
        if distilled_8step:
            # Official LTX-2.5 distilled recipe: fixed 8-step sigma schedule and
            # one positive-only transformer pass per step (no CFG/STG).  Merely
            # setting ValidationConfig.inference_steps=8 would use the wrong
            # scheduler and would retain the standard guidance path.
            from ltx_core.components.diffusion_steps import EulerDiffusionStep
            from ltx_core.model.transformer import X0Model
            from ltx_pipelines.utils.constants import DISTILLED_SIGMAS
            from ltx_pipelines.utils.denoisers import SimpleDenoiser
            from ltx_pipelines.utils.samplers import euler_denoising_loop

            sigmas = DISTILLED_SIGMAS.to(device=device, dtype=torch.float32)
            out_state, _ = euler_denoising_loop(
                sigmas=sigmas,
                video_state=state,
                audio_state=None,
                stepper=EulerDiffusionStep(),
                transformer=X0Model(t),
                denoiser=SimpleDenoiser(v_context=v_pos, a_context=None),
            )
        else:
            out_state, _ = runner._run_denoising(
                transformer=t, video_state=state, audio_state=None, video_clean=clean, audio_clean=None,
                video_frozen=False, audio_frozen=False,
                v_ctx_pos=v_pos, a_ctx_pos=None, v_ctx_neg=v_neg, a_ctx_neg=None,
                device=device, sampling_ctx=_StubCtx())
        return tools.unpatchify(tools.clear_conditioning(out_state)).latent

    def decode_seg(seg_latent):
        """把一段窗口 latent 解成 [C,F,H,W] 像素（用来抽帧给 VLM）。"""
        frl = (seg_latent.shape[2] - 1) * 8 + 1
        tf = runner._create_video_tools(width, height, frl)
        st = tf.create_initial_state(device=device, dtype=torch.bfloat16,
                                     initial_latent=seg_latent.to(torch.bfloat16))
        return runner._decode_video(tf.unpatchify(st), device)             # [C,F,H,W] cpu [0,1]

    # ── VLM brain（仅 gt_seg2_mode=False 用；True 则 seg2 用 GT prompt，不调 VLM）──
    brain = None if direct_seg2_mode else VlmBrain()
    log = {"clip": clip, "engine": engine, "gt_seg2": gt_seg2,
           "entity_anchor_enabled": not disable_entity_anchor,
           "sampler": "official-distilled-8step" if distilled_8step else f"standard-{steps}step",
           "seg2_mode": seg2_mode_label,
           "fixed_seg2_source": fixed_seg2_source,
           "steps": []}

    # ── Phase-2 开关：VLM 从 self-generated seg0/seg1 latent 里挑 entity/scene index ──
    # CW_VLM_RETRIEVE=1 → VLM 挑；否则走 default（entity=first_frame/CW_REFERENCE，
    # scene=CW_SCENE_REFERENCE 目录里 GT-cut 的 pan-peak latent）。
    # 关闭时 = Phase-1（当前跑的这套）；打开时 = 真部署形态（论文 method 章的 SEER 完整形态）。
    # gt_seg2_mode=True 时不调 VLM brain，也就无法做 retrieve，强制退回 default。
    vlm_retrieve_on = (not direct_seg2_mode) and (int(E.get("CW_VLM_RETRIEVE", "0") or 0) != 0)
    # 缓存每段"净增"的 latent（去掉与前段重叠的 prefix 部分），以及对应喂给 VLM 的 pixel-frame idx，
    # 供 seg-2 之前的 retrieve 用。
    seg_lat_net = [None, None, None]        # seg-i 净增 latent（第 0 段 = 整窗；后段去 PREFIX_LAT）
    seg_frame_ids = [None, None, None]      # seg-i 抽给 VLM 的 pixel-frame index 数组
    seg_frames_cache = [None, None, None]   # seg-i 抽给 VLM 的 PIL 帧列表（VLM retrieve 复用，避免二次抽帧）
    retrieve_info = None                    # log 用

    out.mkdir(parents=True, exist_ok=True)
    prefix, acc = seed_prefix, None
    seg2_prompt = None
    seg2_entity_override = None
    seg2_scene_override = None
    for i in range(3):                                 # 固定 3 段（seg0/seg1/seg2）
        ts = time.time()
        # seg0/seg1 用预定义 prompt；seg2 用 GT（gt_seg2_mode）或 VLM 生成的 prompt。
        if i < 2:
            ptext = prompts[i]
        elif direct_seg2_mode:
            ptext = seg2_render_prompt
        else:
            ptext = seg2_prompt
        # seg2：若 VLM retrieve 开且成功，用 override；否则 default（override=None）。
        ent_ov = seg2_entity_override if i == 2 else None
        sce_ov = seg2_scene_override if i == 2 else None
        win = gen_window(prefix, i, ptext, entity_override=ent_ov, scene_override=sce_ov)
        # 新段的"净增"部分（第一段是整窗，后续段去掉重叠的 prefix）
        seg_new = win if i == 0 else win[:, :, PREFIX_LAT:]
        seg_lat_net[i] = seg_new                                     # 缓存供后续 slice
        acc = win if acc is None else torch.cat([acc, seg_new], dim=2)
        prefix = acc[:, :, -PREFIX_LAT:]

        step = {"seg": i, "prompt_used": ptext, "gen_s": round(time.time() - ts, 1)}
        if not direct_seg2_mode:
            # Only a live VLM needs intermediate pixel frames.  Fixed/GT
            # prompt modes keep the rollout in latent space until the single
            # final decode, avoiding three unnecessary VAE decodes per case.
            seg_px = decode_seg(win)
            frames, frame_ids = latent_to_pil_frames(seg_px, n_vlm_frames, return_idx=True)
            seg_frame_ids[i] = frame_ids                             # pixel-frame index
            seg_frames_cache[i] = frames                             # cache for optional retrieval
            if i == 0:
                step["vlm_record"] = brain.observe_opening(frames)
            elif i == 1:
                step["vlm_record"] = brain.observe_event(frames)
                seg2_prompt = brain.reason_revisit()       # 推理出 seg2 的生成 prompt
                step["vlm_reasoned_seg2_prompt"] = seg2_prompt
                # ★ seg1 结束、seg2 生成前：若 VLM retrieve 开，让 VLM 挑 entity/scene 帧
                if vlm_retrieve_on:
                    try:
                        seg0_frames_view = seg_frames_cache[0]
                        seg1_frames_view = seg_frames_cache[1]      # 就是当前的 frames
                        rj = brain.retrieve_references(seg0_frames_view, seg1_frames_view)
                        # 映射：VLM 挑的"第 k 张（seg 内）" → 该 seg 净增 latent 的 index
                        # 每张图对应一个 pixel-frame idx（seg0 内 0..F0-1，seg1 内 0..F1-1）；
                        # pixel-frame → latent index 换算：ceil(f / 8)（同 refs 数据管线的 TEMPORAL_STRIDE=8）
                        f0_pix = seg_frame_ids[0][rj["entity_frame_idx"]]
                        f1_pix = seg_frame_ids[1][rj["scene_frame_idx"]]
                        # seg0 净增 latent 有 WINDOW_LAT 个（seg0 = 整窗）；seg1 净增 (WINDOW_LAT - PREFIX_LAT) 个
                        F0_lat = seg_lat_net[0].shape[2]              # WINDOW_LAT
                        F1_lat = seg_lat_net[1].shape[2]              # WINDOW_LAT - PREFIX_LAT
                        # 换算：一个 seg 的净增 latent 段代表 (F_lat-1)*8+1 像素帧，抽的 pixel-frame idx
                        # 在 [0, (F_lat-1)*8]，除以 8 上取整就是 latent index（cap 到 F_lat-1）
                        import math as _math
                        e_lat = min(F0_lat - 1, _math.ceil(f0_pix / 8))
                        s_lat = min(F1_lat - 1, _math.ceil(f1_pix / 8))
                        # slice 1 个 latent 出来；保持 [B, C, 1, H, W] shape
                        seg2_entity_override = seg_lat_net[0][:, :, e_lat:e_lat + 1].contiguous().to(device).to(torch.bfloat16)
                        seg2_scene_override = seg_lat_net[1][:, :, s_lat:s_lat + 1].contiguous().to(device).to(torch.bfloat16)
                        retrieve_info = {"vlm_pick_entity_frame_view_idx": rj["entity_frame_idx"],
                                         "vlm_pick_scene_frame_view_idx": rj["scene_frame_idx"],
                                         "entity_pixel_frame": int(f0_pix),
                                         "scene_pixel_frame": int(f1_pix),
                                         "entity_latent_in_seg0": int(e_lat),
                                         "scene_latent_in_seg1": int(s_lat),
                                         "reasoning": rj["reasoning"]}
                        step["vlm_retrieve"] = retrieve_info
                        print(f"    [{clip}] VLM retrieve: entity=seg0.lat{e_lat}  scene=seg1.lat{s_lat}", flush=True)
                    except Exception as _re:
                        # 失败 fallback：override 保持 None → seg2 用 default entity/scene ref。
                        retrieve_info = {"error": f"{type(_re).__name__}: {_re}"}
                        step["vlm_retrieve_error"] = retrieve_info["error"]
                        print(f"    [{clip}] ⚠ VLM retrieve 失败，fallback default ref: {retrieve_info['error']}", flush=True)
            else:  # i == 2
                step["vlm_verify"] = brain.verify_revisit(frames)   # 闭环自校验
        log["steps"].append(step)
        print(f"    [{clip}] seg {i+1}/3  {step['gen_s']}s"
              + (f"  → seg2 prompt: {ptext[:70]}…" if i == 1 else ""), flush=True)

    # 最终拼接解码 + 落盘
    fr = (acc.shape[2] - 1) * 8 + 1
    tools_full = runner._create_video_tools(width, height, fr)
    st = tools_full.create_initial_state(device=device, dtype=torch.bfloat16,
                                         initial_latent=acc.to(torch.bfloat16))
    mp4_path = (out / "output.mp4").resolve()
    save_mp4(runner._decode_video(tools_full.unpatchify(st), device), mp4_path)
    log_path = (out / "brain_log.json").resolve()
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2))
    # ★ 汇总：把本 case 实际喂给 seg2 的 prompt + GT 对照追加到 <out_root>/seg2_prompts.jsonl，
    #   边跑边写。gt_seg2_mode 时 seg2 用的就是 GT；VLM 脑模式记 VLM 推理出的 prompt。
    _sum_path = out.parent / "seg2_prompts.jsonl"
    with open(_sum_path, "a", encoding="utf-8") as _sf:
        _sf.write(json.dumps({"clip": clip, "engine": engine,
                              "seg2_mode": seg2_mode_label,
                              "seg2_prompt_used": (seg2_render_prompt if direct_seg2_mode else seg2_prompt),
                              "fixed_seg2_source": fixed_seg2_source,
                              "vlm_seg2_prompt": seg2_prompt, "gt_seg2": gt_seg2},
                             ensure_ascii=False) + "\n")
    print(f"    [{clip}] ✅ 视频已保存: {mp4_path}", flush=True)
    print(f"    [{clip}]    brain_log: {log_path}", flush=True)
    print(f"    [{clip}]    seg2 prompt 已记入: {_sum_path}", flush=True)
    del runner


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="")
    ap.add_argument("--clip-file", default="")
    ap.add_argument("--testset", default="", help="benchmark testset.jsonl：从中读 clip + initial/event/gt prompt")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个（调试）")
    ap.add_argument("--pick", default="",
                    help="从 testset 里只挑这些 clip_id（逗号分隔），仍读 testset 的 prompt。"
                         "与 --clips 不同：--clips 走 captions 回退分支、拿不到 testset prompt")
    ap.add_argument("--shard", default="0/1", help="i/n 分片：多卡并行时每卡跑一片 cases[i::n]")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--lora-yaml", default="")
    ap.add_argument("--lora-strength", type=float, default=1.0,
                    help="我们的 LoRA 融合强度（native loader / --distilled-lora 模式使用）")
    ap.add_argument("--native-lora-loader", action="store_true",
                    help="仅将我们的 LoRA 改用 LTX native fusion loader 加载；不启用 distilled sampler")
    ap.add_argument("--distilled-lora", default="",
                    help="官方 LTX-2.5 distilled LoRA；给定后自动启用官方固定 8-step、无 CFG/STG")
    ap.add_argument("--distilled-lora-strength", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=896)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--vlm-frames", type=int, default=6, help="每段抽几帧喂 VLM")
    ap.add_argument("--gt-seg2", action="store_true",
                    help="seg2 直接用 GT revisit prompt 生成（不调 VLM brain，结果可复现）；默认关=用 VLM 脑")
    ap.add_argument("--fixed-seg2-prompts", default="",
                    help="冻结的逐 case seg2 reasoning JSONL。支持 clip/clip_id 和 "
                         "reasoning_prompt/seg2_prompt_used；严格要求与 testset 的 clip 集完全一致。"
                         "启用后不调用 VLM，且 benchmark GT 仍只保留作评测对照。")
    ap.add_argument("--fixed-seg2-mode", default="precomputed-reasoning",
                    choices=("precomputed-reasoning", "benchmark-blind"),
                    help="固定 seg2 prompt 的审计标签；不改变 prompt 内容。")
    ap.add_argument("--prompt-embeds-dir", default="",
                    help="预编码 prompt embedding 目录（precompute_prompt_embeddings.py 产出）。"
                         "给了则用预编码、runner 跳过 gemma 加载（省时，数值等价）；缺某 clip 的 .pt 自动回退现场编码")
    a = ap.parse_args()
    if a.lora and not a.lora_yaml:
        return int(bool(print("!! --lora 要配 --lora-yaml")))
    if a.native_lora_loader and not a.lora:
        return int(bool(print("!! --native-lora-loader 必须同时给 --lora")))
    if a.distilled_lora and not a.lora:
        return int(bool(print("!! --distilled-lora 模式必须同时给我们的 --lora")))
    if a.gt_seg2 and a.fixed_seg2_prompts:
        return int(bool(print("!! --gt-seg2 与 --fixed-seg2-prompts 互斥")))
    engine = os.environ.get("CW_ENGINE", "i2v")
    if engine not in ("v2v", "i2v"):
        return int(bool(print(f"!! CW_ENGINE 只能是 v2v|i2v，收到 {engine!r}")))
    need = ["CW_CKPT", "CW_GEMMA", "CW_VIDEO_VAE", "CW_LATENTS", "CW_WORK"]
    if engine == "v2v":
        need.append("CW_REFERENCE")
    for k in need:
        if not os.environ.get(k):
            return int(bool(print(f"!! {k} 未设置（请先 source .env）")))

    # cases：list of (clip_id, initial_prompt|None, event_prompt|None, gt_revisit|None)
    cases = []
    if a.testset:
        for line in open(a.testset, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cases.append((r["clip_id"], r.get("initial_prompt"), r.get("event_prompt"),
                          r.get("gt_revisit_prompt")))
    else:
        clip_ids = [c.strip() for c in a.clips.split(",") if c.strip()]
        if a.clip_file:
            clip_ids += [l.strip() for l in open(a.clip_file, encoding="utf-8") if l.strip()]
        cases = [(c, None, None, None) for c in clip_ids]     # 无 testset → 回退 captions
    fixed_seg2_prompts = {}
    fixed_seg2_source = None
    if a.fixed_seg2_prompts:
        if not a.testset:
            return int(bool(print("!! --fixed-seg2-prompts 必须配 --testset")))
        fixed_path = Path(a.fixed_seg2_prompts).resolve()
        if not fixed_path.is_file():
            return int(bool(print(f"!! fixed seg2 prompt 文件不存在: {fixed_path}")))
        duplicates = []
        for line in fixed_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            clip_id = row.get("clip") or row.get("clip_id")
            prompt = str(row.get("reasoning_prompt") or row.get("seg2_prompt_used") or "").strip()
            if not clip_id or len(prompt) < 40:
                return int(bool(print(f"!! 非法 fixed seg2 prompt row: clip={clip_id!r}, prompt={prompt!r}")))
            if clip_id in fixed_seg2_prompts:
                duplicates.append(clip_id)
            fixed_seg2_prompts[clip_id] = prompt
        testset_ids = {row[0] for row in cases}
        if set(fixed_seg2_prompts) != testset_ids:
            missing = sorted(testset_ids - set(fixed_seg2_prompts))
            extra = sorted(set(fixed_seg2_prompts) - testset_ids)
            return int(bool(print(
                f"!! fixed seg2/testset mismatch: missing={missing[:8]} extra={extra[:8]}"
            )))
        fixed_seg2_source = str(fixed_path)
        print(
            f"[deploy] fixed seg2 prompts PASS: {len(fixed_seg2_prompts)} unique "
            f"({len(duplicates)} superseded duplicate rows) <- {fixed_path}",
            flush=True,
        )

    # ★ --pick：从 testset 里【只挑指定 clip】，但仍走 testset 分支读 prompt。
    #   为什么不用 --clips：那是互斥分支（走 captions 回退），拿不到 testset 的
    #   initial/event/gt prompt。smoke 要挑特定 case（如猛兽/火灾）验证强度分档时需要这个。
    if a.pick:
        want = [c.strip() for c in a.pick.split(",") if c.strip()]
        have = {c[0] for c in cases}
        miss = [c for c in want if c not in have]
        if miss:
            print(f"!! --pick 里这些 clip 不在 testset 中: {miss}")
        order = {c: i for i, c in enumerate(want)}            # 保持 --pick 的给定顺序
        cases = sorted([c for c in cases if c[0] in order], key=lambda c: order[c[0]])
        print(f"[deploy] --pick 选中 {len(cases)} 条: {[c[0] for c in cases]}")
    if a.limit > 0:
        cases = cases[:a.limit]
    # 多卡并行：--shard i/n 只取第 i 片（cases[i::n]），每张 renderer 卡一个进程跑一片。
    try:
        si, sn = (int(x) for x in a.shard.split("/"))
    except Exception:
        return int(bool(print(f"!! --shard 格式应是 i/n，收到 {a.shard!r}")))
    if sn > 1:
        cases = cases[si::sn]
    if not cases:
        return int(bool(print("!! 没给 case（--testset 或 --clips/--clip-file），或本分片为空")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(a.out_root)

    use_precomp = bool(a.prompt_embeds_dir)
    scene_ref_on = bool(os.environ.get("CW_SCENE_REFERENCE"))
    vlm_retrieve_env = int(os.environ.get("CW_VLM_RETRIEVE", "0") or 0)
    distilled_8step = bool(a.distilled_lora)
    sampler_name = "official-distilled-8step (no CFG/STG)" if distilled_8step else f"standard-{a.steps}step"
    loader_name = "native-fusion" if distilled_8step or a.native_lora_loader else "peft-yaml"
    direct_seg2_mode = a.gt_seg2 or bool(fixed_seg2_prompts)
    seg2_mode_name = a.fixed_seg2_mode if fixed_seg2_prompts else ("gt" if a.gt_seg2 else "vlm-brain")
    print(f"[deploy] engine={engine}  seg2_mode={seg2_mode_name}  "
          f"sampler={sampler_name}  loader={loader_name}  "
          f"scene_ref={'ON:' + os.environ['CW_SCENE_REFERENCE'] if scene_ref_on else 'OFF'}  "
          f"vlm_retrieve={'ON (seg2 ref from VLM-picked self-gen latent)' if vlm_retrieve_env and not a.gt_seg2 else 'OFF (default rule-cut ref)'}  "
          f"prompt_embeds={'预编码:' + a.prompt_embeds_dir if use_precomp else '现场编码'}  "
          f"{len(cases)} case  "
          f"来源={'testset' if a.testset else 'clips'}  "
          f"VLM={os.environ.get('CW_VLM_URL','http://127.0.0.1:8001/v1')}", flush=True)
    if distilled_8step:
        t = load_transformer_with_fused_loras(
            os.environ["CW_CKPT"],
            ((a.lora, a.lora_strength), (a.distilled_lora, a.distilled_lora_strength)),
            device,
        )
    elif a.native_lora_loader:
        t = load_transformer_with_fused_loras(
            os.environ["CW_CKPT"],
            ((a.lora, a.lora_strength),),
            device,
        )
    else:
        t = load_transformer_with_lora(os.environ["CW_CKPT"], a.lora, a.lora_yaml, device)
    # prompt_encoder（给 VLM 生成的 seg2 prompt 编码）：只有 VLM 脑模式需要。
    # 但预编码模式下，若能保证 testset 每 clip 的 .pt 都齐（gt 模式），可完全不建 encoder。
    # 保守起见：只要不是 gt 模式就建（VLM seg2 要用）。gt 模式一律不建（无论是否预编码）。
    prompt_encoder = None
    if not direct_seg2_mode:
        print("[deploy] 构建独立 prompt encoder（给 VLM 生成的 seg2 prompt 编码）…", flush=True)
        prompt_encoder = make_prompt_encoder(device)

    fresh = bool(int(os.environ.get("CW_DEPLOY_FRESH", "0") or 0))   # =1 强制重跑，不跳过已有
    ok = fail = skip = 0
    for n, (clip, ini_p, evt_p, gt_p) in enumerate(cases, 1):
        out = out_root / clip.replace("/", "__")
        if (out / "output.mp4").is_file() and not fresh:
            skip += 1
            continue
        t0 = time.time()
        try:
            # 预编码：读该 clip 的 .pt；缺文件则 None → run_one_clip 自动回退现场编码。
            pce = load_precomputed_embeds(a.prompt_embeds_dir, clip) if use_precomp else None
            run_one_clip(t, clip, out, device, a.seed, a.width, a.height, a.steps,
                         engine, a.vlm_frames, prompt_encoder,
                         initial_prompt=ini_p, event_prompt=evt_p, gt_revisit=gt_p,
                         gt_seg2_mode=a.gt_seg2,
                         fixed_seg2_prompt=fixed_seg2_prompts.get(clip),
                         fixed_seg2_source=fixed_seg2_source,
                         fixed_seg2_mode=a.fixed_seg2_mode,
                         precomputed_embeds=pce,
                         distilled_8step=distilled_8step)
            (out / "ERROR.txt").unlink(missing_ok=True)
            ok += 1
            print(f"[{n}/{len(cases)}] ✓ {clip}  {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            fail += 1
            out.mkdir(parents=True, exist_ok=True)
            (out / "ERROR.txt").write_text(f"{e}\n\n{traceback.format_exc()}")
            print(f"[{n}/{len(cases)}] ✗ {clip}  {e}", flush=True)
    print(f"\n[deploy] 完成 ok={ok} 跳过={skip} 失败={fail} / 共 {len(cases)}")
    # Per-clip failures are caught above so the remaining shard can continue,
    # but the worker must still report failure to its launcher afterwards.
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
