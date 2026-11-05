"""EvoWorld autoregressive inference pipeline."""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

import torch

from utils import (  # noqa: E402
    PREFIX_LAT,
    WINDOW_LAT,
    SamplingContext,
    load_transformer_with_fused_loras,
    load_transformer_with_lora,
    resolve_pt,
    save_mp4,
    segment_prompts,
)
from vlm_brain import VlmBrain                                             # noqa: E402


def latent_to_pil_frames(pixels, n_frames: int, return_idx: bool = False):
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
    from ltx_trainer.model_loader import load_text_encoder, load_embeddings_processor, embedding_weight_paths
    from ltx_trainer.validation_runner import CachedPromptEmbeddings
    E = os.environ
    te = load_text_encoder(gemma_model_path=E["EW_GEMMA"], device=device, dtype=torch.bfloat16, load_in_8bit=False)
    ep = load_embeddings_processor(
        checkpoint_path=embedding_weight_paths(E["EW_CKPT"], E["EW_GEMMA"]),
        gemma_model_path=E["EW_GEMMA"], device=device, dtype=torch.bfloat16)
    neg_hs, neg_mask = te.encode([""])[0]
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
    from ltx_trainer.config import ValidationConfig, ValidationSample
    from ltx_trainer.validation_runner import ValidationRunner
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
    from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent

    E = os.environ
    disable_entity_anchor = int(E.get("EW_DISABLE_ENTITY_ANCHOR", "0") or 0) != 0
    if disable_entity_anchor and int(E.get("EW_USE_IAA", "0") or 0) != 0:
        raise ValueError(
            "EW_DISABLE_ENTITY_ANCHOR=1 is incompatible with EW_USE_IAA=1: "
            "IAA has no entity keys to bias. Disable IAA for the no-entity ablation."
        )
    win_frames = (WINDOW_LAT - 1) * 8 + 1


    force_captions = int(E.get("EW_FORCE_CAPTIONS_PROMPT", "0") or 0) != 0
    if (not force_captions) and initial_prompt and event_prompt:
        prompts = [initial_prompt, event_prompt]
        gt_seg2 = gt_revisit or ""
    else:
        cap = segment_prompts(E["EW_WORK"], clip)
        if len(cap) < 3:
            raise ValueError(f"{clip} has {len(cap)} caption segments and no explicit initial/event prompts")
        prompts = [cap[0], cap[1]]
        gt_seg2 = cap[2] if not gt_revisit else gt_revisit
        if force_captions:
            print(f"    [{clip}] loading seg0/seg1 from captions.jsonl", flush=True)

    full = torch.load(resolve_pt(E["EW_LATENTS"], clip), map_location="cpu", weights_only=True)["latents"]
    if full.dim() == 5:
        full = full[0]


    required_input_latents = 1 if engine == "i2v" else PREFIX_LAT
    if full.shape[1] < required_input_latents:
        raise ValueError(
            f"{engine} requires at least {required_input_latents} input latents; got {full.shape[1]}"
        )


    first_frame = full[:, :1].unsqueeze(0).to(device).to(torch.bfloat16)
    if engine == "i2v":
        seed_prefix = first_frame
        ref_latent = first_frame
    else:  # v2v
        seed_prefix = full[:, :PREFIX_LAT].unsqueeze(0).to(device).to(torch.bfloat16)
        rf = torch.load(resolve_pt(E["EW_REFERENCE"], clip), map_location="cpu", weights_only=True)["latents"]
        if rf.dim() == 4:
            rf = rf.unsqueeze(0)
        ref_latent = rf.to(device).to(torch.bfloat16)


    scene_ref_latent = None
    if E.get("EW_SCENE_REFERENCE"):

        sr_path = None
        try:
            sr_path = resolve_pt(E["EW_SCENE_REFERENCE"], clip)
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


    if precomputed_embeds is not None:
        cfg = ValidationConfig(samples=[],
                               video_dims=(width, height, win_frames),
                               generate_video=True, generate_audio=False, inference_steps=steps, seed=seed)
    else:
        runner_prompts = prompts if not direct_seg2_mode else prompts + [seg2_render_prompt]
        cfg = ValidationConfig(samples=[ValidationSample(prompt=p) for p in runner_prompts],
                               video_dims=(width, height, win_frames),
                               generate_video=True, generate_audio=False, inference_steps=steps, seed=seed)
    runner = ValidationRunner(config=cfg, model_path=E["EW_CKPT"],
                              text_encoder_path=E["EW_GEMMA"], video_vae_path=E["EW_VIDEO_VAE"])
    if getattr(runner, "_vae_decoder", None) is not None:
        runner._vae_decoder = runner._vae_decoder.to(device)

    def embeds_for_prompt(prompt_text, seg_idx):
        if precomputed_embeds is not None:
            if seg_idx < 2 or direct_seg2_mode:
                ce = precomputed_embeds.get(seg_idx)
                if ce is None:
                    raise ValueError(f"Missing precomputed seg{seg_idx} embedding for {clip}")
                return runner._get_prompt_embeddings(ce, device)

            return runner._get_prompt_embeddings(prompt_encoder(prompt_text), device)

        if seg_idx < 2 or direct_seg2_mode:
            return runner._get_prompt_embeddings(runner._cached_embeddings[seg_idx], device)

        return runner._get_prompt_embeddings(prompt_encoder(prompt_text), device)

    def gen_window(prefix, seg_idx, prompt_text,
                   entity_override=None, scene_override=None):
        tools = runner._create_video_tools(width, height, win_frames)
        state = tools.create_initial_state(device=device, dtype=torch.bfloat16)
        entity_ref_for_this = entity_override if entity_override is not None else ref_latent
        # Optional inference without persistent entity references.
        if not disable_entity_anchor:
            state = VideoConditionByReferenceLatent(
                latent=entity_ref_for_this, downscale_factor=1, temporal_scale_factor=1, strength=1.0).apply_to(state, tools)


        scene_ref_for_this = scene_override if scene_override is not None else scene_ref_latent
        if scene_ref_for_this is not None:
            state = VideoConditionByReferenceLatent(
                latent=scene_ref_for_this, downscale_factor=1, temporal_scale_factor=1, strength=1.0).apply_to(state, tools)
        state = VideoConditionByLatentIndex(latent=prefix, strength=1.0, latent_idx=0).apply_to(state, tools)


        if int(E.get("EW_USE_IAA", "0") or 0) != 0 and seg_idx == 2:
            try:

                import sys as _sys
                _here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if _here not in _sys.path:
                    _sys.path.insert(0, _here)
                from iaa_utils import apply_iaa_to_state

                h_pat = height // 32
                w_pat = width // 32
                state = apply_iaa_to_state(
                    state, revisit_this_seg=True, timestep_ratio=0.0,
                    n_entity_ref_latents=int(E.get("EW_IAA_ENTITY_REFS", "2") or 2),
                    h_pat=h_pat, w_pat=w_pat)
                if state.attention_mask is None or not bool((state.attention_mask > 1).any().item()):
                    raise RuntimeError("IAA returned without any positive attention boost")
                print(f"    [IAA] applied seg2 attention bias (beta={E.get('EW_IAA_BETA','3.0')})", flush=True)
            except Exception as _ie:
                print(f"    [IAA] warning: continuing without attention bias: {_ie}", flush=True)
                # Fail closed unless fallback is explicitly enabled.
                if int(E.get("EW_IAA_STRICT", "1") or 1) != 0:
                    raise RuntimeError("IAA attention bias injection failed in strict mode") from _ie
        clean = state
        g = torch.Generator(device=device).manual_seed(seed + seg_idx)
        state = GaussianNoiser(generator=g)(state, noise_scale=1.0)
        v_pos, _, v_neg, _ = embeds_for_prompt(prompt_text, seg_idx)
        if distilled_8step:
            # Official fixed eight-step distilled sampler.
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
                device=device, sampling_ctx=SamplingContext())
        return tools.unpatchify(tools.clear_conditioning(out_state)).latent

    def decode_seg(seg_latent):
        frl = (seg_latent.shape[2] - 1) * 8 + 1
        tf = runner._create_video_tools(width, height, frl)
        st = tf.create_initial_state(device=device, dtype=torch.bfloat16,
                                     initial_latent=seg_latent.to(torch.bfloat16))
        return runner._decode_video(tf.unpatchify(st), device)             # [C,F,H,W] cpu [0,1]


    brain = None if direct_seg2_mode else VlmBrain()
    log = {"clip": clip, "engine": engine, "gt_seg2": gt_seg2,
           "entity_anchor_enabled": not disable_entity_anchor,
           "sampler": "official-distilled-8step" if distilled_8step else f"standard-{steps}step",
           "seg2_mode": seg2_mode_label,
           "fixed_seg2_source": fixed_seg2_source,
           "steps": []}


    vlm_retrieve_on = (not direct_seg2_mode) and (int(E.get("EW_VLM_RETRIEVE", "0") or 0) != 0)


    seg_lat_net = [None, None, None]
    seg_frame_ids = [None, None, None]
    seg_frames_cache = [None, None, None]
    retrieve_info = None

    out.mkdir(parents=True, exist_ok=True)
    prefix, acc = seed_prefix, None
    seg2_prompt = None
    seg2_entity_override = None
    seg2_scene_override = None
    for i in range(3):
        ts = time.time()

        if i < 2:
            ptext = prompts[i]
        elif direct_seg2_mode:
            ptext = seg2_render_prompt
        else:
            ptext = seg2_prompt

        ent_ov = seg2_entity_override if i == 2 else None
        sce_ov = seg2_scene_override if i == 2 else None
        win = gen_window(prefix, i, ptext, entity_override=ent_ov, scene_override=sce_ov)

        seg_new = win if i == 0 else win[:, :, PREFIX_LAT:]
        seg_lat_net[i] = seg_new
        acc = win if acc is None else torch.cat([acc, seg_new], dim=2)
        prefix = acc[:, :, -PREFIX_LAT:]

        step = {"seg": i, "prompt_used": ptext, "gen_s": round(time.time() - ts, 1)}
        if not direct_seg2_mode:
            # Decode intermediate frames only for live VLM reasoning.
            seg_px = decode_seg(win)
            frames, frame_ids = latent_to_pil_frames(seg_px, n_vlm_frames, return_idx=True)
            seg_frame_ids[i] = frame_ids                             # pixel-frame index
            seg_frames_cache[i] = frames                             # cache for optional retrieval
            if i == 0:
                step["vlm_record"] = brain.observe_opening(frames)
            elif i == 1:
                step["vlm_record"] = brain.observe_event(frames)
                seg2_prompt = brain.reason_revisit()
                step["vlm_reasoned_seg2_prompt"] = seg2_prompt

                if vlm_retrieve_on:
                    try:
                        seg0_frames_view = seg_frames_cache[0]
                        seg1_frames_view = seg_frames_cache[1]
                        rj = brain.retrieve_references(seg0_frames_view, seg1_frames_view)


                        f0_pix = seg_frame_ids[0][rj["entity_frame_idx"]]
                        f1_pix = seg_frame_ids[1][rj["scene_frame_idx"]]

                        F0_lat = seg_lat_net[0].shape[2]              # WINDOW_LAT
                        F1_lat = seg_lat_net[1].shape[2]              # WINDOW_LAT - PREFIX_LAT


                        import math as _math
                        e_lat = min(F0_lat - 1, _math.ceil(f0_pix / 8))
                        s_lat = min(F1_lat - 1, _math.ceil(f1_pix / 8))

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

                        retrieve_info = {"error": f"{type(_re).__name__}: {_re}"}
                        step["vlm_retrieve_error"] = retrieve_info["error"]
                        print(f"    [{clip}] VLM retrieval failed; using default references: {retrieve_info['error']}", flush=True)
            else:  # i == 2
                step["vlm_verify"] = brain.verify_revisit(frames)
        log["steps"].append(step)
        print(f"    [{clip}] seg {i+1}/3  {step['gen_s']}s"
              + (f"  -> seg2 prompt: {ptext[:70]}..." if i == 1 else ""), flush=True)


    fr = (acc.shape[2] - 1) * 8 + 1
    tools_full = runner._create_video_tools(width, height, fr)
    st = tools_full.create_initial_state(device=device, dtype=torch.bfloat16,
                                         initial_latent=acc.to(torch.bfloat16))
    mp4_path = (out / "output.mp4").resolve()
    save_mp4(runner._decode_video(tools_full.unpatchify(st), device), mp4_path)
    log_path = (out / "brain_log.json").resolve()
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2))


    _sum_path = out.parent / "seg2_prompts.jsonl"
    with open(_sum_path, "a", encoding="utf-8") as _sf:
        _sf.write(json.dumps({"clip": clip, "engine": engine,
                              "seg2_mode": seg2_mode_label,
                              "seg2_prompt_used": (seg2_render_prompt if direct_seg2_mode else seg2_prompt),
                              "fixed_seg2_source": fixed_seg2_source,
                              "vlm_seg2_prompt": seg2_prompt, "gt_seg2": gt_seg2},
                             ensure_ascii=False) + "\n")
    print(f"    [{clip}] saved video: {mp4_path}", flush=True)
    print(f"    [{clip}]    brain_log: {log_path}", flush=True)
    print(f"    [{clip}] saved seg2 prompt: {_sum_path}", flush=True)
    del runner


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="")
    ap.add_argument("--clip-file", default="")
    ap.add_argument("--testset", default="", help="Benchmark testset JSONL")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N cases")
    ap.add_argument("--pick", default="",
                    help="Comma-separated clip IDs selected from the testset")
    ap.add_argument("--shard", default="0/1", help="Shard specification i/n")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--lora-yaml", default="")
    ap.add_argument("--lora-strength", type=float, default=1.0,
                    help="EvoWorld LoRA strength")
    ap.add_argument("--native-lora-loader", action="store_true",
                    help="Use the native LTX LoRA fusion loader")
    ap.add_argument("--distilled-lora", default="",
                    help="Official LTX-2.5 distilled LoRA")
    ap.add_argument("--distilled-lora-strength", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=896)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--vlm-frames", type=int, default=6, help="Frames sampled per segment for the VLM")
    ap.add_argument("--gt-seg2", action="store_true",
                    help="Use the provided revisit prompt for seg2")
    ap.add_argument("--fixed-seg2-prompts", default="",
                    help="JSONL containing fixed seg2 reasoning prompts")
    ap.add_argument("--fixed-seg2-mode", default="precomputed-reasoning",
                    choices=("precomputed-reasoning", "benchmark-blind"),
                    help="Audit label for fixed seg2 prompts")
    ap.add_argument("--prompt-embeds-dir", default="",
                    help="Directory containing precomputed prompt embeddings")
    a = ap.parse_args()
    if a.lora and not a.lora_yaml:
        return int(bool(print("--lora requires --lora-yaml")))
    if a.native_lora_loader and not a.lora:
        return int(bool(print("--native-lora-loader requires --lora")))
    if a.distilled_lora and not a.lora:
        return int(bool(print("--distilled-lora requires --lora")))
    if a.gt_seg2 and a.fixed_seg2_prompts:
        return int(bool(print("--gt-seg2 and --fixed-seg2-prompts are mutually exclusive")))
    engine = os.environ.get("EW_ENGINE", "i2v")
    if engine not in ("v2v", "i2v"):
        return int(bool(print(f"EW_ENGINE must be v2v or i2v; got {engine!r}")))
    need = ["EW_CKPT", "EW_GEMMA", "EW_VIDEO_VAE", "EW_LATENTS", "EW_WORK"]
    if engine == "v2v":
        need.append("EW_REFERENCE")
    for k in need:
        if not os.environ.get(k):
            return int(bool(print(f"Missing required environment variable: {k}")))

    # Case tuples: clip ID and optional initial, event, and revisit prompts.
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
        cases = [(c, None, None, None) for c in clip_ids]
    fixed_seg2_prompts = {}
    fixed_seg2_source = None
    if a.fixed_seg2_prompts:
        if not a.testset:
            return int(bool(print("--fixed-seg2-prompts requires --testset")))
        fixed_path = Path(a.fixed_seg2_prompts).resolve()
        if not fixed_path.is_file():
            return int(bool(print(f"Fixed seg2 prompt file not found: {fixed_path}")))
        duplicates = []
        for line in fixed_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            clip_id = row.get("clip") or row.get("clip_id")
            prompt = str(row.get("reasoning_prompt") or row.get("seg2_prompt_used") or "").strip()
            if not clip_id or len(prompt) < 40:
                return int(bool(print(f"Invalid fixed seg2 prompt row: clip={clip_id!r}, prompt={prompt!r}")))
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


    if a.pick:
        want = [c.strip() for c in a.pick.split(",") if c.strip()]
        have = {c[0] for c in cases}
        miss = [c for c in want if c not in have]
        if miss:
            print(f"Clips requested by --pick are missing from the testset: {miss}")
        order = {c: i for i, c in enumerate(want)}
        cases = sorted([c for c in cases if c[0] in order], key=lambda c: order[c[0]])
        print(f"[deploy] selected {len(cases)} cases: {[c[0] for c in cases]}")
    if a.limit > 0:
        cases = cases[:a.limit]

    try:
        si, sn = (int(x) for x in a.shard.split("/"))
    except Exception:
        return int(bool(print(f"--shard must use i/n format; got {a.shard!r}")))
    if sn > 1:
        cases = cases[si::sn]
    if not cases:
        return int(bool(print("No cases were selected for this shard")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(a.out_root)

    use_precomp = bool(a.prompt_embeds_dir)
    scene_ref_on = bool(os.environ.get("EW_SCENE_REFERENCE"))
    vlm_retrieve_env = int(os.environ.get("EW_VLM_RETRIEVE", "0") or 0)
    distilled_8step = bool(a.distilled_lora)
    sampler_name = "official-distilled-8step (no CFG/STG)" if distilled_8step else f"standard-{a.steps}step"
    loader_name = "native-fusion" if distilled_8step or a.native_lora_loader else "peft-yaml"
    direct_seg2_mode = a.gt_seg2 or bool(fixed_seg2_prompts)
    seg2_mode_name = a.fixed_seg2_mode if fixed_seg2_prompts else ("gt" if a.gt_seg2 else "vlm-brain")
    print(f"[deploy] engine={engine}  seg2_mode={seg2_mode_name}  "
          f"sampler={sampler_name}  loader={loader_name}  "
          f"scene_ref={'ON:' + os.environ['EW_SCENE_REFERENCE'] if scene_ref_on else 'OFF'}  "
          f"vlm_retrieve={'ON (seg2 ref from VLM-picked self-gen latent)' if vlm_retrieve_env and not a.gt_seg2 else 'OFF (default rule-cut ref)'}  "
          f"prompt_embeds={'precomputed:' + a.prompt_embeds_dir if use_precomp else 'runtime'}  "
          f"{len(cases)} case  "
          f"source={'testset' if a.testset else 'clips'}  "
          f"VLM={os.environ.get('EW_VLM_URL','http://127.0.0.1:8001/v1')}", flush=True)
    if distilled_8step:
        t = load_transformer_with_fused_loras(
            os.environ["EW_CKPT"],
            ((a.lora, a.lora_strength), (a.distilled_lora, a.distilled_lora_strength)),
            device,
        )
    elif a.native_lora_loader:
        t = load_transformer_with_fused_loras(
            os.environ["EW_CKPT"],
            ((a.lora, a.lora_strength),),
            device,
        )
    else:
        t = load_transformer_with_lora(os.environ["EW_CKPT"], a.lora, a.lora_yaml, device)


    prompt_encoder = None
    if not direct_seg2_mode:
        print("[deploy] building the runtime prompt encoder...", flush=True)
        prompt_encoder = make_prompt_encoder(device)

    fresh = bool(int(os.environ.get("EW_DEPLOY_FRESH", "0") or 0))
    ok = fail = skip = 0
    for n, (clip, ini_p, evt_p, gt_p) in enumerate(cases, 1):
        out = out_root / clip.replace("/", "__")
        if (out / "output.mp4").is_file() and not fresh:
            skip += 1
            continue
        t0 = time.time()
        try:

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
    print(f"\n[deploy] done: ok={ok} skipped={skip} failed={fail} total={len(cases)}")
    # Report failure after completing the remaining cases.
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
