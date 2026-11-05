"""Identity-anchored attention utilities for EvoWorld."""
from __future__ import annotations
import math
import os


def iaa_enabled() -> bool:
    return int(os.environ.get("EW_USE_IAA", "0") or 0) != 0


def _iaa_config() -> dict:
    return {
        "beta": float(os.environ.get("EW_IAA_BETA", "3.0") or 3.0),
        "cutoff": float(os.environ.get("EW_IAA_CUTOFF", "0.5") or 0.5),
        "entity_refs": int(os.environ.get("EW_IAA_ENTITY_REFS", "2") or 2),
    }


def _n_sink_tokens_from_latents(n_entity_ref_latents: int, h_pat: int, w_pat: int) -> int:
    return int(n_entity_ref_latents) * int(h_pat) * int(w_pat)


def training_iaa_ranges(seq_len: int, n_sink: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if n_sink <= 0 or n_sink >= seq_len:
        raise ValueError(f"invalid IAA sink: n_sink={n_sink}, seq_len={seq_len}")
    return (n_sink, seq_len), (0, n_sink)


def inference_iaa_ranges(seq_len: int, n_sink: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if n_sink <= 0 or n_sink >= seq_len:
        raise ValueError(f"invalid IAA sink: n_sink={n_sink}, seq_len={seq_len}")
    split = seq_len - n_sink
    return (0, split), (split, seq_len)


def apply_iaa_to_state(
    state,
    revisit_this_seg: bool,
    timestep_ratio: float = 0.0,
    n_entity_ref_latents: int | None = None,
    h_pat: int | None = None,
    w_pat: int | None = None,
    n_target_latents: int | None = None,
    beta: float | None = None,
    cutoff: float | None = None,
):
    if not iaa_enabled():
        return state
    if not revisit_this_seg:
        return state
    cfg = _iaa_config()
    if cutoff is None:
        cutoff = cfg["cutoff"]
    if timestep_ratio > cutoff:
        return state
    if beta is None:
        beta = cfg["beta"]
    if n_entity_ref_latents is None:
        n_entity_ref_latents = cfg["entity_refs"]

    import torch
    if h_pat is None or w_pat is None:
        return state
    n_sink = _n_sink_tokens_from_latents(n_entity_ref_latents, h_pat, w_pat)
    if n_sink <= 0:
        return state


    attention_mask = state.attention_mask
    if attention_mask is None:
        b, tk, _ = state.latent.shape
        attention_mask = torch.ones(
            b, tk, tk, device=state.latent.device, dtype=state.latent.dtype)
    b, tq, tk = attention_mask.shape
    # inference: [target window | entity reference]
    (ts, default_te), (ss, se) = inference_iaa_ranges(tk, n_sink)
    boost = math.exp(beta)
    new_mask = attention_mask.clone()


    query_mask = state.denoise_mask[..., 0] > 0
    for b_i in range(b):
        q_idx = query_mask[b_i].nonzero(as_tuple=False).flatten()
        if q_idx.numel() == 0:
            continue
        qs = int(q_idx[0].item())
        qe = int(q_idx[-1].item()) + 1
        if q_idx.numel() != qe - qs or qs < ts or qe > default_te:
            raise RuntimeError(
                f"IAA inference queries must be one contiguous span inside "
                f"[{ts},{default_te}), got [{qs},{qe}) count={q_idx.numel()}"
            )
        actual_trailing_reference = tk - qe
        if actual_trailing_reference != n_sink:
            raise RuntimeError(
                "IAA inference reference length mismatch: "
                f"trailing conditioned tokens={actual_trailing_reference}, configured sink={n_sink}"
            )
        if n_target_latents is not None:
            qe = min(qe, qs + n_target_latents * h_pat * w_pat)
        new_mask[b_i, qs:qe, ss:se] = new_mask[b_i, qs:qe, ss:se] * boost

    import dataclasses
    return dataclasses.replace(state, attention_mask=new_mask)
