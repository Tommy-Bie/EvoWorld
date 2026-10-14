"""Identity-Anchored Attention (IAA) —— 训练/推理共享的 log-bias 注入 util。

方法概览
--------
Revisit chunk（离视重现段）生成时，让 attention 更强地 attend 到 entity anchor —— 也就是
包含主体身份的那 2 个片头 reference latent。做法：在 self-attention 的 log-space 加性 bias
上对 (revisit_query_rows, entity_ref_key_cols) 位置**加 +β**。

## LTX-2.5 序列布局（重要：训练和推理不同）

训练的 flexible strategy 使用 `torch.cat([cond_latents, noisy_latents])`，因此：

    train = [ entity reference (前) | target window (后) ]

推理的 `VideoConditionByReferenceLatent.apply_to` 使用
`torch.cat([latent_state.latent, ref_tokens])`，因此：

    infer = [ target window (前) | entity reference (后) ]

无 mask 时 self-attention 对 token 排列等变，两种排列语义一致；IAA 按物理下标修改 mask，
必须分别定位。训练 boost `mask[target_rows_after_ref, first_N_ref_cols]`，推理 boost
`mask[target_rows_before_ref, last_N_ref_cols]`。禁止把推理端的“末尾 reference”假设照搬到训练。

## LTX-2.5 内部路径

    LatentState.attention_mask (值 [0,1])
      → _prepare_self_attention_mask (log-space 转换：bias = log(mask.clamp(eps)))
      → SDPA 当 attn_mask（加性 bias）

利用的性质：log(e^β) = +β。所以我们**把 mask 里被 boost 的位置值设为 e^β**（正好利用 LTX
已有 log 变换）。这样：
  - 未被 boost 的位置 mask=1 → log-bias=0（无变化）
  - 被 boost 的位置 mask=e^β → log-bias=+β
  - 之前完全屏蔽的位置 mask=0 → log-bias=-inf（依然屏蔽）

**完全开关式**：`CW_USE_IAA=0`（默认）时本模块的入口函数 no-op 直接返回 mask，训练/推理行为
与之前 100% 等价。

关键参数（环境变量，yaml 里的 iaa 字段亦可传入训练脚本）：
  CW_USE_IAA          "1" = 启用；空/0 = no-op
  CW_IAA_BETA         float，默认 3.0。log-bias 强度（3.0 ≈ e^3=20× attention 偏好）
  CW_IAA_CUTOFF       预留参数；只有 caller 在每个 denoise step 重新调用本函数并传入
                      timestep_ratio 时才生效。当前 deploy/validation 是全程 bias。
  CW_IAA_ENTITY_REFS  int，默认 2。entity anchor 用几个 latent（head-2 就是 2）

## 训练侧接口

训练必须调用 `FlexibleStrategy._build_iaa_mask`。它能同时看到 reference 拼接前的
`target_window_seq_len` 和拼接后的 `loss_mask`，因此可以验证真实 reference 长度，并只选择
参与 loss 的 generated queries。不要另写一个只靠总长度猜 prefix/target 边界的 helper。

## 推理侧接口

在 deploy_pipeline.py 里 seg2 gen_window 前:

    from iaa_utils import apply_iaa_to_state
    state = apply_iaa_to_state(state, is_revisit_seg=True,
                               n_entity_ref_latents=2,
                               h_pat=..., w_pat=...)
"""
from __future__ import annotations
import math
import os


def iaa_enabled() -> bool:
    """全局开关；default off。"""
    return int(os.environ.get("CW_USE_IAA", "0") or 0) != 0


def _iaa_config() -> dict:
    """读环境变量，返回 IAA 超参 dict。"""
    return {
        "beta": float(os.environ.get("CW_IAA_BETA", "3.0") or 3.0),
        "cutoff": float(os.environ.get("CW_IAA_CUTOFF", "0.5") or 0.5),
        "entity_refs": int(os.environ.get("CW_IAA_ENTITY_REFS", "2") or 2),
    }


def _n_sink_tokens_from_latents(n_entity_ref_latents: int, h_pat: int, w_pat: int) -> int:
    """entity anchor 有 n 个 latent，每个 latent 展开为 h_pat × w_pat 个 token（patchify）。
    默认 LTX 的 patchifier patch_size=1 → 每 latent 每 spatial 位置 = 1 token。
    """
    return int(n_entity_ref_latents) * int(h_pat) * int(w_pat)


def training_iaa_ranges(seq_len: int, n_sink: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """返回训练布局的 ``(query_range, entity_key_range)``。

    flexible training 拼接为 ``[entity_ref | target_window]``。
    """
    if n_sink <= 0 or n_sink >= seq_len:
        raise ValueError(f"invalid IAA sink: n_sink={n_sink}, seq_len={seq_len}")
    return (n_sink, seq_len), (0, n_sink)


def inference_iaa_ranges(seq_len: int, n_sink: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """返回推理布局的 ``(query_range, entity_key_range)``。

    VideoConditionByReferenceLatent 拼接为 ``[target_window | entity_ref]``。
    """
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
    """推理时用：给 LatentState 里的 attention_mask 加 IAA boost，返回新 state。

    - 输入是 LatentState（推理时的整体状态）
    - 单 sample（B=1）
    - 支持 timestep_ratio：ratio > cutoff 时不加 bias（推理 late steps 撤 bias）

    如果 IAA 未启用 / 非 revisit / 已过 cutoff → 返回原 state。
    """
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

    # ReferenceVideoConditioningItem(attention_mask=None) 默认保留全注意力，并不会主动
    # materialize dense mask；旧实现因此在这里直接 no-op，却仍由上层打印“已注入”。IAA
    # 必须在 None 时显式创建全 1 mask，之后再写入 e^beta block。
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
    # denoise_mask>0 是推理端真正需要生成的 query；prefix 与 reference 都是 0。
    # 直接从 state 读取，避免硬编码 V2V=4 / I2V 首段=1 的 prefix 长度。
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
