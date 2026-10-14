"""CPU preflight for the exact IAA masks used by training and inference.

Run before launching a costly multi-GPU job.  This imports the production functions,
not a reimplementation, and verifies layout, revisit gating, clean-prefix exclusion,
None-mask materialization, and the final log-space +beta conversion.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "ltx-trainer" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "ltx-core" / "src"))

import torch

from iaa_utils import apply_iaa_to_state
from ltx_core.model.transformer.transformer_args import TransformerArgsPreprocessor
from ltx_core.types import LatentState
from ltx_trainer.training_strategies.flexible import FlexibleStrategy
from ltx_trainer.validation_runner import ValidationRunner


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    try:
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    except AssertionError as exc:
        raise AssertionError(f"{label} mismatch: {exc}") from exc


def main() -> int:
    old = {key: os.environ.get(key) for key in ("CW_USE_IAA", "CW_IAA_BETA", "CW_IAA_ENTITY_REFS")}
    beta = float(os.environ.get("CW_IAA_BETA", "3.0") or 3.0)
    os.environ.update(CW_USE_IAA="1", CW_IAA_BETA=str(beta), CW_IAA_ENTITY_REFS="2")
    try:
        # Miniature spatial grid: 2x3 tokens/latent. Keep the real temporal layout:
        # entity=2, temporal prefix=4, generated target=11.
        per_latent = 6
        entity = 2 * per_latent
        prefix = 4 * per_latent
        target = 11 * per_latent
        total = entity + prefix + target
        boost = math.exp(beta)

        # Production training path: [entity | clean temporal prefix | generated target].
        loss_mask = torch.zeros(2, total, dtype=torch.bool)
        loss_mask[:, entity + prefix:] = True
        train = FlexibleStrategy._build_iaa_mask(
            batch={"is_revisit": torch.tensor([1, 0])},
            seq_len=total,
            target_window_seq_len=prefix + target,
            loss_mask=loss_mask,
            h_pat=2,
            w_pat=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        expected_train = torch.ones(2, total, total)
        expected_train[0, entity + prefix:, :entity] = boost
        _assert_close(train, expected_train, "training mask")

        # I2V first window: one entity latent, one clean image latent, fourteen
        # generated latents. This specifically guards the variable-prefix branch.
        os.environ["CW_IAA_ENTITY_REFS"] = "1"
        i2v_entity = per_latent
        i2v_prefix = per_latent
        i2v_target = 14 * per_latent
        i2v_total = i2v_entity + i2v_prefix + i2v_target
        i2v_loss = torch.zeros(1, i2v_total, dtype=torch.bool)
        i2v_loss[:, i2v_entity + i2v_prefix:] = True
        train_i2v = FlexibleStrategy._build_iaa_mask(
            batch={"is_revisit": torch.tensor([1])},
            seq_len=i2v_total,
            target_window_seq_len=i2v_prefix + i2v_target,
            loss_mask=i2v_loss,
            h_pat=2,
            w_pat=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        expected_i2v = torch.ones(1, i2v_total, i2v_total)
        expected_i2v[0, i2v_entity + i2v_prefix:, :i2v_entity] = boost
        _assert_close(train_i2v, expected_i2v, "I2V first-window training mask")
        os.environ["CW_IAA_ENTITY_REFS"] = "2"

        # Production inference path: [clean temporal prefix | generated target | entity].
        infer_total = prefix + target + entity
        denoise = torch.zeros(1, infer_total, 1)
        denoise[:, prefix:prefix + target] = 1
        state = LatentState(
            latent=torch.zeros(1, infer_total, 4),
            denoise_mask=denoise,
            positions=torch.zeros(1, 3, infer_total, 2),
            clean_latent=torch.zeros(1, infer_total, 4),
        )
        out = apply_iaa_to_state(
            state,
            revisit_this_seg=True,
            h_pat=2,
            w_pat=3,
            n_entity_ref_latents=2,
            beta=beta,
        )
        if out.attention_mask is None:
            raise AssertionError("inference failed to materialize an attention mask from None")
        expected_infer = torch.ones(1, infer_total, infer_total)
        expected_infer[0, prefix:prefix + target, prefix + target:] = boost
        _assert_close(out.attention_mask, expected_infer, "inference mask")

        # Validation/deployment share this state->Modality adapter. The mask must survive
        # that boundary or both paths silently run unmasked.
        modality = ValidationRunner._modality_from_latent_state(
            out, context=torch.zeros(1, 1, 4), sigma=torch.ones(1))
        if modality.attention_mask is not out.attention_mask:
            raise AssertionError("LatentState attention mask was dropped before transformer input")

        # Transformer must convert multiplier e^beta to additive logit bias +beta.
        bias = TransformerArgsPreprocessor._prepare_self_attention_mask(
            None, out.attention_mask, torch.float32)
        expected_bias = torch.zeros(1, 1, infer_total, infer_total)
        expected_bias[0, 0, prefix:prefix + target, prefix + target:] = beta
        _assert_close(bias, expected_bias, "transformer log-space bias")

        # Non-revisit inference must remain a true no-op (including mask=None).
        untouched = apply_iaa_to_state(state, revisit_this_seg=False, h_pat=2, w_pat=3)
        if untouched is not state or untouched.attention_mask is not None:
            raise AssertionError("non-revisit inference is not a no-op")

        print("IAA runtime preflight: PASS")
        print("  train: [entity|prefix|target], boost target queries -> first entity keys")
        print("  i2v: one-frame entity + one-frame first-window prefix + 14 generated latents verified")
        print("  infer: [prefix|target|entity], materialize mask and boost target queries -> last entity keys")
        print("  gate: non-revisit unchanged; clean prefix excluded; log(e^beta)=+beta verified")
        return 0
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
