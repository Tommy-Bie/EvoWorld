"""IAA token-layout regression tests (pure Python; no GPU or torch required)."""
from __future__ import annotations

import unittest

from iaa_utils import inference_iaa_ranges, training_iaa_ranges


class IaaLayoutTest(unittest.TestCase):
    def test_v2v_training_reference_is_first(self):
        tokens_per_latent = 16 * 28  # 512x896 / VAE stride 32
        entity_tokens = 2 * tokens_per_latent
        total_tokens = (2 + 15) * tokens_per_latent
        queries, keys = training_iaa_ranges(total_tokens, entity_tokens)
        self.assertEqual(keys, (0, 896))
        self.assertEqual(queries, (896, 7616))

    def test_v2v_inference_reference_is_last(self):
        tokens_per_latent = 16 * 28
        entity_tokens = 2 * tokens_per_latent
        total_tokens = (15 + 2) * tokens_per_latent
        queries, keys = inference_iaa_ranges(total_tokens, entity_tokens)
        self.assertEqual(queries, (0, 6720))
        self.assertEqual(keys, (6720, 7616))

    def test_i2v_uses_one_entity_latent_on_opposite_sides(self):
        tokens_per_latent = 16 * 28
        total_tokens = (1 + 15) * tokens_per_latent
        train_queries, train_keys = training_iaa_ranges(total_tokens, tokens_per_latent)
        infer_queries, infer_keys = inference_iaa_ranges(total_tokens, tokens_per_latent)
        self.assertEqual(train_keys, (0, 448))
        self.assertEqual(train_queries, (448, 7168))
        self.assertEqual(infer_queries, (0, 6720))
        self.assertEqual(infer_keys, (6720, 7168))

    def test_invalid_sink_rejected(self):
        for sink in (0, 10):
            with self.assertRaises(ValueError):
                training_iaa_ranges(10, sink)
            with self.assertRaises(ValueError):
                inference_iaa_ranges(10, sink)


if __name__ == "__main__":
    unittest.main()
