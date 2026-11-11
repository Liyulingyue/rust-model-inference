"""Small contracts for the independent GGUF vision reference."""

import unittest

import numpy as np

from tools.oracle.sophon_realtime import vision_oracle as v


class VisionOracleTests(unittest.TestCase):
    def test_gguf_names_cover_blocks_merger_and_deepstack_layer_indexes(self):
        for name, expected in {
            "v.patch_embd.bias": "patch_embed.proj.bias",
            "v.position_embd.weight": "pos_embed.weight",
            "v.blk.2.attn_qkv.weight": "blocks.2.attn.qkv.weight",
            "v.blk.2.attn_out.bias": "blocks.2.attn.proj.bias",
            "v.blk.2.ffn_down.weight": "blocks.2.mlp.linear_fc2.weight",
            "v.blk.2.ln1.bias": "blocks.2.norm1.bias",
            "v.post_ln.weight": "merger.norm.weight",
            "mm.0.weight": "merger.linear_fc1.weight",
            "mm.2.bias": "merger.linear_fc2.bias",
            "v.deepstack.16.fc2.weight": "deepstack_merger_list.1.linear_fc2.weight",
        }.items():
            self.assertEqual(v.tensor_name(name, [8, 16, 24]), expected)
        for name in ("v.deepstack.7.fc1.weight", "v.blk.0.unknown.weight", "output.weight"):
            with self.assertRaises(ValueError):
                v.tensor_name(name, [8, 16, 24])

    def test_temporal_patch_split_restores_channel_then_frame_order(self):
        first = np.array([[10, 20, 30]], dtype=np.float32)
        second = np.array([[11, 21, 31]], dtype=np.float32)
        actual = v.join_patch_weights(first, second, hidden=1, patch=1)
        self.assertEqual(actual.shape, (1, 3, 2, 1, 1))
        np.testing.assert_array_equal(actual.ravel(), [10, 11, 20, 21, 30, 31])


if __name__ == "__main__":
    unittest.main()
