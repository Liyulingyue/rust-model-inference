import unittest
import numpy as np
from tools.oracle.pjit.text_reference import grouped, restore


class LayoutTests(unittest.TestCase):
    def test_restores_grouped_heads_and_preserves_qk(self):
        original = np.arange(32*128, dtype=np.float32).reshape(32*128, 1)
        tiled = original.reshape(16, 2, 128, 1).swapaxes(0, 1).reshape(original.shape)
        np.testing.assert_array_equal(grouped(tiled, 0, 128), original)
        qk = np.full((4096, 1), -1., dtype=np.float32)
        result = restore(np.concatenate([qk, tiled]), 'model.layers.0.linear_attn.in_proj_qkv.weight', (8192, 1))
        np.testing.assert_array_equal(result, np.concatenate([qk, original]))

    def test_restores_norm_offset_and_ssm_log(self):
        np.testing.assert_array_equal(restore(np.array([1.5]), 'model.norm.weight', (1,)), [.5])
        with self.assertRaises(ValueError):
            restore(np.ones(32), 'model.layers.0.linear_attn.A_log', (32,))


if __name__ == '__main__':
    unittest.main()
