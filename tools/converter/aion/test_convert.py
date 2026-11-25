import unittest

from tools.converter.aion.convert import tensor_map


class TensorContractTest(unittest.TestCase):
    def test_expanded_query_and_tied_output(self):
        config = dict(hidden_size=2048, head_dim=128, num_attention_heads=24,
                      num_key_value_heads=8, intermediate_size=8192,
                      vocab_size=200029, num_hidden_layers=1)
        tensors = tensor_map(config)
        self.assertEqual(tensors["model.layers.0.self_attn.q_proj.weight"],
                         ("blk.0.attn_q.weight", (3072, 2048)))
        self.assertEqual(tensors["model.layers.0.self_attn.o_proj.weight"],
                         ("blk.0.attn_output.weight", (2048, 3072)))
        self.assertNotIn("lm_head.weight", tensors)
