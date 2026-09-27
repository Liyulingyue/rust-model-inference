import unittest

from tools.converter.laya.convert_laya import tensor_contracts, validate_config


class ContractTest(unittest.TestCase):
    def test_multilingual_contract_and_wrong_backbone(self):
        encoder = dict(model_type="modernbert", hidden_size=768, num_hidden_layers=22,
                       num_attention_heads=12, intermediate_size=1152, vocab_size=256000,
                       local_attention=128, max_position_embeddings=8192,
                       layer_norm_eps=1e-5, global_attn_every_n_layers=3,
                       attention_bias=False, mlp_bias=False, norm_bias=False,
                       hidden_activation="gelu", rope_parameters={
                           "full_attention": {"rope_theta": 160000},
                           "sliding_attention": {"rope_theta": 160000}},
                       layer_types=["full_attention" if i % 3 == 0 else "sliding_attention"
                                    for i in range(22)])
        agent = dict(encoder="jhu-clsp/mmBERT-base", head_layers=2, max_len=1024,
                     head_max_len=256, act_costs={"escalate": .5},
                     temperature=[1., 1., 1.], temperature_by_options={})
        validate_config(encoder, agent)
        shapes = tensor_contracts(encoder, agent)
        self.assertEqual(len(shapes), 170)
        self.assertEqual(shapes["encoder.embeddings.tok_embeddings.weight"], (256000, 768))
        self.assertNotIn("encoder.layers.0.attn_norm.weight", shapes)
        self.assertEqual(shapes["head.layers.1.self_attn.in_proj_weight"], (2304, 768))
        self.assertEqual(shapes["act_head.0.weight"], (256, 772))
        encoder["model_type"] = "qwen3"
        with self.assertRaisesRegex(ValueError, "model_type"):
            validate_config(encoder, agent)


if __name__ == "__main__":
    unittest.main()
