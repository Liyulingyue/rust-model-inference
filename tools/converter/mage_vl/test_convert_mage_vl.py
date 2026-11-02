import copy
import json
from pathlib import Path
import unittest

from tools.converter.mage_vl.convert_mage_vl import export, map_text_name, validate_config


class ContractTests(unittest.TestCase):
    def test_released_contract_and_reject_changed_operators(self):
        original = json.loads((Path(__file__).resolve().parents[3]/'tests/fixtures/mage_vl/config.json').read_text())
        validate_config(original)
        for part, key, bad in [('text_config','head_dim',80), ('text_config','rope_scaling',{'mrope_section':[24,20,20]}),
                               ('text_config','use_sliding_window',True), ('text_config','layer_types',['sliding_attention']*36),
                               ('vision_config','temporal_patch_size',2), ('vision_config','hidden_act','gelu_pytorch_tanh'),
                               ('vision_config','use_patch_position_encoding',True)]:
            config = copy.deepcopy(original)
            config[part][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(config)

    def test_text_mapping_is_explicit(self):
        self.assertEqual(map_text_name('model.layers.35.self_attn.q_norm.weight'), 'blk.35.attn_q_norm.weight')
        self.assertEqual(map_text_name('model.embed_tokens.weight'), 'token_embd.weight')
        with self.assertRaises(ValueError):
            map_text_name('model.layers.0.self_attn.q_proj.bias')
        with self.assertRaises(ValueError):
            map_text_name('visual.encoder.layers.0.mlp.fc1.weight')

    def test_export_rejects_unknown_type_before_reading_weights(self):
        with self.assertRaisesRegex(ValueError, 'bf16 or f32'):
            export(Path('missing'), Path('unused'), 'q4_k')


if __name__ == '__main__':
    unittest.main()
