import copy
import json
from pathlib import Path
import unittest
import numpy as np

from tools.converter.mage_vl.convert_mage_vl import (
    TEXT_TYPES, encode_text_tensor, export, map_text_name, validate_config,
)
from tools.converter.utils.gguf import GGML_BF16, GGML_F32
from tools.converter.utils.quantize_gguf import BLOCK_FORMATS


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
        with self.assertRaisesRegex(ValueError, 'text type'):
            export(Path('missing'), Path('unused'), 'iq_unknown')

    def test_text_formats_quantize_matrices_and_keep_norms_f32(self):
        class Reader:
            def shape(self, name):
                return (2, 256) if name.endswith('proj.weight') else (256,)

            def dtype(self, name):
                return 'BF16'

            def raw(self, name):
                return (self.f32(name).view('<u4') >> 16).astype('<u2').tobytes()

            def f32(self, name):
                # Exactly representable in BF16; matrix rows have different ranges.
                return np.arange(np.prod(self.shape(name)), dtype=np.float32).reshape(self.shape(name)) / 256

        reader = Reader()
        for mode in TEXT_TYPES:
            with self.subTest(mode=mode):
                dtype, payload = encode_text_tensor(reader, 'q_proj.weight', mode)
                expected_type = GGML_F32 if mode == 'f32' else BLOCK_FORMATS[mode][1]
                self.assertEqual(dtype, expected_type)
                sizes = {'f32': 2048, 'bf16': 1024, 'f16': 1024, 'q8_0': 544,
                         'q4_0': 288, 'q4_1': 320, 'q4_k': 288, 'q5_k': 352, 'q6_k': 420}
                self.assertEqual(len(payload), sizes[mode])
                self.assertEqual(encode_text_tensor(reader, 'norm.weight', mode),
                                 (GGML_F32, reader.f32('norm.weight').tobytes()))
        self.assertEqual(encode_text_tensor(reader, 'q_proj.weight', 'bf16'),
                         (GGML_BF16, reader.raw('q_proj.weight')))


if __name__ == '__main__':
    unittest.main()
