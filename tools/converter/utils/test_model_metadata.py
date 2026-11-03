import struct
import unittest

from tools.converter.utils.gguf import MetaScalar, _gguf_meta_value, model_metadata_value


class ModelMetadataTests(unittest.TestCase):
    def test_llama_cpp_scalar_and_array_types(self):
        for value, expected in [(7, 4), (-1, 5), (1e-6, 6), (True, 7)]:
            data = _gguf_meta_value(model_metadata_value(value))
            self.assertEqual(struct.unpack_from('<I', data)[0], expected)
        for value, expected in [([1, 2, 3], 5), ([0.5, 0.5, 0.5], 6)]:
            data = _gguf_meta_value(model_metadata_value(value))
            self.assertEqual(struct.unpack_from('<II', data), (9, expected))

    def test_explicit_types_and_strings_are_preserved(self):
        value = MetaScalar(10, 1 << 40)
        self.assertIs(model_metadata_value(value), value)
        self.assertEqual(model_metadata_value('qwen35'), 'qwen35')
        self.assertEqual(model_metadata_value(['a', 'b']), ['a', 'b'])

    def test_added_tokens_preserve_control_vs_user_defined(self):
        from tools.converter.intern_decision.convert_intern_decision import _build_vocab as intern_vocab
        from tools.converter.physbrain.convert_physbrain import _build_vocab as physbrain_vocab
        tokenizer = dict(model=dict(vocab={'base': 0}), added_tokens=[
            dict(id=1, content='<control>', special=True),
            dict(id=2, content='<ordinary>', special=False),
        ])
        for tokens, types in [intern_vocab({}, tokenizer, 3), physbrain_vocab({}, tokenizer)]:
            self.assertEqual(tokens[:3], ['base', '<control>', '<ordinary>'])
            self.assertEqual(types[:3], [1, 3, 4])


if __name__ == '__main__':
    unittest.main()
