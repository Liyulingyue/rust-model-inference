import unittest
import json
import struct
import tempfile
from pathlib import Path

from tools.converter.flux3.convert_flux3 import add_tokenizer, copy_tensor, dit_shapes, text_shapes, vae_shapes, validate_config
from tools.converter.utils.gguf import GgufWriter, open_safetensors, read_gguf_directory, read_gguf_tensor_bytes


class Flux3ContractTests(unittest.TestCase):
    def test_distilled_inventory_has_both_times_and_distinct_streams(self):
        shapes = dit_shapes()
        self.assertEqual(len(shapes), 359)
        self.assertEqual(shapes['ref_time_in.in_layer.weight'], (3072, 256))
        self.assertEqual(shapes['txt_in.weight'], (3072, 20480))
        self.assertEqual(shapes['content_mode_blocks.image_cond.4.mlp_in.weight'], (18432, 3072))
        self.assertEqual(shapes['single_blocks.27.norm.key_norm.scale'], (128,))

    def test_rejects_base_and_quantized_packages(self):
        for config in [{'model': 'flux-3-image-klein-base'}, {'model': 'flux-3-image-klein', 'quantization': 'nvfp4'}]:
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_component_inventories(self):
        self.assertEqual(len(text_shapes()), 398)
        self.assertEqual(text_shapes()['blk.35.attn_q.weight'], (4096,2560))
        self.assertEqual(len(vae_shapes()), 250)
        self.assertEqual(vae_shapes()['encoder.quant_conv.weight'], (64,64,1,1))

    def test_tokenizer_preserves_string_merge_rules(self):
        tokenizer = dict(normalizer={'type':'NFC'}, model=dict(type='BPE', vocab={'a':0,'b':1}, merges=['a b','Ġ t']), added_tokens=[])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'tokenizer.json').write_text(json.dumps(tokenizer))
            writer = GgufWriter(root/'tokenizer.gguf')
            add_tokenizer(writer, root)
            self.assertEqual(dict(writer.metadata)['tokenizer.ggml.merges'], ['a b','Ġ t'])

    def test_bf16_copy_preserves_payload_and_reverses_only_dimensions(self):
        raw = struct.pack('<6H', 0, 0x8000, 0x3f80, 0xbf80, 0x0001, 0x7f7f)
        header = json.dumps({'matrix': dict(dtype='BF16', shape=[2,3], data_offsets=[0,len(raw)])}).encode()
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'source.safetensors', Path(directory)/'result.gguf'
            source.write_bytes(struct.pack('<Q',len(header))+header+raw)
            writer = GgufWriter(output)
            copy_tensor(writer, open_safetensors(source), 'matrix', 'mapped')
            writer.write()
            self.assertEqual(read_gguf_directory(output)[1]['mapped'][:2], (30,(3,2)))
            self.assertEqual(read_gguf_tensor_bytes(output,'mapped'), raw)


if __name__ == '__main__':
    unittest.main()
