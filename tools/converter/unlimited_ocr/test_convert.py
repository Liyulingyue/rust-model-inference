import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from tools.converter.unlimited_ocr.convert import convert, tensor_type
from tools.converter.utils import gguf


class ConverterTests(unittest.TestCase):
    def test_quantization_selects_projections_and_keeps_buffers(self):
        matrix = {'shape': [2, 256], 'dtype': 'BF16'}
        for kind, code in [('bf16', 30), ('f16', 1), ('f32', 0), ('q8_0', 8), ('q4_k', 12), ('q6_k', 14)]:
            self.assertEqual(tensor_type('lm_head.weight', matrix, kind), code)
        for name in ['model.clip_model.position_embedding.weight', 'model.sam_model.pos_embed']:
            self.assertEqual(tensor_type(name, matrix, 'q4_k'), 30)
        self.assertEqual(tensor_type('conv.weight', {'shape': [2, 3, 16, 16], 'dtype':'F16'}, 'q8_0'), 1)
        self.assertEqual(tensor_type('positions', {'shape':[256], 'dtype':'I64'}, 'q8_0'), 27)
        self.assertEqual(tensor_type('small.weight', {'shape':[2, 128], 'dtype':'BF16'}, 'q4_k'), 30)

    def fixture(self, root, finite=True):
        config = json.loads(Path('tests/fixtures/unlimited_ocr_config.json').read_text())
        (root/'config.json').write_text(json.dumps(config))
        (root/'tokenizer.json').write_text('{}')
        values = np.arange(512, dtype='<f4')/512
        if not finite:
            values[0] = np.nan
        raw = gguf.f32_to_bf16(values.tobytes())
        tensors = {
            'lm_head.weight': ('BF16',[2,256],raw),
            'model.layers.0.self_attn.q_proj.weight': ('F32',[2,256],values.tobytes()),
            'model.layers.0.self_attn.k_proj.weight': ('F16',[2,256],values.astype('<f2').tobytes()),
            'model.vision_model.position_embedding.weight': ('F32',[2,256],values.tobytes()),
            'model.vision_model.position_ids': ('I64',[16],np.arange(16,dtype='<i8').tobytes()),
        }
        header, offset, payload = {}, 0, b''
        for name,(dtype,shape,data) in tensors.items():
            header[name] = {'dtype':dtype,'shape':shape,'data_offsets':[offset,offset+len(data)]}
            offset += len(data)
            payload += data
        encoded = json.dumps(header).encode()
        (root/'model.safetensors').write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
        return raw

    def test_streamed_formats_metadata_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self.fixture(root)
            for kind in ['bf16','f16','f32','q8_0','q4_k','q6_k']:
                output = root/(kind+'.gguf')
                convert(root, output, kind)
                meta, tensors = gguf._read_gguf(output)
                self.assertEqual(meta['general.architecture'], 'unlimited-ocr')
                self.assertIn('unlimited_ocr.tokenizer_json', meta)
                code, dims, offset, nbytes = tensors['lm_head.weight']
                self.assertEqual(dims, (256,2))
                self.assertEqual(code, tensor_type('lm_head.weight', {'shape':[2,256],'dtype':'BF16'},kind))
                if kind == 'bf16':
                    self.assertEqual(gguf.read_gguf_tensor_bytes(output, 'lm_head.weight'), raw)
                self.assertEqual(gguf.read_gguf_tensor_bytes(output, 'model.vision_model.position_ids'), np.arange(16,dtype='<i8').tobytes())
                for name,dtype in [('q_proj','F32'),('k_proj','F16')]:
                    key=f'model.layers.0.self_attn.{name}.weight'
                    self.assertEqual(tensors[key][0], tensor_type(key,{'dtype':dtype,'shape':[2,256]},kind))
                if kind.startswith('q'):
                    self.assertEqual(tensors['model.vision_model.position_embedding.weight'][0], 0)
                before = output.read_bytes()
                with self.assertRaises(FileExistsError):
                    convert(root,output,kind)
                self.assertEqual(output.read_bytes(), before)

    def test_failure_removes_temporary_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.fixture(root, finite=False)
            output=root/'bad.gguf'
            with self.assertRaises(ValueError):
                convert(root,output,'q8_0')
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
