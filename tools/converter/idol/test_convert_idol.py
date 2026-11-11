import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from tools.converter.idol import convert_idol as c
from tools.converter.utils.gguf import Tensor, _read_gguf


def safetensors(path, tensors):
    header, payload = {}, bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        start = len(payload)
        payload.extend(raw)
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[start, len(payload)])
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class TestConversion(unittest.TestCase):
    def test_all_export_formats_roundtrip_and_preserve_integers(self):
        from gguf.quants import dequantize
        from gguf import GGMLQuantizationType
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in c.CONFIGS.values():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
            values = np.sin(np.arange(512, dtype=np.float32) * 0.13).reshape(2, 256)
            for role, relative in c.FILES.items():
                tensors = {'weight': ('F32', [2, 256], values.astype('<f4').tobytes()),
                           'scalar': ('F32', [], struct.pack('<f', 0.125))}
                if role in ('background', 'vae'):
                    tensors['counter'] = ('I64', [], struct.pack('<q', 9007199254740993))
                safetensors(root / relative, tensors)
            paths = set(c.CONFIGS.values()) | set(c.FILES.values())
            files = [dict(Path=relative, Size=(root / relative).stat().st_size,
                          Sha256=hashlib.sha256((root / relative).read_bytes()).hexdigest()) for relative in paths]
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps([
                dict(directory=entry['Path'].split('/', 1)[0],
                     files=[dict(entry, Path=entry['Path'].split('/', 1)[1])])
                for entry in files]))
            with patch.object(c, 'MANIFEST', manifest):
                for fmt in c.FORMATS:
                    with self.subTest(fmt=fmt), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        output = root / f'{fmt}.gguf'
                        c.convert(root, output, fmt)
                        metadata, tensors = _read_gguf(output)
                        self.assertEqual(metadata['general.architecture'], 'idol')
                        self.assertEqual(metadata['idol.weight_format'], fmt)
                        if fmt == 'original':
                            self.assertNotIn('general.file_type', metadata)
                        self.assertIn('denoiser.scalar', json.loads(metadata['idol.scalar_tensors_json']))
                        kind, dims, size, offset = tensors['denoiser.weight']
                        self.assertEqual(dims, (256, 2))
                        with output.open('rb') as source:
                            source.seek(offset)
                            decoded = dequantize(np.frombuffer(source.read(size), dtype=np.uint8), GGMLQuantizationType(kind)).reshape(2, 256)
                        # Export quality is distinct from native-vs-same-weight implementation error.
                        self.assertLess(np.sqrt(np.mean((decoded - values) ** 2)), 0.09)
                        from types import SimpleNamespace
                        from tools.oracle.idol.gguf_reference import Weights
                        # Exercise the independent decoder without installing PyTorch on the Mac.
                        with patch.dict('sys.modules', torch=SimpleNamespace(from_numpy=lambda x: x)):
                            weights = Weights(output, root)
                            state = weights.load(root / c.FILES['denoiser'])
                            np.testing.assert_array_equal(state['weight'], decoded)
                            self.assertEqual(state['scalar'].shape, ())
                            self.assertEqual(state['scalar'].item(), 0.125)
                            state = weights.load(root / c.FILES['vae'])
                            self.assertEqual(state['counter'].item(), 9007199254740993)
                            self.assertEqual(state['counter'].dtype, np.dtype('int64'))
                            with self.assertRaisesRegex(ValueError, 'unexpected'):
                                weights.load(root / c.FILES['vae'], device='cuda')
                        for scalars in [['vae.counter', 'vae.counter'], ['denoiser.weight'], ['missing']]:
                            bad = dict(metadata, **{'idol.scalar_tensors_json': json.dumps(scalars)})
                            with patch('tools.oracle.idol.gguf_reference._read_gguf', return_value=(bad, tensors)):
                                with self.assertRaisesRegex(ValueError, 'scalar'):
                                    Weights(output, root)
                        for role in ('background', 'vae'):
                            kind, dims, size, offset = tensors[f'{role}.counter']
                            self.assertEqual((kind, dims, size), (27, (1,), 8))
                            with output.open('rb') as source:
                                source.seek(offset)
                                self.assertEqual(struct.unpack('<q', source.read(size))[0], 9007199254740993)
                        before = output.read_bytes()
                        with self.assertRaises(FileExistsError):
                            c.convert(root, output, fmt)
                        self.assertEqual(output.read_bytes(), before)
                with patch.object(c, 'encode', side_effect=ValueError('encoding failed')):
                    with self.assertRaisesRegex(ValueError, 'encoding failed'):
                        c.convert(root, root / 'failed.gguf')
                self.assertFalse((root / 'failed.gguf').exists())
                self.assertEqual(list(root.glob('.failed.gguf.*.tmp')), [])
                (root / c.FILES['denoiser']).write_bytes(b'bad')
                with self.assertRaisesRegex(ValueError, 'denoiser'):
                    c.convert(root, root / 'corrupt.gguf')
                self.assertFalse((root / 'corrupt.gguf').exists())

    def test_float_storage_uses_original_bits_and_rejects_bad_inputs(self):
        for dtype, values in [('F32', np.array([0.5, -1.], dtype='<f4')),
                              ('F16', np.array([0.5, -1.], dtype='<f2')),
                              ('BF16', np.array([0x3f00, 0xbf80], dtype='<u2'))]:
            tensor = Tensor('weight', dtype, (1, 2), values.tobytes())
            self.assertEqual(c.encode(tensor, 'original')[1], tensor.raw)
            self.assertEqual(np.frombuffer(c.encode(tensor, 'f32')[1], dtype='<f4').tolist(), [0.5, -1.])
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            c.encode(Tensor('weight', 'F32', (1, 1), struct.pack('<f', float('nan'))), 'q8_0')
        with self.assertRaisesRegex(ValueError, 'length'):
            c.encode(Tensor('weight', 'F32', (1, 2), b'bad'), 'original')


if __name__ == '__main__':
    unittest.main()
