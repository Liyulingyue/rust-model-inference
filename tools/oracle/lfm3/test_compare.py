"""The checkpoint comparator's coverage and quantisation-amplification rules."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.oracle.lfm3.compare_checkpoints import (
    NAMES,
    OPTIONAL_NAMES,
    PROJECTION_INPUTS,
    compare,
)


def fake_quantize(values):
    """A Q8_0-shaped quantiser: one f32 scale and 32 int8 per block.

    It only has to behave like a quantiser for the property under test: a
    small input difference can land on a different byte.
    """
    count = len(values)
    out = bytearray()
    for block in range(count // 32):
        chunk = values[block * 32:(block + 1) * 32]
        amax = max(abs(v) for v in chunk)
        scale = amax / 127.0 if amax else 0.0
        out += struct.pack('<f', scale)
        for value in chunk:
            quantized = int(round(value / scale)) if scale else 0
            out += struct.pack('<b', max(-127, min(127, quantized)))
    return bytes(out)


class CompareCheckpointTests(unittest.TestCase):
    def build(self, root, steps=3, layers=24, gated=False, drift=()):
        """Write a matching Oracle manifest and Rust trace."""
        oracle_lines = []
        rows = []
        for step in range(steps):
            for layer in range(layers):
                for rust_name in ('attn_norm', 'ffn_norm', 'conv_conv'):
                    oracle_name = NAMES[rust_name] + '-' + str(layer)
                    payload = root / f'{step}-{oracle_name}.f32'
                    payload.write_bytes(struct.pack('<32f', *([0.25] * 32)))
                    oracle_lines.append(f'{step}\t{oracle_name}\t32')
                    path = root / f'rust-{step}-{rust_name}-{layer}.f32'
                    path.write_bytes(struct.pack('<32f', *([0.25] * 32)))
                    rows.append(dict(name=rust_name, layer=layer, step=step,
                                     shape=[32], binary_path=str(path)))
                oracle_name = 'l_out-' + str(layer)
                payload = root / f'{step}-{oracle_name}.f32'
                payload.write_bytes(struct.pack('<32f', *([0.25] * 32)))
                oracle_lines.append(f'{step}\t{oracle_name}\t32')
                path = root / f'rust-{step}-l_out-{layer}.f32'
                path.write_bytes(struct.pack('<32f', *([0.25] * 32)))
                rows.append(dict(name='l_out', layer=layer, step=step,
                                 shape=[32], binary_path=str(path)))
            for rust_name, oracle_name in (('embedding', 'model.embed_tokens'),
                                          ('result_norm', 'result_norm'),
                                          ('result_output', 'result_output')):
                payload = root / f'{step}-{oracle_name}.f32'
                payload.write_bytes(struct.pack('<32f', *([0.25] * 32)))
                oracle_lines.append(f'{step}\t{oracle_name}\t32')
                path = root / f'rust-{step}-{rust_name}.f32'
                path.write_bytes(struct.pack('<32f', *([0.25] * 32)))
                rows.append(dict(name=rust_name, layer=None, step=step,
                                 shape=[32], binary_path=str(path)))
            if gated:
                oracle_name = OPTIONAL_NAMES['attn_values'] + '-0'
                payload = root / f'{step}-{oracle_name}.f32'
                payload.write_bytes(struct.pack('<32f', *([0.25] * 32)))
                oracle_lines.append(f'{step}\t{oracle_name}\t32')
                path = root / f'rust-{step}-attn_values-0.f32'
                path.write_bytes(struct.pack('<32f', *([0.25] * 32)))
                rows.append(dict(name='attn_values', layer=0, step=step,
                                 shape=[32], binary_path=str(path)))
        for rust_name, layer, step in drift:
            path = root / f'rust-{step}-{rust_name}-{layer}.f32'
            path.write_bytes(struct.pack('<32f', *([0.25 + 1e-4] * 32)))
        (root / 'checkpoints.tsv').write_text('\n'.join(oracle_lines))
        (root / 'rust.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
        return root

    def test_reports_every_projection_input_and_the_missing_oracle_ones(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.build(Path(directory))
            result = compare(root, quantize=fake_quantize)
            self.assertGreater(result['compared'], 200)
            self.assertEqual(result['exact_embeddings'], 3)
            # Nothing drifted, so no checkpoint re-quantises differently.
            self.assertEqual(result['q8_amplification_points'], [])
            self.assertIsNone(result['first_norm_q8_change'])
            # The projection inputs the Oracle exposes are available; the
            # gated output and the SwiGLU output are not.
            self.assertEqual(result['projection_inputs_available'],
                             ['attn_norm', 'conv_conv', 'ffn_norm', 'result_norm'])
            self.assertEqual(
                result['projection_inputs_without_oracle_checkpoint'],
                ['attn_values', 'ffn_swiglu'],
            )

    def test_a_drifted_projection_input_is_reported_as_amplified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.build(Path(directory), drift=[('attn_norm', 3, 0)])
            result = compare(root, quantize=fake_quantize)
            points = result['q8_amplification_points']
            self.assertEqual(len(points), 1)
            self.assertEqual(points[0]['checkpoint'], 'attn_norm')
            self.assertEqual(points[0]['layer'], 3)
            self.assertGreater(points[0]['quantized_bytes_differ'], 0)
            self.assertEqual(result['first_norm_q8_change'], points[0])
            self.assertIn('attn_norm', result['projection_inputs_available'])

    def test_the_gated_output_is_probed_once_the_oracle_exposes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.build(Path(directory), gated=True,
                              drift=[('attn_values', 0, 0)])
            result = compare(root, quantize=fake_quantize)
            self.assertIn('attn_values', result['projection_inputs_available'])
            points = [item for item in result['q8_amplification_points']
                      if item['checkpoint'] == 'attn_values']
            self.assertEqual(len(points), 1)
            self.assertNotIn('attn_values',
                             result['projection_inputs_without_oracle_checkpoint'])

    def test_the_gated_output_is_skipped_when_the_oracle_lacks_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.build(Path(directory))
            result = compare(root, quantize=fake_quantize)
            self.assertIn('attn_values',
                          result['projection_inputs_without_oracle_checkpoint'])


if __name__ == '__main__':
    unittest.main()
