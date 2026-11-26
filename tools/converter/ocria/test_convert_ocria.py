"""Unit tests for the OCRia converter's contract, naming and QDQ arithmetic.

The tests that would need the ONNX weights live in `tools/oracle/ocria/`; this
file covers the parts that are pure contract: the tensor naming rule, the Q8_0
eligibility rule, the metadata the GGUF carries and the ONNX `DequantizeLinear`
arithmetic.
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.converter.ocria.convert_ocria import (
    CONTRACT,
    FORMATS,
    GROUPS,
    SOURCES,
    build_metadata,
    component_name,
    config_json,
    convert,
    dequantize,
    flatten_weight,
    quantizable,
    read_charsets,
    source_files,
    tensor_stem,
)

#: The `ocria.config` string the contract fixes, spelled out in full.
CONFIG = ('{"det":{"thresh":0.55,"merge":0.6,"long_side":1600},'
          '"rec":{"groups":["latin","cjk","rtl","world"],'
          '"visual_rtl":[false,false,true,false]}}')


class NamingTest(unittest.TestCase):
    def test_node_paths_drop_the_leading_slash_the_m_and_the_op_segment(self):
        # The examples the contract spells out, one per graph shape.
        self.assertEqual(tensor_stem('/m/stem/stem.0/Conv'), 'stem.stem.0')
        self.assertEqual(tensor_stem('/m/s4/s4.0/dw/Conv'), 's4.s4.0.dw')
        self.assertEqual(tensor_stem('/body/body.0/pw/Conv'), 'body.body.0.pw')
        self.assertEqual(tensor_stem('/m/mix/mix.0/fc1/Conv'), 'mix.mix.0.fc1')
        self.assertEqual(tensor_stem('/group/Gemm'), 'group')
        self.assertEqual(tensor_stem('/flip/Gemm'), 'flip')
        # A graph whose nodes carry no `/m/` prefix, and a lone op segment.
        self.assertEqual(tensor_stem('/stem/stem.0/Conv'), 'stem.stem.0')
        self.assertEqual(tensor_stem('/m/head/Conv'), 'head')
        self.assertEqual(tensor_stem('/m/lat.3/Conv'), 'lat.3')
        # An interior `m` segment is part of the name, only the first is dropped.
        self.assertEqual(tensor_stem('/m/m/m/Conv'), 'm.m')
        # A node with nothing left would collide with every other graph.
        for bad in ('/m/Conv', '/m', 'Conv'):
            with self.assertRaises(ValueError):
                tensor_stem(bad)

    def test_component_prefix(self):
        self.assertEqual(component_name('det'), 'det')
        self.assertEqual(component_name('router'), 'router')
        self.assertEqual(component_name('rec_latin'), 'rec.latin')
        self.assertEqual(component_name('rec_cjk'), 'rec.cjk')
        self.assertEqual(component_name('rec_rtl'), 'rec.rtl')
        self.assertEqual(component_name('rec_world'), 'rec.world')


class ContractTest(unittest.TestCase):
    def test_every_tensor_name_is_a_weight_or_a_bias_of_a_known_component(self):
        prefixes = ('det.', 'router.') + tuple(f'rec.{group}.' for group in GROUPS)
        for name in CONTRACT['tensors']:
            self.assertTrue(name.startswith(prefixes), name)
            self.assertTrue(name.endswith(('.weight', '.bias')), name)
        # the router's two heads are the only `Gemm` nodes
        self.assertEqual(CONTRACT['tensors']['router.group.weight'], [4, 192])
        self.assertEqual(CONTRACT['tensors']['router.group.bias'], [4])
        self.assertEqual(CONTRACT['tensors']['router.flip.weight'], [2, 192])
        self.assertEqual(CONTRACT['tensors']['router.flip.bias'], [2])

    def test_conv_kernels_are_flattened_to_two_dimensions(self):
        # A pointwise conv keeps its channel pair, a depthwise 3x3 becomes a
        # nine-wide row, and no tensor is four-dimensional.
        self.assertEqual(CONTRACT['tensors']['det.stem.stem.0.weight'], [16, 9])
        self.assertEqual(CONTRACT['tensors']['det.s4.s4.0.dw.weight'], [16, 9])
        self.assertEqual(CONTRACT['tensors']['det.s4.s4.0.pw.weight'], [24, 16])
        self.assertEqual(CONTRACT['tensors']['det.lat.3.weight'], [32, 128])
        self.assertNotIn('det.lat.3.bias', CONTRACT['tensors'])
        self.assertEqual(CONTRACT['tensors']['det.head.head.3.weight'], [1, 16])
        self.assertEqual(CONTRACT['tensors']['router.body.body.0.pw.weight'], [32, 16])
        self.assertEqual(CONTRACT['tensors']['rec.cjk.mix.mix.0.dw.weight'], [192, 9])
        self.assertEqual(CONTRACT['tensors']['rec.cjk.mix.mix.0.fc1.weight'], [384, 192])
        # the four heads: one conv for latin/rtl/world, a 96-wide bottleneck
        # and a second conv for cjk
        self.assertEqual(CONTRACT['tensors']['rec.latin.head.weight'], [792, 192])
        self.assertEqual(CONTRACT['tensors']['rec.cjk.head.head.0.weight'], [96, 192])
        self.assertEqual(CONTRACT['tensors']['rec.cjk.head.head.1.weight'], [6214, 96])
        self.assertEqual(CONTRACT['tensors']['rec.rtl.head.weight'], [324, 192])
        self.assertEqual(CONTRACT['tensors']['rec.world.head.weight'], [1712, 192])
        for name, shape in CONTRACT['tensors'].items():
            self.assertIn(len(shape), (1, 2), name)

    def test_the_contract_pins_the_tensor_count_of_each_component(self):
        counts = {}
        for name in CONTRACT['tensors']:
            head = name.split('.')
            key = '.'.join(head[:2]) if head[0] == 'rec' else head[0]
            counts[key] = counts.get(key, 0) + 1
        # one weight per Conv/Gemm plus a bias wherever the node has a third
        # input: det 39, router 15 + 2 Gemm, rec.latin 29, rec.cjk 30,
        # rec.rtl 29, rec.world 29 convolutions
        self.assertEqual(counts, {'det': 74, 'router': 34, 'rec.latin': 58,
                                  'rec.cjk': 60, 'rec.rtl': 58, 'rec.world': 58})
        self.assertEqual(sum(counts.values()), len(CONTRACT['tensors']))


class FlattenTest(unittest.TestCase):
    def test_the_flatten_is_c_order_over_the_kernel_axes(self):
        # a depthwise 3x3 kernel: row `o`, column `ky*3 + kx` (one input channel)
        kernel = np.arange(2 * 1 * 3 * 3, dtype=np.float32).reshape(2, 1, 3, 3)
        flat = flatten_weight(kernel, [3, 3])
        self.assertEqual(flat.shape, (2, 9))
        np.testing.assert_array_equal(flat[0], kernel[0].ravel())
        np.testing.assert_array_equal(flat[1], kernel[1].ravel())
        # a pointwise kernel keeps its channel pair, row = output channel
        pointwise = np.arange(3 * 4, dtype=np.float32).reshape(3, 4, 1, 1)
        flat = flatten_weight(pointwise, [1, 1])
        self.assertEqual(flat.shape, (3, 4))
        np.testing.assert_array_equal(flat[1], np.array([4.0, 5.0, 6.0, 7.0]))
        # the 1-D mix convolution: column `c_in * K + k`
        mix = np.arange(2 * 3 * 9, dtype=np.float32).reshape(2, 3, 9)
        flat = flatten_weight(mix, [9])
        self.assertEqual(flat.shape, (2, 27))
        np.testing.assert_array_equal(flat[0], mix[0].ravel())
        # a transposed flatten would be a different matrix, and is not what we store
        square = np.array([[1.0, 2.0], [3.0, 4.0]]).reshape(2, 2, 1, 1)
        self.assertFalse(np.array_equal(flatten_weight(square), square.reshape(2, 2).T))
        # a kernel that does not match the weight is rejected
        with self.assertRaises(ValueError):
            flatten_weight(np.zeros((2, 1, 3, 3)), [1, 1])
        with self.assertRaises(ValueError):
            flatten_weight(np.zeros((2, 3)), [3, 3])


class QuantizableTest(unittest.TestCase):
    def test_only_row_aligned_two_dimensional_matrices_may_be_quantized(self):
        # pointwise convolutions, the mix `fc` convolutions, the head
        # convolutions and the router's `Gemm` matrices
        self.assertTrue(quantizable([192, 192]))
        self.assertTrue(quantizable([96, 32]))
        self.assertTrue(quantizable([4, 192]))
        self.assertTrue(quantizable([1210, 192]))
        # a row that is not a whole number of 32-value blocks
        self.assertFalse(quantizable([24, 16]))
        self.assertFalse(quantizable([16, 48]))
        # the depthwise 3x3 kernels and the 9-tap mix kernels: row length 9
        self.assertFalse(quantizable([32, 9]))
        self.assertFalse(quantizable([192, 9]))
        # biases and anything else one-dimensional
        self.assertFalse(quantizable([192]))
        self.assertFalse(quantizable([4]))
        # nothing three- or four-dimensional: kernels are stored flattened
        self.assertFalse(quantizable([32, 1, 3, 3]))
        self.assertFalse(quantizable([192, 1, 9]))

    def test_the_rule_holds_for_every_tensor_of_the_contract(self):
        for name, shape in CONTRACT['tensors'].items():
            if quantizable(shape):
                self.assertEqual(len(shape), 2, name)
                self.assertEqual(shape[-1] % 32, 0, name)


class MetadataTest(unittest.TestCase):
    def test_the_config_string_is_the_one_the_contract_fixes(self):
        self.assertEqual(config_json(CONTRACT['config']), CONFIG)
        # and the block round-trips to the contract's own object
        self.assertEqual(json.loads(CONFIG), CONTRACT['config'])

    def test_metadata_carries_the_architecture_format_config_and_charsets(self):
        charsets = {group: json.dumps([f'{group}0', f'{group}1']) for group in GROUPS}
        rows = dict(build_metadata(CONTRACT['config'], charsets, SOURCES, 'q8_0', 'license text'))
        self.assertEqual(rows['general.architecture'], 'ocria')
        self.assertEqual(rows['ocria.format'], 'ocria-nano-1-q8_0-v1')
        self.assertEqual(rows['ocria.config'], CONFIG)
        for group in GROUPS:
            self.assertEqual(rows[f'ocria.rec.{group}.charset'], charsets[group])
        self.assertEqual(rows['ocria.source.repo'], 'llamaindex/OCRia-nano-1')
        self.assertEqual(rows['ocria.source.commit'], SOURCES['commit'])
        self.assertEqual(rows['ocria.source.liteparse'],
                         '44ceb6ab160c64cc8b43aab4ed85251719495629')
        hashes = json.loads(rows['ocria.source.files'])
        self.assertEqual(hashes['det.onnx'],
                         '70ea641d74c4d7d3acb734002cc1507350d8350ad44aeedc6f8ac6017dbcff7a')
        self.assertEqual(rows['ocria.licenses'], 'license text')

    def test_the_two_formats_are_named_as_the_loader_expects(self):
        self.assertEqual(FORMATS, {'f32': 'ocria-nano-1-f32-v1',
                                   'q8_0': 'ocria-nano-1-q8_0-v1'})
        charsets = {group: json.dumps(['a']) for group in GROUPS}
        for quant in FORMATS:
            rows = dict(build_metadata(CONTRACT['config'], charsets, SOURCES, quant, ''))
            self.assertEqual(rows['ocria.format'], FORMATS[quant])

    def test_sources_pin_every_file_the_converter_reads(self):
        pinned = {row['Path']: row for row in SOURCES['files']}
        # the converter hashes exactly the files it reads, and every one of
        # them is pinned
        self.assertEqual(sorted(source_files()), sorted(set(source_files())))
        for name in source_files():
            self.assertIn(name, pinned)
        # 6 graphs + det.meta.json + rec_rtl.meta.json + LICENSE + 4 charsets
        self.assertEqual(len(source_files()), 13)
        # the layer-2 table model is downloaded but not read by this converter
        self.assertIn('tablenet.onnx', pinned)
        self.assertNotIn('tablenet.onnx', source_files())
        self.assertIn('tablenet.bin', pinned)
        for graph in ('det', 'router', 'rec_latin', 'rec_cjk', 'rec_rtl', 'rec_world'):
            self.assertIn(f'{graph}.onnx', pinned)
        for group in GROUPS:
            self.assertIn(f'rec_{group}.charset.json', pinned)
        self.assertIn('det.meta.json', pinned)
        self.assertIn('rec_rtl.meta.json', pinned)
        self.assertIn('LICENSE', pinned)
        for row in SOURCES['files']:
            self.assertEqual(len(row['Sha256']), 64, row['Path'])
            self.assertGreater(row['Size'], 0, row['Path'])
            self.assertEqual(len(row['Revision']), 40, row['Path'])


class DequantizeTest(unittest.TestCase):
    def test_per_tensor_quantization_is_the_onnx_qdq_formula(self):
        # int8 weights with a zero point of 0 and a per-tensor scale
        values = np.array([-128, -1, 0, 1, 127], dtype=np.int8)
        result = dequantize(values, np.float32(0.5), np.array(0, dtype=np.int8))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, np.array([-64.0, -0.5, 0.0, 0.5, 63.5]))
        # a uint8 activation scale with a 128 zero point, the shape rec_latin uses
        activation = np.array([0, 128, 255], dtype=np.uint8)
        result = dequantize(activation, np.float32(2.0 / 255.0), np.array(128, dtype=np.uint8))
        np.testing.assert_allclose(result, np.array([-128 * 2.0 / 255, 0.0, 127 * 2.0 / 255]))

    def test_per_axis_scales_broadcast_along_the_channel_axis(self):
        # a conv weight `[out, in, kh, kw]` with one scale per output channel
        values = np.arange(2 * 3 * 1 * 1, dtype=np.int8).reshape(2, 3, 1, 1)
        scale = np.array([0.5, 2.0], dtype=np.float32)
        zero = np.array([0, 1], dtype=np.int8)
        result = dequantize(values, scale, zero, axis=0)
        self.assertEqual(result.shape, values.shape)
        np.testing.assert_array_equal(result[0].ravel(), np.array([0.0, 0.5, 1.0]))
        np.testing.assert_array_equal(result[1].ravel(), np.array([4.0, 6.0, 8.0]))
        # the same scales along axis 1 broadcast across the other axis
        scale = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        result = dequantize(values, scale, np.zeros(3, dtype=np.int8), axis=1)
        np.testing.assert_array_equal(result[1].ravel(), np.array([1.5, 4.0, 10.0]))

    def test_a_missing_zero_point_is_zero(self):
        values = np.array([-2, 0, 2], dtype=np.int8)
        np.testing.assert_array_equal(dequantize(values, np.float32(0.25), None),
                                      np.array([-0.5, 0.0, 0.5]))

    def test_a_mismatched_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            dequantize(np.zeros((4, 3), dtype=np.int8), np.zeros(4, dtype=np.float32),
                       None, axis=0)
        with self.assertRaises(ValueError):
            dequantize(np.zeros((4,), dtype=np.int8), np.zeros((2, 2), dtype=np.float32), None)


class CharsetTest(unittest.TestCase):
    def test_charsets_are_read_as_released_and_visual_rtl_follows_the_meta_files(self):
        config = CONTRACT['config']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for group in GROUPS:
                (root / f'rec_{group}.charset.json').write_text(json.dumps(['a', 'b']))
            # the release ships rec_rtl.meta.json and no other group's
            (root / 'rec_rtl.meta.json').write_text(json.dumps({'visual_rtl': True}))
            self.assertEqual(read_charsets(root, config)['rtl'], '["a", "b"]')
            # a meta file that disagrees with the contract is a contract error
            (root / 'rec_rtl.meta.json').write_text(json.dumps({'visual_rtl': False}))
            with self.assertRaises(ValueError):
                read_charsets(root, config)
            (root / 'rec_rtl.meta.json').write_text(json.dumps({'visual_rtl': True}))
            self.assertEqual(read_charsets(root, config)['rtl'], '["a", "b"]')
            # an empty charset would shift every id
            (root / 'rec_world.charset.json').write_text('[]')
            with self.assertRaises(ValueError):
                read_charsets(root, config)


class ConvertTest(unittest.TestCase):
    def test_invalid_sources_and_existing_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'result.gguf'
            with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
                convert(root, output)
            self.assertFalse(output.exists())
            output.write_bytes(b'preserve me')
            with self.assertRaises(FileExistsError):
                convert(root, output)
            self.assertEqual(output.read_bytes(), b'preserve me')


if __name__ == '__main__':
    unittest.main()
