"""Unit tests for the LayoutNet converter's naming rule, contract and Q8_0 rule.

The tests that would need the ONNX weights live on the box (`convert_ocria_layout`
reads `tablenet.onnx`); this file covers the parts that are pure contract: the
node -> `.bin` name derivation, the orientation rule, the Q8_0 eligibility rule and
the metadata the GGUF carries. `module_path` is the one piece of logic that broke
twice while it was written (the `/inp/inp.0` and `/unit/unit.0` scopes repeat the
branch name), so it is pinned case by case here.
"""
import unittest

import numpy as np

from tools.converter.ocria.convert_ocria_layout import (
    ARCHITECTURE,
    CONTRACT,
    FORMATS,
    SOURCES,
    config_json,
    module_path,
    quantizable,
    source_pins,
    tensor_name,
)


class FakeNode:
    """The two attributes `tensor_name` reads off an ONNX node."""

    def __init__(self, name, op_type):
        self.name = name
        self.op_type = op_type


class ModulePathTest(unittest.TestCase):
    def test_the_encoder_layer_scopes(self):
        self.assertEqual(module_path('/layers.0/MatMul'), 'enc.layers.0.self_attn')
        self.assertEqual(module_path('/layers.0/out_proj/MatMul'),
                         'enc.layers.0.self_attn.out_proj')
        self.assertEqual(module_path('/layers.5/linear1/MatMul'), 'enc.layers.5.linear1')
        self.assertEqual(module_path('/layers.3/linear2/Add'), 'enc.layers.3.linear2')
        self.assertEqual(module_path('/layers.2/norm1/LayerNormalization'),
                         'enc.layers.2.norm1')

    def test_the_image_branch_scopes(self):
        self.assertEqual(module_path('/cnn/cnn.0/Conv'), 'img.cnn.0')
        # the export repeats the module inside its own scope; the `.bin` does not
        self.assertEqual(module_path('/cnn/cnn.3/cnn.3.0/Conv'), 'img.cnn.3.0')
        self.assertEqual(module_path('/cnn/cnn.5/cnn.5.0/Conv'), 'img.cnn.5.0')
        self.assertEqual(module_path('/unit/unit.0/MatMul'), 'img.unit.0')
        self.assertEqual(module_path('/unit/unit.2/Add'), 'img.unit.2')
        self.assertEqual(module_path('/tok/MatMul'), 'img.tok')
        self.assertEqual(module_path('/norm.1/LayerNormalization'), 'img.norm.1')
        # torch's export de-duplication: the second `out_proj` scope collides
        self.assertEqual(module_path('/out_proj/MatMul'), 'img.attn.0.out_proj')
        self.assertEqual(module_path('/out_proj_1/Add'), 'img.attn.1.out_proj')

    def test_the_head_scopes(self):
        # `/inp/inp.0/MatMul` is `inp.0` in the `.bin`, not `inp.inp.0`
        self.assertEqual(module_path('/inp/inp.0/MatMul'), 'inp.0')
        self.assertEqual(module_path('/inp/inp.2/Add'), 'inp.2')
        for head in ('node', 'qk', 'pair_geo', 'cls'):
            self.assertEqual(module_path(f'/{head}/MatMul'), head)

    def test_an_unnamed_node_is_refused(self):
        with self.assertRaises(ValueError):
            module_path('/MatMul')


class TensorNameTest(unittest.TestCase):
    def test_the_packed_projection_keeps_its_parameter_name(self):
        # `nn.MultiheadAttention` calls it `in_proj_weight` / `in_proj_bias`, and
        # `net.rs` reads those names, so the packed q|k|v projection is the one
        # module whose suffix is not `.weight` / `.bias`.
        node = FakeNode('/layers.0/MatMul', 'MatMul')
        self.assertEqual(tensor_name(node, 'weight'),
                         'enc.layers.0.self_attn.in_proj_weight')
        self.assertEqual(tensor_name(node, 'bias'),
                         'enc.layers.0.self_attn.in_proj_bias')

    def test_every_other_module_uses_weight_and_bias(self):
        self.assertEqual(tensor_name(FakeNode('/layers.0/out_proj/MatMul', 'MatMul'), 'weight'),
                         'enc.layers.0.self_attn.out_proj.weight')
        self.assertEqual(tensor_name(FakeNode('/cnn/cnn.3/cnn.3.0/Conv', 'Conv'), 'bias'),
                         'img.cnn.3.0.bias')
        self.assertEqual(tensor_name(FakeNode('/qk/Add', 'Add'), 'bias'), 'qk.bias')


class ContractTest(unittest.TestCase):
    def test_the_contract_holds_the_110_tensors(self):
        self.assertEqual(len(CONTRACT['tensors']), 110)
        self.assertEqual(sum(int(np.prod(shape)) for shape in CONTRACT['tensors'].values()),
                         4_268_063)

    def test_the_shapes_are_torch_orientation(self):
        tensors = CONTRACT['tensors']
        self.assertEqual(tensors['inp.0.weight'], [256, 66])
        self.assertEqual(tensors['enc.layers.0.self_attn.in_proj_weight'], [768, 256])
        self.assertEqual(tensors['qk.weight'], [480, 256])
        self.assertEqual(tensors['pair_geo.weight'], [5, 15])
        # 4-D convs: `ops::nn::conv2d` takes `[cout, cin, kh, kw]`
        self.assertEqual(tensors['img.cnn.0.weight'], [48, 1, 8, 8])
        self.assertEqual(tensors['img.cnn.4.0.weight'], [64, 48, 3, 3])
        # the four BatchNorm folds the `.bin` does not carry
        for name, channels in (('img.cnn.0.bias', 48), ('img.cnn.3.0.bias', 48),
                               ('img.cnn.4.0.bias', 64), ('img.cnn.5.0.bias', 64)):
            self.assertEqual(tensors[name], [channels])

    def test_the_config_is_the_graph_metadata(self):
        # the loader compares this against `serde_json::to_string` of the contract
        self.assertEqual(config_json(CONTRACT['config']),
                         '{"cfg":{"d":256,"layers":6,"heads":4,"pair_d":48,'
                         '"stack_feats":true,"rule_feats":true,"layout":true,"n_cls":13,'
                         '"img":{"c":64,"xattn":2}},"feat_dim":68}')
        self.assertEqual(CONTRACT['config']['feat_dim'], 68)
        self.assertEqual(CONTRACT['config']['cfg']['img']['xattn'], 2)


class QuantizableTest(unittest.TestCase):
    def test_the_layer_one_rule(self):
        self.assertTrue(quantizable([768, 256]))
        self.assertTrue(quantizable([256, 512]))
        self.assertTrue(quantizable([2, 256]))
        self.assertFalse(quantizable([256, 66]))   # inp.0: 66 is not a block multiple
        self.assertFalse(quantizable([5, 15]))     # pair_geo
        self.assertFalse(quantizable([48]))        # a bias
        # the conv kernels stay F32: they are 4-D and `conv2d` wants them 4-D
        self.assertFalse(quantizable([48, 1, 8, 8]))
        self.assertFalse(quantizable([64, 64, 3, 3]))

    def test_the_loader_accepts_what_this_rule_produces(self):
        # `load_tensor` in `src/models/ocria/layout/mod.rs` quantises when the row
        # (the torch shape's last dim) is a multiple of 32 and the rank is at least 2
        for name, shape in CONTRACT['tensors'].items():
            if quantizable(shape):
                self.assertGreaterEqual(len(shape), 2, name)
                self.assertEqual(shape[-1] % 32, 0, name)


class MetadataTest(unittest.TestCase):
    def test_the_architecture_is_the_one_the_loader_expects(self):
        self.assertEqual(ARCHITECTURE, 'ocria-layout')

    def test_every_source_file_is_pinned(self):
        pins = source_pins()
        self.assertEqual(sorted(pins), ['LICENSE', 'tablenet.bin', 'tablenet.onnx'])
        recorded = {row['Path']: row['Sha256'] for row in SOURCES['files']}
        for name, sha in pins.items():
            self.assertEqual(recorded[name], sha)
        self.assertEqual(pins['tablenet.onnx'],
                         '3b913866067d6e43f73a4e4c9c1eace955b7959447043cc39ea1d30e4c552a95')
        self.assertEqual(pins['tablenet.bin'],
                         '27b5f3ff2d6842d397a21954026b1e6411e8b35085b0b32c616e67e33852d1b4')

    def test_the_formats_are_documented(self):
        self.assertEqual(FORMATS['f32'], 'ocria-layout-f32-v1')
        self.assertEqual(FORMATS['q8_0'], 'ocria-layout-q8_0-v1')


if __name__ == '__main__':
    unittest.main()
