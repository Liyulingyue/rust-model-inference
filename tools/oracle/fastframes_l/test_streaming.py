import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from tools.oracle.fastframes_l.streaming_weights import StreamingWeights
from tools.oracle.fastframes_l.effective_weights import activation_qdq, install_prepared_activations, q4_1_sum_delta


class CrossAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k, self.v = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)

    def text_kv(self, x):
        return self.k(x), self.v(x)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.ModuleDict({name: torch.nn.Linear(4, 4) for name in ['q', 'k', 'v']})
        self.cross_attn = CrossAttention()

    def forward(self, x):
        return sum(layer(x) for layer in self.self_attn.values())


class Trunk(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.blocks = torch.nn.ModuleList([block])

    def embed_text(self, x):
        return x


class StreamingTests(unittest.TestCase):
    def test_repeated_blocks_and_direct_text_cache_keep_values_after_offload(self):
        torch.manual_seed(17)
        dit, text = Trunk(Block()), Trunk(torch.nn.Linear(4, 4))
        modules = [dit, torch.nn.Linear(4, 4), torch.nn.Linear(4, 4), text]
        states = [{k: v.clone() for k, v in m.state_dict().items()} for m in modules]
        x = torch.randn(2, 4)
        expected = dit.blocks[0](x)
        expected_kv = dit.blocks[0].cross_attn.text_kv(x)
        expected_text = text.blocks[0](x)
        for module in modules:
            module.to_empty(device='meta')
        weights = StreamingWeights(states, torch.device('cpu'), torch.float32)
        weights.initialize(modules)
        fast = SimpleNamespace(qkv=[None])
        weights.attach_blocks(dit, text, fast)
        for _ in range(2):
            torch.testing.assert_close(dit.blocks[0](x), expected)
            torch.testing.assert_close(text.blocks[0](x), expected_text)
            self.assertTrue(all(p.is_meta for p in dit.blocks[0].parameters()))
            self.assertIsNone(fast.qkv[0])
            cache = weights.text_cache(dit, x)
            torch.testing.assert_close(cache[0]['k'], expected_kv[0])
            torch.testing.assert_close(cache[0]['v'], expected_kv[1])
            self.assertTrue(all(p.is_meta for p in dit.blocks[0].cross_attn.parameters()))

    def test_attach_blocks_accepts_fast_none_for_no_fast_ops_recipes(self):
        torch.manual_seed(17)
        dit, text = Trunk(Block()), Trunk(torch.nn.Linear(4, 4))
        modules = [dit, torch.nn.Linear(4, 4), torch.nn.Linear(4, 4), text]
        states = [{k: v.clone() for k, v in m.state_dict().items()} for m in modules]
        x = torch.randn(2, 4)
        expected = dit.blocks[0](x)
        for module in modules:
            module.to_empty(device='meta')
        weights = StreamingWeights(states, torch.device('cpu'), torch.float32)
        weights.initialize(modules)
        weights.attach_blocks(dit, text, None)
        for _ in range(2):
            torch.testing.assert_close(dit.blocks[0](x), expected)
            self.assertTrue(all(p.is_meta for p in dit.blocks[0].parameters()))

    def test_decoder_value_falls_back_to_pixel_head_prefix(self):
        torch.manual_seed(17)
        decoder = torch.nn.Linear(4, 4)
        # The official FF-S decoder safetensors uses bare keys; the converted
        # GGUF namespace prefixes them with ``pixel_head.``.
        states = [dict(), dict(), {k: v.clone() for k, v in decoder.state_dict().items()}, dict()]
        for module in [torch.nn.Module(), torch.nn.Module(), decoder, torch.nn.Module()]:
            module.to_empty(device='meta')
        weights = StreamingWeights(states, torch.device('cpu'), torch.float32)
        raw = {f'decoder.pixel_head.{name}': value.detach().numpy().copy()
               for name, value in states[2].items()}
        tensors = {name: SimpleNamespace(data=raw[name], tensor_type=0) for name in raw}
        weights.reader, weights.tensors, weights.formats = object(), tensors, {name: 0 for name in tensors}
        import gguf
        original = gguf.quants.dequantize
        gguf.quants.dequantize = lambda data, ttype: data
        try:
            value = weights.value('decoder', 'weight')
        finally:
            gguf.quants.dequantize = original
        torch.testing.assert_close(value, states[2]['weight'])
        self.assertTrue(isinstance(value, torch.Tensor))

    def test_activation_qdq_handles_zero_and_signed_dominant_blocks(self):
        for kind, width in [(8, 32), (2, 32), (3, 32), (12, 256), (13, 256), (14, 256)]:
            x = torch.zeros(2, width)
            torch.testing.assert_close(activation_qdq(x, kind), x)
            x[0, 0], x[0, 1] = -4., 2.
            result = activation_qdq(x, kind)
            self.assertTrue(torch.isfinite(result).all())
            torch.testing.assert_close(result[0, 0], x[0, 0], atol=.002, rtol=0)
            self.assertLess(abs(float(result[0, 1] - x[0, 1])), .04)

    def test_f16_prepared_activations_round_before_accumulation(self):
        x = torch.tensor([[1.0003, -.12345, 65500.]])
        torch.testing.assert_close(activation_qdq(x, 1), x.half().float(), rtol=0, atol=0)
        self.assertFalse(torch.equal(x, activation_qdq(x, 1)))

    def test_activation_quantization_divides_before_rounding_codes(self):
        # This F32 fixture puts lane 7 at a quantization rounding boundary.
        bits = np.array([
            3214565680, 3214511692, 3202573431, 1069867796, 1066599942, 3216800644,
            1066239163, 3212521752, 3215179503, 1059029414, 3196600292, 3214726010,
            3204629104, 3213627556, 3203566570, 3187558922, 1052246757, 1057982311,
            3213083065, 1066797166, 3213484410, 3208379646, 1063682517, 1066476422,
            1066120626, 3213424243, 3215580435, 3209586140, 3198803490, 1051725897,
            3218479713, 3174319929,
        ], dtype=np.uint32)
        values = bits.view(np.float32)
        maximum = np.abs(values).max()
        inverse = np.divide(np.float32(127), maximum)
        codes = np.rint(values * inverse)
        self.assertEqual(codes[7], -74)
        scale = np.divide(maximum, np.float32(127))
        for kind in [2, 3, 8, 12, 13, 14]:
            with self.subTest(kind=kind):
                x = np.pad(values, (0, 224)) if kind in [12, 13, 14] else values
                step = np.float32(1) / inverse if kind in [12, 13, 14] else np.float16(scale).astype(np.float32)
                expected = torch.from_numpy(codes * step)
                actual = activation_qdq(torch.from_numpy(x)[None], kind)[0, :32]
                torch.testing.assert_close(actual, expected, atol=float(step) / 4, rtol=0)
        total = codes.sum()
        expected_delta = np.float16(total * scale).astype(np.float32) - total * np.float16(scale).astype(np.float32)
        torch.testing.assert_close(q4_1_sum_delta(torch.from_numpy(values)[None]),
                                   torch.tensor([[expected_delta]]), atol=2e-6, rtol=0)

    def test_q4_1_minimum_uses_rounded_q8_1_sum(self):
        dit = torch.nn.Module()
        dit.proj = torch.nn.Linear(32, 2, bias=False)
        minimum = np.array([-.7, .25], dtype='<f2')
        with torch.no_grad():
            dit.proj.weight.copy_(torch.from_numpy(minimum.astype('float32'))[:, None])
        packed = np.zeros((2, 20), dtype=np.uint8)
        packed[:, 2:4] = minimum[:, None].view(np.uint8)
        tensor = SimpleNamespace(data=packed, shape=np.array([32, 2]))
        fast = SimpleNamespace(qkv=[], patch_weight=torch.empty(0))
        formats = {'dit.proj.weight': 3, 'dit.patch_embedding.weight': 0}
        x = torch.linspace(-.31, 1.17, 32)[None]
        raw_scale = float(x.abs().max()) / 127.
        integers = np.rint(x.numpy() * np.float32(127. / float(x.abs().max())))
        q8_1_sum = np.float16(integers.sum() * np.float32(raw_scale)).astype('float32')
        expected = torch.from_numpy(minimum.astype('float32') * q8_1_sum)[None]
        original_linear, original_baddbmm = F.linear, torch.baddbmm
        try:
            install_prepared_activations(
                [dit, torch.nn.Module(), torch.nn.Module(), torch.nn.Module()], fast,
                formats, {'dit.proj.weight': tensor},
            )
            torch.testing.assert_close(dit.proj(x), expected, atol=2e-6, rtol=0)
        finally:
            F.linear, torch.baddbmm = original_linear, original_baddbmm


if __name__ == '__main__':
    unittest.main()
