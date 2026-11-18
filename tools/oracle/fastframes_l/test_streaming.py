import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from tools.oracle.fastframes_l.streaming_weights import StreamingWeights
from tools.oracle.fastframes_l.effective_weights import activation_qdq, install_prepared_activations


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
