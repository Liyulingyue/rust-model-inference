"""BF16 -> Q8_0 conversion must survive numpy 2's integer overflow rules."""

import struct
import unittest

import numpy as np

from tools.converter.utils.gguf import bf16_bytes_to_q8_0, quantize_q8_0


def bf16_words(values):
    """Encode float32 values as raw BF16 words (truncating the mantissa)."""
    words = []
    for value in values:
        bits = struct.unpack('<I', struct.pack('<f', value))[0]
        words.append(struct.pack('<H', bits >> 16))
    return b''.join(words)


class Bf16ToQ8Tests(unittest.TestCase):
    def test_bf16_words_decode_to_their_values(self):
        # numpy 2 keeps the uint16 dtype for `<<`, so `bits << 16` overflows to
        # zero and every weight decodes as 0.0 -- which makes the whole block
        # quantize to a zero scale and a zero payload.
        raw = bf16_words([1.0] * 32)
        out = bf16_bytes_to_q8_0(raw)
        self.assertEqual(len(out), 34)
        scale = np.frombuffer(out[:2], dtype=np.float16)[0]
        self.assertAlmostEqual(float(scale), 1.0 / 127.0, places=6)
        payload = np.frombuffer(out[2:], dtype=np.int8)
        self.assertTrue((payload == 127).all(), payload)

    def test_matches_the_f32_path(self):
        values = [0.5, -0.25, 3.0, -7.5] * 8
        raw = bf16_words(values)
        from_bf16 = bf16_bytes_to_q8_0(raw)
        from_f32 = quantize_q8_0(np.asarray(values, dtype=np.float32))
        self.assertEqual(from_bf16, from_f32)


if __name__ == '__main__':
    unittest.main()
