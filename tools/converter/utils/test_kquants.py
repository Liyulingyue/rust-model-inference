import hashlib

import numpy as np
import pytest

from tools.converter.utils import kquants


def _values():
    raw = np.arange(4096, dtype=np.uint32) * np.uint32(1664525) + np.uint32(1013904223)
    return ((raw % 65536).astype(np.int32) - 32768).astype(np.float32) / np.float32(32768)


def _cases():
    values = _values()
    amplitude = np.float32(3.7654008865356445)
    subblock_values = amplitude * (
        (np.arange(1, 17, dtype=np.float32) + np.float32(0.5)) / np.float32(128)
    )
    subblock_values[0] = amplitude
    return {
        "uniform": values,
        "zeros": np.zeros(256, dtype=np.float32),
        "positive": np.full(256, 0.25, dtype=np.float32),
        "negative": np.full(256, -0.25, dtype=np.float32),
        "ties": np.tile(np.arange(-8, 8, dtype=np.float32), 16),
        "mixed_scales": np.ldexp(
            values[:256].reshape(16, 16), np.arange(-50, 14, 4)[:, None]
        ).astype(np.float32),
        "strided": values.reshape(8, 512)[:, ::2],
        "ordered_sum": np.full(256, 5.202378273010254, dtype=np.float32),
        "underflow": np.tile(np.arange(-8, 8, dtype=np.float32), 16) * np.float32(1e-8),
        "f32_rounding": np.repeat(subblock_values, 16),
    }


@pytest.mark.parametrize(
    "case,expected",
    [
        ("uniform", "a704ce51118d244a392a8cd538350779696141b4e1dace381750d9b371290dde"),
        ("zeros", "eac542cb3768bfcafeb5638149a6ac05472d39d3a29f6782d0b7f7c56458c5a1"),
        ("positive", "70a56db0e2ddf3a0965d65f44af2a41c397e999950efdaad0b45886fd8863a7c"),
        ("negative", "672f9366171945d37163cb38047bc40efbc26f7d6ed4049d4f99cd5f97b0fb32"),
        ("ties", "30a96641692e82c02b96b7dee8fa981eb58ed46ac2d6581e37fea72f905f6f16"),
        ("mixed_scales", "d6b4f328ff13ca8120a7bcf39e5960c35bef60f083c8467d54f638645fc97526"),
        ("strided", "282feb240268e36921a105d190e237bc8b7c6b2b5b708b356acee37216a83558"),
        ("ordered_sum", "da04e10f6070d07fdec9ddec115e85f824f90e38a93b875a5532b54f40a5eab5"),
        ("underflow", "6711f09adf18aa7fd2f681b1455b0da079da072ca4d765d5cca206b91dfabc79"),
        ("f32_rounding", "f0413e98335ea6c26b7c680b0543390c0cffc4e23ec098dee7b9c1099b4568c0"),
    ],
)
def test_q5_k_preserves_scalar_payload(case, expected):
    """Payload digests of the Q5_K encoder."""
    payload = kquants.quantize_q5_k(_cases()[case])
    assert len(payload) == _cases()[case].size // 256 * 176
    assert hashlib.sha256(payload).hexdigest() == expected


def _decode_q5_k(payload: bytes, count: int) -> np.ndarray:
    """Decode a Q5_K payload the way the runtime does.

    The runtime's ``dequantize_row_q5_k`` is the reference for the layout:
    a 2-byte F16 scale, a 2-byte F16 minimum, twelve bytes of six-bit
    sub-scales and sub-minimums, thirty-two bytes of fifth bits and 128
    bytes of four-bit codes.
    """
    blocks = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 176)
    values = np.zeros((blocks.shape[0], 256), dtype=np.float32)
    for index, block in enumerate(blocks):
        d = np.frombuffer(block[0:2].tobytes(), dtype="<f2")[0].astype(np.float32)
        dmin = np.frombuffer(block[2:4].tobytes(), dtype="<f2")[0].astype(np.float32)
        table = block[4:16]
        qh = block[16:48]
        qs = block[48:176]
        scales = np.zeros(8, dtype=np.float32)
        mins = np.zeros(8, dtype=np.float32)
        for j in range(8):
            if j < 4:
                scales[j] = table[j] & 63
                mins[j] = table[j + 4] & 63
            else:
                scales[j] = (table[j + 4] & 0x0F) | ((table[j - 4] >> 6) << 4)
                mins[j] = (table[j + 4] >> 4) | ((table[j] >> 6) << 4)
        for group in range(4):
            low = qs[group * 32 : (group + 1) * 32]
            for sub in range(2):
                j = group * 2 + sub
                mask = 1 << (2 * group + sub)
                for l in range(32):
                    nibble = (low[l] & 0x0F) if sub == 0 else (low[l] >> 4)
                    code = nibble + (16 if qh[l] & mask else 0)
                    values[index, group * 64 + sub * 32 + l] = (
                        d * scales[j] * code - dmin * mins[j]
                    )
    return values.reshape(-1)[:count]


def test_q5_k_decodes_back_within_the_block_error():
    """The Q5_K payload decodes to the source values within the block error."""
    values = _values()[: 256 * 4]
    payload = kquants.quantize_q5_k(values)
    decoded = _decode_q5_k(payload, values.size)
    error = np.abs(decoded - values)
    # Five bits per value over a Gaussian gives a few percent: the block
    # format cannot beat its own bit budget.
    assert np.sqrt((error**2).mean()) / np.sqrt((values**2).mean()) < 0.06


def test_q5_k_rejects_non_finite_input():
    with pytest.raises(ValueError, match="finite"):
        kquants.quantize_q5_k(np.full(256, np.inf, dtype=np.float32))


def test_q5_k_empty_and_incomplete_blocks():
    assert kquants.quantize_q5_k(np.empty(0, dtype=np.float32)) == b""
    with pytest.raises(ValueError, match="multiple of 256"):
        kquants.quantize_q5_k(np.zeros(257, dtype=np.float32))


@pytest.mark.parametrize(
    "case,expected",
    [
        ("uniform", "33279e3509f8e3dc1ded5d363a8876051feb45e44dfbc74fa6e360e6f33c587d"),
        ("zeros", "9e33403d6e41598a790d339e0efdb72ed9b1700e04655543a888a30a50c9f517"),
        ("positive", "37c7468967c85da6e0fd583aef2b49144fef531850ae6a2f75fc37e544a049d1"),
        ("negative", "dc711fe936c85f713e5d824458d9fdab79539804ca44d2cae1029f07bd9100de"),
        ("ties", "cecdacceec30dda0a4e8a82e004ac4ffccd370f75d38f483b3d62d4b5a311ea0"),
        ("mixed_scales", "afebcce3b9f5ee1d2b3548ecf70285d52190a9696f77edb2e05c78c337b11925"),
        ("strided", "c34a89badcced3ea44bb2944ae795c3ab920904880629db51226dca19176a3d2"),
        ("ordered_sum", "213f156ba9e957ed08898038c572ca4359698019b82ee52c2894beab76673443"),
        ("underflow", "ee8aebeada7b3f4e889df9005e49784b89387b083246a7d35f01fa7f11688795"),
        ("f32_rounding", "7ec8850037a47ecbb18bee6641fd942876f37dc97d4d1cce768ab2d1791ba3f9"),
    ],
)
def test_q6_k_preserves_scalar_payload(case, expected):
    """Payload digests captured from the scalar encoder at e7f0cc8."""
    values = _values()
    amplitude = np.float32(3.7654008865356445)
    subblock_values = amplitude * (
        (np.arange(1, 17, dtype=np.float32) + np.float32(0.5)) / np.float32(128)
    )
    subblock_values[0] = amplitude
    cases = {
        "uniform": values,
        "zeros": np.zeros(256, dtype=np.float32),
        "positive": np.full(256, 0.25, dtype=np.float32),
        "negative": np.full(256, -0.25, dtype=np.float32),
        "ties": np.tile(np.arange(-8, 8, dtype=np.float32), 16),
        "mixed_scales": np.ldexp(
            values[:256].reshape(16, 16), np.arange(-50, 14, 4)[:, None]
        ).astype(np.float32),
        "strided": values.reshape(8, 512)[:, ::2],
        "ordered_sum": np.full(256, 5.202378273010254, dtype=np.float32),
        "underflow": np.tile(np.arange(-8, 8, dtype=np.float32), 16) * np.float32(1e-8),
        "f32_rounding": np.repeat(subblock_values, 16),
    }
    payload = kquants.quantize_q6_k(cases[case])
    assert len(payload) == cases[case].size // 256 * 210
    assert hashlib.sha256(payload).hexdigest() == expected


def test_q6_k_chunking_preserves_bytes_and_reports_progress(monkeypatch):
    values = _values()
    expected = kquants.quantize_q6_k(values)
    monkeypatch.setattr(kquants, "K_SEARCH_BUDGET_BYTES", 3 * kquants._Q6K_BYTES_PER_BLOCK)
    progress = []
    payload = kquants.quantize_k("q6_k", values, lambda done, total: progress.append((done, total)))
    assert payload == expected
    assert progress == [(done, 16) for done in (3, 6, 9, 12, 15, 16)]


def test_q6_k_large_input_reports_intermediate_progress():
    progress = []
    values = np.zeros(1025 * 256, dtype=np.float32)
    payload = kquants.quantize_q6_k(values, lambda done, total: progress.append((done, total)))
    assert payload == bytes(1025 * 210)
    assert len(progress) > 1
    assert progress[0][0] < 1025
    assert progress[-1] == (1025, 1025)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_q6_k_rejects_non_finite_input(value):
    with pytest.raises(ValueError, match="finite"):
        kquants.quantize_q6_k(np.full(256, value, dtype=np.float32))


def test_q6_k_empty_and_incomplete_blocks():
    assert kquants.quantize_q6_k(np.empty(0, dtype=np.float32)) == b""
    with pytest.raises(ValueError, match="multiple of 256"):
        kquants.quantize_q6_k(np.zeros(257, dtype=np.float32))
