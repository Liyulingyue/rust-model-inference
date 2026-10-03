import hashlib

import numpy as np
import pytest

from tools.converter.utils import kquants


def _values():
    raw = np.arange(4096, dtype=np.uint32) * np.uint32(1664525) + np.uint32(1013904223)
    return ((raw % 65536).astype(np.int32) - 32768).astype(np.float32) / np.float32(32768)


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
