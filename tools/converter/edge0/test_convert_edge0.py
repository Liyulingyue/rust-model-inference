"""Unit tests for the Edge0 MLX-affine expansion and the multi-precision writer.

The expansion maths is checked against synthetic matrices whose expected values
are derived by hand from the format, and the writer is exercised end to end on
a miniature checkpoint so the ``--quant`` modes are compared without needing the
19 GB Edge0-35B download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.converter.edge0.convert_edge0 import (
    QUANT_MODES,
    TensorEntry,
    emit_packed,
    encoded_nbytes,
    expand_matrix,
    matrix_axes,
    reencode,
)
from tools.converter.edge0.mlx_affine import (
    MLX_GROUP_SIZE,
    bf16_to_f32,
    dequantize_matrix,
)
from tools.converter.utils.gguf import (
    GGML_F16,
    GGML_F32,
    GGML_I32,
    GGML_Q4_0,
    GGML_Q4K,
    GGML_Q6K,
    GGML_Q8_0,
    GgufWriter,
    read_gguf_directory,
    read_gguf_tensor_bytes,
)


def f32_to_bf16_bytes(values: np.ndarray) -> bytes:
    """Truncate F32 to BF16, i.e. keep the top 16 bits of each 32-bit word."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return (bits >> np.uint32(16)).astype("<u2").tobytes()


def make_matrix(n_out: int, n_in: int, bits: int, seed: int):
    """Build a packed affine matrix plus the F32 values it should decode to."""
    rng = np.random.default_rng(seed)
    groups_per_row = n_in // MLX_GROUP_SIZE
    values_per_word = 32 // bits
    q = rng.integers(0, 1 << bits, size=(n_out, n_in), dtype=np.uint32)
    scale = rng.normal(0.0, 0.05, size=n_out * groups_per_row).astype(np.float32)
    bias = rng.normal(0.0, 0.05, size=n_out * groups_per_row).astype(np.float32)

    words = q.reshape(n_out, n_in // values_per_word, values_per_word)
    shifts = (np.arange(values_per_word, dtype=np.uint32) * bits)[None, None, :]
    packed_words = np.bitwise_or.reduce(words << shifts, axis=2)
    packed = packed_words.astype("<u4").tobytes()

    scale_bf16 = f32_to_bf16_bytes(scale)
    bias_bf16 = f32_to_bf16_bytes(bias)
    # Reference values must use the *rounded* BF16 scales, not the originals.
    scale_f32 = bf16_to_f32(scale_bf16)
    bias_f32 = bf16_to_f32(bias_bf16)
    scale_2d = scale_f32.reshape(n_out, groups_per_row).repeat(MLX_GROUP_SIZE, axis=1)
    bias_2d = bias_f32.reshape(n_out, groups_per_row).repeat(MLX_GROUP_SIZE, axis=1)
    expected = (scale_2d * q.astype(np.float32) + bias_2d).astype(np.float32)
    return packed, scale_bf16, bias_bf16, expected


def _entry(
    mapped: str, shape: tuple[int, ...], payload: bytes, tmp_path: Path, dtype: str = "U32"
) -> TensorEntry:
    """A TensorEntry backed by a throwaway file holding exactly ``payload``."""
    path = tmp_path / f"{mapped.replace('.', '_')}.bin"
    path.write_bytes(payload)
    return TensorEntry(mapped, dtype, shape, path, 0, len(payload))


def _affine_triplet(
    tmp_path: Path, stem: str, n_out: int, n_in: int, bits: int, seed: int, experts: int = 1
):
    """Build the weight/scales/biases entries plus the expected F32 values.

    The entry shapes use the real safetensors axis order, so an expert-stacked
    tensor leads with the expert count and each expert owns a contiguous
    ``n_out x groups`` slice of the companions.
    """
    packed, scale_bf16, bias_bf16, expected = make_matrix(n_out, n_in, bits, seed=seed)
    # The last axis holds U32 words, so packed_cols == n_in / values_per_word.
    packed_cols = n_in // (32 // bits)
    groups = n_in // MLX_GROUP_SIZE
    lead = (experts,) if experts > 1 else ()
    weight = _entry(f"{stem}.weight", lead + (n_out, packed_cols), packed * experts, tmp_path)
    scales = _entry(f"{stem}.scales", lead + (n_out, groups), scale_bf16 * experts, tmp_path, "BF16")
    biases = _entry(f"{stem}.biases", lead + (n_out, groups), bias_bf16 * experts, tmp_path, "BF16")
    return weight, scales, biases, expected


def test_bf16_to_f32_matches_manual_shift() -> None:
    raw = f32_to_bf16_bytes(np.array([1.0, -2.5, 0.125], dtype=np.float32))
    assert np.array_equal(bf16_to_f32(raw), np.array([1.0, -2.5, 0.125], dtype=np.float32))


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("shape", [(3, 128), (5, 256)])
def test_dequantize_matrix_matches_hand_derived_values(bits: int, shape: tuple[int, int]) -> None:
    n_out, n_in = shape
    packed, scale_bf16, bias_bf16, expected = make_matrix(n_out, n_in, bits, seed=n_out * 100 + bits)
    row_bytes = (n_in // (32 // bits)) * 4
    got = dequantize_matrix(
        np.frombuffer(packed, dtype=np.uint8).reshape(n_out, row_bytes),
        bf16_to_f32(scale_bf16),
        bf16_to_f32(bias_bf16),
        (n_out, n_in),
        bits,
    )
    assert np.array_equal(got, expected)


def test_dequantize_matrix_rejects_misaligned_width() -> None:
    # n_in = 96 is not a multiple of the 64-value group, so it is rejected
    # before the packed row length is even considered.
    with pytest.raises(ValueError, match="multiple of 64"):
        dequantize_matrix(
            np.zeros((2, 48), dtype=np.uint8),
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            (2, 96),
            4,
        )


def test_dequantize_matrix_rejects_scale_count_mismatch() -> None:
    packed, scale_bf16, bias_bf16, _ = make_matrix(2, 128, 4, seed=1)
    row_bytes = (128 // 8) * 4
    with pytest.raises(ValueError, match="scale/bias count"):
        dequantize_matrix(
            np.frombuffer(packed, dtype=np.uint8).reshape(2, row_bytes),
            bf16_to_f32(scale_bf16)[:-1],
            bf16_to_f32(bias_bf16),
            (2, 128),
            4,
        )


def test_matrix_axes_reads_safetensors_axis_order() -> None:
    # torch axes are (experts?, n_out, packed_cols); the companions end in groups.
    weight = _entry("blk.0.ffn_gate_exps.weight", (256, 512, 256), b"\0" * 16, Path("."))
    scales = TensorEntry("s", "BF16", (256, 512, 32), Path("."), 0, 0)
    assert matrix_axes(weight, scales) == (512, 2048, 256, 4, 256)

    weight = _entry("blk.0.ffn_gate_inp.weight", (256, 512), b"\0" * 16, Path("."))
    scales = TensorEntry("s", "BF16", (256, 32), Path("."), 0, 0)
    # 8-bit routers: packed_cols 512 over 2048 inputs.
    assert matrix_axes(weight, scales) == (256, 2048, 512, 8, 1)


def test_matrix_axes_rejects_row_axis_mismatch() -> None:
    weight = _entry("blk.0.x.weight", (256, 512), b"\0" * 16, Path("."))
    scales = TensorEntry("s", "BF16", (128, 32), Path("."), 0, 0)
    with pytest.raises(ValueError, match="row axis"):
        matrix_axes(weight, scales)


@pytest.mark.parametrize(
    "ggml_type,bytes_per_value",
    [
        (GGML_F32, 4.0), (GGML_F16, 2.0), (GGML_Q8_0, 34 / 32),
        (GGML_Q4_0, 18 / 32), (GGML_Q6K, 210 / 256),
    ],
)
def test_reencode_round_trips_through_the_declared_size(
    ggml_type: int, bytes_per_value: float
) -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(0.0, 1.0, size=256).astype(np.float32)
    payload = reencode(values, ggml_type)
    assert len(payload) == encoded_nbytes(ggml_type, values.size)
    assert len(payload) == pytest.approx(values.size * bytes_per_value, rel=1e-9)


def test_reencode_rejects_unsupported_target() -> None:
    with pytest.raises(ValueError, match="re-encode target"):
        reencode(np.zeros(32, dtype=np.float32), GGML_I32)


def test_emit_packed_streams_the_lossless_bytes_unchanged(tmp_path: Path) -> None:
    n_out, n_in = 4, 128
    weight, scales, biases, _ = _affine_triplet(tmp_path, "blk.0.ffn_gate", n_out, n_in, 4, seed=5)

    out = tmp_path / "lossless.gguf"
    writer = GgufWriter(out)
    emit_packed(writer, weight, scales, biases, None)
    writer.write()

    _, tensors = read_gguf_directory(out)
    assert tensors["blk.0.ffn_gate.weight"][0] == GGML_I32
    assert read_gguf_tensor_bytes(out, "blk.0.ffn_gate.weight") == weight.read()


@pytest.mark.parametrize(
    "ggml_type,decoder",
    [
        (GGML_F32, lambda raw: np.frombuffer(raw, dtype="<f4")),
        (GGML_F16, lambda raw: np.frombuffer(raw, dtype="<f2").astype(np.float32)),
    ],
)
def test_emit_packed_f32_and_f16_reproduce_the_dequantized_values(
    tmp_path: Path, ggml_type: int, decoder
) -> None:
    n_out, n_in = 4, 128
    weight, scales, biases, expected = _affine_triplet(tmp_path, "blk.0.ffn_gate", n_out, n_in, 4, seed=5)

    out = tmp_path / f"out{ggml_type}.gguf"
    writer = GgufWriter(out)
    emit_packed(writer, weight, scales, biases, ggml_type)
    writer.write()

    _, tensors = read_gguf_directory(out)
    name = "blk.0.ffn_gate.weight"
    assert tensors[name][:2] == (ggml_type, (n_in, n_out))
    got = decoder(read_gguf_tensor_bytes(out, name)).reshape(n_out, n_in)
    if ggml_type == GGML_F32:
        assert np.array_equal(got, expected)
    else:
        # F16 is the one lossy hop; the reference is the F32 expansion.
        assert np.allclose(got, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "ggml_type,per_value_bytes",
    [(GGML_Q8_0, 34 / 32), (GGML_Q4_0, 18 / 32), (GGML_Q6K, 210 / 256)],
)
def test_emit_packed_block_quantizes_within_tolerance(
    tmp_path: Path, ggml_type: int, per_value_bytes: float
) -> None:
    n_out, n_in = 4, 256 if ggml_type == GGML_Q6K else 128
    weight, scales, biases, expected = _affine_triplet(tmp_path, "blk.0.ffn_gate", n_out, n_in, 4, seed=5)

    out = tmp_path / f"block{ggml_type}.gguf"
    writer = GgufWriter(out)
    emit_packed(writer, weight, scales, biases, ggml_type)
    writer.write()

    _, tensors = read_gguf_directory(out)
    name = "blk.0.ffn_gate.weight"
    assert tensors[name][:2] == (ggml_type, (n_in, n_out))
    raw = read_gguf_tensor_bytes(out, name)
    assert len(raw) == encoded_nbytes(ggml_type, n_out * n_in)
    # Q4_0 halves the payload versus Q8_0; the F32 expansion is the reference.
    assert len(raw) == pytest.approx(expected.size * per_value_bytes, rel=1e-9)


def test_emit_packed_expert_axis_keeps_the_expert_slice_order(tmp_path: Path) -> None:
    n_out, n_in, experts = 2, 128, 3
    stem = "blk.0.ffn_gate_exps"
    weight, scales, biases, expected = _affine_triplet(tmp_path, stem, n_out, n_in, 4, seed=9, experts=experts)

    out = tmp_path / "experts.gguf"
    writer = GgufWriter(out)
    emit_packed(writer, weight, scales, biases, GGML_F32)
    writer.write()

    _, tensors = read_gguf_directory(out)
    name = f"{stem}.weight"
    assert tensors[name][:2] == (GGML_F32, (n_in, n_out, experts))
    got = np.frombuffer(read_gguf_tensor_bytes(out, name), dtype="<f4").reshape(experts, n_out, n_in)
    for index in range(experts):
        assert np.array_equal(got[index], expected), index



def test_encode_sizes_track_the_ggml_block_contracts() -> None:
    assert encoded_nbytes(GGML_Q8_0, 64) == 2 * 34
    assert encoded_nbytes(GGML_Q4_0, 64) == 2 * 18
    assert encoded_nbytes(GGML_F32, 10) == 40
    assert encoded_nbytes(GGML_F16, 10) == 20
    with pytest.raises(ValueError, match="multiple of 32"):
        encoded_nbytes(GGML_Q8_0, 17)
    with pytest.raises(ValueError, match="multiple of 32"):
        encoded_nbytes(GGML_Q4_0, 17)


def test_quant_modes_cover_lossless_and_the_ggml_targets() -> None:
    assert QUANT_MODES["lossless"] is None
    assert QUANT_MODES["f32"] == GGML_F32
    assert QUANT_MODES["f16"] == GGML_F16
    assert QUANT_MODES["q8_0"] == GGML_Q8_0
    assert QUANT_MODES["q4_0"] == GGML_Q4_0
    assert QUANT_MODES["q4_k"] == GGML_Q4K
    assert QUANT_MODES["q6_k"] == GGML_Q6K
    assert QUANT_MODES["q4_k_m"] is None
    assert set(QUANT_MODES) == {"lossless", "f32", "f16", "q8_0", "q4_0", "q4_k", "q6_k", "q4_k_m"}
