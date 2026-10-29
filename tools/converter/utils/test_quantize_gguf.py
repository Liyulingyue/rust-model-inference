import numpy as np
import pytest

from tools.converter.utils.gguf import (
    GGML_BF16,
    GGML_F32,
    GGML_Q8_0,
    GgufWriter,
    f32_to_bf16,
)
from tools.converter.utils.quantize_gguf import quantize_gguf


def _values(count: int) -> np.ndarray:
    raw = np.arange(count, dtype=np.uint32) * np.uint32(1664525) + np.uint32(1013904223)
    return ((raw % 65536).astype(np.int32) - 32768).astype(np.float32) / np.float32(32768)


def _write_source(path, tensors):
    """Write a small GGUF holding `name -> (type, dims, payload)`."""
    writer = GgufWriter(path)
    writer.add_meta("general.architecture", "test")
    for name, (ggml_type, dims, payload) in tensors.items():
        writer.add_tensor(name, ggml_type, dims, payload)
    writer.write()


def _read(path):
    from tools.converter.utils.gguf import _read_gguf

    return _read_gguf(path)


def test_bf16_source_crosses_to_q8_0_and_decodes_back(tmp_path):
    values = _values(256 * 4)
    pool = _values(4096)
    source = tmp_path / "source.gguf"
    _write_source(
        source,
        {
            "projection": (GGML_BF16, (256, 4), f32_to_bf16(values.tobytes())),
            "norm": (GGML_BF16, (256,), f32_to_bf16(values[:256].tobytes())),
            "conv": (GGML_BF16, (3, 128, 8), f32_to_bf16(pool[: 3 * 128 * 8].tobytes())),
            "kept": (GGML_BF16, (256, 4), f32_to_bf16(values.tobytes())),
        },
    )
    output = tmp_path / "q8_0.gguf"
    quantize_gguf(source, output, target_format="q8_0", keep=("kept",))

    metadata, tensors = _read(output)
    assert metadata["general.file_type"] == "q8_0"
    kinds = {name: value[0] for name, value in tensors.items()}
    assert kinds["projection"] == GGML_Q8_0
    # 1-D vectors, 3-D kernels and kept tensors stay BF16.
    assert kinds["norm"] == GGML_BF16
    assert kinds["conv"] == GGML_BF16
    assert kinds["kept"] == GGML_BF16

    # The Q8_0 payload decodes close to the original values.
    _, _, length, offset = tensors["projection"]
    with output.open("rb") as packed:
        packed.seek(offset)
        payload = packed.read(length)
    blocks = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 34)
    scales = blocks[:, :2].copy().view("<f2").reshape(-1).astype(np.float32)
    quantized = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
    decoded = quantized * scales[:, None]
    assert np.max(np.abs(decoded - values.reshape(-1, 32))) < 0.02


def test_f32_source_still_crosses_to_q8_0(tmp_path):
    values = _values(256 * 4)
    source = tmp_path / "source.gguf"
    _write_source(
        source,
        {
            "projection": (GGML_F32, (256, 4), values.tobytes()),
            "norm": (GGML_F32, (256,), values[:256].tobytes()),
        },
    )
    output = tmp_path / "q8_0.gguf"
    quantize_gguf(source, output, target_format="q8_0")
    _, tensors = _read(output)
    kinds = {name: value[0] for name, value in tensors.items()}
    assert kinds["projection"] == GGML_Q8_0
    assert kinds["norm"] == GGML_F32


def test_unaligned_row_keeps_the_source_type(tmp_path):
    values = _values(3 * 4)
    source = tmp_path / "source.gguf"
    _write_source(source, {"odd": (GGML_BF16, (3, 4), f32_to_bf16(values.tobytes()))})
    output = tmp_path / "q8_0.gguf"
    quantize_gguf(source, output, target_format="q8_0")
    _, tensors = _read(output)
    assert tensors["odd"][0] == GGML_BF16


@pytest.mark.parametrize("fmt", ["q4_k", "q6_k"])
def test_k_quant_sources_round_trip_through_their_own_payload(tmp_path, fmt):
    values = _values(256 * 2)
    source = tmp_path / "source.gguf"
    _write_source(
        source,
        {"projection": (GGML_BF16, (256, 2), f32_to_bf16(values.tobytes()))},
    )
    output = tmp_path / f"{fmt}.gguf"
    quantize_gguf(source, output, target_format=fmt)
    _, tensors = _read(output)
    _, dims, length, _ = tensors["projection"]
    assert dims == (256, 2)
    block = 144 if fmt == "q4_k" else 210
    assert length == block * 2
