"""MLX affine group quantization: the Edge0 source format and its inverse.

Edge0-35B ships 4-bit (8-bit for the routers) MLX affine group quantized
safetensors: each group of 64 values along the input axis is stored as one
unsigned integer per value plus a BF16 ``scale`` and ``bias``::

    value(row, col) = bf16(scales[group]) * q + bf16(biases[group])
    group           = row * (n_in // 64) + col // 64

with ``q`` packed little-endian into U32 words, ``32 // bits`` values per word
and the least significant field first.  This module reproduces that
dequantization so a converter can re-encode the result into an ordinary GGML
block format instead of passing the packed bytes through unchanged.

The reference implementation is ``MlxAffineKernel::value`` in
``src/ops/kernel/mlx_affine.rs``; the two must stay in lockstep.
"""

from __future__ import annotations

import numpy as np


MLX_GROUP_SIZE = 64
MLX_BITS = (4, 8)


def bf16_to_f32(raw: bytes) -> np.ndarray:
    """Decode a little-endian BF16 buffer into F32.

    BF16 is the top half of an IEEE binary32, so widening is a 16-bit left
    shift reinterpreted as F32.
    """
    words = np.frombuffer(raw, dtype="<u2")
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


def packed_width(bits: int) -> int:
    """Bytes one row of packed ``bits``-wide values occupies."""
    values_per_word = 32 // bits
    if 32 % bits:
        raise ValueError(f"{bits}-bit values do not tile a U32 word")
    return (MLX_GROUP_SIZE // values_per_word) * 4


def dequantize_matrix(
    packed: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    shape: tuple[int, int],
    bits: int,
) -> np.ndarray:
    """Expand packed MLX affine rows to F32 in ``(n_out, n_in)`` layout.

    ``packed`` is a ``(n_out, row_bytes)`` uint8 view of the U32 words,
    ``scales`` and ``biases`` are already widened to F32 and hold
    ``n_out * (n_in // MLX_GROUP_SIZE)`` entries each.
    """
    n_out, n_in = shape
    if n_in % MLX_GROUP_SIZE:
        raise ValueError(f"input width {n_in} is not a multiple of {MLX_GROUP_SIZE}")
    if n_out <= 0:
        raise ValueError(f"invalid output rows {n_out}")
    if bits not in MLX_BITS:
        raise ValueError(f"unsupported MLX affine bit width {bits}")

    values_per_word = 32 // bits
    words_per_row = n_in // values_per_word
    row_bytes = words_per_row * 4
    if packed.shape != (n_out, row_bytes):
        raise ValueError(f"packed shape {packed.shape} != {(n_out, row_bytes)}")
    groups_per_row = n_in // MLX_GROUP_SIZE
    if scales.size != n_out * groups_per_row or biases.size != n_out * groups_per_row:
        raise ValueError("MLX affine scale/bias count does not match the matrix shape")

    words = np.ascontiguousarray(packed).view("<u4").reshape(n_out, words_per_row)
    shifts = (np.arange(values_per_word, dtype=np.uint32) * bits)[None, :]
    mask = np.uint32((1 << bits) - 1)
    # (n_out, words_per_row, values_per_word) -> (n_out, n_in) in packed order.
    quantized = ((words[:, :, None] >> shifts) & mask).reshape(n_out, n_in).astype(np.float32)

    scale = scales.reshape(n_out, groups_per_row).repeat(MLX_GROUP_SIZE, axis=1)
    bias = biases.reshape(n_out, groups_per_row).repeat(MLX_GROUP_SIZE, axis=1)
    return scale * quantized + bias


def is_matrix(name: str) -> bool:
    """True for the ``*.weight`` tensors that carry packed affine codes.

    Edge0 splits each quantized projection into a ``weight`` payload plus a
    matching ``scales``/``biases`` pair, and additionally attaches
    ``lora_A``/``lora_B`` for the low-rank adapters folded in at load time.
    Only ``weight`` is repacked; the adapters stay at source precision.
    """
    return name.rsplit(".", 1)[-1] == "weight"
