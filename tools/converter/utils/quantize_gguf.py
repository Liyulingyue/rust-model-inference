"""Quantize an F32 or BF16 GGUF to a smaller GGUF.

The GLiNER family converters emit F32 for byte-exact parity against the
HF reference, and the Audio8 converter emits BF16 because that is what the
merged checkpoint ships. The runtime already supports Q8_0, Q4_0, Q4_K,
Q6_K, F16 and BF16 through ``Weight::from_quantized`` and the
block-quantized row decoder in ``gliner::compute::decode_row`` — the F32
output just never exercises those paths. This script reads an F32 or BF16
GGUF and writes a new one with the same tensor names, switching every
weight whose row width is block-aligned to the requested format. The
runtime's per-tensor ``ggml_type`` dispatch means the resulting mix (F32
norms, F32 small heads, quantized projections) is one file the loader
picks up unchanged.

Formats: ``q8_0`` (32 values per block), ``q4_k`` and ``q6_k`` (256 per
super-block). The encoders are line-by-line ports of ggml's
``quantize_row_q8_0_ref`` (``utils/gguf.py:213``) and
``quantize_row_q4_K_ref`` / ``quantize_row_q6_K_ref``
(``utils/kquants.py``), so the weights decode the way llama.cpp's would.

What cannot be quantized stays in its source type:

  * Every 1-D tensor (norms, biases). They are read through ``load_vec``,
    which decodes F32 and BF16 only.
  * Every 3-D tensor. A convolution kernel is not a projection, and the
    reader for one wants a specific float type: Audio8's audio tower
    decodes its BF16 convolutions by hand.
  * Tensors whose row width is not a multiple of the block size. The
    alignment is on ``dims[0]`` alone: GGUF stores a ``[n_out, n_in]``
    linear as ``[n_in, n_out]``, so the leading dim is the contiguous one
    the block format walks, and ``TensorInfo::checked_nbytes`` only checks
    it. ``token_embd`` shows why the distinction matters — its row is 768
    wide (aligned) while its row count is the 128011-entry vocab (not, and
    irrelevant). The small boundary heads are the other side of it:
    ``compat_mix`` (1 wide), ``inside_weight`` (1), ``length_projection``
    (3) are all under 32 and cannot move at any format here.
  * Tensors the caller names with ``--keep``: a loader that reads a tensor
    as a float vector cannot use a block format for it, whatever its
    width. Audio8's ``frame_len_embedding`` is the case that needs it — it
    is two-dimensional but the runtime reads it through ``load_f32_tensor``.

Usage::

    models/.venv/bin/python -m tools.converter.utils.quantize_gguf \\
        models/gliner2.5-base-v1/gliner2.5-base-v1-f32.gguf \\
        models/gliner2.5-base-v1/gliner2.5-base-v1-q4_k.gguf --format q4_k

    ./.venv/bin/python -m tools.converter.utils.quantize_gguf \\
        models/Audio8-ASR-Infinite/Audio8-ASR-Infinite-BF16.gguf \\
        models/Audio8-ASR-Infinite/Audio8-ASR-Infinite-Q8_0.gguf \\
        --format q8_0 --keep frame_len_embedding.weight
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from tools.converter.utils.gguf import (
    GGML_BF16, GGML_F16, GGML_F32, GGML_Q4K, GGML_Q6K, GGML_Q8_0, _read_gguf,
    GgufWriter, f32_to_bf16, gguf_dims, quantize_q8_0,
)
from tools.converter.utils.kquants import quantize_k

#: Elements per block, and the GGUF type each format produces.
BLOCK_FORMATS = {
    "q8_0": (32, GGML_Q8_0),
    "q4_k": (256, GGML_Q4K),
    "q6_k": (256, GGML_Q6K),
    "f16": (1, GGML_F16),
    "bf16": (1, GGML_BF16),
}
#: Nothing is excluded by name. The alignment rule below is checked against
#: the row width, not the vocabulary: GGUF stores a `[n_out, n_in]` linear as
#: `[n_in, n_out]`, so `dims[0]` is the contiguous dim and it is the one that
#: has to be block-aligned. For `token_embd` that is the 768-wide row, not
#: the 128011-entry vocab.
KEEP_F32_PREFIXES: tuple[str, ...] = ()


def _decode_f32(raw: bytes, source_type: int) -> np.ndarray:
    """Decode a source payload into F32 values."""
    if source_type == GGML_F32:
        return np.frombuffer(raw, dtype="<f4")
    if source_type == GGML_BF16:
        # BF16 is the top half of an F32, so widening is exact.
        widened = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << np.uint32(16)
        return widened.view(np.float32)
    raise ValueError(f"cannot quantize from GGML type {source_type}")


def _pick_target(tensor_name: str, ggml_type: int, dims: tuple[int, ...],
                 block: int, target: int, keep: frozenset[str]) -> int:
    """Decide the GGUF tensor type for `tensor_name` in the output file.

    See the module docstring for the rules. 1-D tensors are read by the
    loader through ``load_vec``, which decodes F32 and BF16 only, so they
    stay as they are; the rest crosses to `target` when the row width is
    block-aligned, and falls back to the source type when it is not.
    """
    if tensor_name in keep:
        return ggml_type
    if ggml_type not in (GGML_F32, GGML_BF16):
        return ggml_type
    if len(dims) != 2:
        # Only a 2-D matrix is a projection. A 1-D tensor is a norm or a
        # bias, and a 3-D one is a convolution kernel whose reader wants a
        # specific float type — Audio8's BF16 convolutions are the case
        # that makes this visible.
        return ggml_type
    # Only `dims[0]` has to align. GGUF stores a `[n_out, n_in]` linear as
    # `[n_in, n_out]`, so the leading dim is the contiguous one and it is
    # what the block format walks; `dims[1..]` is a row count that
    # `TensorInfo::checked_nbytes` multiplies through without checking.
    # `token_embd` is the case that makes this visible: its row is 768 wide
    # (256-aligned) and its row count is the 128011-entry vocab, which is
    # neither and does not need to be.
    if int(dims[0]) % block != 0:
        return ggml_type
    return target


def _quantize_one(
    name: str, raw: bytes, source_type: int, dims: tuple[int, ...],
    target_format: str, keep: frozenset[str],
) -> tuple[int, bytes]:
    """Return ``(target_ggml_type, payload_bytes)`` for one tensor."""
    block, target_type = BLOCK_FORMATS[target_format]
    target = _pick_target(name, source_type, dims, block, target_type, keep)
    if target == source_type:
        return target, raw
    values = _decode_f32(raw, source_type)
    if target == GGML_Q8_0:
        # Per-block `f16 scale / int8 payload` over the F32 dynamic range,
        # which is what `quantize_row_q8_0_ref` in `ggml-quants.c` does.
        return target, quantize_q8_0(values)
    if target == GGML_F16:
        return target, values.astype("<f2").tobytes()
    if target == GGML_BF16:
        return target, f32_to_bf16(values.tobytes())
    if target in (GGML_Q4K, GGML_Q6K):
        # The k-quant search compares weighted squared errors, so its float
        # accumulation order is load-bearing: `kquants.py` is a line-by-line
        # port of `quantize_row_q4_K_ref` / `quantize_row_q6_K_ref` for that
        # reason, and reordering it here would change which scale wins.
        return target, quantize_k(target_format, values)
    raise AssertionError(f"unreachable target {target}")


def quantize_gguf(
    source: Path, output: Path, *,
    target_format: str = "q8_0",
    keep: tuple[str, ...] = (),
    progress_every: int = 32,
) -> None:
    if target_format not in BLOCK_FORMATS:
        raise ValueError(
            f"unknown target format {target_format!r}; "
            f"expected one of {sorted(BLOCK_FORMATS)}"
        )
    if output.exists():
        raise FileExistsError(output)
    keep_names = frozenset(keep)
    metadata, tensors = _read_gguf(source)
    # Build the output in-memory first, then atomically rename it into place
    # so a failed quantization leaves the directory untouched.
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    writer = GgufWriter(temporary)
    wrote_file_type = False
    for key, value in metadata.items():
        if key == "general.file_type":
            writer.add_meta(key, target_format)
            wrote_file_type = True
        else:
            writer.add_meta(key, value)
    if not wrote_file_type:
        # The format is what a reader needs to pick the right tolerance, so
        # it is recorded whether or not the source carried it.
        writer.add_meta("general.file_type", target_format)
    quantised = 0
    kept = 0
    source_types = {value[0] for value in tensors.values()}
    for index, (name, value) in enumerate(sorted(tensors.items())):
        source_type, dims, length, offset = value
        raw = _tensor_bytes(source, offset, length)
        target, payload = _quantize_one(
            name, raw, source_type, dims, target_format, keep_names
        )
        if target != source_type:
            quantised += 1
        else:
            kept += 1
        # `_read_gguf` returns GGUF-order dims (reversed vs torch); the writer's
        # `gguf_dims` helper reverses again, so pass torch-order to round-trip.
        torch_dims = tuple(reversed(dims))
        writer.add_tensor(name, target, gguf_dims(torch_dims), payload)
        if progress_every and (index + 1) % progress_every == 0:
            print(
                f"  [{index + 1}/{len(tensors)}] {name}: "
                f"{source_type} -> {target}",
                file=sys.stderr,
            )
    writer.write()
    temporary.rename(output)
    print(
        f"{output}: {len(tensors)} tensors, "
        f"{quantised} -> {target_format}, "
        f"{kept} kept {sorted(source_types)}",
        file=sys.stderr,
    )


def _tensor_bytes(source: Path, offset: int, length: int) -> bytes:
    with source.open("rb") as stream:
        stream.seek(offset)
        return stream.read(length)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="F32 or BF16 GGUF to quantize")
    parser.add_argument("output", type=Path, help="destination GGUF path")
    parser.add_argument(
        "--format", default="q8_0", choices=sorted(BLOCK_FORMATS),
        help="target quant format (default: q8_0)",
    )
    parser.add_argument(
        "--keep", action="append", default=[], metavar="TENSOR",
        help="tensor to leave in its source type; repeatable",
    )
    args = parser.parse_args()
    quantize_gguf(
        args.source, args.output, target_format=args.format, keep=tuple(args.keep)
    )


if __name__ == "__main__":
    main()