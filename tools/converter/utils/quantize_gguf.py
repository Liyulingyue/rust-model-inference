"""Quantize an F32 GGUF to a smaller GGUF in-place.

The GLiNER family converters emit F32 for byte-exact parity against the
HF reference. The boundary pipeline's ``Weight::from_quantized`` already
supports Q8_0, Q4_0, Q4_K, Q6_K, F16 and BF16 — the only thing the F32
output does not exercise today is that one path. This script reads an F32
GGUF and writes a new one with the same tensors, switching every
boundary-pipeline weight that survives the alignment check to Q8_0. The
runtime's per-tensor ``ggml_type`` dispatch means the file's mixed
F32-encoder / F32-1D / Q8_0-boundary-weights form is one file the loader
picks up unchanged.

What cannot be quantized stays F32:

  * The word embedding table `token_embd.weight` and every 1-D tensor
    (norms, biases). `token_embd` is `[vocab, d]` with a vocab of 128011,
    and GGUF's Q8_0 byte count keys off the first dim, so a
    non-32-aligned vocab cannot be Q8_0 in this layout at all. 1-D
    tensors are read through `load_vec`, which decodes F32 and BF16
    only. Both are the same trade breeze makes
    (``convert_breeze.py:286-329``).
  * A few small output heads in the boundary family land on GGUF dims
    where at least one axis is not a multiple of 32 — e.g. `compat_mix`
    (1 output), `inside_weight` (1 output), `length_projection`
    (3 outputs) on a 768 wide base-v1. Q8_0 blocks every 32 elements
    along each axis, and `TensorInfo::checked_nbytes` rejects the tensor
    when an axis is short, so they stay F32.

Everything else crosses, encoder included: `gliner::compute::encode`
reads the relative-position table and the embedding rows through
`decode_row`, which accepts the block-quantized embedding types, and the
per-layer projections already went through `Weight::from_quantized`.

Quantization is line-by-line ggml's ``quantize_row_q8_0_reference``
(``tools/converter/utils/gguf.py:213``), the same routine the breeze
converter uses. Weight noise is well under 1% per the doc on
``quantize_q8_0``; the boundary pipeline absorbs it through
``ops::kernel::Q8_0``'s dot.

Usage::

    models/.venv/bin/python -m tools.converter.utils.quantize_gguf \\
        models/gliner2.5-base-v1/gliner2.5-base-v1-f32.gguf \\
        models/gliner2.5-base-v1/gliner2.5-base-v1-q8_0.gguf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from tools.converter.utils.gguf import (
    GGML_F32, GGML_Q8_0, _read_gguf, GgufWriter, gguf_dims, quantize_q8_0,
)

Q8_0_BLOCK = 32
#: Tensors the loader reads through an F32-only path even after the encoder
#: gained block-quantized row decoding. `token_embd` is a `[vocab, d]` table and
#: GGUF's Q8_0 byte count keys off the *first* dim, so a vocab of 128011 is
#: never 32-aligned and cannot be Q8_0 in this layout at all — the header
#: would round its size to zero. Same reason breeze keeps its embeddings at
#: full precision (``convert_breeze.py:286-329``).
KEEP_F32_PREFIXES: tuple[str, ...] = ("token_embd.",)


def _pick_target(tensor_name: str, ggml_type: int, dims: tuple[int, ...]) -> int:
    """Decide the GGUF tensor type for `tensor_name` in the output file.

    See the module docstring for the rules. 1-D tensors are read by the
    loader through ``load_vec``, which decodes F32 and BF16 only, so they
    stay as they are; the rest crosses to Q8_0 when every axis aligns.
    """
    if ggml_type != GGML_F32:
        return ggml_type
    if tensor_name.startswith(KEEP_F32_PREFIXES):
        return GGML_F32
    if len(dims) <= 1:
        return GGML_F32
    # Q8_0 blocks every 32 elements along each non-leading axis; both the
    # contiguous dim (GGUF ``dims[0]``) and every other dim must align, or
    # ``TensorInfo::checked_nbytes`` rejects the tensor outright and the
    # loader never sees it.
    for d in dims:
        if int(d) % Q8_0_BLOCK != 0:
            return GGML_F32
    return GGML_Q8_0


def _quantize_one(
    name: str, raw: bytes, source_type: int, dims: tuple[int, ...],
) -> tuple[int, bytes]:
    """Return ``(target_ggml_type, payload_bytes)`` for one tensor."""
    target = _pick_target(name, source_type, dims)
    if target == GGML_F32:
        return target, raw
    if target == GGML_Q8_0:
        # The Rust kernel decodes Q8_0 from F32 weights in a single pass.
        # f32-to-f16 scaling is not safe here: the breeze quantizer emits
        # raw bytes from the fp32 array and treats Q8_0 as a per-block
        # `f16 scale / int8 payload` over the F32 dynamic range, which is
        # what `ggml-quants.c` does too.
        return target, quantize_q8_0(np.frombuffer(raw, dtype="<f4"))


def quantize_gguf(
    source: Path, output: Path, *,
    target_format: str = "Q8_0",
    progress_every: int = 32,
) -> None:
    if target_format != "Q8_0":
        raise ValueError(
            f"only Q8_0 is implemented (got {target_format!r}); K-quants need "
            "their own dispatch and a per-super-block alignment check"
        )
    if output.exists():
        raise FileExistsError(output)
    metadata, tensors = _read_gguf(source)
    # Build the output in-memory first, then atomically rename it into place
    # so a failed quantization leaves the directory untouched.
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    writer = GgufWriter(temporary)
    for key, value in metadata.items():
        if key == "general.file_type":
            writer.add_meta(key, target_format)
        else:
            writer.add_meta(key, value)
    quantised = 0
    kept_f32 = 0
    for index, (name, value) in enumerate(sorted(tensors.items())):
        source_type, dims, length, offset = value
        raw = _tensor_bytes(source, offset, length)
        target, payload = _quantize_one(name, raw, source_type, dims)
        if target != source_type:
            quantised += 1
        else:
            kept_f32 += 1
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
        f"{kept_f32} kept F32",
        file=sys.stderr,
    )


def _tensor_bytes(source: Path, offset: int, length: int) -> bytes:
    with source.open("rb") as stream:
        stream.seek(offset)
        return stream.read(length)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="F32 GGUF to quantize")
    parser.add_argument("output", type=Path, help="destination GGUF path")
    parser.add_argument(
        "--format", default="Q8_0",
        help="target quant format (only Q8_0 is implemented)",
    )
    args = parser.parse_args()
    quantize_gguf(args.source, args.output, target_format=args.format)


if __name__ == "__main__":
    main()