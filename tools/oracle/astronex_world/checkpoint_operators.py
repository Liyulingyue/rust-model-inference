"""Real checkpoint operator fixtures: raw safetensors -> independent scalar C.

Only embedding lookup, linear projections and Conv3D are compared. Inputs are
serialized F32; approximate activations, normalization and GPU paths are absent.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from scalar import FLAGS, Scalar
from trace_official import CONTRACT_COMMIT, REPO, _guard_reference


def tensor(root, name, rows=None):
    index = root / "model.safetensors.index.json"
    file = root / json.loads(index.read_text())["weight_map"][name] if index.exists() else root / "diffusion_pytorch_model.safetensors"
    with file.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))[name]
        shape, dtype = header["shape"], header["dtype"]
        assert dtype in ("BF16", "F32"), (name, dtype)
        itemsize = 2 if dtype == "BF16" else 4
        start, end = header["data_offsets"]
        assert end - start == int(np.prod(shape)) * itemsize
        assert file.stat().st_size >= 8 + size + end
        if rows is None:
            stream.seek(8 + size + start)
            raw = stream.read(end - start)
        else:
            assert len(shape) == 2 and all(0 <= row < shape[0] for row in rows)
            pieces = []
            for row in rows:
                stream.seek(8 + size + start + row * shape[1] * itemsize)
                pieces.append(stream.read(shape[1] * itemsize))
            raw = b"".join(pieces)
            shape = [len(rows), shape[1]]
        assert len(raw) == int(np.prod(shape)) * itemsize
    value = np.frombuffer(raw, dtype="<f4" if dtype == "F32" else "<u2")
    if dtype == "BF16":
        value = (value.astype("<u4") << 16).view("<f4")
    value = value.reshape(shape).copy()
    assert np.isfinite(value).all(), name
    return value, hashlib.sha256(raw).hexdigest()


def generate(weights, output):
    reference = _guard_reference()
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").unlink(missing_ok=True)
    records = []
    cases = [
        ("main", "model.text_embedding.0", "astronex.text_embedding.0"),
        ("main", "model.time_embedding.0", "astronex.time_embedding.0"),
        ("main", "model.time_projection.1", "astronex.time_projection.1"),
        ("main", "model.blocks.0.self_attn.q", "astronex.blocks.0.self_attn.q"),
        ("main", "model.blocks.0.ffn.0", "astronex.blocks.0.ffn.0"),
        ("main", "model.blocks.29.ffn.2", "astronex.blocks.29.ffn.2"),
        ("main", "model.head.head", "astronex.head.head"),
        ("text_encoder", "encoder.block.0.layer.0.SelfAttention.q", "astronex.text.blocks.0.attn.q"),
        ("text_encoder", "encoder.block.23.layer.1.DenseReluDense.wo", "astronex.text.blocks.23.ffn.fc2"),
    ]

    def emit(kind, role, name, x, y, **metadata):
        number = len(records)
        x.astype("<f4", copy=False).tofile(output / f"{number}.input.f32")
        y.astype("<f4", copy=False).tofile(output / f"{number}.expected.f32")
        records.append(dict(kind=kind, role=role, name=name, input_shape=list(x.shape),
                            output_shape=list(y.shape), input=f"{number}.input.f32",
                            expected=f"{number}.expected.f32", **metadata))

    def inputs(shape):
        values = ((np.arange(int(np.prod(shape)), dtype=np.int64) * 17) % 257 - 128).astype("<f4") / 128.0
        return values.reshape(shape)

    with Scalar(output / "compile") as scalar:
        for role, original, mapped in cases:
            root = weights if role == "main" else weights / role
            w, digest = tensor(root, original + ".weight")
            b, bias_digest = tensor(root, original + ".bias") if role == "main" else (None, None)
            x, y = inputs((2, w.shape[1])), np.empty((2, w.shape[0]), dtype="<f4")
            scalar.lib.linear_f32(x.ctypes.data, w.ctypes.data, None if b is None else b.ctypes.data,
                                  y.ctypes.data, 2, w.shape[1], w.shape[0])
            assert np.isfinite(y).all(), mapped
            emit("linear", "main" if role == "main" else "components", mapped, x, y,
                 bias=mapped + ".bias" if b is not None else None,
                 weight_shape=list(w.shape), source_tensor=original + ".weight",
                 source_sha256=digest, bias_sha256=bias_digest)

        for role, original, mapped, shape, stride in [
            ("main", "model.patch_embedding", "astronex.patch_embedding", (48, 2, 2, 2), (1, 2, 2)),
            ("vae", "post_quant_conv", "astronex.vae.conv2", (48, 2, 2, 2), (1, 1, 1)),
        ]:
            root = weights if role == "main" else weights / role
            w, digest = tensor(root, original + ".weight")
            b, bias_digest = tensor(root, original + ".bias")
            out = [(shape[i + 1] - w.shape[i + 2]) // stride[i] + 1 for i in range(3)]
            x, y = inputs(shape), np.empty((w.shape[0], *out), dtype="<f4")
            spec = (ct.c_size_t * 22)(1, *shape, w.shape[0], *w.shape[2:], *out,
                                      *stride, 0, 0, 0, 1, 1, 1, 1)
            scalar.lib.conv3d(x.ctypes.data, w.ctypes.data, b.ctypes.data, y.ctypes.data, spec)
            assert np.isfinite(y).all(), mapped
            emit("conv3d", "main" if role == "main" else "components", mapped, x, y,
                 bias=mapped + ".bias", weight_shape=list(w.shape), stride=list(stride),
                 source_tensor=original + ".weight", source_sha256=digest, bias_sha256=bias_digest)

    ids = [0, 1, 3, 42, 1000, 256383]
    embedding, digest = tensor(weights / "text_encoder", "shared.weight", rows=ids)
    emit("embedding", "components", "astronex.text.token_embedding", np.zeros((1,), dtype="<f4"),
         embedding, token_ids=ids, source_tensor="shared.weight", source_sha256=digest)
    assert len(records) == 12
    manifest = dict(schema_version=1, official_commit=reference, contract_commit=CONTRACT_COMMIT,
                    compiler_flags=FLAGS, arithmetic="scalar C F32, sequential reductions, no FMA or SIMD",
                    scope="real-weight exact operators, not whole-model parity", cases=records)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {len(records)} real-checkpoint exact-operator cases in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, default=REPO / "models/Astronex-World")
    args = parser.parse_args()
    generate(args.weights, args.output)
