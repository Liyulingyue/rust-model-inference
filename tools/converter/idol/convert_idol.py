#!/usr/bin/env python3
"""Bundle the six pinned Idol components into one self-contained GGUF, on the inference host."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.converter.utils.gguf import (
    GGML_F32, GGML_F16, GGML_BF16, GGML_I64, GgufWriter,
    Tensor, _read_gguf, _tensor_nbytes, gguf_dims, open_safetensors,
)
from tools.converter.utils.quantize_gguf import BLOCK_FORMATS, FILE_TYPES, _pick_target, quantize_tensor
from tools.oracle.idol.download import verified
from tools.oracle.idol.oracle import sha256

MANIFEST = Path(__file__).resolve().parents[2] / "oracle/idol/manifest.json"
COMMIT = "4702a7f00850e34b6e83a2170cc123c3e9570744"
FILES = {
    "denoiser": "idol/denoiser.safetensors", "octree": "idol/decoder.safetensors",
    "gaussian": "idol/decoder_gs.safetensors", "dino": "dinov3-vith16plus/model.safetensors",
    "vae": "FLUX.2-dev/vae/diffusion_pytorch_model.safetensors",
    "background": "idol/birefnet.safetensors",
}
CONFIGS = {"idol.config_json": "idol/config.json",
           "idol.dino_config_json": "dinov3-vith16plus/config.json",
           "idol.vae_config_json": "FLUX.2-dev/vae/config.json"}
FORMATS = ("original", "f32", *BLOCK_FORMATS)
STORAGE = {"F32": (GGML_F32, 4), "F16": (GGML_F16, 2),
           "BF16": (GGML_BF16, 2), "I64": (GGML_I64, 8)}


def encode(tensor: Tensor, outtype: str):
    if outtype not in FORMATS or tensor.dtype not in STORAGE:
        raise ValueError(f"unsupported Idol storage {tensor.dtype} / {outtype}")
    kind, width = STORAGE[tensor.dtype]
    if any(type(d) is not int or d <= 0 for d in tensor.shape):
        raise ValueError(f"invalid tensor shape {tensor.name}: {tensor.shape}")
    count = int(np.prod(tensor.shape, dtype=object))
    if len(tensor.raw) != count * width:
        raise ValueError(f"invalid tensor length: {tensor.name}")
    if tensor.dtype == "I64":
        return kind, tensor.raw
    if kind == GGML_BF16:
        values = (np.frombuffer(tensor.raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    else:
        values = np.frombuffer(tensor.raw, dtype="<f2" if kind == GGML_F16 else "<f4").astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite Idol tensor {tensor.name}")
    if outtype == "original":
        return kind, tensor.raw
    raw = values.astype("<f4", copy=False).tobytes()
    if outtype == "f32":
        return GGML_F32, raw
    kind, payload = quantize_tensor(tensor.name, raw, GGML_F32, gguf_dims(tensor.shape or (1,)), outtype)
    if kind in (GGML_F16, GGML_BF16):
        decoded = np.frombuffer(payload, dtype="<f2") if kind == GGML_F16 else (
            np.frombuffer(payload, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        if not np.isfinite(decoded).all():
            raise ValueError(f"Idol tensor {tensor.name} overflows {outtype}")
    return kind, payload


def convert(models: Path, output: Path, outtype="original"):
    if outtype not in FORMATS:
        raise ValueError(f"unknown outtype {outtype}")
    if output.exists():
        raise FileExistsError(output)
    entries = {f"{repo['directory']}/{entry['Path']}": entry
               for repo in json.loads(MANIFEST.read_text()) for entry in repo["files"]}
    required = set(FILES.values()) | set(CONFIGS.values())
    for relative in sorted(required):
        if relative not in entries or not verified(models / relative, entries[relative]):
            raise ValueError(f"missing or corrupt pinned Idol source: {relative}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(filename)
    try:
        writer = GgufWriter(temporary)
        writer.add_meta("general.architecture", "idol")
        writer.add_meta("general.name", "Idol 1024px image-to-3D")
        if outtype != "original":
            writer.add_meta("general.file_type", FILE_TYPES[outtype])
        writer.add_meta("general.quantization_version", 2)
        writer.add_meta("idol.format_version", 1)
        writer.add_meta("idol.oracle_commit", COMMIT)
        writer.add_meta("idol.weight_format", outtype)
        writer.add_meta("idol.source_sha256_json", json.dumps({key: entries[key]["Sha256"] for key in sorted(required)}, sort_keys=True))
        for key, relative in CONFIGS.items():
            writer.add_meta(key, json.dumps(json.loads((models / relative).read_text()), sort_keys=True))
        scalars = []
        directory = {}
        for role, relative in FILES.items():
            source = open_safetensors(models / relative)
            for name in sorted(set(source.header) - {"__metadata__"}):
                info = source.header[name]
                target = f"{role}.{name}"
                if not info['shape']:
                    scalars.append(target)
                dims = gguf_dims(tuple(info['shape']) or (1,))
                kind, _ = STORAGE[info['dtype']]
                if kind != GGML_I64 and outtype != 'original':
                    kind = GGML_F32
                    if outtype != 'f32':
                        block, desired = BLOCK_FORMATS[outtype]
                        kind = _pick_target(name, kind, dims, block, desired, frozenset())
                size = _tensor_nbytes(kind, dims)
                def chunks(source=source, name=name, kind=kind):
                    actual, payload = encode(source.get(name), outtype)
                    if actual != kind:
                        raise ValueError(f"Idol export type changed while writing {name}")
                    yield payload
                writer.add_tensor_chunks(target, kind, dims, size, chunks)
                directory[target] = (kind, dims, size)
            print(f"planned {role}: {len(source.header) - ('__metadata__' in source.header)} tensors", file=sys.stderr, flush=True)
        writer.add_meta("idol.scalar_tensors_json", json.dumps(sorted(scalars)))
        writer.write()
        metadata, actual = _read_gguf(temporary)
        if metadata != dict(writer.metadata) or {k: v[:3] for k, v in actual.items()} != directory:
            raise ValueError("Idol GGUF readback differs from export")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    result = dict(path=str(output), outtype=outtype, bytes=output.stat().st_size,
                  sha256=sha256(output), tensors=len(directory))
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outtype", choices=FORMATS, default="original")
    args = parser.parse_args()
    convert(args.models, args.output, args.outtype)


if __name__ == "__main__":
    main()
