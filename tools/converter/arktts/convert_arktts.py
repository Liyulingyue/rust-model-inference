"""Pack Audio8-TTS-Realistic-Air into native BF16 main / F32 codec GGUFs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch

from tools.converter.utils.gguf import (
    GGML_BF16, GGML_F32, GgufWriter, _read_gguf, gguf_dims, open_safetensors,
)


CONTRACT = json.loads((Path(__file__).resolve().parents[3] / "src/models/arktts/contract.json").read_text())


def validate_main(config, header):
    for name, expected in CONTRACT.items():
        if config.get(name) != expected:
            raise ValueError(f"unsupported config {name}: {config.get(name)!r}, expected {expected!r}")
    shapes = {
        "embeddings.weight": [155776, 896],
        "codebook_embeddings.weight": [40960, 896],
        "fast_embeddings.weight": [4096, 896], "norm.weight": [896],
        "fast_norm.weight": [896], "fast_output.weight": [4096, 896],
    }
    for prefix, count, bias in [("layers", 24, True), ("fast_layers", 4, False)]:
        for layer in range(count):
            p = f"{prefix}.{layer}"
            for suffix, shape in {
                "attention.wqkv.weight": [1152, 896], "attention.wo.weight": [896, 896],
                "attention_norm.weight": [896], "ffn_norm.weight": [896],
                "feed_forward.w1.weight": [4864, 896], "feed_forward.w2.weight": [896, 4864],
                "feed_forward.w3.weight": [4864, 896],
            }.items():
                shapes[f"{p}.{suffix}"] = shape
            if bias:
                shapes[f"{p}.attention.wqkv.bias"] = [1152]
    if set(header) - {"__metadata__"} != set(shapes):
        raise ValueError("main tensor inventory does not match the Realistic Air checkpoint")
    for name, shape in shapes.items():
        if header[name]["dtype"] != "BF16" or header[name]["shape"] != shape:
            raise ValueError(f"invalid BF16 tensor contract: {name}")


def materialize_codec(state):
    """Fold the two published weight-norm storage conventions with PyTorch's kernel."""
    result = {}
    consumed = set()
    for name, value in state.items():
        if name.endswith("parametrizations.weight.original0"):
            prefix = name.removesuffix("parametrizations.weight.original0")
            partner, target = prefix + "parametrizations.weight.original1", prefix + "weight"
        elif name.endswith("weight_g"):
            prefix = name.removesuffix("weight_g")
            partner, target = prefix + "weight_v", prefix + "weight"
        else:
            continue
        if partner not in state or target in state or target in result:
            raise ValueError(f"invalid weight norm pair: {name}")
        g, v = value, state[partner]
        if g.shape != (v.shape[0],) + (1,) * (v.ndim - 1):
            raise ValueError(f"invalid weight norm shape: {name}")
        result[target] = torch._weight_norm(v.float(), g.float(), 0).contiguous()
        consumed.update((name, partner))
    for name, value in state.items():
        if name in consumed:
            continue
        if "parametrizations." in name or name.endswith(("weight_g", "weight_v")):
            raise ValueError(f"unpaired weight norm parameter: {name}")
        if name.endswith(("freqs_cis", "causal_mask")):
            continue
        result[name] = value.detach().float().contiguous()
    if any(not torch.isfinite(v).all() for v in result.values()):
        raise ValueError("codec parameters must be finite")
    return result


def file_chunks(path, offset, length):
    with path.open("rb") as stream:
        stream.seek(offset)
        while length:
            block = stream.read(min(length, 1 << 20))
            if not block:
                raise ValueError(f"truncated tensor in {path}")
            yield block
            length -= len(block)


def verify_main(source, packed):
    _, tensors = _read_gguf(packed)
    with packed.open("rb") as stream:
        for name, info in source.header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            kind, dims, length, offset = tensors[name]
            if (kind, dims, length) != (GGML_BF16, gguf_dims(tuple(info["shape"])), end - start):
                raise ValueError(f"packed tensor contract differs: {name}")
            stream.seek(offset)
            for block in file_chunks(source.path, source.data_offset + start, end - start):
                if stream.read(len(block)) != block:
                    raise ValueError(f"packed BF16 bytes differ: {name}")


def convert(model_dir: Path, out_dir: Path):
    config = json.loads((model_dir / "config.json").read_text())
    source = open_safetensors(model_dir / "model.safetensors")
    validate_main(config, source.header)
    tokenizer = (model_dir / "tokenizer.json").read_text()
    digest = hashlib.sha256()
    for component in ("model.safetensors", "codec.pth"):
        with (model_dir / component).open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    bundle = digest.hexdigest()
    state = torch.load(model_dir / "codec.pth", map_location="cpu", weights_only=True, mmap=True)
    state = state.get("state_dict", state)
    if any("generator." in name for name in state):
        state = {name.replace("generator.", ""): value for name, value in state.items() if "generator." in name}
    codec = materialize_codec(state)
    for name, shape in {
        "encoder.block.0.conv.weight": (64, 1, 7),
        "decoder.model.0.conv.weight": (1536, 1024, 7),
        "quantizer.semantic_quantizer.quantizers.0.codebook.weight": (4096, 8),
        "quantizer.quantizer.quantizers.0.codebook.weight": (1024, 8),
        "quantizer.post_module.layers.0.attention.wqkv.weight": (2048, 1024),
    }.items():
        if name not in codec or tuple(codec[name].shape) != shape:
            raise ValueError(f"invalid codec tensor contract: {name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    main_path, codec_path = [out_dir / f"Audio8-TTS-Realistic-Air-{suffix}.gguf" for suffix in ("BF16", "codec-F32")]
    if main_path.exists() or codec_path.exists():
        raise FileExistsError("output already exists")
    with tempfile.TemporaryDirectory(prefix=".arktts-", dir=out_dir) as temporary:
        temporary = Path(temporary)
        main = GgufWriter(temporary / main_path.name)
        main.add_meta("general.architecture", "arktts")
        main.add_meta("general.name", "Audio8-TTS-Realistic-Air")
        main.add_meta("arktts.bundle_id", bundle)
        main.add_meta("arktts.config_json", json.dumps(config, sort_keys=True))
        main.add_meta("arktts.tokenizer_json", tokenizer)
        for name, info in sorted(source.header.items()):
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            main.add_tensor_chunks(name, GGML_BF16, gguf_dims(tuple(info["shape"])), end - start,
                                   lambda a=start, b=end: file_chunks(source.path, source.data_offset + a, b - a))
        main.write()
        verify_main(source, main.path)
        audio = GgufWriter(temporary / codec_path.name)
        audio.add_meta("general.architecture", "arktts_codec")
        audio.add_meta("arktts.bundle_id", bundle)
        audio.add_meta("arktts_codec.config_json", json.dumps(config, sort_keys=True))
        for name, value in sorted(codec.items()):
            audio.add_tensor(name, GGML_F32, gguf_dims(tuple(value.shape)), value.numpy().tobytes())
        audio.write()
        _, tensors = _read_gguf(audio.path)
        with audio.path.open("rb") as stream:
            for name, value in codec.items():
                stream.seek(tensors[name][3])
                if stream.read(value.numel() * 4) != value.numpy().tobytes():
                    raise ValueError(f"packed F32 codec bytes differ: {name}")
        os.link(main.path, main_path)
        try:
            os.link(audio.path, codec_path)
        except Exception:
            main_path.unlink()
            raise
    print(f"verified {len(source.header) - 1} BF16 main tensors / {len(codec)} F32 codec tensors")
    return main_path, codec_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    print("\n".join(map(str, convert(args.model_dir, args.out_dir or args.model_dir))))
