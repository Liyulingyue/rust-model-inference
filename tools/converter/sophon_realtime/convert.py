"""Convert the pinned Sophon thinker and its original Qwen3-VL vision tower."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from tools.converter.utils.gguf import (
    GGML_F32,
    GgufWriter,
    _read_gguf,
    _tensor_nbytes,
    gguf_dims,
    model_metadata_value,
    open_safetensors,
)
from tools.converter.utils.quantize_gguf import BLOCK_FORMATS, _quantize_one

K_WORKERS = 8
STREAM_CHUNK_BYTES = 1 << 20

REVISION = "990d6a74bbf15c1765a6f7736fa839f423fd5b59"
FORMATS = ("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q4_k", "q5_k", "q6_k")
FILE_TYPES = {
    "f32": 0,
    "f16": 1,
    "bf16": 32,
    "q8_0": 7,
    "q4_0": 2,
    "q4_1": 3,
    "q4_k": 15,
    "q5_k": 17,
    "q6_k": 18,
}
QUANTIZATION = {
    "quant_method": "compressed-tensors",
    "format": "float-quantized",
    "quantization_status": "compressed",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 8,
                "type": "float",
                "strategy": "channel",
                "symmetric": True,
                "dynamic": False,
            },
            "input_activations": {
                "num_bits": 8,
                "type": "float",
                "strategy": "token",
                "symmetric": True,
                "dynamic": True,
            },
        }
    },
    "ignore": [
        "re:.*visual.*",
        "re:.*audio_tower.*",
        "re:.*multi_modal_projector.*",
        "re:.*lm_head.*",
    ],
}
PINNED_TEXT = dict(
    hidden_size=5120,
    intermediate_size=25600,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=128,
    num_hidden_layers=64,
    vocab_size=151936,
    max_position_embeddings=262144,
)
PINNED_VISION = dict(
    depth=27,
    hidden_size=1152,
    intermediate_size=4304,
    num_heads=16,
    out_hidden_size=5120,
    patch_size=16,
    temporal_patch_size=2,
    spatial_merge_size=2,
    num_position_embeddings=2304,
    in_channels=3,
    deepstack_visual_indexes=[8, 16, 24],
    model_type="qwen3_vl_vision",
    hidden_act="gelu_pytorch_tanh",
    rope_parameters={"rope_theta": 10000.0, "rope_type": "axial"},
)
DTYPES = {"F8_E4M3": 1, "F32": 4, "BF16": 2, "F16": 2}


def decode_fp8(codes: np.ndarray) -> np.ndarray:
    """E4M3FN includes signed zero and finite exponent 15; only 0x7f/ff are NaN."""
    values = np.arange(256, dtype=np.uint16)
    exponent, fraction = (values >> 3) & 15, values & 7
    lut = np.where(
        exponent == 0,
        fraction / 512.0,
        (1.0 + fraction / 8.0) * np.exp2(exponent.astype(np.int16) - 7),
    )
    lut = np.where(values & 128, -lut, lut).astype("<f4")
    if np.any((codes & 127) == 127):
        raise ValueError("nonfinite FP8 source")
    return lut[codes]


def validate_config(config: dict, *, validate_pin: bool = True) -> None:
    t = config["text_config"]
    if (
        config.get("model_type") != "qwen3_vl"
        or t.get("hidden_act") != "silu"
        or t.get("attention_bias") is not False
        or config.get("tie_word_embeddings") is not False
    ):
        raise ValueError("unsupported thinker config")
    rope = t["rope_parameters"]
    if rope != dict(
        rope_theta=5000000,
        mrope_interleaved=True,
        mrope_section=[24, 20, 20],
        rope_type="default",
    ):
        raise ValueError("unsupported thinker RoPE config")
    if config.get("quantization_config") != QUANTIZATION:
        raise ValueError("unsupported source FP8 quantization config")
    if t.get("rms_norm_eps") != 1e-6:
        raise ValueError("unsupported RMS norm epsilon")
    for key in PINNED_TEXT:
        if type(t.get(key)) is not int or t[key] <= 0:
            raise ValueError(f"invalid text config {key}")
    if t["num_attention_heads"] % t["num_key_value_heads"]:
        raise ValueError("invalid grouped attention heads")
    if validate_pin:
        for section, expected in (
            (t, PINNED_TEXT),
            (config["vision_config"], PINNED_VISION),
        ):
            for key, value in expected.items():
                if section.get(key) != value:
                    raise ValueError(
                        f"pinned config {key}: expected {value}, got {section.get(key)}"
                    )


def decoder_specs(config):
    t = config["text_config"]
    h, f, d, v = (
        t[k] for k in ("hidden_size", "intermediate_size", "head_dim", "vocab_size")
    )
    q, kv = t["num_attention_heads"] * d, t["num_key_value_heads"] * d
    yield "language_model.model.embed_tokens.weight", "token_embd.weight", (v, h)
    yield "language_model.lm_head.weight", "output.weight", (v, h)
    yield "language_model.model.norm.weight", "output_norm.weight", (h,)
    suffixes = [
        ("input_layernorm.weight", "attn_norm.weight", (h,)),
        ("post_attention_layernorm.weight", "ffn_norm.weight", (h,)),
        ("self_attn.q_proj.weight", "attn_q.weight", (q, h)),
        ("self_attn.k_proj.weight", "attn_k.weight", (kv, h)),
        ("self_attn.v_proj.weight", "attn_v.weight", (kv, h)),
        ("self_attn.o_proj.weight", "attn_output.weight", (h, q)),
        ("self_attn.q_norm.weight", "attn_q_norm.weight", (d,)),
        ("self_attn.k_norm.weight", "attn_k_norm.weight", (d,)),
        ("mlp.gate_proj.weight", "ffn_gate.weight", (f, h)),
        ("mlp.up_proj.weight", "ffn_up.weight", (f, h)),
        ("mlp.down_proj.weight", "ffn_down.weight", (h, f)),
    ]
    for i in range(t["num_hidden_layers"]):
        for source, target, shape in suffixes:
            yield f"language_model.model.layers.{i}.{source}", f"blk.{i}.{target}", shape


class Reader:
    def __init__(self, root: Path):
        index = root / "model.safetensors.index.json"
        weight_map = (
            json.loads(index.read_text())["weight_map"] if index.is_file() else None
        )
        names = (
            sorted(set(weight_map.values()))
            if weight_map is not None
            else ["model.safetensors"]
        )
        self.entries = {}
        for name in names:
            if Path(name).name != name or not name.endswith(".safetensors"):
                raise ValueError(f"invalid shard name {name}")
            source = open_safetensors((root / name).resolve(strict=True))
            intervals = []
            for key, info in source.header.items():
                if key == "__metadata__":
                    continue
                dtype, shape, offsets = (
                    info["dtype"],
                    info["shape"],
                    info["data_offsets"],
                )
                if (
                    dtype not in DTYPES
                    or not shape
                    or any(type(n) is not int or n <= 0 for n in shape)
                ):
                    raise ValueError(f"{key}: invalid source dtype/shape")
                if len(offsets) != 2 or any(type(n) is not int for n in offsets):
                    raise ValueError(f"{key}: invalid source offsets")
                start, end = offsets
                if (
                    start < 0
                    or end - start != math.prod(shape) * DTYPES[dtype]
                    or source.data_offset + end > source.file_size
                ):
                    raise ValueError(f"{key}: truncated or invalid source payload")
                if key in self.entries:
                    raise ValueError(f"duplicate source tensor {key}")
                if weight_map is not None and weight_map.get(key) != name:
                    raise ValueError(f"{key}: source index mismatch")
                self.entries[key] = (source, info)
                intervals.append((start, end))
            intervals.sort()
            if intervals and (
                intervals[0][0] != 0
                or any(a[1] != b[0] for a, b in zip(intervals, intervals[1:]))
                or intervals[-1][1] + source.data_offset != source.file_size
            ):
                raise ValueError(f"{name}: overlapping, missing or trailing payload")
        if weight_map is not None and set(weight_map) != set(self.entries):
            raise ValueError("source index missing tensors")

    def shape(self, name):
        return tuple(self.entries[name][1]["shape"])

    def dtype(self, name):
        return self.entries[name][1]["dtype"]

    def read(self, name, first=0, rows=None):
        source, info = self.entries[name]
        shape = self.shape(name)
        width = math.prod(shape[1:])
        count = math.prod(shape) if rows is None else rows * width
        size = DTYPES[info["dtype"]]
        with source.path.open("rb") as stream:
            stream.seek(
                source.data_offset + info["data_offsets"][0] + first * width * size
            )
            raw = stream.read(count * size)
        if len(raw) != count * size:
            raise ValueError(f"{name}: truncated source")
        dtype = info["dtype"]
        if dtype == "F8_E4M3":
            values = decode_fp8(np.frombuffer(raw, dtype=np.uint8))
        elif dtype == "BF16":
            values = (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")
        else:
            values = np.frombuffer(
                raw, dtype="<f4" if dtype == "F32" else "<f2"
            ).astype("<f4")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name}: nonfinite source")
        return values.reshape((-1, *shape[1:]))

    def scale(self, name):
        scale = self.read(name + "_scale")
        if not np.all(scale > 0):
            raise ValueError(f"{name}: invalid weight scale")
        return scale

    def f32(self, name):
        values = self.read(name)
        if self.dtype(name) == "F8_E4M3":
            values = values * self.scale(name)
        return values


def validate_inventory(reader, config, *, validate_pin=True):
    expected = set()
    for name, _, shape in decoder_specs(config):
        expected.add(name)
        if name not in reader.entries:
            raise ValueError(f"missing decoder tensor {name}")
        if reader.shape(name) != shape:
            raise ValueError(f"{name}: shape {reader.shape(name)} != {shape}")
        scaled = len(shape) == 2 and ".layers." in name
        if scaled:
            expected.add(name + "_scale")
            if name + "_scale" not in reader.entries:
                raise ValueError(f"missing weight scale {name}")
            if (
                reader.dtype(name) != "F8_E4M3"
                or reader.shape(name + "_scale") != (shape[0], 1)
                or reader.dtype(name + "_scale") not in ("BF16", "F32")
            ):
                raise ValueError(f"{name}: invalid FP8 weight/scale dtype or shape")
            reader.scale(name)
        elif reader.dtype(name) not in ("BF16", "F32"):
            raise ValueError(f"{name}: expected unquantized BF16/F32")
    if validate_pin:
        auxiliary = json.loads(
            Path(__file__).with_name("auxiliary-inventory.json").read_text()
        )
        expected.update(auxiliary)
        for name, _, shape in vision_specs(config):
            if (
                name not in reader.entries
                or reader.shape(name) != shape
                or reader.dtype(name) not in ("BF16", "F32")
            ):
                raise ValueError(f"missing or invalid vision tensor {name}")
        for name, info in auxiliary.items():
            if (
                name not in reader.entries
                or reader.dtype(name) != info["dtype"]
                or reader.shape(name) != tuple(info["shape"])
            ):
                raise ValueError(f"missing or invalid auxiliary tensor {name}")
    if set(reader.entries) != expected:
        raise ValueError(
            f"unknown/missing source tensors: extra={sorted(set(reader.entries)-expected)[:8]}, missing={sorted(expected-set(reader.entries))[:8]}"
        )


def target_type(shape, quant):
    if quant == "f32" or len(shape) != 2:
        return GGML_F32
    block, target = BLOCK_FORMATS[quant]
    return target if shape[-1] % block == 0 else GGML_F32


def _encode_chunk(target, raw, dims, quant, kind):
    actual, encoded = _quantize_one(target, raw, GGML_F32, dims, quant, frozenset())
    if actual != kind:
        raise ValueError("quantizer format mismatch")
    return encoded


def add_streamed(writer, reader, name, target, shape, quant, pool=None):
    kind = target_type(shape, quant)

    def chunks():
        width = math.prod(shape[1:])
        step = max(1, STREAM_CHUNK_BYTES // (width * 4))
        scale = reader.scale(name) if reader.dtype(name) == "F8_E4M3" else None
        pending = deque()
        for first in range(0, shape[0], step):
            count = min(step, shape[0] - first)
            values = reader.read(name, first, count)
            if scale is not None:
                values = values * scale[first : first + count]
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name}: nonfinite scaled source")
            raw = values.astype("<f4").tobytes()
            if kind == GGML_F32:
                yield raw
            elif pool is None:
                yield _encode_chunk(
                    target, raw, gguf_dims((count, *shape[1:])), quant, kind
                )
            else:
                pending.append(
                    pool.submit(
                        _encode_chunk,
                        target,
                        raw,
                        gguf_dims((count, *shape[1:])),
                        quant,
                        kind,
                    )
                )
                if len(pending) >= K_WORKERS * 2:
                    yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()

    dims = gguf_dims(shape)
    writer.add_tensor_chunks(target, kind, dims, _tensor_nbytes(kind, dims), chunks)


def tokenizer_metadata(writer, root, config):
    raw = (root / "tokenizer.json").read_text(encoding="utf-8")
    tok = json.loads(raw)
    if tok["model"]["type"] != "BPE" or tok.get("normalizer") != {"type": "NFC"}:
        raise ValueError("expected source BPE/NFC tokenizer")
    vocab_size = config["text_config"]["vocab_size"]
    tokens = [None] * vocab_size
    types = [5] * vocab_size
    for piece, index in tok["model"]["vocab"].items():
        if (
            type(index) is not int
            or not 0 <= index < vocab_size
            or tokens[index] is not None
        ):
            raise ValueError("invalid tokenizer vocab id")
        tokens[index], types[index] = piece, 1
    seen = set()
    for token in tok["added_tokens"]:
        index = token["id"]
        if (
            type(index) is not int
            or not 0 <= index < vocab_size
            or index in seen
            or tokens[index] not in (None, token["content"])
        ):
            raise ValueError("invalid added token id")
        if any(
            token.get(k, False)
            for k in ("single_word", "lstrip", "rstrip", "normalized")
        ):
            raise ValueError("unsupported added token matching flags")
        seen.add(index)
        tokens[index] = token["content"]
        types[index] = 3 if token["special"] else 4
    tokens = [p if p is not None else f"<|reserved_{i}|>" for i, p in enumerate(tokens)]
    merges = [" ".join(p) if isinstance(p, list) else p for p in tok["model"]["merges"]]
    metadata = {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": types,
        "tokenizer.ggml.merges": merges,
        "tokenizer.ggml.normalizer.nfc": True,
        "tokenizer.ggml.add_bos_token": False,
        "tokenizer.ggml.add_eos_token": False,
        "sophon_realtime.tokenizer_json": raw,
        "tokenizer.chat_template": (root / "chat_template.jinja").read_text(
            encoding="utf-8"
        ),
    }
    for name in ("bos", "eos", "pad"):
        token_id = config["text_config"].get(name + "_token_id")
        if token_id is not None:
            if type(token_id) is not int or not 0 <= token_id < vocab_size:
                raise ValueError("invalid special token id")
            metadata[
                f'tokenizer.ggml.{"padding" if name=="pad" else name}_token_id'
            ] = token_id
    for key, value in metadata.items():
        writer.add_meta(key, model_metadata_value(value))


def common_metadata(writer, config, component):
    writer.add_meta("sophon_realtime.component", component)
    writer.add_meta("sophon_realtime.source_revision", REVISION)
    writer.add_meta(
        "sophon_realtime.config_json",
        json.dumps(config, separators=(",", ":"), ensure_ascii=False),
    )
    writer.add_meta(
        "sophon_realtime.source_precision",
        "BF16" if component == "audio" else "channel-scaled E4M3FN + BF16",
    )
    writer.add_meta(
        "sophon_realtime.runtime_quantization",
        (
            "GGUF weight-only"
            if component == "audio"
            else "GGUF weight-only; source dynamic per-token FP8 activation quantization is not reproduced"
        ),
    )


def text_writer(path, reader, root, config, quant, pool=None):
    writer = GgufWriter(path)
    writer.add_meta("general.architecture", "sophon_realtime")
    writer.add_meta("general.name", "Sophon-Realtime-32B-Tag thinker")
    writer.add_meta("general.file_type", FILE_TYPES[quant])
    common_metadata(writer, config, "thinker")
    t = config["text_config"]
    rope = t["rope_parameters"]
    metadata = {
        "block_count": t["num_hidden_layers"],
        "context_length": t["max_position_embeddings"],
        "embedding_length": t["hidden_size"],
        "feed_forward_length": t["intermediate_size"],
        "attention.head_count": t["num_attention_heads"],
        "attention.head_count_kv": t["num_key_value_heads"],
        "attention.key_length": t["head_dim"],
        "attention.value_length": t["head_dim"],
        "attention.layer_norm_rms_epsilon": t["rms_norm_eps"],
        "rope.dimension_count": t["head_dim"],
        "rope.freq_base": float(rope["rope_theta"]),
        "rope.dimension_sections": rope["mrope_section"] + [0],
        "n_deepstack_layers": len(
            config["vision_config"].get("deepstack_visual_indexes", [])
        ),
    }
    for key, value in metadata.items():
        writer.add_meta("qwen3vl." + key, model_metadata_value(value))
    tokenizer_metadata(writer, root, config)
    for name, target, shape in decoder_specs(config):
        add_streamed(writer, reader, name, target, shape, quant, pool)
    return writer


def vision_specs(config):
    v = config["vision_config"]
    h = v["hidden_size"]
    f = v["intermediate_size"]
    m = h * v["spatial_merge_size"] ** 2
    o = v["out_hidden_size"]
    p = v["patch_size"]
    yield "model.visual.patch_embed.proj.weight", "patch", (
        h,
        v["in_channels"],
        v["temporal_patch_size"],
        p,
        p,
    )
    yield "model.visual.patch_embed.proj.bias", "v.patch_embd.bias", (h,)
    yield "model.visual.pos_embed.weight", "v.position_embd.weight", (
        v["num_position_embeddings"],
        h,
    )
    for i in range(v["depth"]):
        for source, target, shape in [
            ("norm1", "ln1", (h,)),
            ("norm2", "ln2", (h,)),
            ("attn.qkv", "attn_qkv", (3 * h, h)),
            ("attn.proj", "attn_out", (h, h)),
            ("mlp.linear_fc1", "ffn_up", (f, h)),
            ("mlp.linear_fc2", "ffn_down", (h, f)),
        ]:
            for suffix in ("weight", "bias"):
                yield f"model.visual.blocks.{i}.{source}.{suffix}", f"v.blk.{i}.{target}.{suffix}", (
                    shape if suffix == "weight" else (shape[0],)
                )
    for source, target, norm in [("merger", "mm", h)] + [
        (f"deepstack_merger_list.{i}", f"v.deepstack.{layer}", m)
        for i, layer in enumerate(v["deepstack_visual_indexes"])
    ]:
        for src, dst, shape in [
            ("norm", "norm" if target != "mm" else "post_ln", (norm,)),
            ("linear_fc1", "fc1" if target != "mm" else "0", (m, m)),
            ("linear_fc2", "fc2" if target != "mm" else "2", (o, m)),
        ]:
            for suffix in ("weight", "bias"):
                yield f"model.visual.{source}.{src}.{suffix}", (
                    f"v.{dst}.{suffix}"
                    if target == "mm" and src == "norm"
                    else f"{target}.{dst}.{suffix}"
                ), (shape if suffix == "weight" else (shape[0],))


def vision_writer(path, reader, root, config):
    v = config["vision_config"]
    writer = GgufWriter(path)
    pre = json.loads((root / "preprocessor_config.json").read_text())
    size = pre.get("size", {})
    minimum = pre.get("min_pixels", size.get("shortest_edge"))
    maximum = pre.get("max_pixels", size.get("longest_edge"))
    if minimum is None or maximum is None or not 0 < minimum <= maximum:
        raise ValueError("missing/invalid source image pixel budgets")
    for key in ("patch_size", "temporal_patch_size", "merge_size"):
        expected = v["spatial_merge_size" if key == "merge_size" else key]
        if key in pre and pre[key] != expected:
            raise ValueError(f"preprocessor {key} mismatch")
    metadata = {
        "general.architecture": "clip",
        "general.type": "mmproj",
        "general.file_type": 0,
        "clip.projector_type": "qwen3vl_merger",
        "clip.has_vision_encoder": True,
        "clip.has_text_encoder": False,
        "clip.use_gelu": True,
        "clip.vision.image_size": math.isqrt(v["num_position_embeddings"])
        * v["patch_size"],
        "clip.vision.patch_size": v["patch_size"],
        "clip.vision.embedding_length": v["hidden_size"],
        "clip.vision.feed_forward_length": v["intermediate_size"],
        "clip.vision.block_count": v["depth"],
        "clip.vision.attention.head_count": v["num_heads"],
        "clip.vision.attention.layer_norm_epsilon": 1e-6,
        "clip.vision.spatial_merge_size": v["spatial_merge_size"],
        "clip.vision.projection_dim": v["out_hidden_size"],
        "clip.vision.image_min_pixels": minimum,
        "clip.vision.image_max_pixels": maximum,
        "clip.vision.image_mean": pre["image_mean"],
        "clip.vision.image_std": pre["image_std"],
        "clip.vision.is_deepstack_layers": [
            i in v["deepstack_visual_indexes"] for i in range(v["depth"])
        ],
        "sophon_realtime.preprocessor_json": json.dumps(pre, separators=(",", ":")),
    }
    for key, value in metadata.items():
        writer.add_meta(key, model_metadata_value(value))
    common_metadata(writer, config, "vision")
    for name, target, shape in vision_specs(config):
        if target == "patch":
            # Same temporal split as mage_flow._patch_embeddings, without retaining the tower.
            values = reader.f32(name)
            for frame in range(2):
                matrix = np.ascontiguousarray(
                    values[:, :, frame, :, :].reshape(shape[0], -1)
                )
                writer.add_tensor(
                    "v.patch_embd.weight" + ("" if frame == 0 else ".1"),
                    GGML_F32,
                    gguf_dims(matrix.shape),
                    matrix.tobytes(),
                )
        else:
            add_streamed(writer, reader, name, target, shape, "f32")
    return writer


def convert(
    model_dir: Path,
    output: Path,
    *,
    quant="f32",
    mmproj: Path | None = None,
    validate_pin=True,
):
    if quant not in FORMATS:
        raise ValueError(f"unknown format {quant}")
    root = Path(model_dir)
    if (root / "thinker" / "config.json").is_file():
        root = root / "thinker"
    outputs = [Path(output)] + ([Path(mmproj)] if mmproj is not None else [])
    if len(set(p.resolve() for p in outputs)) != len(outputs):
        raise ValueError("outputs must differ")
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    config = json.loads((root / "config.json").read_text())
    validate_config(config, validate_pin=validate_pin)
    reader = Reader(root)
    validate_inventory(reader, config, validate_pin=validate_pin)
    pool = (
        ProcessPoolExecutor(max_workers=K_WORKERS)
        if quant in ("q4_k", "q5_k", "q6_k") and K_WORKERS > 1
        else None
    )
    staged = []
    published = []
    try:
        for index, path in enumerate(outputs):
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            os.close(fd)
            temp = Path(name)
            staged.append(temp)
            writer = (
                text_writer(temp, reader, root, config, quant, pool)
                if index == 0
                else vision_writer(temp, reader, root, config)
            )
            writer.write()
            meta, directory = _read_gguf(temp)
            if set(directory) != {t[0] for t in writer.tensors}:
                raise ValueError("GGUF tensor readback mismatch")
            if meta.get("sophon_realtime.source_revision") != REVISION:
                raise ValueError("GGUF metadata readback mismatch")
        for temp, path in zip(staged, outputs):
            os.link(temp, path)
            published.append(path)
    except BaseException:
        for path in published:
            path.unlink()
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        for temp in staged:
            temp.unlink(missing_ok=True)
    return Path(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quant", choices=FORMATS, default="f32")
    parser.add_argument("--mmproj", type=Path)
    args = parser.parse_args()
    convert(args.model_dir, args.output, quant=args.quant, mmproj=args.mmproj)


if __name__ == "__main__":
    main()
