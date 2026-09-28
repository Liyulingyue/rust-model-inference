"""Check real Edge0 embedding, first norm, and QKV F32 bits without MLX/BLAS/SIMD.

Run after a `Hello` CLI trace with RMI_PARITY_FILTER=edge0.embedding,edge0.norm-0,edge0.qkv-0.
The first prompt token is 248045 (<|im_start|>).
"""

import argparse
import json
import math
import struct
from pathlib import Path


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value):
    return struct.pack("<f", value)


def bf16(data, index):
    word = struct.unpack_from("<H", data, 2 * index)[0]
    return struct.unpack("<f", struct.pack("<I", word << 16))[0]


def f16(data, index):
    return struct.unpack_from("<e", data, 2 * index)[0]


def trace(prefix, name):
    return (Path(str(prefix) + "." + name + ".f32")).read_bytes()


def check(model_dir, prefix):
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    headers = {}

    def tensor(name, start=0, size=None):
        shard = index.get(name, "lora_edge0_35b.safetensors")
        path = model_dir / shard
        if shard not in headers:
            with path.open("rb") as source:
                header_size = struct.unpack("<Q", source.read(8))[0]
                headers[shard] = (8 + header_size, json.loads(source.read(header_size)))
        data_start, header = headers[shard]
        info = header[name]
        begin, end = info["data_offsets"]
        length = end - begin
        size = length - start if size is None else size
        if start < 0 or size < 0 or start + size > length:
            raise ValueError(f"out-of-range tensor read: {name}")
        with path.open("rb") as source:
            source.seek(data_start + begin + start)
            data = source.read(size)
        if len(data) != size:
            raise ValueError(f"truncated tensor: {name}")
        return info["shape"], info["dtype"], data

    embedding = "language_model.model.embed_tokens"
    token = 248045
    _, dtype, packed = tensor(embedding + ".weight", token * 1024, 1024)
    _, scale_dtype, scales = tensor(embedding + ".scales", token * 64, 64)
    _, bias_dtype, biases = tensor(embedding + ".biases", token * 64, 64)
    if (dtype, scale_dtype, bias_dtype) != ("U32", "BF16", "BF16"):
        raise ValueError("unexpected embedding types")
    actual = trace(prefix, "edge0.embedding")
    if len(actual) != 2048 * 4:
        raise ValueError("unexpected embedding trace length")
    embedding_values = []
    for col in range(2048):
        word = struct.unpack_from("<I", packed, 4 * (col // 8))[0]
        q = (word >> (4 * (col % 8))) & 15
        expected = f32(f32(bf16(scales, col // 64) * q) + bf16(biases, col // 64))
        embedding_values.append(expected)
        if bits(expected) != actual[4 * col:4 * (col + 1)]:
            raise AssertionError(f"embedding F32 mismatch at column {col}")

    shape, dtype, norm_weight = tensor("language_model.model.layers.0.input_layernorm.weight")
    if (shape, dtype) != ([2048], "BF16"):
        raise ValueError("unexpected first norm contract")
    sum_sq = sum(float(f32(value * value)) for value in embedding_values)
    mean_sq = f32(sum_sq / len(embedding_values))
    eps = f32(json.loads((model_dir / "config.json").read_text())["text_config"]["rms_norm_eps"])
    scale = f32(1.0 / f32(math.sqrt(f32(mean_sq + eps))))
    x = tuple(f32(f32(value * scale) * bf16(norm_weight, col))
              for col, value in enumerate(embedding_values))
    actual_norm = trace(prefix, "edge0.norm-0")
    if len(actual_norm) != 2048 * 4:
        raise ValueError("unexpected first norm trace length")
    for col, expected in enumerate(x):
        if bits(expected) != actual_norm[4 * col:4 * (col + 1)]:
            raise AssertionError(f"first norm F32 mismatch at column {col}")

    qkv = trace(prefix, "edge0.qkv-0")
    if len(qkv) != 8192 * 4:
        raise ValueError("unexpected QKV trace length")
    stem = "language_model.model.layers.0.linear_attn.in_proj_qkv"
    shape, dtype, _ = tensor(stem + ".weight", 0, 0)
    a_shape, a_dtype, a = tensor(stem + ".lora_A")
    b_shape, b_dtype, _ = tensor(stem + ".lora_B", 0, 0)
    if (shape, dtype, a_shape, a_dtype, b_shape, b_dtype) != (
        [8192, 256], "U32", [16, 2048], "F16", [8192, 16], "F16"
    ):
        raise ValueError("unexpected QKV/LoRA contract")
    low = []
    for rank in range(16):
        total = 0.0
        for col, value in enumerate(x):
            total = f32(total + f32(value * f16(a, rank * 2048 + col)))
        low.append(total)
    rows = range(8192)
    for row in rows:
        _, _, packed = tensor(stem + ".weight", row * 1024, 1024)
        _, _, scales = tensor(stem + ".scales", row * 64, 64)
        _, _, biases = tensor(stem + ".biases", row * 64, 64)
        _, _, lora_b = tensor(stem + ".lora_B", row * 32, 32)
        total = 0.0
        for col, value in enumerate(x):
            word = struct.unpack_from("<I", packed, 4 * (col // 8))[0]
            q = (word >> (4 * (col % 8))) & 15
            weight = f32(f32(bf16(scales, col // 64) * q) + bf16(biases, col // 64))
            total = f32(total + f32(value * weight))
        delta = 0.0
        for rank, value in enumerate(low):
            delta = f32(delta + f32(value * f16(lora_b, rank)))
        expected = f32(total + f32(2.0 * delta))
        if bits(expected) != qkv[4 * row:4 * (row + 1)]:
            raise AssertionError(f"QKV F32 mismatch at row {row}")
    print(f"matched 2048 embedding, 2048 first norm, and {len(rows)} QKV F32 words bitwise")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("trace_prefix", type=Path)
    args = parser.parse_args()
    check(args.model_dir, args.trace_prefix)
