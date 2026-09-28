"""Check Edge0's first token through layer 0 without MLX/BLAS/SIMD.

Run after a `Hello` CLI trace with the RMI_PARITY_FILTER in this directory's README.
The first prompt token is 248045 (<|im_start|>).
"""

import argparse
import ctypes
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
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    headers = {}

    def tensor(name, start=0, size=None):
        shard = weight_map.get(name, "lora_edge0_35b.safetensors")
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

    def affine_lora(stem, inputs, output):
        width = len(inputs)
        if width % 64:
            raise ValueError(f"invalid affine input width: {stem}")
        shape, dtype, _ = tensor(stem + ".weight", 0, 0)
        a_shape, a_dtype, a = tensor(stem + ".lora_A")
        b_shape, b_dtype, _ = tensor(stem + ".lora_B", 0, 0)
        if (shape, dtype, a_shape, a_dtype, b_shape, b_dtype) != (
            [output, width // 8], "U32", [16, width], "F16", [output, 16], "F16"
        ):
            raise ValueError(f"unexpected affine/LoRA contract: {stem}")
        low = []
        for rank in range(16):
            total = 0.0
            for col, value in enumerate(inputs):
                total = f32(total + f32(value * f16(a, rank * width + col)))
            low.append(total)
        values = []
        for row in range(output):
            _, _, packed = tensor(stem + ".weight", row * width // 2, width // 2)
            _, _, scales = tensor(stem + ".scales", row * width // 32, width // 32)
            _, _, biases = tensor(stem + ".biases", row * width // 32, width // 32)
            _, _, lora_b = tensor(stem + ".lora_B", row * 32, 32)
            total = 0.0
            for col, value in enumerate(inputs):
                word = struct.unpack_from("<I", packed, 4 * (col // 8))[0]
                q = (word >> (4 * (col % 8))) & 15
                weight = f32(f32(bf16(scales, col // 64) * q) + bf16(biases, col // 64))
                total = f32(total + f32(value * weight))
            delta = 0.0
            for rank, value in enumerate(low):
                delta = f32(delta + f32(value * f16(lora_b, rank)))
            values.append(f32(total + f32(2.0 * delta)))
        return values

    def affine(stem, inputs, output, bits_per_value, expert=None):
        width = len(inputs)
        if width % 64 or bits_per_value not in (4, 8):
            raise ValueError(f"invalid affine matrix: {stem}")
        words = width * bits_per_value // 32
        groups = width // 64
        prefix = [256] if expert is not None else []
        shape, dtype, _ = tensor(stem + ".weight", 0, 0)
        scale_shape, scale_dtype, _ = tensor(stem + ".scales", 0, 0)
        bias_shape, bias_dtype, _ = tensor(stem + ".biases", 0, 0)
        if (shape, dtype, scale_shape, scale_dtype, bias_shape, bias_dtype) != (
            prefix + [output, words], "U32", prefix + [output, groups], "BF16",
            prefix + [output, groups], "BF16"
        ):
            raise ValueError(f"unexpected affine contract: {stem}")
        values = []
        row_bytes = words * 4
        group_bytes = groups * 2
        base = (expert or 0) * output
        per_word = 32 // bits_per_value
        mask = (1 << bits_per_value) - 1
        for row in range(output):
            _, _, packed = tensor(stem + ".weight", (base + row) * row_bytes, row_bytes)
            _, _, scales = tensor(stem + ".scales", (base + row) * group_bytes, group_bytes)
            _, _, biases = tensor(stem + ".biases", (base + row) * group_bytes, group_bytes)
            total = 0.0
            for col, value in enumerate(inputs):
                word = struct.unpack_from("<I", packed, 4 * (col // per_word))[0]
                q = (word >> (bits_per_value * (col % per_word))) & mask
                weight = f32(f32(bf16(scales, col // 64) * q) + bf16(biases, col // 64))
                total = f32(total + f32(value * weight))
            values.append(total)
        return values

    qkv_values = affine_lora("language_model.model.layers.0.linear_attn.in_proj_qkv", x, 8192)
    qkv = trace(prefix, "edge0.qkv-0")
    if len(qkv) != 8192 * 4:
        raise ValueError("unexpected QKV trace length")
    for row, expected in enumerate(qkv_values):
        if bits(expected) != qkv[4 * row:4 * (row + 1)]:
            raise AssertionError(f"QKV F32 mismatch at row {row}")

    shape, dtype, conv_weight = tensor("language_model.model.layers.0.linear_attn.conv1d.weight")
    if (shape, dtype) != ([8192, 4, 1], "BF16"):
        raise ValueError("unexpected first convolution contract")
    conv = trace(prefix, "conv_output_raw-0")
    if len(conv) != 8192 * 4:
        raise ValueError("unexpected first convolution trace length")
    # At the first prompt token the causal four-tap state contains three zeros.
    for channel, value in enumerate(qkv_values):
        total = 0.0
        for tap in range(4):
            previous = value if tap == 3 else 0.0
            total = f32(total + f32(bf16(conv_weight, channel * 4 + tap) * previous))
        if bits(total) != conv[4 * channel:4 * (channel + 1)]:
            raise AssertionError(f"first convolution F32 mismatch at channel {channel}")

    expf = ctypes.CDLL(None).expf
    expf.argtypes = [ctypes.c_float]
    expf.restype = ctypes.c_float

    def silu_scalar(value):
        return f32(value / f32(1.0 + expf(f32(-value))))

    def sigmoid_scalar(value):
        return f32(1.0 / f32(1.0 + expf(f32(-value))))

    conv_values = struct.unpack("<8192f", conv)
    normalized = {}
    for name, start, factor in (
        ("q", 0, f32(1.0 / 128.0)),
        ("k", 2048, f32(1.0 / f32(math.sqrt(128.0)))),
    ):
        actual = trace(prefix, f"{name}_conv_predelta-0")
        if len(actual) != 2048 * 4:
            raise ValueError(f"unexpected first {name} norm trace length")
        normalized[name] = []
        for head in range(16):
            values = [silu_scalar(value)
                      for value in conv_values[start + head * 128:start + (head + 1) * 128]]
            mean_sq = f32(sum(float(f32(value * value)) for value in values) / 128)
            scale = f32(1.0 / f32(math.sqrt(f32(mean_sq + eps))))
            for dimension, value in enumerate(values):
                expected = f32(f32(value * scale) * factor)
                normalized[name].append(expected)
                offset = 4 * (head * 128 + dimension)
                if bits(expected) != actual[offset:offset + 4]:
                    raise AssertionError(f"first {name} norm F32 mismatch at head {head}, dimension {dimension}")

    beta_raw = affine_lora("language_model.model.layers.0.linear_attn.in_proj_b", x, 32)
    beta = [sigmoid_scalar(value) for value in beta_raw]
    actual_beta = trace(prefix, "edge0.beta-0")
    if len(actual_beta) != 32 * 4:
        raise ValueError("unexpected first beta trace length")
    for head, expected in enumerate(beta):
        if bits(expected) != actual_beta[4 * head:4 * (head + 1)]:
            raise AssertionError(f"first beta F32 mismatch at head {head}")

    state = trace(prefix, "new_state-0")
    if len(state) != 32 * 128 * 128 * 4:
        raise ValueError("unexpected first recurrent state trace length")
    for value_head in range(32):
        key_head = value_head // 2
        for value_dim in range(128):
            value = conv_values[4096 + value_head * 128 + value_dim]
            activated = silu_scalar(value)
            delta = f32(activated * beta[value_head])
            for key_dim in range(128):
                expected = f32(0.0 + f32(normalized["k"][key_head * 128 + key_dim] * delta))
                offset = 4 * ((value_head * 128 + value_dim) * 128 + key_dim)
                if bits(expected) != state[offset:offset + 4]:
                    raise AssertionError(
                        f"first recurrent state F32 mismatch at head {value_head}, value {value_dim}, key {key_dim}"
                    )

    z_values = affine_lora("language_model.model.layers.0.linear_attn.in_proj_z", x, 4096)
    actual_z = trace(prefix, "edge0.z-0")
    if len(actual_z) != 4096 * 4:
        raise ValueError("unexpected first z trace length")
    for row, expected in enumerate(z_values):
        if bits(expected) != actual_z[4 * row:4 * (row + 1)]:
            raise AssertionError(f"first z F32 mismatch at index {row}")

    shape, dtype, norm_weight = tensor("language_model.model.layers.0.linear_attn.norm.weight")
    if (shape, dtype) != ([128], "BF16"):
        raise ValueError("unexpected first recurrent output norm contract")
    actual_output = trace(prefix, "final_output-0")
    if len(actual_output) != 4096 * 4:
        raise ValueError("unexpected first recurrent output trace length")
    state_values = struct.unpack("<524288f", state)
    recurrent_values = []
    for value_head in range(32):
        key_head = value_head // 2
        attended = []
        for value_dim in range(128):
            start = (value_head * 128 + value_dim) * 128
            products = (
                float(f32(state_values[start + key_dim] * normalized["q"][key_head * 128 + key_dim]))
                for key_dim in range(128)
            )
            attended.append(f32(sum(products)))
        mean_sq = f32(sum(float(f32(value * value)) for value in attended) / 128)
        scale = f32(1.0 / f32(math.sqrt(f32(mean_sq + eps))))
        for value_dim, value in enumerate(attended):
            expected = f32(f32(f32(value * scale) * bf16(norm_weight, value_dim)) *
                           silu_scalar(z_values[value_head * 128 + value_dim]))
            recurrent_values.append(expected)
            offset = 4 * (value_head * 128 + value_dim)
            if bits(expected) != actual_output[offset:offset + 4]:
                raise AssertionError(f"first recurrent output F32 mismatch at head {value_head}, dimension {value_dim}")

    projected = affine_lora("language_model.model.layers.0.linear_attn.out_proj", recurrent_values, 2048)
    actual_projected = trace(prefix, "edge0.recurrent_projection-0")
    if len(actual_projected) != 2048 * 4:
        raise ValueError("unexpected first recurrent projection trace length")
    for row, expected in enumerate(projected):
        if bits(expected) != actual_projected[4 * row:4 * (row + 1)]:
            raise AssertionError(f"first recurrent projection F32 mismatch at row {row}")

    shape, dtype, post_weight = tensor("language_model.model.layers.0.post_attention_layernorm.weight")
    if (shape, dtype) != ([2048], "BF16"):
        raise ValueError("unexpected first post-attention norm contract")
    residual = [f32(a + b) for a, b in zip(embedding_values, projected)]
    mean_sq = f32(sum(float(f32(value * value)) for value in residual) / 2048)
    scale = f32(1.0 / f32(math.sqrt(f32(mean_sq + eps))))
    actual_moe_input = trace(prefix, "edge0.moe_input-0")
    if len(actual_moe_input) != 2048 * 4:
        raise ValueError("unexpected first MoE input trace length")
    moe_input = []
    for col, value in enumerate(residual):
        expected = f32(f32(value * scale) * bf16(post_weight, col))
        moe_input.append(expected)
        if bits(expected) != actual_moe_input[4 * col:4 * (col + 1)]:
            raise AssertionError(f"first MoE input F32 mismatch at column {col}")

    mlp = "language_model.model.layers.0.mlp."
    logits = affine(mlp + "gate", moe_input, 256, 8)
    actual_router = trace(prefix, "edge0.router-0")
    if len(actual_router) != 256 * 4:
        raise ValueError("unexpected first router trace length")
    for expert, expected in enumerate(logits):
        if bits(expected) != actual_router[4 * expert:4 * (expert + 1)]:
            raise AssertionError(f"first router F32 mismatch at expert {expert}")
    maximum = max(logits)
    exponentials = [expf(f32(value - maximum)) for value in logits]
    denominator = 0.0
    for value in exponentials:
        denominator = f32(denominator + value)
    probabilities = [f32(value / denominator) for value in exponentials]
    chosen = sorted(range(256), key=lambda index: (-probabilities[index], index))[:4]
    actual_chosen = trace(prefix, "edge0.chosen-0")
    if len(actual_chosen) != 4 * 4:
        raise ValueError("unexpected first chosen expert trace length")
    if tuple(chosen) != struct.unpack("<4f", actual_chosen):
        raise AssertionError(f"first chosen experts mismatch: expected={chosen} actual={struct.unpack('<4f', actual_chosen)}")
    chosen_total = 0.0
    for index in chosen:
        chosen_total = f32(chosen_total + probabilities[index])
    routed = [0.0] * 2048
    for expert in chosen:
        gate = affine(mlp + "switch_mlp.gate_proj", moe_input, 512, 4, expert)
        up = affine(mlp + "switch_mlp.up_proj", moe_input, 512, 4, expert)
        intermediate = [f32(silu_scalar(a) * b) for a, b in zip(gate, up)]
        down = affine(mlp + "switch_mlp.down_proj", intermediate, 2048, 4, expert)
        score = f32(probabilities[expert] / chosen_total)
        for col, value in enumerate(down):
            routed[col] = f32(routed[col] + f32(score * value))
    actual_routed = trace(prefix, "edge0.routed-0")
    for col, expected in enumerate(routed):
        if bits(expected) != actual_routed[4 * col:4 * (col + 1)]:
            raise AssertionError(f"first routed MoE F32 mismatch at column {col}; experts={chosen}")

    shared_gate = sigmoid_scalar(affine(mlp + "shared_expert_gate", moe_input, 1, 8)[0])
    shared_gate_proj = affine_lora(mlp + "shared_expert.gate_proj", moe_input, 512)
    shared_up = affine_lora(mlp + "shared_expert.up_proj", moe_input, 512)
    shared_intermediate = [f32(silu_scalar(a) * b) for a, b in zip(shared_gate_proj, shared_up)]
    shared = affine_lora(mlp + "shared_expert.down_proj", shared_intermediate, 2048)
    actual_shared_gate = trace(prefix, "edge0.shared_gate-0")
    if bits(shared_gate) != actual_shared_gate:
        raise AssertionError("first shared expert gate F32 mismatch")
    actual_shared = trace(prefix, "edge0.shared-0")
    for col, expected in enumerate(shared):
        if bits(expected) != actual_shared[4 * col:4 * (col + 1)]:
            raise AssertionError(f"first shared expert F32 mismatch at column {col}")
    actual_layer = trace(prefix, "layer_output-0")
    if len(actual_layer) != 2048 * 4:
        raise ValueError("unexpected first layer output trace length")
    for col in range(2048):
        moe = f32(routed[col] + f32(shared_gate * shared[col]))
        expected = f32(residual[col] + moe)
        if bits(expected) != actual_layer[4 * col:4 * (col + 1)]:
            actual = struct.unpack_from("<f", actual_layer, 4 * col)[0]
            raise AssertionError(
                f"first MoE/layer output F32 mismatch at column {col}: "
                f"expected={expected.hex()} actual={actual.hex()} "
                f"residual={residual[col].hex()} routed={routed[col].hex()} "
                f"shared={shared[col].hex()} shared_gate={shared_gate.hex()}; experts={chosen}"
            )
    print(f"matched 567585 first-token F32 words and 4 expert IDs through layer 0 bitwise; experts={chosen}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("trace_prefix", type=Path)
    args = parser.parse_args()
    check(args.model_dir, args.trace_prefix)
