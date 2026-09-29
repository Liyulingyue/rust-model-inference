"""Independent scalar Edge0 token walk from original safetensors."""

import argparse
import ctypes as ct
import json
import math
import struct
import sys
from array import array
from pathlib import Path


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value):
    return struct.pack("<f", value)


def bf16(data, index):
    word = struct.unpack_from("<H", data, 2 * index)[0]
    return struct.unpack("<f", struct.pack("<I", word << 16))[0]


class Source:
    def __init__(self, root, library):
        self.root = root
        self.weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}
        self.library = ct.CDLL(str(library.resolve()))
        self.affine_fn = self.library.edge0_affine
        self.affine_fn.argtypes = [ct.c_void_p] * 7 + [ct.c_size_t, ct.c_size_t, ct.c_uint,
                                                      ct.c_size_t, ct.c_float]
        self.affine_fn.restype = ct.c_int

    def info(self, name):
        shard = self.weight_map.get(name, "lora_edge0_35b.safetensors")
        if shard not in self.headers:
            with (self.root / shard).open("rb") as source:
                header_size = struct.unpack("<Q", source.read(8))[0]
                self.headers[shard] = (8 + header_size, json.loads(source.read(header_size)))
        base, header = self.headers[shard]
        return shard, base, header.get(name)

    def read(self, name, start=0, size=None):
        shard, base, info = self.info(name)
        if info is None:
            raise ValueError(f"missing tensor: {name}")
        begin, end = info["data_offsets"]
        length = end - begin
        size = length - start if size is None else size
        if start < 0 or size < 0 or start + size > length:
            raise ValueError(f"invalid tensor slice: {name}")
        with (self.root / shard).open("rb") as source:
            source.seek(base + begin + start)
            data = source.read(size)
        if len(data) != size:
            raise ValueError(f"truncated tensor: {name}")
        return info["shape"], info["dtype"], data

    def bf16(self, name):
        shape, dtype, data = self.read(name)
        if dtype != "BF16":
            raise ValueError(f"expected BF16: {name}")
        return shape, [bf16(data, index) for index in range(len(data) // 2)]

    def embedding(self, token):
        stem = "language_model.model.embed_tokens"
        _, _, packed = self.read(stem + ".weight", token * 1024, 1024)
        _, _, scales = self.read(stem + ".scales", token * 64, 64)
        _, _, biases = self.read(stem + ".biases", token * 64, 64)
        values = []
        for col in range(2048):
            word = struct.unpack_from("<I", packed, 4 * (col // 8))[0]
            q = (word >> (4 * (col % 8))) & 15
            values.append(f32(f32(bf16(scales, col // 64) * q) + bf16(biases, col // 64)))
        return values

    def affine(self, stem, values, expert=None):
        width = len(values)
        shape, dtype, _ = self.read(stem + ".weight", 0, 0)
        if dtype != "U32" or len(shape) != (3 if expert is not None else 2):
            raise ValueError(f"unexpected affine weight: {stem}")
        output, words = shape[-2:]
        bits_per_value = words * 32 // width
        groups = width // 64
        if width % 64 or bits_per_value not in (4, 8) or words * 32 != width * bits_per_value:
            raise ValueError(f"unexpected affine dimensions: {stem}")
        prefix = [256] if expert is not None else []
        if shape != prefix + [output, words]:
            raise ValueError(f"unexpected expert count: {stem}")
        for suffix in ("scales", "biases"):
            other_shape, other_dtype, _ = self.read(stem + "." + suffix, 0, 0)
            if (other_shape, other_dtype) != (prefix + [output, groups], "BF16"):
                raise ValueError(f"unexpected affine {suffix}: {stem}")
        expert_index = expert or 0
        _, _, packed = self.read(stem + ".weight", expert_index * output * words * 4,
                                 output * words * 4)
        _, _, scales = self.read(stem + ".scales", expert_index * output * groups * 2,
                                 output * groups * 2)
        _, _, biases = self.read(stem + ".biases", expert_index * output * groups * 2,
                                 output * groups * 2)
        a_name, b_name = stem + ".lora_A", stem + ".lora_B"
        has_a = self.info(a_name)[2] is not None
        has_b = self.info(b_name)[2] is not None
        if has_a != has_b or (expert is not None and has_a):
            raise ValueError(f"invalid LoRA pair: {stem}")
        if has_a:
            a_shape, a_dtype, a = self.read(a_name)
            b_shape, b_dtype, b = self.read(b_name)
            rank = a_shape[0]
            if (a_shape, a_dtype, b_shape, b_dtype) != (
                [rank, width], "F16", [output, rank], "F16"
            ):
                raise ValueError(f"unexpected LoRA dimensions: {stem}")
        else:
            rank, a, b = 0, None, None
        buffers = [ct.create_string_buffer(data) if data is not None else None
                   for data in (packed, scales, biases, a, b)]
        input_buffer = (ct.c_float * width)(*values)
        result = (ct.c_float * output)()
        rc = self.affine_fn(*buffers, input_buffer, result, width, output,
                            bits_per_value, rank, 2.0)
        if rc != 0:
            raise ValueError(f"scalar affine failed ({rc}): {stem}")
        return list(result)


class Scalar:
    def __init__(self, source, trace_prefix):
        self.source = source
        self.trace_prefix = trace_prefix
        self.eps = f32(json.loads((source.root / "config.json").read_text())["text_config"]["rms_norm_eps"])
        self.expf = ct.CDLL(None).expf
        self.expf.argtypes = [ct.c_float]
        self.expf.restype = ct.c_float
        self.logf = ct.CDLL(None).logf
        self.logf.argtypes = [ct.c_float]
        self.logf.restype = ct.c_float
        self.powf = ct.CDLL(None).powf
        self.powf.argtypes = [ct.c_float, ct.c_float]
        self.powf.restype = ct.c_float
        self.trace_index = 0
        self.conv_inputs = {}
        self.recurrent_states = {}
        self.dense_kv = {}

    def silu(self, value):
        return f32(value / f32(1.0 + self.expf(f32(-value))))

    def sigmoid(self, value):
        return f32(1.0 / f32(1.0 + self.expf(f32(-value))))

    def norm(self, values, weight):
        mean_sq = f32(sum(float(f32(value * value)) for value in values) / len(values))
        scale = f32(1.0 / f32(math.sqrt(f32(mean_sq + self.eps))))
        return [f32(f32(value * scale) * w) for value, w in zip(values, weight)]

    def compare(self, name, values):
        suffix = f".{self.trace_index}" if self.trace_index else ""
        data = Path(str(self.trace_prefix) + "." + name + suffix + ".f32").read_bytes()
        if len(data) != 4 * len(values):
            raise ValueError(f"trace length mismatch: {name}")
        for index, value in enumerate(values):
            if bits(value) != data[4 * index:4 * (index + 1)]:
                actual = struct.unpack_from("<f", data, 4 * index)[0]
                raise AssertionError(
                    f"{name}[{index}] expected={value.hex()} actual={actual.hex()}"
                )

    def moe(self, layer, values):
        stem = f"language_model.model.layers.{layer}.mlp."
        logits = self.source.affine(stem + "gate", values)
        maximum = max(logits)
        exponentials = [self.expf(f32(value - maximum)) for value in logits]
        denominator = 0.0
        for value in exponentials:
            denominator = f32(denominator + value)
        probabilities = [f32(value / denominator) for value in exponentials]
        chosen = sorted(range(256), key=lambda index: (-probabilities[index], index))[:4]
        chosen_total = 0.0
        for index in chosen:
            chosen_total = f32(chosen_total + probabilities[index])
        routed = [0.0] * 2048
        for expert in chosen:
            gate = self.source.affine(stem + "switch_mlp.gate_proj", values, expert)
            up = self.source.affine(stem + "switch_mlp.up_proj", values, expert)
            middle = [f32(self.silu(a) * b) for a, b in zip(gate, up)]
            down = self.source.affine(stem + "switch_mlp.down_proj", middle, expert)
            score = f32(probabilities[expert] / chosen_total)
            for col, value in enumerate(down):
                routed[col] = f32(routed[col] + f32(score * value))
        shared_gate = self.sigmoid(self.source.affine(stem + "shared_expert_gate", values)[0])
        gate = self.source.affine(stem + "shared_expert.gate_proj", values)
        up = self.source.affine(stem + "shared_expert.up_proj", values)
        middle = [f32(self.silu(a) * b) for a, b in zip(gate, up)]
        shared = self.source.affine(stem + "shared_expert.down_proj", middle)
        return [f32(a + f32(shared_gate * b)) for a, b in zip(routed, shared)], chosen

    def recurrent_first(self, layer, values):
        stem = f"language_model.model.layers.{layer}.linear_attn."
        qkv = self.source.affine(stem + "in_proj_qkv", values)
        self.conv_inputs[layer] = [qkv]
        z = self.source.affine(stem + "in_proj_z", values)
        b = self.source.affine(stem + "in_proj_b", values)
        _, conv_weight = self.source.bf16(stem + "conv1d.weight")
        activated = []
        for channel, value in enumerate(qkv):
            total = 0.0
            for tap in range(4):
                total = f32(total + f32(conv_weight[channel * 4 + tap] *
                                        (value if tap == 3 else 0.0)))
            activated.append(self.silu(total))
        q, k = [], []
        for start, factor, output in ((0, f32(1.0 / 128.0), q),
                                      (2048, f32(1.0 / f32(math.sqrt(128.0))), k)):
            for head in range(16):
                part = activated[start + head * 128:start + (head + 1) * 128]
                mean_sq = f32(sum(float(f32(value * value)) for value in part) / 128)
                scale = f32(1.0 / f32(math.sqrt(f32(mean_sq + self.eps))))
                output.extend(f32(f32(value * scale) * factor) for value in part)
        beta = [self.sigmoid(value) for value in b]
        state_output = []
        states = array("f")
        _, norm_weight = self.source.bf16(stem + "norm.weight")
        for value_head in range(32):
            key_head = value_head // 2
            head_output = []
            for value_dim in range(128):
                delta = f32(activated[4096 + value_head * 128 + value_dim] * beta[value_head])
                state = [f32(0.0 + f32(k[key_head * 128 + key_dim] * delta))
                         for key_dim in range(128)]
                states.extend(state)
                head_output.append(f32(sum(float(f32(state[key_dim] * q[key_head * 128 + key_dim]))
                                           for key_dim in range(128))))
            normalized = self.norm(head_output, norm_weight)
            state_output.extend(f32(value * self.silu(z[value_head * 128 + dim]))
                                for dim, value in enumerate(normalized))
        self.recurrent_states[layer] = states
        return self.source.affine(stem + "out_proj", state_output)

    def recurrent_next(self, layer, values):
        stem = f"language_model.model.layers.{layer}.linear_attn."
        qkv = self.source.affine(stem + "in_proj_qkv", values)
        if layer == 0:
            self.compare("edge0.qkv-0", qkv)
        z = self.source.affine(stem + "in_proj_z", values)
        beta = [self.sigmoid(value) for value in self.source.affine(stem + "in_proj_b", values)]
        alpha = self.source.affine(stem + "in_proj_a", values)
        _, dt_bias = self.source.bf16(stem + "dt_bias")
        _, a_log = self.source.bf16(stem + "A_log")
        _, conv_weight = self.source.bf16(stem + "conv1d.weight")
        history = self.conv_inputs[layer]
        activated = []
        conv_raw = []
        for channel, value in enumerate(qkv):
            total = 0.0
            for tap in range(4):
                lag = 3 - tap
                sample = value if lag == 0 else history[-lag][channel] if len(history) >= lag else 0.0
                total = f32(total + f32(conv_weight[channel * 4 + tap] * sample))
            conv_raw.append(total)
            activated.append(self.silu(total))
        if layer == 0:
            self.compare("conv_output_raw-0", conv_raw)
        history.append(qkv)
        del history[:-3]
        q, k = [], []
        for start, factor, output in ((0, f32(1.0 / 128.0), q),
                                      (2048, f32(1.0 / f32(math.sqrt(128.0))), k)):
            for head in range(16):
                part = activated[start + head * 128:start + (head + 1) * 128]
                mean_sq = f32(sum(float(f32(value * value)) for value in part) / 128)
                scale = f32(1.0 / f32(math.sqrt(f32(mean_sq + self.eps))))
                output.extend(f32(f32(value * scale) * factor) for value in part)
        if layer == 0:
            self.compare("q_conv_predelta-0", q)
            self.compare("k_conv_predelta-0", k)
        states = self.recurrent_states[layer]
        if layer == 0:
            self.compare("state_predelta-0", states)
        state_output = []
        _, norm_weight = self.source.bf16(stem + "norm.weight")
        for value_head in range(32):
            key_head = value_head // 2
            biased = f32(alpha[value_head] + dt_bias[value_head])
            softplus = biased if biased > 20.0 else self.logf(f32(1.0 + self.expf(biased)))
            decay = self.expf(f32(softplus * f32(-self.expf(a_log[value_head]))))
            head_output = []
            for value_dim in range(128):
                offset = (value_head * 128 + value_dim) * 128
                row = [f32(states[offset + dim] * decay) for dim in range(128)]
                old_projection = f32(sum(float(f32(row[dim] * k[key_head * 128 + dim]))
                                         for dim in range(128)))
                delta = f32(f32(activated[4096 + value_head * 128 + value_dim]
                                - old_projection) * beta[value_head])
                for dim in range(128):
                    row[dim] = f32(row[dim] + f32(delta * k[key_head * 128 + dim]))
                states[offset:offset + 128] = array("f", row)
                head_output.append(f32(sum(float(f32(row[dim] * q[key_head * 128 + dim]))
                                           for dim in range(128))))
            normalized = self.norm(head_output, norm_weight)
            state_output.extend(f32(value * self.silu(z[value_head * 128 + dim]))
                                for dim, value in enumerate(normalized))
        if layer == 0:
            self.compare("new_state-0", states)
            self.compare("final_output-0", state_output)
        projection = self.source.affine(stem + "out_proj", state_output)
        if layer == 0:
            self.compare("edge0.recurrent_projection-0", projection)
        return projection

    def dense_first(self, layer, values):
        stem = f"language_model.model.layers.{layer}.self_attn."
        q = self.source.affine(stem + "q_proj", values)
        k = self.source.affine(stem + "k_proj", values)
        v = self.source.affine(stem + "v_proj", values)
        _, q_weight = self.source.bf16(stem + "q_norm.weight")
        _, k_weight = self.source.bf16(stem + "k_norm.weight")
        q_normed = []
        output = []
        for head in range(16):
            start = head * 512
            q_normed.extend(self.norm(q[start:start + 256], q_weight))
            gate = q[start + 256:start + 512]
            kv_start = (head // 8) * 256
            output.extend(f32(v[kv_start + dim] * self.sigmoid(gate[dim]))
                          for dim in range(256))
        k_normed = []
        for head in range(2):
            start = head * 256
            k_normed.extend(self.norm(k[start:start + 256], k_weight))
        self.dense_kv[layer] = [(k_normed, v)]
        if layer == 3:
            self.compare("Qcur_normed-3", q_normed)
            self.compare("Kcur_normed-3", k_normed)
        # At position zero, RoPE is the identity and each head attends only
        # to its own V with probability one.
        return self.source.affine(stem + "o_proj", output)

    def rope_text(self, values, position):
        values = values.copy()
        theta = f32(position)
        scale = self.powf(f32(1e7), f32(-2.0 / 64.0))
        for index in range(32):
            cosine, sine = f32(math.cos(theta)), f32(math.sin(theta))
            left, right = values[index], values[index + 32]
            values[index] = f32(f32(left * cosine) - f32(right * sine))
            values[index + 32] = f32(f32(left * sine) + f32(right * cosine))
            theta = f32(theta * scale)
        return values

    def dense_next(self, layer, values):
        stem = f"language_model.model.layers.{layer}.self_attn."
        q = self.source.affine(stem + "q_proj", values)
        k = self.source.affine(stem + "k_proj", values)
        v = self.source.affine(stem + "v_proj", values)
        _, q_weight = self.source.bf16(stem + "q_norm.weight")
        _, k_weight = self.source.bf16(stem + "k_norm.weight")
        q_normed = []
        for head in range(16):
            start = head * 512
            q_normed.extend(self.norm(q[start:start + 256], q_weight))
        k_normed = []
        for head in range(2):
            start = head * 256
            k_normed.extend(self.norm(k[start:start + 256], k_weight))
        if layer == 3:
            self.compare("Qcur_normed-3", q_normed)
            self.compare("Kcur_normed-3", k_normed)
        q_rotated = [value for head in range(16) for value in
                     self.rope_text(q_normed[head * 256:(head + 1) * 256], self.trace_index)]
        k_rotated = [value for head in range(2) for value in
                     self.rope_text(k_normed[head * 256:(head + 1) * 256], self.trace_index)]
        cache = self.dense_kv[layer]
        cache.append((k_rotated, v))
        output = []
        attention_scale = f32(1.0 / f32(math.sqrt(256.0)))
        for head in range(16):
            kv_head = head // 8
            query = q_rotated[head * 256:(head + 1) * 256]
            scores = []
            for keys, _ in cache:
                key = keys[kv_head * 256:(kv_head + 1) * 256]
                dot = f32(sum(float(f32(a * b)) for a, b in zip(query, key)))
                scores.append(f32(dot * attention_scale))
            maximum = max(scores)
            exponentials = [self.expf(f32(value - maximum)) for value in scores]
            total = sum(float(value) for value in exponentials)
            probabilities = [f32(value * f32(1.0 / total)) for value in exponentials]
            for dim in range(256):
                attended = f32(sum(float(f32(probabilities[index] *
                                            pair[1][kv_head * 256 + dim]))
                                   for index, pair in enumerate(cache)))
                gate = self.sigmoid(q[head * 512 + 256 + dim])
                output.append(f32(attended * gate))
        return self.source.affine(stem + "o_proj", output)

    def layer_first(self, layer, hidden):
        base = f"language_model.model.layers.{layer}."
        _, input_weight = self.source.bf16(base + "input_layernorm.weight")
        normalized = self.norm(hidden, input_weight)
        if self.trace_index == 1 and layer == 0:
            self.compare("edge0.norm-0", normalized)
        if (layer + 1) % 4 != 0:
            attended = (self.recurrent_first if self.trace_index == 0 else self.recurrent_next)(
                layer, normalized)
        else:
            attended = (self.dense_first if self.trace_index == 0 else self.dense_next)(
                layer, normalized)
        residual = [f32(a + b) for a, b in zip(hidden, attended)]
        _, post_weight = self.source.bf16(base + "post_attention_layernorm.weight")
        moe_input = self.norm(residual, post_weight)
        moe_output, chosen = self.moe(layer, moe_input)
        output = [f32(a + b) for a, b in zip(residual, moe_output)]
        self.compare(f"layer_output-{layer}", output)
        print(f"token {self.trace_index} layer {layer} matched 2048 F32 bits; experts={chosen}", flush=True)
        return output

    def final_first(self, hidden):
        _, weight = self.source.bf16("language_model.model.norm.weight")
        normalized = self.norm(hidden, weight)
        self.compare("result_norm", normalized)
        logits = self.source.affine("language_model.lm_head", normalized)
        self.compare("result_output", logits)
        print(f"final norm and {len(logits)} logits matched F32 bits", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("trace_prefix", type=Path)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--token-ids", default="248045",
                        help="comma-separated prompt/decode token IDs")
    args = parser.parse_args()
    if not 1 <= args.layers <= 40:
        parser.error("scalar walk supports 1 to 40 layers")
    try:
        token_ids = [int(value) for value in args.token_ids.split(",")]
    except ValueError:
        parser.error("--token-ids must contain integers")
    if not token_ids:
        parser.error("--token-ids must contain at least one token")
    suffix = "dylib" if sys.platform == "darwin" else "so"
    library = Path(__file__).resolve().parents[3] / f"target/oracle/libedge0_scalar.{suffix}"
    scalar = Scalar(Source(args.model_dir, library), args.trace_prefix)
    for position, token_id in enumerate(token_ids):
        scalar.trace_index = position
        hidden = scalar.source.embedding(token_id)
        scalar.compare("edge0.embedding", hidden)
        for layer in range(args.layers):
            hidden = scalar.layer_first(layer, hidden)
        if args.layers == 40:
            scalar.final_first(hidden)


if __name__ == "__main__":
    main()
