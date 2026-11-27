"""Convert the Microsoft Aion Instruct checkpoint to the native Aion GGUF."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from tools.converter.utils.gguf import (
    GGML_BF16, GGML_F16, GGML_F32, GGML_Q4_0, GGML_Q4_1, GGML_Q4K,
    GGML_Q5K, GGML_Q6K, GGML_Q8_0, GgufWriter, bf16_to_f16, bf16_to_f32,
    gguf_dims, open_safetensors, quantize_q4_0, quantize_q4_1, quantize_q8_0,
)
from tools.converter.utils.kquants import quantize_k

ARCH = "aion_instruct"
# GGML type, values per block, bytes per block.
FORMATS = {
    "F32": (GGML_F32, 1, 4),
    "F16": (GGML_F16, 1, 2),
    "BF16": (GGML_BF16, 1, 2),
    "Q8_0": (GGML_Q8_0, 32, 34),
    "Q4_0": (GGML_Q4_0, 32, 18),
    "Q4_1": (GGML_Q4_1, 32, 20),
    "Q4_K": (GGML_Q4K, 256, 144),
    "Q5_K": (GGML_Q5K, 256, 176),
    "Q6_K": (GGML_Q6K, 256, 210),
}


def tensor_map(config):
    width = config["hidden_size"]
    dim = config["head_dim"]
    q = config["num_attention_heads"] * dim
    kv = config["num_key_value_heads"] * dim
    ff = config["intermediate_size"]
    tensors = {
        "model.embed_tokens.weight": ("token_embd.weight", (config["vocab_size"], width)),
        "model.norm.weight": ("output_norm.weight", (width,)),
    }
    for layer in range(config["num_hidden_layers"]):
        for hf, gguf, shape in (
            ("input_layernorm", "attn_norm", (width,)),
            ("self_attn.q_proj", "attn_q", (q, width)),
            ("self_attn.k_proj", "attn_k", (kv, width)),
            ("self_attn.v_proj", "attn_v", (kv, width)),
            ("self_attn.o_proj", "attn_output", (width, q)),
            ("self_attn.q_norm", "attn_q_norm", (dim,)),
            ("self_attn.k_norm", "attn_k_norm", (dim,)),
            ("post_attention_layernorm", "ffn_norm", (width,)),
            ("mlp.gate_proj", "ffn_gate", (ff, width)),
            ("mlp.up_proj", "ffn_up", (ff, width)),
            ("mlp.down_proj", "ffn_down", (width, ff)),
        ):
            tensors[f"model.layers.{layer}.{hf}.weight"] = (f"blk.{layer}.{gguf}.weight", shape)
    return tensors


def encode(raw, precision):
    if precision == "BF16":
        return raw
    if precision == "F16":
        return bf16_to_f16(raw)
    f32 = bf16_to_f32(raw)
    if precision == "F32":
        return f32
    values = np.frombuffer(f32, dtype="<f4")
    quantize = {"Q8_0": quantize_q8_0, "Q4_0": quantize_q4_0, "Q4_1": quantize_q4_1}
    if precision in quantize:
        return quantize[precision](values)
    return quantize_k(precision.lower(), values)


def convert(model_dir, output, precision):
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((model_dir / "config.json").read_text())
    generation = json.loads((model_dir / "generation_config.json").read_text())
    if (config["model_type"] != ARCH or config["attention_bias"]
            or config["hidden_act"] != "silu" or not config["tie_word_embeddings"]):
        raise ValueError("expected a tied, bias-free Aion Instruct checkpoint with SwiGLU")
    layers = config["layer_types"]
    full = config["rope_parameters"]["full_attention"]
    sliding = config["rope_parameters"]["sliding_attention"]
    if (len(layers) != config["num_hidden_layers"]
            or set(layers) - {"full_attention", "sliding_attention"}
            or set(full) != {"rope_type", "rope_theta", "factor", "original_max_position_embeddings",
                             "beta_fast", "beta_slow"}
            or set(sliding) != {"rope_type", "rope_theta"}
            or full["rope_type"] != "yarn" or sliding["rope_type"] != "default"):
        raise ValueError("unsupported Aion attention schedule or RoPE parameters")

    source = open_safetensors(model_dir / "model.safetensors")
    tensors = tensor_map(config)
    if set(source.header) - {"__metadata__"} != set(tensors):
        raise ValueError("unexpected or missing tensor names in Aion checkpoint")

    writer = GgufWriter(output)
    writer.add_meta("general.architecture", ARCH)
    writer.add_meta("general.name", model_dir.name)
    writer.add_meta("tokenizer.ggml.model", "huggingface")
    writer.add_meta("tokenizer.huggingface.json", (model_dir / "tokenizer.json").read_text())
    writer.add_meta("tokenizer.ggml.bos_token_id", config["bos_token_id"])
    writer.add_meta("tokenizer.ggml.eos_token_id", config["eos_token_id"])
    writer.add_meta("tokenizer.chat_template", (model_dir / "chat_template.jinja").read_text())
    with source.path.open("rb") as stream:
        writer.add_meta(f"{ARCH}.source_sha256", hashlib.file_digest(stream, "sha256").hexdigest())
    for key, value in {
        "embedding_length": config["hidden_size"],
        "block_count": config["num_hidden_layers"],
        "attention.head_count": config["num_attention_heads"],
        "attention.head_count_kv": config["num_key_value_heads"],
        "attention.key_length": config["head_dim"],
        "attention.sliding_window": config["sliding_window"],
        "attention.layer_norm_rms_epsilon": config["rms_norm_eps"],
        "attention.layer_types": layers,
        "feed_forward_length": config["intermediate_size"],
        "context_length": config["max_position_embeddings"],
        "vocab_size": config["vocab_size"],
        "rope.freq_base": full["rope_theta"],
        "rope.sliding_freq_base": sliding["rope_theta"],
        "rope.scaling.factor": full["factor"],
        "rope.scaling.original_context_length": full["original_max_position_embeddings"],
        "rope.scaling.yarn_beta_fast": full["beta_fast"],
        "rope.scaling.yarn_beta_slow": full["beta_slow"],
        "stop_token_ids": generation["eos_token_id"],
    }.items():
        writer.add_meta(f"{ARCH}.{key}", value)

    def chunks(start, end, kind):
        with source.path.open("rb") as stream:
            stream.seek(source.data_offset + start)
            while start < end:
                size = min(end - start, 2**21)
                raw = stream.read(size)
                if len(raw) != size:
                    raise ValueError("truncated Aion tensor payload")
                yield encode(raw, kind)
                start += size

    spans = []
    for hf_name, (gguf_name, shape) in tensors.items():
        info = source.header[hf_name]
        start, end = info["data_offsets"]
        size = math.prod(shape)
        if (tuple(info["shape"]) != shape or info["dtype"] != "BF16" or start < 0
                or end - start != size * 2 or source.data_offset + end > source.file_size):
            raise ValueError(f"{hf_name} has unexpected dtype, shape or offset")
        spans.append((start, end))
        kind = precision if len(shape) == 2 else "F32"
        type_id, block, block_bytes = FORMATS[kind]
        if shape[-1] % block:
            raise ValueError(f"{hf_name} row width is incompatible with {kind}")
        writer.add_tensor_chunks(
            gguf_name, type_id, gguf_dims(shape), size // block * block_bytes,
            lambda start=start, end=end, kind=kind: chunks(start, end, kind),
        )
    spans.sort()
    if (spans[0][0] != 0 or spans[-1][1] + source.data_offset != source.file_size
            or any(left[1] != right[0] for left, right in zip(spans, spans[1:]))):
        raise ValueError("Aion tensor offsets overlap or leave gaps")

    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, prefix=output.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        writer.path = temporary
        writer.write()
        # Atomic publication with no overwrite, even with concurrent conversions.
        output.hardlink_to(temporary)
    finally:
        temporary.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--precision", choices=FORMATS, default="BF16")
    args = parser.parse_args()
    convert(args.model_dir, args.output, args.precision)
