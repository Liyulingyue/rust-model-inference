"""Export physical SenseNova Loop weights; preserve source precision and FP32 mHC."""

from __future__ import annotations
import argparse
import json
import math
import re
import hashlib
from pathlib import Path

ARCH = "sensenova_loop"
STEMS = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
    "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
    "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.conv1d.weight": "ssm_conv1d.weight",
    "linear_attn.A_log": "ssm_a",
    "linear_attn.dt_bias": "ssm_dt.bias",
    "linear_attn.norm.weight": "ssm_norm.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    **{
        f"{kind}.{suffix}": f"{kind}.{suffix}"
        for kind in ("mhc_attn", "mhc_ffn")
        for suffix in ("weight", "bias")
    },
}


#: The only tensors a --quant mode re-encodes: the linear projections.
#: Norms, biases, the depthwise convolution and the SSM state/alpha/beta stay F32.
LINEAR_PROJECTIONS = (
    "token_embd.weight",
    "output.weight",
    ".attn_qkv.weight",
    ".attn_q.weight",
    ".attn_k.weight",
    ".attn_v.weight",
    ".attn_output.weight",
    ".attn_gate.weight",
    ".ssm_out.weight",
    ".ffn_gate.weight",
    ".ffn_up.weight",
    ".ffn_down.weight",
    ".mhc_attn.weight",
    ".mhc_ffn.weight",
)


def execution_plan(config):
    text = config["text_config"]
    loop = config["loop_settings"]
    n, start, end, count = (
        text["num_hidden_layers"],
        loop["start"],
        loop["end"],
        loop["eval_count"],
    )
    if (
        any(type(v) is not int for v in (n, start, end, count))
        or not 0 <= start < end <= n
        or count < 1
        or n + (end - start) * (count - 1) > 4096
        or loop.get("counts") != [count]
    ):
        raise ValueError("invalid or oversized Loop execution plan")
    types = text["layer_types"]
    if (
        len(types) != n
        or set(types) - {"full_attention", "linear_attention"}
        or "full_attention" not in types
    ):
        raise ValueError("invalid physical layer types")
    mhc = config.get("mhc")
    if not isinstance(mhc, dict):
        raise ValueError("explicit MHC configuration required")
    a, b, s, it = (mhc.get(k) for k in ("start", "end", "streams", "iterations"))
    if (
        any(type(v) is not int for v in (a, b, s, it))
        or not 0 <= a < b <= n
        or s not in (2, 4)
        or not 1 <= it <= 1024
        or type(mhc.get("fixed_identity")) is not bool
        or not any(
            left <= a < b <= right
            for left, right in zip((0, start, end), (start, end, n))
        )
    ):
        raise ValueError("invalid MHC boundaries or routing settings")
    scale = mhc.get("pre_scale", 1.0)
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid MHC pre_scale")
    if set(mhc) - {
        "start",
        "end",
        "streams",
        "iterations",
        "fixed_identity",
        "pre_scale",
    }:
        raise ValueError("unknown MHC fields")
    return list(range(start)) + list(range(start, end)) * count + list(range(end, n))


def tensor_name(name):
    top = {
        "model.language_model.embed_tokens.weight": "token_embd.weight",
        "model.language_model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    if name in top:
        return top[name]
    match = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.(.+)", name)
    if match and match[2] in STEMS:
        return f"blk.{match[1]}.{STEMS[match[2]]}"
    raise ValueError(f"unknown SenseNova text tensor: {name}")


def validate_config(config):
    execution_plan(config)
    text, vision = config["text_config"], config["vision_config"]
    for key, value in {
        "hidden_act": "silu",
        "attention_bias": False,
        "attn_output_gate": True,
        "tie_word_embeddings": False,
    }.items():
        if text.get(key) != value:
            raise ValueError(f"unsupported text_config.{key}")
    rope = text["rope_parameters"]
    if (
        rope.get("rope_type") != "default"
        or rope.get("mrope_interleaved") is not True
        or rope.get("partial_rotary_factor") != text["partial_rotary_factor"]
        or len(rope["mrope_section"]) != 3
        or any(type(v) is not int or v < 0 for v in rope["mrope_section"])
        or 2 * sum(rope["mrope_section"])
        != text["head_dim"] * text["partial_rotary_factor"]
    ):
        raise ValueError("unsupported text RoPE configuration")
    if (
        text["linear_key_head_dim"] != text["linear_value_head_dim"]
        or text["linear_num_value_heads"] % text["linear_num_key_heads"]
    ):
        raise ValueError("unsupported GDN head dimensions")
    if (
        vision.get("deepstack_visual_indexes")
        or vision.get("temporal_patch_size") != 2
        or vision.get("hidden_act") != "gelu_pytorch_tanh"
        or vision.get("in_channels") != 3
        or vision["out_hidden_size"] != text["hidden_size"]
        or config.get("patch_conv3d") is not True
        or math.isqrt(vision["num_position_embeddings"]) ** 2
        != vision["num_position_embeddings"]
    ):
        raise ValueError("unsupported vision configuration")


def add_metadata(writer, root, config):
    text = config["text_config"]
    rope = text["rope_parameters"]
    writer.add_meta("general.architecture", ARCH)
    writer.add_meta("general.name", config["model_name"])
    writer.add_meta(
        f"{ARCH}.source_config_sha256",
        hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
    )
    values = {
        "block_count": text["num_hidden_layers"],
        "context_length": text["max_position_embeddings"],
        "embedding_length": text["hidden_size"],
        "feed_forward_length": text["intermediate_size"],
        "attention.head_count": text["num_attention_heads"],
        "attention.head_count_kv": text["num_key_value_heads"],
        "attention.key_length": text["head_dim"],
        "attention.value_length": text["head_dim"],
        "attention.layer_norm_rms_epsilon": text["rms_norm_eps"],
        "attention.recurrent_layers": [
            v == "linear_attention" for v in text["layer_types"]
        ],
        "rope.dimension_count": int(text["head_dim"] * text["partial_rotary_factor"]),
        "rope.dimension_sections": [*rope["mrope_section"], 0],
        "rope.freq_base": float(rope["rope_theta"]),
        "ssm.conv_kernel": text["linear_conv_kernel_dim"],
        "ssm.state_size": text["linear_key_head_dim"],
        "ssm.group_count": text["linear_num_key_heads"],
        "ssm.time_step_rank": text["linear_num_value_heads"],
        "ssm.inner_size": text["linear_num_value_heads"]
        * text["linear_value_head_dim"],
        "full_attention_interval": text["full_attention_interval"],
        "vocab_size": text["vocab_size"],
        "config": json.dumps(
            {
                "text_config": text,
                "loop_settings": config["loop_settings"],
                "mhc": config["mhc"],
            },
            separators=(",", ":"),
        ),
    }
    for key, value in values.items():
        writer.add_meta(f"{ARCH}.{key}", value)
    tok = json.loads((root / "tokenizer.json").read_text())
    vocab = {i: t for t, i in tok["model"]["vocab"].items()}
    added = tok["added_tokens"]
    vocab.update({t["id"]: t["content"] for t in added})
    tokens = [vocab.get(i, f"<|reserved_{i}|>") for i in range(text["vocab_size"])]
    types = [1 if i in vocab else 5 for i in range(len(tokens))]
    for t in added:
        types[t["id"]] = 3 if t["special"] else 4
    if tok["model"]["type"] != "BPE" or tok.get("normalizer") != {"type": "NFC"}:
        raise ValueError("expected Qwen3.5 BPE/NFC tokenizer")
    merges = tok["model"]["merges"]
    for key, value in {
        "model": "gpt2",
        "pre": "qwen2",
        "tokens": tokens,
        "token_type": types,
        "merges": [" ".join(p) if isinstance(p, list) else p for p in merges],
        "eos_token_id": config["eos_token_id"],
        "add_bos_token": False,
        "add_eos_token": False,
        "normalizer.nfc": True,
    }.items():
        writer.add_meta("tokenizer.ggml." + key, value)
    writer.add_meta(
        "tokenizer.chat_template", (root / "chat_template.jinja").read_text()
    )


def convert(root, output, mmproj, quant="f32"):
    import numpy as np
    from tools.converter.mage_flow.convert_qwen3vl_vision import (
        Reader,
        _patch_embeddings,
    )
    from tools.converter.ovisocr3.convert_ovisocr3 import expected_tensor_shapes
    from tools.converter.utils.gguf import GgufWriter, GGML_F32, gguf_dims

    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != ARCH or config.get("architectures") != [
        "SenseNovaLoopForConditionalGeneration"
    ]:
        raise ValueError("not a SenseNova Loop checkpoint")
    validate_config(config)
    text = config["text_config"]
    vision = config["vision_config"]
    reader = Reader(root)
    expected = expected_tensor_shapes(config)
    h = text["hidden_size"]
    s = config["mhc"]["streams"]
    outputs = 2 * s + (0 if config["mhc"]["fixed_identity"] else s * s)
    expected["lm_head.weight"] = (text["vocab_size"], h)
    for layer in range(config["mhc"]["start"], config["mhc"]["end"]):
        for kind in ("mhc_attn", "mhc_ffn"):
            expected[f"model.language_model.layers.{layer}.{kind}.weight"] = (
                outputs,
                s * h,
            )
            expected[f"model.language_model.layers.{layer}.{kind}.bias"] = (outputs,)
    actual = {name for name in reader.owners if not name.startswith("mtp.")}
    if actual != set(expected):
        raise ValueError(
            f"tensor inventory mismatch: missing={set(expected) - actual}, unexpected={actual - set(expected)}"
        )
    for name, shape in expected.items():
        dtype = reader.dtype(name)
        if reader.shape(name) != shape or dtype != "F32":
            raise ValueError(
                f"{name}: invalid shape or precision: {reader.shape(name)}, {dtype}"
            )
    if output.resolve() == mmproj.resolve():
        raise ValueError("text and projector outputs must differ")
    for path in (
        output,
        mmproj,
        output.with_suffix(".gguf.partial"),
        mmproj.with_suffix(".gguf.partial"),
    ):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    writer = GgufWriter(output.with_suffix(".gguf.partial"))
    add_metadata(writer, root, config)
    keep: set[str] = set()
    for name in expected:
        if name.startswith("model.visual."):
            continue
        target = tensor_name(name)
        shape = reader.shape(name)
        transform = (
            target.endswith(
                (
                    "attn_norm.weight",
                    "post_attention_norm.weight",
                    "attn_q_norm.weight",
                    "attn_k_norm.weight",
                )
            )
            or target == "output_norm.weight"
            or target.endswith(".ssm_a")
        )
        if not any(target.endswith(suffix) for suffix in LINEAR_PROJECTIONS):
            # Everything that is not a linear projection (norms, biases, the
            # depthwise convolution, the SSM state/alpha/beta) stays F32 in
            # every --quant mode.
            keep.add(target)
        if transform:
            values = reader.f32(name)
            values = (
                -np.exp(values) if target.endswith(".ssm_a") else values + np.float32(1)
            )
            writer.add_tensor(
                target,
                GGML_F32,
                gguf_dims(values.shape),
                values.astype("<f4").tobytes(),
            )
        elif ".ssm_conv1d." in target:
            writer.add_tensor(
                target,
                GGML_F32,
                gguf_dims((shape[0], shape[2])),
                reader.f32(name).tobytes(),
            )
        else:
            size = np.prod(shape).item() * 4

            def chunks(name=name):
                yield reader.raw(name)

            writer.add_tensor_chunks(target, GGML_F32, gguf_dims(shape), size, chunks)
    writer.write()
    text_partial = writer.path
    # The existing vision encoder consumes Qwen3.5's physical tower unchanged.
    writer = GgufWriter(mmproj.with_suffix(".gguf.partial"))
    pre = json.loads((root / "preprocessor_config.json").read_text())
    v = vision["hidden_size"]
    merge = vision["spatial_merge_size"]
    metadata = {
        "general.architecture": "clip",
        "general.type": "mmproj",
        "clip.projector_type": "qwen3vl_merger",
        f"{ARCH}.source_config_sha256": hashlib.sha256(
            (root / "config.json").read_bytes()
        ).hexdigest(),
        "clip.vision.image_size": int(math.isqrt(vision["num_position_embeddings"]))
        * vision["patch_size"],
        "clip.vision.patch_size": vision["patch_size"],
        "clip.vision.embedding_length": v,
        "clip.vision.feed_forward_length": vision["intermediate_size"],
        "clip.vision.block_count": vision["depth"],
        "clip.vision.attention.head_count": vision["num_heads"],
        "clip.vision.attention.layer_norm_epsilon": 1e-6,
        "clip.vision.spatial_merge_size": merge,
        "clip.vision.projection_dim": vision["out_hidden_size"],
        "clip.vision.image_min_pixels": pre["size"]["shortest_edge"],
        "clip.vision.image_max_pixels": pre["size"]["longest_edge"],
        "clip.vision.image_mean": pre["image_mean"],
        "clip.vision.image_std": pre["image_std"],
        "clip.use_gelu": True,
    }
    for key, value in metadata.items():
        writer.add_meta(key, value)

    def put(target, name):
        values = reader.f32(name)
        writer.add_tensor(target, GGML_F32, gguf_dims(values.shape), values.tobytes())

    for target, shape, raw in _patch_embeddings(reader):
        writer.add_tensor(
            target, GGML_F32, (vision["patch_size"], vision["patch_size"], 3, v), raw
        )
    put("v.patch_embd.bias", "model.visual.patch_embed.proj.bias")
    put("v.position_embd.weight", "model.visual.pos_embed.weight")
    stems = {
        "norm1": "ln1",
        "norm2": "ln2",
        "attn.qkv": "attn_qkv",
        "attn.proj": "attn_out",
        "mlp.linear_fc1": "ffn_up",
        "mlp.linear_fc2": "ffn_down",
    }
    for layer in range(vision["depth"]):
        for stem, target in stems.items():
            for suffix in ("weight", "bias"):
                put(
                    f"v.blk.{layer}.{target}.{suffix}",
                    f"model.visual.blocks.{layer}.{stem}.{suffix}",
                )
    for stem, target in {
        "norm": "v.post_ln",
        "linear_fc1": "mm.0",
        "linear_fc2": "mm.2",
    }.items():
        for suffix in ("weight", "bias"):
            put(f"{target}.{suffix}", f"model.visual.merger.{stem}.{suffix}")
    writer.write()
    writer.path.rename(mmproj)
    if quant == "f32":
        text_partial.rename(output)
    else:
        quantized = output.with_name(output.name + ".quantized")
        from tools.converter.utils.quantize_gguf import quantize_gguf

        quantize_gguf(
            text_partial, quantized, target_format=quant, keep=tuple(sorted(keep))
        )
        quantized.rename(output)
        text_partial.unlink()


if __name__ == "__main__":
    from tools.converter.utils.quantize_gguf import BLOCK_FORMATS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--mmproj-outfile", type=Path, required=True)
    parser.add_argument(
        "--quant",
        default="f32",
        choices=["f32", *sorted(BLOCK_FORMATS)],
        help="block-quantize the text projections (norms and biases stay F32)",
    )
    args = parser.parse_args()
    convert(args.model, args.outfile, args.mmproj_outfile, args.quant)
