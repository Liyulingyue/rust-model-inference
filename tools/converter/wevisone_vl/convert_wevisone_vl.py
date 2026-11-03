"""Export Tencent's released WeVisOne-VL-4B without discarding its pixel decoder.

Run with the repository .venv: python -m tools.converter.wevisone_vl.convert_wevisone_vl
MODEL_DIR --out-dir OUTPUT_DIR [--text-type bf16|f32|f16|q8_0|q4_k|q6_k].
The decoder and vision tower stay F32; only text matrices are quantized.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from tools.converter.mage_flow.convert_qwen3vl_vision import Reader, _patch_embeddings
from tools.converter.qwen_drive.convert_qwen_drive import sha256_file
from tools.converter.utils.gguf import GgufWriter, GGML_F32, gguf_dims
from tools.converter.utils.quantize_gguf import _quantize_one

ORACLE_COMMIT = "9a6bcd85404505df0d05cfb5d97512ccacf35dee"
MODEL_REVISION = "855d2a73ac0b929f668e6ac4f5deccc667aba99b"
DEEPSTACK = (5, 11, 17)
TEXT_NAMES = {
    "input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm",
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
    "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm",
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
}


def map_text_name(name: str) -> str:
    fixed = {"lm_head.weight": "output.weight",
             "model.language_model.embed_tokens.weight": "token_embd.weight",
             "model.language_model.norm.weight": "output_norm.weight"}
    if name in fixed:
        return fixed[name]
    match = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.(.+)\.weight", name)
    if match and match[2] in TEXT_NAMES:
        return f"blk.{match[1]}.{TEXT_NAMES[match[2]]}.weight"
    raise ValueError(f"unsupported WeVisOne text tensor: {name}")


def validate_config(config: dict) -> None:
    text, vision = config.get("text_config", {}), config.get("vision_config", {})
    if (config.get("model_type"), config.get("architectures")) != (
        "qwen3_vl", ["Qwen3VLForConditionalGeneration"]
    ):
        raise ValueError("expected the released WeVisOne Qwen3-VL architecture")
    expected = {"hidden_size": 2560, "num_hidden_layers": 36, "num_attention_heads": 32,
                "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 9728,
                "vocab_size": 152680, "hidden_act": "silu", "attention_bias": False}
    if any(text.get(k) != v for k, v in expected.items()):
        raise ValueError("unsupported WeVisOne-VL-4B text contract")
    rope = text.get("rope_parameters", text.get("rope_scaling", {}))
    if (rope.get("mrope_interleaved"), rope.get("mrope_section")) != (True, [24, 20, 20]):
        raise ValueError("expected interleaved M-RoPE sections [24,20,20]")
    expected = {"depth": 24, "hidden_size": 1024, "intermediate_size": 4096,
                "num_heads": 16, "patch_size": 16, "temporal_patch_size": 2,
                "spatial_merge_size": 2, "out_hidden_size": 2560,
                "num_position_embeddings": 2304, "deepstack_visual_indexes": list(DEEPSTACK)}
    if any(vision.get(k) != v for k, v in expected.items()):
        raise ValueError("unsupported WeVisOne-VL-4B vision contract")


def validate_tokenizer(tokenizer: dict) -> tuple[list[str], list[int]]:
    vocab = {int(i): s for s, i in tokenizer["model"]["vocab"].items()}
    added = tokenizer["added_tokens"]
    vocab.update({int(t["id"]): t["content"] for t in added})
    strings = set(vocab.values())
    if vocab.get(152678) != "<OBJ>" or any(f"<{i}>" not in strings for i in range(1001)):
        raise ValueError("missing WeVisOne coordinate tokens or <OBJ>=152678")
    if set(vocab) != set(range(152679)):
        raise ValueError("WeVisOne tokenizer must contain exactly 152679 contiguous IDs")
    # The released embedding/output have one padding row absent from tokenizer.json.
    vocab[152679] = ""
    types = [1] * len(vocab)
    types[152679] = 5
    for token in added:
        types[token["id"]] = 3 if token.get("special", False) else 4
    return [vocab[i] for i in range(len(vocab))], types


def _write(writer: GgufWriter) -> Path:
    output = writer.path
    if output.exists():
        raise FileExistsError(output)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(fd)
    writer.path = Path(temporary)
    try:
        writer.write()
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output


def _common(writer: GgufWriter, architecture: str, source_hash: str) -> None:
    writer.add_meta("general.architecture", architecture)
    writer.add_meta("general.name", "WeVisOne-VL-4B")
    writer.add_meta("wevisone_vl.source_revision", MODEL_REVISION)
    writer.add_meta("wevisone_vl.oracle_commit", ORACLE_COMMIT)
    writer.add_meta("wevisone_vl.source_sha256", source_hash)
    writer.add_meta("wevisone_vl.obj_token_id", 152678)


def export_text(root: Path, output: Path, reader: Reader, source_hash: str, dtype: str) -> Path:
    config = json.loads((root / "config.json").read_text())["text_config"]
    tokenizer = json.loads((root / "tokenizer.json").read_text())
    tokens, types = validate_tokenizer(tokenizer)
    writer = GgufWriter(output)
    _common(writer, "qwen3vl", source_hash)
    rope = config.get("rope_parameters", config.get("rope_scaling", {}))
    metadata = {
        "block_count": 36, "context_length": config["max_position_embeddings"],
        "embedding_length": 2560, "feed_forward_length": 9728,
        "attention.head_count": 32, "attention.head_count_kv": 8,
        "attention.key_length": 128, "attention.value_length": 128,
        "attention.layer_norm_rms_epsilon": config["rms_norm_eps"],
        "rope.dimension_count": 128, "rope.dimension_sections": [24, 20, 20, 0],
        "rope.freq_base": float(rope["rope_theta"]), "n_deepstack_layers": 3,
        "vocab_size": 152680,
    }
    for name, value in metadata.items():
        writer.add_meta(f"qwen3vl.{name}", value)
    for name, value in {
        "model": "gpt2", "pre": "qwen2", "tokens": tokens, "token_type": types,
        "merges": [" ".join(v) if isinstance(v, list) else v for v in tokenizer["model"]["merges"]],
        "eos_token_id": 151645, "padding_token_id": 151643,
        "add_bos_token": False, "add_eos_token": False,
    }.items():
        writer.add_meta(f"tokenizer.ggml.{name}", value)
    writer.add_meta("tokenizer.chat_template", (root / "chat_template.jinja").read_text())
    count = 0
    for name in reader.owners:
        if not (name.startswith("model.language_model.") or name == "lm_head.weight"):
            continue
        mapped = map_text_name(name)
        values = reader.f32(name)
        dims = gguf_dims(values.shape)
        raw = values.tobytes()
        target, raw = (GGML_F32, raw) if dtype == "f32" else _quantize_one(mapped, raw, GGML_F32, dims, dtype)
        writer.add_tensor(mapped, target, dims, raw)
        count += 1
    if count != 399:
        raise ValueError(f"expected 399 text tensors, found {count}")
    return _write(writer)


def export_vision(root: Path, output: Path, reader: Reader, source_hash: str) -> Path:
    writer = GgufWriter(output)
    _common(writer, "clip", source_hash)
    for key, value in {
        "image_size": 768, "patch_size": 16, "embedding_length": 1024,
        "feed_forward_length": 4096, "block_count": 24, "attention.head_count": 16,
        "attention.layer_norm_epsilon": 1e-6, "spatial_merge_size": 2,
        "projection_dim": 2560, "image_min_pixels": 16384, "image_max_pixels": 2073600,
        "image_mean": [0.5] * 3, "image_std": [0.5] * 3,
        "is_deepstack_layers": [i in DEEPSTACK for i in range(24)],
    }.items():
        writer.add_meta(f"clip.vision.{key}", value)
    writer.add_meta("clip.use_gelu", True)
    writer.add_meta("clip.projector_type", "qwen3vl_merger")
    for name, shape, raw in _patch_embeddings(reader):
        writer.add_tensor(name, GGML_F32, gguf_dims(shape), raw)
    for name in reader.owners:
        if not name.startswith("model.visual.") or name.endswith("patch_embed.proj.weight"):
            continue
        short = name.removeprefix("model.visual.")
        short = short.replace("patch_embed.proj.", "v.patch_embd.").replace("pos_embed.", "v.position_embd.")
        short = short.replace("merger.norm.", "v.post_ln.")
        short = short.replace("merger.linear_fc1.", "mm.0.").replace("merger.linear_fc2.", "mm.2.")
        match = re.fullmatch(r"blocks\.(\d+)\.(.+)", short)
        if match:
            suffix = match[2]
            for before, after in {"norm1": "ln1", "norm2": "ln2", "attn.qkv": "attn_qkv",
                                  "attn.proj": "attn_out", "mlp.linear_fc1": "ffn_up",
                                  "mlp.linear_fc2": "ffn_down"}.items():
                suffix = suffix.replace(before, after)
            short = f"v.blk.{match[1]}.{suffix}"
        match = re.fullmatch(r"deepstack_merger_list\.(\d+)\.(.+)", short)
        if match:
            suffix = match[2].replace("linear_fc1", "fc1").replace("linear_fc2", "fc2")
            short = f"v.deepstack.{DEEPSTACK[int(match[1])]}.{suffix}"
        if not short.startswith(("v.", "mm.")):
            raise ValueError(f"unsupported vision tensor: {name}")
        values = reader.f32(name)
        writer.add_tensor(short, GGML_F32, gguf_dims(values.shape), values.tobytes())
    if len(writer.tensors) != 316:
        raise ValueError(f"expected 316 vision tensors, found {len(writer.tensors)}")
    return _write(writer)


def export_decoder(sidecar: Path, output: Path) -> Path:
    import torch
    data = torch.load(sidecar, map_location="cpu", weights_only=True)
    state, cfg = data["state_dict"], data["cfg"]
    if (len(state), sum(v.numel() for v in state.values()), data.get("seg_token_id")) != (296, 9609764, 152678):
        raise ValueError("unsupported WeVisOne pixel-decoder checkpoint")
    if cfg.get("architecture_version") != "wevisone_vl_v1":
        raise ValueError("unsupported WeVisOne pixel-decoder version")
    writer = GgufWriter(output)
    _common(writer, "wevisone_vl", sha256_file(sidecar))
    writer.add_meta("wevisone_vl.config", json.dumps(cfg, sort_keys=True))
    for name, value in state.items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"pixel-decoder tensor must be finite F32: {name}")
        writer.add_tensor(name, GGML_F32, gguf_dims(tuple(value.shape)), value.contiguous().numpy().tobytes())
    return _write(writer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--text-type", choices=["bf16", "f32", "f16", "q8_0", "q4_k", "q6_k"], default="bf16")
    args = parser.parse_args()
    root = args.model_dir.resolve()
    validate_config(json.loads((root / "config.json").read_text()))
    validate_tokenizer(json.loads((root / "tokenizer.json").read_text()))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reader = Reader(root)
    source_hash = sha256_file(root / "model.safetensors")
    if source_hash != "002c3ac0783af7c86dc28b282c84dd4bcd0c29dcb3b5e0d584492fe3212acb4a":
        raise ValueError("model.safetensors does not match the pinned ModelScope release")
    print(export_text(root, args.out_dir / f"WeVisOne-VL-4B-{args.text_type}.gguf", reader, source_hash, args.text_type), flush=True)
    print(export_vision(root, args.out_dir / "mmproj-WeVisOne-VL-4B-F32.gguf", reader, source_hash), flush=True)
    print(export_decoder(root / "mask_decoder.pt", args.out_dir / "WeVisOne-VL-decoder-F32.gguf"), flush=True)


if __name__ == "__main__":
    main()
