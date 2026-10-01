"""Losslessly pack the merged Audio8-ASR-Infinite checkpoint into BF16 GGUF."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from tools.converter.utils.gguf import (
    GGML_BF16,
    GgufWriter,
    _read_gguf,
    gguf_dims,
    open_safetensors,
)


def source_contract(model_dir: Path):
    config = json.loads((model_dir / "config.json").read_text())
    if (config.get("model_type"), config.get("weight_format_version")) != (
        "audio8_asr_infinite", 2
    ):
        raise ValueError("expected Audio8 ASR Infinite merged weight format v2")
    if config["audio_config"]["model_type"] != "voxtral_realtime_encoder":
        raise ValueError("expected Voxtral Realtime audio tower")
    if config["text_config"]["model_type"] != "qwen2":
        raise ValueError("this exporter requires the supplied Qwen2 text decoder")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    if set(index.values()) != {"model.safetensors", "semantic_vad_heads.safetensors"}:
        raise ValueError("unexpected safetensors components")
    sources = {name: open_safetensors(model_dir / name) for name in set(index.values())}
    if set(index) != {
        name for source in sources.values() for name in source.header if name != "__metadata__"
    }:
        raise ValueError("safetensors index does not match tensor headers")
    for name, filename in index.items():
        source = sources[filename]
        info = source.header[name]
        start, end = info["data_offsets"]
        if (
            info["dtype"] != "BF16"
            or not info["shape"]
            or min(info["shape"]) <= 0
            or end - start != 2 * math.prod(info["shape"])
            or start < 0
            or source.data_offset + end > source.file_size
        ):
            raise ValueError(f"invalid BF16 tensor: {name}")
    required = {
        "audio_tower.embedder.conv1.weight": [1280, 128, 3],
        "audio_tower.layers.0.self_attn.q_proj.weight": [2048, 1280],
        "multi_modal_projector.linear_1.weight": [2048, 10240],
        "language_model.model.embed_tokens.weight": [151936, 2048],
        "language_model.model.layers.0.ada_rms_norm.linear1.weight": [32, 2048],
        "frame_len_embedding.weight": [3, 2048],
        "semantic_vad_heads.0.weight": [8, 2048],
    }
    for name, shape in required.items():
        if name not in index or sources[index[name]].header[name]["shape"] != shape:
            raise ValueError(f"unexpected tensor shape: {name}")
    return config, index, sources


def gguf_name(name: str) -> str:
    text = {
        "language_model.model.embed_tokens.weight": "token_embd.weight",
        "language_model.model.norm.weight": "output_norm.weight",
    }
    if name in text:
        return text[name]
    prefix = "language_model.model.layers."
    if not name.startswith(prefix):
        return name
    layer, _, suffix = name[len(prefix):].partition(".")
    names = {
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "self_attn.q_proj.bias": "attn_q.bias",
        "self_attn.k_proj.bias": "attn_k.bias",
        "self_attn.v_proj.bias": "attn_v.bias",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
        "ada_rms_norm.linear1.weight": "ada_linear1.weight",
        "ada_rms_norm.linear2.weight": "ada_linear2.weight",
    }
    if not layer.isdigit() or suffix not in names:
        raise ValueError(f"unmapped text tensor: {name}")
    return f"blk.{layer}.{names[suffix]}"


def tokenizer_metadata(writer: GgufWriter, model_dir: Path, vocab_size: int) -> None:
    tokenizer = json.loads((model_dir / "tokenizer.json").read_text())
    if tokenizer["model"]["type"] != "BPE" or tokenizer["normalizer"] != {"type": "NFC"}:
        raise ValueError("expected Qwen2 BPE tokenizer with NFC normalization")
    tokens = [None] * vocab_size
    types = [5] * vocab_size
    for token, token_id in tokenizer["model"]["vocab"].items():
        tokens[token_id], types[token_id] = token, 1
    for entry in tokenizer["added_tokens"]:
        token_id = entry["id"]
        if token_id >= vocab_size or tokens[token_id] is not None:
            raise ValueError(f"invalid added token id: {token_id}")
        tokens[token_id], types[token_id] = entry["content"], 3 if entry["special"] else 4
    tokens = [token if token is not None else f"<|reserved_{i}|>" for i, token in enumerate(tokens)]
    writer.add_meta("tokenizer.ggml.model", "gpt2")
    writer.add_meta("tokenizer.ggml.pre", "qwen2")
    writer.add_meta("tokenizer.ggml.tokens", tokens)
    writer.add_meta("tokenizer.ggml.token_type", types)
    writer.add_meta("tokenizer.ggml.merges", [" ".join(pair) for pair in tokenizer["model"]["merges"]])
    writer.add_meta("tokenizer.ggml.bos_token_id", 151644)
    writer.add_meta("tokenizer.ggml.eos_token_id", 151645)
    writer.add_meta("tokenizer.ggml.add_bos_token", False)
    writer.add_meta("tokenizer.ggml.add_eos_token", False)
    writer.add_meta("tokenizer.ggml.normalizer.nfc", True)


def chunks(path: Path, offset: int, length: int):
    with path.open("rb") as source:
        source.seek(offset)
        while length:
            block = source.read(min(length, 1 << 20))
            if not block:
                raise ValueError(f"truncated source: {path}")
            length -= len(block)
            yield block


def convert(model_dir: Path, output: Path) -> None:
    config, index, sources = source_contract(model_dir)
    if output.exists():
        raise FileExistsError(output)
    writer = GgufWriter(output.with_name(f".{output.name}.partial"))
    if writer.path.exists():
        raise FileExistsError(writer.path)
    writer.add_meta("general.architecture", "audio8_asr_infinite")
    writer.add_meta("general.name", "Audio8-ASR-Infinite")
    writer.add_meta("audio8_asr_infinite.config_json", json.dumps(config, separators=(",", ":")))
    text = config["text_config"]
    for key, value in {
        "embedding_length": text["hidden_size"],
        "block_count": text["num_hidden_layers"],
        "attention.head_count": text["num_attention_heads"],
        "attention.head_count_kv": text["num_key_value_heads"],
        "feed_forward_length": text["intermediate_size"],
        "context_length": text["max_position_embeddings"],
        "vocab_size": text["vocab_size"],
        "rope.freq_base": text["rope_parameters"]["rope_theta"],
        "attention.layer_norm_rms_epsilon": text["rms_norm_eps"],
    }.items():
        writer.add_meta(f"audio8_asr_infinite.{key}", value)
    tokenizer_metadata(writer, model_dir, config["text_config"]["vocab_size"])
    if len({gguf_name(name) for name in index}) != len(index):
        raise ValueError("GGUF tensor names collide")
    for name in sorted(index):
        source = sources[index[name]]
        info = source.header[name]
        start, end = info["data_offsets"]
        writer.add_tensor_chunks(
            gguf_name(name), GGML_BF16, gguf_dims(tuple(info["shape"])), end - start,
            lambda s=source, a=start, b=end: chunks(s.path, s.data_offset + a, b - a),
        )
    try:
        writer.write()
        verify(model_dir, writer.path)
        os.replace(writer.path, output)
    finally:
        writer.path.unlink(missing_ok=True)


def verify(model_dir: Path, output: Path) -> None:
    config, index, sources = source_contract(model_dir)
    metadata, tensors = _read_gguf(output)
    if (metadata.get("general.architecture") != "audio8_asr_infinite"
            or json.loads(metadata["audio8_asr_infinite.config_json"]) != config
            or set(tensors) != {gguf_name(name) for name in index}):
        raise ValueError("GGUF metadata or tensor inventory differs from source")
    with output.open("rb") as packed:
        for name in sorted(index):
            source = sources[index[name]]
            info = source.header[name]
            start, end = info["data_offsets"]
            kind, dims, length, offset = tensors[gguf_name(name)]
            if (kind, dims, length) != (GGML_BF16, gguf_dims(tuple(info["shape"])), end - start):
                raise ValueError(f"GGUF tensor contract differs: {name}")
            packed.seek(offset)
            for block in chunks(source.path, source.data_offset + start, length):
                if packed.read(len(block)) != block:
                    raise ValueError(f"GGUF tensor bytes differ: {name}")
    print(f"verified {len(index)} BF16 tensors: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    (verify if args.verify else convert)(args.model_dir, args.output)
