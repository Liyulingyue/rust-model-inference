#!/usr/bin/env python3
"""Export microsoft/VibeVoice-ASR-Streaming-7B to GGUF + mmproj.

Produces, next to the checkpoint:
  VibeVoice-ASR-Streaming-7B-Q8_0.gguf        — Qwen2.5-7B LLM (arch "qwen2",
                                                llama.cpp names, Q8_0)
  mmproj-VibeVoice-ASR-Streaming-7B-BF16.gguf — acoustic + semantic tokenizer
                                                encoders and speech connectors
                                                (arch "clip", BF16)

The acoustic tokenizer decoder is skipped: ASR never produces audio.

Torch-free: sharded safetensors are read via mmap and the Q8_0 blocks are
built with numpy (GGML layout: f16 scale + 32 int8 per block).

Usage:
  python3 tools/vibevoice/convert_vibevoice_asr.py models/VibeVoice-ASR-Streaming-7B [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dots"))
import convert_dots_tts as _dots  # noqa: E402
from convert_dots_tts import (  # noqa: E402
    GgufWriter,
    Tensor,
    validated_dir,
)
from convert_dots_tts import bf16_to_f32, gguf_dims  # noqa: E402

GGML_F32 = 0
GGML_BF16 = 30
GGML_Q8_0 = 8

Q8_BLOCK = 32
Q8_BLOCK_BYTES = 34  # f16 scale + 32 x int8
LLM_FILENAME = "VibeVoice-ASR-Streaming-7B-Q8_0.gguf"
MMPROJ_FILENAME = "mmproj-VibeVoice-ASR-Streaming-7B-BF16.gguf"


# --------------------------------------------------------------------------- #
# sharded safetensors reader
# --------------------------------------------------------------------------- #


class ShardedSafetensors:
    """Reads tensors from a model.safetensors.index.json checkpoint set.

    Only one shard file is kept open at a time; the index maps tensor names
    to shards up front so coverage can be validated before any conversion.
    """

    def __init__(self, model_dir: Path):
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"missing required input path: {index_path}")
        index = json.loads(index_path.read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self.model_dir = model_dir
        self._open_shard: str | None = None
        self._reader = None

    def tensor(self, name: str) -> Tensor:
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"tensor not in index: {name}")
        if self._open_shard != shard:
            if self._reader is not None:
                self._reader.close()
            self._reader = _open_single_shard(self.model_dir / shard)
            self._open_shard = shard
        assert self._reader is not None
        return self._reader.tensor(name)

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None
            self._open_shard = None


def require_tensor(
    reader: ShardedSafetensors,
    name: str,
    shape: tuple[int, ...],
    dtypes: tuple[str, ...] = ("BF16",),
) -> Tensor:
    tensor = reader.tensor(name)
    if tensor.dtype not in dtypes:
        raise ValueError(f"{name}: expected dtype {dtypes}, got {tensor.dtype}")
    if tensor.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {tensor.shape}")
    return tensor


def _open_single_shard(path: Path):
    from convert_dots_tts import open_safetensors

    reader = open_safetensors(path)
    reader._path_name = str(path)
    return reader


# --------------------------------------------------------------------------- #
# GGML Q8_0 quantization (numpy)
# --------------------------------------------------------------------------- #


def quantize_q8_0(values: np.ndarray) -> bytes:
    """GGML Q8_0: per 32-element block, f16 scale = amax/127, int8 payload.

    Rounding is round-half-away-from-zero to match ggml's roundf."""
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    if flat.size % Q8_BLOCK:
        raise ValueError(f"q8_0 payload {flat.size} is not a multiple of {Q8_BLOCK}")
    blocks = flat.reshape(-1, Q8_BLOCK)
    amax = np.max(np.abs(blocks), axis=1)
    scale = (amax / 127.0).astype(np.float16)
    scale_f32 = scale.astype(np.float32)
    safe = np.where(scale_f32 == 0.0, np.float32(1.0), scale_f32)
    scaled = blocks / safe[:, None]
    q = (np.floor(np.abs(scaled) + 0.5) * np.sign(scaled)).clip(-127, 127).astype(np.int8)
    out = np.empty((blocks.shape[0], Q8_BLOCK_BYTES), dtype=np.uint8)
    out[:, 0:2] = scale.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8).reshape(-1, Q8_BLOCK)
    return out.tobytes()


def emit_q8_0(gguf: GgufWriter, name: str, tensor: Tensor) -> None:
    if tensor.dtype != "BF16":
        raise ValueError(f"{name}: expected BF16, got {tensor.dtype}")
    raw = np.frombuffer(tensor.raw, dtype=np.uint16)
    # convert + quantize in slices so the biggest tensors stay well under a
    # couple of GB of transient memory
    step = 1 << 23  # 8M elements
    parts: list[bytes] = []
    for start in range(0, raw.size, step):
        chunk = raw[start : start + step]
        f32 = (chunk.astype(np.uint32) << np.uint32(16)).view(np.float32)
        parts.append(quantize_q8_0(f32))
    gguf.add_tensor(name, GGML_Q8_0, gguf_dims(tensor.shape), b"".join(parts))


def emit_bf16(gguf: GgufWriter, name: str, tensor: Tensor) -> None:
    if tensor.dtype != "BF16":
        raise ValueError(f"{name}: expected BF16, got {tensor.dtype}")
    gguf.add_tensor(name, GGML_BF16, gguf_dims(tensor.shape), tensor.raw)


def emit_f32(gguf: GgufWriter, name: str, tensor: Tensor) -> None:
    if tensor.dtype != "BF16":
        raise ValueError(f"{name}: expected BF16, got {tensor.dtype}")
    gguf.add_tensor(name, GGML_F32, gguf_dims(tensor.shape), bf16_to_f32(tensor.raw))


# --------------------------------------------------------------------------- #
# tokenizer metadata
# --------------------------------------------------------------------------- #


def add_tokenizer_metadata(gguf: GgufWriter, model_dir: Path, llm_cfg: dict) -> None:
    vocab = json.loads((model_dir / "vocab.json").read_text())
    added = json.loads((model_dir / "added_tokens.json").read_text())
    tok_cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
    merges = (model_dir / "merges.txt").read_text().splitlines()

    all_tokens: dict[int, str] = {int(tid): token for token, tid in vocab.items()}
    added_entries = [{"id": int(tid), "content": token} for token, tid in added.items()]
    for entry in sorted(added_entries, key=lambda e: e["id"]):
        all_tokens.setdefault(entry["id"], entry["content"])
    n_vocab = max(llm_cfg["vocab_size"], max(all_tokens) + 1)
    tokens = [all_tokens.get(i, f"<|reserved_{i}|>") for i in range(n_vocab)]
    token_types = [1] * n_vocab  # NORMAL
    for entry in added_entries:
        if entry["id"] < n_vocab:
            token_types[entry["id"]] = 3  # CONTROL

    gguf.add_meta("tokenizer.ggml.model", "gpt2")
    gguf.add_meta("tokenizer.ggml.pre", "qwen2")
    gguf.add_meta("tokenizer.ggml.tokens", tokens)
    gguf.add_meta("tokenizer.ggml.token_type", token_types)
    gguf.add_meta("tokenizer.ggml.merges", merges)
    gguf.add_meta("tokenizer.ggml.bos_token_id", tok_cfg.get("bos_token_id") or 151643)
    gguf.add_meta("tokenizer.ggml.eos_token_id", tok_cfg.get("eos_token_id") or 151643)
    gguf.add_meta("tokenizer.ggml.add_bos_token", False)
    gguf.add_meta("tokenizer.ggml.add_eos_token", False)
    return n_vocab


# --------------------------------------------------------------------------- #
# LLM export (arch qwen2, Q8_0)
# --------------------------------------------------------------------------- #


def export_llm(model_dir: Path, out_path: Path, shards: ShardedSafetensors, overwrite: bool) -> None:
    cfg = json.loads((model_dir / "config.json").read_text())
    llm_cfg = cfg["decoder_config"]
    n_layer = llm_cfg["num_hidden_layers"]
    n_embd = llm_cfg["hidden_size"]
    n_head = llm_cfg["num_attention_heads"]
    n_kv = llm_cfg["num_key_value_heads"]
    n_ff = llm_cfg["intermediate_size"]
    if n_embd % n_head:
        raise ValueError(f"hidden_size {n_embd} is not divisible by heads {n_head}")
    n_embd_head = n_embd // n_head
    n_kv_embd = n_kv * n_embd_head

    gguf = GgufWriter(out_path)
    gguf.add_meta("general.architecture", "qwen2")
    gguf.add_meta("general.name", "VibeVoice-ASR-Streaming-7B")
    gguf.add_meta("general.file_type", 7)  # mostly Q8_0
    gguf.add_meta("general.quantization_version", 2)
    gguf.add_meta("qwen2.block_count", n_layer)
    gguf.add_meta("qwen2.context_length", llm_cfg["max_position_embeddings"])
    gguf.add_meta("qwen2.embedding_length", n_embd)
    gguf.add_meta("qwen2.feed_forward_length", n_ff)
    gguf.add_meta("qwen2.attention.head_count", n_head)
    gguf.add_meta("qwen2.attention.head_count_kv", n_kv)
    gguf.add_meta("qwen2.attention.layer_norm_rms_epsilon", llm_cfg["rms_norm_eps"])
    gguf.add_meta("qwen2.rope.dimension_count", n_embd_head)
    gguf.add_meta("qwen2.rope.freq_base", llm_cfg.get("rope_theta", 1_000_000.0))
    n_vocab = add_tokenizer_metadata(gguf, model_dir, llm_cfg)
    gguf.add_meta("qwen2.vocab_size", n_vocab)

    emit_q8_0(
        gguf,
        "token_embd.weight",
        require_tensor(
            shards,
            "model.language_model.embed_tokens.weight",
            (llm_cfg["vocab_size"], n_embd),
        ),
    )
    emit_q8_0(
        gguf,
        "output.weight",
        require_tensor(shards, "lm_head.weight", (llm_cfg["vocab_size"], n_embd)),
    )
    emit_f32(
        gguf,
        "output_norm.weight",
        require_tensor(shards, "model.language_model.norm.weight", (n_embd,)),
    )

    layer_map = {
        "input_layernorm.weight": ("attn_norm.weight", "f32", (n_embd,)),
        "post_attention_layernorm.weight": ("ffn_norm.weight", "f32", (n_embd,)),
        "self_attn.q_proj.weight": ("attn_q.weight", "q8", (n_embd, n_embd)),
        "self_attn.k_proj.weight": ("attn_k.weight", "q8", (n_kv_embd, n_embd)),
        "self_attn.v_proj.weight": ("attn_v.weight", "q8", (n_kv_embd, n_embd)),
        "self_attn.q_proj.bias": ("attn_q.bias", "f32", (n_embd,)),
        "self_attn.k_proj.bias": ("attn_k.bias", "f32", (n_kv_embd,)),
        "self_attn.v_proj.bias": ("attn_v.bias", "f32", (n_kv_embd,)),
        "self_attn.o_proj.weight": ("attn_output.weight", "q8", (n_embd, n_embd)),
        "mlp.gate_proj.weight": ("ffn_gate.weight", "q8", (n_ff, n_embd)),
        "mlp.up_proj.weight": ("ffn_up.weight", "q8", (n_ff, n_embd)),
        "mlp.down_proj.weight": ("ffn_down.weight", "q8", (n_embd, n_ff)),
    }
    for layer in range(n_layer):
        for src_key, (dst_key, kind, shape) in layer_map.items():
            tensor = require_tensor(
                shards,
                f"model.language_model.layers.{layer}.{src_key}",
                shape,
            )
            dst = f"blk.{layer}.{dst_key}"
            if kind == "f32":
                emit_f32(gguf, dst, tensor)
            else:
                emit_q8_0(gguf, dst, tensor)
    gguf.write(overwrite=overwrite)
    print(f"wrote {out_path} ({len(gguf.tensors)} tensors)")


# --------------------------------------------------------------------------- #
# mmproj export (arch clip, BF16 encoders + connectors)
# --------------------------------------------------------------------------- #

ENCODER_DEPTHS = [3, 3, 3, 3, 3, 3, 8]


def encoder_tensor_rules(side: str) -> list[tuple[str, str]]:
    """(source suffix under model.<side>_tokenizer.encoder, destination
    suffix under vibevoice.<side>.encoder) for every encoder tensor."""
    rules: list[tuple[str, str]] = []
    for stage in range(len(ENCODER_DEPTHS)):
        rules.append((f"downsample_layers.{stage}.0.conv.conv.weight", f"downsample.{stage}.conv.weight"))
        rules.append((f"downsample_layers.{stage}.0.conv.conv.bias", f"downsample.{stage}.conv.bias"))
        for block in range(ENCODER_DEPTHS[stage]):
            base = f"stages.{stage}.{block}"
            rules.append((f"{base}.norm.weight", f"{base}.norm.weight"))
            rules.append((f"{base}.mixer.conv.conv.conv.weight", f"{base}.mixer.weight"))
            rules.append((f"{base}.mixer.conv.conv.conv.bias", f"{base}.mixer.bias"))
            rules.append((f"{base}.gamma", f"{base}.gamma"))
            rules.append((f"{base}.ffn_norm.weight", f"{base}.ffn_norm.weight"))
            rules.append((f"{base}.ffn.linear1.weight", f"{base}.ffn_linear1.weight"))
            rules.append((f"{base}.ffn.linear1.bias", f"{base}.ffn_linear1.bias"))
            rules.append((f"{base}.ffn.linear2.weight", f"{base}.ffn_linear2.weight"))
            rules.append((f"{base}.ffn.linear2.bias", f"{base}.ffn_linear2.bias"))
            rules.append((f"{base}.ffn_gamma", f"{base}.ffn_gamma"))
    rules.append(("head.conv.conv.weight", "head.conv.weight"))
    rules.append(("head.conv.conv.bias", "head.conv.bias"))
    return rules


def encoder_tensor_shapes(cfg_side: dict, vae_dim: int) -> dict[str, tuple[int, ...]]:
    n_filters = int(cfg_side["encoder_n_filters"])
    strides = [int(value) for value in reversed(cfg_side["encoder_ratios"])]
    shapes: dict[str, tuple[int, ...]] = {}
    for stage, depth in enumerate(ENCODER_DEPTHS):
        n_in = 1 if stage == 0 else n_filters << (stage - 1)
        n_out = n_filters << stage
        kernel = 7 if stage == 0 else strides[stage - 1] * 2
        shapes[f"downsample_layers.{stage}.0.conv.conv.weight"] = (n_out, n_in, kernel)
        shapes[f"downsample_layers.{stage}.0.conv.conv.bias"] = (n_out,)
        ffn = n_out * 4
        for block in range(depth):
            base = f"stages.{stage}.{block}"
            shapes[f"{base}.norm.weight"] = (n_out,)
            shapes[f"{base}.mixer.conv.conv.conv.weight"] = (n_out, 1, 7)
            shapes[f"{base}.mixer.conv.conv.conv.bias"] = (n_out,)
            shapes[f"{base}.gamma"] = (n_out,)
            shapes[f"{base}.ffn_norm.weight"] = (n_out,)
            shapes[f"{base}.ffn.linear1.weight"] = (ffn, n_out)
            shapes[f"{base}.ffn.linear1.bias"] = (ffn,)
            shapes[f"{base}.ffn.linear2.weight"] = (n_out, ffn)
            shapes[f"{base}.ffn.linear2.bias"] = (n_out,)
            shapes[f"{base}.ffn_gamma"] = (n_out,)
    shapes["head.conv.conv.weight"] = (
        vae_dim,
        n_filters << (len(ENCODER_DEPTHS) - 1),
        7,
    )
    shapes["head.conv.conv.bias"] = (vae_dim,)
    return shapes


def export_mmproj(model_dir: Path, out_path: Path, shards: ShardedSafetensors, overwrite: bool) -> None:
    cfg = json.loads((model_dir / "config.json").read_text())
    preproc = json.loads((model_dir / "preprocessor_config.json").read_text())
    acoustic = cfg["acoustic_tokenizer_config"]
    semantic = cfg["semantic_tokenizer_config"]
    decoder_cfg = cfg["decoder_config"]
    n_embd = decoder_cfg["hidden_size"]

    gguf = GgufWriter(out_path)
    gguf.add_meta("general.architecture", "clip")
    gguf.add_meta("general.name", "VibeVoice-ASR-Streaming-7B-mmproj")
    gguf.add_meta("general.file_type", 32)  # mostly BF16
    gguf.add_meta("clip.has_vision_encoder", False)
    gguf.add_meta("clip.has_audio_encoder", True)
    gguf.add_meta("clip.has_gen_audio_encoder", False)
    gguf.add_meta("clip.audio.projector_type", "vibevoice_asr")

    gguf.add_meta("vibevoice.sample_rate", cfg["target_sample_rate"])
    gguf.add_meta("vibevoice.compress_ratio", cfg["speech_tok_compress_ratio"])
    gguf.add_meta("vibevoice.chunk_frames", preproc["chunk_frames"])
    gguf.add_meta("vibevoice.lookahead_frames", preproc["lookahead_frames"])
    gguf.add_meta("vibevoice.llm_hidden_size", n_embd)
    gguf.add_meta("vibevoice.acoustic.vae_dim", cfg["acoustic_vae_dim"])
    gguf.add_meta("vibevoice.semantic.vae_dim", cfg["semantic_vae_dim"])
    gguf.add_meta("vibevoice.encoder.n_filters", acoustic["encoder_n_filters"])
    gguf.add_meta("vibevoice.encoder.kernel_size", 7)
    gguf.add_meta("vibevoice.encoder.last_kernel_size", 7)
    gguf.add_meta("vibevoice.encoder.ffn_expansion", 4)
    gguf.add_meta("vibevoice.encoder.pad_mode", acoustic["pad_mode"])
    gguf.add_meta("vibevoice.encoder.causal", bool(acoustic["causal"]))
    gguf.add_meta("vibevoice.encoder.mixer_layer", acoustic["mixer_layer"])
    gguf.add_meta("vibevoice.encoder.layernorm_eps", acoustic["layernorm_eps"])
    gguf.add_meta("vibevoice.connector.eps", 1e-6)
    gguf.add_meta("vibevoice.acoustic.fix_std", acoustic["fix_std"])
    gguf.add_meta("vibevoice.acoustic.std_dist_type", acoustic["std_dist_type"])
    for key, value in [
        ("vibevoice.encoder.ratios", acoustic["encoder_ratios"]),
        ("vibevoice.encoder.depths", ENCODER_DEPTHS),
    ]:
        gguf.add_meta(key, [int(v) for v in value])

    for side, cfg_side in (("acoustic", acoustic), ("semantic", semantic)):
        if cfg_side["vae_dim"] != cfg[f"{side}_vae_dim"]:
            raise ValueError(f"{side} vae_dim mismatch")
        src_base = f"model.{side}_tokenizer.encoder"
        dst_base = f"vibevoice.{side}.encoder"
        shapes = encoder_tensor_shapes(cfg_side, cfg[f"{side}_vae_dim"])
        for src_suffix, dst_suffix in encoder_tensor_rules(side):
            emit_bf16(
                gguf,
                f"{dst_base}.{dst_suffix}",
                require_tensor(shards, f"{src_base}.{src_suffix}", shapes[src_suffix]),
            )
        conn_base = f"vibevoice.{side}.connector"
        input_dim = cfg[f"{side}_vae_dim"]
        for part, shape in (("fc1", (n_embd, input_dim)), ("fc2", (n_embd, n_embd))):
            emit_bf16(
                gguf,
                f"{conn_base}.{part}.weight",
                require_tensor(shards, f"model.{side}_connector.{part}.weight", shape),
            )
            emit_bf16(
                gguf,
                f"{conn_base}.{part}.bias",
                require_tensor(shards, f"model.{side}_connector.{part}.bias", (n_embd,)),
            )
        emit_bf16(
            gguf,
            f"{conn_base}.norm.weight",
            require_tensor(shards, f"model.{side}_connector.norm.weight", (n_embd,)),
        )

    gguf.write(overwrite=overwrite)
    print(f"wrote {out_path} ({len(gguf.tensors)} tensors)")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def output_paths(out_dir: Path) -> tuple[Path, Path]:
    return out_dir / LLM_FILENAME, out_dir / MMPROJ_FILENAME


def export_model(model_dir: Path, out_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    model_dir = model_dir.resolve()
    out_dir = out_dir.resolve()
    for name in ("model.safetensors.index.json", "config.json", "vocab.json"):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(f"missing required input path: {model_dir / name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    llm_path, mmproj_path = output_paths(out_dir)
    if not overwrite and (llm_path.exists() or mmproj_path.exists()):
        existing = llm_path if llm_path.exists() else mmproj_path
        raise FileExistsError(f"output already exists: {existing}")

    print(f"exporting VibeVoice ASR from {model_dir}")
    shards = ShardedSafetensors(model_dir)
    try:
        export_llm(model_dir, llm_path, shards, overwrite)
        export_mmproj(model_dir, mmproj_path, shards, overwrite)
        covered = set(shards.weight_map)
        # the acoustic tokenizer decoder is intentionally not exported: ASR
        # never synthesizes audio
        skipped = {name for name in covered if ".decoder." in name}
        exported = {
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
            "model.language_model.norm.weight",
        }
        n_layer = json.loads((model_dir / "config.json").read_text())["decoder_config"]["num_hidden_layers"]
        layer_sources = (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias",
            "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        )
        for layer in range(n_layer):
            exported |= {f"model.language_model.layers.{layer}.{src}" for src in layer_sources}
        for side in ("acoustic", "semantic"):
            base = f"model.{side}_tokenizer.encoder"
            exported |= {f"{base}.{src}" for src, _ in encoder_tensor_rules(side)}
            exported |= {
                f"model.{side}_connector.fc1.weight", f"model.{side}_connector.fc1.bias",
                f"model.{side}_connector.fc2.weight", f"model.{side}_connector.fc2.bias",
                f"model.{side}_connector.norm.weight",
            }
        missing_export = (covered - skipped) - exported
        if missing_export:
            raise ValueError(
                f"{len(missing_export)} checkpoint tensors were not exported, e.g. "
                f"{sorted(missing_export)[:5]}"
            )
        print(
            f"coverage: {len(exported)} checkpoint tensors exported, "
            f"{len(skipped)} acoustic-decoder tensors intentionally skipped"
        )
    finally:
        shards.close()
    return llm_path, mmproj_path


def main() -> None:
    parser = argparse.ArgumentParser(description="export VibeVoice ASR to GGUF + mmproj")
    parser.add_argument("model_dir", type=str, help="models/VibeVoice-ASR-Streaming-7B")
    parser.add_argument("--out-dir", default=None, help="output directory (default: model dir parent)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = parser.parse_args()
    model_dir = validated_dir(args.model_dir, must_exist=True)
    out_dir = validated_dir(args.out_dir or str(model_dir.parent), must_exist=False)
    export_model(model_dir, out_dir, args.overwrite)


if __name__ == "__main__":
    main()
