#!/usr/bin/env python3
"""Convert `google/embeddinggemma-2` (multimodal encoder: text + vision + audio)
into a single GGUF the engine can load via the `gemma4-embedding` arch path.

Inputs:
* HF safetensors checkpoint for `google/embeddinggemma-2` containing a single
  `model.safetensors` (1.5 GB BF16) with five top-level sections:
  - `language_model.*` (413 tensors): 24-layer gemma4-text backbone:
      * `language_model.embed_tokens.weight` [262144, 512]
      * `language_model.embedding_projection.weight` [768, 512]
        (text → 768-dim embedding space)
      * `language_model.layers.{0..23}`:
        `input_layernorm`, `layer_scalar`, `self_attn.{q,k,v,o}_proj.{weight}`,
        `self_attn.{q,k}_norm.{weight}`, `ple_block.{per_layer_input_gate,
        per_layer_projection,post_per_layer_input_norm}.{weight}`,
        `pre_feedforward_layernorm`, `post_feedforward_layernorm`,
        `post_attention_layernorm`, `mlp.{gate,up,down}_proj.{weight}`.
      * `language_model.norm.weight` (final pre-head RMSNorm).
      * `language_model.ple.per_layer_model_projection.weight` [12288, 512]
        + `language_model.ple.per_layer_projection_norm.weight` [512]
        (full-vocab per-layer projection shared across all layers).
  - `vision_tower.*` (210 tensors): 16-layer SigLIP-style ViT with axial RoPE,
    patch_size=16, position_embedding_size=10240.
      * `vision_tower.patch_embedder.{input_proj,position_embedding_table}.{weight}`
      * `vision_tower.encoder.layers.{0..15}.{input_layernorm,pre_feedforward_layernorm,
        post_attention_layernorm,post_feedforward_layernorm}.{weight}`,
        `self_attn.{q,k,v,o}_proj.{linear.weight}` + `.{q,k}_norm.{weight}`,
        `mlp.{gate,up,down}_proj.{linear.weight}`.
  - `audio_tower.*` (751 tensors): 12-layer Conformer with TensorRT-LLM
    FP8 calibration stats. **Not currently consumed by the engine**; the
    converter still emits every tensor so the GGUF is byte-complete.
  - `embed_vision.embedding_projection.weight` [512, 768]
  - `embed_audio.embedding_projection.weight` [512, 1536]
    (multimodal projectors: soft-token bottleneck for image / audio).

Output:
* `<out-dir>/embeddinggemma-2-F16.gguf` — single GGUF (arch `gemma4-embedding`)
  containing every tensor from the source safetensors, renamed to the engine's
  internal convention. Text + vision + audio + projector all live in the same
  file; sections are disambiguated by tensor-name prefix.

Reference for tensor naming:
* `src/models/gemma4/trunk/weights.rs` (gemma4_text trunk) for `blk.*` /
  `per_layer_*` / `attn_*` / `ffn_*` names.
* `src/models/gemma4/vision/mod.rs` (gemma4v / gemma4uv vision) for
  `vision_encoder.*` / `mm.0` / `mm.2` names.
* `src/models/gemma_embedding/weights.rs` (gemma-embedding 300M) for
  `dense_2` / `dense_3` post-pooling head.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.converter.utils.gguf import (
    model_metadata_value,
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_Q8_0,
    GgufWriter,
    bf16_bytes_to_q8_0,
    bf16_to_f32,
    gguf_dims,
    open_safetensors,
)


# Text-side dimensions pinned from `config.json` (`text_config`).
TEXT_N_EMBD = 512
TEXT_N_LAYER = 24
TEXT_N_HEAD = 4
TEXT_N_HEAD_KV_DEFAULT = 2
TEXT_HEAD_DIM_DEFAULT = 256
TEXT_HEAD_DIM_FULL = 512
TEXT_INTERMEDIATE_SIZE = 2048
TEXT_VOCAB = 262_144
TEXT_EMBEDDING_DIM = 768  # final embedding width (after embedding_projection)
TEXT_CONTEXT_LENGTH = 262_144
TEXT_NORM_EPS = 1e-6
# Layer indices whose attention is full (head_dim=512, kv_heads=1). The
# others are sliding (head_dim=256, kv_heads=2). The 4-norm-attention E2B
# convention: `is_full_attention(il) = (il+1) % 6 == 5`.
TEXT_FULL_ATTENTION_LAYERS = (5, 11, 17, 23)
TEXT_ROPE_THETA_FULL = 1_000_000.0
TEXT_ROPE_THETA_SLIDING = 10_000.0
TEXT_SLIDING_WINDOW = 512

# Vision-side dimensions (gemma4_vision, SigLIP-style).
VISION_DEPTH = 16
VISION_EMBD = 768
VISION_INTERMEDIATE = 3072
VISION_HEADS = 12
VISION_HEAD_DIM = 64
VISION_PATCH = 16
VISION_MERGE = 3
VISION_PROJECTION = 1536
VISION_POSITIONS = 10_240
VISION_DEFAULT_OUTPUT_LENGTH = 280  # soft tokens per image

# Audio-side dimensions (gemma4_audio, Conformer).
AUDIO_N_LAYER = 12
AUDIO_HIDDEN = 1024
AUDIO_FFN_HIDDEN = 4096
AUDIO_PROJ_DIM = 1536

# Vocabulary specials.
BOA_TOKEN_ID = 256_000
BOI_TOKEN_ID = 255_999
EOA_TOKEN_INDEX = 258_883
EOI_TOKEN_ID = 258_882
IMAGE_TOKEN_ID = 258_880
VIDEO_TOKEN_ID = 258_884
AUDIO_TOKEN_ID = 258_881


def _require(st, name: str):
    if name not in st.header:
        raise KeyError(f"missing safetensors tensor: {name}")
    return st.get(name)


def _emit_f32(gguf: GgufWriter, name: str, raw: bytes, shape: tuple) -> None:
    n = 1
    for d in shape:
        n *= d
    if len(raw) != n * 4:
        raise ValueError(f"{name}: shape {shape} ({n}) != f32 payload {len(raw) // 4}")
    gguf.add_tensor(name, GGML_F32, gguf_dims(shape), raw)


def _emit_bf16_f32(gguf: GgufWriter, name: str, raw: bytes, shape: tuple) -> None:
    n = 1
    for d in shape:
        n *= d
    if len(raw) != n * 2:
        raise ValueError(
            f"{name}: shape {shape} ({n}) != bf16 payload {len(raw) // 2}"
        )
    gguf.add_tensor(name, GGML_F32, gguf_dims(shape), bf16_to_f32(raw))


def _emit_bf16_q8_0(gguf: GgufWriter, name: str, raw: bytes, shape: tuple) -> None:
    n = 1
    for d in shape:
        n *= d
    if len(raw) != n * 2:
        raise ValueError(
            f"{name}: shape {shape} ({n}) != bf16 payload {len(raw) // 2}"
        )
    q8 = bf16_bytes_to_q8_0(raw)
    gguf.add_tensor(name, GGML_Q8_0, gguf_dims(shape), q8)


def _emit_bf16(gguf: GgufWriter, name: str, raw: bytes, shape: tuple) -> None:
    n = 1
    for d in shape:
        n *= d
    if len(raw) != n * 2:
        raise ValueError(
            f"{name}: shape {shape} ({n}) != bf16 payload {len(raw) // 2}"
        )
    gguf.add_tensor(name, GGML_BF16, gguf_dims(shape), raw)


def _is_full_attention(idx: int) -> bool:
    return idx in TEXT_FULL_ATTENTION_LAYERS


def write_gguf(hf_dir: Path, out_path: Path) -> None:
    cfg_path = hf_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing config.json at {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    text = cfg.get("text_config", {})
    vision = cfg.get("vision_config", {})
    audio = cfg.get("audio_config", {})

    # Pin dimensions from config so any retraining of the upstream is caught
    # at conversion time rather than producing a silently incompatible GGUF.
    if text["hidden_size"] != TEXT_N_EMBD:
        raise ValueError(f"text hidden_size={text['hidden_size']} != {TEXT_N_EMBD}")
    if text["num_hidden_layers"] != TEXT_N_LAYER:
        raise ValueError(f"text num_hidden_layers={text['num_hidden_layers']} != {TEXT_N_LAYER}")
    if text["intermediate_size"] != TEXT_INTERMEDIATE_SIZE:
        raise ValueError(f"text intermediate_size={text['intermediate_size']} != {TEXT_INTERMEDIATE_SIZE}")
    if text["vocab_size"] != TEXT_VOCAB:
        raise ValueError(f"text vocab_size={text['vocab_size']} != {TEXT_VOCAB}")
    if vision["hidden_size"] != VISION_EMBD:
        raise ValueError(f"vision hidden_size={vision['hidden_size']} != {VISION_EMBD}")
    if vision["num_hidden_layers"] != VISION_DEPTH:
        raise ValueError(f"vision num_hidden_layers={vision['num_hidden_layers']} != {VISION_DEPTH}")
    if audio["hidden_size"] != AUDIO_HIDDEN:
        raise ValueError(f"audio hidden_size={audio['hidden_size']} != {AUDIO_HIDDEN}")
    if audio["num_hidden_layers"] != AUDIO_N_LAYER:
        raise ValueError(f"audio num_hidden_layers={audio['num_hidden_layers']} != {AUDIO_N_LAYER}")

    tok_cfg_path = hf_dir / "tokenizer_config.json"
    if not tok_cfg_path.is_file():
        raise FileNotFoundError(f"missing tokenizer_config.json at {tok_cfg_path}")
    tok_cfg = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
    tok_json_path = hf_dir / "tokenizer.json"
    if not tok_json_path.is_file():
        raise FileNotFoundError(f"missing tokenizer.json at {tok_json_path}")
    tok_json = json.loads(tok_json_path.read_text(encoding="utf-8"))
    merges = tok_json["model"]["merges"]

    st = open_safetensors(hf_dir / "model.safetensors")

    # Build the GGUF.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gguf = GgufWriter(out_path)

    # ---- general.* ----
    gguf.add_meta("general.architecture", "gemma4-embedding")
    gguf.add_meta("general.name", "embeddinggemma-2")
    gguf.add_meta("general.basename", "embeddinggemma-2")
    gguf.add_meta("general.size_label", "embedding")
    gguf.add_meta("general.file_type", 1)  # F16 convention for the multi-modal asset.
    gguf.add_meta("general.quantization_version", 2)
    gguf.add_meta(
        "general.base_model.0.name", "EmbeddingGemma 2 (multimodal encoder)",
    )
    gguf.add_meta("general.base_model.count", 1)

    # ---- gemma4-embedding.* (text backbone) ----
    gguf.add_meta("gemma4-embedding.embedding_length", TEXT_N_EMBD)
    gguf.add_meta("gemma4-embedding.block_count", TEXT_N_LAYER)
    gguf.add_meta("gemma4-embedding.context_length", TEXT_CONTEXT_LENGTH)
    gguf.add_meta("gemma4-embedding.feed_forward_length", TEXT_INTERMEDIATE_SIZE)
    gguf.add_meta("gemma4-embedding.attention.head_count", TEXT_N_HEAD)
    gguf.add_meta("gemma4-embedding.attention.layer_norm_rms_epsilon", TEXT_NORM_EPS)
    gguf.add_meta("gemma4-embedding.vocab_size", TEXT_VOCAB)
    gguf.add_meta("gemma4-embedding.embedding_dim", TEXT_EMBEDDING_DIM)
    gguf.add_meta("gemma4-embedding.sliding_window", TEXT_SLIDING_WINDOW)
    gguf.add_meta("gemma4-embedding.full_attention_layers",
                  list(TEXT_FULL_ATTENTION_LAYERS))
    # Per-layer head dim / kv heads arrays (length = block_count). Sliding
    # layers use head_dim=256 + kv_heads=2; full layers use head_dim=512 +
    # kv_heads=1. Engine reads these as parallel arrays.
    head_dims = [
        TEXT_HEAD_DIM_FULL if _is_full_attention(i) else TEXT_HEAD_DIM_DEFAULT
        for i in range(TEXT_N_LAYER)
    ]
    kv_heads = [
        1 if _is_full_attention(i) else TEXT_N_HEAD_KV_DEFAULT
        for i in range(TEXT_N_LAYER)
    ]
    gguf.add_meta("gemma4-embedding.attention.head_dim_per_layer",
                  [str(d) for d in head_dims])
    gguf.add_meta("gemma4-embedding.attention.head_count_kv_per_layer",
                  [str(k) for k in kv_heads])
    gguf.add_meta("gemma4-embedding.rope.freq_base_full", TEXT_ROPE_THETA_FULL)
    gguf.add_meta("gemma4-embedding.rope.freq_base_sliding", TEXT_ROPE_THETA_SLIDING)

    # ---- vision + audio (multimodal projectors) ----
    gguf.add_meta("gemma4-embedding.vision.depth", VISION_DEPTH)
    gguf.add_meta("gemma4-embedding.vision.embedding_length", VISION_EMBD)
    gguf.add_meta("gemma4-embedding.vision.feed_forward_length", VISION_INTERMEDIATE)
    gguf.add_meta("gemma4-embedding.vision.attention.head_count", VISION_HEADS)
    gguf.add_meta("gemma4-embedding.vision.patch_size", VISION_PATCH)
    gguf.add_meta("gemma4-embedding.vision.spatial_merge_size", VISION_MERGE)
    gguf.add_meta("gemma4-embedding.vision.projection_dim", VISION_PROJECTION)
    gguf.add_meta("gemma4-embedding.vision.rope_type", "axial")
    gguf.add_meta("gemma4-embedding.vision.position_embedding_size", VISION_POSITIONS)
    gguf.add_meta("gemma4-embedding.vision.default_output_length", VISION_DEFAULT_OUTPUT_LENGTH)
    gguf.add_meta("gemma4-embedding.vision.layer_norm_rms_epsilon", TEXT_NORM_EPS)

    gguf.add_meta("gemma4-embedding.audio.depth", AUDIO_N_LAYER)
    gguf.add_meta("gemma4-embedding.audio.hidden_size", AUDIO_HIDDEN)
    gguf.add_meta("gemma4-embedding.audio.ffn_hidden_size", AUDIO_FFN_HIDDEN)
    gguf.add_meta("gemma4-embedding.audio.projection_dim", AUDIO_PROJ_DIM)

    gguf.add_meta("gemma4-embedding.boi_token_id", BOI_TOKEN_ID)
    gguf.add_meta("gemma4-embedding.eoi_token_id", EOI_TOKEN_ID)
    gguf.add_meta("gemma4-embedding.boa_token_id", BOA_TOKEN_ID)
    gguf.add_meta("gemma4-embedding.eoa_token_id", EOA_TOKEN_INDEX)
    gguf.add_meta("gemma4-embedding.image_token_id", IMAGE_TOKEN_ID)
    gguf.add_meta("gemma4-embedding.video_token_id", VIDEO_TOKEN_ID)
    gguf.add_meta("gemma4-embedding.audio_token_id", AUDIO_TOKEN_ID)

    # ---- tokenizer.ggml.* ----
    vocab = {int(v): k for k, v in tok_json["model"]["vocab"].items()}
    added = {entry["id"]: entry["content"] for entry in tok_json["added_tokens"]}
    all_tokens = {**vocab, **added}
    tokens = [all_tokens.get(i, f"<|reserved_{i}|>") for i in range(TEXT_VOCAB)]
    types = [1] * TEXT_VOCAB
    for tid in added:
        if tid < TEXT_VOCAB:
            types[tid] = 3
    # EmbeddingGemma uses Gemma's character-level BPE: spaces become `▁`,
    # newlines are separate segments, and BOS is always emitted by the
    # Gemma4 tokenizer path. Keeping the model tag explicit avoids routing it
    # through the GPT-2 byte-level pretokenizer.
    gguf.add_meta("tokenizer.ggml.model", "gemma4")
    gguf.add_meta("tokenizer.ggml.tokens", tokens)
    gguf.add_meta("tokenizer.ggml.token_type", types)
    gguf.add_meta("tokenizer.ggml.merges", [" ".join(m) for m in merges])
    token_ids = {token: token_id for token_id, token in all_tokens.items()}
    bos_id = tok_cfg.get("bos_token_id")
    if bos_id is None:
        bos_id = token_ids.get(tok_cfg.get("bos_token", "<bos>"), 1)
    eos_id = tok_cfg.get("eos_token_id")
    if eos_id is None:
        eos_id = token_ids.get(tok_cfg.get("eos_token", "<eos>"), 1)
    gguf.add_meta("tokenizer.ggml.bos_token_id", int(bos_id))
    gguf.add_meta("tokenizer.ggml.eos_token_id", int(eos_id))
    gguf.add_meta("tokenizer.ggml.add_bos_token", True)
    gguf.add_meta("tokenizer.ggml.add_eos_token", True)

    # ---- tensors ----
    # Token embeddings + embedding projection.
    embed = _require(st, "language_model.embed_tokens.weight")
    if tuple(embed.shape) != (TEXT_VOCAB, TEXT_N_EMBD):
        raise ValueError(
            f"embed_tokens shape {embed.shape} != ({TEXT_VOCAB}, {TEXT_N_EMBD})"
        )
    _emit_bf16(gguf, "token_embd.weight", embed.raw, tuple(embed.shape))
    proj_text = _require(st, "language_model.embedding_projection.weight")
    _emit_bf16(
        gguf,
        "text_embproj.weight",
        proj_text.raw,
        tuple(proj_text.shape),
    )

    # Per-layer shared tensors (vocab-sized per-layer projection).
    ple_proj = _require(st, "language_model.ple.per_layer_model_projection.weight")
    _emit_bf16(
        gguf, "per_layer_model_proj.weight", ple_proj.raw, tuple(ple_proj.shape),
    )
    ple_norm = _require(st, "language_model.ple.per_layer_projection_norm.weight")
    _emit_bf16_f32(
        gguf, "per_layer_proj_norm.weight", ple_norm.raw, tuple(ple_norm.shape),
    )

    # Per-layer tensors.
    for i in range(TEXT_N_LAYER):
        layer = f"language_model.layers.{i}"
        blk = f"blk.{i}"
        full_attn = _is_full_attention(i)

        # 5-norm sandwich (input_layernorm, post_attention_layernorm,
        # pre_feedforward_layernorm, post_feedforward_layernorm, + ple).
        for hf_name, gguf_name in [
            ("input_layernorm", "attn_norm"),
            ("post_attention_layernorm", "post_attention_norm"),
            ("pre_feedforward_layernorm", "pre_ffn_norm"),
            ("post_feedforward_layernorm", "post_ffn_norm"),
        ]:
            t = _require(st, f"{layer}.{hf_name}.weight")
            _emit_bf16_f32(gguf, f"{blk}.{gguf_name}.weight", t.raw, tuple(t.shape))

        # `layer_scalar` is a 1-value BF16 → cast to F32.
        ls = _require(st, f"{layer}.layer_scalar")
        _emit_bf16_f32(gguf, f"{blk}.layer_scalar", ls.raw, tuple(ls.shape))

        # PLE block.
        ig = _require(st, f"{layer}.ple_block.per_layer_input_gate.weight")
        _emit_bf16_q8_0(gguf, f"{blk}.ple_input_gate.weight", ig.raw, tuple(ig.shape))
        pp = _require(st, f"{layer}.ple_block.per_layer_projection.weight")
        _emit_bf16_q8_0(gguf, f"{blk}.ple_projection.weight", pp.raw, tuple(pp.shape))
        pn = _require(st, f"{layer}.ple_block.post_per_layer_input_norm.weight")
        _emit_bf16_f32(
            gguf, f"{blk}.ple_post_norm.weight", pn.raw, tuple(pn.shape),
        )

        # Q/K/V/O + Q/K RMSNorm. Sliding layers use n_kv=2; full layers n_kv=1.
        # Q shape: [n_head * head_dim, n_embd] = sliding [4*256, 512] = [1024, 512];
        # full [4*512, 512] = [2048, 512].
        # K shape: [n_kv * head_dim, n_embd] = sliding [2*256, 512] = [512, 512];
        # full [1*512, 512] = [512, 512].
        # V shape same as K.
        # O shape: [n_embd, n_head * head_dim] = sliding [512, 1024]; full [512, 2048].
        q = _require(st, f"{layer}.self_attn.q_proj.weight")
        _emit_bf16_q8_0(gguf, f"{blk}.attn_q.weight", q.raw, tuple(q.shape))
        k = _require(st, f"{layer}.self_attn.k_proj.weight")
        _emit_bf16_q8_0(gguf, f"{blk}.attn_k.weight", k.raw, tuple(k.shape))
        v = _require(st, f"{layer}.self_attn.v_proj.weight")
        _emit_bf16_q8_0(gguf, f"{blk}.attn_v.weight", v.raw, tuple(v.shape))
        o = _require(st, f"{layer}.self_attn.o_proj.weight")
        _emit_bf16_q8_0(gguf, f"{blk}.attn_output.weight", o.raw, tuple(o.shape))
        qn = _require(st, f"{layer}.self_attn.q_norm.weight")
        _emit_bf16_f32(gguf, f"{blk}.attn_q_norm.weight", qn.raw, tuple(qn.shape))
        kn = _require(st, f"{layer}.self_attn.k_norm.weight")
        _emit_bf16_f32(gguf, f"{blk}.attn_k_norm.weight", kn.raw, tuple(kn.shape))

        # FFN: standard SwiGLU.
        gp = _require(st, f"{layer}.mlp.gate_proj.weight")
        _emit_bf16_q8_0(gguf, f"{blk}.ffn_gate.weight", gp.raw, tuple(gp.shape))
        up = _require(st, f"{layer}.mlp.up_proj.weight")
        _emit_bf16_q8_0(gguf, f"{blk}.ffn_up.weight", up.raw, tuple(up.shape))
        dp = _require(st, f"{layer}.mlp.down_proj.weight")
        _emit_bf16_q8_0(gguf, f"{blk}.ffn_down.weight", dp.raw, tuple(dp.shape))

    # Final RMSNorm.
    final_norm = _require(st, "language_model.norm.weight")
    _emit_bf16_f32(gguf, "output_norm.weight", final_norm.raw, tuple(final_norm.shape))

    # ---- Vision tower (SigLIP-style, axial RoPE) ----
    # Patch embedder: input_proj (Conv2d weight) + position_embedding_table.
    pe_input = _require(st, "vision_tower.patch_embedder.input_proj.weight")
    _emit_bf16(
        gguf, "vision.patch_input_proj.weight", pe_input.raw, tuple(pe_input.shape),
    )
    pe_pos = _require(st, "vision_tower.patch_embedder.position_embedding_table")
    _emit_bf16(
        gguf, "vision.position_embedding_table",
        pe_pos.raw,
        tuple(pe_pos.shape),
    )

    # Per-layer tensors (16 layers).
    for i in range(VISION_DEPTH):
        layer = f"vision_tower.encoder.layers.{i}"
        vblk = f"vision.blk.{i}"
        # 4-norm sandwich.
        for hf_name, gguf_name in [
            ("input_layernorm", "ln1"),
            ("post_attention_layernorm", "post_attention_norm"),
            ("pre_feedforward_layernorm", "pre_ffn_norm"),
            ("post_feedforward_layernorm", "post_ffn_norm"),
        ]:
            t = _require(st, f"{layer}.{hf_name}.weight")
            _emit_bf16_f32(
                gguf, f"{vblk}.{gguf_name}.weight", t.raw, tuple(t.shape),
            )
        for hf_name, gguf_name in [
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.o_proj", "attn_output"),
        ]:
            t = _require(st, f"{layer}.{hf_name}.linear.weight")
            _emit_bf16(
                gguf, f"{vblk}.{gguf_name}.weight", t.raw, tuple(t.shape),
            )
        qn = _require(st, f"{layer}.self_attn.q_norm.weight")
        _emit_bf16_f32(
            gguf, f"{vblk}.attn_q_norm.weight", qn.raw, tuple(qn.shape),
        )
        kn = _require(st, f"{layer}.self_attn.k_norm.weight")
        _emit_bf16_f32(
            gguf, f"{vblk}.attn_k_norm.weight", kn.raw, tuple(kn.shape),
        )
        for hf_name, gguf_name in [
            ("mlp.gate_proj", "ffn_gate"),
            ("mlp.up_proj", "ffn_up"),
            ("mlp.down_proj", "ffn_down"),
        ]:
            t = _require(st, f"{layer}.{hf_name}.linear.weight")
            _emit_bf16(
                gguf, f"{vblk}.{gguf_name}.weight", t.raw, tuple(t.shape),
            )

    # Vision output projection to projection_dim=1536 (handled by trunk).
    # EmbeddingGemma2's vision tower doesn't expose a separate output projection
    # in the HF safetensors — the projection is fused with the trunk. The
    # mmproj-equivalent in our GGUF is the soft-token projector below.

    # ---- Multimodal projectors (image→text + audio→text) ----
    embed_vision = _require(st, "embed_vision.embedding_projection.weight")
    _emit_bf16(
        gguf, "vision_embed.weight", embed_vision.raw, tuple(embed_vision.shape),
    )
    embed_audio = _require(st, "embed_audio.embedding_projection.weight")
    _emit_bf16(
        gguf, "audio_embed.weight", embed_audio.raw, tuple(embed_audio.shape),
    )

    # ---- Audio tower (Conformer) — pass through; engine-side forward path
    # is on the TODO list. We emit every tensor verbatim so the GGUF is
    # byte-complete relative to the source safetensors.
    for n in sorted(st.header):
        if n.startswith("audio_tower."):
            t = st.get(n)
            # Rename `audio_tower.layers.{i}.…` -> `audio.layers.{i}.…` so
            # the engine can address it without the prefix.
            new_name = "audio." + n[len("audio_tower."):]
            _emit_bf16(gguf, new_name, t.raw, tuple(t.shape))

    gguf.metadata = [(key, model_metadata_value(value)) for key, value in gguf.metadata]
    gguf.write()
    print(
        f"[gemma4-embedding] wrote {out_path} "
        f"({out_path.stat().st_size // 1024 // 1024} MiB)",
        file=sys.stderr,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert google/embeddinggemma-2 safetensors → gemma4-embedding GGUF",
    )
    ap.add_argument(
        "--hf-dir", required=True, type=Path,
        help="HF checkpoint dir (config.json + tokenizer.json + model.safetensors)",
    )
    ap.add_argument(
        "--out", required=True, type=Path,
        help="Output GGUF path (single file containing all sections)",
    )
    args = ap.parse_args()
    write_gguf(args.hf_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
