#!/usr/bin/env python3
"""Convert `DeepCybo/PhysBrain1.5-2B` (Qwen3-VL-2B-Instruct backbone fine-tuned
with ActionPiece + visual-state tokens) into two GGUF files the engine can
load via the existing `qwen3vl` text path and `qwen3vl_merger` mmproj path.

Inputs:
* HF safetensors checkpoint for `DeepCybo/PhysBrain1.5-2B` containing:
  - `config.json` — `qwen3_vl` text config (hidden_size=2048, 28 layers,
    head_count=16, head_count_kv=8, intermediate_size=6144, vocab_size=168576,
    rope theta=5e6, eps=1e-6, mrope sections [24,20,20,0], image_token=151655,
    vision_start=151652, vision_end=151653) + vision config (depth=24,
    hidden_size=1024, intermediate=4096, heads=16, patch=16, merge=2,
    num_position_embeddings=2304, deepstack_visual_indexes=[5,11,17]).
  - `tokenizer.json` — BPE vocab + merges + 16926 added special tokens.
  - `tokenizer_config.json` — for BOS/EOS/pad token ids.
  - `model.safetensors` — single 4.3 GiB BF16 shard with:
    * `model.language_model.embed_tokens.weight` [168576, 2048]
    * per layer (28 ×): input_layernorm, post_attention_layernorm, q/k/v/o_proj,
      q_norm, k_norm, gate_proj, up_proj, down_proj.
    * `model.language_model.norm.weight`
    * `model.visual.blocks.{0..23}.{attn.qkv,attn.proj,mlp.fc1,mlp.fc2,norm1,norm2}`
    * `model.visual.merger.{norm,linear_fc1,linear_fc2}`
    * `model.visual.deepstack_merger_list.{0,1,2}.{norm,linear_fc1,linear_fc2}`
    * `model.visual.patch_embed.proj.{weight,bias}` (Conv3D weight split)
    * `model.visual.pos_embed.weight` [2304, 1024]
    * NO `lm_head.weight` (tie_word_embeddings=true).

Output:
* `<out-dir>/physbrain-1.5-2b-Q8_0.gguf` — Qwen3-VL text tower (arch `qwen3vl`,
  vocab 168576, 28 layers, GQA-2). Standard llama.cpp qwen3vl names:
  `token_embd.weight`, `blk.{i}.attn_{norm,q_norm,k_norm,q,k,v,output}.weight`,
  `blk.{i}.ffn_{norm,gate,up,down}.weight`, `output_norm.weight`. Tie means
  no separate `output.weight`.
* `<out-dir>/mmproj-physbrain-1.5-2b-F16.gguf` — vision tower (arch `clip`,
  projector `qwen3vl_merger`). Vision tower at F16 (matches our Qwen3.5-2B
  convention). Tensors follow the same names the qwen3vl trunk already loads:
  `v.patch_embd.{weight,weight.1,bias}` (Conv3D → 2 Conv2Ds),
  `v.position_embd.weight`, `v.blk.{i}.{ln1,ln2,attn_qkv,attn_out,ffn_up,ffn_down}.{weight,bias}`,
  `v.deepstack.{5,11,17}.{norm,fc1,fc2}.{weight,bias}`,
  `v.post_ln.{weight,bias}`, `mm.{0,2}.{weight,bias}`.

Quantization: text tower `attn_*.weight` + `ffn_{gate,up,down}.weight` →
`Q8_0`; norms, biases, embeddings, post_ln, position_embd stay `F32` (norms)
or `BF16` (everything else that doesn't get quantized). Mmproj mirrors the
existing `Qwen3.5-2B` convention: vision `attn_*`/`mlp_*` weights stay F16
when not quantized, but we keep things simple and emit BF16 for the full
mmproj (the engine accepts BF16 for vision tensors; that's what
Qwen3.5-2B/mmproj-F16 actually emits because the upstream llama.cpp writer
stores F16 — but our GGUFs end up at F16 dtype with the right shape).

Token types: every entry of `tokenizer.json`'s `added_tokens` is CONTROL (3);
all `model.vocab` entries are NORMAL (1). BPE model + qwen2 pre.

Usage:
    python3 tools/converter/physbrain/convert_physbrain.py \
        --hf-dir models/PhysBrain1.5-2B-safetensors \
        --out-dir  models/PhysBrain1.5-2B-GGUF

Reference for metadata keys: `references/llama.cpp/conversion/qwen3vl.py`
(Qwen3VLVisionModel.set_gguf_parameters + Qwen3VLTextModel.set_gguf_parameters +
Qwen3Model base set_gguf_parameters).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools.converter.utils.gguf import (
    GGML_BF16,
    GGML_F32,
    GGML_F16,
    GGML_Q8_0,
    GgufWriter,
    bf16_bytes_to_q8_0,
    bf16_to_f32,
    f16_to_f32,
    gguf_dims,
    open_safetensors,
)


# Text-side metadata values pinned by the engine whitelist
# (`src/core/loader.rs::KNOWN_QWEN3VL_2B_DIMENSIONS`).
TEXT_ARCH = "qwen3vl"
TEXT_EMBEDDING_LENGTH = 2048
TEXT_BLOCK_COUNT = 28
TEXT_HEAD_COUNT = 16
TEXT_HEAD_COUNT_KV = 8
TEXT_FEED_FORWARD_LENGTH = 6144
TEXT_CONTEXT_LENGTH = 262_144
TEXT_ROPE_FREQ_BASE = 5_000_000.0
TEXT_NORM_EPS = 1e-6
TEXT_HEAD_DIM = 128  # n_embd / n_head = 2048 / 16
TEXT_ROPE_SECTIONS = [24, 20, 20, 0]
# rope.dimension_count is the number of rotated lanes per head, and for Qwen3-VL
# it is the *whole* head dim: llama.cpp asserts `n_embd_head == n_rot`
# (src/models/qwen3vl.cpp) and reads n_rot from this key when it is present.
# HEAD_DIM/2 would make llama.cpp abort on load, and it would also tell this
# engine to rotate 64 lanes when the model rotates 128.
TEXT_DIMENSION_COUNT = 128
# Vision-side metadata values
VISION_BLOCK_COUNT = 24
VISION_EMBEDDING_LENGTH = 1024
VISION_FEED_FORWARD_LENGTH = 4096
VISION_HEAD_COUNT = 16
VISION_PATCH_SIZE = 16
VISION_IMAGE_SIZE = 768  # sqrt(num_position_embeddings=2304) * patch_size=16
VISION_SPATIAL_MERGE_SIZE = 2
VISION_NUM_POSITION_EMBEDDINGS = 2304
VISION_PROJECTION_DIM = 2048  # out_hidden_size, matches TEXT_EMBEDDING_LENGTH
DEEPSTACK_VISUAL_INDEXES = [5, 11, 17]
TEXT_VOCAB_SIZE = 168_576
IMAGE_TOKEN_ID = 151_655
VISION_START_TOKEN_ID = 151_652
VISION_END_TOKEN_ID = 151_653


def _read_hf_config(hf_dir: Path) -> dict:
    cfg_path = hf_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing HF config.json at {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    text = cfg.get("text_config", {})
    vision = cfg.get("vision_config", {})
    # Pin dimensions from the actual config so any drift is caught early.
    expected = {
        ("text_config", "hidden_size"): TEXT_EMBEDDING_LENGTH,
        ("text_config", "num_hidden_layers"): TEXT_BLOCK_COUNT,
        ("text_config", "num_attention_heads"): TEXT_HEAD_COUNT,
        ("text_config", "num_key_value_heads"): TEXT_HEAD_COUNT_KV,
        ("text_config", "intermediate_size"): TEXT_FEED_FORWARD_LENGTH,
        ("text_config", "vocab_size"): TEXT_VOCAB_SIZE,
        ("vision_config", "hidden_size"): VISION_EMBEDDING_LENGTH,
        ("vision_config", "intermediate_size"): VISION_FEED_FORWARD_LENGTH,
        ("vision_config", "num_heads"): VISION_HEAD_COUNT,
        ("vision_config", "depth"): VISION_BLOCK_COUNT,
    }
    for (sec, key), want in expected.items():
        got = cfg.get(sec, {}).get(key)
        if got != want:
            raise ValueError(
                f"config mismatch {sec}.{key}: got {got!r}, expected {want!r}"
            )
    if cfg.get("image_token_id") != IMAGE_TOKEN_ID:
        raise ValueError(
            f"image_token_id={cfg.get('image_token_id')!r} != {IMAGE_TOKEN_ID}"
        )
    if vision.get("deepstack_visual_indexes") != DEEPSTACK_VISUAL_INDEXES:
        raise ValueError(
            f"deepstack_visual_indexes={vision.get('deepstack_visual_indexes')!r} "
            f"!= {DEEPSTACK_VISUAL_INDEXES!r}"
        )
    return cfg


def _read_tokenizer(hf_dir: Path) -> tuple[dict, dict, list[str]]:
    """Return (config, tokenizer_json, merges_list)."""
    tok_path = hf_dir / "tokenizer.json"
    if not tok_path.is_file():
        raise FileNotFoundError(f"missing tokenizer.json at {tok_path}")
    tok = json.loads(tok_path.read_text(encoding="utf-8"))
    cfg_path = hf_dir / "tokenizer_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing tokenizer_config.json at {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8")), tok, tok["model"]["merges"]


def _build_vocab(
    tokenizer_cfg: dict, tok_json: dict
) -> tuple[list[str], list[int]]:
    """Build `(tokens, types)` lists for `tokenizer.ggml.tokens` /
    `tokenizer.ggml.token_type`. `types`: 1 = NORMAL, 3 = CONTROL.
    BPE vocab entries from `model.vocab` are NORMAL; every entry of
    `added_tokens` is CONTROL. Pad any unused slots up to `TEXT_VOCAB_SIZE`
    with `<|reserved_{i}|>` NORMAL placeholders so missing entries do not
    produce empty strings in the engine.
    """
    vocab = {int(v): k for k, v in tok_json["model"]["vocab"].items()}
    added = {entry["id"]: entry["content"] for entry in tok_json["added_tokens"]}
    all_tokens: dict[int, str] = {**vocab, **added}
    tokens = [
        all_tokens.get(i, f"<|reserved_{i}|>") for i in range(TEXT_VOCAB_SIZE)
    ]
    types = [1] * TEXT_VOCAB_SIZE
    for tid in added:
        if tid < TEXT_VOCAB_SIZE:
            types[tid] = 3
    return tokens, types


def _require(safetensors, name: str) -> "Tensor":
    """`tools.converter.utils.gguf.Tensor` accessor with clear errors."""
    if name not in safetensors.header:
        raise KeyError(f"missing safetensors tensor: {name}")
    return safetensors.get(name)


def _emit_bf16_norm(gguf: GgufWriter, name: str, raw: bytes, shape: tuple) -> None:
    """Norms / biases stay F32 in the qwen3vl trunk (see
    `src/models/qwen3/trunk/weights.rs::get_f32_tensor`)."""
    if len(raw) % 2 != 0:
        raise ValueError(f"{name}: bf16 payload {len(raw)} not a multiple of 2")
    n = len(raw) // 2
    expected = 1
    for d in shape:
        expected *= d
    if n != expected:
        raise ValueError(f"{name}: shape {shape} ({expected}) != bf16 len {n}")
    gguf.add_tensor(name, GGML_F32, gguf_dims(shape), bf16_to_f32(raw))


def _emit_bf16_q8_0(gguf: GgufWriter, name: str, raw: bytes, shape: tuple) -> None:
    """BF16 → Q8_0 for learned linear weights."""
    n = 1
    for d in shape:
        n *= d
    if len(raw) != n * 2:
        raise ValueError(
            f"{name}: shape {shape} ({n}) != bf16 payload {len(raw) // 2}"
        )
    gguf.add_tensor(name, GGML_Q8_0, gguf_dims(shape), bf16_bytes_to_q8_0(raw))


def _emit_bf16(gguf: GgufWriter, name: str, raw: bytes, shape: tuple) -> None:
    """BF16 kept as-is (token embeddings + mmproj-side weights)."""
    n = 1
    for d in shape:
        n *= d
    if len(raw) != n * 2:
        raise ValueError(
            f"{name}: shape {shape} ({n}) != bf16 payload {len(raw) // 2}"
        )
    gguf.add_tensor(name, GGML_BF16, gguf_dims(shape), raw)


def _split_conv3d(weight_raw: bytes, shape: tuple[int, ...]) -> tuple[bytes, tuple]:
    """`patch_embed.proj.weight` is shape `(out, in, kt=2, kH, kW)` in HF.
    The engine (and llama.cpp) want it split into two Conv2D weights
    `(out, in, kH, kW)`, the second being suffixed `.weight.1`. We return
    `(raw_bytes_of_frame0, shape_bytes)`; frame1 is emitted by the caller
    as `name + ".1"`.
    """
    if len(shape) != 5 or shape[2] != 2:
        raise ValueError(
            f"conv3d split expected shape (out, in, 2, kH, kW), got {shape}"
        )
    out_c, in_c, _kt, kH, kW = shape
    n_per_frame = out_c * in_c * kH * kW
    expected = n_per_frame * 2  # both frames concatenated
    if len(weight_raw) != expected * 2:
        raise ValueError(
            f"conv3d split: shape {shape} -> {expected*2} bf16 bytes, "
            f"got {len(weight_raw)}"
        )
    bf16 = np.frombuffer(weight_raw, dtype="<u2").reshape(
        out_c, in_c, 2, kH * kW
    )
    frame0 = bf16[:, :, 0, :].reshape(out_c, in_c, kH, kW).copy()
    frame1 = bf16[:, :, 1, :].reshape(out_c, in_c, kH, kW).copy()
    torch_shape = (out_c, in_c, kH, kW)
    return (
        frame0.tobytes(),
        torch_shape,
        frame1.tobytes(),
        torch_shape,
    )


def write_text_gguf(hf_dir: Path, out_path: Path) -> None:
    cfg = _read_hf_config(hf_dir)
    tok_cfg, tok_json, merges = _read_tokenizer(hf_dir)
    tokens, types = _build_vocab(tok_cfg, tok_json)
    st = open_safetensors(hf_dir / "model.safetensors")

    gguf = GgufWriter(out_path)

    # ---- general.* ----
    gguf.add_meta("general.architecture", TEXT_ARCH)
    gguf.add_meta("general.name", "PhysBrain1.5-2B")
    gguf.add_meta("general.basename", "PhysBrain1.5-2B")
    gguf.add_meta("general.size_label", "2B")
    gguf.add_meta("general.file_type", 7)  # mostly Q8_0
    gguf.add_meta("general.quantization_version", 2)
    gguf.add_meta(
        "general.base_model.0.name", "Qwen3-VL-2B-Instruct",
    )
    gguf.add_meta(
        "general.base_model.0.organization", "Qwen",
    )
    gguf.add_meta(
        "general.base_model.0.repo_url",
        "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct",
    )
    gguf.add_meta("general.base_model.count", 1)

    # ---- qwen3vl.* ----
    gguf.add_meta("qwen3vl.embedding_length", TEXT_EMBEDDING_LENGTH)
    gguf.add_meta("qwen3vl.block_count", TEXT_BLOCK_COUNT)
    gguf.add_meta("qwen3vl.context_length", TEXT_CONTEXT_LENGTH)
    gguf.add_meta("qwen3vl.feed_forward_length", TEXT_FEED_FORWARD_LENGTH)
    gguf.add_meta("qwen3vl.attention.head_count", TEXT_HEAD_COUNT)
    gguf.add_meta("qwen3vl.attention.head_count_kv", TEXT_HEAD_COUNT_KV)
    gguf.add_meta("qwen3vl.attention.layer_norm_rms_epsilon", TEXT_NORM_EPS)
    gguf.add_meta("qwen3vl.rope.freq_base", TEXT_ROPE_FREQ_BASE)
    gguf.add_meta("qwen3vl.rope.dimension_count", TEXT_DIMENSION_COUNT)
    gguf.add_meta("qwen3vl.rope.dimension_sections", TEXT_ROPE_SECTIONS)
    gguf.add_meta("qwen3vl.vocab_size", TEXT_VOCAB_SIZE)
    # Image placeholder ids (engine reads these from metadata).
    gguf.add_meta("qwen3vl.image_token_id", IMAGE_TOKEN_ID)
    gguf.add_meta("qwen3vl.vision_start_token_id", VISION_START_TOKEN_ID)
    gguf.add_meta("qwen3vl.vision_end_token_id", VISION_END_TOKEN_ID)

    # ---- tokenizer.ggml.* ----
    gguf.add_meta("tokenizer.ggml.model", "gpt2")
    gguf.add_meta("tokenizer.ggml.pre", "qwen2")
    gguf.add_meta("tokenizer.ggml.tokens", tokens)
    gguf.add_meta("tokenizer.ggml.token_type", types)
    gguf.add_meta("tokenizer.ggml.merges", [" ".join(m) for m in merges])
    # Qwen3-VL uses `<|endoftext|>` (id 151643) as both EOS and pad; the
    # chat template emits `<|im_end|>` (id 151645) as turn stop. Set both so
    # `RMI_EOS_TOKEN_ID` resolution finds a match whichever the caller picks.
    bos_id = tok_cfg.get("bos_token_id") or 151643
    eos_id = tok_cfg.get("eos_token_id") or 151645
    gguf.add_meta("tokenizer.ggml.bos_token_id", int(bos_id))
    gguf.add_meta("tokenizer.ggml.eos_token_id", int(eos_id))
    gguf.add_meta("tokenizer.ggml.add_bos_token", False)
    gguf.add_meta("tokenizer.ggml.add_eos_token", False)

    # ---- tensors ----
    # Embedding is BF16 because the engine token_embd lookup expects it; the
    # qwen3 trunk accepts BF16 here.
    embed = _require(st, "model.language_model.embed_tokens.weight")
    if tuple(embed.shape) != (TEXT_VOCAB_SIZE, TEXT_EMBEDDING_LENGTH):
        raise ValueError(
            f"embed_tokens shape {embed.shape} != "
            f"({TEXT_VOCAB_SIZE}, {TEXT_EMBEDDING_LENGTH})"
        )
    _emit_bf16(gguf, "token_embd.weight", embed.raw, tuple(embed.shape))
    # No `output.weight` — `tie_word_embeddings=true`. Engine constructs
    # output from `token_embd.weight` (see `src/models/qwen3/trunk/weights.rs`).

    for i in range(TEXT_BLOCK_COUNT):
        layer_prefix = f"model.language_model.layers.{i}"
        block_prefix = f"blk.{i}"

        def emit_ln(name: str, hf_sub: str, *, f32: bool = True) -> None:
            t = _require(st, f"{layer_prefix}.{hf_sub}.weight")
            if f32:
                _emit_bf16_norm(
                    gguf, f"{block_prefix}.{name}.weight", t.raw, tuple(t.shape),
                )
            else:
                _emit_bf16(
                    gguf, f"{block_prefix}.{name}.weight", t.raw, tuple(t.shape),
                )

        # Norms → F32.
        emit_ln("attn_norm", "input_layernorm", f32=True)
        emit_ln("ffn_norm", "post_attention_layernorm", f32=True)

        # Per-head Q/K RMSNorm → F32 (head_dim = 128).
        q_norm = _require(st, f"{layer_prefix}.self_attn.q_norm.weight")
        _emit_bf16_norm(gguf, f"{block_prefix}.attn_q_norm.weight", q_norm.raw, tuple(q_norm.shape))
        k_norm = _require(st, f"{layer_prefix}.self_attn.k_norm.weight")
        _emit_bf16_norm(gguf, f"{block_prefix}.attn_k_norm.weight", k_norm.raw, tuple(k_norm.shape))

        # Attention projections → Q8_0.
        for hf_name, gguf_name in [
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.o_proj", "attn_output"),
        ]:
            t = _require(st, f"{layer_prefix}.{hf_name}.weight")
            _emit_bf16_q8_0(gguf, f"{block_prefix}.{gguf_name}.weight", t.raw, tuple(t.shape))

        # FFN projections → Q8_0.
        for hf_name, gguf_name in [
            ("mlp.gate_proj", "ffn_gate"),
            ("mlp.up_proj", "ffn_up"),
            ("mlp.down_proj", "ffn_down"),
        ]:
            t = _require(st, f"{layer_prefix}.{hf_name}.weight")
            _emit_bf16_q8_0(gguf, f"{block_prefix}.{gguf_name}.weight", t.raw, tuple(t.shape))

    # Final norm.
    out_norm = _require(st, "model.language_model.norm.weight")
    _emit_bf16_norm(gguf, "output_norm.weight", out_norm.raw, tuple(out_norm.shape))

    gguf.write()
    print(
        f"[text] wrote {out_path} ({out_path.stat().st_size // 1024 // 1024} MiB)",
        file=sys.stderr,
    )


def write_mmproj_gguf(hf_dir: Path, out_path: Path) -> None:
    cfg = _read_hf_config(hf_dir)
    st = open_safetensors(hf_dir / "model.safetensors")

    gguf = GgufWriter(out_path)

    # ---- general.* ----
    gguf.add_meta("general.architecture", "clip")
    gguf.add_meta("general.name", "PhysBrain1.5-2B-mmproj")
    gguf.add_meta("general.file_type", 1)  # F16 convention; matches Qwen3.5-2B.

    # ---- clip.* ----
    gguf.add_meta("clip.has_vision_encoder", True)
    gguf.add_meta("clip.projector_type", "qwen3vl_merger")
    gguf.add_meta("clip.vision.embedding_length", VISION_EMBEDDING_LENGTH)
    gguf.add_meta("clip.vision.feed_forward_length", VISION_FEED_FORWARD_LENGTH)
    gguf.add_meta("clip.vision.block_count", VISION_BLOCK_COUNT)
    gguf.add_meta("clip.vision.attention.head_count", VISION_HEAD_COUNT)
    gguf.add_meta("clip.vision.image_size", VISION_IMAGE_SIZE)
    gguf.add_meta("clip.vision.patch_size", VISION_PATCH_SIZE)
    gguf.add_meta("clip.vision.projection_dim", VISION_PROJECTION_DIM)
    gguf.add_meta("clip.vision.spatial_merge_size", VISION_SPATIAL_MERGE_SIZE)
    # `clip.vision.attention.layer_norm_epsilon` is the rms_norm eps used
    # inside ViT blocks; reuse text-side value (Qwen3-VL convention; see
    # `references/llama.cpp/conversion/qwen3vl.py::set_gguf_parameters`).
    gguf.add_meta(
        "clip.vision.attention.layer_norm_epsilon",
        cfg.get("text_config", {}).get("rms_norm_eps", TEXT_NORM_EPS),
    )
    gguf.add_meta("clip.use_gelu", True)
    gguf.add_meta("clip.vision.image_mean", [0.5, 0.5, 0.5])
    gguf.add_meta("clip.vision.image_std", [0.5, 0.5, 0.5])
    # Mark which ViT layers carry a deepstack merger; engine disables a
    # layer if the corresponding `v.deepstack.{i}.norm.weight` is missing.
    deepstack = [False] * VISION_BLOCK_COUNT
    for idx in DEEPSTACK_VISUAL_INDEXES:
        deepstack[idx] = True
    gguf.add_meta("clip.vision.is_deepstack_layers", deepstack)

    # ---- tensors ----
    # Patch embed: Conv3D [out, in, kt=2, kH, kW] → 2 Conv2D weights + bias.
    patch_w = _require(st, "model.visual.patch_embed.proj.weight")
    frame0_raw, frame0_shape, frame1_raw, frame1_shape = _split_conv3d(
        patch_w.raw, tuple(patch_w.shape),
    )
    _emit_bf16(gguf, "v.patch_embd.weight", frame0_raw, frame0_shape)
    _emit_bf16(gguf, "v.patch_embd.weight.1", frame1_raw, frame1_shape)
    patch_b = _require(st, "model.visual.patch_embed.proj.bias")
    _emit_bf16(gguf, "v.patch_embd.bias", patch_b.raw, tuple(patch_b.shape))

    pos = _require(st, "model.visual.pos_embed.weight")
    if tuple(pos.shape) != (VISION_NUM_POSITION_EMBEDDINGS, VISION_EMBEDDING_LENGTH):
        raise ValueError(
            f"pos_embed shape {pos.shape} != "
            f"({VISION_NUM_POSITION_EMBEDDINGS}, {VISION_EMBEDDING_LENGTH})"
        )
    _emit_bf16(gguf, "v.position_embd.weight", pos.raw, tuple(pos.shape))

    for i in range(VISION_BLOCK_COUNT):
        layer_prefix = f"model.visual.blocks.{i}"
        block_prefix = f"v.blk.{i}"

        def emit_ln(hf_sub: str, gguf_name: str) -> None:
            w = _require(st, f"{layer_prefix}.{hf_sub}.weight")
            _emit_bf16_norm(
                gguf, f"{block_prefix}.{gguf_name}.weight", w.raw, tuple(w.shape),
            )
            if (b := st.get(f"{layer_prefix}.{hf_sub}.bias")) is not None:
                _emit_bf16_norm(
                    gguf,
                    f"{block_prefix}.{gguf_name}.bias",
                    b.raw,
                    tuple(b.shape),
                )

        # Norms → F32 (engine reads them with `get_f32_tensor`-style paths).
        emit_ln("norm1", "ln1")
        emit_ln("norm2", "ln2")

        for hf_name, gguf_name in [
            ("attn.qkv", "attn_qkv"),
            ("attn.proj", "attn_out"),
            ("mlp.linear_fc1", "ffn_up"),
            ("mlp.linear_fc2", "ffn_down"),
        ]:
            w = _require(st, f"{layer_prefix}.{hf_name}.weight")
            _emit_bf16(gguf, f"{block_prefix}.{gguf_name}.weight", w.raw, tuple(w.shape))
            if (b := st.get(f"{layer_prefix}.{hf_name}.bias")) is not None:
                _emit_bf16_norm(
                    gguf,
                    f"{block_prefix}.{gguf_name}.bias",
                    b.raw,
                    tuple(b.shape),
                )

    # Deepstack mergers: map ds_idx → absolute ViT layer index, write under
    # the engine's `v.deepstack.{layer_idx}.{norm/fc1/fc2}.{weight/bias}` names.
    for ds_idx, layer_idx in enumerate(DEEPSTACK_VISUAL_INDEXES):
        prefix = f"model.visual.deepstack_merger_list.{ds_idx}"
        out = f"v.deepstack.{layer_idx}"

        for hf_name, gguf_name in [
            ("norm", "norm"),
            ("linear_fc1", "fc1"),
            ("linear_fc2", "fc2"),
        ]:
            w = _require(st, f"{prefix}.{hf_name}.weight")
            if hf_name == "norm":
                _emit_bf16_norm(gguf, f"{out}.{gguf_name}.weight", w.raw, tuple(w.shape))
            else:
                _emit_bf16(gguf, f"{out}.{gguf_name}.weight", w.raw, tuple(w.shape))
            if (b := st.get(f"{prefix}.{hf_name}.bias")) is not None:
                _emit_bf16_norm(
                    gguf, f"{out}.{gguf_name}.bias", b.raw, tuple(b.shape),
                )

    # Final merger: norm → v.post_ln, fc1 → mm.0, fc2 → mm.2.
    out_prefix = "model.visual.merger"
    norm_w = _require(st, f"{out_prefix}.norm.weight")
    _emit_bf16_norm(gguf, "v.post_ln.weight", norm_w.raw, tuple(norm_w.shape))
    if (norm_b := st.get(f"{out_prefix}.norm.bias")) is not None:
        _emit_bf16_norm(
            gguf, "v.post_ln.bias", norm_b.raw, tuple(norm_b.shape),
        )

    fc1_w = _require(st, f"{out_prefix}.linear_fc1.weight")
    _emit_bf16(gguf, "mm.0.weight", fc1_w.raw, tuple(fc1_w.shape))
    if (fc1_b := st.get(f"{out_prefix}.linear_fc1.bias")) is not None:
        _emit_bf16_norm(
            gguf, "mm.0.bias", fc1_b.raw, tuple(fc1_b.shape),
        )

    fc2_w = _require(st, f"{out_prefix}.linear_fc2.weight")
    _emit_bf16(gguf, "mm.2.weight", fc2_w.raw, tuple(fc2_w.shape))
    if (fc2_b := st.get(f"{out_prefix}.linear_fc2.bias")) is not None:
        _emit_bf16_norm(
            gguf, "mm.2.bias", fc2_b.raw, tuple(fc2_b.shape),
        )

    gguf.write()
    print(
        f"[mmproj] wrote {out_path} ({out_path.stat().st_size // 1024 // 1024} MiB)",
        file=sys.stderr,
    )


def convert(hf_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    text_path = out_dir / "physbrain-1.5-2b-Q8_0.gguf"
    mmproj_path = out_dir / "mmproj-physbrain-1.5-2b-F16.gguf"
    write_text_gguf(hf_dir, text_path)
    write_mmproj_gguf(hf_dir, mmproj_path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert DeepCybo/PhysBrain1.5-2B safetensors → 2 GGUFs",
    )
    ap.add_argument(
        "--hf-dir", required=True, type=Path,
        help="HF checkpoint dir (config.json + tokenizer.json + model.safetensors)",
    )
    ap.add_argument(
        "--out-dir", required=True, type=Path,
        help="Output directory (text + mmproj GGUFs will be written here)",
    )
    ap.add_argument(
        "--only", choices=["text", "mmproj"], default=None,
        help="Emit only one of the two outputs (debug aid).",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "text"):
        write_text_gguf(args.hf_dir, args.out_dir / "physbrain-1.5-2b-Q8_0.gguf")
    if args.only in (None, "mmproj"):
        write_mmproj_gguf(
            args.hf_dir, args.out_dir / "mmproj-physbrain-1.5-2b-F16.gguf",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
