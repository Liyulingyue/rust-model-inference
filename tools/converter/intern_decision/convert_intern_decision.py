#!/usr/bin/env python3
"""Convert `Shanghai_AI_Laboratory/Intern-Decision-{0.8B,4B}` (Qwen3.5 multimodal
hybrid + a vision projector) into a `qwen35` text GGUF and a
`clip`/`qwen3vl_merger` mmproj GGUF.

Inputs:
* HF safetensors checkpoint for `Intern-Decision-{0.8B,4B}` containing:
  - `config.json` — Qwen3.5 hybrid text config + a vision config (depth=12,
    hidden=768, FFN=3072, 12 heads, num_position_embeddings=2304) +
    `architectures=["Qwen3_5ForConditionalGeneration"]`.
  - `tokenizer.json` + `tokenizer_config.json` — BPE vocab + merges.
  - `model-language-*.safetensors` — text shards with names
    `model.language_model.layers.{i}.{linear_attn,self_attn,mlp}.*` (linear
    for the SSM layers, self_attn for the periodic full-attention layers
    every 4th step), `embed_tokens.weight`, `norm.weight`. Per-shape:
    - 0.8B: 24 layers / hidden 1024 / FFN 3584 / 8Q:2KV / GQA-4 / vocab 248320
    - 4B:  36 layers / hidden 2560 / FFN 9728 / 16Q:2KV / GQA-8 / vocab 248320
  - `model-projector-*.safetensors` — vision merger weights
    (`model.visual.merger.{norm,linear_fc1,linear_fc2}`).
  - Vision tower (`model.visual.{blocks,patch_embed,pos_embed}`) ships
    separately in the language shards but its config (hidden_size=768) matches
    our existing `qwen3vl_merger` path.

Output (per variant):
* `<out-dir>/intern-decision-{0.8b,4b}-Q8_0.gguf` — text GGUF (arch=`qwen35`).
* `<out-dir>/mmproj-intern-decision-{0.8b,4b}-F16.gguf` — mmproj GGUF
  (arch=`clip`, `clip.projector_type=qwen3vl_merger`).

Reference for metadata keys + tensor naming:
* `references/llama.cpp/conversion/qwen3vl.py` (mmproj metadata)
* `models/Qwen3.5-{0.8B,2B,4B}-GGUF/*.gguf` (text metadata, already shipped
  in this repo — same arch `qwen35`, same field names).

Usage:
    python3 tools/converter/intern_decision/convert_intern_decision.py \\
        --hf-dir models/intern-decision-0.8b \\
        --variant 0.8b \\
        --out-dir models/intern-decision-0.8b-GGUF
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

from tools.converter.utils.gguf import (
    model_metadata_value,
    GGML_BF16,
    GGML_F32,
    GGML_F16,
    GGML_Q8_0,
    GgufWriter,
    bf16_bytes_to_q8_0,
    bf16_to_f16,
    bf16_to_f32,
    gguf_dims,
    open_safetensors,
)


# Per-variant constants. Pinned by hand from `config.json` so any drift
# (e.g. a retrained variant that retunes head_dim) is caught at conversion.
VARIANT_TABLE = {
    "0.8b": dict(
        text_n_embd=1024,
        text_n_layer=24,
        text_n_head=8,
        text_n_head_kv=2,
        text_n_ff=3584,
        text_head_dim=256,  # Explicit head_dim differs from hidden_size / head_count.
        text_vocab=248_320,
        vision_depth=12,
        vision_embd=768,
        vision_ff=3072,
        vision_head=12,
        vision_patch=16,
        vision_image=768,
        vision_merge=2,
        vision_pos=2304,
        vision_proj=1024,
    ),
    "4b": dict(
        text_n_embd=2560,
        text_n_layer=36,
        text_n_head=16,
        text_n_head_kv=2,
        text_n_ff=9728,
        text_head_dim=160,  # = 2560 / 16
        text_vocab=248_320,
        vision_depth=12,
        vision_embd=768,
        vision_ff=3072,
        vision_head=12,
        vision_patch=16,
        vision_image=768,
        vision_merge=2,
        vision_pos=2304,
        vision_proj=2560,
    ),
}

TEXT_ARCH = "qwen35"
TEXT_CONTEXT_LENGTH = 262_144
TEXT_NORM_EPS = 1e-6
TEXT_ROPE_FREQ_BASE = 10_000_000.0
TEXT_DIMENSION_COUNT = 64
TEXT_ROPE_SECTIONS = [11, 11, 10, 0]
SSM_CONV_KERNEL = 4
SSM_INNER_SIZE = 2048
SSM_STATE_SIZE = 128
SSM_TIME_STEP_RANK = 16
SSM_GROUP_COUNT = 16
FULL_ATTENTION_INTERVAL = 4
IMAGE_TOKEN_ID = 248056
VISION_START_TOKEN_ID = 248053
VISION_END_TOKEN_ID = 248054


def _read_hf_config(hf_dir: Path) -> dict:
    cfg_path = hf_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing HF config.json at {cfg_path}")
    return json.loads(cfg_path.read_text())


def _read_tokenizer(hf_dir: Path) -> tuple[dict, dict, list[str]]:
    tok_path = hf_dir / "tokenizer.json"
    if not tok_path.is_file():
        raise FileNotFoundError(f"missing tokenizer.json at {tok_path}")
    tok = json.loads(tok_path.read_text())
    cfg_path = hf_dir / "tokenizer_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing tokenizer_config.json at {cfg_path}")
    return (
        json.loads(cfg_path.read_text()),
        tok,
        tok["model"]["merges"],
    )


def _build_vocab(
    tokenizer_cfg: dict, tok_json: dict, vocab_size: int
) -> tuple[list[str], list[int]]:
    """Build `(tokens, types)` lists for `tokenizer.ggml.tokens`. 1 = NORMAL,
    3 = CONTROL (special added tokens). Pad any unused slots with reserved
    placeholders.
    """
    vocab = {int(v): k for k, v in tok_json["model"]["vocab"].items()}
    added = {entry["id"]: entry["content"] for entry in tok_json["added_tokens"]}
    all_tokens: dict[int, str] = {**vocab, **added}
    tokens = [all_tokens.get(i, f"<|reserved_{i}|>") for i in range(vocab_size)]
    types = [1] * vocab_size
    for entry in tok_json["added_tokens"]:
        tid = entry["id"]
        if tid < vocab_size:
            types[tid] = 3 if entry["special"] else 4
    return tokens, types


def _open_safetensors(hf_dir: Path, glob: str):
    """Lazy mmap each matching shard; we never need to load more than a
    single tensor at a time so a per-shard dict is fine."""
    shards = sorted(hf_dir.glob(glob))
    if not shards:
        raise FileNotFoundError(f"no shards matching {glob} under {hf_dir}")
    return [(shard, open_safetensors(shard)) for shard in shards]


def _require(st_dicts, name: str):
    """`Tensor` accessor with clear errors; searches across all opened shards."""
    for _shard, st in st_dicts:
        if name in st.header:
            return st.get(name)
    raise KeyError(f"missing safetensors tensor: {name}")


def _emit_f32_or_bf16_to_f32(
    gguf: GgufWriter, name: str, raw: bytes, shape: tuple, src_dtype: str,
    *, add_one: bool = False, neg_exp: bool = False,
) -> None:
    """Coerce either an F32 source (already F32) or a BF16 source (cast to
    F32 with the same width) into a F32 GGUF tensor. The qwen35 trunk
    expects these as F32; we keep the source dtype intact otherwise."""
    n = 1
    for d in shape:
        n *= d
    if src_dtype == "F32":
        if len(raw) != n * 4:
            raise ValueError(
                f"{name}: shape {shape} ({n}) != f32 payload {len(raw) // 4}"
            )
        payload = raw
    elif src_dtype == "BF16":
        if len(raw) != n * 2:
            raise ValueError(
                f"{name}: shape {shape} ({n}) != bf16 payload {len(raw) // 2}"
            )
        payload = bf16_to_f32(raw)
    else:
        raise ValueError(f"{name}: unsupported source dtype {src_dtype!r}")
    if add_one or neg_exp:
        values = struct.unpack(f"<{n}f", payload)
        if add_one:
            values = tuple(value + 1.0 for value in values)
        if neg_exp:
            values = tuple(-math.exp(value) for value in values)
        payload = struct.pack(f"<{n}f", *values)
    gguf.add_tensor(name, GGML_F32, gguf_dims(shape), payload)


def _emit_f32(gguf: GgufWriter, name: str, raw: bytes, shape: tuple) -> None:
    """Norms / biases stored as F32 in the qwen35 trunk
    (`src/models/qwen35/trunk/weights.rs::get_f32_tensor`)."""
    n = 1
    for d in shape:
        n *= d
    if len(raw) != n * 4:
        raise ValueError(f"{name}: shape {shape} ({n}) != f32 payload {len(raw) // 4}")
    gguf.add_tensor(name, GGML_F32, gguf_dims(shape), raw)


def _emit_bf16_f32(
    gguf: GgufWriter, name: str, raw: bytes, shape: tuple, *, add_one: bool = False,
) -> None:
    """BF16 -> F32 conversion for BF16 norms / biases the engine reads as F32."""
    n = 1
    for d in shape:
        n *= d
    if len(raw) != n * 2:
        raise ValueError(
            f"{name}: shape {shape} ({n}) != bf16 payload {len(raw) // 2}"
        )
    _emit_f32_or_bf16_to_f32(
        gguf, name, raw, shape, "BF16", add_one=add_one,
    )


def _emit_bf16_q8_0(gguf: GgufWriter, name: str, raw: bytes, shape: tuple) -> None:
    """BF16 -> Q8_0 for learned linear weights."""
    n = 1
    for d in shape:
        n *= d
    if len(raw) != n * 2:
        raise ValueError(
            f"{name}: shape {shape} ({n}) != bf16 payload {len(raw) // 2}"
        )
    gguf.add_tensor(name, GGML_Q8_0, gguf_dims(shape), bf16_bytes_to_q8_0(raw))


def _emit_bf16(gguf: GgufWriter, name: str, raw: bytes, shape: tuple) -> None:
    """Keep BF16 (token embeddings, mmproj weights)."""
    n = 1
    for d in shape:
        n *= d
    if len(raw) != n * 2:
        raise ValueError(
            f"{name}: shape {shape} ({n}) != bf16 payload {len(raw) // 2}"
        )
    gguf.add_tensor(name, GGML_BF16, gguf_dims(shape), raw)


def _emit_bf16_f16(gguf: GgufWriter, name: str, raw: bytes, shape: tuple) -> None:
    """Convert BF16 source weights to the F16 storage used by vision linear layers."""
    n = 1
    for d in shape:
        n *= d
    if len(raw) != n * 2:
        raise ValueError(f"{name}: shape {shape} ({n}) != bf16 payload {len(raw) // 2}")
    gguf.add_tensor(name, GGML_F16, gguf_dims(shape), bf16_to_f16(raw))


def _is_full_attention(idx: int) -> bool:
    """Qwen3.5 layer-type pattern: layers [3, 7, 11, ...] (every 4th) are full
    attention; everything else is linear (Mamba SSM)."""
    return (idx + 1) % FULL_ATTENTION_INTERVAL == 0


def write_text_gguf(
    hf_dir: Path,
    out_path: Path,
    *,
    variant: str,
) -> None:
    cfg = _read_hf_config(hf_dir)
    tok_cfg, tok_json, merges = _read_tokenizer(hf_dir)
    p = VARIANT_TABLE[variant]

    text = cfg.get("text_config", {})
    if (
        text["hidden_size"] != p["text_n_embd"]
        or text["num_hidden_layers"] != p["text_n_layer"]
        or text["num_attention_heads"] != p["text_n_head"]
        or text["num_key_value_heads"] != p["text_n_head_kv"]
        or text["intermediate_size"] != p["text_n_ff"]
        or text["vocab_size"] != p["text_vocab"]
    ):
        raise ValueError(
            f"text_config dims mismatch with variant {variant!r}: "
            f"hidden={text['hidden_size']}, layers={text['num_hidden_layers']}, "
            f"heads={text['num_attention_heads']}/{text['num_key_value_heads']}, "
            f"ff={text['intermediate_size']}, vocab={text['vocab_size']}"
        )

    tokens, types = _build_vocab(tok_cfg, tok_json, p["text_vocab"])
    lang_st = _open_safetensors(hf_dir, "model-language*.safetensors")

    gguf = GgufWriter(out_path)

    # ---- general.* ----
    gguf.add_meta("general.architecture", TEXT_ARCH)
    gguf.add_meta("general.name", f"Intern-Decision-{variant}")
    gguf.add_meta("general.basename", f"Intern-Decision-{variant}")
    gguf.add_meta("general.size_label", "0.8B" if variant == "0.8b" else "4B")
    gguf.add_meta("general.file_type", 7)  # mostly Q8_0
    gguf.add_meta("general.quantization_version", 2)
    gguf.add_meta(
        "general.base_model.0.name",
        "Qwen3.5-0.8B" if variant == "0.8b" else "Qwen3.5-4B",
    )
    gguf.add_meta(
        "general.base_model.0.organization", "Qwen",
    )
    gguf.add_meta(
        "general.base_model.0.repo_url",
        "https://huggingface.co/Qwen/"
        + ("Qwen3.5-0.8B" if variant == "0.8b" else "Qwen3.5-4B"),
    )
    gguf.add_meta("general.base_model.count", 1)

    # ---- qwen35.* ----
    gguf.add_meta("qwen35.embedding_length", p["text_n_embd"])
    gguf.add_meta("qwen35.block_count", p["text_n_layer"])
    gguf.add_meta("qwen35.context_length", TEXT_CONTEXT_LENGTH)
    gguf.add_meta("qwen35.feed_forward_length", p["text_n_ff"])
    gguf.add_meta("qwen35.attention.head_count", p["text_n_head"])
    gguf.add_meta("qwen35.attention.head_count_kv", p["text_n_head_kv"])
    gguf.add_meta("qwen35.attention.key_length", p["text_head_dim"])
    gguf.add_meta("qwen35.attention.value_length", p["text_head_dim"])
    gguf.add_meta("qwen35.attention.layer_norm_rms_epsilon", TEXT_NORM_EPS)
    gguf.add_meta("qwen35.rope.freq_base", TEXT_ROPE_FREQ_BASE)
    gguf.add_meta("qwen35.rope.dimension_count", TEXT_DIMENSION_COUNT)
    gguf.add_meta("qwen35.rope.dimension_sections", TEXT_ROPE_SECTIONS)
    gguf.add_meta("qwen35.vocab_size", p["text_vocab"])
    # SSM (Mamba) constants.
    gguf.add_meta("qwen35.ssm.conv_kernel", SSM_CONV_KERNEL)
    gguf.add_meta("qwen35.ssm.inner_size", SSM_INNER_SIZE)
    gguf.add_meta("qwen35.ssm.state_size", SSM_STATE_SIZE)
    gguf.add_meta("qwen35.ssm.time_step_rank", SSM_TIME_STEP_RANK)
    gguf.add_meta("qwen35.ssm.group_count", SSM_GROUP_COUNT)
    gguf.add_meta("qwen35.full_attention_interval", FULL_ATTENTION_INTERVAL)
    # Image placeholder ids (engine reads these from metadata).
    gguf.add_meta("qwen35.image_token_id", IMAGE_TOKEN_ID)
    gguf.add_meta("qwen35.vision_start_token_id", VISION_START_TOKEN_ID)
    gguf.add_meta("qwen35.vision_end_token_id", VISION_END_TOKEN_ID)

    # ---- tokenizer.ggml.* ----
    gguf.add_meta("tokenizer.ggml.model", "gpt2")
    gguf.add_meta("tokenizer.ggml.pre", "qwen2")
    gguf.add_meta("tokenizer.ggml.tokens", tokens)
    gguf.add_meta("tokenizer.ggml.token_type", types)
    gguf.add_meta("tokenizer.ggml.merges", [" ".join(m) for m in merges])
    bos_id = tok_cfg.get("bos_token_id") or 248044
    eos_id = tok_cfg.get("eos_token_id") or 248044
    gguf.add_meta("tokenizer.ggml.bos_token_id", int(bos_id))
    gguf.add_meta("tokenizer.ggml.eos_token_id", int(eos_id))
    gguf.add_meta("tokenizer.ggml.add_bos_token", False)
    gguf.add_meta("tokenizer.ggml.add_eos_token", False)

    # ---- tensors ----
    embed = _require(lang_st, "model.language_model.embed_tokens.weight")
    if tuple(embed.shape) != (p["text_vocab"], p["text_n_embd"]):
        raise ValueError(
            f"embed_tokens shape {embed.shape} != "
            f"({p['text_vocab']}, {p['text_n_embd']})"
        )
    _emit_bf16(gguf, "token_embd.weight", embed.raw, tuple(embed.shape))
    # tie_word_embeddings=true → no `output.weight`; engine constructs output
    # from `token_embd.weight` (see `src/models/qwen35/trunk/weights.rs`).

    for i in range(p["text_n_layer"]):
        layer = f"model.language_model.layers.{i}"
        blk = f"blk.{i}"
        full_attn = _is_full_attention(i)

        # Norms → F32 (BF16 source converted to F32).
        ln1 = _require(lang_st, f"{layer}.input_layernorm.weight")
        _emit_bf16_f32(
            gguf, f"{blk}.attn_norm.weight", ln1.raw, tuple(ln1.shape), add_one=True,
        )
        ln2 = _require(lang_st, f"{layer}.post_attention_layernorm.weight")
        _emit_bf16_f32(
            gguf, f"{blk}.post_attention_norm.weight", ln2.raw, tuple(ln2.shape),
            add_one=True,
        )

        # FFN → Q8_0.
        for hf_name, gguf_name in [
            ("mlp.gate_proj", "ffn_gate"),
            ("mlp.up_proj", "ffn_up"),
            ("mlp.down_proj", "ffn_down"),
        ]:
            t = _require(lang_st, f"{layer}.{hf_name}.weight")
            _emit_bf16_q8_0(
                gguf, f"{blk}.{gguf_name}.weight", t.raw, tuple(t.shape),
            )

        if full_attn:
            # Q/K/V/O projections + per-head QK RMSNorm → engine expects
            # `blk.{i}.attn_{q,k,v,output}.weight` + `attn_{q,k}_norm.weight`.
            for hf_name, gguf_name in [
                ("self_attn.q_proj", "attn_q"),
                ("self_attn.k_proj", "attn_k"),
                ("self_attn.v_proj", "attn_v"),
                ("self_attn.o_proj", "attn_output"),
            ]:
                t = _require(lang_st, f"{layer}.{hf_name}.weight")
                _emit_bf16_q8_0(
                    gguf, f"{blk}.{gguf_name}.weight", t.raw, tuple(t.shape),
                )
            qn = _require(lang_st, f"{layer}.self_attn.q_norm.weight")
            _emit_bf16_f32(
                gguf, f"{blk}.attn_q_norm.weight", qn.raw, tuple(qn.shape), add_one=True,
            )
            kn = _require(lang_st, f"{layer}.self_attn.k_norm.weight")
            _emit_bf16_f32(
                gguf, f"{blk}.attn_k_norm.weight", kn.raw, tuple(kn.shape), add_one=True,
            )
        else:
            # Linear (Mamba) SSM block: in_proj_qkv / in_proj_z / in_proj_a /
            # in_proj_b / conv1d / norm / out_proj / A_log / dt.bias.
            qkv = _require(lang_st, f"{layer}.linear_attn.in_proj_qkv.weight")
            _emit_bf16_q8_0(
                gguf, f"{blk}.attn_qkv.weight", qkv.raw, tuple(qkv.shape),
            )
            z = _require(lang_st, f"{layer}.linear_attn.in_proj_z.weight")
            _emit_bf16_q8_0(
                gguf, f"{blk}.attn_gate.weight", z.raw, tuple(z.shape),
            )
            alpha = _require(lang_st, f"{layer}.linear_attn.in_proj_a.weight")
            _emit_bf16_q8_0(
                gguf, f"{blk}.ssm_alpha.weight", alpha.raw, tuple(alpha.shape),
            )
            beta = _require(lang_st, f"{layer}.linear_attn.in_proj_b.weight")
            _emit_bf16_q8_0(
                gguf, f"{blk}.ssm_beta.weight", beta.raw, tuple(beta.shape),
            )
            conv1d = _require(lang_st, f"{layer}.linear_attn.conv1d.weight")
            if len(conv1d.shape) != 3 or conv1d.shape[1] != 1:
                raise ValueError(f"unexpected depthwise conv shape: {conv1d.shape}")
            # `ssm_conv1d.weight` is F32 in the engine (matches what
            # Qwen3.5-0.8B / Qwen3.5-2B / Qwen3.5-4B GGUF ships); we cast the
            # BF16 source to F32 with a width-preserving conversion.
            _emit_bf16_f32(
                gguf,
                f"{blk}.ssm_conv1d.weight",
                conv1d.raw,
                (conv1d.shape[0], conv1d.shape[2]),
            )
            ssm_norm = _require(lang_st, f"{layer}.linear_attn.norm.weight")
            # `linear_attn.norm.weight` is F32 in HF; some older exports
            # carry BF16 (engine reads it as F32 either way).
            _emit_f32_or_bf16_to_f32(
                gguf, f"{blk}.ssm_norm.weight", ssm_norm.raw, tuple(ssm_norm.shape),
                ssm_norm.dtype,
            )
            out_proj = _require(lang_st, f"{layer}.linear_attn.out_proj.weight")
            _emit_bf16_q8_0(
                gguf, f"{blk}.ssm_out.weight", out_proj.raw, tuple(out_proj.shape),
            )
            # `linear_attn.A_log` is F32 in HF (`dtype: 'F32'` per the shard
            # dump) and is small (16 values). Cast BF16 to F32 for safety.
            a_log = _require(lang_st, f"{layer}.linear_attn.A_log")
            _emit_f32_or_bf16_to_f32(
                gguf, f"{blk}.ssm_a", a_log.raw, tuple(a_log.shape),
                a_log.dtype, neg_exp=True,
            )
            # `dt.bias` is BF16 in HF; cast to F32 (matches Qwen3.5-0.8B
            # GGUF which already emits `blk.{i}.ssm_dt.bias` as F32).
            dt_bias = _require(lang_st, f"{layer}.linear_attn.dt_bias")
            _emit_f32_or_bf16_to_f32(
                gguf, f"{blk}.ssm_dt.bias", dt_bias.raw, tuple(dt_bias.shape),
                dt_bias.dtype,
            )

    # Final norm.
    out_norm = _require(lang_st, "model.language_model.norm.weight")
    _emit_bf16_f32(
        gguf, "output_norm.weight", out_norm.raw, tuple(out_norm.shape), add_one=True,
    )

    gguf.metadata = [(key, model_metadata_value(value)) for key, value in gguf.metadata]
    gguf.write()
    print(
        f"[text] wrote {out_path} ({out_path.stat().st_size // 1024 // 1024} MiB)",
        file=sys.stderr,
    )


def write_mmproj_gguf(
    hf_dir: Path,
    out_path: Path,
    *,
    variant: str,
) -> None:
    cfg = _read_hf_config(hf_dir)
    p = VARIANT_TABLE[variant]
    proj_st = _open_safetensors(hf_dir, "model-projector*.safetensors")
    # Vision tower weights live in their own shard in this release.
    vision_st = _open_safetensors(hf_dir, "model-vision*.safetensors")

    gguf = GgufWriter(out_path)

    gguf.add_meta("general.architecture", "clip")
    gguf.add_meta("general.name", f"Intern-Decision-{variant}-mmproj")
    gguf.add_meta("general.file_type", 1)  # F16 convention.

    gguf.add_meta("clip.has_vision_encoder", True)
    gguf.add_meta("clip.projector_type", "qwen3vl_merger")
    gguf.add_meta("clip.vision.embedding_length", p["vision_embd"])
    gguf.add_meta("clip.vision.feed_forward_length", p["vision_ff"])
    gguf.add_meta("clip.vision.block_count", p["vision_depth"])
    gguf.add_meta("clip.vision.attention.head_count", p["vision_head"])
    gguf.add_meta("clip.vision.image_size", p["vision_image"])
    gguf.add_meta("clip.vision.patch_size", p["vision_patch"])
    gguf.add_meta("clip.vision.projection_dim", p["vision_proj"])
    gguf.add_meta("clip.vision.spatial_merge_size", p["vision_merge"])
    gguf.add_meta(
        "clip.vision.attention.layer_norm_epsilon",
        cfg.get("text_config", {}).get("rms_norm_eps", TEXT_NORM_EPS),
    )
    gguf.add_meta("clip.use_gelu", True)
    gguf.add_meta("clip.vision.image_mean", [0.5, 0.5, 0.5])
    gguf.add_meta("clip.vision.image_std", [0.5, 0.5, 0.5])
    # Intern-Decision ships with deepstack_visual_indexes=[]; emit an
    # all-False list so the engine does not try to load `v.deepstack.{i}.*`.
    gguf.add_meta(
        "clip.vision.is_deepstack_layers",
        [False] * p["vision_depth"],
    )

    # The vision tower weights live in `model-vision*.safetensors` for
    # Intern-Decision (separate shard from the language model).
    lang_st = vision_st

    # Vision tower: 12 ViT blocks (hidden=768, FFN=3072). The trunk already
    # loads `v.blk.{i}.{ln1,ln2,attn_qkv,attn_out,ffn_up,ffn_down}.{weight,bias}`
    # for the qwen3vl merger (see `src/models/qwen3/vision/mod.rs`).
    for i in range(p["vision_depth"]):
        layer = f"model.visual.blocks.{i}"
        blk = f"v.blk.{i}"
        for hf_name, gguf_name in [
            ("norm1.weight", "ln1.weight"),
            ("norm1.bias", "ln1.bias"),
            ("norm2.weight", "ln2.weight"),
            ("norm2.bias", "ln2.bias"),
        ]:
            t = _require(lang_st, f"{layer}.{hf_name}")
            _emit_bf16_f32(
                gguf, f"{blk}.{gguf_name}", t.raw, tuple(t.shape),
            )
        for hf_name, gguf_name in [
            ("attn.qkv.weight", "attn_qkv.weight"),
            ("attn.qkv.bias", "attn_qkv.bias"),
            ("attn.proj.weight", "attn_out.weight"),
            ("attn.proj.bias", "attn_out.bias"),
            ("mlp.linear_fc1.weight", "ffn_up.weight"),
            ("mlp.linear_fc1.bias", "ffn_up.bias"),
            ("mlp.linear_fc2.weight", "ffn_down.weight"),
            ("mlp.linear_fc2.bias", "ffn_down.bias"),
        ]:
            t = _require(lang_st, f"{layer}.{hf_name}")
            if hf_name.endswith(".weight"):
                _emit_bf16_f16(gguf, f"{blk}.{gguf_name}", t.raw, tuple(t.shape))
            else:
                _emit_bf16_f32(gguf, f"{blk}.{gguf_name}", t.raw, tuple(t.shape))

    # Patch embed: Conv3D [out, in, kt=2, kH, kW] -> two Conv2D weights + bias.
    patch_w = _require(lang_st, "model.visual.patch_embed.proj.weight")
    out_c, in_c, kt, kH, kW = patch_w.shape
    if kt != 2:
        raise ValueError(
            f"Intern-Decision patch_embed Conv3D requires kt=2, got {kt}"
        )
    import numpy as np
    bf16 = np.frombuffer(patch_w.raw, dtype="<u2").reshape(
        out_c, in_c, 2, kH * kW
    )
    frame0 = bf16[:, :, 0, :].reshape(out_c, in_c, kH, kW).copy()
    frame1 = bf16[:, :, 1, :].reshape(out_c, in_c, kH, kW).copy()
    torch_shape = (out_c, in_c, kH, kW)
    _emit_bf16_f16(gguf, "v.patch_embd.weight", frame0.tobytes(), torch_shape)
    _emit_bf16_f16(gguf, "v.patch_embd.weight.1", frame1.tobytes(), torch_shape)
    patch_b = _require(lang_st, "model.visual.patch_embed.proj.bias")
    _emit_bf16_f32(
        gguf, "v.patch_embd.bias", patch_b.raw, tuple(patch_b.shape),
    )
    pos = _require(lang_st, "model.visual.pos_embed.weight")
    if tuple(pos.shape) != (p["vision_pos"], p["vision_embd"]):
        raise ValueError(
            f"pos_embed shape {pos.shape} != "
            f"({p['vision_pos']}, {p['vision_embd']})"
        )
    _emit_bf16_f32(gguf, "v.position_embd.weight", pos.raw, tuple(pos.shape))

    # Merger: norm -> v.post_ln; fc1 -> mm.0; fc2 -> mm.2.
    out_prefix = "model.visual.merger"
    nw = _require(proj_st, f"{out_prefix}.norm.weight")
    _emit_bf16_f32(gguf, "v.post_ln.weight", nw.raw, tuple(nw.shape))
    if (nb := _require_safe(proj_st, f"{out_prefix}.norm.bias")) is not None:
        _emit_bf16_f32(gguf, "v.post_ln.bias", nb.raw, tuple(nb.shape))

    fc1w = _require(proj_st, f"{out_prefix}.linear_fc1.weight")
    _emit_bf16_f16(gguf, "mm.0.weight", fc1w.raw, tuple(fc1w.shape))
    if (fc1b := _require_safe(proj_st, f"{out_prefix}.linear_fc1.bias")) is not None:
        _emit_bf16_f32(gguf, "mm.0.bias", fc1b.raw, tuple(fc1b.shape))

    fc2w = _require(proj_st, f"{out_prefix}.linear_fc2.weight")
    _emit_bf16_f16(gguf, "mm.2.weight", fc2w.raw, tuple(fc2w.shape))
    if (fc2b := _require_safe(proj_st, f"{out_prefix}.linear_fc2.bias")) is not None:
        _emit_bf16_f32(gguf, "mm.2.bias", fc2b.raw, tuple(fc2b.shape))

    gguf.metadata = [(key, model_metadata_value(value)) for key, value in gguf.metadata]
    gguf.write()
    print(
        f"[mmproj] wrote {out_path} ({out_path.stat().st_size // 1024 // 1024} MiB)",
        file=sys.stderr,
    )


def _require_safe(st_dicts, name: str):
    for _shard, st in st_dicts:
        if name in st.header:
            return st.get(name)
    return None


def convert(hf_dir: Path, out_dir: Path, *, variant: str) -> None:
    if variant not in VARIANT_TABLE:
        raise ValueError(f"unknown variant {variant!r}; pick one of {list(VARIANT_TABLE)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "Q8_0"
    text_path = out_dir / f"intern-decision-{variant}-{suffix}.gguf"
    mmproj_path = out_dir / f"mmproj-intern-decision-{variant}-F16.gguf"
    write_text_gguf(hf_dir, text_path, variant=variant)
    write_mmproj_gguf(hf_dir, mmproj_path, variant=variant)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert Intern-Decision safetensors -> qwen35 + mmproj GGUFs",
    )
    ap.add_argument(
        "--hf-dir", required=True, type=Path,
        help="HF checkpoint dir (config.json + tokenizer.json + model-language-*.safetensors + model-projector-*.safetensors)",
    )
    ap.add_argument(
        "--variant", required=True, choices=["0.8b", "4b"],
        help="Pin which model variant the converter expects.",
    )
    ap.add_argument(
        "--out-dir", required=True, type=Path,
        help="Output directory (text + mmproj GGUFs will be written here).",
    )
    ap.add_argument(
        "--only", choices=["text", "mmproj"], default=None,
    )
    args = ap.parse_args()
    if args.variant not in VARIANT_TABLE:
        raise SystemExit(f"unknown variant {args.variant!r}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "text"):
        write_text_gguf(
            args.hf_dir,
            args.out_dir / f"intern-decision-{args.variant}-Q8_0.gguf",
            variant=args.variant,
        )
    if args.only in (None, "mmproj"):
        write_mmproj_gguf(
            args.hf_dir,
            args.out_dir / f"mmproj-intern-decision-{args.variant}-F16.gguf",
            variant=args.variant,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
