"""Convert `fastino/GLiNER2.5-Decide-1B` to F32 GGUF.

This checkpoint is the odd one out in the family: every other variant fine-tunes
a **DeBERTa-v2** encoder, and this one fine-tunes **ModernBERT**
(`jhu-clsp/ettin-enc-from-dec-1b`, `architectures: ModernBertForMaskedLM`).
Nothing in `convert_gliner.py` applies — the two encoders share only the task
head. The differences that matter, all read off the published encoder config
rather than inferred:

* **LayerNorm, not RMSNorm**, with `eps = norm_eps = 1e-5` and `bias = False`.
  `src/ops/norm.rs`'s `rms_norm` is the wrong operator.
* **Hybrid attention.** `layer_id % global_attn_every_n_layers != 0` is a
  *local* layer with a 128-wide sliding window; every third layer (0, 3, 6, …)
  is global. All 28 layers' tensors are named identically, so the schedule is
  the only thing distinguishing them and it has to be written into the GGUF.
* **Interleaved QKV.** `Wqkv` is `[3 * hidden, hidden]` but is read as
  `view(bs, seq, 3, heads, head_dim)`, so query/key/value interleave along the
  middle axis rather than sitting in three contiguous blocks.
* **GLU, not SwiGLU.** `mlp.Wi` is `[2 * intermediate, hidden]` and the halves
  are `chunk(2)`-ed as `act(first) * second` with `hidden_activation = "gelu"`.
  `src/ops/activation`'s `silu_mul_inplace` is the wrong gate.
* **No bias anywhere** in the encoder (`attention_bias = mlp_bias = norm_bias =
  False`), which is why the shapes are single matrices.
* **No position embeddings** (`position_embedding_type = "sans_pos"`); position
  enters only through RoPE with `theta = 160000.0`, applied to Q and K in
  NeoX style (rotate-half, not interleaved pairs).
* `layer 0` has `attn_norm = Identity()`, so its block skips that LayerNorm.
* `head_dim = hidden / heads = 1792 / 28 = 64`.

The task head is the same contract as 2.5-Decide: `classifier.0`
`[2 * hidden, hidden]` + ReLU + `classifier.2` `[1, 2 * hidden]`. `span_rep`,
`count_embed` and `count_pred` are dropped, as in every other span checkpoint.

The tokenizer is a ByteLevel BPE (`tokenizer.json`, `model.type = "BPE"`,
50280 pieces, 50009 merges) with an NFC normalizer, and it is embedded whole as
`gliner2.tokenizer_json` — the same `hf-json` route the DeBERTa checkpoints use
when they ship no `spm.model`. Its 126 added tokens include business vocabulary
(`|||IP_ADDRESS|||` is id 0), and the eleven schema specials sit at
50368..50377, i.e. they are *not* past the end of the vocabulary the way they are
for every SentencePiece variant in this family. That is why
`spm.piece_count` cannot be the anchor here and the ids are read from
`tokenizer.json` directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools.converter.utils.gguf import GgufWriter, GGML_F32, gguf_dims, open_safetensors

ARCH = "gliner2"
VARIANT = "gliner2-decide-ettin-1b"

# `jhu-clsp/ettin-enc-from-dec-1b`, as `AutoConfig.from_pretrained` resolves it.
# These are ModernBERT's own fields, not DeBERTa's: there is no relative
# attention, no position bucket and no disentangled-attention scale divisor, so
# none of `convert_gliner.py`'s ENCODER tables apply.
ENCODER = {
    "model_name": "jhu-clsp/ettin-enc-from-dec-1b",
    "architecture": "modernbert",
    "hidden_size": 1792,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "head_dim": 64,
    "intermediate_size": 3840,
    "hidden_activation": "gelu",
    "norm_eps": 1e-5,
    "norm_bias": False,
    "attention_bias": False,
    "mlp_bias": False,
    # Hybrid attention schedule: `layer_id % 3 != 0` is local with a
    # 128-wide window, every third layer is global.
    "local_attention": 128,
    "global_attn_every_n_layers": 3,
    "rope_theta": 160000.0,
    "max_position_embeddings": 7999,
    "position_embedding_type": "sans_pos",
}

# Only the two gliner-level fields the reference reads, so a checkpoint that
# stops declaring them is rejected rather than silently converted.
EXPECTED_CONFIG = {"architecture": "span", "config_version": 3, "token_pooling": "first"}

# `modeling_modernbert.py:542` sets `attn_norm = nn.Identity()` at layer 0, so the
# checkpoint stores no such tensor. The loader reads this list rather than
# inferring it, so the schedule is visible in the GGUF.
IDENTITY_ATTN_NORM_LAYERS = frozenset({0})

# The eleven appended schema markers, in the order `SPECIAL_TOKENS` uses
# everywhere else in this family. Their ids are *not* contiguous with the end of
# the vocabulary on this checkpoint: the tokenizer carries 98 non-schema added
# tokens, and the specials start after those.
SCHEMA_SPECIALS = [
    "[SEP_STRUCT]", "[SEP_TEXT]", "[P]", "[C]", "[E]", "[R]", "[L]",
    "[EXAMPLE]", "[OUTPUT]", "[DESCRIPTION]",
]

# The task head, which is the Decide contract: `create_mlp(hidden, [hidden*2], 1)`.
def head_shapes(hidden: int) -> dict[str, tuple]:
    wide = hidden * 2
    return {
        "token_embd.weight": None,  # filled in with the real vocab
        "classifier.0.weight": (wide, hidden),
        "classifier.0.bias": (wide,),
        "classifier.2.weight": (1, wide),
        "classifier.2.bias": (1,),
    }


def source_contracts() -> dict[str, str]:
    """GGUF tensor name -> safetensors key."""
    mapping = {
        "token_embd.weight": "encoder.embeddings.tok_embeddings.weight",
        "embd_norm.weight": "encoder.embeddings.norm.weight",
        "final_norm.weight": "encoder.final_norm.weight",
        "classifier.0.weight": "classifier.0.weight",
        "classifier.0.bias": "classifier.0.bias",
        "classifier.2.weight": "classifier.2.weight",
        "classifier.2.bias": "classifier.2.bias",
    }
    for i in range(ENCODER["num_hidden_layers"]):
        src = f"encoder.layers.{i}."
        dst = f"blk.{i}."
        mapping[dst + "attn_qkv.weight"] = src + "attn.Wqkv.weight"
        mapping[dst + "attn_out.weight"] = src + "attn.Wo.weight"
        # Layer 0's `attn_norm` is `nn.Identity()`, so the checkpoint has no
        # tensor for it. It is absent from the mapping rather than pointed at a
        # placeholder, and `identity_attn_norm_layers` tells the loader to skip
        # the normalization — matching `modeling_modernbert.py:542`.
        if i not in IDENTITY_ATTN_NORM_LAYERS:
            mapping[dst + "attn_norm.weight"] = src + "attn_norm.weight"
        mapping[dst + "mlp_in.weight"] = src + "mlp.Wi.weight"
        mapping[dst + "mlp_out.weight"] = src + "mlp.Wo.weight"
        mapping[dst + "mlp_norm.weight"] = src + "mlp_norm.weight"
    return mapping


def tensor_contracts(vocab_size: int) -> dict[str, tuple]:
    d = ENCODER["hidden_size"]
    wide = d * 2
    shapes = {
        "token_embd.weight": (vocab_size, d),
        "embd_norm.weight": (d,),
        "final_norm.weight": (d,),
        "classifier.0.weight": (wide, d),
        "classifier.0.bias": (wide,),
        "classifier.2.weight": (1, wide),
        "classifier.2.bias": (1,),
    }
    f = ENCODER["intermediate_size"]
    for i in range(ENCODER["num_hidden_layers"]):
        if i not in IDENTITY_ATTN_NORM_LAYERS:
            shapes[f"blk.{i}.attn_norm.weight"] = (d,)
        p = f"blk.{i}."
        shapes.update({
            p + "attn_qkv.weight": (3 * d, d),
            p + "attn_out.weight": (d, d),
            p + "mlp_in.weight": (2 * f, d),
            p + "mlp_out.weight": (d, f),
            p + "mlp_norm.weight": (d,),
        })
    return shapes


def read_tokenizer(model_dir: Path) -> dict:
    """The checkpoint's `tokenizer.json`, plus the schema specials' real ids.

    Unlike every SentencePiece variant in this family, the appended specials are
    not a block past the end of the vocabulary, so their ids are read from
    `added_tokens` and cross-checked against the embedding table's row count
    rather than assumed to be contiguous.
    """
    raw = (model_dir / "tokenizer.json").read_text()
    fast = json.loads(raw)
    model = fast.get("model", {})
    if model.get("type") != "BPE":
        raise ValueError(f"expected a BPE tokenizer.json, got {model.get('type')!r}")
    if not model.get("vocab") or not model.get("merges"):
        raise ValueError("tokenizer.json is missing its vocabulary or merges")
    pre = fast.get("pre_tokenizer", {})
    if pre.get("type") != "ByteLevel":
        raise ValueError(f"expected a ByteLevel pre-tokenizer, got {pre.get('type')!r}")
    if fast.get("normalizer", {}).get("type") != "NFC":
        raise ValueError(
            f"expected an NFC normalizer, got {fast.get('normalizer', {}).get('type')!r}"
        )
    declared = {entry["content"]: int(entry["id"]) for entry in fast.get("added_tokens", [])}
    missing = [token for token in SCHEMA_SPECIALS if token not in declared]
    if missing:
        raise ValueError(f"tokenizer.json is missing schema specials: {missing}")
    specials = {token: declared[token] for token in SCHEMA_SPECIALS}
    ordered = sorted(specials.values())
    if ordered != list(range(ordered[0], ordered[0] + len(ordered))):
        raise ValueError(f"schema specials are not contiguous: {specials}")
    # The embedding table is longer than `model.vocab`: 28 of the added tokens
    # sit inside the BPE vocabulary and 98 extend past it, so the row count is
    # `max(len(vocab), highest added id + 1)` rather than either alone. Taking
    # `len(vocab)` alone is what the shape contract just caught.
    highest = max(declared.values())
    vocab_size = max(len(model["vocab"]), highest + 1)
    return {"json": raw, "vocab_size": vocab_size, "specials": specials,
            "added": declared, "vocab_len": len(model["vocab"]), "highest": highest}


def validate_config(config: dict) -> None:
    if config.get("model_name") != ENCODER["model_name"]:
        raise ValueError(
            f"this converter handles {ENCODER['model_name']!r}; got "
            f"{config.get('model_name')!r}"
        )
    for key, value in EXPECTED_CONFIG.items():
        if config.get(key) != value:
            raise ValueError(
                f"unsupported config {key}: {config.get(key)!r}; expected {value!r}"
            )


def convert(model_dir: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((model_dir / "config.json").read_text())
    validate_config(config)

    tokenizer = read_tokenizer(model_dir)
    vocab_size = tokenizer["vocab_size"]

    contracts = tensor_contracts(vocab_size)
    sources = source_contracts()

    source = open_safetensors(model_dir / "model.safetensors")
    present = {name for name in source.header if name != "__metadata__"}
    needed = set(sources.values())
    if missing := needed - present:
        raise ValueError(f"checkpoint is missing {sorted(missing)[:4]} ({len(missing)} tensors)")
    dropped = present - needed
    # Same contract as the DeBERTa converter: only the training-time heads may
    # go, and a typo in a tensor name must not be silently accepted as one.
    stray = sorted(name for name in dropped if not name.startswith(
        ("span_rep.", "count_embed.", "count_pred.")))
    if stray:
        raise ValueError(f"refusing to drop unexpected tensors: {stray[:4]}")

    for name, key in sources.items():
        shape = tuple(source.header[key]["shape"])
        if shape != contracts[name]:
            raise ValueError(f"invalid shape for {key}: {shape} != {contracts[name]}")

    writer = GgufWriter(output)
    writer.add_meta("general.architecture", ARCH)
    writer.add_meta("general.name", model_dir.name)
    writer.add_meta("general.file_type", "F32")

    writer.add_meta(f"{ARCH}.variant", VARIANT)
    writer.add_meta(f"{ARCH}.encoder_architecture", ENCODER["architecture"])
    writer.add_meta(f"{ARCH}.block_count", ENCODER["num_hidden_layers"])
    writer.add_meta(f"{ARCH}.embedding_length", ENCODER["hidden_size"])
    writer.add_meta(f"{ARCH}.feed_forward_length", ENCODER["intermediate_size"])
    writer.add_meta(f"{ARCH}.attention.head_count", ENCODER["num_attention_heads"])
    writer.add_meta(f"{ARCH}.attention.head_dim", ENCODER["head_dim"])
    # No `scale_divisor`: ModernBERT uses plain 1/sqrt(head_dim) SDPA scaling,
    # not DeBERTa's 1 + |pos_att_type|.
    writer.add_meta(f"{ARCH}.attention.norm_epsilon", ENCODER["norm_eps"])
    writer.add_meta(f"{ARCH}.norm_bias", ENCODER["norm_bias"])
    writer.add_meta(f"{ARCH}.attention.bias", ENCODER["attention_bias"])
    writer.add_meta(f"{ARCH}.mlp.bias", ENCODER["mlp_bias"])
    # The hybrid schedule. Written as two numbers rather than a per-layer flag
    # list so the loader derives it the same way `modeling_modernbert.py:484`
    # does: `layer_id % global_attn_every_n_layers != 0` means local.
    writer.add_meta(f"{ARCH}.attention.local_window", ENCODER["local_attention"])
    writer.add_meta(f"{ARCH}.attention.global_every_n_layers",
                    ENCODER["global_attn_every_n_layers"])
    writer.add_meta(f"{ARCH}.rope.theta", ENCODER["rope_theta"])
    writer.add_meta(f"{ARCH}.rope.max_position_embeddings", ENCODER["max_position_embeddings"])
    writer.add_meta(f"{ARCH}.position_embedding_type", ENCODER["position_embedding_type"])
    # Which layers skip `attn_norm` (ModernBERT sets it to Identity at layer 0).
    writer.add_meta(f"{ARCH}.identity_attn_norm_layers",
                    sorted(IDENTITY_ATTN_NORM_LAYERS))
    writer.add_meta(f"{ARCH}.hidden_act", ENCODER["hidden_activation"])
    writer.add_meta(f"{ARCH}.vocab_size", vocab_size)
    writer.add_meta(f"{ARCH}.classifier.intermediate_size", ENCODER["hidden_size"] * 2)
    writer.add_meta(f"{ARCH}.classifier.activation", "relu")
    writer.add_meta(f"{ARCH}.source_architecture", json.dumps(
        {k: config.get(k) for k in ("model_name", "model_type", *EXPECTED_CONFIG)}))
    writer.add_meta(f"{ARCH}.source_config", json.dumps(ENCODER))
    writer.add_meta(f"{ARCH}.special_tokens", SCHEMA_SPECIALS)
    writer.add_meta(f"{ARCH}.special_token_ids",
                    [tokenizer["specials"][t] for t in SCHEMA_SPECIALS])

    # ByteLevel BPE travels as the whole tokenizer.json, the same `hf-json` route
    # the DeBERTa checkpoints take when they ship no `spm.model`.
    writer.add_meta("tokenizer.ggml.model", "hf-json")
    writer.add_meta("tokenizer.ggml.vocab_size", vocab_size)
    # The BPE vocabulary is shorter than the table; say so rather than leaving
    # the loader to infer a row count that would mis-tokenise the tail.
    writer.add_meta("tokenizer.ggml.bpe_vocab_size", tokenizer["vocab_len"])
    writer.add_meta("tokenizer.ggml.pre", "llama-bpe")
    raw = tokenizer["json"]
    writer.add_meta(f"{ARCH}.tokenizer_json", raw)
    writer.add_meta(f"{ARCH}.tokenizer_sha256", hashlib.sha256(raw.encode()).hexdigest())

    with source.path.open("rb") as stream:
        writer.add_meta(f"{ARCH}.source_sha256", hashlib.file_digest(stream, "sha256").hexdigest())

    def chunks(key: str):
        info = source.header[key]
        start, end = info["data_offsets"]
        dtype = "<f2" if info["dtype"] == "F16" else "<f4"
        with source.path.open("rb") as stream:
            stream.seek(source.data_offset + start)
            remaining = end - start
            while remaining:
                raw = stream.read(min(remaining, 8 * 1024 * 1024))
                if not raw:
                    raise ValueError(f"truncated tensor {key}")
                values = np.frombuffer(raw, dtype=dtype).astype("<f4")
                if not np.isfinite(values).all():
                    raise ValueError(f"non-finite tensor {key}")
                yield values.tobytes()
                remaining -= len(raw)

    for name, key in sources.items():
        writer.add_tensor_chunks(name, GGML_F32, gguf_dims(contracts[name]),
                                 math.prod(contracts[name]) * 4,
                                 lambda key=key: chunks(key))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    writer.path = temporary
    writer.write()
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.model_dir, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
