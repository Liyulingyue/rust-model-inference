"""Convert the GLiNER2.5-Decide safetensors checkpoint to F32 GGUF.

The model is a DeBERTa-v3-large encoder with a 2-layer ReLU MLP on top of the
hidden state at each ``[L]`` marker. Only that path is converted: the NER
heads (``span_rep``, ``count_embed``, ``count_pred``) never run in
``classify_text`` and are dropped.

The GGUF is self-contained: it embeds the checkpoint's ``tokenizer.json`` when
present, or the SentencePiece vocabulary and normalizer from ``spm.model``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.converter.gliner.common import (
    ARCH, TOKENIZER_CLASSES, add_encoder_meta, add_source_sha256,
    add_tokenizer_meta, emit_tensor, encoder_tensor_contracts,
    fast_tokenizer_pieces, parse_spm, resolve_encoder, tensor_chunks, write_atomic,
)
from tools.converter.utils.gguf import GgufWriter, open_safetensors

# Added on top of the 128000 SentencePiece pieces; ids 128000..128010 in
# `tokenizer_config.json`. [MASK] leads because GLiNER2 appends its ten schema
# markers after whatever the base tokenizer already declared.
#
# Pre-2.5 configs (e.g. fastino/gliner2-large-v1) omit `architecture`,
# `config_version`, and `token_pooling`; the field is present only from
# gliner2.5 onwards. Same encoder + head contract, just an older config schema.
EXPECTED_CONFIG_OPTIONAL = {"architecture": "span", "config_version": 3,
                            "token_pooling": "first"}

# Encoder size per `config.json`'s `model_name`. Only the four size fields vary
# across the DeBERTa-v3 checkpoints GLiNER2 ships; the relative-attention
# settings in ENCODER_COMMON are what `AutoConfig.from_pretrained` resolves
# identically for every v3 size.
#
# `position_buckets * 2` is the row count of `rel_embeddings`, and `att_span`
# (the clamp range inside `disentangled_attention_bias`) is the bucket count.
#
# Sizes are read off the published `microsoft/deberta-v3-*` configs, not
# inferred. Three of the four are re-derived from the checkpoint's tensor shapes
# by `encoder_tensor_contracts`, so a wrong entry here fails the conversion;
# `num_attention_heads` appears in no tensor shape, which makes it the one field
# only the byte-exact oracle can catch.
ENCODER_SIZES = {
    "microsoft/deberta-v3-large": {"hidden_size": 1024, "num_hidden_layers": 24,
                                   "num_attention_heads": 16, "intermediate_size": 4096},
    "microsoft/deberta-v3-base": {"hidden_size": 768, "num_hidden_layers": 12,
                                  "num_attention_heads": 12, "intermediate_size": 3072},
    # fastino/gliner2-multi-v1. mDeBERTa-v3 is DeBERTa-v2 with a 250k
    # multilingual SPM vocab; its published config is field-for-field identical
    # to deberta-v3-base apart from `vocab_size` (251000 vs 128100), so every
    # number here is the base row again. The vocab is not in this table: it is
    # derived from the tokenizer below and re-checked against the checkpoint's
    # `word_embeddings` shape.
    "microsoft/mdeberta-v3-base": {"hidden_size": 768, "num_hidden_layers": 12,
                                   "num_attention_heads": 12, "intermediate_size": 3072},
}


def resolve_span_encoder(model_name: str) -> dict:
    return resolve_encoder(model_name, ENCODER_SIZES, "span")


HEADERS = ("piece", "score", "type")
SPECIAL_HEADERS = ("task", "labels", "multi_label", "cls_threshold", "class_act")
SPECIAL_TOKENS = ("[MASK]", "[SEP_STRUCT]", "[SEP_TEXT]", "[P]", "[C]", "[E]", "[R]",
                  "[L]", "[EXAMPLE]", "[OUTPUT]", "[DESCRIPTION]")


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def validate_config(config: dict) -> dict:
    """Check the config against the contract and return the encoder dims.

    `model_type` is the family marker both sizes share; `model_name` selects the
    size and is resolved rather than compared, since more than one is supported.
    """
    if config.get("model_type") != "extractor":
        raise ValueError(f"Unsupported model_type: {config.get('model_type')!r}; expected 'extractor'")
    encoder = resolve_span_encoder(config.get("model_name"))
    for key, value in EXPECTED_CONFIG_OPTIONAL.items():
        if config.get(key) not in (value, None):
            raise ValueError(f"Unsupported config {key}: {config.get(key)!r}; expected {value!r} or missing")
    return encoder


def added_tokens(tokenizer_config: dict, spm_pieces: list[str],
                 fast: dict | None = None) -> dict[str, int]:
    """Tokens appended past the SentencePiece vocab, id-ordered.

    `added_tokens_decoder` also lists tokens that are *already* in the SPM
    vocab, and those are not a mistake: the four base specials always are, and
    mDeBERTa-v3 (`fastino/gliner2-multi-v1`) additionally declares its 100
    `<extra_id_N>` sentinel pieces there. So declarations are split by id — those
    below the SPM length must match the piece they claim, which is a stricter
    check than ignoring them, and only the rest are the appended block the GGUF
    has to carry.
    """
    base = len(spm_pieces)
    decoder = tokenizer_config.get("added_tokens_decoder") or {}
    declared = {entry["content"]: int(index) for index, entry in decoder.items()}
    # The family ships two declaration styles. Older repos carry an
    # `added_tokens_decoder` in `tokenizer_config.json`; the guardrail repos and
    # the whole boundary family carry only `extra_special_tokens` (ten names, no
    # `[MASK]`) and leave the ids to `tokenizer.json`. When both are present they
    # must agree, which is a stronger check than trusting either alone.
    from_fast = {entry["content"]: int(entry["id"])
                 for entry in (fast or {}).get("added_tokens", [])}
    if declared and from_fast:
        disagree = sorted(token for token in set(declared) & set(from_fast)
                          if declared[token] != from_fast[token])
        if disagree:
            raise ValueError(
                f"tokenizer_config and tokenizer.json disagree on {disagree[:3]}"
            )
    if not declared:
        declared = from_fast
    for content, index in (("[PAD]", 0), ("[CLS]", 1), ("[SEP]", 2), ("[UNK]", 3)):
        if declared.get(content) != index or spm_pieces[index] != content:
            raise ValueError(f"base special token {content} does not match the SPM piece at {index}")
    mismatched = [content for content, index in declared.items()
                  if index < base and spm_pieces[index] != content]
    if mismatched:
        raise ValueError(f"added token {mismatched[0]!r} does not match the SPM piece at its id")
    missing = [token for token in SPECIAL_TOKENS if token not in declared]
    if missing:
        raise ValueError(f"tokenizer_config is missing added tokens: {missing}")
    added = {content: index for content, index in declared.items() if index >= base}
    if sorted(added.values()) != list(range(base, base + len(added))):
        raise ValueError(f"added tokens are not a contiguous block at {base}: {added}")
    return added


CLASSIFIER_LAST_LAYER = 2  # ReLU sits at index 1; the output linear at 2.


def tensor_contracts(encoder: dict, vocab_size: int) -> dict[str, tuple]:
    return encoder_tensor_contracts(encoder, vocab_size, CLASSIFIER_LAST_LAYER)


def source_contracts(encoder: dict, vocab_size: int) -> dict[str, str]:
    """Map GGUF tensor name -> safetensors key."""
    mapping = {
        "token_embd.weight": "encoder.embeddings.word_embeddings.weight",
        "tok_norm.weight": "encoder.embeddings.LayerNorm.weight",
        "tok_norm.bias": "encoder.embeddings.LayerNorm.bias",
        "rel_embeddings.weight": "encoder.encoder.rel_embeddings.weight",
        "rel_norm.weight": "encoder.encoder.LayerNorm.weight",
        "rel_norm.bias": "encoder.encoder.LayerNorm.bias",
        "classifier.0.weight": "classifier.0.weight",
        "classifier.0.bias": "classifier.0.bias",
        "classifier.2.weight": "classifier.2.weight",
        "classifier.2.bias": "classifier.2.bias",
    }
    layer = "attention.self.{q}.{k}"
    ff = "attention.output.dense"
    for i in range(encoder["num_hidden_layers"]):
        src = f"encoder.encoder.layer.{i}."
        dst = f"blk.{i}."
        for role, name in (("q", "query_proj"), ("k", "key_proj"), ("v", "value_proj")):
            mapping[dst + f"attn_{role}.weight"] = (src + layer.format(q=name, k="weight"))
            mapping[dst + f"attn_{role}.bias"] = (src + layer.format(q=name, k="bias"))
        mapping[dst + "attn_output.weight"] = src + ff + ".weight"
        mapping[dst + "attn_output.bias"] = src + ff + ".bias"
        mapping[dst + "attn_out_norm.weight"] = src + "attention.output.LayerNorm.weight"
        mapping[dst + "attn_out_norm.bias"] = src + "attention.output.LayerNorm.bias"
        mapping[dst + "ffn_up.weight"] = src + "intermediate.dense.weight"
        mapping[dst + "ffn_up.bias"] = src + "intermediate.dense.bias"
        mapping[dst + "ffn_down.weight"] = src + "output.dense.weight"
        mapping[dst + "ffn_down.bias"] = src + "output.dense.bias"
        mapping[dst + "output_norm.weight"] = src + "output.LayerNorm.weight"
        mapping[dst + "output_norm.bias"] = src + "output.LayerNorm.bias"
    return mapping


# ---------------------------------------------------------------------------

def convert(model_dir: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((model_dir / "config.json").read_text())
    encoder = validate_config(config)
    tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text())
    # fastino ships both the slow `DebertaV2Tokenizer` (gliner2-large-v1,
    # GLiNER2.5-Decide) and the fast `DebertaV2TokenizerFast` (gliner2-base-v1)
    # for what is the same DeBERTa-v2 SPM vocabulary. The class name is only a
    # flavour marker; the properties that decide the pieces and the normalizer
    # are checked on the next line, so accept exactly these two rather than
    # widening to any `Deberta*`.
    if tokenizer_config.get("tokenizer_class") not in TOKENIZER_CLASSES:
        raise ValueError(f"Expected one of {sorted(TOKENIZER_CLASSES)}, "
                         f"got {tokenizer_config.get('tokenizer_class')!r}")
    if tokenizer_config.get("vocab_type") != "spm" or tokenizer_config.get("do_lower_case"):
        raise ValueError("Expected a case-sensitive SentencePiece tokenizer")
    tokenizer_json_path = model_dir / "tokenizer.json"
    fast_json = (
        tokenizer_json_path.read_text()
        if tokenizer_json_path.exists() and not (model_dir / "spm.model").exists()
        else None
    )
    fast = json.loads(fast_json) if fast_json is not None else None
    if fast is not None:
        pieces = fast_tokenizer_pieces(fast, SPECIAL_TOKENS, 0)
        spm = None
    else:
        spm = parse_spm((model_dir / "spm.model").read_bytes())
        if spm["normalizer"].get("name") != "nmt_nfkc":
            raise ValueError(f"Unsupported normalizer {spm['normalizer'].get('name')!r}")
        pieces = spm["pieces"]
    added = added_tokens(tokenizer_config, pieces, fast)
    vocab_size = max(added.values()) + 1
    contracts = tensor_contracts(encoder, vocab_size)
    sources = source_contracts(encoder, vocab_size)

    source = open_safetensors(model_dir / "model.safetensors")
    present = {name for name in source.header if name != "__metadata__"}
    needed = set(sources.values())
    if missing := needed - present:
        raise ValueError(f"checkpoint is missing {sorted(missing)[:4]} ({len(missing)} tensors)")
    dropped = present - needed
    if not dropped <= {name for name in dropped if name.startswith(("span_rep.", "count_embed.", "count_pred."))}:
        raise ValueError(f"refusing to drop unexpected tensors: {sorted(dropped - needed)[:4]}")
    for name, key in sources.items():
        info = source.header[key]
        shape = tuple(info["shape"])
        if shape != contracts[name]:
            raise ValueError(f"Invalid shape for {key}: {shape} != {contracts[name]}")

    tokens = list(pieces) + [token for token, _ in sorted(added.items(), key=lambda kv: kv[1])]
    if len(tokens) != vocab_size:
        raise ValueError(f"vocab mismatch: {len(tokens)} tokens vs {vocab_size} ids")

    writer = GgufWriter(output)
    add_encoder_meta(writer, model_dir.name, encoder, vocab_size)
    # The reference hardcodes `activation="relu"` in
    # `SpanExtractorModel.__init__`'s `create_mlp` call.
    writer.add_meta(f"{ARCH}.classifier.activation", "relu")
    source_keys = ("model_name", "model_type", *EXPECTED_CONFIG_OPTIONAL)
    writer.add_meta(f"{ARCH}.source_architecture", json.dumps({k: config.get(k) for k in source_keys}))
    writer.add_meta(f"{ARCH}.source_config", json.dumps(encoder))
    add_tokenizer_meta(writer, tokens, vocab_size, spm, fast_json, added)
    add_source_sha256(writer, source.path)

    for name, shape in contracts.items():
        key = sources[name]
        info = source.header[key]
        emit_tensor(writer, name, shape,
                    lambda key=key, info=info: tensor_chunks(
                        source.path, source.data_offset, info, key))
    write_atomic(writer, output)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.model_dir, args.output)
