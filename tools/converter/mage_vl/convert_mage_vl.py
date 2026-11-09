"""Export the pinned Mage-VL-Cowork release as explicit mage_vl text + Mage-ViT GGUFs.

python -m tools.converter.mage_vl.convert_mage_vl MODEL_DIR --out-dir OUT [--text-type q8_0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from tools.converter.mage_flow.convert_qwen3vl_vision import Reader
from tools.converter.utils.gguf import GgufWriter, GGML_F32, GGML_BF16, gguf_dims
from tools.converter.utils.quantize_gguf import BLOCK_FORMATS, _quantize_one
from tools.converter.wevisone_vl.convert_wevisone_vl import TEXT_NAMES, _write

REVISION = "287c5fdc8c83e43c4d207d21bc51024a52fdff38"
TEXT_TYPES = ("f32", *BLOCK_FORMATS)
SHARDS = {
    "model-00001-of-00002.safetensors": "86424086d1a35e416ae6475248856b517933aaf87b8e2418b63e74ac0f920241",
    "model-00002-of-00002.safetensors": "3ec6ed805a4664b0e39f57615f02e81989fbe08ae95ac96e3be6af309115e048",
}


def validate_config(config: dict) -> None:
    if config.get("model_type") != "llava_onevision2":
        raise ValueError("expected Mage-VL-Cowork llava_onevision2")
    text = {"model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36,
            "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
            "intermediate_size": 9728, "vocab_size": 151936, "hidden_act": "silu",
            "attention_bias": False, "rope_theta": 5000000, "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262144, "rope_scaling": None,
            "sliding_window": None, "use_sliding_window": False,
            "layer_types": ["full_attention"] * 36}
    vision = {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16,
              "intermediate_size": 4096, "num_channels": 3, "patch_size": 16,
              "spatial_merge_size": 2, "temporal_patch_size": 1, "out_hidden_size": 2560,
              "rope_theta": 10000.0, "hidden_act": "gelu", "layer_norm_type": "layer_norm",
              "layer_norm_eps": 1e-6, "use_head": False}
    for part, expected in [("text_config", text), ("vision_config", vision)]:
        if any(config.get(part, {}).get(k) != v for k, v in expected.items()):
            raise ValueError(f"unsupported Mage-VL-Cowork {part}")
    if config["vision_config"].get("use_patch_position_encoding", False):
        raise ValueError("unsupported Mage-ViT patch position embeddings")
    if any(config.get(k) != v for k,v in {"image_token_id":151655,"vision_start_token_id":151652,"vision_end_token_id":151653}.items()):
        raise ValueError("invalid Mage-VL image token")


def map_text_name(name: str) -> str:
    fixed = {"lm_head.weight": "output.weight", "model.embed_tokens.weight": "token_embd.weight",
             "model.norm.weight": "output_norm.weight"}
    if name in fixed:
        return fixed[name]
    m = re.fullmatch(r"model\.layers\.(\d+)\.(.+)\.weight", name)
    if m and m[2] in TEXT_NAMES:
        return f"blk.{m[1]}.{TEXT_NAMES[m[2]]}.weight"
    raise ValueError(f"unsupported Mage-VL text tensor: {name}")


def tokenizer_metadata(tokenizer: dict) -> dict:
    vocab = {int(i): s for s, i in tokenizer["model"]["vocab"].items()}
    added = tokenizer["added_tokens"]
    vocab.update({int(t["id"]): t["content"] for t in added})
    if set(vocab) != set(range(max(vocab)+1)) or len(vocab) > 151936:
        raise ValueError("invalid Mage-VL tokenizer ID range")
    for i, token in {151643: "<|endoftext|>", 151645: "<|im_end|>", 151655: "<|image_pad|>"}.items():
        if vocab.get(i) != token:
            raise ValueError(f"invalid Mage-VL token {i}")
    tokens = [vocab.get(i, "") for i in range(151936)]
    types = [1 if i in vocab else 5 for i in range(151936)]
    for t in added:
        types[t["id"]] = 3 if t.get("special", False) else 4
    return {"model": "gpt2", "pre": "qwen2", "tokens": tokens, "token_type": types,
            "merges": [" ".join(m) if isinstance(m, list) else m for m in tokenizer["model"]["merges"]],
            "eos_token_id": 151645, "padding_token_id": 151643,
            "add_bos_token": False, "add_eos_token": False}


def encode_text_tensor(reader: Reader, name: str, text_type: str) -> tuple[int, bytes]:
    shape = reader.shape(name)
    if len(shape) == 1 or text_type == "f32":
        return GGML_F32, reader.f32(name).tobytes()
    if reader.dtype(name) != "BF16":
        raise ValueError(f"expected BF16 source: {name}")
    return _quantize_one(name, reader.raw(name), GGML_BF16, gguf_dims(shape),
                         text_type, frozenset())


def export(root: Path, out: Path, text_type: str) -> tuple[Path, Path]:
    if text_type not in TEXT_TYPES:
        raise ValueError(f"unsupported Mage-VL text type {text_type!r}; expected {TEXT_TYPES}")
    config = json.loads((root/"config.json").read_text())
    validate_config(config)
    hashes = {}
    for name, expected in SHARDS.items():
        with (root/name).open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"{name}: SHA256 does not match {REVISION}")
        hashes[name] = actual
    out.mkdir(parents=True, exist_ok=True)
    paths = (out/f"Mage-VL-Cowork-{text_type}.gguf", out/"mmproj-Mage-VL-Cowork-F32.gguf")
    if any(p.exists() for p in paths):
        raise FileExistsError("Mage-VL output already exists")
    reader = Reader(root)
    writers = [GgufWriter(p) for p in paths]
    text, vision = writers
    for writer, arch in zip(writers, ["mage_vl", "clip"]):
        writer.add_meta("general.architecture", arch)
        writer.add_meta("general.name", "Mage-VL-Cowork")
        writer.add_meta("mage_vl.source_revision", REVISION)
        writer.add_meta("mage_vl.source_sha256", json.dumps(hashes, sort_keys=True))
    text.add_meta("general.file_type", text_type)
    for name, value in {"embedding_length": 2560, "block_count": 36, "feed_forward_length": 9728,
                        "context_length": 262144, "attention.head_count": 32, "attention.head_count_kv": 8,
                        "attention.key_length": 128, "attention.value_length": 128,
                        "attention.layer_norm_rms_epsilon": 1e-6, "rope.freq_base": 5000000.0,
                        "rope.dimension_count": 128, "vocab_size": 151936}.items():
        text.add_meta(f"mage_vl.{name}", value)
    for name, value in tokenizer_metadata(json.loads((root/"tokenizer.json").read_text())).items():
        text.add_meta(f"tokenizer.ggml.{name}", value)
    # Python Jinja removes the final source newline; the shared Rust renderer preserves it.
    text.add_meta("tokenizer.chat_template", (root/"chat_template.jinja").read_text().removesuffix("\n"))
    vision.add_meta("clip.projector_type", "mage_vl")
    vision.add_meta("clip.has_vision_encoder", True)
    vision.add_meta("clip.has_audio_encoder", False)
    vision.add_meta("mage_vl.preprocessor", (root/"preprocessor_config.json").read_text())
    counts = [0, 0]
    contract = {}
    for name in reader.owners:
        is_vision = name.startswith("visual.")
        idx = int(is_vision)
        mapped = name if is_vision else map_text_name(name)
        shape = reader.shape(name)
        contract[name] = {"shape": shape, "dtype": reader.dtype(name), "gguf_name": mapped}
        if is_vision:
            dtype, raw = GGML_F32, reader.f32(name).tobytes()
        else:
            dtype, raw = encode_text_tensor(reader, name, text_type)
        writers[idx].add_tensor(mapped, dtype, gguf_dims(shape), raw)
        counts[idx] += 1
    if counts != [399, 297]:
        raise ValueError(f"unexpected Mage-VL component tensor counts: {counts}")
    for writer in writers:
        _write(writer)
    (out/"source-tensors.json").write_text(json.dumps(contract, indent=2))
    return paths


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--text-type", choices=TEXT_TYPES, default="bf16")
    a = p.parse_args()
    for path in export(a.model, a.out_dir, a.text_type):
        print(path, flush=True)
