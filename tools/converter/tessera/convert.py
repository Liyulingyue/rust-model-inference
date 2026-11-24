"""Package the released Tessera-small EMA, palette and frozen CLIP text tower in one GGUF."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer

from tools.converter.utils.gguf import GGML_F32, GgufWriter, gguf_dims, model_metadata_value
from tools.converter.utils.quantize_gguf import FILE_TYPES, quantize_tensor

# This adapter targets the named release. Pin the inputs before reading the checkpoints,
# including the tokenizer assets from which transformers constructs its actual backend.
CHECKPOINT_SHA = "772a289de232950bc411bbfeaa4d60c320450b04cf5b7d8280c9c0b36880fec3"
CLIP_SHA = {
    "pytorch_model.bin": "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f",
    "config.json": "b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770",
    "tokenizer.json": "b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842",
    "tokenizer_config.json": "34b7336e4bee12e0a9730eaf5189f582ef3c3eea5027f65730e5717256755aad",
    "special_tokens_map.json": "f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf",
    "vocab.json": "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363",
    "merges.txt": "f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85",
}


def verify(path: Path, expected: str) -> None:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"{path} does not match the Tessera-small release (SHA256 {digest})")


def convert(checkpoint: Path, clip: Path, output: Path, fmt: str) -> None:
    verify(checkpoint, CHECKPOINT_SHA)
    for name, digest in CLIP_SHA.items():
        verify(clip / name, digest)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    clip_weights = torch.load(clip / "pytorch_model.bin", map_location="cpu", weights_only=True)
    tensors = dict(ckpt["ema"])
    tensors.update({name: weight for name, weight in clip_weights.items()
                    if name.startswith("text_model.") and weight.is_floating_point()})
    tokenizer = AutoTokenizer.from_pretrained(clip, local_files_only=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        path = Path(tmp) / output.name
        writer = GgufWriter(path)
        writer.add_meta("general.architecture", "tessera")
        writer.add_meta("general.name", "Tessera-small")
        writer.add_meta("general.file_type", model_metadata_value(FILE_TYPES[fmt]))
        writer.add_meta("general.license", "cc-by-nc-4.0")
        writer.add_meta("general.author", "Jagrat Patel (TheJagStudio)")
        writer.add_meta("tessera.model_config", json.dumps(ckpt["model_config"], sort_keys=True))
        writer.add_meta("tessera.text_encoder", ckpt["text_encoder"])
        writer.add_meta("tessera.tokenizer_json", tokenizer.backend_tokenizer.to_str())
        for name, weight in tensors.items():
            dims = gguf_dims(tuple(weight.shape))
            data = weight.float().contiguous().numpy().astype("<f4", copy=False).tobytes()
            ggml_type = GGML_F32
            # The fixed color features, embeddings used by the color MLP, convolution
            # kernels, modulation offsets and normalization vectors stay dense F32.
            # The 40-wide color projection also stays dense: the native F16
            # Vulkan dispatch requires widths >= 32 to be a multiple of 32.
            dense = name in ("tok_emb.weight", "color_mlp.0.weight")
            if fmt != "f32" and weight.ndim == 2 and name.endswith(".weight") and not dense:
                ggml_type, data = quantize_tensor(name, data, GGML_F32, dims, fmt)
            writer.add_tensor(name, ggml_type, dims, data)
        writer.write()
        path.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="released tessera-small-fp16.pt")
    parser.add_argument("clip", type=Path, help="local openai/clip-vit-base-patch32 directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--format", choices=tuple(FILE_TYPES), default="f16")
    args = parser.parse_args()
    convert(args.checkpoint, args.clip, args.output, args.format)


if __name__ == "__main__":
    main()
