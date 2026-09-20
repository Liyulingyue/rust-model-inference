#!/usr/bin/env python3
"""Export text-only NeoHorse with the pinned llama.cpp converter and NFC metadata."""

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys

LLAMA_PIN = "b96806d96061049a5b574269b049bf6241d63d46"


def validate_source(model: Path) -> None:
    config = json.loads((model / "config.json").read_text())
    if config.get("architectures") != ["Qwen3_5ForCausalLM"]:
        raise ValueError("expected text-only Qwen3_5ForCausalLM")
    tokenizer = json.loads((model / "tokenizer.json").read_text())
    if tokenizer.get("normalizer") != {"type": "NFC"}:
        raise ValueError("expected NeoHorse NFC tokenizer")
    names = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    if any(name.startswith(("mtp.", "model.mtp.")) for name in names):
        raise ValueError("this exporter expects NeoHorse without MTP weights")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--outfile", type=Path, required=True)
    args = parser.parse_args()
    validate_source(args.model)
    if args.outfile.exists():
        parser.error(f"output already exists: {args.outfile}")
    commit = subprocess.check_output(
        ["git", "-C", str(args.llama_cpp), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != LLAMA_PIN:
        parser.error(f"llama.cpp must be {LLAMA_PIN}, got {commit}")
    sys.path[:0] = [str(args.llama_cpp), str(args.llama_cpp / "gguf-py")]
    import gguf
    from conversion.qwen import Qwen3_5TextModel

    class NeoHorseModel(Qwen3_5TextModel):
        model_arch = gguf.MODEL_ARCH.QWEN35
        no_mtp = True

        def set_vocab(self):
            super().set_vocab()
            self.gguf_writer.add_bool("tokenizer.ggml.normalizer.nfc", True)

    logging.basicConfig(level=logging.INFO)
    model = NeoHorseModel(args.model, gguf.LlamaFileType.MOSTLY_BF16, args.outfile)
    model.write()


if __name__ == "__main__":
    main()
