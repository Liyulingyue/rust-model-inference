"""Measure the official CPU sampler and Rust CLI at the same F16 weights and seed."""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from reference import load_packed


@torch.no_grad()
def measure(args):
    sys.path.insert(0, str((args.assets / "oracle").resolve()))
    from tessera.pipeline import load_model_from_checkpoint
    from tessera.sampler import sample
    from tessera.text_encoder import FrozenTextEncoder
    from tessera.viz import tokens_to_images

    prompt, threads = "golden crown with blue gems", 8
    weight = args.assets / "tessera-small-f16.gguf"
    torch.set_num_threads(threads)
    model, palette, _ = load_model_from_checkpoint(args.assets / "tessera-small-fp16.pt", device="cpu")
    encoder = FrozenTextEncoder(str(args.assets / "clip"), model.cfg.text_len)
    load_packed(weight, model, encoder)
    results = {"prompt": prompt, "seed": 0, "steps": 16, "threads": threads,
               "weight": weight.name, "official_pytorch_cpu_seconds": [], "rust_cli_seconds": {}}
    for _ in range(args.runs):
        start = time.monotonic()
        states, pooled = encoder(encoder.tokenize([prompt]))
        tokens = sample(model, states, pooled, generator=torch.Generator().manual_seed(0))
        tokens_to_images(tokens, palette)
        results["official_pytorch_cpu_seconds"].append(round(time.monotonic() - start, 3))
    del encoder, model
    with tempfile.TemporaryDirectory() as tmp:
        for policy in ["cpu", "auto", "vulkan"]:
            values = []
            for _ in range(args.runs):
                start = time.monotonic()
                subprocess.run([args.bin_dir / "rust-model-inference", "--model", weight,
                                "--prompt", prompt, "--threads", str(threads), "--seed", "0",
                                "--compute", policy, "--out", Path(tmp) / "image.png", "--overwrite"],
                               stdout=sys.stderr, check=True)
                values.append(round(time.monotonic() - start, 3))
            results["rust_cli_seconds"][policy] = values
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("bin_dir", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    print(json.dumps(measure(args), indent=2))
