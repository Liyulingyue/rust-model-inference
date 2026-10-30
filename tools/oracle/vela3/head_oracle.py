#!/usr/bin/env python3
"""Compare each d3 Vela package's real SpanHeadV2 without loading its large base."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    fixtures = []
    for root in args.models:
        root = root.resolve()
        source = root / "vela3_span.py"
        spec = importlib.util.spec_from_file_location("vela_head_" + root.name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        config = json.loads((root / "vela3_config.json").read_text())
        width = json.loads((root / "config.json").read_text())["text_config"]["hidden_size"]
        words = torch.sin(torch.arange(width * 3).float() * .017).reshape(3, width)
        labels = torch.cos(torch.arange(width * 2).float() * .031).reshape(2, width)
        for route in ("span", "ner", "relation", "image_halu"):
            item = config["span"] if route == "span" else config["span"].get(route)
            if not item:
                continue
            file = root / item["head_file"]
            head = module.SpanHeadV2(width).eval()
            head.load_state_dict(load_file(file), strict=True)
            logits = head(words, labels)[0]
            fixtures.append(dict(model=str(root), route=route, file=str(file), width=width,
                words=words.flatten().tolist(), labels=labels.flatten().tolist(), logits=logits.flatten().tolist(),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                weights_sha256=hashlib.file_digest(file.open("rb"), "sha256").hexdigest()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(torch=torch.__version__, fixtures=fixtures), allow_nan=False))
    print(json.dumps(dict(fixture=str(args.out), heads=len(fixtures))))


if __name__ == "__main__":
    main()
