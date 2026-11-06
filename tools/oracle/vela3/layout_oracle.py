#!/usr/bin/env python3
"""Pinned UltraVBERT preprocessing across aspect ratios, without loading weights."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import GPT2Tokenizer
from transformers.models.idefics3.processing_idefics3 import Idefics3Processor

from vision_oracle import official_classes, dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    _, _, Processor, core = official_classes(root)
    cfg = json.loads((root / "processor_config.json").read_text())
    processor = Idefics3Processor(Processor(**cfg["image_processor"]),
                                GPT2Tokenizer.from_pretrained(root), image_seq_len=cfg["image_seq_len"])
    layouts = []
    for width, height in [(37, 21), (511, 513), (1, 200), (200, 1), (3, 3000), (3000, 3)]:
        y, x = np.indices((height, width))
        image = Image.fromarray(np.stack(((x * 13 + y * 7) % 256, (x * 5 + y * 19) % 256,
                                         (x * 23 + y * 3) % 256), -1).astype(np.uint8))
        filename = f"image_{width}_{height}.png"
        image.save(out / filename)
        for edge in (512, 1024):
            processor.image_processor.size = {"longest_edge": edge}
            processor.image_processor.do_image_splitting = edge > 512
            ids, pixels, _ = core.image_prefix(processor, [image])
            _, _, grid = core.image_cells(processor, ids)
            layouts.append({"image": filename, "edge": edge, "grid": list(grid), "input_ids": ids,
                            "pixels": dump(out, f"pixels_{width}_{height}_{edge}", pixels)})
    (out / "layouts.json").write_text(json.dumps({"source_root": str(root), "layouts": layouts}, indent=2) + "\n")
    print(json.dumps({"layouts": len(layouts), "out": str(out)}))


if __name__ == "__main__":
    main()
