"""Generate CPU F32 fixtures using the pinned, unmodified Tencent decoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from tools.converter.wevisone_vl.convert_wevisone_vl import ORACLE_COMMIT, export_decoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    commit = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if commit != ORACLE_COMMIT or subprocess.check_output(["git", "-C", str(args.source), "status", "--porcelain"]):
        raise RuntimeError("Oracle must be the pinned, clean Tencent checkout")
    sys.path.insert(0, str(args.source.resolve()))
    from wevisone_vl.config import Qwen3VLSegConfig
    from wevisone_vl.mask_decoder import WeVisOneMaskDecoder
    from wevisone_vl.feature_bridge import premerge_tokens_to_2d
    from wevisone_vl.prompts import build_prompt
    args.out.mkdir(parents=True, exist_ok=True)
    sidecar = args.model / "mask_decoder.pt"
    data = torch.load(sidecar, map_location="cpu", weights_only=True)
    decoder = WeVisOneMaskDecoder([1024] * 4, 2560, Qwen3VLSegConfig.from_dict(data["cfg"])).eval()
    decoder.load_state_dict(data["state_dict"], strict=True)
    torch.set_num_threads(2)
    torch.manual_seed(71)
    fixtures = {}

    def save(name, value):
        value = value.detach().float().cpu().contiguous()
        fixtures[name] = list(value.shape)
        (args.out / f"{name}.f32").write_bytes(value.numpy().tobytes())

    image = torch.rand(1, 3, 32, 32)
    features = [torch.randn(1, 1024, 4, 6) * 0.25 for _ in range(4)]
    hidden = torch.randn(2, 2560)
    boxes = torch.tensor([[0.1, 0.2, 0.8, 0.9], [0.2, 0.1, 0.7, 0.6]])
    with torch.inference_mode():
        save("image", image)
        for i, f in enumerate(features):
            save(f"feature{i}", f)
        save("hidden", hidden)
        save("boxes", boxes)
        levels, memory, shapes, starts = decoder.visual_encoder(features)
        for i, f in enumerate(levels):
            save(f"encoded{i}", f)
        content, position = decoder.query_builder(hidden, boxes)
        queries = decoder.transformer(content[None], position[None], boxes, memory, shapes, starts)[0]
        save("queries", queries)
        save("pixels", decoder.pixel_fusion(image, levels))
        save("mask_logits", decoder(image, features, hidden, boxes, (4, 6))["mask_logits"])
        tokens = torch.arange(24 * 3).reshape(24, 3).float()
        save("premerge", premerge_tokens_to_2d(tokens, (4, 6), 2))
    export_decoder(sidecar, args.out / "decoder.gguf")
    (args.out / "fixture.json").write_text(json.dumps({"oracle_commit": commit, "torch": torch.__version__, "shapes": fixtures}, indent=2))
    tasks = ["detection", "referring", "point", "count_box", "count_point", "segmentation", "matting", "ocr", "ocr_grounding", "freeform"]
    (args.out / "prompts.json").write_text(json.dumps({task: build_prompt(task, "cat") for task in tasks}, indent=2))


if __name__ == "__main__":
    main()
