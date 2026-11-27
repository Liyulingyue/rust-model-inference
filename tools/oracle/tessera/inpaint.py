"""Compare RGBA inpainting, including color quantization and polish, through the real CLI."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def small_fixture(assets, fmt, output):
    """Punch four holes in the pinned reference sprite and repaint with this GGUF."""
    from reference import load_packed
    sys.path.insert(0, str((assets / "oracle").resolve()))
    from tessera.pipeline import TesseraPipeline, load_model_from_checkpoint
    from tessera.text_encoder import FrozenTextEncoder

    torch.set_num_threads(4)
    model, palette, _ = load_model_from_checkpoint(assets / "tessera-small-fp16.pt", device="cpu")
    encoder = FrozenTextEncoder(str(assets / "clip"), model.cfg.text_len)
    load_packed(assets / f"tessera-small-{fmt}.gguf", model, encoder)
    pipe = TesseraPipeline(model, encoder, palette, "cpu")
    with Image.open(assets / "parity/original/default.png") as sprite:
        source = np.array(sprite.convert("RGBA"))
    mask = np.zeros((16, 16), dtype=bool)
    mask[(5, 7, 8, 13), (3, 8, 7, 14)] = True
    prompt = "golden crown with blue gems"
    reference, = pipe.inpaint(prompt, Image.fromarray(source), mask, num=1, steps=1, cfg_scale=1.,
                              temperature=0., choice_temperature=0.)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "reference.png"
    reference.save(path)
    expected = palette.tokens_to_rgba(palette.quantize_rgba(source))
    assert np.array_equal(np.asarray(reference)[~mask], expected[~mask])
    options = {"choice_temperature": 0, "canvas": [
        None if regenerate else pixel.tolist()
        for regenerate, pixel in zip(mask.flat, source.reshape(-1, 4))]}
    return path, options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("cli", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str((args.assets / "oracle").resolve()))
    from tessera.pipeline import TesseraPipeline, load_model_from_checkpoint
    from tessera.text_encoder import FrozenTextEncoder

    torch.set_num_threads(4)
    model, palette, _ = load_model_from_checkpoint(args.assets / "tessera-small-fp16.pt", device="cpu")
    encoder = FrozenTextEncoder(str(args.assets / "clip"), model.cfg.text_len)
    pipe = TesseraPipeline(model, encoder, palette, "cpu")

    # The frame includes arbitrary sRGB values, alpha on both sides of 128,
    # and an opaque black pixel that must remain distinct from transparency.
    y, x = np.indices((16, 16))
    source = np.full((16, 16, 4), [161, 29, 67, 127], dtype=np.uint8)
    source[1:15, 1:15] = [11, 120, 68, 255]
    source[2:14, 2:14] = [230, 185, 34, 128]
    source[2, 7] = [0, 0, 0, 255]
    mask = (x >= 3) & (x < 13) & (y >= 3) & (y < 13)
    prompt = "an emerald shield with golden trim"
    settings = dict(temperature=0.85, temperature_end=0.4, choice_temperature=1.5,
                    cfg_schedule="cosine", top_k=16, top_p=0.94, min_p=0.01,
                    polish_steps=2, polish_frac=0.06)
    reference, = pipe.inpaint(prompt, Image.fromarray(source), mask, num=1, steps=5,
                              cfg_scale=2.5, seed=73, **settings)
    args.output.mkdir(parents=True, exist_ok=True)
    reference.save(args.output / "inpaint-oracle.png")
    options = dict(settings)
    del options["temperature"]
    options["canvas"] = [None if regenerate else pixel.tolist()
                         for regenerate, pixel in zip(mask.flat, source.reshape(-1, 4))]
    opts = args.output / "inpaint-options.json"
    opts.write_text(json.dumps(options))
    image = args.output / "inpaint-rust.png"
    subprocess.run([str(args.cli.resolve()), "--model", str(args.assets / "tessera-small-f32.gguf"),
                    "--compute", "cpu", "--threads", "4", "--steps", "5", "--seed", "73",
                    "--guidance", "2.5", "--temperature", "0.85", "--input", str(opts),
                    "--prompt", prompt, "--out", str(image), "--overwrite"], check=True)
    with Image.open(image) as result:
        assert result.mode == "RGBA" and result.size == (16, 16)
        actual = np.asarray(result)
    same = np.all(np.asarray(reference) == actual, axis=2)
    kept = np.all(palette.tokens_to_rgba(palette.quantize_rgba(source)) == actual, axis=2)
    report = {"prompt": prompt, "seed": 73, "steps": 5, "guidance": 2.5, **settings,
              "kept": int((~mask).sum()), "kept_match_input_palette": int(kept[~mask].sum()),
              "generated": int(mask.sum()), "generated_match_reference": int(same[mask].sum()),
              "total_match_reference": int(same.sum())}
    (args.output / "inpaint.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if not kept[~mask].all() or not same.all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
