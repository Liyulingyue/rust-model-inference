"""Upstream PyTorch reference for Occamy-1.0.

The GGUF is a conversion of `Accio-Lab/occamy-1.0`; this script runs the
upstream weights through `transformers`' own `Qwen3_5MoeForConditionalGeneration`
so the engine's numbers can be compared against the implementation the model
was trained with, not only against llama.cpp.

Two dumps are produced, both in the same layout the Rust parity trace writes:

  * `text`: the logits of the final prefill row, teacher-forced over the
    reference history, so the comparison runs on identical histories.
  * `vision`: the projected image features of the vision tower, for the same
    solid-blue input the `llama-mtmd-debug` reference uses.

Usage:

    ./.venv/bin/python tools/oracle/pr169/pytorch_reference.py \
        --model models/Occamy-1.0 \
        --prompt 'What is 2+2?' \
        --out target/pr169/pytorch-reference
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


def chat_prompt(processor, question: str) -> str:
    """Render the checkpoint's own chat template for one user turn."""
    messages = [{"role": "user", "content": question}]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def dump_f32(path: Path, values) -> None:
    array = np.asarray(values, dtype="<f4")
    array.tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="directory with the safetensors")
    parser.add_argument("--prompt", default="What is 2+2?")
    parser.add_argument(
        "--input-ids",
        default=None,
        help="comma separated token ids; use this to run the engine's exact prompt",
    )
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda; the CPU path is the authoritative reference",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="torch dtype for the weights (bfloat16 is what the repo ships)",
    )
    parser.add_argument("--image", default=None, help="optional PNG for the vision dump")
    parser.add_argument(
        "--generate",
        type=int,
        default=0,
        help="greedy tokens to generate after the prompt",
    )
    parser.add_argument(
        "--vision-input",
        default="blue",
        help="'blue' for the solid RGB [0,0,1] square the oracle uses, or a PNG path",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=dtype, device_map=None
    ).eval()
    model = model.to(args.device)

    if args.input_ids:
        input_ids = torch.tensor(
            [[int(v) for v in args.input_ids.split(",")]], dtype=torch.long
        ).to(args.device)
        prompt = "<explicit ids>"
    else:
        prompt = chat_prompt(processor, args.prompt)
        tokenizer = processor.tokenizer
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(args.device)
    print(f"prompt tokens: {input_ids.shape[-1]}")
    print(f"prompt ids: {input_ids[0].tolist()}")

    with torch.no_grad():
        output = model(input_ids=input_ids, use_cache=False)
    logits = output.logits[0, -1].to(torch.float32)
    dump_f32(out / "text-logits.f32", logits.cpu().numpy())
    top = torch.topk(logits, 5)
    print("top-5:", list(zip(top.indices.tolist(), [round(v, 4) for v in top.values.tolist()])))

    if args.generate:
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.generate,
            do_sample=False,
            use_cache=True,
        )
        tail = generated[0, input_ids.shape[-1] :].tolist()
        print(f"generated ids: {tail}")
        (out / "generated-ids.json").write_text(json.dumps(tail))

    if args.vision_input != "blue":
        return

    # The oracle's vision reference is a solid RGB [0,0,1] square of already
    # normalized floats, not a preprocessed PNG. Feed the same pixels through
    # the upstream tower so the comparison is on identical input.
    side = 512
    pixels = torch.zeros((1, 3, side, side), dtype=torch.float32)
    pixels[:, 2] = 1.0
    # The tower is a Qwen3-VL style patchified CLIP: it needs the grid, and
    # 512/16 = 32 patches per side with a merge of 2 gives 16x16 = 256 tokens,
    # the same count the engine reports for this input.
    patch = model.config.vision_config.patch_size
    merge = model.config.vision_config.spatial_merge_size
    grid = side // patch
    grid_thw = torch.tensor([[1, grid, grid]], dtype=torch.long)
    print(f"vision grid: {grid_thw.tolist()} -> {grid * grid // (merge * merge)} tokens")
    # The tower takes the patchified sequence, not the image: (num_patches,
    # C * patch * patch) in the Qwen3-VL layout.
    patches = (
        pixels.reshape(1, 3, grid, patch, grid, patch)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(grid * grid, 3 * patch * patch)
    )
    with torch.no_grad():
        features = model.get_image_features(
            pixel_values=patches.to(args.device),
            image_grid_thw=grid_thw.to(args.device),
        )
    if isinstance(features, (list, tuple)):
        features = features[0]
    features = features.to(torch.float32)
    dump_f32(out / "vision-features.f32", features.reshape(-1).cpu().numpy())
    print(f"vision features: {tuple(features.shape)}")


if __name__ == "__main__":
    main()
