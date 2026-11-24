"""Run the pinned Tessera Space directly on CPU and export little-endian parity fixtures."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def write_tensor(directory, name, tensor, dtype):
    tensor.detach().cpu().numpy().astype(dtype, copy=False).tofile(directory / name)


def load_packed(path, model, text_encoder):
    # Independent GGUF decoder. Restore the exact exported weights to the unmodified
    # upstream modules, leaving their position/rotary buffers and operations in place.
    from gguf import GGUFReader, dequantize

    denoiser, clip = {}, {}
    for tensor in GGUFReader(path).tensors:
        target = clip if tensor.name.startswith("text_model.") else denoiser
        array = dequantize(tensor.data, tensor.tensor_type).reshape(tuple(reversed(tensor.shape)))
        target[tensor.name.removeprefix("text_model.")] = torch.from_numpy(array.astype(np.float32, copy=True))
    model.load_state_dict(denoiser, strict=True)
    text_encoder.model.load_state_dict(clip, strict=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path, help="contains clip/, oracle/tessera/ and the released .pt")
    parser.add_argument("output", type=Path)
    parser.add_argument("--gguf", type=Path, help="compare the same dequantized GGUF weights")
    parser.add_argument("--default-sample", action="store_true", help="also run the official stochastic 16-step preset")
    parser.add_argument("--prompt", default="golden crown with blue gems")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str((args.assets / "oracle").resolve()))
    from tessera.pipeline import load_model_from_checkpoint
    from tessera.sampler import sample_iter
    from tessera.text_encoder import FrozenTextEncoder
    from tessera.viz import tokens_to_images

    model, palette, _ = load_model_from_checkpoint(args.assets / "tessera-small-fp16.pt", device="cpu")
    encoder = FrozenTextEncoder(str(args.assets / "clip"), model.cfg.text_len)
    if args.gguf is not None:
        load_packed(args.gguf, model, encoder)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    prompts = [args.prompt, "", "  café\t蓝色   crown! 🐈 ", "red " * 48]
    token_ids = encoder.tokenize(prompts)
    (out / "tokenizer.json").write_text(json.dumps(list(zip(prompts, token_ids.tolist())), ensure_ascii=False))
    start = time.perf_counter()
    states, pooled = encoder(token_ids[:1])
    timings = {"clip_seconds": time.perf_counter() - start}
    write_tensor(out, "ids.u32", token_ids[0], "<u4")
    write_tensor(out, "states.f32", states, "<f4")
    write_tensor(out, "pooled.f32", pooled, "<f4")

    step = 0

    def trace(module, inputs, logits):
        nonlocal step
        step += 1
        if step in (1, 2):
            suffix = "" if step == 1 else "_2"
            write_tensor(out, f"logits_cond{suffix}.f32", logits[0], "<f4")
            write_tensor(out, f"logits_uncond{suffix}.f32", logits[1], "<f4")
        if step == 2:
            write_tensor(out, "sc-step-2.f32", inputs[4][0], "<f4")

    hook = model.register_forward_hook(trace)
    start = time.perf_counter()
    for st in sample_iter(model, states, pooled, steps=4, cfg_scale=3., temperature=0., choice_temperature=0.):
        write_tensor(out, f"greedy-step-{st.index + 1}.u16", st.tokens, "<u2")
    timings["greedy_four_steps_with_trace_seconds"] = time.perf_counter() - start
    hook.remove()
    tokens_to_images(st.tokens, palette)[0].save(out / "greedy.png")

    if args.default_sample:
        generator = torch.Generator(device="cpu").manual_seed(0)
        start = time.perf_counter()
        for st in sample_iter(model, states, pooled, generator=generator):
            write_tensor(out, f"default-step-{st.index + 1}.u16", st.tokens, "<u2")
        timings["default_sixteen_steps_seconds"] = time.perf_counter() - start
        tokens_to_images(st.tokens, palette)[0].save(out / "default.png")
    (out / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    print(json.dumps(timings))


if __name__ == "__main__":
    main()
