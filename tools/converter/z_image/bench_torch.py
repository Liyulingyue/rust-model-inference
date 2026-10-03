"""What does Z-Image-Turbo actually do on this GB10?

The Rust CPU path takes 140 s per denoise step at 512x512 and 20.5 minutes for
an 8-step render. The GB10 in this box is a Blackwell part with a few hundred
TFLOPS of bf16 tensor core throughput, so almost any of the explanations offered
for that number -- bandwidth, dispatch overhead, the row-at-a-time loop shape --
are worth checking against the one measurement that settles it: run the model the
way it was trained, on the GPU, and read the clock.

This is also the reference for whether the Vulkan work is worth continuing. A
hand-written int8 shader on this device has to compete with cuBLAS, not with the
CPU, and the gap between those two is what the port has to close.

The prompt and seed match the Rust CLI runs, and both 512x512 (what the Rust
path is validated at) and 1024x1024 (what the published timings quote) are
measured, since comparing across resolutions is how a 90x claim gets made.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

PROMPT = "A red fox sleeping beneath a pine tree"


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Z-Image-Turbo directory")
    parser.add_argument(
        "--resolutions",
        default="512,1024",
        help="comma-separated square resolutions to measure",
    )
    parser.add_argument("--steps", type=int, default=9, help="num_inference_steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="/tmp/zimage_torch")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    args = parser.parse_args()

    from diffusers import ZImagePipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device : {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"gpu    : {props.name}  sm_{props.major}{props.minor}")
        print(f"memory : {props.total_memory / 1e9:.1f} GB")
    dtype = getattr(torch, args.dtype)
    print(f"dtype  : {args.dtype}")
    print(f"steps  : {args.steps} (Z-Image Turbo counts one fewer DiT forward)")

    t0 = time.time()
    pipe = ZImagePipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    pipe.to(device)
    sync()
    print(f"load   : {time.time() - t0:.2f} s\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for resolution in (int(r) for r in args.resolutions.split(",")):
        print(f"== {resolution}x{resolution} ==")
        generator = torch.Generator(device).manual_seed(args.seed)

        # One warm pass so the first measurement is not paying for kernel
        # autotuning, cuDNN algorithm selection or the allocator growing.
        warm = pipe(
            prompt=PROMPT,
            height=resolution,
            width=resolution,
            num_inference_steps=2,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        sync()
        del warm
        if device == "cuda":
            torch.cuda.empty_cache()

        generator = torch.Generator(device).manual_seed(args.seed)
        t0 = time.time()
        image = pipe(
            prompt=PROMPT,
            height=resolution,
            width=resolution,
            num_inference_steps=args.steps,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        sync()
        elapsed = time.time() - t0

        path = out_dir / f"torch_{resolution}_{args.steps}step.png"
        image.save(path)
        forwards = args.steps - 1  # the last step is the sigma->0 terminal update
        print(f"  total     : {elapsed:.2f} s")
        print(f"  per DiT fwd: {elapsed / max(forwards, 1) * 1000:.1f} ms  ({forwards} forwards)")
        if device == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  peak vram : {peak:.1f} GB")
        print(f"  saved     : {path}\n")


if __name__ == "__main__":
    main()
