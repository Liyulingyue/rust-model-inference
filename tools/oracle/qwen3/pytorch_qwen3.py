#!/usr/bin/env python3
"""PyTorch reference for the Qwen3 GPU/CPU dispatch work.

Runs greedy generation on the original safetensors weights so the Vulkan and
CPU paths in this repository can be compared against a third implementation
that nobody here wrote. The point is the throughput ceiling of the device:
if the reference is also slower than the CPU path, the dispatch policy has to
send this shape to the CPU no matter how good the kernels get.

    ./.venv/bin/python tools/oracle/qwen3/pytorch_qwen3.py \
        --model models/Qwen3-0.6B --device cuda --max-tokens 64
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="directory with config.json and safetensors")
    parser.add_argument("--prompt", default="用两句话解释什么是扩散模型。")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("cuda is not available")
        return 1

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model = model.to(device).eval()

    prompt_ids = tokenizer(args.prompt, return_tensors="pt").to(device)
    prompt_len = prompt_ids["input_ids"].shape[1]

    with torch.inference_mode():
        started = time.perf_counter()
        output = model.generate(
            **prompt_ids,
            max_new_tokens=args.max_tokens,
            do_sample=False,
            use_cache=True,
        )
        elapsed = time.perf_counter() - started
    generated = output[0, prompt_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    total = generated.shape[0]
    print(f"device={device.type} dtype={args.dtype} prompt_tokens={prompt_len} generated={total}")
    print(f"wall={elapsed * 1e3:.1f}ms prefill={prompt_len / elapsed:.1f} t/s generation={total / elapsed:.1f} t/s")
    print(f"Output: {text}")
    if device.type == "cuda":
        print(f"peak_gpu_MiB={torch.cuda.max_memory_allocated() / 2**20:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
