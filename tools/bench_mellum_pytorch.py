#!/usr/bin/env python3
"""Benchmark Mellum 2.1 PyTorch native inference on GB10.

Reference point for the native Rust implementation
(`src/models/mellum/`), which runs the same Q8_0 GGUF.

The GB10 has unified memory, so a CUDA run needs the bf16 weights resident
in GPU memory *plus* the CPU staging copy that `from_pretrained` uses while
sharding -- roughly 2x the model size in peak RSS. When another process
already holds most of the unified memory (e.g. a local inference server),
the CUDA path cannot fit; `--device cpu` runs the identical graph on the CPU
so the Rust-vs-PyTorch comparison is still apples-to-apples.
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "models/Mellum2.1-12B-A2.5B-Thinking"
PROMPT = "What is 2+2?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    if args.device == "cuda":
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: cpu")
    print(f"Loading model from {args.model_dir}...")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    t_load = time.time() - t0
    print(f"Loaded in {t_load:.1f}s")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    active_params = total_params  # MoE: all params are "active" in PyTorch
    print(f"Total params: {total_params/1e9:.2f}B")

    # Build prompt
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(args.device)
    input_ids = inputs["input_ids"]
    print(f"Prompt: {text[:100]}...")
    print(f"Input tokens: {input_ids.shape[1]}")

    # Prefill
    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    past_kv = outputs.past_key_values
    logits = outputs.logits
    t_prefill = time.time() - t0
    prefill_tok_s = input_ids.shape[1] / t_prefill
    print(f"Prefill: {t_prefill*1000:.0f}ms ({prefill_tok_s:.1f} tok/s)")

    # Decode
    generated = []
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    t0 = time.time()
    for _ in range(args.max_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                past_key_values=past_kv,
                use_cache=True,
            )
        past_kv = outputs.past_key_values
        logits = outputs.logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())
    t_decode = time.time() - t0
    decode_tok_s = len(generated) / t_decode
    print(f"Decode: {t_decode*1000:.0f}ms ({decode_tok_s:.1f} tok/s)")

    # Total
    total_tokens = len(generated)
    total_time = t_decode
    print(f"\nTotal: {total_tokens} tokens in {total_time*1000:.0f}ms ({decode_tok_s:.1f} tok/s)")

    # Decode text
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"Output: {text}")

    # Memory
    if args.device == "cuda":
        print(
            f"\nGPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, "
            f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GB peak"
        )


if __name__ == "__main__":
    main()
