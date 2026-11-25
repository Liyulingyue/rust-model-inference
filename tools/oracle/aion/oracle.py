"""Dump Aion eager PyTorch logits for fixed prefill and teacher-forced decode."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tools.converter.aion.convert import tensor_map


def round_projection_input(input, bits):
    width = 256 if bits == "q8_k" else 32
    values = input.reshape(*input.shape[:-1], -1, width)
    if bits == "q8_k":
        i = values.abs().argmax(dim=-1, keepdim=True)
        signed_max = values.gather(-1, i)
        divisor = torch.where(signed_max == 0, torch.ones_like(signed_max), signed_max)
        # Tensor/tensor division preserves the single F32 division before
        # rounding codes. Scalar/tensor uses reciprocal then multiply.
        inverse_scale = torch.full_like(divisor, -127.0) / divisor
        codes = torch.round(values * inverse_scale).clamp(-127, 127)
        rounded = codes * (1.0 / inverse_scale)
        correction = None
    else:
        scale = values.abs().amax(dim=-1, keepdim=True) / 127.0
        divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
        codes = torch.round(values * (1.0 / divisor)).clamp(-127, 127)
        rounded = codes * scale.to(torch.float16).to(torch.float32)
        # Q4_1's offset uses a separately rounded activation-block sum.
        sums = (codes.sum(dim=-1) * scale.squeeze(-1)).to(torch.float16).float()
        correction = sums - rounded.sum(dim=-1)
    return rounded.reshape_as(input), correction


def projection_quantization(tensor):
    from gguf import GGMLQuantizationType

    quant = tensor.tensor_type
    if quant in {GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K, GGMLQuantizationType.Q6_K}:
        return "q8_k"
    if quant in {GGMLQuantizationType.Q8_0, GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q4_1}:
        return "q8_0"
    return None


def projection_minima(tensor, data):
    from gguf import GGMLQuantizationType

    if tensor.tensor_type != GGMLQuantizationType.Q4_1:
        return None
    blocks = data.reshape(data.shape[0], -1, 20)
    values = blocks[:, :, 2:4].copy().view("<f2").reshape(data.shape[0], -1)
    return torch.from_numpy(values.astype("<f4"))


def quantized_linear(module, bits, minima):
    original = module.forward

    def forward(input):
        rounded, correction = round_projection_input(input, bits)
        result = original(rounded)
        if minima is not None:
            result = result + torch.nn.functional.linear(correction, minima)
        return result

    module.forward = forward


@torch.no_grad()
def load_weights(model, path, round_activations):
    from gguf import GGUFReader, dequantize

    reader = GGUFReader(path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    parameters = dict(model.named_parameters())
    for name, (gguf_name, shape) in tensor_map(model.config.to_dict()).items():
        tensor = tensors[gguf_name]
        # gguf-py independently decodes each format, including packed K-quants.
        # Tile the vocabulary matrix to bound the temporary F32 allocation.
        weight = parameters[name]
        if len(shape) == 1:
            weight.copy_(torch.from_numpy(dequantize(tensor.data, tensor.tensor_type).copy()))
            continue
        for start in range(0, shape[0], 1024):
            end = min(start + 1024, shape[0])
            data = dequantize(tensor.data[start:end], tensor.tensor_type).reshape(end - start, shape[1])
            weight[start:end].copy_(torch.from_numpy(data.copy()))
        if not round_activations:
            continue
        mode = projection_quantization(tensor)
        if mode is None:
            continue
        minima = projection_minima(tensor, tensor.data)
        module = model.lm_head if name == "model.embed_tokens.weight" else model.get_submodule(name.removesuffix(".weight"))
        quantized_linear(module, mode, minima)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gguf", type=Path, help="install the independently dequantized GGUF weights")
    parser.add_argument("--round-activations", action="store_true")
    parser.add_argument("--date", default="2026-11-26")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--follow", type=Path, help="teacher-force another run's generated token IDs")
    parser.add_argument("--long-context", action="store_true")
    parser.add_argument("--case", choices=("arithmetic", "unicode", "tool_history"))
    args = parser.parse_args()
    if args.round_activations and args.gguf is None:
        parser.error("--round-activations requires --gguf")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True,
        dtype=torch.float32, attn_implementation="eager",
    ).eval()
    if args.gguf:
        load_weights(model, args.gguf, args.round_activations)
    cases = [
        ("arithmetic", [{"role": "user", "content": "Answer with just the number. What is 84 * 3 / 2?"}], []),
        ("unicode", [{"role": "system", "content": "You are a translator."},
                     {"role": "user", "content": "Translate to English:  你好，café 👋\n\t123456 \r\n"}], []),
        ("tool_history", [
            {"role": "user", "content": "What is the weather in Hangzhou?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "weather", "arguments": {"city": "杭州"}}}]},
            {"role": "tool", "content": "Sunny, 18°C."},
        ], [{"type": "function", "function": {"name": "weather", "description": "Look up weather.",
                                               "parameters": {"type": "object", "properties": {
                                                   "city": {"type": "string"}}, "required": ["city"]}}}]),
    ]
    if args.long_context:
        cases = [("sliding_boundary",
                  [{"role": "user", "content": " x" * 2050 + "\nReply with OK."}], [])]
    elif args.case:
        cases = [case for case in cases if case[0] == args.case]
    follow = {}
    if args.follow:
        follow = {c["name"]: c["greedy"] for c in json.loads(args.follow.read_text())["cases"]}

    results = []
    for name, turns, tools in cases:
        text = tokenizer.apply_chat_template(turns, tools=tools, current_date=args.date,
                                             add_generation_prompt=True, tokenize=False)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_ids = torch.tensor([ids])
        greedy, forced, checkpoints = [], [], []
        cache = None
        start = time.perf_counter()
        timings = []
        with (args.output / f"{name}.f32").open("wb") as logits_file:
            for step in range(args.steps):
                result = model(input_ids=token_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = result.past_key_values
                logits = result.logits[0, -1]
                logits.numpy().astype("<f4", copy=False).tofile(logits_file)
                checkpoints.append(float(logits.abs().max()))
                token = int(logits.argmax())
                greedy.append(token)
                forced.append(follow[name][step] if name in follow else token)
                token_ids = torch.tensor([[forced[-1]]])
                end = time.perf_counter()
                timings.append(end - start)
                start = end
        results.append(dict(name=name, messages=turns, tools=tools, prompt=text, ids=ids,
                            greedy=greedy, forced=forced, seconds=timings, max_abs=checkpoints))
        print(f"{name}: {len(ids)} input IDs, greedy {greedy}, seconds {timings}", flush=True)
    (args.output / "cases.json").write_text(json.dumps(
        dict(date=args.date, torch=torch.__version__, threads=args.threads, rounded_activations=args.round_activations, cases=results),
        indent=2, ensure_ascii=False,
    ) + "\n")


if __name__ == "__main__":
    main()
