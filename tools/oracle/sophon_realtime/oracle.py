"""Independent Transformers reference for the Sophon Qwen3-VL thinker GGUF.

GGUF parsing/dequantization uses upstream gguf-py, not this repo's converter.
This evaluates weight-only exports; it does not emulate native dynamic FP8.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import tempfile
import time
from pathlib import Path

import numpy as np


# Declared before real inference in validation.md; implementation error only.
GATES = {
    "f32": (0.001, 0.999999), "f16": (0.005, 0.99998), "bf16": (0.005, 0.99998),
    **{name: (0.03, 0.999) for name in ("q8_0", "q4_0", "q4_1", "q4_k", "q5_k", "q6_k")},
}


def meets_gate(result, precision):
    if precision not in GATES:
        raise ValueError(f"unknown precision: {precision}")
    relative, cosine = result["relative_rmse"], result["cosine"]
    maximum, minimum = GATES[precision]
    return relative is not None and np.isfinite(relative) and np.isfinite(cosine) and relative <= maximum and cosine >= minimum


def metrics(actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or not actual.size or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("outputs must have identical nonempty shapes and finite values")
    error = actual - expected
    rmse = float(np.sqrt(np.mean(error * error)))
    reference = float(np.sqrt(np.mean(expected * expected)))
    denominator = float(np.linalg.norm(actual) * np.linalg.norm(expected))
    return {
        "max_abs": float(np.max(np.abs(error))), "rmse": rmse,
        "relative_rmse": rmse / reference if reference else (0.0 if not rmse else None),
        "cosine": float(np.dot(actual.ravel(), expected.ravel()) / denominator) if denominator else float(np.array_equal(actual, expected)),
    }


def tensor_name(name):
    top = {"token_embd.weight": "embed_tokens.weight", "output_norm.weight": "norm.weight", "output.weight": "lm_head.weight"}
    if name in top:
        return top[name]
    parts = name.split(".")
    if len(parts) != 4 or parts[0] != "blk" or parts[3] != "weight":
        raise ValueError(f"unexpected thinker tensor: {name}")
    suffix = {
        "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
        "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
        "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj",
        "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
    }.get(parts[2])
    if suffix is None:
        raise ValueError(f"unexpected thinker tensor: {name}")
    return f"layers.{int(parts[1])}.{suffix}.weight"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_weight(tensor, dequantize, scratch):
    shape = tuple(reversed(tensor.shape))
    values = np.asarray(dequantize(tensor.data, tensor.tensor_type), dtype=np.float32).reshape(shape)
    if int(tensor.tensor_type) == 0:  # F32 already borrows the GGUF mmap.
        return values
    # A widened 32B model exceeds the host's memory limit; retain decoded weights on remote disk.
    offset = scratch.seek(0, 2)
    values.tofile(scratch)
    scratch.flush()
    mapped = np.memmap(scratch, dtype=np.float32, mode="r+", shape=shape, offset=offset)
    scratch.seek(offset + values.nbytes)
    return mapped


def thinker_activations(model, kinds):
    """Reuse independent PreparedRows QDQ with the thinker's GGUF name map."""
    from .code2wav_oracle import RuntimeActivations

    names = {tensor_name(name): name for name in kinds if name != "output.weight"}
    return RuntimeActivations(model, kinds, names)


def full_prefix_logits(model, head, prompt_ids, teacher, steps, activation=None):
    """Evaluate the same causal rows once, retaining the Rust teacher history."""
    import torch

    if not prompt_ids or steps < 1 or len(teacher) < steps - 1:
        raise ValueError("full-prefix requires a prompt and sufficient teacher tokens")
    prefix = torch.tensor([list(prompt_ids) + list(teacher[:steps - 1])], dtype=torch.long)
    positions = torch.arange(prefix.shape[1])[None, None].expand(3, 1, -1)
    started = time.perf_counter()
    output = model(input_ids=prefix, position_ids=positions, use_cache=False)
    hidden = output.last_hidden_state[0, len(prompt_ids) - 1:len(prompt_ids) - 1 + steps]
    if activation is not None:
        hidden = activation.apply(hidden, "output.weight")
    logits = torch.nn.functional.linear(hidden, head).detach().float().numpy()
    return logits, time.perf_counter() - started


def run(args):
    import torch
    import transformers
    import gguf
    from gguf.quants import dequantize
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    if args.steps < 1 or args.threads < 1:
        raise ValueError("steps and threads must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    # Source rope_parameters is the Transformers 5 spelling of this same contract.
    config_path = args.model_dir / "thinker/config.json"
    expected_hash = "b48dc9247bf75aa54ba2af1109ae9e2e5414db54ce0331dba8169ccd6f4405ba"
    if sha256(config_path) != expected_hash:
        raise ValueError("source thinker config does not match the pinned release")
    text = json.loads(config_path.read_text())["text_config"]
    rope = text.pop("rope_parameters")
    text["rope_theta"] = rope["rope_theta"]
    text["rope_scaling"] = {"rope_type": "default", "mrope_section": rope["mrope_section"], "mrope_interleaved": True}
    config = Qwen3VLTextConfig(**text)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir / "thinker", local_files_only=True)
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    trace = [json.loads(line) for line in args.rust_trace.read_text().splitlines()] if args.rust_trace else []
    if trace:
        rust_ids = next(record["token_ids"] for record in trace if record["name"] == "prompt_ids")
        if rust_ids != ids:
            raise ValueError("Rust and official tokenizer/template prompt IDs differ")
    teacher = next((r["token_ids"] for r in trace if r["name"] == "generated_ids"), [])
    if trace and len(teacher) < args.steps:
        raise ValueError("Rust trace has fewer decode tokens than the requested steps")
    if args.reference_execution == "full-prefix" and (args.kv != "f32" or not teacher):
        raise ValueError("full-prefix comparison requires Rust teacher tokens and F32 KV")

    reader = gguf.GGUFReader(str(args.gguf))
    def field_string(key):
        field = reader.fields.get(key)
        return bytes(field.parts[field.data[0]]).decode("utf-8") if field else None
    if field_string("general.architecture") != "sophon_realtime" or field_string("sophon_realtime.component") != "thinker" or field_string("sophon_realtime.source_revision") != "990d6a74bbf15c1765a6f7736fa839f423fd5b59":
        raise ValueError("GGUF is not the pinned Sophon thinker")
    state = {}
    with tempfile.TemporaryFile(prefix="reference-weights-", dir=args.output) as scratch:
        for tensor in reader.tensors:
            name = tensor_name(tensor.name)
            state[name] = torch.from_numpy(reference_weight(tensor, dequantize, scratch))
        scratch_bytes = scratch.tell()
    print(json.dumps({"reference_scratch_bytes": scratch_bytes}), flush=True)
    head = state.pop("lm_head.weight")
    with torch.device("meta"):
        model = Qwen3VLTextModel(config)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    if any(buffer.is_meta for buffer in model.rotary_emb.buffers()):
        model.rotary_emb = type(model.rotary_emb)(config=config, device="cpu")
    model.eval()
    kinds = {tensor.name: int(tensor.tensor_type) for tensor in reader.tensors}
    activation = thinker_activations(model, kinds) if args.activation == "runtime" else None

    class ReferenceCache(DynamicCache):
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            if args.kv == "f16":
                key_states = key_states.half().float()
                value_states = value_states.half().float()
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

    cache = ReferenceCache(config=config)
    current = torch.tensor([ids], dtype=torch.long)
    next_position = 0
    records = []
    generated = []
    checkpoints = [r for r in trace if r["name"] == "result_output"]
    prefix_logits, forward_seconds = (
        full_prefix_logits(model, head, ids, teacher, args.steps, activation)
        if args.reference_execution == "full-prefix" else (None, 0.0)
    )
    for step in range(args.steps):
        if prefix_logits is None:
            positions = torch.arange(next_position, next_position + current.shape[1])[None, None, :].expand(3, 1, -1)
            started = time.perf_counter()
            output = model(input_ids=current, position_ids=positions, past_key_values=cache, use_cache=True)
            head_input = output.last_hidden_state[:, -1, :]
            if activation is not None:
                head_input = activation.apply(head_input, "output.weight")
            logits = torch.nn.functional.linear(head_input, head)[0].float().numpy()
            elapsed = time.perf_counter() - started
            forward_seconds += elapsed
        else:
            logits, elapsed = prefix_logits[step], None
        logits.tofile(args.output / f"logits-{step}.f32")
        selected = int(np.argmax(logits))
        generated.append(selected)
        top = np.sort(logits)[-2:]
        record = {"step": step, "greedy_id": selected, "greedy_margin": float(top[-1] - top[-2]), "seconds": elapsed}
        if checkpoints:
            # CPU traces can include one result per prefill row; the final prompt row is the decode origin.
            if len(checkpoints) < len(ids) + args.steps - 1:
                if len(checkpoints) != args.steps:
                    raise ValueError("unexpected result_output checkpoint count")
                index = step
            else:
                index = len(ids) - 1 + step
            path = Path(checkpoints[index]["binary_path"])
            record["rust_metrics"] = metrics(np.fromfile(path, dtype="<f4"), logits)
            record["rust_greedy_id"] = int(np.argmax(np.fromfile(path, dtype="<f4")))
            record["numerical_gate_pass"] = bool(meets_gate(record["rust_metrics"], args.precision))
            record["greedy_match"] = record["rust_greedy_id"] == selected
        records.append(record)
        print(json.dumps(record), flush=True)
        next_position += current.shape[1]
        current = torch.tensor([[teacher[step] if teacher else selected]], dtype=torch.long)
    result = {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "transformers_source_sha256": sha256(Path(inspect.getfile(Qwen3VLTextModel))),
        "gguf_sha256": sha256(args.gguf), "kv": args.kv, "threads": args.threads,
        "gguf_file_type": int(reader.fields["general.file_type"].parts[reader.fields["general.file_type"].data[0]][0]),
        "reference_scratch_bytes": scratch_bytes,
        "reference_execution": args.reference_execution,
        "reference_forward_seconds": forward_seconds,
        "source_config_sha256": expected_hash, "semantics": "F32 compute of independently dequantized weight-only GGUF",
        "prompt": prompt, "prompt_ids": ids, "greedy_ids": generated, "records": records,
        "teacher_tokens": teacher[:args.steps] if teacher else None,
        "comparison_history": "Rust teacher-forced history" if teacher else "autonomous reference greedy",
        "tokenizer_samples": [{"text": sample, "ids": tokenizer.encode(sample, add_special_tokens=False)} for sample in ["", "你好\n\t  世界", "café 👩‍💻 １２３", "<|audio|>[laughter]<strong>你好</strong>"]],
        "precision": args.precision, "gate": dict(zip(("maximum_relative_rmse", "minimum_cosine"), GATES[args.precision])),
        "activation": args.activation,
        "activation_projections": activation.description if activation is not None else [],
        "tensor_types": {name: sum(t.tensor_type.name == name for t in reader.tensors) for name in sorted({t.tensor_type.name for t in reader.tensors})},
        "parity_pass": all(r["numerical_gate_pass"] and r["greedy_match"] for r in records) if trace else None,
    }
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if activation is not None:
        activation.close()
    if result["parity_pass"] is False:
        raise SystemExit("numerical or greedy parity gate failed; inspect result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rust-trace", type=Path)
    parser.add_argument("--prompt", default="你好，请简短介绍自己。")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--kv", choices=["f16", "f32"], default="f32")
    parser.add_argument("--precision", choices=list(GATES), default="f32")
    parser.add_argument("--activation", choices=["f32", "runtime"], default="f32")
    parser.add_argument("--reference-execution", choices=["cached", "full-prefix"], default="cached")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
