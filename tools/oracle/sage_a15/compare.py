#!/usr/bin/env python3
"""Sage Q8_0: exact tokens/embeddings/routing, bounded lossy checkpoints."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess

MODEL_SHA256 = "f7247f6faef3f6c616b953595e1fc7c3fbf257d02d9915af5fdf15e808ea8ee4"
ROUTER_NAMES = {"ffn_moe_logits", "ffn_moe_probs", "ffn_moe_topk", "ffn_moe_weights_norm"}


def key(record):
    name, layer = record["name"], record.get("layer")
    if layer is not None and name.endswith(f"-{layer}"):
        name = name[: -len(f"-{layer}")]
    step = record.get("step")
    return name, -1 if layer is None else layer, record["occurrence"] if step is None else step


def selected(name, layer):
    return name in {"model.input_embed", "result_norm", "result_output", "l_out"} or (name in ROUTER_NAMES and layer in {0, 3, 39})


def load(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    out = {}
    for record in records:
        if "binary_path" not in record:
            continue
        k = key(record)
        if selected(*k[:2]):
            assert k not in out, f"duplicate checkpoint {k} in {path}"
            raw = Path(record["binary_path"]).read_bytes()
            assert len(raw) == record["len"] * 4, (k, "truncated checkpoint")
            out[k] = (record, raw)
    return records, out


def compare(oracle, rust, steps):
    required = {(name, -1, step) for name in ["model.input_embed", "result_norm", "result_output"] for step in range(steps)}
    required |= {("l_out", layer, step) for layer in range(40) for step in range(steps)}
    required |= {(name, layer, step) for name in ROUTER_NAMES for layer in [0, 3, 39] for step in range(steps)}
    assert set(oracle) == required, ("oracle checkpoint coverage", sorted(required - oracle.keys()), sorted(oracle.keys() - required))
    assert set(rust) == required, ("Rust checkpoint coverage", sorted(required - rust.keys()), sorted(rust.keys() - required))
    summary = []
    def graph_order(k):
        name, layer, step = k
        if name == "model.input_embed":
            return step, -1, 0
        if layer >= 0:
            names = ["ffn_moe_logits", "ffn_moe_probs", "ffn_moe_topk", "ffn_moe_weights_norm", "l_out"]
            return step, layer, names.index(name)
        return step, 40, ["result_norm", "result_output"].index(name)

    for k in sorted(required, key=graph_order):
        a, raw_a = oracle[k]
        b, raw_b = rust[k]
        # Single-row checkpoints differ only in the explicit singleton batch axis.
        shape_a = [d for d in a["shape"] if d != 1]
        shape_b = [d for d in b["shape"] if d != 1]
        assert shape_a == shape_b and len(raw_a) == len(raw_b), (k, "shape mismatch")
        if k[0] == "ffn_moe_topk":
            assert raw_a == raw_b, (k, "lossless checkpoint differs")
            summary.append({"key": k, "exact": True})
            continue
        lhs = struct.unpack(f"<{len(raw_a)//4}f", raw_a)
        rhs = struct.unpack(f"<{len(raw_b)//4}f", raw_b)
        assert all(math.isfinite(x) for x in (*lhs, *rhs)), (k, "non-finite checkpoint")
        if k[0] == "model.input_embed":
            assert raw_a == raw_b, (k, "lossless checkpoint differs")
            summary.append({"key": k, "exact": True})
            continue
        delta2 = sum((x-y)**2 for x, y in zip(lhs, rhs))
        norm_a, norm_b = sum(x*x for x in lhs), sum(x*x for x in rhs)
        relative_l2 = math.sqrt(delta2 / max(norm_a, 1e-30))
        cosine = sum(x*y for x, y in zip(lhs, rhs)) / max(math.sqrt(norm_a * norm_b), 1e-30)
        max_abs = max(abs(x-y) for x, y in zip(lhs, rhs))
        assert relative_l2 <= 0.01 and (cosine >= 0.99995 or norm_a < 1e-20), (k, relative_l2, cosine, max_abs)
        if k[0] == "result_output":
            assert max_abs <= 0.5, (k, "logit error", max_abs)
        summary.append({"key": k, "relative_l2": relative_l2, "cosine": cosine, "max_abs": max_abs})
    return summary


def run(command, log, env=None):
    with log.open("w") as out:
        subprocess.run(list(map(str, command)), stdout=out, stderr=subprocess.STDOUT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["model", "oracle", "rust-test", "out"]:
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--backend", choices=["cpu", "vulkan"], default="cpu")
    args = parser.parse_args()
    assert args.model.stat().st_size == 37802150112
    with args.model.open("rb") as model:
        assert hashlib.file_digest(model, "sha256").hexdigest() == MODEL_SHA256
    args.out.mkdir(parents=True, exist_ok=True)
    for case, prompt in [("raw", "Hello"), ("chat", "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")]:
        directory = (args.out / case).resolve()
        directory.mkdir(exist_ok=False)
        oracle_trace, rust_trace = directory / "oracle.jsonl", directory / "rust.jsonl"
        env = dict(os.environ, RMI_PARITY_TRACE=str(oracle_trace), RMI_PARITY_FILTER=",".join(["model.input_embed", "result_norm", "result_output", *sorted(ROUTER_NAMES), *[f"l_out-{layer}" for layer in range(40)]]))
        run([args.oracle, "-m", args.model, "--no-repack", "-p", prompt, "--no-escape", "-n", "4", "-c", "128", "-b", "1", "-ub", "1", "-t", "8", "-tb", "8", "-ngl", "0", "-ctk", "f32", "-ctv", "f32", "-fa", "off", "--temp", "0", "--repeat-penalty", "1"], directory / "oracle.log", env)
        env.pop("RMI_SCALAR", None)
        env.update(RMI_SAGE_BACKEND=args.backend, RMI_PARITY_TRACE=str(rust_trace), RMI_SAGE_MODEL=str(args.model), RMI_SAGE_PROMPT=prompt, RMI_SAGE_TOKENS=str(directory / "tokens.json"))
        run([args.rust_test, "--ignored", "--exact", "sage_dump_trace", "--nocapture"], directory / "rust.log", env)
        assert f"Sage trace backend: {args.backend}" in (directory / "rust.log").read_text()
        records, oracle = load(oracle_trace)
        _, rust = load(rust_trace)
        tokens = json.loads((directory / "tokens.json").read_text())
        processed = [token for r in records if r["name"] == "qwen35moe.prompt_ids" for token in r["token_ids"]]
        generated = [r["token_ids"] for r in records if r["name"] == "qwen35.greedy_token_ids"]
        summary = compare(oracle, rust, len(processed))
        assert processed == tokens["processed"] and generated == [tokens["generated"]], ("token mismatch", processed, generated, tokens)
        (directory / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(case, len(summary), "checkpoints; greedy", tokens["generated"], flush=True)


if __name__ == "__main__":
    main()
