#!/usr/bin/env python3
"""Check trace structure exactly and reduced-precision values numerically."""
import argparse
import json
import math
import struct
from pathlib import Path

# Quantized/BF16 projections and SIMD math are numerical checks, not bit gates.
RELATIVE_L2_LIMIT = 0.01
SCALED_MAX_ABS_LIMIT = 0.01


def load(path):
    tensors, tokens = {}, {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if "token_ids" in row:
            if row["name"] in tokens:
                raise ValueError("duplicate token record")
            tokens[row["name"]] = row["token_ids"]
        else:
            key = (row["name"], row["layer"], row["step"])
            if key in tensors:
                raise ValueError(f"duplicate checkpoint {key}")
            tensors[key] = row
    return tensors, tokens


def payload(row):
    raw = Path(row["binary_path"]).read_bytes()
    if len(raw) != math.prod(row["shape"]) * 4:
        raise ValueError("checkpoint payload/shape mismatch")
    values = [v[0] for v in struct.iter_unpack("<f", raw)]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("empty or non-finite checkpoint")
    return raw, values


def compare(oracle, rust, first_step=0):
    expected, expected_tokens = load(oracle)
    actual, actual_tokens = load(rust)
    failures, metrics = [], []
    for name in ("prompt_ids", "generated_ids"):
        if name not in expected_tokens or name not in actual_tokens:
            failures.append(f"missing {name}")
        elif expected_tokens[name] != actual_tokens[name]:
            failures.append(f"{name}: oracle={expected_tokens[name]} rust={actual_tokens[name]}")
    steps = len(expected_tokens.get("prompt_ids", [])) + len(expected_tokens.get("generated_ids", []))
    batches = expected_tokens.get("prefill_batches", [1] * steps)
    if actual_tokens.get("prefill_batches", [1] * steps) != batches:
        failures.append("prefill batch schedule differs")
    if not batches or any(type(n) is not int or not 1 <= n <= 32 for n in batches) or sum(batches) != steps:
        failures.append("invalid prefill batch schedule")
        batches = []
    contract = []
    step = 0
    for size in batches:
        for name, layer in [("embd", None)] + [("l_out", i) for i in range(192)]:
            contract.extend((name, layer, s) for s in range(max(first_step, step), step + size))
        if step + size - 1 >= first_step:
            contract.extend((name, None, step + size - 1) for name in ("result_norm", "result_output"))
        step += size
    if not expected or list(expected) != contract or list(actual) != contract:
        failures.append("checkpoint names/layers/steps/order differ or trace is empty")
    for key, row in expected.items():
        other = actual.get(key)
        if other is None:
            continue
        if row["shape"] != other["shape"]:
            failures.append(f"{key}: shapes differ")
            continue
        raw_o, values_o = payload(row)
        raw_r, values_r = payload(other)
        squared_error = math.fsum((a - b) ** 2 for a, b in zip(values_o, values_r))
        squared_norm = math.fsum(a * a for a in values_o)
        relative_l2 = math.sqrt(squared_error / squared_norm) if squared_norm else math.sqrt(squared_error)
        max_abs = max(abs(a - b) for a, b in zip(values_o, values_r))
        scaled_max_abs = max_abs / max(1.0, max(abs(a) for a in values_o))
        exact = key[0] == "embd"
        ok = raw_o == raw_r if exact else (
            relative_l2 <= RELATIVE_L2_LIMIT and scaled_max_abs <= SCALED_MAX_ABS_LIMIT
        )
        metric = dict(name=key[0], layer=key[1], step=key[2], relative_l2=relative_l2,
                      max_abs=max_abs, scaled_max_abs=scaled_max_abs, exact=exact, passed=ok)
        metrics.append(metric)
        if not ok:
            failures.append(str(metric))
    return dict(passed=not failures, checked=len(metrics), first_step=first_step, failures=failures,
                tokens=actual_tokens, limits=dict(relative_l2=RELATIVE_L2_LIMIT,
                scaled_max_abs=SCALED_MAX_ABS_LIMIT),
                worst_relative_l2=max(metrics, key=lambda x: x["relative_l2"], default=None),
                worst_max_abs=max(metrics, key=lambda x: x["max_abs"], default=None),
                metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle")
    parser.add_argument("rust")
    parser.add_argument("--first-step", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.first_step < 0:
        parser.error("--first-step must be non-negative")
    result = compare(args.oracle, args.rust, args.first_step)
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    summary = {k: v for k, v in result.items() if k not in ("metrics", "failures")}
    summary["failure_count"] = len(result["failures"])
    summary["first_failures"] = result["failures"][:5]
    print(json.dumps(summary, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
