#!/usr/bin/env python3
"""Compare a Hy-MT3 oracle run against this repository's parity run.

Both sides write the same artifact layout (see `oracle.cpp` and
`examples/hy_mt3_parity.rs`), so this script is symmetric: it never cares which
directory came from llama.cpp.

Discrete outputs (prompt token IDs, greedy token IDs) must match **exactly** —
they are the contract. Floating-point outputs are compared with metrics, because
the two sides legitimately differ: this repository uses its own SIMD kernels,
an approximate SiLU (`silu_mul_approx_inplace`) and a different summation order,
and for quantized checkpoints the routed-expert set can flip on a near-tie.

Usage
-----
    python3 tools/oracle/hy_mt3/compare.py <oracle-dir> <rust-dir> [--json]

Exit status is 0 when the discrete contracts match and every numeric metric is
inside the thresholds below, 1 otherwise. The thresholds are fixed here (not
chosen per run) and are the ones quoted in `validation.md`.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

# Cosine similarity of the final-position logits. 0.999 is what the dense
# Q8_0 siblings reach on the same kernels; anything below 0.99 means the two
# sides disagree about the distribution, not just about rounding.
MIN_COSINE = 0.99
# RMSE relative to the logit scale (max |logit|). 1% keeps a quantized MoE with
# discrete routing honest without demanding bit equality from lossy operators.
MAX_REL_RMSE = 0.01


def load_ids(path: pathlib.Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data[0]["ids"] if data and isinstance(data[0], dict) else data
    return data.get("ids") or data.get("prompt_tokens")


def load_generated(path: pathlib.Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["generated_tokens"] if isinstance(data, dict) else data


def metrics(reference: np.ndarray, actual: np.ndarray):
    error = reference - actual
    rmse = float(np.sqrt(np.mean(error * error)))
    denom = float(np.linalg.norm(reference) * np.linalg.norm(actual))
    cosine = float(np.clip(np.dot(reference, actual) / denom, -1.0, 1.0)) if denom else 0.0
    scale = float(np.max(np.abs(reference)))
    return {
        "n": int(reference.size),
        "max_abs": float(np.max(np.abs(error))),
        "rmse": rmse,
        "relative_rmse": rmse / max(scale, 1e-12),
        "cosine": cosine,
        "scale": scale,
        "top1_reference": int(np.argmax(reference)),
        "top1_actual": int(np.argmax(actual)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle")
    parser.add_argument("rust")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    oracle = pathlib.Path(args.oracle)
    rust = pathlib.Path(args.rust)
    report = {}
    failures = []

    for name, loader in (("prompt_tokens", load_ids), ("generated_tokens", load_generated)):
        left = loader(oracle / f"{name}.json")
        right = loader(rust / f"{name}.json")
        if left != right:
            first = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
            failures.append(f"{name} differ at index {first}: oracle={left[:first + 1][-6:]} rust={right[:first + 1][-6:]}")
        report[name] = {"equal": left == right, "n": len(left)}

    reference = np.fromfile(oracle / "logits_prompt.f32", dtype="<f4").astype(np.float64)
    actual = np.fromfile(rust / "logits_prompt.f32", dtype="<f4").astype(np.float64)
    if reference.shape != actual.shape:
        failures.append(f"logits shape {reference.shape} vs {actual.shape}")
    else:
        values = metrics(reference, actual)
        report["logits"] = values
        if values["cosine"] < MIN_COSINE:
            failures.append(f"cosine {values['cosine']:.6f} < {MIN_COSINE}")
        if values["relative_rmse"] > MAX_REL_RMSE:
            failures.append(f"relative_rmse {values['relative_rmse']:.6f} > {MAX_REL_RMSE}")
        if values["top1_reference"] != values["top1_actual"]:
            failures.append(f"top1 differs: {values['top1_reference']} vs {values['top1_actual']}")

    if args.json:
        print(json.dumps({"report": report, "failures": failures}, indent=2))
    else:
        print(f"prompt_tokens equal: {report['prompt_tokens']['equal']} (n={report['prompt_tokens']['n']})")
        print(f"generated_tokens equal: {report['generated_tokens']['equal']} (n={report['generated_tokens']['n']})")
        if "logits" in report:
            v = report["logits"]
            print(
                f"logits: max_abs={v['max_abs']:.4e} rmse={v['rmse']:.4e} "
                f"rel_rmse={v['relative_rmse']:.4%} cosine={v['cosine']:.9f} "
                f"top1={v['top1_reference']}/{v['top1_actual']}"
            )
        print("PASS" if not failures else "FAIL")
        for failure in failures:
            print(f"  - {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
