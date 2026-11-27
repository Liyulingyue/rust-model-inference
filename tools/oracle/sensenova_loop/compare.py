"""Compare all official Loop visit states/logits, with floating-point error gates."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def metrics(actual, expected):
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"nonfinite or mismatched tensors: {a.shape} vs {b.shape}")
    delta = a - b
    rmse = float(np.sqrt(np.mean(delta * delta)))
    rms = float(np.sqrt(np.mean(b * b)))
    max_abs = float(np.max(np.abs(delta)))
    scale = float(np.max(np.abs(b)))
    denom = float(np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()))
    cosine = (
        float(np.dot(a.ravel(), b.ravel()) / denom)
        if denom
        else float(np.array_equal(a, b))
    )
    return {
        "max_abs": max_abs,
        "rmse": rmse,
        "relative_rmse": rmse / max(rms, 1e-12),
        "cosine": cosine,
        "pass": max_abs <= 1e-3 + 1e-4 * scale and rmse <= 1e-5 + 1e-4 * rms,
    }


def compare(oracle, native):
    fixtures = json.loads((oracle / "fixtures.json").read_text(encoding="utf-8"))
    results = json.loads((native / "results.json").read_text(encoding="utf-8"))
    summary = {"tokenizer_exact": True, "generated_exact": True, "checks": []}
    for case, fixture in enumerate(fixtures):
        records = [
            json.loads(line)
            for line in (native / f"case{case}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        groups = defaultdict(list)
        logits_at_row = {}
        seen_rows = 0
        final_visit = f"loop.visit.{len(fixture['steps'][0]['visits']) - 1}"
        for record in records:
            groups[record["name"]].append(record)
            if record["name"] == final_visit:
                seen_rows += record["shape"][0]
            elif record["name"] == "loop.logits":
                logits_at_row[seen_rows] = record
        total_rows = sum(info["rows"] for info in fixture["steps"])
        visits = len(fixture["steps"][0]["visits"])
        if any(len(groups[f"loop.visit.{v}"]) != total_rows for v in range(visits)):
            raise ValueError(f"case{case}: checkpoint count mismatch")
        generated = results[case]["generated"]
        summary["generated_exact"] &= generated == fixture["generated"]
        offset = 0
        for step, info in enumerate(fixture["steps"]):
            rows = info["rows"]
            for visit, shape in enumerate(info["visits"]):
                name = f"loop.visit.{visit}"
                chosen = groups[name][offset : offset + rows]
                if len(chosen) != rows:
                    raise ValueError(f"missing checkpoint {case}/{step}/{name}")
                if any(record["shape"] != [1, shape[1]] for record in chosen):
                    raise ValueError(f"wrong checkpoint shape {case}/{step}/{name}")
                actual = np.concatenate(
                    [
                        np.fromfile(record["binary_path"], dtype="<f4")
                        for record in chosen
                    ]
                ).reshape(shape)
                expected = np.load(oracle / f"case{case}.step{step}.visit{visit}.npy")
                summary["checks"].append(
                    {
                        "case": case,
                        "step": step,
                        "name": name,
                        **metrics(actual, expected),
                    }
                )
            record = logits_at_row.get(offset + rows)
            if record is None:
                raise ValueError(f"missing final-position logits {case}/{step}")
            actual = np.fromfile(record["binary_path"], dtype="<f4")
            expected = np.load(oracle / f"case{case}.step{step}.logits.npy")
            summary["checks"].append(
                {
                    "case": case,
                    "step": step,
                    "name": "loop.logits",
                    **metrics(actual, expected),
                }
            )
            offset += rows
        if "image" in fixture:
            expected = np.load(oracle / "vision.npy")
            actual = np.fromfile(native / "vision.f32", dtype="<f4").reshape(
                expected.shape
            )
            summary["checks"].append(
                {
                    "case": case,
                    "step": 0,
                    "name": "vision.output",
                    **metrics(actual, expected),
                }
            )
    summary["pass"] = summary["generated_exact"] and all(
        row["pass"] for row in summary["checks"]
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = compare(args.oracle, args.native)
    args.output.write_text(json.dumps(summary, indent=2))
    failed = [r for r in summary["checks"] if not r["pass"]]
    print(
        json.dumps(
            {
                "pass": summary["pass"],
                "generated_exact": summary["generated_exact"],
                "checkpoints": len(summary["checks"]),
                "failed": len(failed),
                "first_failure": next(iter(failed), None),
                "worst": max(summary["checks"], key=lambda r: r["relative_rmse"]),
            },
            indent=2,
        )
    )
    raise SystemExit(0 if summary["pass"] else 1)
