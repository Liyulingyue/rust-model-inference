#!/usr/bin/env python3
"""Exact discrete contracts; tolerance-based checks for approximate/quantized math."""
import argparse
import json
from pathlib import Path
import numpy as np


def values(record):
    a = np.fromfile(record["binary_path"], dtype="<f4").astype(np.float64)
    assert a.size == int(np.prod(record["shape"])) and a.size > 0
    assert np.isfinite(a).all(), "non-finite checkpoint"
    return a


def metrics(a, b):
    assert a.shape == b.shape
    error = a-b
    rms = float(np.sqrt(np.mean(error*error)))
    denom = float(np.linalg.norm(a)*np.linalg.norm(b))
    cosine = float(np.clip(np.dot(a,b)/denom, -1, 1)) if denom else float(np.array_equal(a,b))
    return dict(max_abs=float(np.max(np.abs(error))), rms=rms,
                relative_rms=rms/max(float(np.sqrt(np.mean(a*a))), 1e-12),
                cosine=cosine)


def records(root):
    found = {}
    for line in (root / "trace.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["name"] in {"hy.enc_inp", "hy.after_transformer", "hy.after_proj2", "hy.projected", "hy.logits"} or r["name"].startswith("hy.projection."):
            key = (r["name"], r["occurrence"])
            assert key not in found, f"duplicate trace key {key}"
            found[key] = r
    return found


def compare(reference, actual, lossy_baseline=None):
    reference, actual = Path(reference), Path(actual)
    baseline = Path(lossy_baseline) if lossy_baseline else None
    if baseline:
        assert baseline.resolve() not in {reference.resolve(), actual.resolve()}, "calibration must use a separate Oracle run"
        assert all((p / "turns.json").is_file() for p in (reference, actual, baseline)), "calibration requires generated turns from all three runs"
    result = {}
    for name in ("tokens.json", "turns.json"):
        if (reference / name).exists():
            left = json.loads((reference / name).read_text())
            right = json.loads((actual / name).read_text())
            assert left == right, f"{name}: discrete output differs"
            if baseline:
                assert left == json.loads((baseline / name).read_text()), "reference paths have different contexts"
            result[name] = {"exact": True, "cases": len(left)}
    if (reference / "trace.jsonl").exists():
        left, right = records(reference), records(actual)
        assert left and left.keys() == right.keys(), "checkpoint names/order/count differ"
        assert list(left) == list(right), "checkpoint order differs"
        logit_bounds = (0.15, 0.005, 0.99998)
        if baseline:
            other = records(baseline)
            assert list(left) == list(other), "calibration checkpoints differ"
            calibration = []
            for key, ref in left.items():
                assert ref["shape"] == other[key]["shape"], "calibration shape differs"
                if key[0] == "hy.logits":
                    calibration.append(metrics(values(ref), values(other[key])))
            assert calibration, "calibration requires logits"
            # A whole quantized MoE compounds rounding discontinuities. Use an
            # independent Oracle-vs-Oracle baseline, with 2x margin and hard caps.
            logit_bounds = (
                max(0.15, 2*max(m["max_abs"] for m in calibration)),
                min(0.2, max(0.005, 2*max(m["relative_rms"] for m in calibration))),
                max(0.98, 1-max(0.00002, 2*max(1-m["cosine"] for m in calibration))),
            )
            result["lossy_calibration"] = dict(reference_path_metrics=calibration,
                                              bounds=dict(zip(("max_abs", "relative_rms", "cosine"), logit_bounds)))
        checked = []
        for key, ref in left.items():
            got = right[key]
            assert ref["shape"] == got["shape"], f"{key}: shape differs"
            a, b = values(ref), values(got)
            m = metrics(a, b)
            if key[0] == "hy.logits":
                limits = logit_bounds
            elif key[0].startswith("hy.projection."):
                limits = (2e-5, 2e-6, 1-2e-11)
            else:
                limits = (0.002 + 0.001*float(np.max(np.abs(a))), 0.003, 0.99999)
            assert m["max_abs"] <= limits[0], f"{key}: max_abs={m['max_abs']}"
            assert m["relative_rms"] <= limits[1], f"{key}: relative_rms={m['relative_rms']}"
            assert m["cosine"] >= limits[2], f"{key}: cosine={m['cosine']}"
            checked.append(dict(name=key[0], occurrence=key[1], shape=ref["shape"], **m))
        result["checkpoints"] = checked
    assert result, "reference has no comparison artifacts"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("actual")
    parser.add_argument("--lossy-baseline", help="same pinned Oracle with CPU repacking enabled")
    args = parser.parse_args()
    print(json.dumps(compare(args.reference, args.actual, args.lossy_baseline), indent=2))
