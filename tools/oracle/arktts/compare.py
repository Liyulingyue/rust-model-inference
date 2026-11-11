"""Compare complete official / Rust checkpoint streams with FP32 tolerances."""
import argparse
import json
from pathlib import Path

import numpy as np

# Same-weight implementation gates, distinct from quantization quality versus the original weights.
# F16 Prepared uses native half accumulators; the independent PyTorch Oracle accumulates widely.
TOLERANCES = {
    "float": (1e-4, 5e-5, 2e-3, 2e-4, 2e-5),
    # Frozen from the original three prompts before running the separate validation suite.
    # RMS/RoPE rounding can cross activation-quantizer bins and amplify across FastAR layers.
    # Same-input projection fixtures separately enforce 5e-5 relative RMSE for block kernels.
    "quantized": (2e-1, 1.5e-1, 5.0, 6e-3, 1e-3),
    "f16": (1e-2, 1e-2, 2.5e-1, 1e-3, 1e-4),
}


def precision_profile(kinds):
    if set(kinds) & {2, 3, 8, 12, 13, 14}:
        return "quantized"
    return "f16" if 1 in kinds else "float"


def compare(oracle, rust, profile="float"):
    if profile not in TOLERANCES:
        raise ValueError(f"unknown precision profile {profile}")
    relative_limit, normalized_limit, logits_limit, audio_max, audio_rmse = TOLERANCES[profile]
    expected = json.loads(Path(oracle).read_text())
    actual = [row for line in Path(rust).read_text().splitlines() if (row := json.loads(line)).get("binary_path")]
    if not expected:
        raise AssertionError("empty official checkpoint stream")
    if len(expected) != len(actual):
        raise AssertionError(f"checkpoint counts differ: official={len(expected)}, Rust={len(actual)}")
    metrics = []
    for index, (a, b) in enumerate(zip(expected, actual)):
        for key in ("name", "layer", "step", "shape"):
            if a.get(key) != b.get(key):
                raise AssertionError(f"checkpoint {index} {key}: {a.get(key)!r} != {b.get(key)!r}")
        x = np.fromfile(a["binary_path"], dtype="<f4").astype(np.float64)
        y = np.fromfile(b["binary_path"], dtype="<f4").astype(np.float64)
        if x.size != np.prod(a["shape"]) or y.size != x.size or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise AssertionError(f"invalid checkpoint data at {index}")
        delta = x - y
        max_abs = float(np.max(np.abs(delta)))
        rmse = float(np.sqrt(np.mean(delta * delta)))
        relative_rmse = rmse / max(float(np.sqrt(np.mean(x * x))), 1e-12)
        normalized_max_abs = max_abs / max(float(np.max(np.abs(x))), 1.0)
        cosine = float(np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-24))
        result = {"index": index, "name": a["name"], "layer": a["layer"], "step": a["step"],
                  "max_abs": max_abs, "rmse": rmse, "relative_rmse": relative_rmse,
                  "normalized_max_abs": normalized_max_abs, "cosine": cosine}
        metrics.append(result)
        # Residual activations exceed 10,000; scale their FP32 error by the reference magnitude.
        if normalized_max_abs > normalized_limit or relative_rmse > relative_limit:
            raise AssertionError(f"first out-of-tolerance checkpoint: {result}")
        if a["name"].endswith(".logits") and max_abs > logits_limit:
            raise AssertionError(f"logits out of tolerance: {result}")
        if a["name"] == "arktts.codec.audio" and (max_abs >= audio_max or rmse >= audio_rmse):
            raise AssertionError(f"waveform out of tolerance: {result}")
    print(json.dumps({"checkpoints": len(metrics), "worst_absolute": max(metrics, key=lambda x: x["max_abs"]),
                      "worst_relative": max(metrics, key=lambda x: x["relative_rmse"]),
                      "minimum_cosine": min(x["cosine"] for x in metrics)}, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle")
    parser.add_argument("rust")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=TOLERANCES, default="float")
    args = parser.parse_args()
    results = compare(args.oracle, args.rust, args.profile)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
