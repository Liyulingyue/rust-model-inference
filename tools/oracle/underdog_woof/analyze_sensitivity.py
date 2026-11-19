"""Locate free-generation drift and measure official flow-head rounding sensitivity.

This diagnostic preserves failed strict gates; its perturbations do not change
the model files, Rust implementation, acceptance thresholds or official source.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from tools.oracle.underdog_woof.check_interfaces import metrics
from tools.oracle.underdog_woof.export_reference import export, export_public, public_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--test-binary", type=Path, required=True,
                        help="lib test executable built with vulkan,parity-trace")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    text = "Your order costs $12.50. It ships on March 3rd."
    reference = args.output / "public"
    export_public(args.directory, reference, text=text, voice="mira", gguf=args.gguf)
    expected = np.fromfile(reference / "waveform.f32", dtype="<f4")
    results = {"perturbations": [], "stages": []}
    for ulps in (1, 4, 16):
        for direction in (np.inf, -np.inf):
            tts, _ = public_model(args.directory, voice="mira", gguf=args.gguf)
            original = tts.model.speech._head
            error, power, count, maximum = 0., 0., 0, 0.

            def head(*a, **kw):
                nonlocal error, power, count, maximum
                value = original(*a, **kw)
                rounded = value
                for _ in range(ulps):
                    rounded = np.nextafter(rounded, np.float32(direction))
                delta = rounded.astype(np.float64) - value
                error += np.sum(delta ** 2)
                power += np.sum(value.astype(np.float64) ** 2)
                count += value.size
                maximum = max(maximum, float(np.max(np.abs(delta))))
                return rounded

            tts.model.speech._head = head
            waveform = tts(text, seed=0)
            assert len(waveform) and np.isfinite(waveform).all() and count
            item = {"ulps": ulps, "direction": str(direction), "samples": len(waveform),
                    "head_max_abs": maximum, "head_relative_rmse": float(np.sqrt(error / power)),
                    "waveform": metrics(waveform, expected) if len(waveform) == len(expected) else None}
            results["perturbations"].append(item)
    phrase = "Your order costs twelve dollars and fifty cents."
    natural = args.output / "natural"
    export(args.directory, natural, text=phrase, voice="mira", frames=750,
           gguf=args.gguf, eos_threshold=0.9)
    frames = json.loads((natural / "manifest.json").read_text())["frames"]
    forced = args.output / "forced"
    export(args.directory, forced, text=phrase, voice="mira", frames=frames,
           gguf=args.gguf, force_latents=natural / "latents.f32")
    for fixture in (natural, forced):
        trace = fixture / "rust.jsonl"
        environment = {**os.environ, "RMI_WOOF_FIXTURE": str(fixture), "RMI_WOOF_COMPUTE": "cpu",
                       "RMI_WOOF_RESULT": str(fixture / "rust-result.json"), "RMI_PARITY_TRACE": str(trace)}
        with (fixture / "rust.log").open("w") as log:
            process = subprocess.run([str(args.test_binary.resolve()), "official_pipeline_fixture",
                                      "--ignored", "--nocapture"], env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=120)
        manifest = json.loads((fixture / "manifest.json").read_text())
        groups = {}
        for line in trace.read_text().splitlines():
            item = json.loads(line)
            groups.setdefault(item["name"], []).append(np.fromfile(item["binary_path"], dtype="<f4"))
        summary = {"case": fixture.name, "exit": process.returncode,
                   "first_float_difference": None, "first_relative_gate_failure": None,
                   "peak_relative_error": 0.}
        for name, chunks in groups.items():
            if name not in manifest["checkpoints"]:
                continue
            actual = np.concatenate(chunks)
            expected_stage = np.fromfile(fixture / manifest["checkpoints"][name]["file"], dtype="<f4")
            measured = metrics(actual, expected_stage)
            if not np.array_equal(actual.view(np.uint32), expected_stage.view(np.uint32)):
                if summary["first_float_difference"] is None:
                    summary["first_float_difference"] = {"checkpoint": name, "metrics": measured}
            if measured["relative_rmse"] > 2e-4 and summary["first_relative_gate_failure"] is None:
                summary["first_relative_gate_failure"] = {"checkpoint": name, "metrics": measured}
            summary["peak_relative_error"] = max(summary["peak_relative_error"], measured["relative_rmse"])
        if fixture == forced:
            assert process.returncode == 0, "teacher-forced same-weight gate failed"
        results["stages"].append(summary)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
