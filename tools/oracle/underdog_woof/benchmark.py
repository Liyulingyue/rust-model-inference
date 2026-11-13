"""Measure same-host public NumPy synthesis and release CLI CPU/Auto/Vulkan attempts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import tempfile
import time

from tools.converter.underdog_woof.convert_underdog_woof import CONTRACT, sha256
from tools.oracle.underdog_woof.check_interfaces import pcm
from tools.oracle.underdog_woof.export_reference import public_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bin-dir", type=Path, default=Path("target/release"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    assert args.repeats >= 1
    text = "Your order costs $12.50. It ships on March 3rd."
    started = time.perf_counter()
    official, _ = public_model(args.directory)
    construction = time.perf_counter() - started
    timings, digests = [], []
    for iteration in range(args.repeats + 1):
        started = time.perf_counter()
        waveform = official(text, seed=0)
        elapsed = time.perf_counter() - started
        digests.append(hashlib.sha256(waveform.tobytes()).hexdigest())
        if iteration:
            timings.append(elapsed)
    assert len(set(digests)) == 1, "official seeded requests are unstable"
    result = {"profile": "release (opt-level=3, fat LTO, codegen-units=1), features=vulkan",
              "host": platform.platform(), "cpu": "AMD EPYC 9334", "threads": 2,
              "numpy_thread_environment": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
              "source_revision": CONTRACT["source"]["revision"], "text": text,
              "voice": "juno", "seed": 0, "temperature": 0.5, "samples": len(waveform),
              "official_numpy": {"construction_seconds": construction, "scope": "synthesis only, model already loaded",
                                 "seconds": timings, "median_seconds": statistics.median(timings)}, "rust": {}}
    model = args.directory / "woof-f32.gguf"
    result["gguf_sha256"] = sha256(model)
    environment = {k: v for k, v in os.environ.items() if k not in ("RMI_COMPUTE_TRACE", "RMI_PARITY_TRACE")}
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix="woof-bench-") as temporary:
        path = Path(temporary) / "out.wav"
        for backend in ("cpu", "auto", "vulkan"):
            samples, errors, timings, executions, digests = [], [], [], [], []
            command = [str((args.bin_dir / "rust-model-inference").resolve()), "--tts", "--model", str(model),
                       "--threads", "2", "--compute", backend, "--prompt", text, "--seed", "0", "--out", str(path)]
            for iteration in range(args.repeats + 1 if backend == "cpu" else 1):
                path.unlink(missing_ok=True)
                started = time.perf_counter()
                process = subprocess.run(command, env=environment, capture_output=True, timeout=120)
                elapsed = time.perf_counter() - started
                log = process.stderr.decode(errors="replace")
                executions.extend(re.findall(r"Underdog Woof: .*", log))
                if process.returncode:
                    errors.append(log)
                    timings.append(elapsed)
                    assert backend != "cpu", log
                    continue
                data = path.read_bytes()
                samples.append(len(pcm(data)))
                digests.append(hashlib.sha256(data).hexdigest())
                if iteration or backend != "cpu":
                    timings.append(elapsed)
            assert len(set(digests)) <= 1, "Rust seeded requests are unstable"
            result["rust"][backend] = {"scope": "CLI process wall time, including startup/model load",
                "seconds": timings, "median_seconds": statistics.median(timings), "samples": samples,
                "execution": executions, "errors": errors, "command": command}
    result["notes"] = ["CPU includes model loading; NumPy synthesis-only timing excludes construction/imports.",
                       "Auto CPU fallback is not GPU throughput. A failed strict attempt is not a completed inference.",
                       "No external acceleration library is linked into Rust; NumPy is the read-only oracle."]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"numpy": result["official_numpy"]["median_seconds"],
                      "rust": {k: v["median_seconds"] for k, v in result["rust"].items()}}))


if __name__ == "__main__":
    main()
