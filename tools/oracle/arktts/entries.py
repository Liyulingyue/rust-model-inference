"""Exercise real CLI/HTTP synthesis and cloning for each exported precision/backend."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
import wave

import numpy as np


def run(root, binaries, formats, computes, output_dir=None):
    root, binaries = root.resolve(), binaries.resolve()
    evidence = output_dir.resolve() if output_dir else root / "entries"
    evidence.mkdir(parents=True, exist_ok=False)
    samples = np.fromfile(root.parent / "oracle-validation-base/reference.f32", dtype="<f4")
    reference = evidence / "reference.wav"
    with wave.open(str(reference), "wb") as wav:
        wav.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
        wav.writeframes((samples.clip(-1, 1) * 32767).round().astype("<i2").tobytes())
    rows = []
    for fmt in formats:
        for compute in computes:
            folder = evidence / f"{fmt}-{compute}"
            folder.mkdir()
            main = root / fmt / f"Audio8-TTS-Realistic-Air-{fmt.upper()}.gguf"
            codec = root / fmt / f"Audio8-TTS-Realistic-Air-codec-{fmt.upper()}.gguf"
            args = ["--tts", "--model", str(main), "--mmproj", str(codec), "--threads", "4",
                    "--temperature", "0", "--max-tokens", "4", "--compute", compute]
            environment = {**os.environ, "RAYON_NUM_THREADS": "4", "RMI_COMPUTE_TRACE": "1"}
            row = {"format": fmt, "compute": compute, "cli": [], "http": []}
            for clone in (False, True):
                output = folder / ("clone.wav" if clone else "plain.wav")
                command = [str(binaries / "rust-model-inference"), *args, "--prompt", "One, two, three.", "--out", str(output)]
                if clone:
                    command += ["--ref-audio", str(reference), "--ref-text", "A different reference."]
                with output.with_suffix(".log").open("w") as log:
                    began = time.perf_counter()
                    result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=180)
                if result.returncode:
                    raise RuntimeError(f"CLI failed: {output.with_suffix('.log')}")
                with wave.open(str(output), "rb") as wav:
                    assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 44100)
                    assert 0 < wav.getnframes() <= 8192
                row["cli"].append({"clone": clone, "seconds": time.perf_counter() - began,
                                   "command": command, "bytes": output.stat().st_size})
            with socket.socket() as address:
                address.bind(("127.0.0.1", 0))
                port = address.getsockname()[1]
            server_command = [str(binaries / "rust-model-server"), *args, "--host", "127.0.0.1", "--port", str(port)]
            with (folder / "server.log").open("w") as log:
                server = subprocess.Popen(server_command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 60
                    while True:
                        try:
                            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                                break
                        except (OSError, urllib.error.URLError):
                            if server.poll() is not None or time.monotonic() > deadline:
                                raise RuntimeError(f"server failed: {folder / 'server.log'}")
                            time.sleep(.1)
                    for clone in (False, True):
                        for response_format in ("wav", "pcm"):
                            payload = {"model": "audio8", "input": "One, two, three.", "response_format": response_format}
                            if clone:
                                payload.update(voice=reference.as_uri(), reference_text="A different reference.")
                            request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/audio/speech",
                                        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                            began = time.perf_counter()
                            with urllib.request.urlopen(request, timeout=180) as response:
                                output = response.read()
                                assert response.status == 200
                            expected = (folder / ("clone.wav" if clone else "plain.wav")).read_bytes()
                            assert output == (expected if response_format == "wav" else expected[44:]), (fmt, compute, clone, response_format)
                            row["http"].append({"request": payload, "status": 200, "bytes": len(output),
                                                "seconds": time.perf_counter() - began, "equals_cli": True})
                finally:
                    server.terminate()
                    server.wait(timeout=10)
            logs = "\n".join(path.read_text() for path in folder.glob("*.log"))
            row["gpu_linear_calls"] = logs.count("backend=Vulkan")
            row["cpu_linear_calls"] = logs.count("backend=Cpu")
            row["last_device_counters"] = re.findall(r"ComputeStats \{[^\n]+", logs)[-1:]
            if compute != "cpu":
                assert row["gpu_linear_calls"] > 0, "Auto fallback is not GPU verification"
            else:
                assert row["gpu_linear_calls"] == 0
            rows.append(row)
            (evidence / "results.json").write_text(json.dumps(rows, indent=2))
            print(fmt, compute, "CLI/HTTP WAV/PCM + clone passed", flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="multiq directory")
    parser.add_argument("--binaries", type=Path, default=Path("target/release-fast"))
    parser.add_argument("--formats", nargs="+", default=["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q4_k", "q5_k", "q6_k"])
    parser.add_argument("--compute", nargs="+", choices=["cpu", "auto", "vulkan"], default=["cpu", "auto", "vulkan"])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.model_dir, args.binaries, args.formats, args.compute, args.output_dir)
