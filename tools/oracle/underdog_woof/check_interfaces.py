"""Run real CLI/HTTP format/backend probes; keep failures visible in a JSON matrix."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
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

from tools.converter.underdog_woof.convert_underdog_woof import FORMATS, sha256


def pcm(data):
    with wave.open(io.BytesIO(data)) as audio:
        assert (audio.getnchannels(), audio.getframerate(), audio.getsampwidth()) == (1, 24000, 2)
        values = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    assert len(values) and np.any(values), "empty/silent audio"
    return values


def metrics(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    delta = actual - expected
    return {"max_abs": float(np.max(np.abs(delta))),
            "relative_rmse": float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-30)),
            "cosine": float(np.dot(actual, expected) /
                            max(np.linalg.norm(actual) * np.linalg.norm(expected), 1e-30))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bin-dir", type=Path, default=Path("target/release-fast"))
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--backends", nargs="+", choices=("cpu", "auto", "vulkan"),
                        default=["cpu", "auto", "vulkan"])
    parser.add_argument("--public-fixtures", type=Path, default=Path("target/underdog-woof"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cli, server = [str((args.bin_dir / name).resolve())
                   for name in ("rust-model-inference", "rust-model-server")]
    environment = {**os.environ, "RMI_COMPUTE_TRACE": "1"}
    rows = []
    for fmt in args.formats:
        model = (args.directory / f"woof-{fmt}.gguf").resolve()
        for backend in args.backends:
            root = args.output / f"{fmt}-{backend}"
            root.mkdir()
            common = ["--tts", "--model", str(model), "--threads", "2", "--compute", backend]
            row = {"format": fmt, "requested": backend, "model_sha256": sha256(model),
                   "cli": [], "http": [], "checks": [], "errors": [], "oracle_failures": []}

            def run(text, voice, name, extra=()):
                path = root / f"{name}.wav"
                command = [cli, *common, "--prompt", text, "--voice", voice, "--seed", "0",
                           "--temp", "0.5", "--out", str(path), *extra]
                started = time.perf_counter()
                result = subprocess.run(command, env=environment, capture_output=True, timeout=120)
                elapsed = time.perf_counter() - started
                log = result.stderr.decode(errors="replace")
                (root / f"{name}.log").write_text(log)
                item = {"command": command, "exit": result.returncode, "seconds": elapsed,
                        "execution": re.findall(r"Underdog Woof: .*", log)}
                row["cli"].append(item)
                if result.returncode:
                    assert not path.exists(), "failed request wrote audio"
                    return None
                data = path.read_bytes()
                item["samples"] = len(pcm(data))
                return data

            wavs = {voice: run("Hi.", voice, voice) for voice in ("juno", "mira")}
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            address = f"http://127.0.0.1:{port}"

            def http(path, payload=None):
                request = urllib.request.Request(address + path,
                    data=None if payload is None else json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        return response.status, response.read(), response.headers.get_content_type()
                except urllib.error.HTTPError as error:
                    return error.code, error.read(), error.headers.get_content_type()

            log_path = root / "server.log"
            command = [server, *common, "--host", "127.0.0.1", "--port", str(port)]
            row["server_command"] = command
            with log_path.open("w") as log:
                process = subprocess.Popen(command, env=environment, stdout=log, stderr=log)
                try:
                    deadline = time.monotonic() + 30
                    while True:
                        try:
                            status, data, _ = http("/v1/models")
                            assert status == 200
                            model_id = json.loads(data)["data"][0]["id"]
                            break
                        except (urllib.error.URLError, ConnectionError):
                            if process.poll() is not None or time.monotonic() > deadline:
                                raise RuntimeError("server failed to start: " + log_path.read_text()[-2000:])
                            time.sleep(0.1)
                    for voice, expected in wavs.items():
                        payload = {"input": "Hi.", "voice": voice, "seed": 0, "temperature": 0.5,
                                   "model": model_id}
                        status, data, kind = http("/v1/audio/speech", payload)
                        item = {"route": "/v1/audio/speech", "request": payload, "status": status,
                                "content_type": kind}
                        row["http"].append(item)
                        if expected is None:
                            assert backend == "vulkan" and status >= 500, (status, data)
                            row["errors"].append(data.decode(errors="replace"))
                            continue
                        assert (status, kind, data) == (200, "audio/wav", expected), (status, data[:200])
                        item["samples"] = len(pcm(data))
                        status, data, kind = http("/v1/audio/speech", {**payload, "response_format": "pcm"})
                        assert (status, kind, data) == (200, "audio/pcm", pcm(expected).tobytes())
                        reference = args.public_fixtures / f"public-f32-{voice}-short/waveform.f32"
                        if fmt == "f32" and reference.exists():
                            values = np.fromfile(reference, dtype="<f4")
                            measured = metrics(pcm(expected) / 32767., np.clip(values, -1, 1))
                            item["public_oracle"] = measured
                            # Natural generation plus PCM16 serialization; declared before entry probes.
                            assert measured["relative_rmse"] <= 0.002, measured
                            assert measured["cosine"] >= 0.99999 and measured["max_abs"] <= 0.01, measured
                    if all(wavs.values()):
                        payload = {"input": "Hi.", "voice": "juno", "seed": 0, "temperature": 0.5}
                        assert http("/v1/audio/speech", {"input": "Hi."})[1] == wavs["juno"]
                        for patch, expected_status in [
                            ({"voice": "other"}, 400), ({"temperature": -1}, 400),
                            ({"seed": -1}, 422), ({"speed": 2}, 422), ({"model": "wrong"}, 400),
                            ({"input": " "}, 400), ({"input": "a" * 4097}, 400),
                            ({"response_format": "mp3"}, 400), ({"extra": True}, 422),
                        ]:
                            status, data, _ = http("/v1/audio/speech", {**payload, **patch})
                            assert status == expected_status, (patch, status, data)
                        for route in ("/v1/chat/completions", "/v1/embeddings", "/v1/jev"):
                            assert http(route, {})[0] == 404
                        assert http("/v1/audio/speech", payload)[1] == wavs["juno"]
                        row["checks"].extend(["CLI=HTTP WAV and PCM", "request reset", "invalid requests", "route isolation"])
                        if fmt == "f32" and backend == "cpu":
                            controlled = run("Hi.", "mira", "seed7-temp025",
                                             ["--seed", "7", "--temperature", "0.25"])
                            assert controlled != wavs["mira"]
                            assert http("/v1/audio/speech", {"input": "Hi.", "voice": "mira",
                                "seed": 7, "temperature": 0.25, "normalize": False})[1] == controlled
                            row["checks"].append("nondefault seed/temperature and normalize=false")
                            for voice in ("juno", "mira"):
                                data = run("Your order costs $12.50. It ships on March 3rd.", voice,
                                           f"split-{voice}")
                                ref = args.public_fixtures / f"public-f32-{voice}-split/waveform.f32"
                                measured = metrics(pcm(data) / 32767., np.clip(np.fromfile(ref, dtype="<f4"), -1, 1))
                                if measured["relative_rmse"] > 0.002 or measured["cosine"] < 0.99999:
                                    row["oracle_failures"].append({"case": f"split-{voice}", "metrics": measured,
                                        "threshold": {"relative_rmse": 0.002, "cosine": 0.99999}})
                                row["checks"].append({"normalized_multi_utterance": voice, "metrics": measured})
                            for extra in (["--speed", "2"], ["--kv-cache", "f16"],
                                          ["--language", "Chinese"], ["--max-tokens", "10"]):
                                assert run("Hi.", "juno", "bad", extra) is None, extra
                            with ThreadPoolExecutor(max_workers=4) as pool:
                                statuses = list(pool.map(lambda _: http("/v1/audio/speech",
                                    {**payload, "input": "Hello world. " * 10})[0], range(4)))
                            assert 200 in statuses and 429 in statuses and set(statuses) <= {200, 429}, statuses
                            assert http("/v1/audio/speech", payload)[1] == wavs["juno"]
                            row["checks"].append({"concurrency": statuses})
                except Exception as error:
                    row["errors"].append(str(error))
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            row["execution"] = re.findall(r"Underdog Woof: .*", log_path.read_text())
            row["entry_passed"] = not row["errors"] and all(wavs.values())
            row["passed"] = row["entry_passed"] and not row["oracle_failures"]
            rows.append(row)
            (args.output / "matrix.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(fmt, backend, "PASS" if row["passed"] else "FAIL",
                  row["errors"] + row["oracle_failures"], flush=True)
    assert all(row["passed"] for row in rows), "matrix includes failed/unverified entries; see matrix.json"


if __name__ == "__main__":
    main()
