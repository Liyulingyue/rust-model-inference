#!/usr/bin/env python3
"""Real CLI/HTTP checks; write every command, response and device trace under model-dir."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request


def comparable(value):
    if isinstance(value, dict):
        return {k: comparable(v) for k, v in value.items() if k not in {"latency_ms", "prefill_ms"}}
    if isinstance(value, list):
        return [comparable(v) for v in value]
    return value


def compare(a, b, path=""):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a:
            compare(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path + f"/{i}")
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert math.isfinite(a) and abs(a - b) <= 1e-5, (path, a, b)
    else:
        assert a == b, (path, a, b)


def check(model_dir, binaries, fmt, compute, extended):
    output = model_dir / "entry-validation" / f"{fmt}-{compute}"
    output.mkdir(parents=True, exist_ok=True)
    model = model_dir / f"THX-02-{fmt}.gguf"
    common = ["--model", str(model), "--compute", compute, "--threads", "4", "--max-context", "8192"]
    env = dict(os.environ, RMI_COMPUTE_TRACE="1")
    records = []

    def cli(name, flags, stdin=None):
        cmd = [str(binaries / "rust-model-inference"), *common, *flags]
        start = time.monotonic()
        result = subprocess.run(cmd, input=stdin, capture_output=True, text=True, env=env, timeout=1800)
        (output / f"{name}.stderr.log").write_text(result.stderr)
        (output / f"{name}.stdout.json").write_text(result.stdout)
        record = {"entry": "CLI", "name": name, "command": cmd, "seconds": time.monotonic() - start, "exit": result.returncode}
        records.append(record)
        assert result.returncode == 0, (name, result.stderr[-1500:])
        if compute in {"auto", "vulkan"}:
            assert "device_submissions=" in result.stderr, (name, "no actual device trace")
            assert "device_submissions=0 " not in result.stderr, (name, "no device submissions")
        return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]

    text = json.loads((model_dir / "request-smoke.json").read_text())
    image = json.loads((model_dir / "request-image.json").read_text())
    texts = cli("native-text", ["--decision-request", str(model_dir / "request-smoke.json")])[0]
    images = cli("native-image", ["--decision-request", str(model_dir / "request-image.json")])[0]
    score_options = ["Not delivered:0", "Delivered:10"]
    jev_args = ["--jev", "--jev-output", "json", "--jev-context", "The package arrived.", "--jev-question", "Has the package arrived?"]
    for option in score_options:
        jev_args += ["--jev-option", option]
    scores = cli("jev-score", jev_args)
    image_options = ["Mostly red", "Mostly blue"]
    jev_image_args = ["--jev", "--jev-output", "json", "--jev-context", "Classify the image.", "--jev-question", "What is the main color?", "--image", str(model_dir / "oracle-colors.png")]
    for option in image_options:
        jev_image_args += ["--jev-option", option]
    image_scores = cli("jev-image", jev_image_args)
    if extended:
        compare(comparable(texts), comparable(cli("native-stdin", ["--decision-request", "-"], json.dumps(text))[0]))
        local_image = copy.deepcopy(image)
        local_image["images"] = [str(model_dir / "oracle-colors.png")]
        compare(comparable(images), comparable(cli("native-local-image", ["--decision-request", "-"], json.dumps(local_image))[0]))

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    cmd = [str(binaries / "rust-model-server"), *common, "--host", "127.0.0.1", "--port", str(port)]
    server_log = (output / "server.log").open("w")
    server = subprocess.Popen(cmd, stdout=server_log, stderr=server_log, env=env)
    records.append({"entry": "HTTP server", "command": cmd})

    def http(name, route, body, expected=200):
        start = time.monotonic()
        request = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                status, data = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, data = error.code, error.read()
        result = json.loads(data) if data.strip() else None
        (output / f"{name}.http.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        records.append({"entry": "HTTP", "name": name, "route": route, "request": body, "status": status, "seconds": time.monotonic() - start})
        assert status == expected, (name, status, result)
        return result

    try:
        deadline = time.monotonic() + 300
        while True:
            assert server.poll() is None, "server exited before listening"
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=.2):
                    break
            except OSError:
                assert time.monotonic() < deadline, "server startup timeout"
                time.sleep(.1)
        batch = http("native-batch", "/v1/decide/batch", {"requests": [text, image]})
        compare(comparable([texts, images]), comparable(batch["results"]))
        request = {"context": "The package arrived.", "questions": [{"text": "Has the package arrived?", "options": score_options}]}
        compare(comparable(scores), comparable(http("jev-score", "/v1/jev/score", request)["results"]))
        request = {"context": "Classify the image.", "questions": [{"text": "What is the main color?", "options": image_options}], "image_url": image["images"][0]}
        compare(comparable(image_scores), comparable(http("jev-image", "/v1/jev/image", request)["results"]))
        if extended:
            compare(comparable(texts), comparable(http("native-single", "/v1/decide", text)))
            http("invalid-batch", "/v1/decide/batch", {"requests": [text, {"state": "invalid", "questions": []}]}, 400)
            http("stream", "/v1/decide", dict(text, stream=True), 400)
            http("missing-thinker", "/v1/decide", dict(text, think=True), 400)
            http("local-image", "/v1/decide", dict(text, images=[str(model_dir / "oracle-colors.png")]), 400)
            http("chat", "/v1/chat/completions", {"messages": []}, 404)
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        server_log.close()
        (output / "commands.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    return {"format": fmt, "compute": compute, "passed": True, "evidence": str(output), "records": len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--binaries", type=Path, default=Path("target/release-fast"))
    parser.add_argument("--formats", nargs="+", required=True)
    parser.add_argument("--computes", nargs="+", default=["cpu", "auto", "vulkan"])
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    summaries = []
    for fmt in args.formats:
        for compute in args.computes:
            try:
                result = check(args.model_dir.resolve(), args.binaries.resolve(), fmt, compute, args.extended)
            except Exception as error:
                result = {"format": fmt, "compute": compute, "passed": False, "error": repr(error)}
            summaries.append(result)
            print(json.dumps(result), flush=True)
            (args.model_dir / "entry-validation" / f"summary-{fmt}-{compute}.json").write_text(json.dumps(result, indent=2))
    assert all(row["passed"] for row in summaries), "entrypoint validation failed; see each evidence directory"


if __name__ == "__main__":
    main()
