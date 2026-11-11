#!/usr/bin/env python3
"""Focused real Q8_0/Vulkan entry checks; keep evidence outside the main matrix."""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

from verify_entries import comparable, compare


def check_no_device(model_dir, binaries):
    output = model_dir / "extra-validation" / time.strftime("no-device-text-%Y%m%d-%H%M%S")
    output.mkdir(parents=True)
    missing = str(output / "missing-vulkan-icd.json")
    overrides = {"VK_DRIVER_FILES": missing, "VK_ICD_FILENAMES": missing, "RMI_COMPUTE_TRACE": "1"}
    env = dict(os.environ, **overrides)
    model = model_dir / "THX-02-Q8_0.gguf"
    request_path = model_dir / "request-smoke.json"
    baseline_path = model_dir / "entry-validation/Q8_0-cpu/native-text.stdout.json"
    baseline = json.loads(baseline_path.read_text())
    records = []
    summary = {"passed": False, "environment_overrides": overrides, "cpu_reference": str(baseline_path),
               "output": str(output), "not_run": ["no-device image inference", "thinker"]}

    def save():
        (output / "commands.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    def common(compute):
        return ["--model", str(model), "--compute", compute, "--threads", "4", "--max-context", "8192"]

    try:
        for compute in ["auto", "vulkan"]:
            command = [str(binaries / "rust-model-inference"), *common(compute), "--decision-request", str(request_path)]
            started = time.monotonic()
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=1800)
            (output / f"{compute}.stdout.json").write_text(result.stdout)
            (output / f"{compute}.stderr.log").write_text(result.stderr)
            records.append({"entry": "CLI", "compute": compute, "command": command,
                            "exit": result.returncode, "seconds": time.monotonic() - started})
            save()
            if compute == "auto":
                assert result.returncode == 0, result.stderr
                compare(comparable(baseline), comparable(json.loads(result.stdout)))
                assert not any(int(n) > 0 for n in re.findall(r"device_submissions=(\d+)", result.stderr))
            else:
                assert result.returncode != 0 and "vulkan" in result.stderr.lower(), result
        for compute in ["auto", "vulkan"]:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            command = [str(binaries / "rust-model-server"), *common(compute), "--host", "127.0.0.1", "--port", str(port)]
            log_path = output / f"{compute}.server.log"
            with log_path.open("w") as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
                record = {"entry": "HTTP server", "compute": compute, "command": command, "pid": process.pid}
                records.append(record)
                save()
                try:
                    deadline, listening = time.monotonic() + 300, False
                    while process.poll() is None:
                        try:
                            with socket.create_connection(("127.0.0.1", port), timeout=.2):
                                listening = True
                                break
                        except OSError:
                            assert time.monotonic() < deadline, "server startup timeout"
                            time.sleep(.1)
                    if not listening:
                        record["startup_exit"] = process.returncode
                        assert compute == "vulkan" and process.returncode != 0, record
                        assert "vulkan" in log_path.read_text().lower()
                        record["rejected_before_listening"] = True
                        continue
                    url = f"http://127.0.0.1:{port}/v1/decide"
                    request = urllib.request.Request(url, data=request_path.read_bytes(), headers={"Content-Type": "application/json"})
                    started = time.monotonic()
                    try:
                        with urllib.request.urlopen(request, timeout=1800) as response:
                            status, data = response.status, response.read()
                    except urllib.error.HTTPError as error:
                        status, data = error.code, error.read()
                    payload = json.loads(data)
                    (output / f"{compute}.http.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
                    records.append({"entry": "HTTP", "compute": compute, "url": url, "request_file": str(request_path),
                                    "status": status, "seconds": time.monotonic() - started})
                    save()
                    if compute == "auto":
                        assert status == 200, payload
                        compare(comparable(baseline), comparable(payload))
                    else:
                        assert status >= 400 and "vulkan" in json.dumps(payload).lower(), (status, payload)
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            trace = log_path.read_text()
            assert not any(int(n) > 0 for n in re.findall(r"device_submissions=(\d+)", trace))
        summary["passed"] = True
    except Exception as error:
        summary["error"] = repr(error)
        (output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        summary["records"] = len(records)
        save()
        print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--binaries", type=Path, default=Path("target/release-fast"))
    parser.add_argument("--no-device-only", action="store_true")
    args = parser.parse_args()
    model_dir, binaries = args.model_dir.resolve(), args.binaries.resolve()
    if args.no_device_only:
        check_no_device(model_dir, binaries)
        return
    output = model_dir / "extra-validation" / time.strftime("Q8_0-vulkan-%Y%m%d-%H%M%S")
    output.mkdir(parents=True)
    model = model_dir / "THX-02-Q8_0.gguf"
    env = dict(os.environ, RMI_COMPUTE_TRACE="1")
    records, checks = [], []
    server, server_log = None, None
    port = None
    summary = {"passed": False, "output": str(output), "invocation": sys.argv, "model": str(model)}
    summary["binaries"] = {
        name: {"sha256": hashlib.sha256((binaries / name).read_bytes()).hexdigest(),
               "mtime_ns": (binaries / name).stat().st_mtime_ns}
        for name in ["rust-model-inference", "rust-model-server"]
    }
    summary["model_size"] = model.stat().st_size
    summary["model_mtime_ns"] = model.stat().st_mtime_ns

    def save():
        (output / "commands.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
        (output / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2))
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    def passed(name, **detail):
        checks.append({"name": name, "passed": True, **detail})
        save()
        print(json.dumps(checks[-1]), flush=True)

    def common(context):
        return ["--model", str(model), "--compute", "vulkan", "--threads", "4", "--max-context", str(context)]

    def cli(name, flags, stdin=None, context=8192):
        cmd = [str(binaries / "rust-model-inference"), *common(context), *flags]
        start = time.monotonic()
        if stdin is not None:
            (output / f"{name}.stdin.json").write_text(stdin)
        result = subprocess.run(cmd, input=stdin, capture_output=True, text=True, env=env, timeout=1800)
        (output / f"{name}.stderr.log").write_text(result.stderr)
        (output / f"{name}.stdout.json").write_text(result.stdout)
        submissions = [int(n) for n in re.findall(r"device_submissions=(\d+)", result.stderr)]
        records.append({"entry": "CLI", "name": name, "command": cmd, "exit": result.returncode,
                        "seconds": time.monotonic() - start, "device_submissions": submissions})
        save()
        assert result.returncode == 0, (name, result.stderr[-2000:])
        assert submissions and all(n > 0 for n in submissions), (name, "missing Vulkan submissions")
        return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]

    def stop_server():
        nonlocal server, server_log
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            server = None
            server_log.close()
            server_log = None

    def start_server(context):
        nonlocal server, server_log, port
        assert server is None
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cmd = [str(binaries / "rust-model-server"), *common(context), "--host", "127.0.0.1", "--port", str(port)]
        server_log = (output / f"server-{context}.log").open("w")
        server = subprocess.Popen(cmd, stdout=server_log, stderr=server_log, env=env)
        records.append({"entry": "HTTP server", "command": cmd, "pid": server.pid})
        save()
        deadline = time.monotonic() + 300
        while True:
            assert server.poll() is None, "server exited before listening"
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=.2):
                    return
            except OSError:
                assert time.monotonic() < deadline, "server startup timeout"
                time.sleep(.1)

    def http(name, route, body, expected=200):
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        (output / f"{name}.request.json").write_bytes(payload)
        start = time.monotonic()
        request = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=payload,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                status, data = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, data = error.code, error.read()
        result = json.loads(data) if data.strip() else None
        (output / f"{name}.http.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        records.append({"entry": "HTTP", "name": name, "route": route, "status": status,
                        "seconds": time.monotonic() - start})
        save()
        if expected is not None:
            assert status == expected, (name, status, result)
        return status, result

    try:
        save()
        text = json.loads((model_dir / "request-smoke.json").read_text())
        image = json.loads((model_dir / "request-image.json").read_text())
        native_text = cli("native-file", ["--decision-request", str(model_dir / "request-smoke.json")])[0]
        compare(comparable(native_text), comparable(cli("native-stdin", ["--decision-request", "-"], json.dumps(text))[0]))
        passed("native file and stdin match")
        native_image = cli("native-data-image", ["--decision-request", str(model_dir / "request-image.json")])[0]
        local_image = dict(image, images=[str(model_dir / "oracle-colors.png")])
        compare(comparable(native_image), comparable(cli("native-local-image", ["--decision-request", "-"], json.dumps(local_image))[0]))
        passed("native data URL and local image match")

        jev_cases = []
        base = ["--jev", "--jev-output", "json"]
        for image_mode in [False, True]:
            context = "Inspect the supplied image." if image_mode else "The basket contains red apples and yellow bananas."
            for mode in ["multi_select", "block_choice"]:
                groups = ([{"label": "pair_1", "options": ["No red item", "Red item"]},
                           {"label": "pair_2", "options": ["No blue item", "Blue item"]}]
                          if mode == "multi_select" else
                          [{"label": "color", "options": ["Red:0", "Blue:10", "Green:20"]},
                           {"label": "shape", "options": ["Round", "Square"]}])
                question = "Classify the visible items."
                flags = [*base, "--jev-context", context, "--jev-question", question]
                if mode == "multi_select":
                    flags += ["--jev-multi"]
                for group in groups:
                    if mode == "block_choice":
                        flags += ["--jev-block", group["label"]]
                    for option in group["options"]:
                        flags += ["--jev-option", option]
                if image_mode:
                    flags += ["--image", str(model_dir / "oracle-colors.png")]
                body = {"context": context, "questions": [{"text": question, "groups": groups}], "mode": mode}
                if image_mode:
                    body["image_url"] = image["images"][0]
                name = f"jev-{'image-' if image_mode else ''}{mode}"
                jev_cases.append((name, "/v1/jev/image_grouped" if image_mode else "/v1/jev/grouped", body, cli(name, flags)))
        for name, image_mode, positive, options in [
            ("jev-score", False, None, ["No:0", "Yes:10"]),
            ("jev-positive", False, "A", ["No", "Yes"]),
            ("jev-image", True, None, ["Red", "Blue", "Green"]),
        ]:
            context = "Inspect the supplied image." if image_mode else "The package arrived."
            question = "Which color dominates?" if image_mode else "Has the package arrived?"
            flags = [*base, "--jev-context", context, "--jev-question", question]
            for option in options:
                flags += ["--jev-option", option]
            body = {"context": context, "questions": [{"text": question, "options": options}]}
            if positive:
                flags += ["--jev-positive", positive]
                body["positive"] = positive
            if image_mode:
                flags += ["--image", str(model_dir / "oracle-colors.png")]
                body["image_url"] = image["images"][0]
            jev_cases.append((name, "/v1/jev/image" if image_mode else "/v1/jev/score", body, cli(name, flags)))
        passed("all four JEV routes have real CLI references", cases=len(jev_cases))

        tta = {"state": "Alice has 2 apples and gets 3 more.", "questions": {"q": {
            "type": "choice", "instructions": "How many apples does Alice have?",
            "criteria": {"five": "5 apples", "six": "6 apples", "four": "4 apples"}}}, "tta": 4}
        tta_cli = cli("tta4", ["--decision-request", "-"], json.dumps(tta))[0]
        assert tta_cli["usage"]["sequences"] == 4
        split = json.loads((model_dir / "request-split-117.json").read_text())
        split_cli = cli("facet-split-117", ["--decision-request", "-"], json.dumps(split), context=117)[0]
        assert split_cli["usage"] == {"input_tokens": 208, "output_tokens": 0, "sequences": 2}, split_cli
        passed("CLI TTA and official 117-token facet boundary", split_usage=split_cli["usage"])

        start_server(8192)
        _, batch = http("native-batch", "/v1/decide/batch", {"requests": [text, image]})
        compare(comparable([native_text, native_image]), comparable(batch["results"]))
        passed("native HTTP batch matches CLI including image")
        for name, route, body, expected in jev_cases:
            _, result = http(name, route, body)
            compare(comparable(expected), comparable(result["results"]))
            passed(name + " HTTP matches CLI")

        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            pending = executor.submit(http, "tta4", "/v1/decide", tta)
            busy = None
            for attempt in range(200):
                time.sleep(.05)
                if pending.done():
                    break
                status, response = http(f"busy-probe-{attempt}", "/v1/decide", dict(text, stream=True), expected=None)
                if status == 429:
                    busy = response
                    break
                assert status == 400, (status, response)
            _, tta_http = pending.result()
        assert busy is not None, "no real busy response observed"
        compare(comparable(tta_cli), comparable(tta_http))
        passed("busy slot returns HTTP 429 while TTA request succeeds")
        criteria = tta["questions"]["q"]["criteria"]
        keys = list(criteria)
        orders = [[0, 1, 2], [2, 1, 0], [1, 2, 0], [2, 0, 1]]
        variants = []
        for order in orders:
            variant = copy.deepcopy(tta)
            variant["tta"] = 1
            variant["questions"]["q"]["criteria"] = {keys[i]: criteria[keys[i]] for i in order}
            variants.append(variant)
        _, permutations = http("tta-independent-orders", "/v1/decide/batch", {"requests": variants})
        geometric = {key: math.exp(sum(math.log(max(row["answers"]["q"]["probabilities"][key], 1e-9))
                                       for row in permutations["results"]) / 4) for key in keys}
        total = sum(geometric.values())
        geometric = {key: value / total for key, value in geometric.items()}
        compare(geometric, tta_http["answers"]["q"]["probabilities"])
        passed("TTA4 matches independent geometric mean", probabilities=geometric)

        invalids = [
            ("invalid-batch", "/v1/decide/batch", {"requests": [text, {"state": "bad", "questions": []}]}),
            ("truncated-image", "/v1/decide", dict(text, images=["data:image/png;base64,iVBORw0KGgo="])),
            ("bad-image-batch", "/v1/decide/batch", {"requests": [text, dict(text, images=["data:image/png;base64,iVBORw0KGgo="])]}),
            ("stream", "/v1/decide", dict(text, stream=True)),
            ("missing-thinker", "/v1/decide", dict(text, think=True)),
            ("missing-auto-thinker", "/v1/decide", dict(text, think="auto")),
            ("local-image", "/v1/decide", dict(text, images=[str(model_dir / "oracle-colors.png")])),
            ("remote-image-disabled", "/v1/decide", dict(text, images=["http://127.0.0.1/image.png"])),
            ("empty-batch", "/v1/decide/batch", {"requests": []}),
            ("bad-tta", "/v1/decide", dict(text, tta=0)),
            ("bad-temperature", "/v1/decide", dict(text, temperature=0)),
            ("nonfinite-json", "/v1/decide", b'{"state":1e999,"questions":{}}'),
            ("missing-jev-image", "/v1/jev/image", {"questions": [{"text": "q", "options": ["a", "b"]}]}),
        ]
        trace_before = (output / "server-8192.log").read_text().count("compute scope=thx02")
        for name, route, body in invalids:
            http(name, route, body, 400)
        http("unsupported-chat", "/v1/chat/completions", {"messages": []}, 404)
        trace_after = (output / "server-8192.log").read_text().count("compute scope=thx02")
        assert trace_before == trace_after, "an invalid request ran model inference"
        passed("invalid inputs return 400 before model forward; chat returns 404", cases=len(invalids) + 1)
        stop_server()

        start_server(117)
        _, split_http = http("facet-split-117", "/v1/decide", split)
        compare(comparable(split_cli), comparable(split_http))
        halves = []
        for subset in [["q0"], ["q1", "q2"]]:
            halves.append(dict(split, questions={key: split["questions"][key] for key in subset}))
        _, separate = http("official-split-halves", "/v1/decide/batch", {"requests": halves})
        answers = {key: answer for result in separate["results"] for key, answer in result["answers"].items()}
        compare(answers, split_http["answers"])
        assert [row["usage"]["input_tokens"] for row in separate["results"]] == [91, 117]
        passed("facet bisection matches official floor midpoint, CLI and independent halves", usage=split_http["usage"])
        long_request = dict(text, state="word " * 256)
        http("context-overflow", "/v1/decide", long_request, 400)
        _, after_error = http("context-error-recovery", "/v1/decide", text)
        compare(comparable(native_text), comparable(after_error))
        passed("context overflow returns 400 and releases the slot")
        stop_server()
        for context in [8192, 117]:
            trace = (output / f"server-{context}.log").read_text()
            submissions = [int(n) for n in re.findall(r"device_submissions=(\d+)", trace)]
            assert submissions and all(n > 0 for n in submissions)
            passed(f"HTTP context {context} used real Vulkan operators", requests=len(submissions), submissions=submissions)
        summary["passed"] = True
    except Exception as error:
        summary["error"] = repr(error)
        (output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        stop_server()
        summary["records"] = len(records)
        summary["checks"] = len(checks)
        save()
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
