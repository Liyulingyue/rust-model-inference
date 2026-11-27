"""Exercise the public CLI and both HTTP chat modes on a real Aion GGUF."""
import argparse
import json
import os
import re
import socket
import subprocess
import time
import urllib.request
from pathlib import Path


def api(port, path, body=None):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read().decode()


def wait_ready(server, port):
    deadline = time.monotonic() + 25
    while server.poll() is None and time.monotonic() < deadline:
        try:
            api(port, "/health")
            return
        except OSError:
            time.sleep(0.20)
    raise RuntimeError(f"server failed to become ready, exit={server.poll()}")


def device_receipt(log):
    data = log.read_text()
    counts = [int(n) for n in re.findall(r"submissions: (\d+),", data)]
    return {
        "vulkan_projections": data.count("scope=host_linear backend=Vulkan"),
        "cpu_projections": data.count("scope=host_linear backend=Cpu"),
        "vulkan_attention_calls": data.count("scope=aion_attention_kv backend=Vulkan"),
        "cpu_attention_calls": data.count("scope=aion_attention_kv backend=Cpu"),
        "device_submissions": max(counts, default=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--bin-dir", type=Path, default=Path("target/release-fast"))
    parser.add_argument("--compute", choices=("cpu", "auto", "vulkan"), default="cpu")
    parser.add_argument("--kv", choices=("f32", "f16"), default="f32")
    parser.add_argument("--jev", action="store_true")
    parser.add_argument("--tools", action="store_true")
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    common = ["--model", str(args.gguf), "--threads", "8", "--compute", args.compute,
              "--kv-cache", args.kv, "--prefill-batch-size", "64"]
    env = dict(os.environ, RMI_COMPUTE_TRACE="1")
    prompt = "Reply exactly with OK."
    request = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 3}
    cli_log = args.log_dir / "cli.log"
    with cli_log.open("w") as log:
        start = time.monotonic()
        cli = subprocess.run(
            [args.bin_dir / "rust-model-inference", *common, "--prompt", prompt,
             "--temp", "0", "--max-tokens", "3", "--profile"],
            env=env, stdout=subprocess.PIPE, stderr=log, text=True, timeout=180, check=True,
        ).stdout.rstrip("\n")
        cli_seconds = time.monotonic() - start
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server_log = args.log_dir / "server.log"
    with server_log.open("w") as log:
        server = subprocess.Popen(
            [args.bin_dir / "rust-model-server", *common, "--host", "127.0.0.1", "--port", str(port)],
            env=env, stdout=log, stderr=log,
        )
        try:
            wait_ready(server, port)
            start = time.monotonic()
            chat = json.loads(api(port, "/v1/chat/completions", request))
            http_seconds = time.monotonic() - start
            sse = api(port, "/v1/chat/completions", dict(request, stream=True))
            chunks = [json.loads(line[6:]) for line in sse.splitlines()
                      if line.startswith("data: ") and line[6:] != "[DONE]"]
            stream_text = "".join(
                chunk["choices"][0]["delta"].get("content") or "" for chunk in chunks if chunk["choices"]
            )
            choice = chat["choices"][0]
            assert sse.rstrip().endswith("data: [DONE]"), sse
            assert cli and cli == choice["message"]["content"] == stream_text, (cli, chat, sse)
            results = dict(format=args.gguf.stem, compute=args.compute, kv=args.kv, text=cli,
                           usage=chat["usage"], cli_seconds=cli_seconds, http_seconds=http_seconds)
            if args.jev:
                question = "What color is the sky?"
                context = "The sky is blue."
                options = ["Blue", "Orange"]
                scoring = json.loads(api(port, "/v1/jev/score", {
                    "context": context, "questions": [{"text": question, "options": options}],
                }))
                grouped = json.loads(api(port, "/v1/jev/grouped", {
                    "context": context, "mode": "block_choice",
                    "questions": [{"text": question, "groups": [{"label": "color", "options": options}]}],
                }))
                result = subprocess.run(
                    [args.bin_dir / "rust-model-inference", *common, "--jev",
                     "--jev-context", context, "--jev-question", question, "--jev-option", options[0],
                     "--jev-option", options[1], "--jev-output", "json"],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, timeout=180, check=True,
                )
                cli_score = json.loads(result.stdout)
                http_score = scoring["results"][0]
                for key in ("labels", "choice", "probabilities"):
                    assert cli_score[key] == http_score[key], (key, cli_score, http_score)
                grouped_cli = subprocess.run(
                    [args.bin_dir / "rust-model-inference", *common, "--jev",
                     "--jev-context", context, "--jev-question", question, "--jev-block", "color",
                     "--jev-option", options[0], "--jev-option", options[1], "--jev-output", "json"],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, timeout=180, check=True,
                )
                grouped_cli = json.loads(grouped_cli.stdout)
                grouped_http = grouped["results"][0]
                assert grouped_cli["groups"] == grouped_http["groups"], (grouped_cli, grouped_http)
                results["jev"] = {"cli": cli_score, "http": http_score,
                                  "grouped_cli": grouped_cli, "grouped_http": grouped_http}
            if args.tools:
                weather = {"type": "function", "function": {
                    "name": "weather", "description": "Get local weather.",
                    "parameters": {"type": "object", "required": ["city"],
                                   "properties": {"city": {"type": "string"}}},
                }}
                tool_request = {
                    "messages": [{"role": "user", "content": "What is the weather in Hangzhou?"}],
                    "tools": [weather], "tool_choice": "required", "temperature": 0, "max_tokens": 72,
                }
                response = json.loads(api(port, "/v1/chat/completions", tool_request))
                calls = response["choices"][0]["message"]["tool_calls"]
                assert len(calls) == 1 and calls[0]["function"]["name"] == "weather", response
                call_args = json.loads(calls[0]["function"]["arguments"])
                assert call_args["city"].lower() == "hangzhou", response
                results["tool_call"] = calls[0]["function"]
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    results["cli"] = device_receipt(cli_log)
    results["http"] = device_receipt(server_log)
    if args.compute == "vulkan":
        assert all(results[entry]["vulkan_projections"] > 0 and results[entry]["cpu_projections"] == 0
                   and results[entry]["vulkan_attention_calls"] > 0 and results[entry]["cpu_attention_calls"] == 0
                   for entry in ("cli", "http")), results
    (args.log_dir / "result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
