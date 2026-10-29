#!/usr/bin/env python3
"""Probe an already-running native OCR server with the pinned page fixture."""
import argparse
import base64
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def check_chat_stream(output):
    chunks, finishes, done = [], [], False
    for line in output.splitlines():
        if line == "data: [DONE]":
            done = True
        elif line.startswith("data: "):
            data = json.loads(line[6:])
            assert data.get("error") is None, data
            for choice in data.get("choices", []):
                chunks.append(choice.get("delta", {}).get("content", ""))
                if choice.get("finish_reason") is not None:
                    finishes.append(choice["finish_reason"])
    assert "".join(chunks) == "<PAGE>" and finishes == ["length"] and done, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []

    def request(name, path, body, expected=200):
        start = time.monotonic()
        req = urllib.request.Request(
            args.url.rstrip("/") + path,
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=480) as response:
                status, output = response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            status, output = error.code, error.read().decode()
        records.append(dict(name=name, path=path, status=status,
                            seconds=time.monotonic() - start, body=output))
        assert status == expected, (name, status, output)
        return output

    png = base64.b64encode((Path(__file__).parent / "fixtures/page-1.png").read_bytes()).decode()
    uri = "data:image/png;base64," + png
    chat = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": uri}}
    ]}], "max_tokens": 1}
    anthropic = {"messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}}
    ]}], "max_tokens": 1}
    try:
        for key, value in [("temperature", 0.5), ("top_p", 0.9),
                           ("enable_thinking", True), ("max_tokens", 131072)]:
            request(key, "/v1/chat/completions", chat | {key: value}, 400)
        request("no_image", "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "Parse"}]}, 400)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            running = pool.submit(request, "chat", "/v1/chat/completions", chat)
            time.sleep(2)
            request("concurrent", "/v1/chat/completions", chat, 429)
            result = json.loads(running.result())
        assert result["choices"][0]["message"]["content"] == "<PAGE>", result
        assert result["usage"]["completion_tokens"] == 1
        count = json.loads(request("count_tokens", "/v1/messages/count_tokens", anthropic))
        assert count["input_tokens"] == result["usage"]["prompt_tokens"], count
        result = json.loads(request("anthropic", "/v1/messages", anthropic))
        assert result["content"][0]["text"] == "<PAGE>", result
        output = request("chat_sse", "/v1/chat/completions", chat | {"stream": True})
        check_chat_stream(output)
        result = json.loads(request("responses", "/v1/responses", {
            "input": [{"role": "user", "content": [{"type": "input_image", "image_url": uri}]}],
            "max_output_tokens": 1, "store": False,
        }))
        assert result["output"][0]["content"][0]["text"] == "<PAGE>", result
        print(f"PASS: {len(records)} real HTTP requests")
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
