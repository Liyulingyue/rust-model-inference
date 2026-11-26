"""Run the shipped CLI and HTTP server serially for one model/backend pair."""

import argparse
import base64
import json
import re
import subprocess
import time
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("bin_dir", "model", "mmproj", "red_png", "log_dir"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("policy", choices=("cpu", "auto", "vulkan"))
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--jev", action="store_true")
    parser.add_argument("--scores-only", action="store_true", help="exercise single and grouped JEV, skipping generation")
    parser.add_argument("--template-file", type=Path)
    parser.add_argument("--port", type=int, default=18550)
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    settings = ["--model", args.model, "--compute", args.policy, "--threads", "8",
                "--max-context", "512", "--prefill-batch-size", "16", "--jinja", "--no-thinking"]
    if args.template_file:
        settings.extend(["--chat-template-file", args.template_file])
    draft = ["--mtp-draft", "2"] if args.mtp else []
    media = ["--mmproj", args.mmproj, "--image", args.red_png]
    uri = "data:image/png;base64," + base64.b64encode(args.red_png.read_bytes()).decode()
    prompts = {"text": "Name the capital of France. Answer with one word.",
               "image": "What color is the image? Answer with one word."}
    result = {}
    groups = [{"label": "Hue", "options": ["red", "blue"]},
              {"label": "Brightness", "options": ["bright", "dark"]}]
    group_question = "Choose a hue and a brightness."
    group_flags = []
    for group in groups:
        group_flags.extend(["--jev-block", group["label"]])
        for option in group["options"]:
            group_flags.extend(["--jev-option", option])

    def record(key, value):
        result[key] = value
        print(json.dumps({key: value}, ensure_ascii=False), flush=True)
        (args.log_dir / "results.json").write_text(json.dumps(result, indent=2))

    def cli(key, flags):
        run = subprocess.run([args.bin_dir / "rust-model-inference", *settings, *flags],
                             capture_output=True, text=True, timeout=480)
        (args.log_dir / (key + ".log")).write_text(run.stdout + "\n" + run.stderr)
        run.check_returncode()
        return run

    for kind, prompt in prompts.items():
        if not args.scores_only:
            flags = media if kind == "image" else draft
            run = cli("cli_" + kind, [*flags, "--prompt", prompt, "--temperature", "0", "--max-tokens", "6"])
            answer = run.stdout.split("--- Generation ---\n", 1)[1].split("\n--- End ---", 1)[0].strip()
            record("cli_" + kind, answer)
            expected = "Paris" if kind == "text" else "Red"
            assert answer.rstrip(".").casefold() == expected.casefold(), (kind, expected, answer)
            if args.mtp and kind == "text":
                rounds = re.search(r"\[mtp\] (\d+) rounds, (\d+)/(\d+) drafts accepted", run.stderr)
                assert rounds and int(rounds[3]) > 0, run.stderr
                record("mtp_rounds_accepted_drafted", [int(n) for n in rounds.groups()])
        if args.jev or args.scores_only:
            for name, context, question, flags in [
                ("jev", "Red", "Color?", ["--jev-option", "red", "--jev-option", "blue"]),
                ("grouped", "Bright red.", group_question, group_flags),
            ]:
                key = f"cli_{name}_{kind}"
                run = cli(key, [*(media if kind == "image" else []), "--jev", "--jev-context", context,
                                "--jev-question", question, *flags, "--jev-output", "json"])
                record(key, next(json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")))

    opener = build_opener(ProxyHandler({}))

    def request(path, body=None):
        url = f"http://127.0.0.1:{args.port}/v1/{path}"
        data = None if body is None else json.dumps(body).encode()
        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=480) as response:
            return response.read().decode()

    with (args.log_dir / "server.log").open("w") as log:
        server = subprocess.Popen([args.bin_dir / "rust-model-server", *settings, *draft,
                                  "--mmproj", args.mmproj, "--host", "127.0.0.1",
                                  "--port", str(args.port)], stdout=log, stderr=log)
        try:
            for _ in range(240):
                if server.poll() is not None:
                    raise RuntimeError("server stopped during startup")
                try:
                    if json.loads(request("models"))["data"]:
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError("server did not start")
            if args.jev or args.scores_only:
                for kind in prompts:
                    for name, route, payload in [
                        ("jev", "image" if kind == "image" else "score",
                         {"context": "Red", "questions": [{"text": "Color?", "options": ["red", "blue"]}]}),
                        ("grouped", "image_grouped" if kind == "image" else "grouped",
                         {"context": "Bright red.", "mode": "block_choice",
                          "questions": [{"text": group_question, "groups": groups}]}),
                    ]:
                        if kind == "image":
                            payload["image_url"] = uri
                        score = json.loads(request("jev/" + route, payload))["results"][0]
                        record(f"http_{name}_{kind}", score)
                        other = result[f"cli_{name}_{kind}"]
                        for a, b in zip(score.get("groups", [score]), other.get("groups", [other]), strict=True):
                            assert a.get("label") == b.get("label") and a["choice"] == b["choice"]
                            x, y = a["probabilities"], b["probabilities"]
                            assert x.keys() == y.keys() and max(abs(x[k] - y[k]) for k in x) < 0.01
            if args.scores_only:
                return
            for kind, prompt in prompts.items():
                content = prompt if kind == "text" else [
                    {"type": "image_url", "image_url": {"url": uri}}, {"type": "text", "text": prompt}]
                body = {"messages": [{"role": "user", "content": content}], "temperature": 0,
                        "enable_thinking": False, "max_tokens": 6}
                reply = json.loads(request("chat/completions", body))
                answer = reply["choices"][0]["message"]["content"].strip()
                record("http_" + kind, answer)
                events = [line[6:] for line in request("chat/completions", dict(body, stream=True)).splitlines()
                          if line.startswith("data: ")]
                assert events[-1] == "[DONE]"
                chunks = (json.loads(event).get("choices", []) for event in events[:-1])
                streamed = "".join(c[0]["delta"].get("content", "") or "" for c in chunks if c).strip()
                record("sse_" + kind, streamed)
                assert streamed == answer == result["cli_" + kind], result

        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
