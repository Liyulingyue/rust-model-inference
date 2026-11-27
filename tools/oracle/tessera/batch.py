"""Replay batched CLI, JSON and SSE output against the pinned author's sample_iter."""
import argparse
import base64
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from endpoints import image_pixels, request

PROMPT = "golden crown with blue gems"
OTHER = "a red apple with a leaf"
STEPS = 4
SEED = 17
OPTIONS = dict(top_k=17, top_p=.92, min_p=.01, polish_steps=1)


def reference(args, directory):
    import torch
    from reference import load_packed
    sys.path.insert(0, str((args.assets / "oracle").resolve()))
    from tessera.pipeline import load_model_from_checkpoint
    from tessera.sampler import sample_iter
    from tessera.text_encoder import FrozenTextEncoder
    from tessera.viz import tokens_to_images

    torch.set_num_threads(8)
    model, palette, _ = load_model_from_checkpoint(args.assets / "tessera-small-fp16.pt", device="cpu")
    encoder = FrozenTextEncoder(str(args.assets / "clip"), model.cfg.text_len)
    load_packed(args.assets / f"tessera-small-{args.fmt}.gguf", model, encoder)
    for name, prompts in [("repeat", [PROMPT, PROMPT]), ("mixed", [PROMPT, OTHER])]:
        states, pooled = encoder(encoder.tokenize(prompts))
        run = sample_iter(model, states, pooled, steps=STEPS, **OPTIONS,
                          generator=torch.Generator().manual_seed(SEED))
        for step in run:
            for i, image in enumerate(tokens_to_images(step.preview, palette)):
                image.save(directory / f"{name}-step-{step.index:03}-{i:03}.png")
        for i, image in enumerate(tokens_to_images(step.tokens, palette)):
            image.save(directory / f"{name}-{i:03}.png")


def stream(url, payload):
    req = urllib.request.Request(url, data=json.dumps(dict(payload, stream=True)).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as response:
        assert response.headers.get_content_type() == "text/event-stream"
        event = None
        for raw in response:
            name, _, value = raw.decode().strip().partition(":")
            if name == "event":
                event = value.strip()
            elif name == "data":
                yield event, json.loads(value)


def check_disconnect(url):
    payload = {"prompt": "fox", "steps": 32, "stream": True}
    first = stream(url, payload)
    event, _ = next(first)
    assert event == "image.preview"
    def status(payload):
        try:
            request(url, payload)
            return 200
        except urllib.error.HTTPError as exc:
            return exc.code
    try:
        assert status({"prompt": "x", "n": 0}) == 400
        assert status({"prompt": "x"}) == 429
    finally:
        first.close()
    # The blocking worker owns the generation slot until it sees the closed
    # preview channel. A fully kept canvas makes the retry cost bounded.
    until = time.monotonic() + 60
    while time.monotonic() < until:
        code = status({"prompt": "", "pixel_options": {"canvas": [0] * 256}})
        if code == 200:
            return
        assert code == 429
        time.sleep(.1)
    raise TimeoutError("generation slot still busy after disconnect")


def replay(args, oracle, run):
    base = ["--model", str(args.assets / f"tessera-small-{args.fmt}.gguf"),
            "--compute", args.policy, "--threads", "8"]
    options = run / "options.json"
    options.write_text(json.dumps(OPTIONS))
    with (run / "cli.log").open("w") as log:
        subprocess.run([args.bin_dir / "rust-model-inference", *base, "--prompt", PROMPT, "--steps",
                        str(STEPS), "--seed", str(SEED), "--num-samples", "2", "--input", options,
                        "--preview-dir", run, "--out", run / "repeat.png", "--overwrite"],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    with (run / "server.log").open("w") as log:
        server = subprocess.Popen([args.bin_dir / "rust-model-server", *base, "--port", str(port)],
                                  stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(600):
                if server.poll() is not None:
                    raise RuntimeError("server exited")
                try:
                    request(url + "/v1/models")
                    break
                except urllib.error.URLError:
                    time.sleep(.1)
            url += "/v1/images/generations"
            check_disconnect(url)
            # A finite CFG multiplier can overflow the guided logits. A
            # failed generation must end the stream without publishing a PNG.
            overflow = list(stream(url, {"prompt": PROMPT, "steps": 1, "guidance": 3.4028234e38}))
            assert len(overflow) == 1 and overflow[0][0] == "error", overflow
            for name, prompt, n in [("repeat", PROMPT, 2), ("mixed", [PROMPT, OTHER], 1)]:
                payload = dict(prompt=prompt, n=n, steps=STEPS, seed=SEED, pixel_options=OPTIONS)
                final = request(url, payload)["data"]
                seen = []
                matches = []
                def check(path, encoded):
                    data = base64.b64decode(encoded, validate=True)
                    if name == "repeat":
                        assert data == (run / path).read_bytes(), path
                    actual, expected = image_pixels(data), image_pixels((oracle / path).read_bytes())
                    matches.append(sum(a == b for a, b in zip(actual, expected)))
                for event, data in stream(url, payload):
                    if event == "image.preview":
                        step, i = data["step"], data["index"]
                        assert data["total_steps"] == STEPS + 1
                        assert data["phase"] == ("polish" if step == STEPS else "decode")
                        seen.append((step, i))
                        check(f"{name}-step-{step:03}-{i:03}.png", data["b64_json"])
                    else:
                        assert event == "image.completed" and data["data"] == final and len(final) == 2
                        seen.append("completed")
                        for i, item in enumerate(final):
                            check(f"{name}-{i:03}.png", item["b64_json"])
                assert seen == [(k,i) for k in range(STEPS + 1) for i in range(2)] + ["completed"]
                print(json.dumps(dict(case=name, policy=args.policy, format=args.fmt,
                                      preview_pixels=matches[:-2], final_pixels=matches[-2:])), flush=True)
        finally:
            server.terminate()
            server.wait(timeout=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fmt", default="f32")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--bin-dir", type=Path)
    parser.add_argument("--policy", default="cpu")
    args = parser.parse_args()
    oracle = args.output / args.fmt / "oracle"
    oracle.mkdir(exist_ok=True, parents=True)
    if args.reference:
        reference(args, oracle)
    if args.bin_dir:
        run = args.output / args.fmt / args.policy
        run.mkdir(exist_ok=True, parents=True)
        replay(args, oracle, run)
