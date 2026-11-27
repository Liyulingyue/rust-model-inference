"""Compare CLI and HTTP RGBA output to each other and to the same-weight PyTorch fixture."""
import argparse
import base64
import io
import json
import math
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


def request(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def image_pixels(data):
    with Image.open(io.BytesIO(data)) as im:
        assert im.size == (16, 16) and im.mode == "RGBA"
        raw = im.tobytes()
        return [raw[i:i + 4] for i in range(0, len(raw), 4)]


def run(args, fmt, policy, reference, options):
    case = args.output / fmt / policy
    case.mkdir(parents=True, exist_ok=True)
    prompt = "golden crown with blue gems"
    weight = args.assets / f"tessera-small-{fmt}.gguf"
    common = ["--model", str(weight), "--compute", policy, "--threads", str(args.threads)]
    # Give both public interfaces only the seed for the default sampler. This
    # also exercises their own defaults for steps, CFG, temperatures and NMS.
    cli = [str(args.bin_dir / "rust-model-inference"), *common, "--prompt", prompt,
           "--seed", "0", "--out", str(case / "cli.png"), "--overwrite"]
    payload = {"prompt": prompt, "seed": 0, "size": "16x16"}
    if not args.default_sample:
        steps, guidance = (1, 1) if args.inpaint else (4, 3)
        settings = case / "options.json"
        settings.write_text(json.dumps(options))
        cli.extend(["--steps", str(steps), "--guidance", str(guidance), "--temperature", "0",
                    "--input", str(settings)])
        payload.update(steps=steps, guidance=guidance, temperature=0, pixel_options=options)
    start = time.monotonic()
    with (case / "cli.log").open("w") as log:
        subprocess.run(cli, stdout=log, stderr=subprocess.STDOUT, check=True)
    cli_time = time.monotonic() - start

    if args.repeat:
        replay = cli.copy()
        replay[replay.index("--out") + 1] = str(case / "cli-repeat.png")
        with (case / "cli-repeat.log").open("w") as log:
            subprocess.run(replay, stdout=log, stderr=subprocess.STDOUT, check=True)

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    with (case / "server.log").open("w") as log:
        server = subprocess.Popen([str(args.bin_dir / "rust-model-server"), *common,
                                   "--host", "127.0.0.1", "--port", str(port)],
                                  stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(1200):
                if server.poll() is not None:
                    raise RuntimeError(f"server exited {server.returncode}: {case / 'server.log'}")
                try:
                    request(url + "/v1/models")
                    break
                except urllib.error.URLError:
                    time.sleep(0.1)
            else:
                raise TimeoutError("waiting for /v1/models")
            start = time.monotonic()
            response = request(url + "/v1/images/generations", payload)
            http_time = time.monotonic() - start
            png = base64.b64decode(response["data"][0]["b64_json"], validate=True)
            (case / "http.png").write_bytes(png)
            if args.repeat:
                replay = request(url + "/v1/images/generations", payload)
                (case / "http-repeat.png").write_bytes(base64.b64decode(replay["data"][0]["b64_json"], validate=True))
        finally:
            server.terminate()
            server.wait(timeout=30)
    cli_img = image_pixels((case / "cli.png").read_bytes())
    http_img = image_pixels(png)
    oracle = image_pixels(reference.read_bytes())
    result = {"format": fmt, "policy": policy, "cli_s": round(cli_time, 3), "http_s": round(http_time, 3),
            "cli_http_pixels": sum(a == b for a, b in zip(cli_img, http_img)),
            "cli_oracle_pixels": sum(a == b for a, b in zip(cli_img, oracle)),
            "http_oracle_pixels": sum(a == b for a, b in zip(http_img, oracle))}
    if args.repeat:
        for label, first in (("cli", cli_img), ("http", http_img)):
            replay_pixels = image_pixels((case / f"{label}-repeat.png").read_bytes())
            result[f"{label}_repeat_pixels"] = sum(a == b for a, b in zip(first, replay_pixels))
    if args.default_sample:
        # Composite over black so transparent pixels do not contribute arbitrary RGB.
        squared_error = sum((c * actual[3] / 255. - r * expected[3] / 255.) ** 2
                            for actual, expected in zip(cli_img, oracle)
                            for c, r in zip(actual[:3], expected[:3]))
        result["rgb_rmse_255"] = round(math.sqrt(squared_error / (3 * len(oracle))), 4)
        intersection = sum(a[3] > 0 and b[3] > 0 for a, b in zip(cli_img, oracle))
        union = sum(a[3] > 0 or b[3] > 0 for a, b in zip(cli_img, oracle))
        result["alpha_iou"] = round(intersection / union, 6) if union else 1.
    if args.inpaint:
        kept = [i for i, pixel in enumerate(options["canvas"]) if pixel is not None]
        generated = [i for i, pixel in enumerate(options["canvas"]) if pixel is None]
        for label, image in (("cli", cli_img), ("http", http_img)):
            result[f"{label}_kept_of_{len(kept)}"] = sum(image[i] == oracle[i] for i in kept)
            result[f"{label}_generated_of_{len(generated)}"] = sum(image[i] == oracle[i] for i in generated)
            if result[f"{label}_kept_of_{len(kept)}"] != len(kept):
                raise RuntimeError(f"kept pixels changed: {result}")
    if args.default_sample or args.inpaint:
        for label, log in (("cli", "cli.log"), ("http", "server.log")):
            counts = re.search(r"Tessera denoiser projections: cpu=(\d+) vulkan=(\d+)", (case / log).read_text())
            if counts is None:
                raise RuntimeError(f"missing projection counts in {case / log}")
            result[f"{label}_cpu_gpu"] = [int(n) for n in counts.groups()]
    if cli_img != http_img:
        raise RuntimeError(f"CLI and HTTP disagree: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("bin_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--formats", nargs="+", default="f32 f16 bf16 q8_0 q4_0 q4_1 q4_k q5_k q6_k".split())
    parser.add_argument("--policies", nargs="+", default=["cpu", "auto", "vulkan"])
    parser.add_argument("--threads", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--default-sample", action="store_true", help="compare the official seed-zero stochastic 16-step preset")
    mode.add_argument("--inpaint", action="store_true", help="repaint four pixels of the published-weight sample")
    parser.add_argument("--repeat", action="store_true", help="repeat on a fresh CLI process and on the same HTTP server")
    args = parser.parse_args()
    for fmt in args.formats:
        if args.inpaint:
            from inpaint import small_fixture
            reference, options = small_fixture(args.assets, fmt, args.output / fmt)
        elif args.default_sample:
            reference = args.assets / "parity" / fmt / "default.png"
            options = {}
        else:
            reference = args.assets / "parity" / fmt / "greedy.png"
            options = {"choice_temperature": 0}
        for policy in args.policies:
            result = run(args, fmt, policy, reference, options)
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
