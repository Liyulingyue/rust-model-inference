#!/usr/bin/env python3
"""Run real CLIs and compare canonical checkpoints as little-endian F32 bits."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import struct
import subprocess

FIXTURES = [
    "What is the capital of France?",
    " \t\n",
    "  Héllo\t世界! Café e\u0301\n🙂  [MASK] [SEP]",
    "The capital of Germany is Berlin.",
    "Photosynthesis converts light into chemical energy.",
    "hello world " * 40,
]


def run(command, trace, scalar=False):
    env = {**os.environ, "RMI_PARITY_TRACE": str(trace)}
    env.pop("RMI_PARITY_FILTER", None)
    if scalar:
        env["RMI_SCALAR"] = "1"
    with trace.with_suffix(".log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=log, check=True)
    return [json.loads(line) for line in trace.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model, oracle, rust = (str(p.resolve()) for p in (args.model, args.oracle, args.rust))
    summary = {"model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
               "model_bytes": args.model.stat().st_size,
               "oracle_commit": "b96806d96061049a5b574269b049bf6241d63d46",
               "oracle_binary_sha256": hashlib.sha256(Path(oracle).read_bytes()).hexdigest(),
               "rust_binary_sha256": hashlib.sha256(Path(rust).read_bytes()).hexdigest(),
               "platform": platform.platform(), "threads": 1, "fixtures": []}
    assert summary["model_sha256"] == "c4743c5cfcf2b1c6fe0bb548b13d8ecf828cfc9a8d2efd48df926f746e2dcd22", "unexpected model"
    for index, prompt in enumerate(FIXTURES):
        reference = run([oracle, "-m", model, "-p", prompt, "-t", "1", "-tb", "1",
                         "-ngl", "0", "-fa", "off", "-b", "512", "-ub", "512"],
                        args.output / f"oracle-{index}.jsonl")
        actual = run([rust, "--model", model, "--prompt", prompt, "--embedding",
                      "--threads", "1", "--embedding-output", "raw"],
                     args.output / f"rust-{index}.jsonl", scalar=True)
        assert len(reference) == 77, (index, "expected 76 tensors and token IDs")
        assert len(reference) == len(actual), (index, "checkpoint count", len(reference), len(actual))
        count = 0
        for expected, observed in zip(reference, actual):
            for field in ("name", "layer", "occurrence"):
                assert expected.get(field) == observed.get(field), (index, field, expected, observed)
            name = expected["name"]
            if "token_ids" in expected:
                assert expected["token_ids"] == observed["token_ids"], (index, "tokens", expected, observed)
                continue
            assert expected["shape"] == observed["shape"], (index, name, "shape", expected, observed)
            left = Path(expected["binary_path"]).read_bytes()
            right = Path(observed["binary_path"]).read_bytes()
            assert len(left) == len(right), (index, name, "length")
            if left != right:
                for offset in range(0, len(left), 4):
                    a, b = struct.unpack_from("<I", left, offset)[0], struct.unpack_from("<I", right, offset)[0]
                    if a != b:
                        raise AssertionError(f"fixture {index}, {name}, layer {expected.get('layer')}, "
                                             f"value {offset // 4}: oracle=0x{a:08x}, rust=0x{b:08x}")
            count += len(left) // 4
        summary["fixtures"].append({"prompt": prompt, "tokens": reference[0]["token_ids"],
                                    "checkpoints": len(reference) - 1, "f32_values": count})
        print(f"fixture {index}: {len(reference) - 1} checkpoints, {count} F32 values match", flush=True)
    (args.output / "verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
