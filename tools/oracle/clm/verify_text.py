"""Bitwise CLM text scoring check against scalar llama.cpp and the CLM head Oracle."""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path

from tools.oracle.clm.scalar import run as run_heads


CASES = (("hello", "hello world"), ("你好，世界", "hello world"))
STAGES = ("inp", "gelu1", "hidden", "norm", "gelu2", "out", "unit")


def command(args: list[str], env: dict[str, str] | None = None) -> str:
    result = subprocess.run(args, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed:\n{result.stderr[-4000:]}")
    return result.stdout


def equal_bits(name: str, left: bytes, right: bytes) -> int:
    if len(left) != len(right) or len(left) % 4:
        raise AssertionError(f"{name}: byte lengths {len(left)} != {len(right)}")
    if left != right:
        actual_words = struct.iter_unpack("<I", left)
        expected_words = struct.iter_unpack("<I", right)
        for index, (actual, expected) in enumerate(zip(actual_words, expected_words)):
            if actual != expected:
                raise AssertionError(f"{name}[{index}]: rust={actual[0]:08x} oracle={expected[0]:08x}")
    return len(left) // 4


def llama_embedding(llama: Path, encoder: Path, text: str, root: Path) -> tuple[bytes, bytes, list[int]]:
    outputs = []
    for normalize in (-1, 2):
        directory = root / str(normalize)
        directory.mkdir(parents=True)
        command([
            str(llama), "-m", str(encoder), "-p", text, "-t", "1", "-tb", "1",
            "-c", "32", "-b", "32", "-ub", "32", "-ngl", "0", "-fa", "off",
            "-ctk", "f32", "-ctv", "f32", "--no-repack", "--no-warmup",
            "--embedding", "--pooling", "last", "--embd-normalize", str(normalize),
            "--save-logits", "--logits-output-dir", str(directory),
        ])
        stem = f"llamacpp-{encoder.stem}-embeddings"
        data = (directory / f"{stem}.bin").read_bytes()
        tokens = (directory / f"{stem}-tokens.bin").read_bytes()
        outputs.append((data, list(struct.unpack(f"<{len(tokens) // 4}i", tokens))))
    if outputs[0][1] != outputs[1][1]:
        raise AssertionError(f"{text!r}: llama tokenization changed between runs")
    return outputs[0][0], outputs[1][0], outputs[0][1]


def verify(checkpoint: Path, heads: Path, encoder: Path, rust: Path, llama: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="clm-text-parity-") as directory:
        root = Path(directory)
        references = {
            text: llama_embedding(llama, encoder, text, root / f"prompt-{index}")
            for index, text in enumerate(dict.fromkeys(text for case in CASES for text in case))
        }
        encoder_words = head_words = 0
        for index, (state, action) in enumerate(CASES):
            trace = root / f"rust-{index}.jsonl"
            env = os.environ.copy()
            env.update(RMI_SCALAR="1", RMI_PARITY_TRACE=str(trace))
            names = [f"clm.{head}.{stage}" for head in ("state", "action") for stage in STAGES]
            names.append("clm.logits")
            env["RMI_PARITY_FILTER"] = ",".join(["clm.tokens", "clm.embedding.pooled", "clm.embedding.final", *names])
            stdout = command([
                str(rust), "--model", str(encoder), "--jev", "--clm-head", str(heads),
                "--jev-context", state, "--jev-question", "", "--jev-option", action,
                "--jev-option", state, "--threads", "1",
            ], env)
            tokens = [record["token_ids"] for record in map(json.loads, trace.read_text().splitlines())
                      if record["name"] == "clm.tokens"]
            if tokens != [references[state][2], references[action][2], references[state][2]]:
                raise AssertionError(f"case {index}: token IDs {tokens} differ from llama.cpp")
            for prompt_index, prompt in enumerate((state, action)):
                suffix = ".1" if prompt_index else ""
                for stage, reference in zip(("pooled", "final"), references[prompt][:2]):
                    actual = Path(f"{trace}.clm.embedding.{stage}{suffix}.f32").read_bytes()
                    encoder_words += equal_bits(f"case {index} {prompt} {stage}", actual, reference)

            oracle = root / f"oracle-{index}"
            state_file = root / f"state-{index}.f32"
            action_file = root / f"action-{index}.f32"
            state_file.write_bytes(references[state][1])
            action_file.write_bytes(references[action][1])
            run_heads(checkpoint, state_file, action_file, oracle)
            for name in names:
                head_words += equal_bits(
                    f"case {index} {name}",
                    Path(f"{trace}.{name}.f32").read_bytes(),
                    Path(f"{oracle}.{name}.f32").read_bytes(),
                )
            score_bits = struct.unpack("<I", Path(f"{trace}.clm.logits.f32").read_bytes())[0]
            if f"score={struct.unpack('<f', struct.pack('<I', score_bits))[0]:.4f}" not in stdout:
                raise AssertionError(f"case {index}: CLI score differs from trace: {stdout!r}")
            print(f"case {index}: tokens {tokens}, logit {score_bits:08x}")
        print(f"PASS: {encoder_words} encoder and {head_words} head F32 words bitwise identical")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "heads", "encoder", "rust", "llama"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    verify(*(getattr(args, name).resolve() for name in ("checkpoint", "heads", "encoder", "rust", "llama")))
