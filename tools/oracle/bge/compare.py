#!/usr/bin/env python3
"""Compare BGE embeddings with a fixed llama.cpp Oracle."""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
from pathlib import Path


PROMPTS = (
    "hello",
    "Hello, 世界! 123",
    "What is the capital of China?",
    "中国的首都是北京。",
)


def _bits(path: Path) -> list[int]:
    raw = path.read_bytes()
    if len(raw) % 4:
        raise ValueError(f"{path}: F32 file has {len(raw)} bytes")
    return list(struct.unpack(f"<{len(raw) // 4}I", raw))


def _tokens(path: Path) -> list[int]:
    raw = path.read_bytes()
    if len(raw) % 4:
        raise ValueError(f"{path}: token file has {len(raw)} bytes")
    return list(struct.unpack(f"<{len(raw) // 4}I", raw))


def _only(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"{path}: expected one {pattern}, found {len(matches)}")
    return matches[0]


def _run_oracle(
    oracle: Path, model: Path, prompt: str, output_dir: Path, normalize: int, pooling: str
) -> tuple[list[int], list[int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(oracle),
        "-m",
        str(model),
        "-p",
        prompt,
        "-t",
        "1",
        "-tb",
        "1",
        "-b",
        "2048",
        "-ub",
        "512",
        "-ngl",
        "0",
        "-fa",
        "off",
        "-ctk",
        "f32",
        "-ctv",
        "f32",
        "--embedding",
        "--pooling",
        pooling,
        "--embd-normalize",
        str(normalize),
        "--save-logits",
        "--logits-output-dir",
        str(output_dir),
        "--no-warmup",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            f"llama.cpp failed for {prompt!r} (normalize={normalize}):\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return (
        _bits(_only(output_dir, "*-embeddings.bin")),
        _tokens(_only(output_dir, "*-embeddings-tokens.bin")),
    )


def _run_rust(rust: Path, model: Path, prompt: str, trace: Path) -> tuple[list[int], list[int], list[int]]:
    for path in (trace, Path(f"{trace}.embedding.pooled.f32"), Path(f"{trace}.embedding.final.f32")):
        path.unlink(missing_ok=True)
    env = {
        **os.environ,
        "RMI_SCALAR": "1",
        "RMI_PARITY_TRACE": str(trace),
        "RMI_PARITY_FILTER": "embedding.tokens,embedding.pooled,embedding.final",
    }
    command = [
        str(rust),
        "--model",
        str(model),
        "--prompt",
        prompt,
        "--embedding",
        "--threads",
        "1",
        "--kv-cache",
        "f32",
        "--embedding-output",
        "raw",
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Rust failed for {prompt!r}:\n{result.stdout}\n{result.stderr}")
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    token_records = [record for record in records if record.get("name") == "embedding.tokens"]
    if len(token_records) != 1:
        raise RuntimeError(f"{trace}: expected one embedding.tokens record")
    return (
        token_records[0]["token_ids"],
        _bits(Path(f"{trace}.embedding.pooled.f32")),
        _bits(Path(f"{trace}.embedding.final.f32")),
    )


def _assert_equal(label: str, actual: list[int], expected: list[int]) -> None:
    if actual == expected:
        return
    limit = min(len(actual), len(expected))
    index = next((i for i in range(limit) if actual[i] != expected[i]), limit)
    raise AssertionError(
        f"{label} mismatch at {index}: Rust={actual[index:index + 3]} "
        f"Oracle={expected[index:index + 3]} (lengths {len(actual)}/{len(expected)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--pooling", choices=("cls", "last"), default="cls")
    parser.add_argument("--prompt", action="append", dest="prompts")
    args = parser.parse_args()
    prompts = tuple(args.prompts or PROMPTS)
    if not prompts:
        raise SystemExit("at least one --prompt is required")

    args.work_dir.mkdir(parents=True)
    for index, prompt in enumerate(prompts):
        raw, oracle_tokens = _run_oracle(
            args.oracle, args.model, prompt, args.work_dir / f"oracle-{index}-raw", -1, args.pooling
        )
        final, final_tokens = _run_oracle(
            args.oracle, args.model, prompt, args.work_dir / f"oracle-{index}-final", 2, args.pooling
        )
        if final_tokens != oracle_tokens:
            raise AssertionError(f"Oracle token IDs are not stable for {prompt!r}")
        rust_tokens, rust_raw, rust_final = _run_rust(
            args.rust, args.model, prompt, args.work_dir / f"rust-{index}.jsonl"
        )
        _assert_equal(f"{prompt!r} token IDs", rust_tokens, oracle_tokens)
        _assert_equal(f"{prompt!r} pooled F32 bits", rust_raw, raw)
        _assert_equal(f"{prompt!r} final F32 bits", rust_final, final)
        print(f"{index}: {prompt!r}: token IDs, pooled, final all bit-exact")


if __name__ == "__main__":
    main()
