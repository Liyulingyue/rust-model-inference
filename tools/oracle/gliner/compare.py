"""Fail unless two GLiNER scalar traces have identical tokens and F32 bits."""

import argparse
import json
import struct
from pathlib import Path


def compare(official: Path, rust: Path) -> None:
    reference = json.loads(official.read_text())
    if not reference.get("scalar_kernels"):
        raise ValueError("official trace was not produced with --scalar")
    actual = [json.loads(line) for line in rust.read_text().splitlines()]
    token_record = next((record for record in actual if record["name"] == "gliner.token_ids"), None)
    if token_record is None or token_record["token_ids"] != reference["token_ids"]:
        raise ValueError("token IDs differ")
    expected = {record["name"]: record for record in reference["checkpoints"]}
    observed_items = [
        (f"gliner.layer.{record['layer']}" if record["name"] == "gliner.layer" else record["name"], record)
        for record in actual
        if record.get("binary_path")
    ]
    observed = dict(observed_items)
    required = ["gliner.embeddings", "gliner.q", "gliner.k", "gliner.v", "gliner.context"]
    required += [f"gliner.layer.{index}" for index in range(24)]
    required += ["gliner.logits"]
    if [record["name"] for record in reference["checkpoints"]] != required:
        raise ValueError("official checkpoint order or count differs")
    if [name for name, _ in observed_items if name in required] != required:
        raise ValueError("Rust checkpoint order or count differs")
    words = 0
    for name in required:
        if name not in expected or name not in observed:
            raise ValueError(f"missing checkpoint {name}")
        oracle_shape = expected[name]["shape"]
        rust_shape = observed[name]["shape"]
        aligned_shape = oracle_shape[:-1] if name == "gliner.logits" else oracle_shape[-2:]
        if aligned_shape != rust_shape:
            raise ValueError(f"{name}: shape mismatch {oracle_shape} vs {rust_shape}")
        left = Path(expected[name]["binary_path"]).read_bytes()
        right = Path(observed[name]["binary_path"]).read_bytes()
        if len(left) != len(right):
            raise ValueError(f"{name}: shape mismatch {len(left)} vs {len(right)} bytes")
        words += len(left) // 4
        if left != right:
            first = next(index for index in range(0, len(left), 4) if left[index:index + 4] != right[index:index + 4])
            old = struct.unpack_from("<I", left, first)[0]
            new = struct.unpack_from("<I", right, first)[0]
            raise ValueError(f"{name}: first mismatch at F32[{first // 4}] oracle=0x{old:08x} rust=0x{new:08x}")
    print(f"PASS: token IDs and {len(required)} checkpoints, {words} F32 words bit-identical")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official", type=Path)
    parser.add_argument("rust", type=Path)
    args = parser.parse_args()
    compare(args.official, args.rust)


if __name__ == "__main__":
    main()
