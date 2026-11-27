#!/usr/bin/env python3
"""Compare exact F32 operators against a pinned, scalar GGML build."""

import argparse
import hashlib
import json
import os
import random
import struct
import subprocess
from pathlib import Path


def compare(oracle: Path, rust: Path, work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=False)
    rng = random.Random(172)
    total = 0
    cases = []
    for n in [1, 3, 16, 32, 64, 256, 384, 768, 1024, 2048, 4096]:
        for kind in ["ordinary", "constant", "small", "cancellation"]:
            x = [rng.randrange(-1024, 1025) / 128 for _ in range(n)]
            if kind == "constant":
                x = [0.125] * n
            if kind == "small":
                x = [v * 2**-16 for v in x]
            if kind == "cancellation":
                x = [(-1 if i % 2 else 1) * (2**16 + (i % 3) * 2**-7) for i in range(n)]
            matrix = [rng.randrange(-1024, 1025) / 128 for _ in range(n * 5)]
            weight = [rng.randrange(1, 256) / 128 for _ in range(n)]
            bias = [rng.randrange(-128, 129) / 128 for _ in range(n)]
            other = [rng.randrange(-128, 129) / 128 for _ in range(n)]
            fixture = work / f"{n}-{kind}.bin"
            values = x + matrix + weight + bias + other
            fixture.write_bytes(
                struct.pack("<II", n, 5) + struct.pack(f"<{len(values)}f", *values)
            )
            outputs = []
            for binary in [oracle, rust]:
                result = subprocess.run(
                    [str(binary.resolve()), str(fixture.resolve())],
                    env={**os.environ, "RMI_SCALAR": "1"},
                    capture_output=True,
                    text=True,
                    check=True,
                )
                outputs.append(json.loads(result.stdout))
            if outputs[0].keys() != outputs[1].keys():
                raise AssertionError("operator inventory differs")
            count = 0
            for op, expected in outputs[0].items():
                actual = outputs[1][op]
                if len(expected) != len(actual):
                    raise AssertionError(f"{n}/{kind}/{op}: shape differs")
                for i, (a, b) in enumerate(zip(expected, actual)):
                    if a != b:
                        raise AssertionError(
                            f"{n}/{kind}/{op}[{i}]: oracle={a:#010x}, rust={b:#010x}"
                        )
                count += len(expected)
            total += count
            cases.append(
                dict(
                    width=n,
                    case=kind,
                    values=count,
                    sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                )
            )
    report = dict(
        cases=cases,
        values_compared=total,
        operators=list(outputs[0]),
        skipped=[
            "quantization",
            "F16/BF16 activation rounding",
            "GELU/GELU-quick F16 tables",
            "approximate SIMD",
            "FMA",
            "F16 KV",
            "pipeline output parity",
        ],
    )
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.oracle, args.rust, args.work_dir)
    print(
        f"Matched {len(report['cases'])} same-input cases, {report['values_compared']} raw F32 values across {len(report['operators'])} operators."
    )
