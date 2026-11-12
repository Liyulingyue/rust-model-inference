#!/usr/bin/env python3
"""Compare the Rust `qwen35moe` text tower against the pinned llama.cpp oracle.

The MoE routing is the part that can falsify a claim: the same experts, in
the same order, with the same weights. This script runs both sides over the
same token ids and compares the MoE checkpoints and the final logits.

Usage:
    # 1. oracle (llama.cpp at the pinned commit, built with the qwen35 trace
    #    and qwen35moe ffn-dump patches)
    # 2. rust (tests/minicpmv_reference.rs::dump_qwen35moe_logits)
    tools/oracle/minicpmv/compare_text.py \
        --model $MODEL --oracle $ORACLE --rust-test $RUST_TEST --out $OUT

Exit code 0 means every compared checkpoint is bit-identical.
"""

import argparse
import json
import math
import os
import struct
import subprocess
from collections import defaultdict
from pathlib import Path

# The MoE checkpoints, the final norm and the logits. The router ids
# (`ffn_moe_topk`) are the ones that must match exactly; the rest are the
# same shape on both sides.
FILTER = ",".join(
    [
        "model.input_embed",
        "result_norm",
        "result_output",
        "ffn_moe_logits",
        "ffn_moe_probs",
        "ffn_moe_topk",
        "ffn_moe_weights",
        "ffn_moe_weights_norm",
        "ffn_moe_out",
        "ffn_moe_input",
        "ffn_shexp_gate",
        "ffn_shexp_gated",
        "ffn_moe_combined",
        "ffn_moe_scale",
        "l_out",
    ]
)

PROMPT = "user\nWhat is 2+2?\nassistant\n<think>\n\n</think>\n\n"


def run(command, log, env=None):
    with log.open("w", encoding="utf-8") as out:
        subprocess.run(
            command,
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
        )


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def values(record):
    raw = Path(record["binary_path"]).read_bytes()
    return struct.unpack("<%df" % record["len"], raw[: 4 * record["len"]])


def step_of(record):
    """The token index a record belongs to.

    The Rust side labels the per-step checkpoints with `step` and counts the
    final norm and the logits by `occurrence` only; the oracle labels
    everything with `step`.
    """
    step = record.get("step")
    return record.get("occurrence") or 0 if step is None else step


def by_name(path):
    """The checkpoints of one side, keyed by (name, layer, step).

    The layer is normalized so that the final norm and the logits -- which
    one side calls layer `None` and the other `-1` -- end up together.
    """
    groups = {}
    for record in records(path):
        if "binary_path" not in record:
            continue
        layer = record.get("layer")
        key = (record["name"], -1 if layer is None else layer, step_of(record))
        groups[key] = record
    return groups


def prompt_ids(path):
    """The token ids the oracle tokenized the prompt into."""
    ids = []
    for record in records(path):
        if record["name"].endswith("prompt_ids"):
            ids.extend(record["token_ids"])
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--rust-test", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    oracle_trace = args.out / "oracle.jsonl"
    env = dict(
        os.environ,
        RMI_PARITY_TRACE=str(oracle_trace),
        RMI_PARITY_FILTER=FILTER,
    )
    run(
        [
            str(args.oracle),
            "-m",
            str(args.model),
            "--no-repack",
            "-p",
            PROMPT,
            "--no-escape",
            "-n",
            "4",
            "-c",
            "128",
            "-b",
            "1",
            "-ub",
            "1",
            "-t",
            "8",
            "-tb",
            "8",
            "-ngl",
            "0",
            "-ctk",
            "f32",
            "-ctv",
            "f32",
            "-fa",
            "off",
            "--temp",
            "0",
            "--repeat-penalty",
            "1",
        ],
        args.out / "oracle.log",
        env,
    )

    # The Rust side takes the token ids explicitly: the repository's CLI wraps
    # a prompt in the GGUF chat template, the oracle tokenizes the raw text.
    ids = prompt_ids(oracle_trace)
    assert ids, "the oracle trace has no prompt ids"
    env = dict(
        os.environ,
        RMI_PARITY_TRACE=str(args.out / "rust.jsonl"),
        RMI_PARITY_FILTER=FILTER,
        RMI_MOE_LOGITS_MODEL=str(args.model),
        RMI_MOE_LOGITS_OUT=str(args.out / "logits.bin"),
        RMI_MOE_LOGITS_PROMPT_IDS=",".join(str(value) for value in ids),
    )
    run(
        [
            str(args.rust_test),
            "--ignored",
            "--exact",
            "dump_qwen35moe_logits",
            "--nocapture",
        ],
        args.out / "rust.log",
        env,
    )

    rust = by_name(args.out / "rust.jsonl")
    oracle = by_name(oracle_trace)
    summary = []
    for key in sorted(set(rust) & set(oracle)):
        a, b = rust[key], oracle[key]
        assert a["len"] == b["len"], (key, a["len"], b["len"])
        x, y = values(a), values(b)
        relative = 0.0
        if x != y:
            numerator = math.sqrt(sum((p - q) ** 2 for p, q in zip(x, y)))
            denominator = math.sqrt(sum(q * q for q in y))
            relative = numerator / denominator if denominator > 0 else 0.0
        summary.append(
            {
                "name": key[0],
                "layer": key[1],
                "step": key[2],
                "exact": x == y,
                "max_relative_l2": relative,
            }
        )
        assert x == y, (key, relative)
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    names = sorted({entry["name"] for entry in summary})
    for name in names:
        count = sum(1 for entry in summary if entry["name"] == name)
        print("%-20s %3d checkpoints, all bit-identical" % (name, count))
    print("%d checkpoints in total, all bit-identical" % len(summary))
    print("ALL DONE")


if __name__ == "__main__":
    main()
