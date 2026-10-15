#!/usr/bin/env python3
"""Bit-exact comparison of the Rust Mellum trace against the pinned llama.cpp one.

Both traces are JSONL manifests of F32 sidecars:

* oracle  — `llama-eval-callback` writes `<path>.<name>[-<layer>].f32` with
  ``{"name","layer","step","shape","len","occurrence","binary_path"}``.
  Layer checkpoints are named ``<base>-<layer>``; `layer` is null for the
  global ones. The I32 ``ffn_moe_topk`` result is emitted as ``int_ids``.
* Rust    — ``RMI_PARITY_TRACE`` writes ``<path>.<name>.<occurrence>.f32``
  with ``{"name","layer","step","shape","len","occurrence","binary_path"}``
  (see `src/parity_trace.rs`). ``layer`` is the layer index or null.

This script indexes both by ``(name, layer, step)`` and compares the raw
little-endian F32 words. Any bit difference is reported with the first
divergent index so the search stops at the *first* divergence, per the
repo's parity contract (no tolerances, no cosine similarity).

Usage:
    python3 tools/oracle/mellum/check_scalar.py ORACLE_TRACE RUST_TRACE [--max-report N]
"""

import argparse
import json
import os
import struct
import sys
from collections import defaultdict


def _norm_oracle_name(name, layer):
    """Oracle layer checkpoints are named `<base>-<layer>`; strip the suffix."""
    if layer is not None and name.endswith(f"-{layer}"):
        return name[: -(len(str(layer)) + 1)]
    return name


def load_manifest(path, oracle_side):
    records = {}
    int_records = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "token_ids" in entry:
                records[(entry["name"], None, None)] = entry["token_ids"]
                continue
            layer = entry.get("layer")
            step = entry.get("step")
            name = entry["name"]
            if oracle_side:
                name = _norm_oracle_name(name, layer)
            key = (name, layer, step)
            if "int_ids" in entry:
                int_records[key] = entry["int_ids"]
                continue
            records[key] = entry
    return records, int_records


def read_f32(path):
    with open(path, "rb") as handle:
        data = handle.read()
    count = len(data) // 4
    return struct.unpack("<%df" % count, data), data


def load_tensor(entry):
    values, raw = read_f32(entry["binary_path"])
    return values, raw


def trim(values, entry, key):
    """`kq` / `kq_soft_max` are dumped at ggml's padded KV width.

    llama.cpp pads the unified KV cache so `ne[0]` is a multiple of 256 while
    only `step + 1` slots are live; the Rust side emits exactly the live
    window. Compare the first `step + 1` entries of each head.
    """
    name, _layer, step = key
    if name not in ("kq", "kq_soft_max"):
        return values
    live = step + 1
    stride = entry["shape"][0]
    if stride == live:
        return values
    return [values[head * stride + index] for head in range(len(values) // stride) for index in range(live)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle")
    parser.add_argument("rust")
    parser.add_argument("--max-report", type=int, default=20)
    args = parser.parse_args()

    oracle_records, oracle_ints = load_manifest(args.oracle, True)
    rust_records, rust_ints = load_manifest(args.rust, False)

    failures = []
    checked = 0
    bit_diffs = 0
    shown = 0

    # Token sequences must match first — a tokenizer divergence makes every
    # later number meaningless.
    for key in ("prompt_ids", "generated_ids"):
        oracle_ids = oracle_records.get((key, None, None))
        rust_ids = rust_records.get((key, None, None))
        if oracle_ids is None or rust_ids is None:
            continue
        if oracle_ids != rust_ids:
            print(f"[tokenizer] {key}: oracle={oracle_ids} rust={rust_ids}")
            failures.append(f"{key} token ids differ")
        else:
            print(f"[tokenizer] {key}: match ({len(oracle_ids)} ids)")

    all_keys = set(oracle_records) | set(rust_records)
    all_keys.discard(("prompt_ids", None, None))
    all_keys.discard(("generated_ids", None, None))
    # `ffn_moe_topk` is an int tensor on the oracle side; it is compared
    # separately below so drop it from the tensor loop.
    all_keys = {k for k in all_keys if k[0] != "ffn_moe_topk"}

    # Compare in graph order so the first reported divergence is the earliest.
    def sort_key(key):
        name, layer, step = key
        order = [
            "embd",
            "attn_norm",
            "Qcur_normed",
            "Kcur_normed",
            "Vcur",
            "Qcur",
            "Kcur",
            "kq",
            "kq_soft_max",
            "kqv",
            "ffn_inp",
            "ffn_norm",
            "ffn_moe_logits",
            "ffn_moe_probs",
            "ffn_moe_topk",
            "ffn_moe_weights",
            "ffn_moe_up",
            "ffn_moe_gate",
            "ffn_moe_swiglu",
            "ffn_moe_down",
            "ffn_moe_weighted",
            "ffn_moe_out",
            "ffn_out",
            "l_out",
            "result_norm",
            "result_output",
        ]
        try:
            base = order.index(name)
        except ValueError:
            base = len(order)
        return (step if step is not None else 0, base, layer if layer is not None else -1)

    for key in sorted(all_keys, key=sort_key):
        name, layer, step = key
        oracle_entry = oracle_records.get(key)
        rust_entry = rust_records.get(key)
        if oracle_entry is None:
            continue
        if rust_entry is None:
            failures.append(f"{key}: missing on the Rust side")
            continue
        oracle_values, oracle_raw = load_tensor(oracle_entry)
        rust_values, rust_raw = load_tensor(rust_entry)
        # `kq` / `kq_soft_max` are dumped at ggml's padded KV width; trim to the
        # live window before comparing so the shapes line up.
        if name in ("kq", "kq_soft_max"):
            oracle_values = trim(oracle_values, oracle_entry, key)
            rust_values = trim(rust_values, rust_entry, key)
        checked += 1
        if len(oracle_values) != len(rust_values):
            failures.append(
                f"{key}: length oracle={len(oracle_values)} rust={len(rust_values)}"
            )
            continue
        o_bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in oracle_values]
        r_bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in rust_values]
        if o_bits == r_bits:
            continue
        first = next((i for i in range(len(o_bits)) if o_bits[i] != r_bits[i]), None)
        bit_diffs += 1
        ndiff = sum(1 for i in range(len(o_bits)) if o_bits[i] != r_bits[i])
        if shown < args.max_report:
            shown += 1
            print(
                f"[diff] {key}: {ndiff}/{len(o_bits)} words differ; first at index {first} "
                f"(oracle=0x{o_bits[first]:08x} rust=0x{r_bits[first]:08x})"
            )
        failures.append(f"{key}: bitwise mismatch")

    # int records: oracle emits ffn_moe_topk as int_ids (I32); Rust emits the
    # same values as an f32 checkpoint (small ints round-trip exactly).
    for key in sorted(set(oracle_ints) | set(rust_ints)):
        name, layer, step = key
        o = oracle_ints.get(key)
        r_ids = None
        r_entry = rust_records.get(key)
        if r_entry is not None:
            r_values, _ = load_tensor(r_entry)
            r_ids = [int(v) for v in r_values]
        if o is None or r_ids is None:
            failures.append(f"{key}: topk present on only one side")
        elif o != r_ids:
            failures.append(f"{key}: oracle={o} rust={r_ids}")
        else:
            checked += 1


    print(f"\nchecked {checked} tensors, {bit_diffs} bitwise mismatches, {len(failures)} failures")
    if failures:
        print("\nFIRST divergences (fix these before anything else):")
        for item in failures[: args.max_report]:
            print(f"  - {item}")
        return 1
    print("PARITY OK: all compared tensors are bit-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
