#!/usr/bin/env python3
"""Bit-exact comparison of the Rust lfm2moe trace against the pinned llama.cpp one.

Both traces are JSONL manifests of F32 sidecars:

* oracle  -- `llama-eval-callback` writes `<path>.<name>.f32` with
  ``{"name","layer","step","shape","len","occurrence","binary_path"}``.
  The I32 `ffn_moe_topk` node is emitted under the same key with the ids
  written as raw little-endian I32.
* Rust    -- `RMI_PARITY_TRACE` writes `<path>.<name>.<occurrence>.f32` with
  the same fields (see `src/parity_trace.rs`).

This script indexes both by ``(name, layer, step)`` and compares the raw
little-endian F32 words. Any bit difference is reported with the first
divergent index so the search stops at the *first* divergence, per the repo's
parity contract (no tolerances, no cosine similarity).

`kq` / `kq_soft_max` are trimmed on the oracle side by the harness itself
(ggml pads the KV width to a multiple of 256), so both manifests already hold
the live `step + 1` entries per head.

Usage:
    python3 tools/oracle/lfm2moe/check_scalar.py ORACLE_TRACE RUST_TRACE [--max-report N]
"""

import argparse
import json
import struct
import sys

# Graph order, so the first reported divergence is the earliest one.
ORDER = [
    "embd",
    "attn_norm",
    "q_proj", "k_proj", "v_proj",
    "q_norm", "k_norm",
    "q_rope", "k_rope",
    "attn_scores", "attn_probs", "attn_values",
    "attn_proj",
    "conv_in_proj", "conv_conv", "conv_out_proj",
    "post_attn_residual",
    "ffn_norm",
    "ffn_up", "ffn_gate", "ffn_swiglu", "ffn_down",
    "ffn_moe_logits", "ffn_moe_probs", "ffn_moe_probs_biased", "ffn_moe_topk",
    "ffn_moe_weights", "ffn_moe_weights_sum", "ffn_moe_weights_sum_clamped",
    "ffn_moe_weights_norm", "ffn_moe_up", "ffn_moe_gate", "ffn_moe_swiglu",
    "ffn_moe_down", "ffn_moe_weighted", "ffn_moe_out",
    "l_out",
    "result_norm", "result_output",
]


def load_manifest(path):
    tensors = {}
    tokens = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "token_ids" in entry:
                tokens[entry["name"]] = entry["token_ids"]
                continue
            if "int_ids" in entry or "binary_path" not in entry:
                continue
            tensors[(entry["name"], entry.get("layer"), entry.get("step"))] = entry
    return tensors, tokens


def read_words(path, name="", oracle_side=False):
    with open(path, "rb") as handle:
        data = handle.read()
    # `ffn_moe_topk` is an I32 node on the oracle side and an F32 checkpoint on
    # the Rust side; expert ids round-trip exactly through f32, so compare the
    # integer values rather than the raw words. Everything else is F32 on both
    # sides and is compared as raw bits.
    if name == "ffn_moe_topk":
        if oracle_side:
            return [int(v) for v in struct.unpack("<%di" % (len(data) // 4), data)]
        return [int(round(v)) for v in struct.unpack("<%df" % (len(data) // 4), data)]
    return list(struct.unpack("<%dI" % (len(data) // 4), data))


def sort_key(key):
    name, layer, step = key
    base = ORDER.index(name) if name in ORDER else len(ORDER)
    return (step if step is not None else 0, base, layer if layer is not None else -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle")
    parser.add_argument("rust")
    parser.add_argument("--max-report", type=int, default=20)
    parser.add_argument("--layers", type=int, default=0, help="only compare the first N layers")
    args = parser.parse_args()

    oracle_tensors, oracle_tokens = load_manifest(args.oracle)
    rust_tensors, rust_tokens = load_manifest(args.rust)

    failures = []
    checked = 0
    bit_diffs = 0
    shown = 0
    skipped_embd = 0

    for name in ("prompt_ids", "generated_ids"):
        if name not in oracle_tokens and name not in rust_tokens:
            continue
        if oracle_tokens.get(name) != rust_tokens.get(name):
            print(f"[tokenizer] {name}: oracle={oracle_tokens.get(name)} rust={rust_tokens.get(name)}")
            failures.append(f"{name} token ids differ")
        else:
            print(f"[tokenizer] {name}: match ({len(oracle_tokens[name])} ids)")

    keys = set(oracle_tensors) | set(rust_tensors)
    for key in sorted(keys, key=sort_key):
        name, layer, step = key
        if args.layers and layer is not None and layer >= args.layers:
            continue
        oracle_entry = oracle_tensors.get(key)
        rust_entry = rust_tensors.get(key)
        if oracle_entry is None:
            if name == "embd":
                # llama.cpp materialises the get_rows embedding as a graph
                # input, so the eval callback never fires for it and the oracle
                # trace has no record. The value is still pinned indirectly:
                # layer 0's `attn_norm` is RMSNorm over exactly this buffer and
                # it matches bit for bit on every step.
                skipped_embd += 1
                continue
            failures.append(f"{key}: missing on the oracle side")
            continue
        if rust_entry is None:
            failures.append(f"{key}: missing on the Rust side")
            continue
        oracle_words = read_words(oracle_entry["binary_path"], name, oracle_side=True)
        rust_words = read_words(rust_entry["binary_path"], name, oracle_side=False)
        checked += 1
        if len(oracle_words) != len(rust_words):
            failures.append(
                f"{key}: length oracle={len(oracle_words)} rust={len(rust_words)}"
            )
            continue
        if oracle_words == rust_words:
            continue
        first = next(i for i in range(len(oracle_words)) if oracle_words[i] != rust_words[i])
        bit_diffs += 1
        ndiff = sum(1 for i in range(len(oracle_words)) if oracle_words[i] != rust_words[i])
        if shown < args.max_report:
            shown += 1
            if name == "ffn_moe_topk":
                print(f"[diff] {key}: {ndiff}/{len(oracle_words)} experts differ; "
                      f"first at index {first} (oracle={oracle_words[first]} rust={rust_words[first]})")
            else:
                print(
                    f"[diff] {key}: {ndiff}/{len(oracle_words)} words differ; first at index {first} "
                    f"(oracle=0x{oracle_words[first]:08x} rust=0x{rust_words[first]:08x})"
                )
        failures.append(f"{key}: bitwise mismatch")

    print(f"\nchecked {checked} tensors, {bit_diffs} bitwise mismatches, {len(failures)} failures"
          + (f" ({skipped_embd} `embd` records skipped: graph input, not a callback node)"
             if skipped_embd else ""))
    if failures:
        print("\nFIRST divergences (fix these before anything else):")
        for item in failures[: args.max_report]:
            print(f"  - {item}")
        return 1
    print("PARITY OK: all compared tensors are bit-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
