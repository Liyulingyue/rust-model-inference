#!/usr/bin/env python3
"""Join the pinned Oracle's F32 checkpoints to one RMI parity-trace manifest."""
import json
import math
from pathlib import Path
import struct
import sys

NAMES = {
    "embedding": "model.embed_tokens",
    "attn_norm": "model.layers.{}.operator_norm",
    "conv_in_proj": "model.layers.{}.conv.in_proj",
    "conv_conv": "model.layers.{}.conv.conv",
    "conv_out_proj": "model.layers.{}.conv.out_proj",
    "post_operator_norm": "model.layers.{}.post_operator_norm",
    "q_norm": "model.layers.{}.self_attn.q_layernorm",
    "k_norm": "model.layers.{}.self_attn.k_layernorm",
    "q_rope": "lfm3_q_rope",
    "k_rope": "lfm3_k_rope",
    "attn_gate": "model.layers.{}.self_attn.gate",
    "ffn_norm": "model.layers.{}.ffn_norm",
    "post_ffn_norm": "model.layers.{}.post_ffn_norm",
    "l_out": "l_out",
    "result_norm": "result_norm",
    "result_output": "result_output",
}


def shape(dims):
    dims = list(dims)
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()
    return dims


def compare(directory):
    directory = Path(directory)
    oracle = {}
    for line in (directory / "checkpoints.tsv").read_text().splitlines():
        step, name, *dims = line.split("\t")
        if " (" in name:
            continue
        key = int(step), name
        assert key not in oracle, f"duplicate reference checkpoint: {key}"
        oracle[key] = shape(map(int, dims))
    seen = set()
    first = None
    worst = (0.0, None)
    embedding_count = 0
    for line in (directory / "rust.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["name"] not in NAMES:
            continue
        name = NAMES[row["name"]]
        if row["layer"] is not None:
            name += "-" + str(row["layer"])
        key = row["step"], name
        assert key not in seen, f"duplicate Rust checkpoint: {key}"
        seen.add(key)
        assert oracle[key] == shape(row["shape"]), (key, oracle[key], row["shape"])
        actual = Path(row["binary_path"]).read_bytes()
        expected = (directory / f"{key[0]}-{key[1]}.f32").read_bytes()
        assert len(actual) == len(expected) == math.prod(row["shape"]) * 4, key
        if row["name"] == "embedding":
            assert actual == expected, f"lossless embedding differs: {key}"
            embedding_count += 1
        if actual == expected:
            continue
        a = struct.unpack(f"<{len(actual)//4}f", actual)
        b = struct.unpack(f"<{len(expected)//4}f", expected)
        assert all(math.isfinite(v) for v in (*a, *b)), key
        error = max(abs(x-y) for x, y in zip(a, b))
        if first is None:
            first = (key, error)
        if error > worst[0]:
            worst = error, key
    projected_steps = {step for step, name in seen if name == "result_output"}
    expected = {key for key in oracle
                if key[1].rsplit("-", 1)[0] in NAMES.values()
                and (key[1] not in ("result_norm", "result_output") or key[0] in projected_steps)}
    assert seen == expected, f"checkpoint coverage differs: missing={expected-seen}, extra={seen-expected}"
    assert embedding_count > 0 and len(seen) > 200, "incomplete checkpoint coverage"
    result = dict(compared=len(seen), exact_embeddings=embedding_count, first_bit_difference=first, worst_abs=worst)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    compare(sys.argv[1])
