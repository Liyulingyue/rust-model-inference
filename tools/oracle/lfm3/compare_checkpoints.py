#!/usr/bin/env python3
"""Join the pinned Oracle's F32 checkpoints to one RMI parity-trace manifest."""
import json
import ctypes
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

# Checkpoints the Oracle only exposes after an extra cb() is applied. The
# gated attention output needs the one in checkpoints.patch; the SwiGLU
# output lives inside the shared build_ffn(), so exposing it would touch
# every model's graph and it stays unprobed.
OPTIONAL_NAMES = {
    "attn_values": "lfm3_attn_gated",
}

# Every Rust checkpoint that feeds a quantized projection. Re-quantizing
# these with the Oracle's own quantizer shows where a small input
# difference turns into a different Q8_0 byte, which is where the numerical
# margin is thin. It measures margin, not parity: the residual it explains
# is the lossy part of quantisation itself, not a defect to chase.
PROJECTION_INPUTS = {
    "attn_norm",    # operator_norm -> q/k/v/head-gate projections
    "attn_values",  # gated attention output -> output projection
    "conv_conv",    # shortconv output -> shortconv output projection
    "ffn_norm",     # ffn_norm -> gate/up projections
    "ffn_swiglu",   # SwiGLU output -> down projection
    "result_norm",  # output norm -> output head
}


def shape(dims):
    dims = list(dims)
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()
    return dims


def load_quantizer(library_path):
    """Bind the pinned Oracle's CPU Q8_0 quantizer, never a model backend."""
    library = ctypes.CDLL(str(library_path))
    entry = library.quantize_row_q8_0
    entry.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64]
    entry.restype = None

    def quantize(values):
        count = len(values)
        out = ctypes.create_string_buffer(count // 32 * 34)
        entry((ctypes.c_float * count).from_buffer_copy(values), out, count)
        return out.raw

    return quantize


def compare(directory, quantizer_library=None, quantize=None):
    directory = Path(directory)
    if quantize is None and quantizer_library:
        quantize = load_quantizer(quantizer_library)
    amplification = []
    available = set()
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
        optional = row["name"] in OPTIONAL_NAMES
        if row["name"] not in NAMES and not optional:
            continue
        name = NAMES.get(row["name"]) or OPTIONAL_NAMES[row["name"]]
        if row["layer"] is not None:
            name += "-" + str(row["layer"])
        key = row["step"], name
        if optional and key not in oracle:
            # The Oracle was built without the extra cb(), so there is
            # nothing to compare against. The Rust side still traces it.
            continue
        assert key not in seen, f"duplicate Rust checkpoint: {key}"
        seen.add(key)
        assert oracle[key] == shape(row["shape"]), (key, oracle[key], row["shape"])
        actual = Path(row["binary_path"]).read_bytes()
        expected = (directory / f"{key[0]}-{key[1]}.f32").read_bytes()
        assert len(actual) == len(expected) == math.prod(row["shape"]) * 4, key
        if row["name"] == "embedding":
            assert actual == expected, f"lossless embedding differs: {key}"
            embedding_count += 1
        if row["name"] in PROJECTION_INPUTS:
            # The Oracle exposes this one, so its margin can be measured
            # whenever the two sides differ.
            available.add(row["name"])
        if actual == expected:
            continue
        a = struct.unpack(f"<{len(actual)//4}f", actual)
        b = struct.unpack(f"<{len(expected)//4}f", expected)
        assert all(math.isfinite(v) for v in (*a, *b)), key
        error = max(abs(x-y) for x, y in zip(a, b))
        if quantize and row["name"] in PROJECTION_INPUTS:
            n = len(a)
            assert n % 32 == 0, "Q8_0 input must be block aligned"
            qa = quantize(list(a))
            qb = quantize(list(b))
            amplification.append(dict(
                checkpoint=row["name"], layer=row["layer"], step=row["step"],
                input_max_abs=error,
                quantized_bytes_differ=sum(x != y for x, y in zip(qa, qb)),
            ))
        if first is None:
            first = (key, error)
        if error > worst[0]:
            worst = error, key
    projected_steps = {step for step, name in seen if name == "result_output"}
    known = set(NAMES.values()) | set(OPTIONAL_NAMES.values())
    expected = {key for key in oracle
                if key[1].rsplit("-", 1)[0] in known
                and (key[1] not in ("result_norm", "result_output") or key[0] in projected_steps)}
    assert seen == expected, f"checkpoint coverage differs: missing={expected-seen}, extra={seen-expected}"
    assert embedding_count > 0 and len(seen) > 200, "incomplete checkpoint coverage"
    result = dict(compared=len(seen), exact_embeddings=embedding_count, first_bit_difference=first, worst_abs=worst)
    if quantize:
        amplification.sort(key=lambda item: (-item["quantized_bytes_differ"], -item["input_max_abs"]))
        result["q8_amplification_points"] = amplification
        result["first_norm_q8_change"] = amplification[0] if amplification else None
        result["projection_inputs_available"] = sorted(available)
        # The Oracle exposes no checkpoint for these, so their margin stays
        # unmeasured until one is added.
        result["projection_inputs_without_oracle_checkpoint"] = sorted(
            set(PROJECTION_INPUTS) - available
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
