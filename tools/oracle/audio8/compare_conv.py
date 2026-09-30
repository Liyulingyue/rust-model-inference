"""Compare the official Audio8 first-frame graph to Rust using scalar F32 C."""

import argparse
import ctypes
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from safetensors import safe_open


def compare(left: np.ndarray, right: np.ndarray, name: str) -> None:
    assert left.shape == right.shape, (name, left.shape, right.shape)
    mismatch = np.flatnonzero(left.view("<u4") != right.view("<u4"))
    if mismatch.size:
        index = int(mismatch[0])
        raise AssertionError(
            f"{name}: first mismatch at {index}: "
            f"official=0x{int(left.view('<u4')[index]):08x}, "
            f"Rust=0x{int(right.view('<u4')[index]):08x}; "
            f"{mismatch.size}/{left.size} differing"
        )
    print(f"{name}: {left.size}/{left.size} raw F32 bits match")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="directory holding original model.safetensors")
    parser.add_argument("rust_trace", type=Path, help="RMI_PARITY_TRACE JSONL from the first window")
    parser.add_argument("--frame-trace", type=Path, help="optional focused first-frame Rust trace")
    parser.add_argument("--mel-f32", type=Path, help="time-major normalized Mel input for a focused trace")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.rust_trace.read_text().splitlines()]
    if args.frame_trace:
        records = [record for record in records if record["name"] not in
                   ("audio8.norm1", "audio8.layer_output")]
        records += [json.loads(line) for line in args.frame_trace.read_text().splitlines()]
    by_name = {record["name"]: record for record in records if record.get("occurrence") == 0}
    if args.mel_f32:
        mel = np.fromfile(args.mel_f32, dtype="<f4").reshape(-1, 128)
    else:
        record = by_name["asr.normalized_mel"]
        mel = np.fromfile(record["binary_path"], dtype="<f4")
        mel = mel.reshape(128, record["shape"][1]).T.copy()
    with tempfile.TemporaryDirectory(prefix="audio8-scalar-") as temporary:
        library = Path(temporary) / "conv.dylib"
        subprocess.run(
            ["clang", "-O2", "-ffp-contract=off", "-fno-vectorize", "-fno-slp-vectorize",
             "-shared", "-fPIC", str(Path(__file__).with_name("conv_scalar.c")), "-o", str(library)],
            check=True,
        )
        scalar = ctypes.CDLL(str(library))
        function = scalar.causal_conv_gelu
        pointer = ctypes.c_void_p
        function.argtypes = [pointer, pointer, pointer, pointer, ctypes.c_size_t,
                             ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        function.restype = None
        norm = scalar.rms_norm_f32
        norm.argtypes = [pointer, pointer, pointer, ctypes.c_size_t, ctypes.c_float]
        norm.restype = None
        linear = scalar.linear_f32
        linear.argtypes = [pointer, pointer, pointer, pointer, ctypes.c_size_t, ctypes.c_size_t]
        linear.restype = None
        add = scalar.add_f32
        add.argtypes = [pointer, pointer, ctypes.c_size_t]
        add.restype = None
        add_scalar = scalar.add_scalar_f32
        add_scalar.argtypes = [pointer, ctypes.c_float, ctypes.c_size_t]
        add_scalar.restype = None
        multiply = scalar.mul_f32
        multiply.argtypes = [pointer, pointer, ctypes.c_size_t]
        multiply.restype = None
        silu_mul = scalar.silu_mul_f32
        silu_mul.argtypes = [pointer, pointer, ctypes.c_size_t]
        silu_mul.restype = None
        gelu = scalar.gelu_f32
        gelu.argtypes = [pointer, ctypes.c_size_t]
        gelu.restype = None
        make_condition = scalar.time_condition_f32
        make_condition.argtypes = [pointer, pointer]
        make_condition.restype = None
        rope = scalar.rope_f32
        rope.argtypes = [pointer, ctypes.c_size_t]
        rope.restype = None
        attend = scalar.attention_f32
        attend.argtypes = [pointer, pointer, pointer, pointer, ctypes.c_size_t]
        attend.restype = None
        text_rope = scalar.rope_text_f32
        text_rope.argtypes = [pointer, ctypes.c_size_t, ctypes.c_size_t]
        text_rope.restype = None
        text_attend = scalar.attention_text_f32
        text_attend.argtypes = [pointer, pointer, pointer, pointer, ctypes.c_size_t]
        text_attend.restype = None

        with safe_open(args.source / "model.safetensors", framework="pt", device="cpu") as source:
            current = mel
            for layer, in_channels, stride in ((1, 128, 1), (2, 1280, 2)):
                prefix = f"audio_tower.embedder.conv{layer}"
                weight = source.get_tensor(f"{prefix}.weight").float().numpy().copy()
                bias = source.get_tensor(f"{prefix}.bias").float().numpy().copy()
                assert weight.shape == (1280, in_channels, 3)
                assert bias.shape == (1280,)
                out_frames = (len(current) + 3 - stride - 3) // stride + 1
                output = np.empty((out_frames, 1280), dtype="<f4")
                function(current.ctypes.data, weight.ctypes.data, bias.ctypes.data,
                         output.ctypes.data, len(current), in_channels, 1280, stride)
                name = f"audio8.conv{layer}"
                record = by_name[name]
                assert record["shape"] == list(output.shape)
                rust = np.fromfile(record["binary_path"], dtype="<f4")
                compare(output.reshape(-1), rust, name)
                current = output

            def tensor(name: str) -> np.ndarray:
                return source.get_tensor(name).float().numpy().copy()

            def normalized(values: np.ndarray, name: str, epsilon=1e-5) -> np.ndarray:
                weight = tensor(name)
                output = np.empty_like(values)
                norm(values.ctypes.data, weight.ctypes.data, output.ctypes.data,
                     values.size, ctypes.c_float(epsilon))
                return output

            def projected(values: np.ndarray, name: str, bias: bool = False) -> np.ndarray:
                weight = tensor(f"{name}.weight")
                offset = tensor(f"{name}.bias") if bias else None
                output = np.empty(weight.shape[0], dtype="<f4")
                linear(values.ctypes.data, weight.ctypes.data,
                       offset.ctypes.data if offset is not None else None,
                       output.ctypes.data, values.size, output.size)
                return output

            frame_records = {
                (record["name"], record.get("step") or 0, record.get("layer")): record
                for record in records if record["name"].startswith("audio8.")
            }

            def check(values: np.ndarray, name: str, position: int, layer=None) -> None:
                record = frame_records.get((name, position, layer))
                if position < 2:
                    assert record is not None, (name, position, layer)
                if record is not None:
                    rust = np.fromfile(record["binary_path"], dtype="<f4")
                    compare(values, rust, f"{name}.{position}.{layer}")

            projection_records = [record for record in records
                                  if record["name"] == "audio8.project2"]
            if args.mel_f32:
                assert len(projection_records) == len(mel) // 8
            positions = (range(4 * len(projection_records)) if projection_records else
                         sorted({step for name, step, _ in frame_records
                                 if name == "audio8.layer_output"}))
            keys = [[] for _ in range(32)]
            values = [[] for _ in range(32)]
            outputs = []
            for position in positions:
                hidden = current[position].copy()
                for layer in range(32):
                    prefix = f"audio_tower.layers.{layer}"
                    normal = normalized(hidden, f"{prefix}.self_attn_layer_norm.weight")
                    if layer == 0:
                        check(normal, "audio8.norm1", position, layer)
                    query = projected(normal, f"{prefix}.self_attn.q_proj", bias=True)
                    key = projected(normal, f"{prefix}.self_attn.k_proj")
                    value = projected(normal, f"{prefix}.self_attn.v_proj", bias=True)
                    rope(query.ctypes.data, position)
                    rope(key.ctypes.data, position)
                    if layer == 0:
                        check(query, "audio8.query", position, layer)
                        check(key, "audio8.key", position, layer)
                    keys[layer].append(key)
                    values[layer].append(value)
                    past_keys = np.stack(keys[layer])
                    past_values = np.stack(values[layer])
                    attention = np.empty(2048, dtype="<f4")
                    attend(query.ctypes.data, past_keys.ctypes.data,
                           past_values.ctypes.data, attention.ctypes.data,
                           len(keys[layer]))
                    if layer == 0:
                        check(attention, "audio8.attention", position, layer)
                    update = projected(attention, f"{prefix}.self_attn.o_proj", bias=True)
                    add(hidden.ctypes.data, update.ctypes.data, hidden.size)
                    normal = normalized(hidden, f"{prefix}.final_layer_norm.weight")
                    gate = projected(normal, f"{prefix}.mlp.gate_proj")
                    up = projected(normal, f"{prefix}.mlp.up_proj")
                    silu_mul(gate.ctypes.data, up.ctypes.data, gate.size)
                    down = projected(gate, f"{prefix}.mlp.down_proj", bias=True)
                    add(hidden.ctypes.data, down.ctypes.data, hidden.size)
                    check(hidden, "audio8.layer_output", position, layer)
                output = normalized(hidden, "audio_tower.norm.weight")
                check(output, "audio8.encoder_norm", position)
                outputs.append(output)
            projected_groups = []
            for group, record in enumerate(projection_records):
                padded = np.zeros(8 * 1280, dtype="<f4")
                padded[:4 * 1280] = np.concatenate(outputs[group * 4:(group + 1) * 4])
                output = projected(padded, "multi_modal_projector.linear_1")
                gelu(output.ctypes.data, output.size)
                output = projected(output, "multi_modal_projector.linear_2")
                compare(output, np.fromfile(record["binary_path"], dtype="<f4"),
                        f"audio8.project2.{group}")
                projected_groups.append(output)
            if "model.input_embed" in by_name:
                token_ids = ([151644, 151667] +
                             [151665] * max(0, len(projected_groups) - 2))[:len(projected_groups)]
                weights = source.get_slice("language_model.model.embed_tokens.weight")
                embeddings = []
                input_records = [record for record in records
                                 if record["name"] == "model.input_embed"]
                for token, (token_id, audio) in enumerate(zip(token_ids, projected_groups)):
                    embedding = weights[token_id].float().numpy().copy()
                    add(embedding.ctypes.data, audio.ctypes.data, embedding.size)
                    compare(embedding,
                            np.fromfile(input_records[token]["binary_path"], dtype="<f4"),
                            f"model.input_embed.{token}")
                    embeddings.append(embedding)
                frame_embedding = tensor("frame_len_embedding.weight")[0].copy()
                condition = np.empty(2048, dtype="<f4")
                make_condition(frame_embedding.ctypes.data, condition.ctypes.data)
                compare(condition,
                        np.fromfile(by_name["audio8.time_condition"]["binary_path"],
                                    dtype="<f4"), "audio8.time_condition")
                text_names = ("attn_norm-0", "Qcur-0", "Kcur-0", "kqv_out-0",
                              "ffn_out-0", "audio8.text_layer_output",
                              "result_norm", "result_output")
                text_records = {}
                for record in records:
                    if record["name"] in text_names:
                        row = record["occurrence"] // 36 if record["name"] == \
                            "audio8.text_layer_output" else record["occurrence"]
                        text_records[(record["name"], row, record.get("layer"))] = record

                def check_text(values: np.ndarray, name: str, row: int, layer=None) -> None:
                    record = text_records[(name, row, layer)]
                    compare(values, np.fromfile(record["binary_path"], dtype="<f4"),
                            f"{name}.{row}.{layer}")

                key_cache = [[] for _ in range(36)]
                value_cache = [[] for _ in range(36)]
                for token, embedding in enumerate(embeddings):
                    hidden = embedding
                    for layer in range(36):
                        prefix = f"language_model.model.layers.{layer}"
                        normal = normalized(hidden, f"{prefix}.input_layernorm.weight", 1e-6)
                        if layer == 0:
                            check_text(normal, "attn_norm-0", token, 0)
                        query = projected(normal, f"{prefix}.self_attn.q_proj", bias=True)
                        key = projected(normal, f"{prefix}.self_attn.k_proj", bias=True)
                        value = projected(normal, f"{prefix}.self_attn.v_proj", bias=True)
                        text_rope(query.ctypes.data, 16, token)
                        text_rope(key.ctypes.data, 2, token)
                        if layer == 0:
                            check_text(query, "Qcur-0", token, 0)
                            check_text(key, "Kcur-0", token, 0)
                        key_cache[layer].append(key)
                        value_cache[layer].append(value)
                        past_keys = np.stack(key_cache[layer])
                        past_values = np.stack(value_cache[layer])
                        attention = np.empty(2048, dtype="<f4")
                        text_attend(query.ctypes.data, past_keys.ctypes.data,
                                    past_values.ctypes.data, attention.ctypes.data, token + 1)
                        if layer == 0:
                            check_text(attention, "kqv_out-0", token, 0)
                        update = projected(attention, f"{prefix}.self_attn.o_proj")
                        add(hidden.ctypes.data, update.ctypes.data, hidden.size)
                        normal = normalized(hidden, f"{prefix}.post_attention_layernorm.weight", 1e-6)
                        scale = projected(condition, f"{prefix}.ada_rms_norm.linear1")
                        gelu(scale.ctypes.data, scale.size)
                        scale = projected(scale, f"{prefix}.ada_rms_norm.linear2")
                        add_scalar(scale.ctypes.data, ctypes.c_float(1.0), scale.size)
                        multiply(normal.ctypes.data, scale.ctypes.data, normal.size)
                        gate = projected(normal, f"{prefix}.mlp.gate_proj")
                        up = projected(normal, f"{prefix}.mlp.up_proj")
                        silu_mul(gate.ctypes.data, up.ctypes.data, gate.size)
                        down = projected(gate, f"{prefix}.mlp.down_proj")
                        if layer == 0:
                            check_text(down, "ffn_out-0", token, 0)
                        add(hidden.ctypes.data, down.ctypes.data, hidden.size)
                        check_text(hidden, "audio8.text_layer_output", token, layer)
                    normal = normalized(hidden, "language_model.model.norm.weight", 1e-6)
                    check_text(normal, "result_norm", token)
                    logits = np.empty(151936, dtype="<f4")
                    for start in range(0, logits.size, 4096):
                        end = min(start + 4096, logits.size)
                        chunk = weights[start:end].float().numpy().copy()
                        linear(normal.ctypes.data, chunk.ctypes.data, None,
                               logits[start:end].ctypes.data, normal.size, end - start)
                    check_text(logits, "result_output", token)


if __name__ == "__main__":
    main()
