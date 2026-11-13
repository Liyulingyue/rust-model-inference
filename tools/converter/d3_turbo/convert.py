"""Stream standalone d3 weights to GGUF, keeping at most a few matrix rows in RAM.

python -m tools.converter.d3_turbo.convert models/d3-turbo models/d3-turbo-q8_0.gguf --type q8_0
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools.converter.utils.gguf import (
    GGML_F32, GGML_F16, GGML_BF16, GgufWriter, bf16_to_f32, f32_to_bf16,
    gguf_dims, open_safetensors, quantize_q4_0, quantize_q4_1, quantize_q8_0,
)
from tools.converter.utils.kquants import quantize_k
from tools.converter.utils.quantize_gguf import BLOCK_FORMATS, FILE_TYPES

FORMATS = ("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q4_k", "q5_k", "q6_k")
CONFIG_FILES = ("config.json", "decision_config.json", "tokenizer.json", "tokenizer_config.json",
                "chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json")


def verify_manifest(root, names):
    path = root / "modelscope-files.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    files = {item["Path"]: item for item in manifest["files"]}
    for name in names:
        expected = files[name]
        source = root / name
        if source.stat().st_size != expected["Size"]:
            raise ValueError(f"source size differs from ModelScope manifest: {name}")
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected["Sha256"]:
            raise ValueError(f"source SHA256 differs from ModelScope manifest: {name}")
    return manifest


def encode_matrix(values, kind):
    if kind not in FORMATS:
        raise ValueError(f"unsupported d3 weight format: {kind}")
    values = np.asarray(values, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite d3 weights")
    if kind == "f32":
        return GGML_F32, values.tobytes()
    block, dtype = BLOCK_FORMATS[kind]
    if values.shape[-1] % block:
        return GGML_F32, values.tobytes()
    flat = values.ravel()
    if kind == "f16":
        converted = flat.astype("<f2")
        if not np.isfinite(converted).all():
            raise ValueError("d3 F16 export overflow")
        return dtype, converted.tobytes()
    if kind == "bf16":
        return dtype, f32_to_bf16(flat.tobytes())
    quantizer = {"q8_0": quantize_q8_0, "q4_0": quantize_q4_0, "q4_1": quantize_q4_1}.get(kind)
    return dtype, quantizer(flat) if quantizer else quantize_k(kind, flat)


def convert(root, output, kind, workers=1):
    root, output = Path(root), Path(output)
    config = json.loads((root / "config.json").read_text())
    decision = json.loads((root / "decision_config.json").read_text())
    if config.get("model_type") != "qwen3_5" or decision.get("format_id") != "d3-code-readout-v1" \
            or decision.get("attention_mode") != "prefix_bidir":
        raise ValueError("expected an explicit prefix_bidir d3 Qwen3.5 checkpoint")
    if kind not in FORMATS:
        raise ValueError(f"unsupported d3 weight format: {kind}")
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(index.values()))
    if any(Path(name).name != name for name in shards):
        raise ValueError("d3 shards must be local filenames")
    manifest = verify_manifest(root, [*shards, "readout.safetensors", "model.safetensors.index.json", *CONFIG_FILES])
    sources = [open_safetensors(root / name) for name in shards + ["readout.safetensors"]]
    owners = {}
    for source in sources:
        for name in source.header:
            if name == "__metadata__":
                continue
            if name in owners:
                raise ValueError(f"duplicate d3 tensor {name}")
            owners[name] = source
    if set(owners) != set(index) | {"weight"}:
        raise ValueError("d3 shard inventory differs from index/readout")
    hidden = config["text_config"]["hidden_size"]
    if owners["weight"].header["weight"]["shape"] != [255, hidden] or owners["weight"].header["weight"]["dtype"] != "F32":
        raise ValueError("d3 readout must use F32 storage with shape [255, hidden_size]")
    writer = GgufWriter(partial)
    writer.add_meta("general.architecture", "d3_turbo")
    writer.add_meta("general.name", "vllm-sr/d3-turbo")
    writer.add_meta("general.file_type", FILE_TYPES[kind])
    writer.add_meta("d3_turbo.compute_dtype", "float32")
    for name in CONFIG_FILES:
        data = (root / name).read_bytes()
        writer.add_meta("d3_turbo." + name, data.decode())
        writer.add_meta("d3_turbo." + name + ".sha256", hashlib.sha256(data).hexdigest())
    if manifest is not None:
        writer.add_meta("d3_turbo.source_revision", manifest["revision"])
        writer.add_meta("d3_turbo.source_manifest", json.dumps(manifest))
    source_types = {"F32": (GGML_F32, 4), "F16": (GGML_F16, 2), "BF16": (GGML_BF16, 2)}
    for name, source in owners.items():
        info = source.header[name]
        shape = tuple(info["shape"])
        if info["dtype"] not in source_types or not shape or not all(d > 0 for d in shape):
            raise ValueError(f"invalid d3 tensor {name}: {info}")
        source_type, element = source_types[info["dtype"]]
        matrix = name.startswith("language_model.") and len(shape) == 2
        rows = math.prod(shape[:-1])
        width = shape[-1]
        row_batch = max(1, (1 << 20) // (width * element))
        dtype, sample = encode_matrix(np.zeros((1, width), np.float32), kind if matrix else "f32")
        size = rows * len(sample)

        def chunks(name=name, source=source, shape=shape, width=width, rows=rows, element=element,
                   row_batch=row_batch, source_type=source_type, matrix=matrix):
            start, stop = source.header[name]["data_offsets"]
            if stop - start != math.prod(shape) * element:
                raise ValueError(f"invalid d3 tensor byte count: {name}")
            with source.path.open("rb") as stream:
                stream.seek(source.data_offset + start)
                pending = deque()
                for row in range(0, rows, row_batch):
                    count = min(row_batch, rows - row)
                    data = stream.read(count * width * element)
                    if len(data) != count * width * element:
                        raise ValueError(f"truncated d3 tensor: {name}")
                    values = (np.frombuffer(bf16_to_f32(data), dtype="<f4") if source_type == GGML_BF16 else
                              np.frombuffer(data, dtype="<f2" if source_type == GGML_F16 else "<f4").astype(np.float32))
                    target = kind if matrix else "f32"
                    if pool is not None and target in ("q4_k", "q5_k", "q6_k"):
                        pending.append(pool.submit(encode_matrix, values.reshape(count, width), target))
                        if len(pending) >= 2 * workers:
                            yield pending.popleft().result()[1]
                    else:
                        yield encode_matrix(values.reshape(count, width), target)[1]
                while pending:
                    yield pending.popleft().result()[1]

        writer.add_tensor_chunks(name, dtype, gguf_dims(shape), size, chunks)
    try:
        with (ProcessPoolExecutor(max_workers=workers) if workers > 1 and kind in ("q4_k", "q5_k", "q6_k") else nullcontext(None)) as pool:
            writer.write()
        partial.rename(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--type", choices=FORMATS, default="bf16")
    parser.add_argument("--workers", type=int, default=1, help="bounded parallel K-quant conversion")
    args = parser.parse_args()
    convert(args.model, args.output, args.type, args.workers)
