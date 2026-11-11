"""Independent sampled export-degradation check; needs torch, safetensors and gguf."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize
from safetensors import safe_open
import torch

from tools.converter.sophon_realtime.convert import (
    FORMATS,
    REVISION,
    Reader,
    decoder_specs,
    target_type,
)
from tools.converter.utils.gguf import _read_gguf, _tensor_nbytes, gguf_dims

# Calibrated on real layer-0 Q projection before whole-export checks; not task accuracy.
RELATIVE_RMSE_BUDGET = dict(
    f32=0.0,
    f16=0.0005,
    bf16=0.005,
    q8_0=0.01,
    q4_0=0.15,
    q4_1=0.15,
    q4_k=0.12,
    q5_k=0.08,
    q6_k=0.04,
)


def verify(model_dir: Path, output: Path, quant: str, *, hash_output=True):
    root = Path(model_dir)
    if (root / "thinker" / "config.json").is_file():
        root = root / "thinker"
    metadata, directory = _read_gguf(output)
    component = metadata.get("sophon_realtime.component")
    if component == "audio":
        from tools.converter.sophon_realtime.audio import audio_specs

        config = json.loads((root / "audio" / "config.json").read_text())
        reader = Reader(root / "audio")
        specs = list(audio_specs(config))
        architecture = "sophon_realtime_audio"
    else:
        config = json.loads((root / "config.json").read_text())
        reader = Reader(root)
        specs = list(decoder_specs(config))
        architecture = "sophon_realtime"
    if (
        metadata.get("general.architecture") != architecture
        or component not in ("thinker", "audio")
        or metadata.get("sophon_realtime.source_revision") != REVISION
    ):
        raise ValueError("not a matching Sophon export")
    if set(directory) != {target for _, target, _ in specs}:
        raise ValueError("tensor inventory mismatch")
    elements = 0
    squared_error = 0.0
    source_squared = 0.0
    output_squared = 0.0
    dot = 0.0
    maximum = 0.0
    maximum_tensor_relative = 0.0
    budget = RELATIVE_RMSE_BUDGET[quant]
    with Path(output).open("rb") as stream:
        for name, target, shape in specs:
            kind, dims, _, offset = directory[target]
            if kind != target_type(shape, quant) or dims != gguf_dims(shape):
                raise ValueError(f"{target}: wrong format or shape")
            rows = (
                range(1)
                if len(shape) == 1
                else sorted({0, shape[0] // 2, shape[0] - 1})
            )
            tensor_error = 0.0
            tensor_source = 0.0
            tensor_maximum = 0.0
            source, _ = reader.entries[name]
            with safe_open(source.path, framework="pt", device="cpu") as reference:
                for row in rows:
                    value = (
                        reference.get_tensor(name).float()
                        if len(shape) == 1
                        else reference.get_slice(name)[row : row + 1].float()
                    )
                    if reader.dtype(name) == "F8_E4M3":
                        value = (
                            value
                            * reference.get_slice(name + "_scale")[
                                row : row + 1
                            ].float()
                        )
                    expected = value.numpy().reshape(-1)
                    length = _tensor_nbytes(kind, (expected.size,))
                    stream.seek(offset + (0 if len(shape) == 1 else row * length))
                    raw = np.frombuffer(stream.read(length), dtype=np.uint8)
                    actual = dequantize(raw, GGMLQuantizationType(kind)).reshape(-1)
                    if not np.all(np.isfinite(actual)):
                        raise ValueError(f"{target}: nonfinite export")
                    if kind == GGMLQuantizationType.F32 or quant in ("f16", "bf16"):
                        rounded = (
                            value.to(
                                {
                                    "f32": torch.float32,
                                    "f16": torch.float16,
                                    "bf16": torch.bfloat16,
                                }[quant]
                            )
                            .float()
                            .numpy()
                            .reshape(-1)
                            if kind != 0
                            else expected
                        )
                        if not np.array_equal(actual, rounded):
                            raise ValueError(
                                f"{target}: independent float casting mismatch"
                            )
                    a = actual.astype(np.float64)
                    b = expected.astype(np.float64)
                    delta = a - b
                    error = float(np.sum(delta * delta))
                    norm = float(np.sum(b * b))
                    squared_error += error
                    source_squared += norm
                    output_squared += float(np.sum(a * a))
                    dot += float(np.sum(a * b))
                    tensor_error += error
                    tensor_source += norm
                    tensor_maximum = max(tensor_maximum, float(np.max(np.abs(delta))))
                    maximum = max(maximum, tensor_maximum)
                    elements += a.size
            if tensor_source == 0.0 and tensor_error > 0.0:
                raise ValueError(f"{target}: export degradation for zero source")
            tensor_relative = (
                math.sqrt(tensor_error / tensor_source) if tensor_source else 0.0
            )
            if tensor_relative > budget:
                raise ValueError(
                    f"{target}: export degradation relative RMSE {tensor_relative} "
                    f"exceeds {budget}; max_abs={tensor_maximum}"
                )
            maximum_tensor_relative = max(maximum_tensor_relative, tensor_relative)
    relative = math.sqrt(squared_error / source_squared) if source_squared else 0.0
    if relative > budget:
        raise ValueError(
            f"export degradation relative RMSE {relative} exceeds {budget}"
        )
    result = dict(
        status="PASS",
        component=component,
        format=quant,
        tensors=len(directory),
        sampled_elements=elements,
        max_abs=maximum,
        rmse=math.sqrt(squared_error / elements),
        relative_rmse=relative,
        relative_rmse_budget=budget,
        tensor_relative_rmse_budget=budget,
        max_tensor_relative_rmse=maximum_tensor_relative,
        cosine=(
            dot / math.sqrt(source_squared * output_squared)
            if source_squared and output_squared
            else 1.0
        ),
        bytes=Path(output).stat().st_size,
        source_revision=REVISION,
        reference="safetensors.safe_open PyTorch F32 dequantization; independent gguf.quants payload decoding",
        scope="full vectors and first/middle/last matrix rows; export degradation only",
    )
    print(json.dumps(result), file=sys.stderr, flush=True)
    if hash_output:
        start = time.monotonic()
        with Path(output).open("rb") as source:
            result["sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
        result["hash_seconds"] = time.monotonic() - start
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quant", choices=FORMATS, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.model_dir, args.output, args.quant), indent=2))


if __name__ == "__main__":
    main()
