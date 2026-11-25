"""Replay native quantized projections with the exact captured input rows."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader, dequantize

from tools.oracle.aion.oracle import (
    projection_minima, projection_quantization, round_projection_input,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    checkpoint_rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
    reader = GGUFReader(args.gguf)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    largest = (0.0, None)
    largest_rmse = 0.0
    count = 0
    violations = []
    for inputs, outputs in zip(checkpoint_rows[::2], checkpoint_rows[1::2], strict=True):
        name = inputs["name"].removeprefix("aion.").removesuffix(".input")
        assert outputs["name"] == f"aion.{name}.output"
        assert (inputs["layer"], inputs["step"]) == (outputs["layer"], outputs["step"])
        key = f"blk.{inputs['layer']}.{name}.weight" if inputs["layer"] is not None else "token_embd.weight"
        tensor = tensors[key]
        x = torch.from_numpy(np.fromfile(inputs["binary_path"], dtype="<f4"))
        actual = np.fromfile(outputs["binary_path"], dtype="<f4")
        mode = projection_quantization(tensor)
        if mode is not None:
            x, correction = round_projection_input(x, mode)
        expected = np.empty_like(actual)
        for start in range(0, actual.size, 4096):
            end = min(start + 4096, actual.size)
            data = tensor.data[start:end]
            weight = torch.from_numpy(dequantize(data, tensor.tensor_type).reshape(end - start, x.numel()).copy())
            projected = torch.nn.functional.linear(x, weight)
            minima = projection_minima(tensor, data)
            if minima is not None:
                projected = projected + torch.nn.functional.linear(correction, minima)
            expected[start:end] = projected.numpy()
        error = actual.astype(np.float64) - expected
        scale = max(1.0, float(np.abs(expected).max()))
        max_abs = float(np.abs(error).max())
        rmse = float(np.sqrt(np.mean(error * error)))
        if max_abs / scale > largest[0]:
            largest = (max_abs / scale, key)
        largest_rmse = max(largest_rmse, rmse / scale)
        if max_abs / scale > 2e-4 or rmse / scale > 5e-5:
            violations.append(dict(tensor=key, step=inputs["step"], max_abs=max_abs, rmse=rmse))
        count += 1
    assert count >= 197, f"incomplete projection trace: {count}"
    result = dict(gguf=args.gguf.stem, count=count, max_scaled_abs=largest[0],
                  largest_error_tensor=largest[1], max_scaled_rmse=largest_rmse,
                  violations=violations)
    print(json.dumps(result), flush=True)
    assert not violations, "same-input projection replay exceeded the numeric budget"


if __name__ == "__main__":
    main()
