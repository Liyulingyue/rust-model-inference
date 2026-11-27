#!/usr/bin/env python3
"""Isolate weight decoding and CPU activation rounding from recurrent drift."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vela3"))
from gguf_weights import Weights, activation_roundtrip, q4_1_min_correction


def main(args):
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    record = next(json.loads(line) for line in args.native.read_text().splitlines()
                  if json.loads(line).get("name") == "vela3.d3.input_norm")
    values = np.fromfile(record["binary_path"], dtype="<f4").reshape(record["shape"])
    assert values.ndim == 2 and np.isfinite(values).all()
    input_path = args.out / "input.f32"
    values.tofile(input_path)
    input_tensor = torch.from_numpy(values)
    name = "language_model.layers.0.linear_attn.in_proj_qkv.weight"
    fixtures = []
    for kind in args.formats:
        path = args.ggufs / f"d3-turbo-{kind}.gguf"
        weights = Weights(path)
        dtype = int(weights.tensors[name].tensor_type)
        activations = activation_roundtrip(input_tensor, dtype) if dtype in (1, 2, 3, 8, 12, 13, 14) else input_tensor
        expected = activations @ weights.value(name).T
        if dtype == 3:
            expected += q4_1_min_correction(input_tensor, weights.tensors[name])
        assert expected.isfinite().all()
        output = args.out / f"{kind}.f32"
        expected.numpy().astype("<f4").tofile(output)
        fixtures.append(dict(format=kind, model=str(path.resolve()), weight=name,
            input=str(input_path.resolve()), expected=str(output.resolve()), rows=len(values),
            source_trace_sha256=hashlib.sha256(args.native.read_bytes()).hexdigest()))
        del weights
    (args.out / "fixtures.json").write_text(json.dumps(fixtures, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--ggufs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--formats", nargs="+", required=True)
    p.add_argument("--threads", type=int, default=4)
    main(p.parse_args())
