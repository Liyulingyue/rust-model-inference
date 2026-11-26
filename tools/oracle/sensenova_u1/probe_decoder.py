#!/usr/bin/env python3
"""Run the oracle's ConvDecoder on a Rust-dumped input, to isolate decoder bugs.

Reads `step0_decoder_in.f32` ([4096, token_h, token_w], raw LE f32) from a
`RMI_SENSENOVA_TRACE` directory, rebuilds the ConvDecoder from the GGUF's
`fm_modules.fm_head.*` tensors in torch, and writes `oracle_decoder_out.f32`.
If the Rust decoder's `step0_x_pred_cond` matches this, the decoder is correct
and the divergence is upstream in the trunk.

  probe_decoder.py <gguf> <trace_dir> <token_h> <token_w>
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gguf import GGUFReader


class ConvDecoder(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.ps1 = nn.PixelShuffle(2)
        self.conv1 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.act1 = nn.GELU()
        self.ps2 = nn.PixelShuffle(2)
        self.conv2 = nn.Conv2d(256, 192, 3, padding=1)
        self.ps3 = nn.PixelShuffle(8)
        self.conv1.weight.data.copy_(sd["conv1.weight"])
        self.conv1.bias.data.copy_(sd["conv1.bias"])
        self.conv2.weight.data.copy_(sd["conv2.weight"])
        self.conv2.bias.data.copy_(sd["conv2.bias"])

    def forward(self, x):
        x = self.act1(self.conv1(self.ps1(x)))
        return self.ps3(self.conv2(self.ps2(x)))


def main() -> int:
    gguf, trace_dir, th, tw = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    reader = GGUFReader(gguf)
    want = {
        "conv1.weight": (1024, 1024, 3, 3),
        "conv1.bias": (1024,),
        "conv2.weight": (192, 256, 3, 3),
        "conv2.bias": (192,),
    }
    sd = {}
    for t in reader.tensors:
        key = t.name.split("fm_head.")[-1]
        if key in want:
            # GGUF header dims are the reversed torch shape; the payload is
            # already torch-contiguous, so reshape with the reversed header.
            arr = np.array(t.data, dtype=np.float32)
            sd[key] = torch.from_numpy(arr.reshape(want[key]).copy())
            print(f"{key}: gguf dims {list(getattr(t, "shape", getattr(t, "dims", [])))} -> torch {want[key]}")
    model = ConvDecoder(sd).eval()

    flat = np.fromfile(trace_dir / "step0_decoder_in.f32", dtype="<f4")
    x = torch.from_numpy(flat.reshape(1, 4096, th, tw).copy())
    with torch.no_grad():
        y = model(x)
    print("out shape", tuple(y.shape))
    y.numpy().astype("<f4").tofile(trace_dir / "oracle_decoder_out.f32")
    print("wrote", trace_dir / "oracle_decoder_out.f32")
    return 0


if __name__ == "__main__":
    sys.exit(main())
