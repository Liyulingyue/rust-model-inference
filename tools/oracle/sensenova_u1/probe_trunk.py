#!/usr/bin/env python3
"""Capture the oracle's per-layer hidden states on the generation branch.

The Rust port matches the oracle everywhere except the generation-branch forward
(`Trunk::denoise`). This script localises that divergence to a decoder layer by
hooking the decoder layers and dumping their outputs during a **single** denoise
step, which is all that is needed (the divergence is already present at step 0).

Serial CFG (`cfg_batching=False`) is used so the forward order is
`prefix_cond, prefix_uncond, denoise_cond, denoise_uncond` — matching the Rust
port, which also runs the two branches separately.

Writes raw little-endian f32 as `denoise_layer{NN}_{cond,uncond}.f32` into
`--out_dir`, so `compare_trace.py`-style tooling can diff them against the Rust
`RMI_SENSENOVA_TRACE` dumps of the same name.

  probe_trunk.py --model_path <gguf> --out_dir <dir> [--device cpu] [--dtype float32]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

LAYERS = (0, 1, 2, 21, 41)

# One step is enough: the port already diverges at step 0.
CONTRACT = {
    "prompt": "A studio photograph of a red apple on a white table, soft daylight.",
    "width": 512,
    "height": 512,
    "num_steps": 1,
    "seed": 42,
    "cfg_scale": 4.0,
    "cfg_norm": "none",
    "cfg_interval": [0.0, 1.0],
    "timestep_shift": 3.0,
    "t_eps": 0.02,
    "batch_size": 1,
    "method": "euler",
    "cfg_batching": False,
    "think_mode": False,
    "attn_backend": "sdpa",
    # Required by the replay driver; kept minimal because this probe only
    # wants the per-layer captures, not the KV dumps.
    "kv_seq_slice": 8,
    "kv_layers": [0],
}


class Capture:
    """Collects every decoder-layer call so the prefix passes can be skipped."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.calls: list[dict[int, torch.Tensor]] = []
        self.current: dict[int, torch.Tensor] = {}

    def hook(self, layer_index: int):
        def fn(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.current[layer_index] = hidden.detach().float().cpu()

        return fn

    def flush(self) -> None:
        if self.current:
            self.calls.append(self.current)
            self.current = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_format", default="gguf")
    ap.add_argument("--gguf_checkpoint", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import sensenova_u1
    from sensenova_u1.models.neo_unify import modeling_neo_chat as mnc
    from sensenova_u1.utils import load_model_and_tokenizer_for_format

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_reference import Capture as RefCapture, run_t2i_contract  # noqa: E402

    sensenova_u1.set_attn_backend(CONTRACT["attn_backend"])
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"[probe] loading {args.model_path}", flush=True)
    model, tokenizer = load_model_and_tokenizer_for_format(
        args.model_path,
        model_format=args.model_format,
        dtype=dtype,
        device=args.device,
        gguf_checkpoint=args.gguf_checkpoint,
        for_offload=False,
    )
    print("[probe] loaded", flush=True)

    cap = Capture(out_dir)
    handles = []
    layers = model.language_model.model.layers
    for index in LAYERS:
        handles.append(layers[index].register_forward_hook(cap.hook(index)))

    # `_t2i_predict_cfg` calls `_t2i_predict_v` once per branch, so flush after
    # each call by wrapping it.
    original = model._t2i_predict_v

    def wrapped(*a, **kw):
        cap.current = {}
        result = original(*a, **kw)
        cap.flush()
        return result

    model._t2i_predict_v = wrapped

    try:
        # The replay driver writes its own .npy dumps; give it a scratch dir so
        # this probe only has to add the per-layer captures.
        scratch = out_dir / "_scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        run_t2i_contract(mnc, model, tokenizer, CONTRACT, RefCapture(scratch))
    finally:
        for handle in handles:
            handle.remove()

    print(f"[probe] captured {len(cap.calls)} denoise forward(s)", flush=True)
    if len(cap.calls) < 2:
        print("[probe] expected a cond and an uncond denoise forward", file=sys.stderr)
        return 1

    for branch, call in (("cond", cap.calls[0]), ("uncond", cap.calls[1])):
        for index in LAYERS:
            tensor = call.get(index)
            if tensor is None:
                print(f"[probe] missing layer {index} for {branch}", file=sys.stderr)
                continue
            flat = tensor.reshape(-1).numpy().astype("<f4")
            name = f"denoise_layer{index:02}_{branch}.f32"
            flat.tofile(out_dir / name)
            print(f"[probe] {name} shape={tuple(tensor.shape)} n={flat.size}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
