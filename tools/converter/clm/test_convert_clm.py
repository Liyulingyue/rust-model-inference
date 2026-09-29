"""Round-trip check for the CLM head converter.

Writes a GGUF from the reference checkpoint and asserts the bytes match the
torch state dict under the GGML ne0-contiguous contract (element (i, o) is
flat[i + o*n_in], i.e. torch's [out, in] row-major bytes), plus the scalar
metadata the runtime reads.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from tools.converter.clm.convert_clm import convert
from tools.converter.utils.gguf import GGML_F32

ROOT = Path(__file__).resolve().parents[3]


def test_convert(tmp_path: Path) -> None:
    ckpt = ROOT / "models" / "CLM-v0.1-8B" / "CLM_v0.1-8B.pt"
    if not ckpt.exists():
        print(f"skip: {ckpt} not present")
        return
    out = tmp_path / "clm.gguf"
    convert(ckpt, out)

    from gguf.gguf_reader import GGUFReader

    reader = GGUFReader(out)
    fields = {f.name: f for f in reader.fields.values()}
    tensors = {t.name: t for t in reader.tensors}

    def meta(key):
        v = fields[key].contents()
        return v

    assert meta("general.architecture") == "clm"
    assert meta("clm.hidden_size") == 4096
    assert meta("clm.width") == 1536
    assert meta("clm.depth") == 3
    assert meta("clm.projection_dim") == 512
    # logit_scale is a raw F32 payload, so index it rather than decode.
    scale = tensors["clm.logit_scale"]
    assert scale.tensor_type.name == "F32" and tuple(scale.shape) == (1,)
    got = torch.frombuffer(bytearray(scale.data), dtype=torch.float32).item()
    assert math.isclose(got, 100.0, rel_tol=1e-6), got

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    for head in ("state_head", "action_head"):
        for prefix in ("inp", "hidden.0", "out"):
            name = f"clm.{head}.{prefix}.weight"
            t = tensors[name]
            flat = torch.frombuffer(bytearray(t.data), dtype=torch.float32)
            want = ck[head][f"{prefix}.weight"].float().contiguous()
            # declared dims are (in, out); torch's [out, in] bytes are the
            # same buffer, so the flat views must be identical.
            assert tuple(t.shape) == tuple(reversed(tuple(want.shape))), name
            assert torch.equal(flat, want.flatten()), name

    # biases and LayerNorm params are 1-D: byte-identical, no layout question.
    for head in ("state_head", "action_head"):
        for prefix in ("inp", "hidden.0", "out"):
            t = tensors[f"clm.{head}.{prefix}.bias"]
            flat = torch.frombuffer(bytearray(t.data), dtype=torch.float32)
            assert torch.equal(flat, ck[head][f"{prefix}.bias"].float()), prefix
        for i, key in enumerate(("weight", "bias")):
            t = tensors[f"clm.{head}.norms.0.{key}"]
            flat = torch.frombuffer(bytearray(t.data), dtype=torch.float32)
            assert torch.equal(flat, ck[head][f"norms.0.{key}"].float()), key
    print("ok: clm gguf round-trips")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_convert(Path(d))
