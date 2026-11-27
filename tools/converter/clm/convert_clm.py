"""Convert a Contrastive-LM projection-head checkpoint to GGUF.

The reference checkpoint (``CLM_v0.1-8B.pt``) is a ``torch.save`` dict with
``state_head`` / ``action_head`` state dicts, a scalar ``logit_scale`` and a
``cfg``.  Each head is a small MLP:

    h = GELU(inp(x))                    # hidden -> width
    h = GELU(LayerNorm(hidden_i(h)))    # width -> width   (depth-2 times)
    z = out(h)                          # width -> projection_dim
    z = L2_normalize(z)                 # applied at scoring time, not stored

The score of a (state, candidate) pair is
``min(exp(logit_scale), 100.0) * dot(z_state, z_candidate)``.

Only the heads are exported; the encoder (a frozen Qwen3-8B) is a separate
GGUF the runtime loads alongside.  Weights are written as F32 (the file is
75 MB, so there is nothing to gain from quantising).

Tensor names follow the ``clm.`` prefix so they cannot collide with any
llama.cpp arch:

    clm.state_head.inp.weight      [width, hidden]      -> gguf (hidden, width)
    clm.state_head.inp.bias        [width]
    clm.state_head.hidden.{i}.weight/bias               (depth - 2 of them)
    clm.state_head.norms.{i}.weight/bias                (only when layernorm)
    clm.state_head.out.weight      [proj, width]        -> gguf (width, proj)
    clm.state_head.out.bias        [proj]
    clm.action_head.*                                  (same)
    clm.logit_scale                [1]                  (already clamped)

Metadata carries the architecture string and the head geometry so the Rust
side can reject a file it does not understand instead of mis-reading shapes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.converter.utils.gguf import GGML_F32, GgufWriter, _gguf_meta_value, gguf_dims

ARCH = "clm"

# nn.LayerNorm default; heads.py builds the norm with no explicit eps.
LAYERNORM_EPS = 1e-5


def _linear(sd: dict, prefix: str) -> tuple[str, torch.Tensor, torch.Tensor | None]:
    """Pull ``{prefix}weight`` / ``{prefix}bias`` out of a state dict."""
    weight = sd[f"{prefix}weight"]
    bias = sd.get(f"{prefix}bias")
    return prefix.rstrip("."), weight, bias


def _validate_cfg(cfg: dict) -> dict:
    """Only the geometry this converter and runtime know how to emit."""
    expected_keys = {"width", "depth"}
    missing = expected_keys - set(cfg)
    if missing:
        raise ValueError(f"cfg is missing {sorted(missing)}")
    out = dict(
        width=int(cfg["width"]),
        depth=int(cfg["depth"]),
        proj=int(cfg.get("projection_dim", cfg.get("proj", 512))),
        hidden=int(cfg.get("hidden_size", 4096)),
        activation=str(cfg.get("activation", "gelu")),
        layernorm=bool(cfg.get("layernorm", False)),
        residual=bool(cfg.get("residual", False)),
    )
    if out["activation"] != "gelu":
        # nn.GELU() defaults to the exact erf form; SiLU/ReLU would change the
        # reference forward, so refuse rather than silently diverge.
        raise ValueError(f"Unsupported activation {out['activation']!r}; expected 'gelu'")
    if out["depth"] < 2:
        raise ValueError(f"depth must be >= 2, got {out['depth']}")
    if out["residual"]:
        # The residual branch adds the pre-block activation;
        # the Rust runtime would need an extra buffer per hidden block.
        raise ValueError("residual heads are not supported yet")
    return out



def _as_is(t: torch.Tensor) -> torch.Tensor:
    """Keep torch's [out, in] row-major bytes.

    The tensor directory declares (in, out), which is how a reader following
    the GGML ne0-contiguous contract interprets element (i, o) as
    flat[i + o*n_in] -- exactly torch's [out, in] row-major bytes.  Do NOT
    transpose here: doing so double-transposes and silently swaps weights.
    """
    return t.float().contiguous()


def check(sd: dict, head: str, prefix: str, torch_shape: tuple) -> None:
    """Validate the checkpoint tensor against the torch shape from cfg."""
    w = sd[f"{prefix}.weight"]
    if tuple(w.shape) != torch_shape:
        raise ValueError(f"{head}.{prefix}.weight shape {tuple(w.shape)} != {torch_shape}")


def convert(ckpt: Path, out_path: Path) -> dict:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    for key in ("state_head", "action_head", "logit_scale", "cfg"):
        if key not in ck:
            raise ValueError(f"checkpoint is missing {key!r}")

    cfg = _validate_cfg(ck["cfg"])
    width, depth, proj, hidden = cfg["width"], cfg["depth"], cfg["proj"], cfg["hidden"]
    ln = cfg["layernorm"]

    # logit_scale is stored as a log; the runtime wants the linear factor, and
    # the reference clamps it at 100.0 (see _load() in clm/heads.py).
    scale = float(ck["logit_scale"].float().exp().clamp(max=100.0))

    writer = GgufWriter(out_path)
    writer.add_meta("general.architecture", ARCH)
    writer.add_meta("general.type", "clm-heads")
    writer.add_meta(f"{ARCH}.hidden_size", hidden)
    writer.add_meta(f"{ARCH}.width", width)
    writer.add_meta(f"{ARCH}.depth", depth)
    writer.add_meta(f"{ARCH}.projection_dim", proj)
    writer.add_meta(f"{ARCH}.layernorm", ln)
    writer.add_meta(f"{ARCH}.layernorm_eps", LAYERNORM_EPS)
    writer.add_meta(f"{ARCH}.activation", "gelu")
    writer.add_meta(f"{ARCH}.logit_scale", scale)
    # The encoder is a separate GGUF; name it so the runtime can refuse a
    # mismatched base instead of producing meaningless scores.
    writer.add_meta(f"{ARCH}.base_model", str(ck["cfg"].get("model", "Qwen/Qwen3-8B")))

    n_params = 0
    for head in ("state_head", "action_head"):
        sd = ck[head]

        # Bytes stay torch's [out, in]; gguf_dims labels them (in, out).
        check(sd, head, "inp", (width, hidden))
        w = _as_is(sd["inp.weight"])          # bytes (width, hidden) -> labelled (hidden, width)
        b = sd["inp.bias"].float().contiguous()
        writer.add_tensor(f"{ARCH}.{head}.inp.weight", GGML_F32, gguf_dims(tuple(w.shape)), w.numpy().tobytes())
        writer.add_tensor(f"{ARCH}.{head}.inp.bias", GGML_F32, gguf_dims(tuple(b.shape)), b.numpy().tobytes())
        n_params += w.numel() + b.numel()

        # (depth - 2) identical hidden blocks, each with an optional LayerNorm.
        for i in range(depth - 2):
            w = _as_is(sd[f"hidden.{i}.weight"])   # square, no ambiguity
            b = sd[f"hidden.{i}.bias"].float().contiguous()
            check(sd, head, f"hidden.{i}", (width, width))
            writer.add_tensor(f"{ARCH}.{head}.hidden.{i}.weight", GGML_F32, gguf_dims(tuple(w.shape)), w.numpy().tobytes())
            writer.add_tensor(f"{ARCH}.{head}.hidden.{i}.bias", GGML_F32, gguf_dims(tuple(b.shape)), b.numpy().tobytes())
            n_params += w.numel() + b.numel()
            if ln:
                nw = sd[f"norms.{i}.weight"].float().contiguous()
                nb = sd[f"norms.{i}.bias"].float().contiguous()
                if tuple(nw.shape) != (width,) or tuple(nb.shape) != (width,):
                    raise ValueError(f"{head}.norms.{i} shape != {(width,)}")
                writer.add_tensor(f"{ARCH}.{head}.norms.{i}.weight", GGML_F32, gguf_dims(tuple(nw.shape)), nw.numpy().tobytes())
                writer.add_tensor(f"{ARCH}.{head}.norms.{i}.bias", GGML_F32, gguf_dims(tuple(nb.shape)), nb.numpy().tobytes())
                n_params += nw.numel() + nb.numel()
            elif f"norms.{i}.weight" in sd:
                raise ValueError(f"{head}: cfg says layernorm=False but norms.{i} exists")

        check(sd, head, "out", (proj, width))
        w = _as_is(sd["out.weight"])            # bytes (proj, width) -> labelled (width, proj)
        b = sd["out.bias"].float().contiguous()
        writer.add_tensor(f"{ARCH}.{head}.out.weight", GGML_F32, gguf_dims(tuple(w.shape)), w.numpy().tobytes())
        writer.add_tensor(f"{ARCH}.{head}.out.bias", GGML_F32, gguf_dims(tuple(b.shape)), b.numpy().tobytes())
        n_params += w.numel() + b.numel()

    import struct

    writer.add_tensor(f"{ARCH}.logit_scale", GGML_F32, (1,), struct.pack("<f", scale))
    writer.write()

    return {"config": cfg, "scale": scale, "n_params": n_params,
            "tensors": len(writer.tensors), "output": str(out_path)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt", type=Path, help="CLM_v0.1-8B.pt")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output GGUF (default: <ckpt dir>/clm-v0.1-8B-heads-f32.gguf)")
    ap.add_argument("--json", action="store_true", help="print a summary as JSON")
    args = ap.parse_args()

    out = args.output or args.ckpt.parent / "clm-v0.1-8B-heads-f32.gguf"
    summary = convert(args.ckpt, out)
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        c = summary["config"]
        print(f"wrote {summary['output']}")
        print(f"  head: {c['hidden']} -> {c['width']} -> {c['proj']} (depth {c['depth']}, "
              f"layernorm={c['layernorm']})")
        print(f"  logit_scale = {summary['scale']}")
        print(f"  {summary['tensors']} tensors, {summary['n_params']} params")


if __name__ == "__main__":
    main()
