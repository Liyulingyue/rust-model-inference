#!/usr/bin/env python3
"""Capture the oracle's layer-0 understanding-branch internals during the prefix pass.

The port carries a ~1e-2 relative error from layer 0 onward in *both* branches.
The understanding branch is the cheaper place to bisect (causal mask, no prefix
cache, same code), so this script runs **only** the T2I prefix forward and dumps
the points the Rust `prefix_l0_*` traces mirror:

  embed_tokens -> layers[0].input_layernorm -> q/k/v_proj -> (norm+rope) -> o_proj -> mlp

Module-level points are captured with forward hooks; the post-RoPE q/k are not a
module, so they are recomputed here from the captured `q_norm`/`k_norm` output
using the same three-segment rotation the Rust side applies.

  probe_prefix_l0.py --model_path <gguf> --out_dir <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROMPT = "A studio photograph of a red apple on a white table, soft daylight."


def dump(out_dir: Path, name: str, tensor: torch.Tensor) -> None:
    flat = tensor.detach().float().cpu().reshape(-1).numpy().astype("<f4")
    flat.tofile(out_dir / f"{name}.f32")
    print(f"[probe] {name} shape={tuple(tensor.shape)} n={flat.size}", flush=True)


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

    sensenova_u1.set_attn_backend("sdpa")
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

    # ---- build exactly the conditional prefix the T2I path builds ----------
    think_content = "<think>\n\n</think>\n\n" + "<img>"
    query = model._build_t2i_query(PROMPT, system_message=mnc.SYSTEM_MESSAGE_FOR_GEN, append_text=think_content)
    input_ids, indexes, mask = model._build_t2i_text_inputs(tokenizer, query)
    print(f"[probe] prefix tokens = {input_ids.shape[1]}", flush=True)

    # ---- hooks ------------------------------------------------------------
    captured: dict[str, torch.Tensor] = {}

    def hook(name):
        def fn(_m, _i, output):
            captured[name] = output[0] if isinstance(output, tuple) else output

        return fn

    lm = model.language_model.model
    layer0 = lm.layers[0]
    handles = [
        lm.embed_tokens.register_forward_hook(hook("prefix_embed")),
        layer0.input_layernorm.register_forward_hook(hook("prefix_l0_normed")),
        layer0.self_attn.q_proj.register_forward_hook(hook("prefix_l0_q")),
        layer0.self_attn.k_proj.register_forward_hook(hook("prefix_l0_k")),
        layer0.self_attn.v_proj.register_forward_hook(hook("prefix_l0_v")),
        layer0.self_attn.q_norm.register_forward_hook(hook("prefix_l0_qn_t")),
        layer0.self_attn.q_norm_hw.register_forward_hook(hook("prefix_l0_qn_hw")),
        layer0.self_attn.k_norm.register_forward_hook(hook("prefix_l0_kn_t")),
        layer0.self_attn.k_norm_hw.register_forward_hook(hook("prefix_l0_kn_hw")),
        layer0.self_attn.o_proj.register_forward_hook(hook("prefix_l0_o")),
        layer0.mlp.register_forward_hook(hook("prefix_l0_mlp")),
    ]
    try:
        with torch.no_grad():
            model._t2i_prefix_forward(input_ids, indexes, mask)
    finally:
        for handle in handles:
            handle.remove()

    for name in (
        "prefix_embed",
        "prefix_l0_normed",
        "prefix_l0_q",
        "prefix_l0_k",
        "prefix_l0_v",
        "prefix_l0_qn_t",
        "prefix_l0_qn_hw",
        "prefix_l0_kn_t",
        "prefix_l0_kn_hw",
        "prefix_l0_o",
        "prefix_l0_mlp",
    ):
        if name in captured:
            dump(out_dir, name, captured[name])
        else:
            print(f"[probe] MISSING {name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
