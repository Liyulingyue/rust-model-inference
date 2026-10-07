"""Generate a golden for BoundaryQueryHead.forward.

Reuses ``BoundaryEncoder.forward`` via the reference ``BoundaryExtractorModel``
forward path. The input matches ``tests/gliner2_5_base_v1_forward.rs``'s
smoke. Output is the start/end logits + cumulative inside prefix for a
fixed query.

Regenerate with:
    PYTHONPATH=target/gliner2-oracle \\
        models/.venv/bin/python3 \\
        tools/oracle/gliner_boundary/dump_boundary_query_head.py
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))

from gliner2.models.boundary.encoding import BoundaryEncoder  # noqa: E402
from gliner2.models.boundary.heads import BoundaryQueryHead  # noqa: E402

HIDDEN_SIZE = 768
BOUNDARY_DIM = 128
REFINEMENT_LAYERS = 1
FFN_MULTIPLIER = 2.0
ATTENTION_LAYERS = 2
ATTENTION_HEADS = 4
ATTENTION_WINDOW = 128


def load_boundary_encoder(state_dict, prefix):
    renames = {
        f"{prefix}.bos_state": "bos_state",
        f"{prefix}.eos_state": "eos_state",
        f"{prefix}.left_projection.weight": "left_projection.weight",
        f"{prefix}.left_projection.bias": "left_projection.bias",
        f"{prefix}.right_projection.weight": "right_projection.weight",
        f"{prefix}.right_projection.bias": "right_projection.bias",
        f"{prefix}.output_projection.weight": "output_projection.weight",
        f"{prefix}.output_projection.bias": "output_projection.bias",
        f"{prefix}.layer_norm.weight": "layer_norm.weight",
        f"{prefix}.layer_norm.bias": "layer_norm.bias",
    }
    for i in range(ATTENTION_LAYERS):
        for j in ["norm.weight", "norm.bias", "qkv_projection.weight",
                  "qkv_projection.bias", "output_projection.weight",
                  "output_projection.bias"]:
            renames[f"{prefix}.attention_blocks.{i}.{j}"] = f"attention_blocks.{i}.{j}"
    for i in range(REFINEMENT_LAYERS):
        for j in ["norm.weight", "norm.bias", "input_projection.weight",
                  "input_projection.bias", "output_projection.weight",
                  "output_projection.bias"]:
            renames[f"{prefix}.refinement_blocks.{i}.{j}"] = f"refinement_blocks.{i}.{j}"
    encoder = BoundaryEncoder(
        hidden_size=HIDDEN_SIZE,
        boundary_dim=BOUNDARY_DIM,
        refinement_layers=REFINEMENT_LAYERS,
        ffn_multiplier=FFN_MULTIPLIER,
        attention_layers=ATTENTION_LAYERS,
        attention_heads=ATTENTION_HEADS,
        attention_window=ATTENTION_WINDOW,
    )
    renamed = {renames[k]: v for k, v in state_dict.items() if k in renames}
    encoder.load_state_dict(renamed, strict=False)
    encoder.eval()
    return encoder


def load_query_head(state_dict):
    prefix = "boundary_head.boundary_query_head"
    renames = {
        f"{prefix}.start_boundary_projection.weight": "start_boundary_projection.weight",
        f"{prefix}.start_boundary_projection.bias": "start_boundary_projection.bias",
        f"{prefix}.start_query_projection.weight": "start_query_projection.weight",
        f"{prefix}.start_query_projection.bias": "start_query_projection.bias",
        f"{prefix}.end_boundary_projection.weight": "end_boundary_projection.weight",
        f"{prefix}.end_boundary_projection.bias": "end_boundary_projection.bias",
        f"{prefix}.end_query_projection.weight": "end_query_projection.weight",
        f"{prefix}.end_query_projection.bias": "end_query_projection.bias",
        f"{prefix}.inside_text_projection.weight": "inside_text_projection.weight",
        f"{prefix}.inside_text_projection.bias": "inside_text_projection.bias",
        f"{prefix}.inside_query_projection.weight": "inside_query_projection.weight",
        f"{prefix}.inside_query_projection.bias": "inside_query_projection.bias",
    }
    head = BoundaryQueryHead(hidden_size=HIDDEN_SIZE, boundary_dim=BOUNDARY_DIM)
    renamed = {renames[k]: v for k, v in state_dict.items() if k in renames}
    head.load_state_dict(renamed, strict=False)
    head.eval()
    return head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "models" / "gliner2.5-base-v1")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "gliner2.5-base-v1" / "boundary-query-head-golden.json",
    )
    args = parser.parse_args()

    state_dict = load_file(str(args.model_dir / "model.safetensors"))
    encoder = load_boundary_encoder(state_dict, "boundary_head.boundary_encoder")
    head = load_query_head(state_dict)

    seq_len = 8
    torch.manual_seed(20260930)
    text_states = (torch.arange(seq_len * HIDDEN_SIZE, dtype=torch.float32) * 0.011).sin() - 1.0
    text_states = text_states.view(1, seq_len, HIDDEN_SIZE)
    text_mask = torch.tensor([[True] * 6 + [False] * 2])

    # Run BoundaryEncoder to get boundary_states.
    with torch.inference_mode():
        encoding = encoder(text_states, text_mask)

    # Run BoundaryQueryHead with deterministic query states. We use the
    # encoder's own output as queries (size hidden_size=HIDDEN_SIZE == q_dim
    # for this checkpoint) so the test exercises the real q_dim path.
    # Two queries: [0]=positive (taken from row 0), [1]=negative (row 1).
    query_states = torch.tensor(
        [
            [
                text_states[0, 0].tolist(),  # positive intent
                text_states[0, 1].tolist(),  # negative intent
            ]
        ],
        dtype=torch.float32,
    )
    query_mask = torch.tensor([[True, True]])

    with torch.inference_mode():
        marginals = head(
            encoding.states,
            encoding.mask,
            text_states,
            text_mask,
            query_states,
            query_mask,
        )

    fixture = {
        "config": {
            "hidden_size": HIDDEN_SIZE,
            "boundary_dim": BOUNDARY_DIM,
            "seq_len": seq_len,
            "valid_tokens": 6,
            "q_count": 2,
        },
        "start_logits": marginals.start_logits.flatten().tolist(),
        "end_logits": marginals.end_logits.flatten().tolist(),
        "inside_prefix": marginals.inside_prefix.flatten().tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture) + "\n")
    print(
        f"{args.out} (start={len(fixture['start_logits'])}, end={len(fixture['end_logits'])}, "
        f"inside_prefix={len(fixture['inside_prefix'])})"
    )


if __name__ == "__main__":
    main()