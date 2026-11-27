"""Generate a golden for BoundaryProposer::score_explicit_pairs.

Same input shape as the boundary-encoder / boundary-query-head goldens:
1 sample × 6 valid tokens + 2 padding. Two queries, taken from rows
0 and 1 of text_states. Indices enumerate a few explicit (start, end)
pairs including legal and illegal (start >= end) pairs.

The reference path is
``gliner2.models.boundary.proposal.SparseBoundaryProposer.score_explicit_pairs``.
We construct the proposer from the safetensors weights and run it on
the same fixed input.

Regenerate with:
    PYTHONPATH=target/gliner2-oracle \\
        models/.venv/bin/python3 \\
        tools/oracle/gliner_boundary/dump_score_explicit_pairs.py
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))

from gliner2.models.boundary.proposal import (  # noqa: E402
    ProposalSettings, SparseBoundaryProposer,
)
from gliner2.models.boundary.heads import BoundaryMarginals  # noqa: E402
from gliner2.models.boundary.encoding import BoundaryEncoder  # noqa: E402

HIDDEN_SIZE = 768
BOUNDARY_DIM = 128


def load_boundary_encoder(state_dict):
    prefix = "boundary_head.boundary_encoder"
    renames = {
        f"{prefix}.bos_state": "bos_state", f"{prefix}.eos_state": "eos_state",
        f"{prefix}.left_projection.weight": "left_projection.weight",
        f"{prefix}.left_projection.bias": "left_projection.bias",
        f"{prefix}.right_projection.weight": "right_projection.weight",
        f"{prefix}.right_projection.bias": "right_projection.bias",
        f"{prefix}.output_projection.weight": "output_projection.weight",
        f"{prefix}.output_projection.bias": "output_projection.bias",
        f"{prefix}.layer_norm.weight": "layer_norm.weight",
        f"{prefix}.layer_norm.bias": "layer_norm.bias",
    }
    for i in range(2):
        for j in ["norm.weight", "norm.bias", "qkv_projection.weight",
                  "qkv_projection.bias", "output_projection.weight",
                  "output_projection.bias"]:
            renames[f"{prefix}.attention_blocks.{i}.{j}"] = f"attention_blocks.{i}.{j}"
    for i in range(1):
        for j in ["norm.weight", "norm.bias", "input_projection.weight",
                  "input_projection.bias", "output_projection.weight",
                  "output_projection.bias"]:
            renames[f"{prefix}.refinement_blocks.{i}.{j}"] = f"refinement_blocks.{i}.{j}"
    encoder = BoundaryEncoder(
        hidden_size=HIDDEN_SIZE, boundary_dim=BOUNDARY_DIM,
        refinement_layers=1, ffn_multiplier=2.0,
        attention_layers=2, attention_heads=4, attention_window=128,
    )
    encoder.load_state_dict({renames[k]: v for k, v in state_dict.items() if k in renames}, strict=False)
    encoder.eval()
    return encoder


def load_proposer(state_dict):
    prefix = "boundary_head.boundary_proposer"
    renames = {
        f"{prefix}.start_pair_projection.weight": "start_pair_projection.weight",
        f"{prefix}.start_pair_projection.bias": "start_pair_projection.bias",
        f"{prefix}.end_key_projection.weight": "end_key_projection.weight",
        f"{prefix}.end_key_projection.bias": "end_key_projection.bias",
        f"{prefix}.start_query_projection.weight": "start_query_projection.weight",
        f"{prefix}.start_query_projection.bias": "start_query_projection.bias",
    }
    settings = ProposalSettings(
        start_top_k=24, end_top_k=24,
        ends_per_start=12, starts_per_end=12,
        candidate_budget=192, training_candidate_budget=192,
        max_gold_per_query=64, end_block_size=256,
        bidirectional=True, export_mode="auto",
        vectorized_pair_elements=16_777_216,
        enable_rotary_endpoints=True, rotary_base=10000.0,
        boundary_top_k_alpha=0.0, boundary_top_k_max=128,
        boundary_top_k_bucket=8,
    )
    proposer = SparseBoundaryProposer(
        boundary_dim=BOUNDARY_DIM, query_dim=HIDDEN_SIZE, settings=settings,
    )
    proposer.load_state_dict({renames[k]: v for k, v in state_dict.items() if k in renames}, strict=False)
    proposer.eval()
    return proposer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "models" / "gliner2.5-base-v1")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "gliner2.5-base-v1" / "score-explicit-pairs-golden.json",
    )
    args = parser.parse_args()

    state_dict = load_file(str(args.model_dir / "model.safetensors"))
    encoder = load_boundary_encoder(state_dict)
    proposer = load_proposer(state_dict)

    seq_len = 8
    text_states = (torch.arange(seq_len * HIDDEN_SIZE, dtype=torch.float32) * 0.011).sin() - 1.0
    text_states = text_states.view(1, seq_len, HIDDEN_SIZE)
    text_mask = torch.tensor([[True] * 6 + [False] * 2])

    with torch.inference_mode():
        encoding = encoder(text_states, text_mask)

    # Two queries: row 0 (positive intent) and row 1 (negative intent).
    query_states = torch.tensor(
        [[text_states[0, 0].tolist(), text_states[0, 1].tolist()]],
        dtype=torch.float32,
    )

    # Six explicit (start, end) candidates per query: 3 legal, 1 invalid
    # (start == end), 1 invalid (start > end), 1 padded (start >= seq_len).
    indices = torch.tensor(
        [
            [
                # q=0
                [
                    [0, 2],
                    [1, 4],
                    [3, 6],
                    [5, 5],  # start == end, invalid
                    [4, 2],  # start > end, invalid
                    [7, 8],  # past valid boundary, invalid
                ],
                # q=1
                [
                    [0, 2],
                    [1, 4],
                    [3, 6],
                    [5, 5],
                    [4, 2],
                    [7, 8],
                ],
            ]
        ],
        dtype=torch.long,
    )
    text_lengths = text_mask.sum(dim=1).long()
    legal = (
        (indices[..., 0] >= 0)
        & (indices[..., 1] > indices[..., 0])
        & (indices[..., 1] <= text_lengths.view(1, 1, 1))
    )

    with torch.inference_mode():
        compat = proposer.score_explicit_pairs(
            encoding.states, query_states, indices, legal
        )

    fixture = {
        "indices": indices.flatten().tolist(),
        "valid_mask": legal.flatten().tolist(),
        "compatibility": compat.flatten().tolist(),
        "config": {
            "hidden_size": HIDDEN_SIZE,
            "boundary_dim": BOUNDARY_DIM,
            "seq_len": seq_len,
            "valid_tokens": 6,
            "q_count": 2,
            "c_count": 6,
            "enable_rotary_endpoints": True,
            "rotary_base": 10000.0,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture) + "\n")
    print(
        f"{args.out} (compatibility={len(fixture['compatibility'])}, "
        f"valid={fixture['valid_mask'].count(True)})"
    )


if __name__ == "__main__":
    main()