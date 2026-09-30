"""Generate a golden for the LIMITED PairScorer (no span content, no
inside evidence, no endpoint difference features).

Same boundary-encoder + boundary-query-head + score_explicit_pairs
oracles above; this oracle additionally runs the pair_scorer with
`enable_span_content=False, use_inside_evidence=False,
endpoint_difference_features=False` to match the Rust port that
intentionally omits these features (tracked in glinerTODO.md).

The reference path is
``gliner2.models.boundary.scoring.SparseBoundaryPairScorer.forward``
with the limited config. The output is a per-(B, Q, C) score vector.

Regenerate with:
    PYTHONPATH=target/gliner2-oracle \\
        models/.venv/bin/python3 \\
        tools/oracle/gliner_boundary/dump_pair_scorer_limited.py
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
from gliner2.models.boundary.proposal import (  # noqa: E402
    ProposalSettings, SparseBoundaryProposer, BoundaryProposals,
)
from gliner2.models.boundary.scoring import (  # noqa: E402
    SparseBoundaryPairScorer, mask_invalid_candidate_logits,
)

HIDDEN_SIZE = 768
BOUNDARY_DIM = 128
PAIR_DIM = 128


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
    head.load_state_dict({renames[k]: v for k, v in state_dict.items() if k in renames}, strict=False)
    head.eval()
    return head


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


def load_pair_scorer(state_dict):
    prefix = "boundary_head.pair_scorer"
    renames = {
        f"{prefix}.start_endpoint_projection.weight": "start_endpoint_projection.weight",
        f"{prefix}.start_endpoint_projection.bias": "start_endpoint_projection.bias",
        f"{prefix}.end_endpoint_projection.weight": "end_endpoint_projection.weight",
        f"{prefix}.end_endpoint_projection.bias": "end_endpoint_projection.bias",
        f"{prefix}.query_gate.weight": "query_gate.weight",
        f"{prefix}.query_gate.bias": "query_gate.bias",
        f"{prefix}.compat_mix.weight": "compat_mix.weight",
        f"{prefix}.compat_mix.bias": "compat_mix.bias",
        f"{prefix}.length_query_projection.weight": "length_query_projection.weight",
        f"{prefix}.length_query_projection.bias": "length_query_projection.bias",
    }
    scorer = SparseBoundaryPairScorer(
        boundary_dim=BOUNDARY_DIM,
        query_dim=HIDDEN_SIZE,
        pair_dim=PAIR_DIM,
        # Limited scorer: matches the Rust port that omits the three
        # optional features.
        use_inside_evidence=False,
        enable_span_content=False,
        enable_rotary_endpoints=True,
        rotary_base=10000.0,
        query_conditioned_inside_weight=False,
        endpoint_difference_features=False,
        reranker_endpoint_compat=True,
        multihead_pair_compat_heads=8,
        content_dim=64,
        content_hidden_size=HIDDEN_SIZE,
    )
    scorer.load_state_dict({renames[k]: v for k, v in state_dict.items() if k in renames}, strict=False)
    scorer.eval()
    return scorer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "models" / "gliner2.5-base-v1")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "gliner2.5-base-v1" / "pair-scorer-limited-golden.json",
    )
    args = parser.parse_args()

    state_dict = load_file(str(args.model_dir / "model.safetensors"))
    encoder = load_boundary_encoder(state_dict)
    head = load_query_head(state_dict)
    proposer = load_proposer(state_dict)
    scorer = load_pair_scorer(state_dict)

    seq_len = 8
    text_states = (torch.arange(seq_len * HIDDEN_SIZE, dtype=torch.float32) * 0.011).sin() - 1.0
    text_states = text_states.view(1, seq_len, HIDDEN_SIZE)
    text_mask = torch.tensor([[True] * 6 + [False] * 2])

    with torch.inference_mode():
        encoding = encoder(text_states, text_mask)
        marginals = head(
            encoding.states, encoding.mask, text_states, text_mask,
            torch.tensor([[
                text_states[0, 0].tolist(), text_states[0, 1].tolist()
            ]], dtype=torch.float32),
            torch.tensor([[True, True]]),
        )

    query_states = torch.tensor(
        [[text_states[0, 0].tolist(), text_states[0, 1].tolist()]],
        dtype=torch.float32,
    )
    query_mask = torch.tensor([[True, True]])
    text_lengths = text_mask.sum(dim=1).long()

    indices = torch.tensor(
        [
            [
                [[0, 2], [1, 4], [3, 6], [5, 5], [4, 2], [7, 8]],
                [[0, 2], [1, 4], [3, 6], [5, 5], [4, 2], [7, 8]],
            ]
        ],
        dtype=torch.long,
    )
    legal = (
        (indices[..., 0] >= 0)
        & (indices[..., 1] > indices[..., 0])
        & (indices[..., 1] <= text_lengths.view(1, 1, 1))
    )

    with torch.inference_mode():
        compat = proposer.score_explicit_pairs(
            encoding.states, query_states, indices, legal
        )
        proposals = BoundaryProposals(
            indices=indices, logits=None, valid_mask=legal, compat_logits=compat,
        )
        scores = scorer(
            encoding.states, query_states, proposals,
            marginals.start_logits, marginals.end_logits,
            None,  # inside_prefix omitted in the limited scorer
            text_lengths, text_states, text_mask,
            inside_prefix_mean=None,
        )

    fixture = {
        "indices": indices.flatten().tolist(),
        "valid_mask": legal.flatten().tolist(),
        "compatibility": compat.flatten().tolist(),
        "scores": scores.flatten().tolist(),
        "config": {
            "hidden_size": HIDDEN_SIZE,
            "boundary_dim": BOUNDARY_DIM,
            "pair_dim": PAIR_DIM,
            "seq_len": seq_len,
            "valid_tokens": 6,
            "q_count": 2,
            "c_count": 6,
            "enable_rotary_endpoints": True,
            "rotary_base": 10000.0,
            "multihead_pair_compat_heads": 8,
            "limited_features": [
                "enable_span_content=false",
                "use_inside_evidence=false",
                "endpoint_difference_features=false",
            ],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture) + "\n")
    print(f"{args.out} (scores={len(fixture['scores'])})")


if __name__ == "__main__":
    main()