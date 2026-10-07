"""Generate a golden for the FULL `BoundaryHead.score_explicit_spans` path.

Unlike ``dump_pair_scorer_limited.py`` (which hand-built a scorer with three
features switched off to match a partial Rust port), this oracle loads the
reference ``BoundaryHead`` straight from the checkpoint's ``config.json`` +
``model.safetensors`` and calls the real
``BoundaryHead.score_explicit_spans`` (``boundary/model.py:274``). Every
feature flag therefore comes from the checkpoint itself, not from this file —
which is the point: it is the path the reference engine uses to score
caller-supplied spans (entity classification, entity attributes, joint-IE),
and it is the path the Rust ``score_spans`` mirrors.

The chain exercised here is
``BoundaryEncoder`` -> ``BoundaryQueryHead`` ->
``SparseBoundaryProposer.score_explicit_pairs`` ->
``SparseBoundaryPairScorer.forward`` with the published base-v1 settings:
``enable_span_content``, ``use_inside_evidence``,
``query_conditioned_inside_weight`` and ``endpoint_difference_features`` all
on, ``content_dim = 64``, ``pair_dim = 128``,
``multihead_pair_compat_heads = 8``.

Regenerate with:
    PYTHONPATH=target/gliner2-oracle \\
        models/.venv/bin/python3 \\
        tools/oracle/gliner_boundary/dump_score_explicit_spans_full.py
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))

from gliner2.configuration import BoundaryHeadSettings  # noqa: E402
from gliner2.models.boundary.model import BoundaryHead  # noqa: E402

HIDDEN_SIZE = 768

# Two cases with different sequence lengths and mask patterns: the second one
# exercises a short document where a span reaches the final boundary and an
# invalid candidate whose index is out of range.
CASES = [
    {"seq_len": 8, "valid_tokens": 6, "q_count": 2, "c_count": 6,
     "candidates": [[0, 2], [1, 4], [3, 6], [5, 5], [4, 2], [7, 8]]},
    {"seq_len": 5, "valid_tokens": 4, "q_count": 3, "c_count": 4,
     "candidates": [[0, 1], [1, 4], [2, 3], [4, 9]]},
]


def build_head(model_dir: Path) -> BoundaryHead:
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("architecture") != "boundary":
        raise ValueError(f"expected architecture='boundary', got {config.get('architecture')!r}")
    settings = BoundaryHeadSettings(**config["boundary_head"])
    head = BoundaryHead(
        HIDDEN_SIZE, settings, query_dim=HIDDEN_SIZE,
        build_candidate_states=settings.enable_records,
    )
    state_dict = load_file(str(model_dir / "model.safetensors"))
    prefix = "boundary_head."
    sub = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    # `load_state_dict` is strict on purpose: a missing or extra key here means
    # this oracle would be scoring something other than the checkpoint.
    head.load_state_dict(sub, strict=True)
    head.eval()
    return head


def synthetic_states(seq_len: int) -> torch.Tensor:
    flat = (torch.arange(seq_len * HIDDEN_SIZE, dtype=torch.float32) * 0.011).sin() - 1.0
    return flat.view(1, seq_len, HIDDEN_SIZE)


def run_case(head: BoundaryHead, case: dict) -> dict:
    seq_len = case["seq_len"]
    q_count = case["q_count"]
    c_count = case["c_count"]
    text_states = synthetic_states(seq_len)
    text_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    text_mask[0, : case["valid_tokens"]] = True

    # Queries reuse text rows 0..q_count-1, as in the earlier stage oracles.
    query_states = text_states[:, :q_count].contiguous()
    query_mask = torch.ones(1, q_count, dtype=torch.bool)

    # Same candidate list for every query; the reference expands it to
    # [B, Q, C, 2] itself in `score_explicit_spans`.
    pairs = torch.tensor([case["candidates"]], dtype=torch.long)
    indices = pairs.view(1, 1, c_count, 2).expand(1, q_count, c_count, 2).contiguous()

    text_lengths = text_mask.sum(dim=1).long()
    starts = indices[..., 0]
    ends = indices[..., 1]
    legal = (
        (starts >= 0)
        & (ends > starts)
        & (ends <= text_lengths.view(1, 1, 1))
        & query_mask.unsqueeze(-1)
    )

    with torch.inference_mode():
        scores = head.score_explicit_spans(
            text_states, text_mask, query_states, query_mask, indices, legal
        )

    return {
        "seq_len": seq_len,
        "valid_tokens": case["valid_tokens"],
        "q_count": q_count,
        "c_count": c_count,
        "text_states": text_states.flatten().tolist(),
        "text_mask": text_mask.flatten().tolist(),
        "query_states": query_states.flatten().tolist(),
        "query_mask": query_mask.flatten().tolist(),
        "indices": indices.flatten().tolist(),
        "valid_mask": legal.flatten().tolist(),
        "scores": scores.flatten().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "models" / "gliner2.5-base-v1")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "gliner2.5-base-v1"
        / "score-explicit-spans-full-golden.json",
    )
    args = parser.parse_args()

    head = build_head(args.model_dir)
    settings = head.settings
    fixture = {
        "hidden_size": HIDDEN_SIZE,
        "config": {
            "boundary_dim": settings.boundary_dim,
            "pair_dim": settings.pair_dim,
            "content_dim": settings.content_dim,
            "content_soft_max_pool": settings.content_soft_max_pool,
            "multihead_pair_compat_heads": settings.multihead_pair_compat_heads,
            "enable_rotary_endpoints": settings.enable_rotary_endpoints,
            "rotary_base": settings.rotary_base,
            "reranker_endpoint_compat": settings.reranker_endpoint_compat,
            "use_inside_evidence": settings.use_inside_evidence,
            "query_conditioned_inside_weight": settings.query_conditioned_inside_weight,
            "enable_span_content": settings.enable_span_content,
            "endpoint_difference_features": settings.endpoint_difference_features,
        },
        "cases": [run_case(head, case) for case in CASES],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture) + "\n")
    total = sum(len(case["scores"]) for case in fixture["cases"])
    print(f"{args.out} (cases={len(fixture['cases'])}, scores={total})")


if __name__ == "__main__":
    main()
