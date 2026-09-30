"""Golden for ``DocumentCandidatePool`` — the shared document span pool.

``gliner2.5-base-v1`` sets ``candidate_pool = "shared"``, so this module
(``boundary/pool.py:107``), not ``SparseBoundaryProposer``, decides which
spans the model actually considers. ``BoundaryHead.forward`` calls it as
``shared_pool_builder(encoding.states, encoding.mask, query_mask,
start_logits, end_logits)`` at inference: no gold pairs, no stats.

The algorithm, in order:

  1. Union the per-query start / end marginals with ``amax`` over queries, so
     the pool is query-agnostic but still evidence-driven.
  2. Take the top ``pool_boundary_top_k`` starts and ends (32 for base-v1,
     clamped down to ``n_boundaries``). ``stable=True`` means ties break
     toward the lower boundary index.
  3. Pair them as a Cartesian product, keeping ``end > start``.
  4. Score each pair by ``(start_proj * end_proj).sum(-1) / sqrt(boundary_dim)``
     plus both union marginals.
  5. Reserve each query's top ``min_pool_per_query`` pairs (8) with a score
     band above every global score, so a query cannot be crowded out.
  6. Deduplicate by ``start * n + end`` keeping the best-scoring occurrence,
     then take the top ``pool_size`` (192) by score.

Every step is order-sensitive: ``_deduplicate_pool`` sorts twice with a stable
comparator, so a Rust port that breaks a tie differently produces a different
pool and a wrong score for reasons that look like a numeric bug.

Regenerate with:
    PYTHONPATH=target/gliner2-oracle \\
        models/.venv/bin/python3 \\
        tools/oracle/gliner_boundary/dump_document_candidate_pool.py
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    CASES,
    build_head,
    build_inputs,
    dump_json,
    fixture_dir,
    run_head_stages,
)


def run_case(head, case: dict) -> dict:
    inputs = build_inputs(case)
    stages = run_head_stages(head, inputs)
    encoding = stages["encoding"]
    marginals = stages["marginals"]
    builder = head.shared_pool_builder
    with torch.inference_mode():
        pooled = builder(
            encoding.states,
            encoding.mask,
            inputs["query_mask"],
            marginals.start_logits,
            marginals.end_logits,
        )
    assert pooled.gold_mask is None and pooled.stats is None
    return {
        "seq_len": inputs["seq_len"],
        "valid_tokens": int(inputs["text_mask"].sum(dim=1)[0]),
        "q_count": inputs["q_count"],
        "text_states": inputs["text_states"].flatten().tolist(),
        "text_mask": inputs["text_mask"].flatten().tolist(),
        "query_states": inputs["query_states"].flatten().tolist(),
        "query_mask": inputs["query_mask"].flatten().tolist(),
        "indices": pooled.indices.flatten().tolist(),
        "mask": pooled.mask.flatten().tolist(),
        "proposal_logits": pooled.proposal_logits.flatten().tolist(),
        "compat_logits": pooled.compat_logits.flatten().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=fixture_dir() / "document-candidate-pool-golden.json")
    args = parser.parse_args()

    head = build_head()
    settings = head.settings
    fixture = {
        "config": {
            "boundary_dim": settings.boundary_dim,
            "pool_boundary_top_k": settings.pool_boundary_top_k,
            "pool_size": settings.pool_size,
            "min_pool_per_query": settings.min_pool_per_query,
        },
        "cases": [run_case(head, case) for case in CASES],
    }
    dump_json(args.out, fixture)


if __name__ == "__main__":
    main()
