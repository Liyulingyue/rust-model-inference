"""Golden for ``SharedPoolScorer`` — scoring the shared pool against every query.

The other half of the ``candidate_pool = "shared"`` mainline (see
``dump_document_candidate_pool.py`` for the candidate construction). Where
``SparseBoundaryPairScorer`` adds independent scalar terms per (query, span),
``SharedPoolScorer`` scores each candidate once as a ``pair_dim`` vector and
dot-products it with a per-query vector, then FiLM-conditions the candidate on
the query before a small MLP:

    candidate = start_rep + end_rep
              + length_projection([log1p(len), len/tokens, rsqrt(len)])
              + prior_projection(pooled.compat_logits)
              + content_projection(span_content)
    candidate = candidate_norm(candidate) * pool_mask
    score     = <candidate, query> / sqrt(pair_dim)
              + film_output(gelu(film_hidden(candidate * (1 + gamma) + beta)))
              + start_logit + end_logit + inside_interval / sqrt(width)

``BoundaryHead.forward`` calls it as
``shared_pool_scorer(encoding.states, query_states, query_mask, pooled,
start_logits, end_logits, inside_prefix, text_lengths, token_states, text_mask,
inside_prefix_mean=marginals.inside_prefix_mean)``.

``pair_logits`` comes back in the reference's candidate-major ``[B, C, Q]``
order, *not* the public ``[B, Q, C]`` contract — ``to_candidate_batch``
transposes it. The Rust port keeps the internal order and says so.

The fixture stores the full ``pair_logits`` for every slot, plus a sample of
``candidate`` rows per case. The score already depends on the candidate vector,
so the sample is a readability aid rather than extra coverage.

Regenerate with:
    PYTHONPATH=target/gliner2-oracle \\
        models/.venv/bin/python3 \\
        tools/oracle/gliner_boundary/dump_shared_pool_scorer.py
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    CASES,
    HIDDEN_SIZE,
    build_head,
    build_inputs,
    dump_json,
    fixture_dir,
    run_head_stages,
)

# How many valid candidate rows to store per case.
CANDIDATE_SAMPLE = 8


def run_case(head, case: dict) -> dict:
    inputs = build_inputs(case)
    stages = run_head_stages(head, inputs)
    encoding = stages["encoding"]
    marginals = stages["marginals"]

    with torch.inference_mode():
        pooled = head.shared_pool_builder(
            encoding.states,
            encoding.mask,
            inputs["query_mask"],
            marginals.start_logits,
            marginals.end_logits,
        )
        inside_prefix = (
            marginals.inside_prefix if head.use_inside_evidence else None
        )
        pair_logits, candidate = head.shared_pool_scorer(
            encoding.states,
            inputs["query_states"],
            inputs["query_mask"],
            pooled,
            marginals.start_logits,
            marginals.end_logits,
            inside_prefix,
            inputs["text_lengths"],
            inputs["text_states"],
            inputs["text_mask"],
            inside_prefix_mean=marginals.inside_prefix_mean,
        )

    # `BoundaryHead.forward` builds the *public* `candidate_states` separately
    # from the scorer's internal `candidate` (`model.py:441-447`): it projects the
    # two endpoint boundary states through `candidate_encoder`
    # (`Linear(2 * boundary_dim, hidden_size)`, no activation) and zeroes invalid
    # slots. The two tensors have different widths — the scorer's is `pair_dim`,
    # this one is `hidden_size` — and conflating them is what made our
    # `DocumentCandidateBatch::candidate_states` a misnomer.
    from gliner2.models.boundary.indexing import gather_rows

    pooled_candidate_states = None
    if head.candidate_encoder is not None:
        ps = gather_rows(encoding.states, pooled.indices[..., 0])
        pe = gather_rows(encoding.states, pooled.indices[..., 1])
        pooled_candidate_states = head.candidate_encoder(
            torch.cat((ps, pe), -1)
        ).masked_fill(~pooled.mask.unsqueeze(-1), 0.0)

    valid = pooled.mask[0].nonzero().flatten().tolist()
    sampled = valid[:CANDIDATE_SAMPLE]
    return {
        "seq_len": inputs["seq_len"],
        "valid_tokens": int(inputs["text_mask"].sum(dim=1)[0]),
        "q_count": inputs["q_count"],
        "text_states": inputs["text_states"].flatten().tolist(),
        "text_mask": inputs["text_mask"].flatten().tolist(),
        "query_states": inputs["query_states"].flatten().tolist(),
        "query_mask": inputs["query_mask"].flatten().tolist(),
        "pool_indices": pooled.indices.flatten().tolist(),
        "pool_mask": pooled.mask.flatten().tolist(),
        "pool_compat_logits": pooled.compat_logits.flatten().tolist(),
        "pair_logits": pair_logits.flatten().tolist(),
        "candidate_slots": sampled,
        # The scorer's internal, `pair_dim`-wide candidate vector.
        "candidate_rows": [
            candidate[0, slot].tolist() for slot in sampled
        ],
        # The public `candidate_states`: `hidden_size`-wide, for the record head.
        "candidate_state_rows": [
            pooled_candidate_states[0, slot].tolist() for slot in sampled
        ]
        if pooled_candidate_states is not None
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=fixture_dir() / "shared-pool-scorer-golden.json")
    args = parser.parse_args()

    head = build_head()
    settings = head.settings
    fixture = {
        "config": {
            "boundary_dim": settings.boundary_dim,
            "pair_dim": settings.pair_dim,
            "query_dim": HIDDEN_SIZE,
            "content_dim": settings.content_dim,
            "pool_size": settings.pool_size,
            "candidate_attention_layers": settings.candidate_attention_layers,
            "query_attention_layers": settings.query_attention_layers,
            "use_inside_evidence": settings.use_inside_evidence,
            "enable_span_content": settings.enable_span_content,
            "pair_temperature": settings.pair_temperature,
            "candidate_sample": CANDIDATE_SAMPLE,
        },
        "cases": [run_case(head, case) for case in CASES],
    }
    dump_json(args.out, fixture)


if __name__ == "__main__":
    main()
