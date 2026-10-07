"""Oracle for the GLiNER2.5 BoundaryExtractor relation head.

Dumps, from the reference implementation, everything
``src/models/gliner_boundary/relations.rs`` consumes:

  1. ``TypedRelationPairGenerator.generate`` — the typed, capped pair proposals
  2. ``SparseRelationScorer.forward``       — the per-pair logits
  3. ``BoundaryHead._build_rel_specs``     — the relation query states, so the
     ``directional_relation_states`` concatenation is pinned too

The generator is pure (it only reads the candidate batch), so its cases are
synthetic and need no model. The scorer needs real weights, so it runs on the
synthetic document's text states like the other stage oracles.

Cases deliberately cover the three ways a naive port goes wrong:

  * **ties.** Two mentions with identical probability, same span, different
    query, and adjacent spans with identical probability. The mention order is
    the tie-break for the score sort, so these pin it.
  * **padding.** Fewer qualifying arguments than ``heads_per_relation``, so the
    padded slots are exercised. A real head paired with a padded tail must be
    dropped (``pair_valid = hvalid & tvalid``), which is invisible unless the
    document runs out of arguments.
  * **same span.** A head and tail that are the same mention are removed, and
    the threshold boundary is probed at exactly ``argument_threshold``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "target" / "gliner2-oracle") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))

from gliner2.configuration import BoundaryHeadSettings  # noqa: E402
from gliner2.models.base import QueryLayout  # noqa: E402
from gliner2.models.boundary.model import BoundaryHead  # noqa: E402
from gliner2.models.boundary.relations import (  # noqa: E402
    RelationProposalSettings,
    RelationTypeSpec,
    SparseRelationScorer,
    TypedRelationPairGenerator,
)
from gliner2.models.outputs import CandidateTensorBatch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from common import HIDDEN_SIZE, fixture_dir, model_dir, synthetic_states  # noqa: E402

CASES = [
    # One relation type, head=query 0, tail=query 1. More candidates than the
    # cap, so the top-k and the `pair_cap` truncation both do work.
    {
        "name": "single_type_over_cap",
        "seq_len": 12,
        "q_count": 2,
        "c_count": 5,
        "candidates": [[[0, 2], [3, 5], [1, 4], [6, 9], [2, 3]],
                       [[0, 2], [3, 5], [7, 8], [2, 3], [6, 9]]],
        # Two mentions share a probability exactly, and two adjacent spans do
        # too, so the mention tie-break has to come from (start, end).
        "logits": [[2.0, 1.0, 0.5, -0.25, 0.5],
                   [2.0, 1.0, 0.5, 0.5, -0.25]],
        "valid": [[True] * 5, [True] * 5],
        "specs": [("works_at", 0, 1)],
    },
    # Fewer qualifying arguments than the cap: heads_per_relation is 32 for
    # base-v1, so this document's 2 candidates leave 30 padded slots. A head
    # paired with a padded tail must not survive the pair top-k.
    {
        "name": "padding_and_threshold",
        "seq_len": 8,
        "q_count": 2,
        "c_count": 2,
        "candidates": [[[1, 3], [4, 6]],
                       [[1, 3], [4, 6]]],
        # One candidate sits exactly on argument_threshold (0.2 -> logit
        # logit(0.2/0.8) = -1.3862943611...), one just under. The reference
        # uses `>=`, so the exact one qualifies.
        "logits": [[-1.3862943611198906, -2.0],
                   [-1.3862943611198906, -1.3862943611198906]],
        "valid": [[True, True], [True, True]],
        "specs": [("at", 0, 1)],
    },
    # Two relation types over the same mentions, plus a self pair
    # (query 0 candidate 0 against query 0 candidate 0) that must be removed.
    {
        "name": "two_types_and_self_span",
        "seq_len": 10,
        "q_count": 3,
        "c_count": 3,
        "candidates": [[[0, 2], [3, 4], [5, 8]],
                       [[0, 2], [3, 4], [5, 8]],
                       [[2, 3], [6, 9], [0, 2]]],
        "logits": [[3.0, 1.5, 0.5],
                   [3.0, 1.5, 0.5],
                   [2.0, 0.75, 1.25]],
        "valid": [[True] * 3, [True] * 3, [True] * 3],
        "specs": [("founded", 0, 1), ("located_in", 1, 2)],
    },
    # Invalid (padded) candidates mixed in, and a query the spec does not name,
    # so the role membership filter has to drop them.
    {
        "name": "invalid_mask_and_unnamed_query",
        "seq_len": 9,
        "q_count": 3,
        "c_count": 3,
        "candidates": [[[0, 2], [3, 4], [6, 8]],
                       [[0, 2], [3, 4], [6, 8]],
                       [[0, 2], [3, 4], [6, 8]]],
        "logits": [[4.0, 2.0, 1.0], [4.0, 2.0, 1.0], [9.0, 9.0, 9.0]],
        # Query 2's candidates are all invalid, so a spec naming it as head
        # must produce nothing.
        "valid": [[True, True, True], [True, True, True], [False, False, False]],
        "specs": [("named", 0, 1), ("dead_head", 2, 1)],
    },
]


def settings() -> BoundaryHeadSettings:
    config = json.loads((model_dir() / "config.json").read_text())
    return BoundaryHeadSettings(**config["boundary_head"])


def build_scorer(set_: BoundaryHeadSettings, hidden: int = HIDDEN_SIZE) -> SparseRelationScorer:
    scorer = SparseRelationScorer(
        hidden,
        dropout=0.0,
        relation_query_dim=(
            2 * hidden if set_.directional_relation_states else hidden
        ),
        use_biaffine_content=set_.relation_biaffine_content,
    )
    state_dict = load_file(str(model_dir() / "model.safetensors"))
    prefix = "relation_scorer."
    sub = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    scorer.load_state_dict(sub, strict=True)
    scorer.eval()
    return scorer


def build_query_states(seq_len: int, specs, hidden: int = HIDDEN_SIZE):
    """The `directional_relation_states` rule, exercised directly.

    ``_build_rel_specs`` sets a relation's query state to the concatenation of
    *its own* head and tail role states when ``directional_relation_states`` is
    set, and their mean otherwise (``model.py:1494-1498``). Porting the scorer
    without this rule would still pass a scorer-only fixture, so the oracle
    records the states themselves.
    """
    states = synthetic_states(seq_len)[0]
    out = []
    for spec in specs:
        head = states[spec.head_query_ids[0]]
        tail = states[spec.tail_query_ids[0]]
        if settings().directional_relation_states:
            out.append(torch.cat((head, tail), dim=-1))
        else:
            out.append(torch.stack((head, tail)).mean(dim=0))
    return torch.stack(out).unsqueeze(0)


def main() -> None:
    set_ = settings()
    generator = TypedRelationPairGenerator(
        RelationProposalSettings(
            heads_per_relation=set_.relation_heads_per_type,
            tails_per_relation=set_.relation_tails_per_type,
            pair_cap=set_.relation_pair_cap,
            argument_threshold=set_.relation_argument_proposal_threshold,
        )
    )
    scorer = build_scorer(set_)
    relation_dim = 2 * HIDDEN_SIZE if set_.directional_relation_states else HIDDEN_SIZE

    records = []
    for case in CASES:
        q_count, c_count = case["q_count"], case["c_count"]
        indices = torch.tensor(case["candidates"], dtype=torch.long).view(
            1, q_count, c_count, 2
        )
        candidates = CandidateTensorBatch(
            indices=indices,
            proposal_logits=torch.zeros(1, q_count, c_count),
            pair_logits=torch.tensor(case["logits"], dtype=torch.float32).view(
                1, q_count, c_count
            ),
            valid_mask=torch.tensor(case["valid"], dtype=torch.bool).view(
                1, q_count, c_count
            ),
            query_mask=torch.ones(1, q_count, dtype=torch.bool),
            candidate_states=torch.zeros(1, q_count, c_count, HIDDEN_SIZE),
        )
        specs = [
            RelationTypeSpec(name, head_query_ids=(head,), tail_query_ids=(tail,))
            for name, head, tail in case["specs"]
        ]
        pairs = generator.generate(candidates, [QueryLayout(queries=())], specs)
        query_states = build_query_states(case["seq_len"], specs)
        logits = scorer(
            synthetic_states(case["seq_len"]),
            query_states,
            candidates,
            pairs,
        )
        records.append({
            "name": case["name"],
            "seq_len": case["seq_len"],
            "q_count": q_count,
            "c_count": c_count,
            "candidates": case["candidates"],
            "logits": case["logits"],
            "valid": case["valid"],
            "specs": [[name, head, tail] for name, head, tail in case["specs"]],
            "pair_count": len(pairs),
            "relation_types": list(pairs.relation_types),
            "pairs": [
                [
                    int(pairs.head_start[i]), int(pairs.head_end[i]),
                    int(pairs.tail_start[i]), int(pairs.tail_end[i]),
                    float(pairs.head_prob[i]), float(pairs.tail_prob[i]),
                    float(logits[i]),
                ]
                for i in range(len(pairs))
            ],
        })
        print(
            f"{case['name']}: {len(pairs)} pair(s) "
            f"{list(pairs.relation_types)}",
            file=sys.stderr,
        )

    payload = {
        "source": "typed_relation_pair_generator + sparse_relation_scorer",
        "hidden_size": HIDDEN_SIZE,
        "relation_query_dim": relation_dim,
        "directional_relation_states": set_.directional_relation_states,
        "relation_biaffine_content": set_.relation_biaffine_content,
        "heads_per_relation": set_.relation_heads_per_type,
        "tails_per_relation": set_.relation_tails_per_type,
        "pair_cap": set_.relation_pair_cap,
        "argument_threshold": set_.relation_argument_proposal_threshold,
        "cases": records,
    }
    out = fixture_dir() / "relations-golden.json"
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {out} ({len(records)} cases)", file=sys.stderr)


if __name__ == "__main__":
    main()
