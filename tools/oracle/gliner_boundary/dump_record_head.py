"""Oracle for the record head: ``forward_group`` and ``decode_group``.

Dumps ``RecordHead.forward_group`` and ``decode_group`` over synthetic
candidates, so the whole instance-formation plus assignment path is pinned
without a model forward pass. The weights are the checkpoint's own
``record_decoder.*``; only the *inputs* are synthetic.

The synthetic candidates follow the pattern the relations oracle uses: a
document of ``seq_len`` words, queries that are its first rows, and a candidate
batch whose ``candidate_states`` are derived from the same sine pattern so the
values are reproducible from the fixture alone.

Cases, chosen for where the three modes and the assignment solver diverge:

``natural_*``
    Instances are the anchor field's candidates and the object logits are the
    pool's ``pair_logits`` *straight through* — ``object_head`` is not applied.
    A case where applying it would change the answer is included by giving the
    anchor a distinctive ``pair_logits`` spread.
``latent_*``
    Every field's candidates seed an instance, scored by ``latent_seed_head``.
    ``natural`` and ``latent`` therefore produce different instance counts for
    the same candidates, which is the thing to pin.
``anchorless_*``
    ``instance_embed`` is the instance state, refined by one attention pass over
    all candidates, and gated by ``object_threshold`` rather than
    ``anchor_threshold``. The case sets the two thresholds *differently* so a
    decoder that used the wrong one is caught.
``assignment_*``
    The exclusive-field solver. ``allows_absent`` false makes the ABSENT column
    one emergency cost broadcast to every row — a per-row max would let a row
    with cheap candidates escape an assignment a row with expensive ones has to
    take, so the case has rows with deliberately different cost scales.
``list_*``
    List fields do not go through the solver: exclusive ones award each candidate
    to its single strongest instance, non-exclusive ones threshold per candidate.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "target" / "gliner2-oracle") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from safetensors.torch import load_file  # noqa: E402

from common import HIDDEN_SIZE, build_head, fixture_dir, model_dir  # noqa: E402
from gliner2.models.base import QueryLayout, QuerySpec  # noqa: E402
from gliner2.models.boundary.records import decode_group  # noqa: E402
from gliner2.models.outputs import CandidateTensorBatch  # noqa: E402
from gliner2.processing.records import compile_record_specs  # noqa: E402


def build_record_head(settings):
    from gliner2.models.boundary.records import RecordHead

    head = RecordHead(
        HIDDEN_SIZE, settings.record_dim, settings.record_instance_queries
    )
    state = load_file(str(model_dir() / "model.safetensors"))
    prefix = "record_decoder."
    head.load_state_dict(
        {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)},
        strict=True,
    )
    head.eval()
    return head


# The synthetic candidate states need a scale and offset that actually exercise
# the head, and both values were chosen by measurement rather than taste. The
# shared document sits in roughly [-2, 0], so a naive mean lands every
# `latent_seed_head` logit well below the 0.5 threshold and *no* latent instance
# survives — the fixture then decodes zero records while still passing, because
# "the head agreed with the reference on nothing" is indistinguishable from
# agreement. That happened twice, once before the states became query-independent
# and once after, which is why `the_fixture_keeps_its_discriminating_cases` now
# fails if any case drops to zero records.
#
# At these values the seed probabilities come out around [0.33, 0.63, 0.90]: two
# of three clear the threshold, so both the selecting and the rejecting branch
# are exercised. `scale=3.0` pushes all three below (0.11 / 0.13 / 0.38), and
# `scale=1.0, off=1.0` leaves only one above.
#
# The Rust test rebuilds these from the same formula, so both constants are part
# of the contract and appear in `tests/gliner2_5_base_v1_record_head_parity.rs`.
CANDIDATE_STATE_SCALE = 1.0
CANDIDATE_STATE_OFFSET = 2.0


def states(seq_len: int) -> torch.Tensor:
    """`(arange(seq_len * hidden) * 0.011).sin() - 1.0`, the shared synthetic doc."""
    flat = (torch.arange(seq_len * HIDDEN_SIZE, dtype=torch.float32) * 0.011).sin() - 1.0
    return flat.view(1, seq_len, HIDDEN_SIZE)


def build_case(case: dict):
    """A `[1, Q, C]` candidate batch with 768-wide candidate states.

    The state depends on the *slot* only, never on the query or on that query's
    span. The real pool's `candidate_states` are query-independent: the pool is
    built per document, and `to_candidate_batch` reaches `[B, Q, C, H]` with an
    `expand`, a broadcast view whose values do not depend on `q`. Deriving the
    synthetic state from a per-query span would produce a fixture the production
    candidate-major layout cannot represent at all, and `main` asserts the
    query-independence so a future edit cannot reintroduce it.
    """
    q_count, c_count = case["q_count"], case["c_count"]
    doc = states(case["seq_len"])[0]
    indices = torch.tensor(case["candidates"], dtype=torch.long).view(1, q_count, c_count, 2)
    candidate_states = torch.zeros(1, q_count, c_count, HIDDEN_SIZE)
    seq_len = case["seq_len"]
    for c in range(c_count):
        mid = 1 + (c * 2) % max(seq_len - 2, 1)
        span = doc[mid : min(mid + 2, seq_len)]
        state = span.mean(dim=0) * CANDIDATE_STATE_SCALE + CANDIDATE_STATE_OFFSET * (c % 5)
        for q in range(q_count):
            candidate_states[0, q, c] = state
    return CandidateTensorBatch(
        indices=indices,
        proposal_logits=torch.zeros(1, q_count, c_count),
        pair_logits=torch.tensor(case["logits"], dtype=torch.float32).view(
            1, q_count, c_count
        ),
        valid_mask=torch.tensor(case["valid"], dtype=torch.bool).view(1, q_count, c_count),
        query_mask=torch.ones(1, q_count, dtype=torch.bool),
        candidate_states=candidate_states,
    )


def make_layout(roles):
    return QueryLayout(
        queries=tuple(
            QuerySpec(
                query_id=i,
                task_index=0,
                task_type="json_structures",
                task_name="record",
                role_index=i,
                role_name=name,
            )
            for i, name in enumerate(roles)
        )
    )


# `record_*` settings come from the checkpoint, so the thresholds below are the
# reference's own; the decode knobs are per case.
CASES = [
    {
        "name": "natural_two_fields",
        "seq_len": 10,
        "q_count": 3,
        "c_count": 3,
        "roles": ["name", "employer", "city"],
        "candidates": [[[0, 2], [2, 4], [5, 7]], [[0, 2], [3, 6], [7, 9]],
                       [[1, 3], [4, 5], [8, 10]]],
        # A wide spread on the anchor field, so applying `object_head` instead of
        # passing the pair logits through would change which instances survive.
        "logits": [[4.0, 2.0, -1.0], [0.5, 1.5, 0.0], [1.0, -0.5, 2.0]],
        "valid": [[True, True, True], [True, True, True], [True, True, True]],
        "record_metadata": {
            "record": {
                "mode": "natural",
                "anchor": "name",
                "fields": {
                    "employer": {"cardinality": "optional_one", "exclusive": True},
                    "city": {"cardinality": "zero_or_more"},
                },
            }
        },
        "field_dtypes": {},
        "decode": {},
    },
    {
        "name": "natural_anchor_absent_from_candidates",
        # Every anchor candidate is invalid, so there are no instances at all and
        # the non-anchor fields must not invent any.
        "seq_len": 8,
        "q_count": 2,
        "c_count": 3,
        "roles": ["name", "employer"],
        "candidates": [[[0, 1], [2, 3], [4, 5]], [[1, 3], [3, 5], [5, 7]]],
        "logits": [[9.0, 9.0, 9.0], [1.0, 2.0, 3.0]],
        "valid": [[False, False, False], [True, True, True]],
        "record_metadata": {
            "record": {"mode": "natural", "anchor": "name",
                       "fields": {"employer": {"cardinality": "optional_one"}}}
        },
        "field_dtypes": {},
        "decode": {},
    },
    {
        "name": "latent_seeds_every_field",
        # Same shape as natural_two_fields but latent: 9 instances (3 fields x 3
        # candidates) instead of 3, and the object logits come from
        # `latent_seed_head` rather than the pool.
        "seq_len": 10,
        "q_count": 3,
        "c_count": 3,
        "roles": ["name", "employer", "city"],
        "candidates": [[[0, 2], [2, 4], [5, 7]], [[0, 2], [3, 6], [7, 9]],
                       [[1, 3], [4, 5], [8, 10]]],
        "logits": [[4.0, 2.0, -1.0], [0.5, 1.5, 0.0], [1.0, -0.5, 2.0]],
        "valid": [[True, True, True], [True, True, True], [True, True, True]],
        "record_metadata": {
            "record": {
                "mode": "latent",
                "fields": {
                    "name": {"cardinality": "required_one"},
                    "employer": {"cardinality": "required_one", "exclusive": True},
                    "city": {"cardinality": "zero_or_more"},
                },
            }
        },
        "field_dtypes": {},
        "decode": {},
    },
    {
        "name": "anchorless_uses_object_threshold",
        # `anchor_threshold` and `object_threshold` are set far apart: a decoder
        # that gated anchorless instances with the anchor threshold would select a
        # different set, so this pins which one is read.
        "seq_len": 9,
        "q_count": 2,
        "c_count": 3,
        "roles": ["name", "employer"],
        "candidates": [[[0, 2], [3, 5], [6, 8]], [[1, 3], [4, 6], [7, 9]]],
        "logits": [[2.0, 1.0, 0.0], [1.5, 0.5, 2.5]],
        "valid": [[True, True, True], [True, True, True]],
        "record_metadata": {
            "record": {"mode": "anchorless",
                       "fields": {"name": {"cardinality": "optional_one"},
                                  "employer": {"cardinality": "zero_or_more"}}}
        },
        "field_dtypes": {},
        "decode": {"anchor_threshold": 0.99, "object_threshold": 0.1},
    },
    {
        # Every candidate invalid, so `ctx` is empty and `_anchorless_states`
        # returns `instance_embed` untouched. That isolates the parameter from the
        # attention pooling: if `instance_embed` is read with the wrong layout the
        # 32 rows are a permutation of each other, and because the table is
        # `randn * 0.02` the resulting object logits differ by only ~0.01 — small
        # enough to hide behind a loose tolerance and large enough to fail a
        # strict one. Deltas of that size across all 32 rows, with no structure,
        # are the signature of a permutation rather than of numerics.
        "name": "anchorless_without_candidates_isolates_instance_embed",
        "seq_len": 8,
        "q_count": 2,
        "c_count": 2,
        "roles": ["name", "employer"],
        "candidates": [[[0, 2], [3, 5]], [[0, 2], [3, 5]]],
        "logits": [[1.0, 1.0], [1.0, 1.0]],
        "valid": [[False, False], [False, False]],
        "record_metadata": {
            "record": {"mode": "anchorless",
                       "fields": {"name": {"cardinality": "optional_one"},
                                  "employer": {"cardinality": "zero_or_more"}}}
        },
        "field_dtypes": {},
        "decode": {"anchor_threshold": 0.5, "object_threshold": 0.01},
    },
    {
        "name": "assignment_required_one_uses_broadcast_emergency",
        # `allows_absent = false` (required_one) makes the ABSENT column a single
        # emergency cost broadcast to every row. The rows have very different
        # scales, so a per-row max instead of a broadcast would let the row with
        # cheap candidates skip the assignment the other row is forced into.
        "seq_len": 12,
        "q_count": 2,
        "c_count": 4,
        "roles": ["name", "employer"],
        "candidates": [[[0, 2], [3, 5], [6, 8], [9, 11]], [[0, 2], [3, 5], [6, 8], [9, 11]]],
        "logits": [[5.0, 5.0, 5.0, 5.0], [0.0, 0.0, 0.0, 0.0]],
        "valid": [[True] * 4, [True] * 4],
        "record_metadata": {
            "record": {"mode": "natural", "anchor": "name",
                       "fields": {"employer": {"cardinality": "required_one",
                                               "exclusive": True}}}
        },
        "field_dtypes": {},
        "decode": {},
    },
    {
        "name": "assignment_optional_one_allows_absent",
        # The same shape but optional_one, so a row whose best candidate is weak
        # takes the ABSENT column instead of being forced onto it.
        "seq_len": 12,
        "q_count": 2,
        "c_count": 4,
        "roles": ["name", "employer"],
        "candidates": [[[0, 2], [3, 5], [6, 8], [9, 11]], [[0, 2], [3, 5], [6, 8], [9, 11]]],
        "logits": [[5.0, 5.0, 5.0, 5.0], [0.0, 0.0, 0.0, 0.0]],
        "valid": [[True] * 4, [True] * 4],
        "record_metadata": {
            "record": {"mode": "natural", "anchor": "name",
                       "fields": {"employer": {"cardinality": "optional_one",
                                               "exclusive": True}}}
        },
        "field_dtypes": {},
        "decode": {},
    },
    {
        "name": "list_exclusive_awards_candidate_to_strongest",
        "seq_len": 11,
        "q_count": 2,
        "c_count": 3,
        "roles": ["name", "city"],
        "candidates": [[[0, 2], [3, 5], [6, 8]], [[0, 2], [3, 5], [6, 8]]],
        "logits": [[3.0, 1.0, 0.5], [3.0, 1.0, 0.5]],
        "valid": [[True] * 3, [True] * 3],
        "record_metadata": {
            "record": {"mode": "natural", "anchor": "name",
                       "fields": {"city": {"cardinality": "zero_or_more",
                                           "exclusive": True}}}
        },
        "field_dtypes": {},
        "decode": {},
    },
    {
        "name": "list_non_exclusive_thresholds_each_candidate",
        "seq_len": 11,
        "q_count": 2,
        "c_count": 3,
        "roles": ["name", "city"],
        "candidates": [[[0, 2], [3, 5], [6, 8]], [[0, 2], [3, 5], [6, 8]]],
        "logits": [[3.0, 1.0, 0.5], [3.0, 1.0, 0.5]],
        "valid": [[True] * 3, [True] * 3],
        "record_metadata": {
            "record": {"mode": "natural", "anchor": "name",
                       "fields": {"city": {"cardinality": "zero_or_more"}}}
        },
        "field_dtypes": {},
        "decode": {},
    },
    {
        "name": "natural_orders_records_by_anchor_span",
        # The anchor field's candidates run *backwards* in the document, so a
        # decoder that kept the probability order instead of sorting by anchor
        # span produces the same set in a different order.
        "seq_len": 10,
        "q_count": 1,
        "c_count": 3,
        "roles": ["name"],
        "candidates": [[[6, 8], [0, 2], [3, 5]]],
        "logits": [[3.0, 3.0, 3.0]],
        "valid": [[True, True, True]],
        "record_metadata": {"record": {"mode": "natural", "anchor": "name"}},
        "field_dtypes": {},
        "decode": {},
    },
    {
        "name": "latent_dedups_identical_field_sets",
        # Two instances that bind the same spans collapse to one, the
        # higher-scoring one winning. A non-deduplicating decoder reports both.
        "seq_len": 8,
        "q_count": 2,
        "c_count": 2,
        "roles": ["name", "employer"],
        "candidates": [[[0, 2], [4, 6]], [[0, 2], [4, 6]]],
        "logits": [[3.0, 3.0], [3.0, 3.0]],
        "valid": [[True, True], [True, True]],
        "record_metadata": {
            "record": {"mode": "latent",
                       "fields": {"name": {"cardinality": "required_one"},
                                  "employer": {"cardinality": "zero_or_more"}}}
        },
        "field_dtypes": {},
        "decode": {},
    },
]


def main() -> None:
    head = build_head()
    settings = head.settings
    record_head = build_record_head(settings)

    records = []
    for case in CASES:
        candidates = build_case(case)
        layout = make_layout(case["roles"])
        specs = compile_record_specs(
            query_layout=layout,
            record_metadata=case["record_metadata"],
            field_dtypes=case["field_dtypes"],
        )
        spec = next(iter(specs.values()))
        query_states = states(case["seq_len"])[0, : case["q_count"]].contiguous()

        with torch.inference_mode():
            group = record_head.forward_group(spec, query_states, candidates, 0)
            decode_kwargs = dict(
                anchor_threshold=settings.record_anchor_threshold,
                field_threshold=settings.record_field_threshold,
                object_threshold=settings.record_anchor_proposal_threshold,
                temperature=settings.record_temperature,
            )
            decode_kwargs.update(case["decode"])
            decoded = decode_group(group, **decode_kwargs)

        # Assert the property the candidate-major layout depends on, so a future
        # edit that makes the synthetic states query-dependent fails here rather
        # than as a confusing mismatch in the Rust test.
        for q in range(case["q_count"]):
            for c in range(case["c_count"]):
                if not torch.equal(
                    candidates.candidate_states[0, q, c],
                    candidates.candidate_states[0, 0, c],
                ):
                    raise ValueError(
                        f"{case['name']}: candidate_states differ across queries "
                        f"at slot {c}, but the production layout is candidate-major"
                    )
        records.append({
            "name": case["name"],
            "seq_len": case["seq_len"],
            "q_count": case["q_count"],
            "c_count": case["c_count"],
            "roles": case["roles"],
            "candidates": case["candidates"],
            "logits": case["logits"],
            "valid": case["valid"],
            "record_metadata": case["record_metadata"],
            "decode": {
                "anchor_threshold": decode_kwargs["anchor_threshold"],
                "field_threshold": decode_kwargs["field_threshold"],
                "object_threshold": decode_kwargs["object_threshold"],
                "temperature": decode_kwargs["temperature"],
            },
            "mode": spec.mode,
            "anchor_query_id": spec.anchor_query_id,
            "field_specs": [
                [f.query_id, f.name, f.cardinality.value, f.is_anchor, f.exclusive]
                for f in spec.fields
            ],
            "object_logits": [float(v) for v in group.object_logits.detach()],
            "assign_logits": [
                [float(v) for v in row] for rows in group.assign_logits for row in rows
            ],
            "assign_shape": [
                len(group.assign_logits),
                group.assign_logits[0].shape[0] if group.assign_logits else 0,
                group.assign_logits[0].shape[1] if group.assign_logits else 0,
            ],
            "instance_spans": [
                None if s is None else [int(s[0]), int(s[1])]
                for s in group.instance_spans
            ],
            "records": [
                {
                    "anchor_span": None if r.anchor_span is None
                    else [int(r.anchor_span[0]), int(r.anchor_span[1])],
                    "score": float(r.score),
                    "fields": {
                        str(qid): [[int(s[0]), int(s[1])] for s in spans]
                        for qid, spans in sorted(r.fields.items())
                    },
                    "field_scores": {
                        str(qid): [float(v) for v in scores]
                        for qid, scores in sorted(r.field_scores.items())
                    },
                }
                for r in decoded
            ],
        })
        print(
            f"  {case['name']}: {group.object_logits.shape[0]} instance(s) -> "
            f"{len(decoded)} record(s)",
            file=sys.stderr,
        )

    out = fixture_dir() / "record-head-golden.json"
    out.write_text(json.dumps({
        "hidden_size": HIDDEN_SIZE,
        "record_dim": settings.record_dim,
        "instance_queries": settings.record_instance_queries,
        "cases": records,
    }, indent=1) + "\n")
    print(f"wrote {out} ({len(records)} cases)", file=sys.stderr)


if __name__ == "__main__":
    main()
