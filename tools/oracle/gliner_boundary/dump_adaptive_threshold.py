"""Oracle for the count-head-guided candidate admission (`adaptive_threshold`).

Every shipped boundary checkpoint sets ``boundary_head.adaptive_threshold =
false``, so this path contributes nothing to the 12/12 parity results — it is
dead code *for those checkpoints* and a live decode for any future one that
turns it on. The algorithm is still part of the family's feature surface, so it
is pinned here on real candidates and real count-head output.

What this pins
--------------
``_group_scored_candidates`` (``boundary/model.py:949-1023``) is the only
consumer of ``count_log_rates``, and its adaptive branch is a **union**, never
a filter::

    predicted_count = round(exp(count_log_rates)).clamp(0, C)
    keep = (eligible & probs >= threshold) | (eligible & rank < predicted_count)

Three properties a port has to get right, and each is easy to get wrong:

1. **Union, not replace.** Count guidance can *add* candidates the threshold
   rejected; it can never remove one the threshold kept. An implementation that
   computes ``predicted_count`` and truncates to it is not this function.
2. **Rank is over *eligible* candidates only.** The rank tensor is built from
   ``probs.masked_fill(~eligible, MASK_LOGIT)``, so padding slots consume no rank.
   Ranking the raw probability vector instead shifts every rank by however many
   padded slots precede the candidate.
3. **Rank ties break by index** (``argsort(..., stable=True)`` on a descending
   sort). Without stability, equal-probability candidates swap rank and the
   admitted set becomes an artifact of the sort implementation.

The fixture carries three admissions per case: the threshold-only baseline the
shipped checkpoints actually run, the adaptive admission driven by the
checkpoint's own count head, and an adaptive admission driven by *synthetic*
count rates. The synthetic rates are what make the union observable: the real
head's rates are a property of the checkpoint, and a case that only exercised
them would pass just as well with the branch deleted.
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402

from common import (  # noqa: E402
    DEFAULT_MODEL, build_head, dump_json, encoder_dir, fixture_dir,
)
from dump_extract_spans_end_to_end import (  # noqa: E402
    load_checkpoint_encoder,
)
from gliner2.models.boundary.model import _group_scored_candidates  # noqa: E402
from gliner2.processor import SchemaTransformer  # noqa: E402


# ``count_log_rates`` the head actually produces is a Poisson log-rate, so
# ``exp`` of it lands in [0, ~C]. A threshold high enough to reject most
# candidates is what leaves room for the count branch to add any: at 0.02 the
# baseline keeps almost everything and the union is a no-op that certifies
# nothing.
CASES = [
    (
        "Marie Curie worked in Paris with Pierre Curie in London.",
        {"json_structures": [{"person": {"name": [], "city": []}}]},
        0.9,
    ),
    (
        "Marie Curie worked in Paris and later in London.",
        {"json_structures": [{"person": {"name": [], "city": []}}]},
        0.7,
    ),
    (
        # Entities rather than structures, so the case does not depend on the
        # `[C]` group routing.
        "Marie Curie worked in Paris with Pierre Curie in London.",
        {"entities": {"person": {}, "city": {}}},
        0.9,
    ),
    (
        # Low threshold: the baseline survives, and the union must be a superset
        # of it rather than a replacement. A port that truncates to
        # ``predicted_count`` fails here even though it can pass a high
        # threshold.
        "Marie Curie worked in Paris with Pierre Curie in London.",
        {"entities": {"person": {}, "city": {}}},
        0.02,
    ),
    (
        "nothing structured here at all",
        {"entities": {"person": {}, "city": {}}},
        0.99,
    ),
]

# ``exp`` of these are the predicted counts. Zero is the case that must change
# nothing (rank < 0 is empty), and the large one is the case that must admit
# every eligible candidate regardless of the threshold.
SYNTHETIC_RATES = [0.0, 1.0, 2.0, 3.0]

# `_group_scored_candidates` is a pure function over a `CandidateTensorBatch`, so
# the properties the real encoder cannot be made to exhibit are pinned with
# hand-built candidate batches. On every real case above the invalid slots'
# pair logits saturate negative (probability 0.0 at index 191), which makes
# `masked_fill(~eligible, MASK_LOGIT)` a no-op — so a port that ranked the raw
# probability vector instead of the masked one would pass all of them.
#
# Every rate below is `ln(n)` for a small `n`, so `exp(rate)` rounds to exactly
# that `n` and the admitted set is decided by *ranking* rather than by the
# clamp. A rate large enough to clamp to the candidate count admits every
# eligible slot and would pass a port that never ranks at all.
#
# Each entry is `(name, pair_logits, valid_mask, count_rate, threshold)`.
LN2 = math.log(2.0)
LN3 = math.log(3.0)
SYNTHETIC_BATCHES = [
    (
        # Interleaved validity with the *invalid* slots scoring highest, and a
        # predicted count of 2. Masked, the two highest *eligible* candidates
        # (5.0, 4.0) take ranks 0 and 1. Ranked unmasked, ranks 0 and 1 go to
        # the invalid slots at 9.0 and 8.0, so `eligible & rank < 2` selects
        # nothing and the admitted set collapses to the threshold hit alone.
        "masked_rank_interleaved",
        [9.0, 5.0, 8.0, 4.0, 1.0, 0.5],
        [False, True, False, True, True, False],
        LN2,
        0.99,
    ),
    (
        # All six eligible candidates share one logit, so every rank is a tie
        # and the admitted three are decided purely by the stable descending
        # sort's index order. An unstable sort still admits three candidates
        # and still looks right, but not this set.
        "all_tied",
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        [True, True, True, True, True, True],
        LN3,
        0.99,
    ),
    (
        # Two invalid slots tied with the eligible ones at the top, and a
        # threshold no candidate clears, so the count branch is the only thing
        # that can admit anything. Masked it admits the two eligible leaders;
        # unmasked both ranks 0 and 1 land on invalid slots and the result is
        # empty.
        "tie_with_padding",
        [7.0, 7.0, 7.0, 7.0, 1.0, 1.0],
        [False, False, True, True, True, True],
        LN2,
        0.9999,
    ),
    (
        # `exp(1.0) = 2.718` rounds to 3, so the rate itself is not an integer.
        # A port that rounds the log-rate before exponentiating gets
        # `round(1.0) = 1` and admits a single candidate; the reference
        # exponentiates first and admits three.
        "exp_before_round",
        [3.0, 2.5, 2.0, 1.5, 1.0, 0.5],
        [True, True, True, True, True, True],
        1.0,
        0.99,
    ),
    (
        # A count of zero alongside a threshold that keeps every candidate: the
        # union must not remove them. This is the case a truncating port fails,
        # and the only one here where `threshold_only` is non-empty.
        "zero_count_keeps_threshold_hits",
        [4.0, 3.0, 0.1, 0.1, 0.1, 0.1],
        [True, True, True, True, True, True],
        -12.0,
        0.5,
    ),
    (
        # A negative rate far enough that `exp` underflows toward zero, which
        # must still round to a count of zero rather than to one.
        "underflowing_rate",
        [4.0, 3.0, 0.1, 0.1, 0.1, 0.1],
        [True, True, True, True, True, True],
        -100.0,
        0.5,
    ),
    (
        # `exp(12)` is ~162754, far past the six candidates, so the clamp to `C`
        # is what bounds the count. Observational note: because `rank` cannot
        # reach the clamped value, this case passes whether or not the clamp is
        # implemented — it is here to pin the reference's answer, not to
        # discriminate the clamp.
        "count_clamped_to_c",
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        [True, True, True, True, True, True],
        12.0,
        0.99,
    ),
]


def run_synthetic(name, pair_logits, valid_mask, rate, threshold):
    """`_group_scored_candidates` over a hand-built `[1, 1, C]` candidate batch.

    The single query is the only shape the boundary decode uses, and keeping it
    to one query per batch means a failure names the property rather than a
    coordinate in a query grid.
    """
    from gliner2.models.outputs import CandidateTensorBatch

    count = len(pair_logits)
    indices = torch.arange(count, dtype=torch.long).view(1, 1, count, 1)
    indices = indices.expand(1, 1, count, 2).contiguous()
    candidates = CandidateTensorBatch(
        indices=indices,
        proposal_logits=None,
        pair_logits=torch.tensor([[pair_logits]], dtype=torch.float32),
        valid_mask=torch.tensor([[valid_mask]], dtype=torch.bool),
        query_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    counts = torch.tensor([[float(rate)]], dtype=torch.float32)
    baseline = _group_scored_candidates(candidates, threshold=threshold)[0][0]
    adaptive = _group_scored_candidates(
        candidates,
        threshold=threshold,
        count_log_rates=counts,
        adaptive_threshold=True,
    )[0][0]
    return {
        "name": name,
        "pair_logits": [float(v) for v in pair_logits],
        "valid_mask": [bool(v) for v in valid_mask],
        "count_log_rate": float(rate),
        "threshold": threshold,
        "threshold_only": [[float(p), int(a), int(b)] for p, a, b in baseline],
        "adaptive": [[float(p), int(a), int(b)] for p, a, b in adaptive],
    }


def run_case(processor, encoder, head, text, schema, threshold):
    batch = processor._collate_batch(
        [(text, copy.deepcopy(schema))],
        max_len=None,
        error_policy="raise",
        build_targets=False,
    )
    with torch.inference_mode():
        hidden = encoder(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask
        ).last_hidden_state
    width = hidden.shape[-1]

    def gather_routed(indices, mask):
        safe = indices.clamp(0, hidden.shape[1] - 1)
        states = hidden.gather(1, safe.unsqueeze(-1).expand(-1, -1, width))
        return states * mask.unsqueeze(-1).to(states.dtype)

    text_states = gather_routed(batch.text_word_indices, batch.text_word_mask)
    query_states = gather_routed(batch.query_marker_indices, batch.query_marker_mask)
    with torch.inference_mode():
        out = head(
            text_states,
            batch.text_word_mask,
            query_states,
            batch.query_marker_mask,
            return_candidates=True,
        )
    candidates = out.candidates
    counts = out.count_log_rates
    if counts is None:
        raise ValueError("count head is disabled; adaptive_threshold cannot run")

    baseline = _group_scored_candidates(candidates, threshold=threshold)[0]
    with_counts = _group_scored_candidates(
        candidates,
        threshold=threshold,
        count_log_rates=counts,
        adaptive_threshold=True,
    )[0]
    synthetic = {}
    for rate in SYNTHETIC_RATES:
        shaped = torch.full_like(counts, float(rate))
        synthetic[str(rate)] = _group_scored_candidates(
            candidates,
            threshold=threshold,
            count_log_rates=shaped,
            adaptive_threshold=True,
        )[0]

    def plain(grouped):
        return [[[float(p), int(a), int(b)] for p, a, b in q] for q in grouped]

    return {
        "text": text,
        "threshold": threshold,
        "schema": schema,
        "count_log_rates": [float(v) for v in counts.reshape(-1)],
        "candidate_indices": [
            [[int(candidates.indices[0, q, c, 0]),
              int(candidates.indices[0, q, c, 1])] for c in
             range(candidates.indices.shape[2])]
            for q in range(candidates.indices.shape[1])
        ],
        "valid_mask": [
            [bool(candidates.valid_mask[0, q, c]) for c in
             range(candidates.valid_mask.shape[2])]
            for q in range(candidates.valid_mask.shape[1])
        ],
        "pair_logits": [
            [float(candidates.pair_logits[0, q, c]) for c in
             range(candidates.pair_logits.shape[2])]
            for q in range(candidates.pair_logits.shape[1])
        ],
        "threshold_only": plain(baseline),
        "adaptive_real_counts": plain(with_counts),
        "adaptive_synthetic_counts": {
            rate: plain(grouped) for rate, grouped in synthetic.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="boundary checkpoint to score; picks the base encoder and head width",
    )
    parser.add_argument(
        "--out", type=Path, default=fixture_dir() / "adaptive-threshold-golden.json"
    )
    args = parser.parse_args()

    base = encoder_dir(args.model)
    processor = SchemaTransformer(
        str(base), token_pooling="first", word_splitter=None
    )
    tokenizer = AutoTokenizer.from_pretrained(str(base))
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(str(base)))
    load_checkpoint_encoder(encoder, args.model)
    encoder.eval()
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SchemaTransformer.SPECIAL_TOKENS}
    )
    if added != len(SchemaTransformer.SPECIAL_TOKENS):
        raise ValueError(f"added {added} special tokens")

    head = build_head(model=args.model)
    if head.count_head is None:
        raise ValueError("checkpoint has no count head")

    payload = {
        "model": args.model,
        "note": (
            "Shipped boundary checkpoints set adaptive_threshold=false; the "
            "head is built from config unmodified and only the admission call "
            "passes adaptive_threshold=True."
        ),
        "cases": [
            run_case(processor, encoder, head, text, schema, threshold)
            for text, schema, threshold in CASES
        ],
        "synthetic_batches": [
            run_synthetic(name, logits, mask, rate, threshold)
            for name, logits, mask, rate, threshold in SYNTHETIC_BATCHES
        ],
    }
    dump_json(args.out, payload)


if __name__ == "__main__":
    main()
