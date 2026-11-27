r"""Oracle for `_deduplicate_relation_edges` — the four-stage edge canonicalizer.

Relation proposals score a capped head x tail cross-product on purpose, so the
raw edge list contains every occurrence combination and every contained partial
mention. `_deduplicate_relation_edges` (`engine.py:899-1002`) collapses that
into semantic edges before anything is reported.

It is a `@staticmethod` over a list of dicts, so the fixture is a truth table and
the test needs no GGUF. Edges are `[head_text, head_start, head_end, tail_text,
tail_start, tail_end, score]`, the tuple order the reference destructures.

What this pins
--------------
Four stages, each removing a different kind of redundancy, and the order matters
because each feeds the next:

1. **Per-side containment canonicalization.** For each side independently, every
   mention is replaced by the *longest* mention that contains it, ties going to
   the **earlier** start (`-candidate[1]`). This is what removes a partial
   mention that is a prefix of a longer one — the same canonical head is chosen
   for the head side and the tail side *independently*, so a single edge can have
   its head and tail canonicalized against different mention sets.
2. **Exact `(h0,h1,t0,t1)` dedup**, keeping the **higher score**. Note the
   winner is the edge with the max score but the winner's *text* comes from
   whichever edge supplied it, and the key ignores text entirely — so two edges
   at identical offsets with different surface text collapse on offsets alone.
3. **Case- and whitespace-folded semantic-text dedup.** The key is the folded
   text pair, so `"Marie Curie"` and `"marie  curie"` are the same entity. The
   survivor is chosen by a rank of `(distance, -score, head_start, tail_start)`
   where distance is the gap between the two spans — so **closer beats higher
   scoring**, which is the opposite of what stage 2 does. The comparison is a
   strict `<`, so an exact tie keeps the incumbent, and iteration order of a dict
   is insertion order, so "incumbent" is the first-inserted at that key.
4. **Token-subset dominance removal.** An edge is dropped when one argument is a
   *strict* subset of another edge's same-side token set and the other side is
   *exactly* equal. Strict (`<`, not `<=`) is what keeps two identical-token
   edges from deleting each other.

The result is sorted by `(head_start, tail_start, -score)`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))

from gliner2.inference.engine import BoundaryExtractor  # noqa: E402


def edge(head, hs, he, tail, ts, te, score):
    """One edge in the shape the reference destructures."""
    return {
        "head": [head, hs, he],
        "tail": [tail, ts, te],
        "score": score,
    }


def flat(item):
    return [
        item["head"][0], item["head"][1], item["head"][2],
        item["tail"][0], item["tail"][1], item["tail"][2],
        item["score"],
    ]


def dedup(edges):
    return BoundaryExtractor._deduplicate_relation_edges(edges)


CASES = [
    ("single_edge", [edge("Marie", 0, 5, "Paris", 22, 27, 0.9)]),
    ("empty", []),
    # Stage 1: a contained partial mention folds into the longer one, so the
    # edge pair that differed only by mention length becomes one exact edge.
    (
        "contained_head_folds_to_the_longer_mention",
        [
            edge("Marie", 0, 5, "Paris", 22, 27, 0.4),
            edge("Marie Curie", 0, 11, "Paris", 22, 27, 0.8),
        ],
    ),
    # Stage 1 both sides: head and tail canonicalize independently, so this
    # needs *two* mentions to collapse a pair that shares neither endpoint
    # verbatim with the survivor.
    (
        "both_sides_canonicalize_independently",
        [
            edge("Marie", 0, 5, "Paris", 22, 27, 0.5),
            edge("Marie Curie", 0, 11, "Paris", 22, 27, 0.7),
            edge("Marie", 0, 5, "Paris France", 22, 33, 0.6),
            edge("Marie Curie", 0, 11, "Paris France", 22, 33, 0.9),
        ],
    ),
    # Containment picks the **longest** containing mention, so "A Paris" (7 chars)
    # wins over "B Paris" and "Paris" for the span (5, 10) — the `-start`
    # tie-break never engages here because the lengths already differ.
    (
        "containment_prefers_the_longest_mention",
        [
            edge("Paris", 5, 10, "x", 30, 31, 0.5),
            edge("A Paris", 3, 10, "x", 30, 31, 0.5),
            edge("B Paris", 4, 10, "x", 30, 31, 0.5),
        ],
    ),
    # Stage 2: identical offsets, different score, and *different text* — the
    # key is offsets only, so they collapse and the higher score wins.
    (
        "exact_offsets_collapse_ignoring_text",
        [
            edge("Marie", 0, 5, "Paris", 22, 27, 0.3),
            edge("M. Curie", 0, 5, "Paris", 22, 27, 0.8),
        ],
    ),
    # Stage 3: case and whitespace folding make these the same entity, and the
    # survivor is the *closer* pair even though the other scores higher.
    (
        "semantic_text_folds_case_and_whitespace",
        [
            edge("Marie Curie", 0, 11, "Paris", 22, 27, 0.9),
            edge("marie  curie", 0, 11, "paris", 40, 45, 0.4),
        ],
    ),
    # Same offsets, different text, different score. The oracle shows this
    # collapsing in **stage 2**, not stage 3: the key is offsets only, so the
    # winner's score comes from the higher-scoring edge while the head/tail text
    # comes from the *last* edge at those coordinates. That split of provenance
    # between the winning edge and the canonical mention is the subtle part.
    (
        "equal_offsets_take_the_incumbents_text_and_the_winning_score",
        [
            edge("Marie", 0, 5, "Paris", 22, 27, 0.9),
            edge("MARIE", 0, 5, "PARIS", 22, 27, 0.2),
        ],
    ),
    # Stage 4: a strict token subset with an exactly-equal opposite endpoint is
    # dominated. `"paris"` is a strict subset of `"paris france"`.
    (
        "token_subset_dominance",
        [
            edge("Marie", 0, 5, "Paris", 22, 27, 0.9),
            edge("Marie", 0, 5, "Paris France", 22, 33, 0.4),
        ],
    ),
    # Stage 4 needs *strict* subset, so equal token sets do not dominate and
    # both survive.
    (
        "equal_token_sets_collapse_on_offsets_not_on_tokens",
        [
            edge("Marie Curie", 0, 11, "Paris", 22, 27, 0.9),
            edge("Curie Marie", 0, 11, "Paris", 22, 27, 0.4),
        ],
    ),
    # Stage 4 on the head side, with equal tails.
    (
        "head_token_subset_dominance",
        [
            edge("Paris", 22, 27, "Marie", 0, 5, 0.9),
            edge("Paris France", 22, 33, "Marie", 0, 5, 0.4),
        ],
    ),
    # Stage 4 needs **strict** subset, so two edges with the *same* token set on a
    # side and the same opposite side both survive. A non-strict comparison would
    # delete both, leaving nothing — which is why this case exists.
    (
        "identical_token_sets_both_survive",
        [
            # "Marie Curie" and "Curie Marie" fold to *different* strings, so
            # stage 3 keeps both, but they share a token set, so stage 4 is the
            # only thing that could drop either. Distinct offsets keep stage 1
            # and stage 2 out of it. A non-strict subset would delete both.
            edge("Marie Curie", 0, 11, "Paris", 22, 27, 0.9),
            edge("Curie Marie", 40, 51, "Paris", 60, 65, 0.4),
        ],
    ),
    # Several relations at once, to pin the final sort by
    # (head_start, tail_start, -score).
    (
        "final_sort_is_by_head_then_tail_then_score",
        [
            edge("z", 10, 11, "t", 50, 51, 0.5),
            edge("a", 0, 1, "t", 50, 51, 0.9),
            edge("b", 0, 2, "s", 40, 41, 0.7),
            edge("c", 0, 3, "s", 40, 41, 0.7),
            edge("c", 0, 3, "s", 40, 41, 0.2),
        ],
    ),
    # Every stage at once: containment, then exact, then semantic, then subset.
    (
        "all_four_stages",
        [
            edge("Marie", 0, 5, "Paris", 22, 27, 0.3),
            edge("Marie Curie", 0, 11, "Paris", 22, 27, 0.5),
            edge("marie curie", 0, 11, "paris", 22, 27, 0.4),
            edge("Marie Curie", 0, 11, "Paris France", 22, 33, 0.6),
        ],
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "gliner2.5-base-v1"
        / "relation-dedup-golden.json",
    )
    args = parser.parse_args()

    cases = []
    for name, edges in CASES:
        deduped = dedup([dict(
            head=list(edge_item["head"]),
            tail=list(edge_item["tail"]),
            score=edge_item["score"],
        ) for edge_item in edges])
        cases.append({
            "name": name,
            "edges": [flat(item) for item in edges],
            "deduped": [flat(item) for item in deduped],
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cases": cases}) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
