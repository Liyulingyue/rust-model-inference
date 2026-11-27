"""Golden for ``resolve_overlaps`` — the span-conflict resolver.

``gliner2.5-base-v1`` ships ``overlap_policy = "flat"``, which normalizes to
``disallow``: the maximum-total-score set of non-overlapping spans. That is
weighted interval scheduling, not a greedy pass, so threshold-and-sort alone
leaves overlapping spans in the output.

This is a pure function, so the fixture is a table of hand-written cases rather
than model states — which makes it cheap to cover the paths that matter and
impossible to pass by accident:

* exact-boundary duplicates collapsing to the best-ranked representative
* containment (``nested``) versus crossing
* ``longest`` dropping strictly contained spans
* the three ``disallow`` tie-breaks in order: total score, then set size, then
  the lexicographically better ranking. The last one only fires on an exact
  score tie, so there is a case built to hit it.
* alias normalization (``flat``/``disallow``/``no_overlap``/``non_overlapping``)

Regenerate with:
    PYTHONPATH=target/gliner2-oracle \\
        models/.venv/bin/python3 \\
        tools/oracle/gliner_boundary/dump_overlap_resolution.py
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))

from gliner2.inference.overlap import (  # noqa: E402
    normalize_overlap_policy,
    resolve_overlaps,
)

# (name, policy, items) where each item is (score, start, end).
CASES = [
    ("empty", "flat", []),
    ("single", "flat", [(0.9, 0, 2)]),
    ("disjoint", "flat", [(0.9, 0, 2), (0.8, 3, 5)]),
    # Crossing spans: (0,3) and (2,5) overlap but neither contains the other.
    ("crossing_flat_picks_better", "flat", [(0.6, 0, 3), (0.9, 2, 5)]),
    # Three-way crossing where the greedy-by-score choice is *not* optimal:
    # taking the middle span blocks both outer ones.
    ("crossing_needs_dp_not_greedy", "flat",
     [(0.55, 0, 4), (0.5, 3, 7), (0.45, 6, 10)]),
    # Containment: the outer span scores higher, so `disallow` keeps it whole.
    ("nested_flat_keeps_whole", "flat", [(0.9, 0, 6), (0.8, 2, 4)]),
    # Containment where the inner span scores higher.
    ("nested_flat_prefers_inner", "flat", [(0.3, 0, 6), (0.9, 2, 4)]),
    # Exact score tie between two disjoint spans: both survive.
    ("tie_disjoint", "flat", [(0.5, 0, 2), (0.5, 4, 6)]),
    # Exact score tie between crossing spans, same start: the tie-break must
    # prefer the shorter/lexicographically-better one deterministically.
    ("tie_crossing_same_start", "flat", [(0.5, 0, 4), (0.5, 0, 6)]),
    # Zero-confidence entries, which is what a chunked merge produces. Without
    # the "larger set" tie-break these can beat a real span by sorting first.
    ("zero_score_vs_real", "flat", [(0.0, 0, 3), (0.7, 1, 4)]),
    ("all_zero_score", "flat", [(0.0, 0, 2), (0.0, 1, 3)]),
    # Touching half-open spans do not overlap, so all survive.
    ("touching", "flat", [(0.5, 0, 2), (0.4, 2, 4)]),
    # Same boundaries twice, different scores: the better one represents both.
    ("duplicate_boundaries", "flat", [(0.2, 1, 3), (0.8, 1, 3)]),
    ("duplicate_boundaries_three", "flat",
     [(0.2, 1, 3), (0.8, 1, 3), (0.5, 1, 3)]),
    # Nested policy keeps containment, rejects crossing.
    ("nested_keeps_containment", "nested", [(0.3, 0, 6), (0.9, 2, 4)]),
    ("nested_rejects_crossing", "nested", [(0.6, 0, 3), (0.9, 2, 5)]),
    ("nested_mixed", "nested",
     [(0.1, 0, 2), (0.9, 1, 5), (0.2, 4, 8), (0.3, 3, 9)]),
    # Allow keeps every distinct span, ranked.
    ("allow_keeps_all", "allow", [(0.1, 0, 6), (0.9, 2, 4), (0.5, 0, 3)]),
    # Longest drops strictly contained spans but keeps ties and disjoint spans.
    ("longest_drops_contained", "longest",
     [(0.1, 0, 6), (0.9, 2, 4), (0.5, 7, 9)]),
    ("longest_identical_survives", "longest", [(0.4, 1, 3), (0.9, 1, 3)]),
    # Alias normalization.
    ("alias_allow_all", "all", [(0.1, 0, 4), (0.9, 2, 6)]),
    ("alias_no_overlap", "no-overlap", [(0.6, 0, 3), (0.9, 2, 5)]),
    ("alias_non_overlapping", "non_overlapping", [(0.6, 0, 3), (0.9, 2, 5)]),
    ("alias_allow_nested", "allow_nested", [(0.3, 0, 6), (0.9, 2, 4)]),
    ("alias_keep_longest", "keep_longest", [(0.1, 0, 6), (0.9, 2, 4)]),
]

# A long chain, to exercise the DP beyond a handful of spans.
CHAIN = [(0.9 - 0.1 * (i % 5), i, i + 2) for i in range(12)]
CASES.append(("chain_flat", "flat", CHAIN))
CASES.append(("chain_nested", "nested", CHAIN))
CASES.append(("chain_longest", "longest", CHAIN))
CASES.append(("chain_allow", "allow", CHAIN))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "tests" / "fixtures"
        / "gliner2.5-base-v1" / "overlap-resolution-golden.json",
    )
    args = parser.parse_args()

    cases = []
    for name, policy, items in CASES:
        canonical = normalize_overlap_policy(policy, default="disallow")
        kept = resolve_overlaps(
            items, policy, score=lambda item: item[0], start=lambda item: item[1],
            end=lambda item: item[2],
        )
        cases.append({
            "name": name,
            "policy": policy,
            "canonical": canonical,
            "items": [list(item) for item in items],
            "kept": [list(item) for item in kept],
        })

    unknown = []
    for bad in ("bogus", "", "  "):
        try:
            normalize_overlap_policy(bad, default="disallow")
            unknown.append({"policy": bad, "error": None})
        except ValueError as error:
            unknown.append({"policy": bad, "error": str(error)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cases": cases, "unknown_policies": unknown}) + "\n")
    print(f"{args.out} (cases={len(cases)})")


if __name__ == "__main__":
    main()
