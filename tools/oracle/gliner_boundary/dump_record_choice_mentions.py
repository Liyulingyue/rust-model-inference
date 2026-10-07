r"""Oracle for `_record_local_choice_mentions` — document-level enum assignment.

For a record whose field carries `choices`, the reference first looks for the
choice literals **in the document text** and, if it finds any, assigns them to
the records by anchor position before falling back to scoring the schema-prefix
enum tokens. This file pins that assignment; the prefix-token fallback is a
separate stage.

`_record_local_choice_mentions` (`engine.py:595-654`) is a pure function of
`(text, choices, anchor_char_spans)`, so the fixture is a truth table over text
rather than a model run.

What this pins
--------------
The docstring calls the rule "more reliable than nearest-distance assignment for
records whose fields follow a short anchor across one or more clauses", and the
rule is **preceding-anchor, not nearest**:

- Every occurrence of a choice literal is found with ``(?<!\w)`` and ``(?!\w)``
  around it, case-insensitively, so ``paris`` does not match inside ``parisian``
  and a choice that is a substring of a longer word is not reported.
- A mention between two anchors belongs to the **preceding** one. A mention
  before the first anchor binds to the first. The reference's own example is
  ``Amazon ... (books). Amazon ...`` — the second ``(books)`` goes to the second
  record, not to whichever anchor happens to be closest.
- Mentions before the first anchor and the trailing ownership are the two
  easy-to-miss parts; the code sorts anchors by their start (not by the record
  index) before assigning.
- Values are **semantic sets**: one source occurrence per value per record,
  kept in source order. So a value repeated inside one record appears once, and
  the retained occurrence is the first.

The choice values are reported **verbatim as declared** — the regex match is
case-insensitive but the reported text is the schema's own casing, not the
document's.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))

from gliner2.inference.engine import BoundaryExtractor  # noqa: E402

# (name, text, choices, anchors)
#   anchors are `[start, end)` character spans, or `None` for a record whose
#   anchor could not be resolved. A `None` anchor is skipped for assignment but
#   still counts as a record.
CASES = [
    (
        "one_anchor_one_choice",
        "The trip was great. Paris was lovely.",
        ["paris"],
        [(0, 3)],
    ),
    (
        "two_anchors_mention_between_belongs_to_preceding",
        "Amazon sells books. Amazon ships fast.",
        ["books"],
        [(0, 6), (19, 25)],
    ),
    (
        "mention_before_first_anchor_binds_to_first",
        "In Paris, Alice lives.",
        ["paris"],
        [(12, 17)],
    ),
    (
        "no_mention_anywhere",
        "Nothing relevant here.",
        ["paris"],
        [(0, 7)],
    ),
    (
        "word_boundary_excludes_substring",
        "A Parisian cafe and a real Paris.",
        ["paris"],
        [(0, 14)],
    ),
    (
        "case_insensitive_but_reported_verbatim",
        "We visited PARIS and paris.",
        ["Paris"],
        [(0, 10)],
    ),
    (
        "repeated_value_in_one_record_kept_once",
        "Paris Paris Paris.",
        ["paris"],
        [(0, 5)],
    ),
    (
        "two_mentions_two_anchors_each_get_one",
        "Rome then Rome, then Rome again.",
        ["rome"],
        [(0, 4), (16, 20)],
    ),
    (
        "anchor_none_is_skipped",
        "Paris is nice.",
        ["paris"],
        [None, (13, 17)],
    ),
    (
        "empty_choices",
        "Paris is nice.",
        [],
        [(0, 5)],
    ),
    (
        "empty_text",
        "",
        ["paris"],
        [(0, 0)],
    ),
    (
        "all_choices_absent",
        "Berlin is nice.",
        ["paris", "london"],
        [(0, 6)],
    ),
    (
        "some_choices_present",
        "Paris is nice, Berlin too.",
        ["paris", "berlin"],
        [(0, 6)],
    ),
    (
        "mention_spanning_anchor_boundary",
        "xxparisxx and paris.",
        ["paris"],
        [(0, 3), (12, 17)],
    ),
    (
        "multiple_choices_different_owners",
        "Paris books. London books.",
        ["paris", "books", "london"],
        [(0, 5), (16, 22)],
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "gliner2.5-base-v1"
        / "record-choice-mentions-golden.json",
    )
    args = parser.parse_args()

    # A `@staticmethod`, so it is called straight off the class.
    mention = BoundaryExtractor._record_local_choice_mentions

    cases = []
    for name, text, choices, anchors in CASES:
        has_literal, assigned = mention(text, choices, anchors)
        cases.append({
            "name": name,
            "text": text,
            "choices": choices,
            "anchors": [[None] if a is None else [a[0], a[1]] for a in anchors],
            "has_literal_choices": bool(has_literal),
            # Keyed by record index, the reference's own return shape.
            "assigned": {
                str(index): [[choice, start, end] for choice, start, end in mentions]
                for index, mentions in sorted(assigned.items())
            },
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cases": cases}, ensure_ascii=False) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
