r"""Oracle for the public result formatters.

`format_results` (`runtime.py:73`) is the last thing every extract call does. It
is a module-level function over the **raw** results dict — before formatting,
where a prediction is still a Python tuple — and it decides per key whether the
value is a classification, a relation, or a structure, purely by looking at the
value's runtime type. That type sniffing is the whole contract, so the fixture
below drives it with tuples and lets the reference decide, rather than
reimplementing the dispatch here.

This is the piece F-3 接线 needs: `merge_chunk_results` walks the *formatted*
JSON, but this port's `extract()` returns a typed `Extraction` that is not
`Serialize`. Porting these formatters is what lets `Extraction` become the
reference's public payload, so the formatters need ground truth of their own —
and they are pure functions over plain data, so no GGUF is required.

What this pins
--------------
**Dispatch is by type sniffing, not by declaration.** A key is a relation if its
first element is a 2-tuple or a dict with `head`/`tail` — and once the sniff
says relation, `requested_relations` is never consulted again, so a sniffed
relation ships whether or not anyone asked for it. `requested_relations` only
*adds* an empty list for names that were missing. A key in `classification_tasks`
short-circuits the sniff entirely, which means a name that is both a
classification task and a requested relation appears **twice** in one payload,
in two different shapes.

**`include_confidence` is not a uniform toggle.** It only affects the branches
that *build* a value. The dict branch of `format_entity_dict` / `format_struct`
passes dicts through **unchanged**, so a span dict keeps its `start`/`end` even
with confidence off; the tuple branch instead *constructs*
`{"text", "confidence"}` and **drops the offsets**. Feeding the same logical
span as a tuple versus as a dict therefore produces a different payload shape,
which is the single most surprising thing here and is why both are pinned.

**Empty is not absent.** An empty list under key `entities` becomes `{}`; under
any other key it stays `[]`. A falsy scalar field becomes `null` in both struct
formatters — `spans or None` and an explicit `None` agree, so the two formatters
are *not* distinguished by falsy handling.

**Requested relations are always present.** Every name in
`requested_relations` gets an entry, empty list if nothing was found, and they
are all nested under one `relation_extraction` key rather than left at top
level. Entries sniffed as relations join them there even if unrequested.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))

from gliner2.inference.runtime import format_results  # noqa: E402


def span_tuple(text, conf, start, end):
    """A raw span prediction: the 4-tuple the tuple branches destructure."""
    return (text, conf, start, end)


def span_dict(text, conf=None, start=None, end=None):
    """A raw span prediction in dict form; `start`/`end` are optional."""
    out = {"text": text}
    if conf is not None:
        out["confidence"] = conf
    if start is not None:
        out["start"] = start
    if end is not None:
        out["end"] = end
    return out


def rel(head, tail, score):
    return {"head": head, "tail": tail, "score": score}


# Sentinels standing in for a pair that must survive as a real `list`. The
# restore pass turns any 2-element list whose first item is a string and whose
# second is a number back into a `tuple`, which is exactly what the case above
# must *not* be, so these are objects the pass leaves alone and the driver
# swaps for the real lists right before calling the reference.
RAW_LIST_PAIR = "raw-list-pair"
RAW_LIST_PAIR_2 = "raw-list-pair-2"


CASES = [
    # ---- the argument-driven dispatch -------------------------------------
    # A 2-tuple list is a relation *by type alone* — `requested_relations` is
    # never consulted, because the sniff already decided. So no argument turns
    # this into labels; a label list is a list of 2-element *lists*, which is
    # the next case.
    (
        "two_tuple_list_is_a_relation_whatever_the_arguments",
        {"works_in": [("Marie", 0.9), ("ACME", 0.8)]},
        {"requested_relations": [], "classification_tasks": []},
    ),
    # A list of 2-element *lists* — genuinely lists, not tuples — fails the
    # relation sniff. It then fails every remaining branch too (the value is a
    # list whose first element is a list, so it is neither a dict, a tuple, nor
    # a scalar), and lands in the bare `else`, which stores it unchanged. So
    # "not a relation" does not mean "formatted as labels": the list survives
    # completely unformatted, with no {label, confidence} rewrite. This is the
    # one case where the payload keeps its nested-list shape end to end.
    (
        "list_of_real_lists_is_not_sniffed_and_not_formatted",
        {"sentiment": [RAW_LIST_PAIR, RAW_LIST_PAIR_2]},
        {"requested_relations": [], "classification_tasks": []},
    ),
    # A head/tail dict sniffs as a relation by its keys, and the final
    # `if relations:` merge publishes it **whether or not it was requested** —
    # `requested_relations` only ever *adds* empty entries for missing names.
    # So a sniffed relation with an empty request list still ships.
    (
        "sniffed_relation_ships_without_being_requested",
        {"linked": [rel("Marie", "ACME", 0.9)]},
        {"requested_relations": [], "classification_tasks": []},
    ),
    # Requested but absent: the name still appears, as an empty list, so a
    # caller can index every requested relation without a presence check.
    (
        "requested_relation_absent_becomes_an_empty_list",
        {"works_in": [("Marie", 0.9)]},
        {"requested_relations": ["worked_in", "works_in"], "classification_tasks": []},
    ),
    # A key that is both a classification task and a requested relation shows up
    # **twice**: formatted as a classification at top level, and again as an
    # empty list inside `relation_extraction`, because the classification branch
    # consumed the value and never filled `relations[mentions]`. The payload
    # therefore carries one key in two different shapes at once.
    (
        "classification_and_requested_relation_appear_in_both_places",
        {"mentions": [("Marie", 0.9), ("ACME", 0.8)]},
        {"requested_relations": ["mentions"], "classification_tasks": ["mentions"]},
    ),
    # Two requested relations with the same first element type: both nest under
    # the one `relation_extraction` key rather than staying at top level.
    (
        "all_relations_nest_under_one_key",
        {
            "works_in": [("Marie", 0.9)],
            "founded": [("ACME", 0.7)],
        },
        {"requested_relations": ["works_in", "founded"], "classification_tasks": []},
    ),

    # ---- include_confidence is not a uniform toggle -----------------------
    # The tuple branch *constructs* {"text", "confidence"} and drops the
    # offsets. The dict branch passes through untouched, so it keeps
    # start/end. Same span, same flag, different payload — the core surprise.
    # NOTE only the **first** entity-type dict is formatted: `format_results`
    # passes `value[0]`, so the `org` entry below is discarded entirely rather
    # than formatted. That is why this case has one surviving type and why the
    # offsets question has to be split into its own case.
    (
        "tuple_spans_lose_their_offsets",
        {
            "entities": [
                {"person": [span_tuple("Marie", 0.9, 0, 5)]},
                {"org": [span_dict("ACME", 0.8, 10, 14)]},
            ]
        },
        {"requested_relations": [], "classification_tasks": []},
    ),
    # The same input with the flag off: the tuple collapses to a bare string.
    # The `org` dict is discarded here for the same reason as above, so this
    # case does not show the dict branch at all.
    (
        "without_confidence_tuples_become_bare_strings",
        {
            "entities": [
                {"person": [span_tuple("Marie", 0.9, 0, 5)]},
                {"org": [span_dict("ACME", 0.8, 10, 14)]},
            ]
        },
        {"requested_relations": [], "classification_tasks": [], "include_confidence": False},
    ),
    # Two dict spans of the same text at different offsets: **both** survive.
    # The dict branch keys on `(text.lower(), start, end)`, so distinct offsets
    # keep them apart. This is the case that shows the dict branch passing
    # spans through with their offsets intact, which the tuple branch never
    # does.
    (
        "dict_spans_with_distinct_offsets_are_both_kept",
        {
            "entities": [
                {"person": [
                    span_dict("Marie", 0.9, 0, 5),
                    span_dict("Marie", 0.4, 20, 25),
                ]}
            ]
        },
        {"requested_relations": [], "classification_tasks": []},
    ),
    # Dedup keys on `text.lower()`, but the check is `not in seen` against a set
    # of `(lower, start, end)` triples — so the differing offsets are what keep
    # these two apart, *not* the casing. Lowercasing the key only matters when
    # the offsets also match, which the first-wins case below pins.
    (
        "case_does_not_dedupe_across_different_offsets",
        {
            "entities": [
                {"person": [
                    span_tuple("Marie", 0.4, 0, 5),
                    span_tuple("MARIE", 0.9, 20, 25),
                ]}
            ]
        },
        {"requested_relations": [], "classification_tasks": []},
    ),
    # Two spans at the **same** offsets but different casing: this is the case
    # that actually collapses, and it pins that the survivor is the *first*
    # occurrence rather than the higher-scoring one. The value is appended, never
    # replaced, so the lower confidence (0.4) is what ships. The offsets being
    # in the dedup key is what makes this collapse at all — drop them and the
    # case stops deduping, which is how the property is verified.
    (
        "same_offsets_dedupe_and_the_first_one_wins",
        {
            "entities": [
                {"person": [
                    span_tuple("Marie", 0.4, 0, 5),
                    span_tuple("MARIE", 0.9, 0, 5),
                ]}
            ]
        },
        {"requested_relations": [], "classification_tasks": []},
    ),
    # An empty-text span is skipped outright by the tuple and dict branches
    # (`if text and ...`), so it never appears in the output.
    (
        "empty_text_spans_are_dropped",
        {
            "entities": [
                {"person": [span_tuple("", 0.9, 0, 0), span_tuple("Marie", 0.5, 3, 8)]}
            ]
        },
        {"requested_relations": [], "classification_tasks": []},
    ),

    # ---- empty is not absent ---------------------------------------------
    # An empty list is `{}` for `entities` but `[]` everywhere else. This is a
    # real difference in payload type that a caller can trip over.
    (
        "empty_entities_list_becomes_an_empty_object",
        {"entities": []},
        {"requested_relations": [], "classification_tasks": []},
    ),
    (
        "empty_non_entity_list_stays_a_list",
        {"attributes": []},
        {"requested_relations": [], "classification_tasks": []},
    ),
    # A struct whose fields are all falsy: `format_struct` writes an explicit
    # `None` per field, so the key survives.
    (
        "falsy_struct_fields_become_none",
        {"record": {"author": "", "role": None}},
        {"requested_relations": [], "classification_tasks": []},
    ),
    # The same falsy fields through `format_entity_dict`. The two branches read
    # differently (`spans or None` versus an explicit `None`) but agree: a
    # falsy value becomes null either way, so the empty string is null here too.
    # The two formatters are not distinguished by falsy handling at all — that
    # was the first guess, and the oracle contradicted it.
    (
        "falsy_entity_fields_become_none_too",
        {"entities": [{"author": "", "role": None}]},
        {"requested_relations": [], "classification_tasks": []},
    ),
    # A scalar tuple (not a list) is a single labelled value: it becomes a bare
    # label, or a one-key dict with confidence.
    (
        "scalar_tuple_is_one_label",
        {"sentiment": ("positive", 0.9)},
        {"requested_relations": [], "classification_tasks": ["sentiment"]},
    ),
    # Two labels is a *list* of labels, not a (label, score) pair: the
    # `len(value) == 2` check also requires the first element to be a string,
    # and here the first element is itself a list.
    (
        "two_labels_stay_a_list_of_labels",
        {"topics": [("ml", 0.9), ("safety", 0.8)]},
        {"requested_relations": [], "classification_tasks": ["topics"]},
    ),
    # A single-element list *is* a list-of-pairs (the inner check only needs
    # `value[0]` to be a pair), so it is formatted like any other label list —
    # there is no "too short to be a list" special case.
    (
        "one_element_label_list_is_still_formatted",
        {"topics": [("ml", 0.9)]},
        {"requested_relations": [], "classification_tasks": ["topics"]},
    ),
    # A two-element list of plain strings with a *non*-numeric second element
    # is not a (label, score) pair, so it survives as a list of strings.
    (
        "two_strings_are_not_a_label_score_pair",
        {"topics": ["ml", "safety"]},
        {"requested_relations": [], "classification_tasks": ["topics"]},
    ),
    # A bool in the score slot is not a score (`_is_score` excludes bool), so
    # this is a two-element list, not a (label, score) pair.
    (
        "bool_score_is_not_a_score",
        {"flagged": ["yes", True]},
        {"requested_relations": [], "classification_tasks": ["flagged"]},
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "gliner2.5-base-v1"
        / "format-results-golden.json",
    )
    args = parser.parse_args()

    cases = []
    for name, results, options in CASES:
        keep_lists = "list_of_real_lists" in name
        # Round-trip through JSON so tuples become lists, exactly as they
        # arrive at this layer in the real pipeline. `format_results` sniffs
        # `isinstance(value[0], tuple)`, which a JSON list is *not*, so the
        # tuple-shape cases need the tuples restored after loading.
        # Longest sentinel first: `RAW_LIST_PAIR` is a prefix of
        # `RAW_LIST_PAIR_2`, so replacing the short one first would rewrite the
        # long one into `"["negative", 0.8]-2"`. That mistake is silent — the
        # JSON still parses — and it turns the intended non-tuple pair into a
        # string, which then sniffs as nothing and the case quietly stops
        # testing what its name claims.
        encoded = json.dumps(results)
        encoded = encoded.replace(f'"{RAW_LIST_PAIR_2}"', '["negative", 0.8]')
        encoded = encoded.replace(f'"{RAW_LIST_PAIR}"', '["positive", 0.9]')
        # `keep_lists` skips the restore pass for this case. Restoring is what
        # turns a 2-element list into a tuple, and the whole point of the case
        # is a pair that stays a list — so restoring it would silently undo the
        # setup and re-sniff the value as a relation.
        payload = json.loads(encoded)
        if not keep_lists:
            payload = _restore_tuples(payload)
        formatted = format_results(
            payload,
            include_confidence=options.get("include_confidence", True),
            requested_relations=options.get("requested_relations"),
            classification_tasks=options.get("classification_tasks"),
        )
        # The stored `results` is the *resolved* input, not the sentinels, so a
        # reader of the fixture sees the pairs the reference actually saw. The
        # sentinel only exists to survive the round trip above.
        resolved = results
        if keep_lists:
            resolved = json.loads(encoded)
        cases.append({
            "name": name,
            "results": resolved,
            "options": options,
            "formatted": formatted,
            # JSON has no `tuple`, and the reference's dispatcher branches on
            # exactly that (`isinstance(value[0], tuple)`). One case's whole
            # point is a pair that stays a *list*, so the fixture records which
            # nested pairs are tuples and which are lists, and the port is told
            # rather than guessing from shape. Without this the port cannot
            # reproduce the case at all: `["a", 0.9]` is the same JSON whether
            # the reference saw a tuple or a list.
            "tuple_pairs": keep_lists is False,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cases": cases}) + "\n")
    print(args.out)


def _restore_tuples(value):
    """Turn the encoded 2- and 4-tuples back into real tuples.

    `json.dumps` has no tuple type, so a span 4-tuple and a label 2-tuple both
    arrive as lists. The reference distinguishes them by `len` and by the type
    of the first element, so a list of four is restored as a 4-tuple and a list
    of two whose second element is a number is restored as a 2-tuple. Anything
    else stays a list, which is what makes the `flagged`/`yes, True` case come
    out as a plain list rather than a pair.
    """
    if isinstance(value, list):
        if len(value) == 4 and all(
            isinstance(item, (int, float, str)) and not isinstance(item, bool)
            for item in value
        ):
            return tuple(value)
        if len(value) == 2 and isinstance(value[0], str) and _is_number(value[1]):
            return tuple(value)
        return [_restore_tuples(item) for item in value]
    if isinstance(value, dict):
        return {key: _restore_tuples(item) for key, item in value.items()}
    return value


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


if __name__ == "__main__":
    main()
