r"""Oracle for a `choices` field inside a **record** group — the four-way dispatch.

A legacy `json_structures` field with `choices` is scored once per field, against
the literal enum tokens the schema prefix embeds. A **record** field with
`choices` is not: `decode_group` has already picked an instance-conditioned
candidate for it, and for a choices field that candidate is one of the same
prefix tokens. Resolving it the ordinary way — mapping its token range to
character offsets — fails, because those tokens sit *before* the document and the
offset map only covers document tokens.

`_format_field` (`engine.py:1044-1281`) therefore resolves a record's choice field
through **four** paths, tried in this order. Which one fires is not visible from
the output's shape, so the fixture names each case after the path it is meant to
reach and the comment says why that path is the one that will.

1. **Prefix token** (`choices and te_raw <= offset`). The record's own candidate
   is a prefix literal, so the surface comes from the prefix tokens and the
   probability is `min(candidate_probability, assignment_probability)` — the
   record's own per-instance signal, not the field's.
   **The reported offsets are the record's *anchor span*, not the choice's.** A
   choice has no document location, and the anchor is the closest thing the
   record has to one; with no anchor the offsets are `0, 0`.
2. **Local literal mention.** If the record has an anchor and a declared choice
   occurs literally inside an anchor span, the field is rescored over *just those
   choices* (`preferred_choices`). A scalar field keeps only the mention closest
   to its anchor; with an anchor but no local mention a scalar field reports
   `None` and a list field `[]`.
3. **Matched surface.** Whatever the prefix path produced, if its surface
   casefolds to a declared choice it is reported as-is.
4. **Field-level fallback.** `_decode_choice_field` over all choices — the same
   query, no per-record signal, so **every record in the document gets an
   identical value and confidence** no matter which record asked.

Path 4 is what this whole mechanism exists to avoid, and the reference's own
comment says so. A port that stops at path 4 produces output that looks entirely
reasonable and is wrong in a way no single-record test can see — with one record
there is nothing to compare. So the cases below deliberately put **two or more
records** in the same document wherever they are checking per-record signal.

What this pins
--------------
- **Offsets are the anchor's.** Same value, same confidence, different `start`/`end`
  per record.
- **A record with no anchor reports `0, 0`**, not the choice token's position.
- **`min(candidate, assignment)` is a real floor.** The reported probability can be
  strictly below both inputs, so a port that reports either one alone is wrong in a
  way a tolerance on the score would hide.
- **Scalar and list render differently** under path 2's no-local-mention case:
  `None` versus `[]`.
- **Path 4 is uniform across records**, which is the observable signature of the
  fallback being taken.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402

from gliner2.processor import SchemaTransformer  # noqa: E402

from common import build_head, encoder_dir  # noqa: E402
from dump_extract_spans_end_to_end import load_checkpoint_encoder  # noqa: E402
from dump_records_end_to_end import build_record_scorer, run_case  # noqa: E402


def record_with_choices(
    choice_fields,
    *,
    mode="latent",
    anchor=None,
    plain_fields=(),
    name="person",
    dtypes=None,
    description="a person and how they feel",
):
    """A `json_structures` group carrying `record_metadata` and choice fields.

    `choice_fields` is a list of `(field, choices)` pairs; `plain_fields` are
    ordinary span fields alongside them. `mode` selects the record path:
    `"natural"` requires `anchor` to name one of the plain fields, while
    `"latent"` and `"anchorless"` do not.

    Two details are load-bearing and were both wrong in the first version:

    * **`record_metadata` is keyed by the *group* name** (`{"person": {...}}`), not
      by `"<group>.__mode__"`. The wrong key is accepted without complaint and the
      group then decodes through the **legacy** structure path, which emits zero
      records — so the oracle ran clean and produced nothing.
    * **`json_descriptions[group]` is a field -> description map**, not a string on
      the group. A plain string there raises inside `_process_json_structures`,
      and `error_policy="fallback"` turns that into a dummy `[E] entity` record —
      the same silent-to-nothing failure by a different route.
    """
    # Declaration order is the prompt order, and the reference's
    # `field_orders` follows the schema, so the order written here is the order
    # the model sees. Choice fields come first here because that is the order
    # the cases below read most clearly; nothing depends on it being one way.
    roles = [field for field, _ in choice_fields] + list(plain_fields)
    entry = {
        role: {"value": "", "choices": dict(choice_fields).get(role)}
        if role in dict(choice_fields)
        else []
        for role in roles
    }
    schema = {
        "json_structures": [{name: entry}],
        "json_descriptions": {
            name: {role: f"{description} for {role}" for role in roles}
        },
        "field_metadata": {
            f"{name}.{field}": {
                "dtype": (dtypes or {}).get(field, "str"),
                "choices": choices,
            }
            for field, choices in choice_fields
        },
    }
    for field in plain_fields:
        schema["field_metadata"].setdefault(
            f"{name}.{field}", {"dtype": "str"}
        )
    spec = {"mode": mode}
    if anchor is not None:
        spec["anchor"] = anchor
    schema["record_metadata"] = {name: spec}
    return schema


MOODS = ["positive", "negative", "neutral"]
# Repeated so a two-record document has a literal to attach to in one window and
# not the other — that is what separates path 1 from path 2.
TWO_RECORDS = (
    "Marie Curie was positive about physics. "
    "Pierre Curie was negative about chemistry. "
    "Ada Lovelace was neutral about computation."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "gliner2.5-base-v1"
        / "record-choices-golden.json",
    )
    args = parser.parse_args()

    base = encoder_dir("gliner2.5-base-v1")
    processor = SchemaTransformer(str(base), token_pooling="first", word_splitter=None)
    tokenizer = AutoTokenizer.from_pretrained(str(base))
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(str(base)))
    load_checkpoint_encoder(encoder, "gliner2.5-base-v1")
    encoder.eval()
    tokenizer.add_special_tokens(
        {"additional_special_tokens": SchemaTransformer.SPECIAL_TOKENS}
    )
    head = build_head(model="gliner2.5-base-v1")
    record_head = build_record_scorer(head.settings)

    cases = [
        # Two records, each with a literal mood inside its own window, so path 2
        # has an anchor and a local mention per record and the reported values
        # differ per record.
        (
            "two_records_literal_moods",
            TWO_RECORDS,
            record_with_choices(
                [("name", MOODS)], mode="latent",
                plain_fields=["who"], dtypes={"who": "str"},
            ),
            0.3,
        ),
        # The same schema on text with no literal mood anywhere: path 2 finds
        # nothing, so the value has to come from the prefix tokens (path 1) or the
        # field-level fallback (path 4). Which one fired is the whole question,
        # and with two records it is observable.
        (
            "no_literal_mood_in_the_text",
            "Marie Curie worked in Paris for many years. Pierre Curie joined her there.",
            record_with_choices(
                [("name", MOODS)], mode="latent",
                plain_fields=["who"], dtypes={"who": "str"},
            ),
            0.3,
        ),
        # A scalar choice field beside a list one, so the two render differently
        # and a port that treats them alike is caught.
        (
            "scalar_beside_list_choice_fields",
            TWO_RECORDS,
            record_with_choices(
                [("name", MOODS), ("mood", MOODS)], mode="latent",
                plain_fields=["who"], dtypes={"who": "str"},
            ),
            0.3,
        ),
        # `natural` mode anchors each record on the `who` field, which is what
        # gives the choice field its offsets.
        (
            "natural_mode_anchored_records",
            TWO_RECORDS,
            record_with_choices(
                [("name", MOODS)], mode="natural", anchor="who",
                plain_fields=["who"], dtypes={"who": "str"},
            ),
            0.3,
        ),
        # `anchorless`: no anchor at all, so a choice field has no offsets to
        # report and must emit `0, 0` — not the prefix token's position.
        (
            "anchorless_records_have_no_choice_offsets",
            TWO_RECORDS,
            record_with_choices([("name", MOODS)], mode="anchorless"),
            0.3,
        ),
        # A single-record document, where path 4's uniformity is invisible.
        # Present deliberately: it is the case a *different* bug hides behind, and
        # without it the suite cannot tell "uniform because there is one record"
        # from "uniform because the fallback was taken".
        (
            "one_record_cannot_reveal_path_four",
            "Marie Curie was positive about physics.",
            record_with_choices(
                [("name", MOODS)], mode="latent",
                plain_fields=["who"], dtypes={"who": "str"},
            ),
            0.3,
        ),
    ]

    out_cases = []
    for name, text, schema, threshold in cases:
        dumped = run_case(processor, encoder, head, record_head, text, schema, threshold)
        dumped["name"] = name
        out_cases.append(dumped)
        records = dumped.get("records") or []
        print(f"  {name}: {len(records)} record(s)")
        for record in records:
            for qid, field in record["fields"].items():
                print(f"      qid={qid} spans={field['spans']} text={field['text']} "
                      f"scores={[round(s, 4) for s in field['scores']]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cases": out_cases}, indent=1) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()