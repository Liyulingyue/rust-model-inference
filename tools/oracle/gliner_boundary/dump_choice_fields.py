"""Oracle for `field_metadata.choices` — the literal-enum structure field.

A structure field may declare a fixed set of literal values instead of asking
the model to find a span. The reference builds the enum into a **prefix on the
text token stream** (`processor.py:825-858`), scores each choice token as a
one-token span, and reads the value off `_decode_choice_field`
(`engine.py:656`).

What this pins
--------------
The prefix is the whole difficulty, and it is easy to get wrong in three
separate ways:

1. **The prefix goes on the text stream, not the schema stream.**
   ``_transform_record`` does ``text_tokens = prefix + text_tokens``
   (``processor.py:645``) and records ``len_prefix``. The schema tokens are
   untouched, so the ``[C]`` group's marker stride — which every other group
   kind depends on — does not move. A port that appended the choices to the
   schema prefix would shift every marker after it by a variable amount.

2. **Every later word index is offset by ``len_prefix``.** A choice span is
   reported in text-token coordinates, so ``start - offset`` is needed before
   the span indexes the word list. Getting this wrong reports a choice span
   from the middle of the document.

3. **A choice is one token, found by exact lower-cased match**, and it is found
   in the *prefix region only* (``_find_choice_idx`` searches
   ``text_tokens[:len_prefix]``, ``runtime.py:1206``). The same word appearing in
   the document does not make a second choice.

The render is::

    ( <parent>: <field> ( <c1> | <c2> | <c3> ) , <field2> ( <c4> ) )

with the choices shuffled and the field order shuffled when training; at
inference both keep declaration order. The fixture records the prefix tokens
verbatim, so the rendering is pinned rather than re-derived.
"""
from __future__ import annotations

import argparse
import copy
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
from gliner2.processor import SchemaTransformer  # noqa: E402

TEXT = "Marie Curie was born in Warsaw and worked in Paris."


def choice_schema(fields, name="trip", descriptions=None):
    """A `json_structures` group whose fields carry literal-enum `choices`.

    The field value itself is a dict carrying `value` and `choices`, which is
    what `_build_classification_prefix` filters on
    (`processor.py:830-835`): a plain `[]` field does not become a choice field.
    """
    schema = {
        "json_structures": [
            {name: {field: {"value": "", "choices": list(values)}
                    for field, values in fields.items()}}
        ],
    }
    if descriptions:
        schema["json_descriptions"] = {
            name: {field: desc for field, desc in descriptions.items()}
        }
    return schema


CASES = [
    (
        "single_choice_field",
        choice_schema({"mood": ["happy"]}),
        0.5,
        "One choice, so there is nothing to choose between; the fixture pins the "
        "prefix shape rather than the verdict.",
    ),
    (
        "three_choices_two_fields",
        {
            "json_structures": [
                {"trip": {
                    "mood": {"value": "", "choices": ["happy", "sad", "neutral"]},
                    "status": {"value": "", "choices": ["done", "pending"]},
                }}
            ]
        },
        0.5,
        "Two choice fields in one group, so the prefix holds two parenthesised "
        "runs and the field order is observable.",
    ),
    (
        "choices_mixed_with_a_plain_field",
        {
            "json_structures": [
                {"trip": {
                    "mood": {"value": "", "choices": ["happy", "sad"]},
                    "destination": [],
                }}
            ]
        },
        0.5,
        "A plain field alongside a choice field: the plain one takes no "
        "parenthesised run, so the prefix is not a uniform shape.",
    ),
    (
        "choices_alongside_a_plain_group",
        {
            "json_structures": [
                {"trip": {"mood": {"value": "", "choices": ["happy", "sad"]}}},
                {"person": {"name": []}},
            ]
        },
        0.5,
        "Two groups, one with choices and one without, so the prefix belongs to "
        "the group that has them and the other group's queries are unaffected.",
    ),
    (
        "choice_values_also_appear_in_the_document",
        {
            "json_structures": [
                {"trip": {"mood": {"value": "", "choices": ["paris", "happy"]}}}
            ]
        },
        0.5,
        "`paris` is both a choice and a word in the text. A port that searched "
        "the whole text stream would find the document occurrence instead of "
        "the prefix one.",
    ),
    (
        "uppercase_choice_values",
        {
            "json_structures": [
                {"trip": {"mood": {"value": "", "choices": ["Happy", "SAD"]}}}
            ]
        },
        0.5,
        "Choice literals are matched case-insensitively by "
        "`_find_choice_idx`, and the prefix is rendered verbatim, so the "
        "lowercased comparison is load-bearing.",
    ),
    (
        "single_character_choices",
        {
            "json_structures": [
                {"trip": {"grade": {"value": "", "choices": ["a", "b", "c"]}}}
            ]
        },
        0.5,
        "One-character choices sit next to the `(` and `|` punctuation in the "
        "prefix, so a tokenizer that merges them would change every index.",
    ),
]


def run_case(processor, encoder, head, schema, threshold):
    batch = processor._collate_batch(
        [(TEXT, copy.deepcopy(schema))],
        max_len=None,
        error_policy="raise",
        build_targets=False,
    )
    prefix = processor._build_classification_prefix(schema)
    text_tokens = list(batch.text_tokens[0])
    # The reference's own bookkeeping: the prefix is prepended and its length
    # recorded, so a choice index is a *text-stream* index and a document index
    # is the same index minus this.
    offset = len(prefix)
    combined = prefix + text_tokens

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

    # `_find_choice_idx` over the prefix region only, lower-cased on both sides.
    found = []
    for group in schema.get("json_structures", []):
        for parent, fields in group.items():
            for field, value in fields.items():
                if not (isinstance(value, dict) and "choices" in value):
                    continue
                entries = []
                seen = set()
                for choice in value["choices"]:
                    if choice in seen:
                        continue
                    seen.add(choice)
                    index = -1
                    for position, token in enumerate(combined[:offset]):
                        if token.lower() == choice.lower():
                            index = position
                            break
                    entries.append([choice, index])
                found.append([f"{parent}.{field}", entries])

    return {
        "text": TEXT,
        "threshold": threshold,
        "schema": schema,
        "prefix_tokens": list(prefix),
        "prefix_length": offset,
        "text_tokens": text_tokens,
        "combined_tokens": combined,
        "input_ids": [int(v) for v in batch.input_ids[0]],
        "text_word_indices": [int(v) for v in batch.text_word_indices[0]],
        "choice_lookup": found,
        "schema_tokens_list": [
            [str(token) for token in tokens] for tokens in batch.schema_tokens_list[0]
        ],
        "count_log_rates": [float(v) for v in out.count_log_rates.reshape(-1)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="boundary checkpoint whose encoder word-routes the text stream",
    )
    parser.add_argument(
        "--out", type=Path,
        default=fixture_dir() / "choice-fields-golden.json",
    )
    args = parser.parse_args()

    base = encoder_dir(args.model)
    processor = SchemaTransformer(str(base), token_pooling="first", word_splitter=None)
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

    payload = {
        "model": args.model,
        "text": TEXT,
        "note": (
            "The choice prefix is prepended to the TEXT token stream "
            "(processor.py:645), not to the schema stream, so the `[C]` marker "
            "stride is unaffected. A choice index is a text-stream index; a "
            "document index is the same index minus prefix_length."
        ),
        "cases": [
            dict(
                name=name,
                note=note,
                **run_case(processor, encoder, head, schema, threshold),
            )
            for name, schema, threshold, note in CASES
        ],
    }
    dump_json(args.out, payload)


if __name__ == "__main__":
    main()
