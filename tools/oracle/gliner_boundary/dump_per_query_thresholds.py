"""Oracle for per-entity, per-field and per-relation thresholds.

The reference resolves a configured threshold in three *different* places, and
the three do not agree on which key they read or on whether the candidate
stage sees them at all:

1. ``_query_thresholds`` (``engine.py:197-226``) builds a ``[B, Q]`` tensor
   that ``_group_scored_candidates`` thresholds the candidate pool with. It
   reads ``entity_metadata[<label>]["threshold"]`` for ``entities`` queries and
   ``field_metadata["<group>.<field>"]["threshold"]`` for ``json_structures``
   queries — and nothing else. Relations are absent from the ``if``/``elif``
   chain, so a relation query keeps the caller's global threshold.
2. ``_decode_relations`` (``engine.py:849-853``) re-applies a per-type
   threshold itself, from ``relation_metadata[<type>]``. So a relation's
   configured threshold is applied once here and is *not* the candidate-stage
   threshold — the candidate stage saw the global one.
3. ``_decode_choice_field`` reads ``field_metadata["<group>.<field>"]
   ["threshold"]`` again for choice fields.

What this pins
--------------
* Which task types the candidate stage honours. A port that lets a relation's
  configured threshold reach the candidate stage changes which pairs are even
  generated, not just which are kept, so this is observable in the pair count
  and not only in the edge list.
* That the two relation thresholds are independent: setting
  ``relation_metadata[t]["threshold"]`` high must not empty the candidate pool
  for that relation's queries.
* ``relation_metadata`` lookup happens on the *resolved* type name, after the
  ``"<name>: <description>"`` alias map is inverted (``engine.py:837-843``), so
  a relation declared with a description is configured under its bare name.
* An explicit ``null`` threshold falls back to the caller's, rather than
  comparing against zero — ``engine.py:851-853`` re-checks for ``None`` after
  the ``.get(..., threshold)`` default, which is redundant except when the key
  is present and null.
* Absent metadata leaves every threshold at the caller's default, including for
  labels the schema never mentions.

Each case dumps the resolved ``[B, Q]`` tensor next to the decode, so a port
that gets the right spans for the wrong reason is still caught.
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
from gliner2.inference.engine import BoundaryExtractor  # noqa: E402
from gliner2.processor import SchemaTransformer  # noqa: E402

ENTITIES = {"entities": {"person": {}, "city": {}}}
STRUCTURES = {
    "json_structures": [{"trip": {"traveller": [], "destination": []}}],
}
# `relations` is a list of single-key dicts, the same shape as
# `json_structures` (`processor.py:1081-1085`).
RELATIONS = {
    "relations": [{"works_at": {"head": "person", "tail": "city"}}],
}

TEXT = "Marie Curie worked in Paris with Pierre Curie in London."


def case(name, schema, threshold, note=""):
    return (name, schema, threshold, note)


CASES = [
    case(
        "no_metadata_everything_uses_the_caller_threshold",
        ENTITIES,
        0.5,
        "The baseline: a schema that says nothing about thresholds must behave "
        "exactly like one that has no per-field support at all.",
    ),
    case(
        "per_entity_thresholds_differ_per_label",
        {
            "entities": {"person": {}, "city": {}},
            "entity_metadata": {
                "person": {"threshold": 0.05},
                "city": {"threshold": 0.9},
            },
        },
        0.5,
        "One query admits a wide net, the other nearly nothing. A port that "
        "applied the first entity's threshold to every query, or that kept "
        "the global one, disagrees on both queries.",
    ),
    case(
        "per_field_threshold_on_a_structure",
        {
            "json_structures": [{"trip": {"traveller": [], "destination": []}}],
            "field_metadata": {
                "trip.traveller": {"threshold": 0.02},
                "trip.destination": {"threshold": 0.95},
            },
        },
        0.5,
        "`field_metadata` is keyed `<group>.<field>`, and the lookup is by "
        "task name rather than by field name alone.",
    ),
    case(
        "field_metadata_does_not_leak_across_groups",
        {
            "json_structures": [
                {"trip": {"traveller": [], "destination": []}},
                {"visit": {"who": [], "where": []}},
            ],
            "field_metadata": {
                "trip.traveller": {"threshold": 0.02},
                "trip.destination": {"threshold": 0.02},
            },
        },
        0.5,
        "`visit`'s fields have no configured threshold, so they must stay at "
        "the caller's even though a same-named-looking group is configured.",
    ),
    case(
        "per_relation_threshold",
        {
            "relations": [
                {"works_at": {"head": "person", "tail": "city"}},
                {"studied_in": {"head": "person", "tail": "city"}},
            ],
            "relation_metadata": {
                "works_at": {"threshold": 0.02},
                "studied_in": {"threshold": 0.99},
            },
        },
        0.5,
        "The candidate stage keeps the global threshold for relation queries; "
        "the per-type threshold is applied once, in the decoder.",
    ),
    case(
        "relation_threshold_is_looked_up_by_resolved_type",
        {
            "relations": [{"works_at": {"head": "person", "tail": "city"}}],
            "relation_descriptions": {"works_at": "where they are employed"},
            "relation_metadata": {"works_at": {"threshold": 0.02}},
        },
        0.5,
        "The relation is declared with a description, so the pair generator "
        "reports `\"works_at: where they are employed\"`. The metadata key is "
        "still the bare name, because the alias map is inverted first.",
    ),
    case(
        "explicit_null_relation_threshold_falls_back",
        {
            "relations": [{"works_at": {"head": "person", "tail": "city"}}],
            "relation_metadata": {"works_at": {"threshold": None}},
        },
        0.5,
        "A present-but-null threshold must fall back to the caller's rather "
        "than compare against zero, which would admit everything.",
    ),
    case(
        "entities_and_structures_and_relations_together",
        {
            "entities": {"person": {}, "city": {}},
            "json_structures": [{"trip": {"traveller": [], "destination": []}}],
            "relations": [{"works_at": {"head": "person", "tail": "city"}}],
            "entity_metadata": {"person": {"threshold": 0.02}},
            "field_metadata": {"trip.destination": {"threshold": 0.02}},
            "relation_metadata": {"works_at": {"threshold": 0.02}},
        },
        0.5,
        "All three resolution sites in one schema, so the query-index mapping "
        "has to be right for every task type at once.",
    ),
    case(
        "relation_role_names_collide_with_entity_labels",
        {
            "entities": {"person": {}, "city": {}},
            "relations": [{"works_at": {"head": "person", "tail": "city"}}],
            # `head` and `tail` are the relation group's own field names, and a
            # schema may also declare entity labels called `head`/`tail`. A port
            # that resolved relation queries through `entity_metadata` would pick
            # these entries up and throttle the relation's candidate stage, which
            # the reference never does.
            "entity_metadata": {
                "person": {"threshold": 0.02},
                "head": {"threshold": 0.99},
                "tail": {"threshold": 0.99},
            },
            "relation_metadata": {"works_at": {"threshold": 0.02}},
        },
        0.5,
        "Entity labels that collide with a relation role name must not leak into "
        "the relation's queries.",
    ),
    case(
        "high_global_threshold_with_low_overrides",
        {
            "entities": {"person": {}, "city": {}},
            "entity_metadata": {
                "person": {"threshold": 0.02},
                "city": {"threshold": 0.02},
            },
        },
        0.99,
        "Overrides below the caller's threshold must be honoured; a port that "
        "took the max of the two would report nothing.",
    ),
    case(
        "overrides_above_the_caller_threshold_can_only_empty",
        {
            "entities": {"person": {}, "city": {}},
            "entity_metadata": {
                "person": {"threshold": 0.99},
                "city": {"threshold": 0.99},
            },
        },
        0.02,
        "The mirror image: overrides above the caller's threshold suppress "
        "everything, which a port that treated them as a floor would not.",
    ),
]


def build_extractor(base: Path, model: str):
    processor = SchemaTransformer(str(base), token_pooling="first", word_splitter=None)
    tokenizer = AutoTokenizer.from_pretrained(str(base))
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(str(base)))
    load_checkpoint_encoder(encoder, model)
    encoder.eval()
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SchemaTransformer.SPECIAL_TOKENS}
    )
    if added != len(SchemaTransformer.SPECIAL_TOKENS):
        raise ValueError(f"added {added} special tokens")
    head = build_head(model=model)
    return processor, encoder, head


def run_case(processor, encoder, head, schema, threshold):
    """Resolve the threshold tensor for one schema.

    `_query_thresholds` is a `@staticmethod` and the relation-side resolution is
    inline in the decoder, so neither needs a constructed `BoundaryExtractor` —
    which would drag in a loaded model for a feature that is pure schema input.
    """
    from gliner2.inference.engine import BoundaryExtractor
    from gliner2.models.boundary.model import _extractive_field_names

    batch = processor._collate_batch(
        [(TEXT, copy.deepcopy(schema))],
        max_len=None,
        error_policy="raise",
        build_targets=False,
    )

    specs = []
    for tokens in batch.schema_tokens_list[0]:
        names = _extractive_field_names(tokens)
        task_type = {
            "[E]": "entities",
            "[C]": "json_structures",
            "[R]": "relations",
        }.get(tokens[4] if len(tokens) > 4 else "", "entities")
        task_name = tokens[2].split(" [DESCRIPTION] ")[0]
        for name in names:
            specs.append({
                "task_type": task_type,
                "task_name": task_name,
                "field_name": name,
            })

    thresholds = BoundaryExtractor._query_thresholds(
        [specs],
        [{"entity_metadata": schema.get("entity_metadata") or {},
          "field_metadata": schema.get("field_metadata") or {}}],
        threshold,
        torch.device("cpu"),
    )

    # The resolved per-relation threshold, computed the way the decoder does it:
    # the candidate stage never sees it.
    relation_thresholds = {}
    for name, config in (schema.get("relation_metadata") or {}).items():
        value = (config or {}).get("threshold", threshold)
        if value is None:
            value = threshold
        relation_thresholds[name] = None if value is None else float(value)

    return {
        "schema": schema,
        "threshold": threshold,
        "query_specs": specs,
        "query_thresholds": [[float(v) for v in row] for row in thresholds],
        "relation_thresholds": relation_thresholds,
        "input_ids": [int(v) for v in batch.input_ids[0]],
        "text_word_count": int(batch.text_word_counts[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="boundary checkpoint whose config supplies the decode settings",
    )
    parser.add_argument(
        "--out", type=Path,
        default=fixture_dir() / "per-query-thresholds-golden.json",
    )
    args = parser.parse_args()

    base = encoder_dir(args.model)
    processor, encoder, head = build_extractor(base, args.model)

    payload = {
        "model": args.model,
        "text": TEXT,
        "note": (
            "Every shipped boundary checkpoint leaves the global threshold at "
            "the caller default, so these thresholds are pure schema input. "
            "The candidate stage sees `query_thresholds`; relations get their "
            "configured threshold re-applied in the decoder instead."
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
