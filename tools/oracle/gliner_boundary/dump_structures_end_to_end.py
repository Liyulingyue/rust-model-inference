"""Oracle for the legacy `json_structures` decode.

Dumps the reference's ``_decode_json_structures`` (``engine.py:506-588``) and
``_format_structure_field`` (``engine.py:762``) over real text and real schemas:
real text + schema -> structure instances.

What this pins
--------------
A ``json_structures`` group takes the **record** path only when the schema
annotates it with a ``mode`` in ``record_metadata``. Without that annotation it
decodes here, and the two paths are not two implementations of one thing:

* records form *instances* (several ``person`` records per schema, exclusive
  fields solved jointly);
* legacy structures emit exactly **one** instance per group. The reference's own
  docstring gives the reason: "Boundary checkpoints do not have the span
  architecture's count-slot axis, so legacy structures are emitted as one
  instance containing all list-valued fields and the best scalar value for each
  scalar field."

So a port that quietly routes an unannotated group through the record head, or
through the plain span path, produces something that looks like a structure and
is not one.

Three details are load-bearing:

1. **A scalar field binds ``spans[0]``** — the first *resolved* span, and the
   rest are discarded. The resolver returns ``(-score, start, end)``, so the
   candidate order decides which value a ``str`` field reports.
2. **Field order is the schema's**, via ``metadata["field_orders"]``, not the
   routed query order.
3. **A structure whose fields all come back empty is dropped**, not emitted as an
   empty object. So the negative case is "absent from the results", not "present
   and empty".

Cases cover scalar-only, list-only, mixed, an unannotated group beside an
annotated one (so the two paths coexist in one schema), a group where a scalar
field's best span and its runner-up differ, and the negative.
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

from transformers import AutoModel, AutoTokenizer  # noqa: E402

from common import build_head, dump_json, fixture_dir  # noqa: E402
from dump_extract_spans_end_to_end import (  # noqa: E402
    ENCODER_DIR,
    load_checkpoint_encoder,
)
from gliner2.processor import SchemaTransformer  # noqa: E402

HIDDEN = 768


def structure_schema(fields, dtypes=None, description=None, name="person"):
    """A ``json_structures`` group with **no** ``record_metadata``.

    ``dtypes`` becomes the ``field_metadata`` the engine reads for
    ``dtype == "str"``. Absent dtypes default to ``"list"``, matching
    ``field_metadata.get("dtype", "list")``.
    """
    schema = {
        "json_structures": [{name: {field: [] for field in fields}}],
    }
    if description:
        schema["json_descriptions"] = {
            name: {field: f"{description} for {field}" for field in fields}
        }
    if dtypes:
        schema["field_metadata"] = {
            f"{name}.{field}": {"dtype": dtype} for field, dtype in dtypes.items()
        }
    return schema


CASES = [
    (
        "Marie Curie worked in Paris with Pierre Curie in London.",
        structure_schema(
            ["name", "city"],
            {"name": "str", "city": "list"},
            description="a person and where they were",
        ),
        0.5,
    ),
    (
        # Two cities, so the `list` field has more than one value to report.
        "Marie Curie worked in Paris and later in London.",
        structure_schema(
            ["name", "city"],
            {"name": "str", "city": "list"},
            description="a person and where they were",
        ),
        0.5,
    ),
    (
        # Every field a list: the default when the schema says nothing about
        # dtypes, and the shape a caller gets for free.
        "Marie Curie worked in Paris with Pierre Curie in London.",
        structure_schema(["name", "city"], description="a person and where they were"),
        0.5,
    ),
    (
        # Every field a scalar: each binds exactly one span and discards the rest,
        # which is the behaviour a span-per-field port cannot reproduce.
        "Marie Curie worked in Paris with Pierre Curie in London.",
        structure_schema(
            ["name", "city"],
            {"name": "str", "city": "str"},
            description="a person and where they were",
        ),
        0.5,
    ),
    (
        # A low threshold so the pool's runner-up spans survive into the list and
        # the scalar's "best span" choice is actually a choice.
        "Marie Curie worked in Paris with Pierre Curie in London.",
        structure_schema(
            ["name", "city"],
            {"name": "str", "city": "list"},
            description="a person and where they were",
        ),
        0.02,
    ),
    (
        # A non-record group and a record group in the same schema: the annotation
        # is per group, so one takes the legacy path and the other does not.
        "Marie Curie worked in Paris.",
        {
            "json_structures": [
                {"person": {"name": [], "city": []}},
                {"trip": {"traveller": [], "destination": []}},
            ],
            "json_descriptions": {
                "person": {"name": "the name", "city": "the city"},
            },
            "record_metadata": {
                "trip": {
                    "mode": "natural",
                    "anchor": "traveller",
                    "fields": {
                        "destination": {"cardinality": "optional_one", "exclusive": True}
                    },
                }
            },
            "field_metadata": {"person.name": {"dtype": "str"}},
        },
        0.5,
    ),
    (
        "nothing structured here at all",
        structure_schema(
            ["name", "city"],
            {"name": "str", "city": "list"},
        ),
        0.99,
    ),
]


def query_layout_from_batch(batch, sample: int = 0):
    """Extractive query names in routed order, for the `[C]` group's field map."""
    from gliner2.models.boundary.model import _extractive_field_names

    names = []
    for tokens in batch.schema_tokens_list[sample]:
        names.extend(_extractive_field_names(tokens))
    return names


def legacy_groups(schema):
    """`{group: {field: dtype}}` for groups the schema left unannotated."""
    annotated = set()
    for name, config in (schema.get("record_metadata") or {}).items():
        if isinstance(config, dict) and config.get("mode"):
            annotated.add(name)
    out = {}
    for item in schema.get("json_structures", []):
        for name, occurrence in item.items():
            if name in annotated:
                continue
            fields = list(occurrence.keys()) if isinstance(occurrence, dict) else list(occurrence)
            dtypes = {}
            for field in fields:
                meta = (schema.get("field_metadata") or {}).get(f"{name}.{field}", {})
                dtypes[field] = meta.get("dtype", "list")
            out[name] = {"fields": fields, "dtypes": dtypes}
    return out


def run_case(processor, encoder, head, text, schema, threshold, overlap_policy):
    from gliner2.models.boundary.model import _group_scored_candidates

    normalized = processor._normalize_text(text)
    words = [token for token, _, _ in processor.word_splitter(normalized, lower=True)]
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
    grouped = _group_scored_candidates(candidates, threshold=threshold)[0]

    # Drive the reference's own decoder. It is an instance method on the engine
    # and needs a fair amount of state, so the two pieces it actually reads
    # (`_resolve_spans` and `_format_structure_field`) are applied here in the
    # same order, which is the whole of its body for a group with no `choices`.
    from gliner2.models.boundary.engine import _resolve_spans

    names = query_layout_from_batch(batch)
    groups = legacy_groups(schema)
    offset = max(int(batch.text_word_counts[0]) - len(words), 0)
    text_len = len(words)
    results = []
    for group_name, info in groups.items():
        instance = []
        for field in info["fields"]:
            query_id = names.index(field)
            spans = []
            for probability, start, end in _resolve_spans(
                grouped[query_id] if query_id < len(grouped) else [], overlap_policy
            ):
                token_start, token_end = start - offset, end - offset
                if not (0 <= token_start < token_end <= text_len):
                    continue
                surface = " ".join(words[token_start:token_end]).strip()
                if not surface:
                    continue
                spans.append((surface, float(probability), token_start, token_end))
            is_scalar = info["dtypes"].get(field) == "str"
            if not is_scalar:
                instance.append({
                    field: [
                        {"text": s, "confidence": p, "start": a, "end": b}
                        for s, p, a, b in spans
                    ]
                })
            elif spans:
                surface, probability, char_start, char_end = spans[0]
                instance.append({
                    field: {
                        "text": surface,
                        "confidence": probability,
                        "start": char_start,
                        "end": char_end,
                    }
                })
            else:
                instance.append({field: None})
        merged = {}
        for entry in instance:
            merged.update(entry)
        if any(value is not None and value != [] for value in merged.values()):
            results.append({"task": group_name, "fields": merged})

    return {
        "text": text,
        "threshold": threshold,
        "overlap_policy": overlap_policy,
        "schema": schema,
        "text_words": words,
        "query_names": names,
        "grouped_spans": [
            [[float(p), int(a), int(b)] for p, a, b in query] for query in grouped
        ],
        "structures": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=fixture_dir() / "structures-e2e-golden.json"
    )
    args = parser.parse_args()

    processor = SchemaTransformer(
        "models/deberta-v3-base", token_pooling="first", word_splitter=None
    )
    tokenizer = AutoTokenizer.from_pretrained(str(ENCODER_DIR))
    encoder = AutoModel.from_pretrained(str(ENCODER_DIR))
    load_checkpoint_encoder(encoder)
    encoder.eval()
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SchemaTransformer.SPECIAL_TOKENS}
    )
    if added != len(SchemaTransformer.SPECIAL_TOKENS):
        raise ValueError(f"added {added} special tokens")

    head = build_head()
    overlap_policy = "flat"  # base-v1's boundary_head.overlap_policy

    cases = [
        run_case(processor, encoder, head, text, schema, threshold, overlap_policy)
        for text, schema, threshold in CASES
    ]
    dump_json(args.out, {"overlap_policy": overlap_policy, "cases": cases})
    for case in cases:
        print(
            f"  {case['text']!r} @{case['threshold']}: "
            f"{len(case['structures'])} structure(s) "
            f"{[(s['task'], {k: v for k, v in s['fields'].items()}) for s in case['structures']]}",
            file=sys.stderr,
        )
    print(f"wrote {args.out} ({len(cases)} cases)", file=sys.stderr)


if __name__ == "__main__":
    main()
