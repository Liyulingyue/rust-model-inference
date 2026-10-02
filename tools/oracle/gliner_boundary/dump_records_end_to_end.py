"""End-to-end golden for the record head: real text + record schema -> records.

Same encoder path as ``dump_extract_spans_end_to_end.py`` and
``dump_relations_end_to_end.py`` (the checkpoint's fine-tuned ``encoder.*`` over
``microsoft/deberta-v3-base``), extended through the record stage:

    SchemaTransformer        ->  input_ids + word/marker routing
    DeBERTa-v3-base          ->  last_hidden_state
    BoundaryHead             ->  pool, pair logits, and the real
                                ``candidate_encoder`` states
    compile_record_specs     ->  RecordSpec bound to the routed query ids
    RecordHead.forward_group ->  instance states, object logits, assign logits
    decode_group             ->  records

Why this oracle exists when ``dump_record_head.py`` already covers the head
-----------------------------------------------------------------------------
The head fixture feeds the head a *synthetic* ``candidate_states`` — an invented
formula — so it pins the head's arithmetic but not the handoff. Here the states
come from the checkpoint's ``candidate_encoder`` over the real refined boundary
states, the candidate batch is the real pool, and the field/query ids come from
the real prompt routing. Every coordinate, layout and broadcast mistake in that
handoff lives here and nowhere else.

That class of mistake is not hypothetical: a one-column offset in the assignment
cost matrix shipped green through the head fixture and was only caught by
comparing against the reference, and a ``width`` shadowing bug only surfaced once
the real shapes were involved.

Cases
-----
``natural_*``
    Instances are the anchor field's candidates, scored by the pool's
    ``pair_logits`` straight through.
``exclusive_greedy_would_lose``
    The case the global assignment exists for: two people and one place, the place
    declared ``exclusive``. A greedy pass that lets the highest-scoring instance
    claim its favourite candidate forces the other instance onto an unrelated
    span (or onto nothing). Asserted semantically as well as against the
    reference, because "the reference agrees" is not the same as "the assignment
    is right".
``latent_*`` / ``anchorless_*``
    The other two instance modes, over real text.
``mixed_*``
    An entity group *and* a record group, so the record's field query ids have to
    start after the entity queries. The one case where that arithmetic is
    observable: a lone record group agrees with any implementation.
``negative`` and ``low_threshold``
    No records above the anchor threshold, and a threshold low enough that
    several instances survive so the ABSENT columns get exercised.

Regenerate with:
    HF_HUB_OFFLINE=1 models/.venv/bin/python \\
        tools/oracle/gliner_boundary/dump_records_end_to_end.py

Needs ``models/deberta-v3-base``.
"""
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
from gliner2.models.boundary.records import decode_group  # noqa: E402
from gliner2.processor import SchemaTransformer  # noqa: E402

HIDDEN = 768


def record_schema(roles, mode, anchor=None, fields=None, description=None):
    """A single ``json_structures`` group, annotated as a record.

    ``roles`` is the field list in declaration order. ``fields`` is the
    per-field metadata mapping the reference's ``normalize_record_metadata``
    reads; the group description becomes the ``[P]`` prompt.
    """
    # The field list comes from `roles` — the declared field names, in order. It
    # must NOT come from `fields`, which is only the per-field *metadata* map: a
    # schema whose `fields` omits a role (e.g. an unannotated non-anchor) would
    # then silently lose that field, and a `natural` group's `anchor` would name
    # something the layout never routed.
    # The reference's own processing (`_process_json_structures`) tolerates a bare
    # `{field: gold_span}` dict as a single occurrence and unions its keys; the
    # list-of-occurrences form also works but routes through a different legacy
    # branch that expects dicts. The dict form is what the engine accepts at
    # inference and what this oracle pins.
    entry = {role: [] for role in roles}
    schema = {"json_structures": [{"person": entry}]}
    spec = {"mode": mode}
    if anchor is not None:
        spec["anchor"] = anchor
    if fields:
        spec["fields"] = fields
    schema["record_metadata"] = {"person": spec}
    if description:
        # `json_descriptions[parent]` is a **field -> description map**, not a
        # single string on the group: `_process_json_structures` does
        # `descs.get(parent, {})` and then iterates `.items()`, filtering to the
        # declared fields. A plain string there raises `AttributeError`, and
        # `error_policy="fallback"` swallows it into a dummy `[E] entity` record —
        # so the schema silently stops being a record schema at all.
        # No parentheses in the description: the Rust `Task::validate` rejects
        # "(" in a label/prompt because it is a reserved schema marker, and the
        # reference's descriptions never need one.
        schema["json_descriptions"] = {
            "person": {role: f"{description} for {role}" for role in roles}
        }
    return schema


CASES = [
    (
        "Ada Lovelace worked in London for Babbage Analytical Engine Co.",
        record_schema(
            ["name", "employer"],
            "natural",
            anchor="name",
            fields={"employer": {"cardinality": "optional_one", "exclusive": True}},
            description="a person and who they work for",
        ),
        0.5,
    ),
    (
        # The greedy-failure case. Two people, one place, place is exclusive.
        "Marie Curie worked with Pierre Curie in Paris.",
        record_schema(
            ["name", "city"],
            "natural",
            anchor="name",
            fields={"city": {"cardinality": "optional_one", "exclusive": True}},
            description="a person and where they are",
        ),
        0.5,
    ),
    (
        "Marie Curie worked with Pierre Curie in Paris.",
        record_schema(
            ["name", "city"],
            "natural",
            anchor="name",
            fields={"city": {"cardinality": "optional_one", "exclusive": True}},
            description="a person and where they are",
        ),
        0.02,
    ),
    (
        "Marie Curie worked with Pierre Curie in Paris and later in London.",
        record_schema(
            ["name", "city"],
            "natural",
            anchor="name",
            fields={"city": {"cardinality": "zero_or_more", "exclusive": True}},
            description="a person and every place they were",
        ),
        0.02,
    ),
    (
        "Tim Cook is the CEO of Apple Inc. in Cupertino.",
        record_schema(
            ["name", "company", "city"],
            "natural",
            anchor="name",
            fields={
                "company": {"cardinality": "optional_one", "exclusive": True},
                "city": {"cardinality": "zero_or_more"},
            },
            description="a person, their company and their city",
        ),
        0.5,
    ),
    (
        "Marie Curie worked with Pierre Curie in Paris.",
        record_schema(
            ["name", "city"],
            "latent",
            fields={
                "name": {"cardinality": "required_one"},
                "city": {"cardinality": "zero_or_more"},
            },
            description="a person and where they are",
        ),
        0.5,
    ),
    (
        "Marie Curie worked with Pierre Curie in Paris.",
        record_schema(
            ["name", "city"],
            "anchorless",
            fields={
                "name": {"cardinality": "optional_one"},
                "city": {"cardinality": "zero_or_more"},
            },
            description="a person and where they are",
        ),
        0.5,
    ),
    (
        # Entities *and* a record group: the record's fields must take the query
        # ids after the entity queries.
        "Ada Lovelace worked in London.",
        {
            "entities": {"person": ["Ada Lovelace"], "location": ["London"]},
            "entity_descriptions": {
                "person": "an individual human being",
                "location": "a city, country or other place",
            },
            "json_structures": [
                {"person": {"name": [], "employer": []}}
            ],
            "record_metadata": {
                "person": {
                    "mode": "natural",
                    "anchor": "name",
                    "fields": {
                        "employer": {
                            "cardinality": "optional_one",
                            "exclusive": True,
                        }
                    },
                }
            },
            "json_descriptions": {
                "person": {"name": "the full name", "employer": "who they work for"}
            },
        },
        0.5,
    ),
    (
        "nothing structured here at all",
        record_schema(
            ["name", "employer"],
            "natural",
            anchor="name",
            fields={"employer": {"cardinality": "optional_one", "exclusive": True}},
        ),
        0.5,
    ),
]


def build_record_scorer(settings):
    from gliner2.models.boundary.records import RecordHead
    from safetensors.torch import load_file

    head = RecordHead(
        HIDDEN, settings.record_dim, settings.record_instance_queries
    )
    state = load_file(
        str(REPO_ROOT / "models" / "gliner2.5-base-v1" / "model.safetensors")
    )
    prefix = "record_decoder."
    head.load_state_dict(
        {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)},
        strict=True,
    )
    head.eval()
    return head


def query_layout_from_batch(batch, sample: int = 0):
    """Rebuild the `QueryLayout` the record spec compiles against.

    Mirrors the reference's query numbering: walk the schema groups in order,
    skip classification groups entirely (they emit no extractive query), and give
    every remaining field the next global id. That id space is shared with the
    relation head, which is why a mixed schema is the only way to observe it.
    """
    from gliner2.models.base import QueryLayout, QuerySpec
    from gliner2.models.boundary.model import (
        _extractive_field_names,
        _schema_group_name,
    )

    queries = []
    query_id = 0
    for group_index, tokens in enumerate(batch.schema_tokens_list[sample]):
        task_type = batch.task_types[sample][group_index]
        if task_type == "classifications":
            continue
        name = _schema_group_name(tokens)
        for role_index, role in enumerate(_extractive_field_names(tokens)):
            queries.append(
                QuerySpec(
                    query_id=query_id,
                    task_index=group_index,
                    task_type=task_type,
                    task_name=name,
                    role_index=role_index,
                    role_name=role,
                )
            )
            query_id += 1
    return QueryLayout(queries=tuple(queries))


def field_dtypes_from_schema(schema):
    """`{task_name: {field: dtype}}` from the schema's `json_descriptions`.

    The reference uses the declared dtype to pick a default cardinality, so a
    schema that describes fields has to supply it or the defaults shift.
    """
    out = {}
    for parent, fields in (schema.get("json_descriptions") or {}).items():
        if isinstance(fields, dict):
            out[parent] = {name: "str" for name in fields}
    return out


def run_case(processor, encoder, head, record_head, text, schema, threshold):
    from gliner2.processing.records import compile_record_specs

    settings = head.settings
    normalized = processor._normalize_text(text)
    words = [token for token, _, _ in processor.word_splitter(normalized, lower=True)]
    batch = processor._collate_batch(
        [(text, copy.deepcopy(schema))],
        max_len=None,
        # `error_policy="fallback"` would replace a malformed schema with a dummy
        # `[E] entity` record and keep going, so a schema typo shows up as "no
        # records" instead of as the schema error it is. Raise instead.
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

    layout = query_layout_from_batch(batch)
    specs = compile_record_specs(
        query_layout=layout,
        record_metadata=schema.get("record_metadata"),
        field_dtypes=field_dtypes_from_schema(schema),
    )
    spec_errors = []
    records = []
    for spec in specs.values():
        # `forward_group` takes the sample's query states as `[Q, H]`, 2-D — the
        # engine passes `query_states_i`, one sample slice, not the `[1, Q, H]`
        # batch. Passing 3-D makes the reference broadcast the field query against
        # the instance batch and blow up with a shape error deep in
        # `_assign_logits`.
        query_states_i = query_states[0]
        with torch.inference_mode():
            group = record_head.forward_group(spec, query_states_i, candidates, 0)
        decoded = decode_group(
            group,
            anchor_threshold=settings.record_anchor_threshold,
            field_threshold=settings.record_field_threshold,
            object_threshold=settings.record_anchor_proposal_threshold,
            temperature=settings.record_temperature,
        )
        for record in decoded:
            records.append(
                {
                    "task": spec.task_name,
                    "mode": spec.mode,
                    "score": float(record.score),
                    "anchor_span": None
                    if record.anchor_span is None
                    else [int(record.anchor_span[0]), int(record.anchor_span[1])],
                    "fields": {
                        str(qid): {
                            "spans": [[int(s[0]), int(s[1])] for s in spans],
                            "text": [
                                " ".join(words[s[0] : s[1]]) for s in spans
                            ],
                            "scores": [float(v) for v in record.field_scores[qid]],
                        }
                        for qid, spans in sorted(record.fields.items())
                    },
                }
            )

    return {
        "text": text,
        "threshold": threshold,
        "schema": schema,
        "input_ids": batch.input_ids[0].tolist(),
        "text_words": words,
        "text_word_first_positions": batch.text_word_indices[0].tolist(),
        "query_marker_indices": batch.query_marker_indices[0].tolist(),
        "query_names": [
            q.role_name for q in layout.queries
        ],
        "query_ids": [q.query_id for q in layout.queries],
        "query_tasks": [q.task_name for q in layout.queries],
        "specs": [
            {
                "task_name": spec.task_name,
                "mode": spec.mode,
                "anchor_query_id": spec.anchor_query_id,
                "fields": [
                    {
                        "query_id": f.query_id,
                        "name": f.name,
                        "cardinality": f.cardinality.value,
                        "is_anchor": f.is_anchor,
                        "exclusive": f.exclusive,
                    }
                    for f in spec.fields
                ],
            }
            for spec in specs.values()
        ],
        "pair_logits": candidates.pair_logits[0].flatten().tolist(),
        "candidate_states_present": candidates.candidate_states is not None,
        "candidate_state_width": (
            int(candidates.candidate_states.shape[-1])
            if candidates.candidate_states is not None
            else 0
        ),
        "record_errors": spec_errors,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="boundary checkpoint to score; picks the base encoder and head width",
    )
    parser.add_argument(
        "--out", type=Path, default=fixture_dir() / "records-e2e-golden.json"
    )
    args = parser.parse_args()

    base = encoder_dir(args.model)
    # The reference tokenizes with the *base* encoder's tokenizer and then adds
    # the ten schema specials, so point SchemaTransformer at the same directory
    # rather than the GLiNER repo, whose tokenizer.json already has them.
    processor = SchemaTransformer(
        str(base), token_pooling="first", word_splitter=None
    )
    tokenizer = AutoTokenizer.from_pretrained(str(base))
    # Built from the config rather than `from_pretrained`: every tensor is
    # replaced by the checkpoint's own weights in `load_checkpoint_encoder`,
    # which raises unless its key set matches exactly. Loading the base weights
    # first would only cost a gigabyte per encoder and add a way to be wrong.
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(str(base)))
    load_checkpoint_encoder(encoder, args.model)
    encoder.eval()
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SchemaTransformer.SPECIAL_TOKENS}
    )
    if added != len(SchemaTransformer.SPECIAL_TOKENS):
        raise ValueError(f"added {added} special tokens")

    head = build_head(model=args.model)
    record_head = build_record_scorer(head.settings)

    cases = []
    for text, schema, threshold in CASES:
        case = run_case(processor, encoder, head, record_head, text, schema, threshold)
        cases.append(case)
        print(
            f"  {text!r} @{threshold} mode="
            f"{case['specs'][0]['mode'] if case['specs'] else '?'}: "
            f"{len(case['records'])} record(s) "
            f"{[(r['task'], {k: v['text'] for k, v in r['fields'].items()}) for r in case['records']]}",
            file=sys.stderr,
        )

    dump_json(
        args.out,
        {
            "threshold_default": 0.5,
            "cases": cases,
        },
    )
    print(f"wrote {args.out} ({len(cases)} cases)", file=sys.stderr)


if __name__ == "__main__":
    main()
