"""End-to-end golden for the relation head: real text + relation schema -> edges.

Same encoder path as ``dump_extract_spans_end_to_end.py`` (the checkpoint's
fine-tuned ``encoder.*`` over ``microsoft/deberta-v3-base``), extended to the
relation stage:

    SchemaTransformer  ->  input_ids + word/marker routing
    DeBERTa-v3-base    ->  last_hidden_state
    BoundaryHead       ->  pool + pair logits (the mention candidates)
    _build_rel_specs   ->  relation query states
    TypedRelationPairGenerator -> capped pairs
    SparseRelationScorer -> logits
    _decode_relations  ->  edges

Two things this pins that a generator-or-scorer fixture cannot:

1. **Query-id assignment.** ``_build_rel_specs`` numbers extractive queries in
   schema-group order and takes a relation group's *first two* fields as head and
   tail. A schema with more than one group makes the arithmetic observable — with
   a single group every implementation agrees that head is 0 and tail is 1.
2. **The relation type string.** ``_schema_group_name`` recovers the
   prompt-joined name, and ``_decode_relations``' alias table maps it back to the
   bare schema name. A description that leaks into the output is only visible
   with a ``relation_descriptions`` entry present.

The cases include one where a relation and an entity group share the schema, so
the group ordering and the mixed prompt are both exercised.

Regenerate with:
    HF_HUB_OFFLINE=1 PYTHONPATH=target/gliner2-oracle \\
        models/.venv/bin/python3 \\
        tools/oracle/gliner_boundary/dump_relations_end_to_end.py

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

from transformers import AutoModel, AutoTokenizer  # noqa: E402

from common import build_head, dump_json, fixture_dir  # noqa: E402
from dump_extract_spans_end_to_end import (  # noqa: E402
    ENCODER_DIR,
    load_checkpoint_encoder,
)
from gliner2.models.boundary.relations import (  # noqa: E402
    RelationProposalSettings,
    SparseRelationScorer,
    TypedRelationPairGenerator,
)
from gliner2.processor import SchemaTransformer  # noqa: E402


def _relation_schema():
    """A relation schema plus, in one case, an entity group.

    The relation values are the *gold spans*, which `_process_relations` uses to
    build training targets. At inference only the role field names reach the
    prompt, so the values here are placeholders that make the shape obvious.
    """
    return {
        "relations": [
            {"worked_in": {"head": "person", "tail": "location"}},
            {"collaborated_with": {"head": "person", "tail": "person"}},
        ],
        "relation_descriptions": {
            "worked_in": "who worked in which place",
            "collaborated_with": "who worked with whom",
        },
    }


def _mixed_schema():
    """An entity group *and* relation groups, to pin the query-id arithmetic.

    `_transform_record` emits json_structures, then entities, then relations,
    then classifications, so the entity queries take the lower ids and the
    relation roles start after them.
    """
    return {
        "entities": {
            "person": ["Ada Lovelace"],
            "location": ["London"],
        },
        "entity_descriptions": {
            "person": "an individual human being",
            "location": "a city, country or other place",
        },
        "relations": [
            {"worked_in": {"head": "person", "tail": "location"}},
        ],
        "relation_descriptions": {"worked_in": "who worked in which place"},
    }


CASES = [
    ("Ada Lovelace worked in London.", _relation_schema(), 0.5),
    ("Marie Curie worked with Pierre Curie in Paris.", _relation_schema(), 0.5),
    ("Marie Curie worked with Pierre Curie in Paris.", _relation_schema(), 0.02),
    ("Ada Lovelace worked in London.", _mixed_schema(), 0.5),
    ("nothing here should produce a relation", _relation_schema(), 0.5),
]


def build_relation_scorer(settings, hidden: int) -> SparseRelationScorer:
    from safetensors.torch import load_file

    scorer = SparseRelationScorer(
        hidden,
        dropout=0.0,
        relation_query_dim=(
            2 * hidden if settings.directional_relation_states else hidden
        ),
        use_biaffine_content=settings.relation_biaffine_content,
    )
    state = load_file(str(REPO_ROOT / "models" / "gliner2.5-base-v1" / "model.safetensors"))
    prefix = "relation_scorer."
    scorer.load_state_dict(
        {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)},
        strict=True,
    )
    scorer.eval()
    return scorer


def relation_specs_from_batch(batch, sample: int = 0):
    """`model.py:1440-1520`'s `rel_specs`, rebuilt from the collated batch.

    Mirrors the loop directly: walk the groups in order, skip classification
    groups, number the remaining fields, and treat a relation group's first two
    fields as the head and tail roles.
    """
    from gliner2.models.boundary.model import _extractive_field_names, _schema_group_name
    from gliner2.models.boundary.relations import RelationTypeSpec

    specs, emb_index = [], 0
    for group_index, tokens in enumerate(batch.schema_tokens_list[sample]):
        task_type = batch.task_types[sample][group_index]
        if task_type == "classifications":
            continue
        names = _extractive_field_names(tokens)
        first = emb_index
        emb_index += len(names)
        if task_type == "relations" and emb_index - first >= 2:
            specs.append(
                RelationTypeSpec(
                    _schema_group_name(tokens),
                    head_query_ids=(first,),
                    tail_query_ids=(first + 1,),
                )
            )
    return specs, emb_index


def run_case(processor, encoder, head, scorer, generator, text: str, schema: dict,
             threshold: float) -> dict:
    normalized = processor._normalize_text(text)
    words = [token for token, _, _ in processor.word_splitter(normalized, lower=True)]
    batch = processor._collate_batch(
        [(text, copy.deepcopy(schema))],
        max_len=None,
        error_policy="fallback",
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
    specs, query_count = relation_specs_from_batch(batch)

    relation_states = []
    for spec in specs:
        head_state = query_states[0, spec.head_query_ids[0]]
        tail_state = query_states[0, spec.tail_query_ids[0]]
        if head_state.shape[-1] == width:
            relation_states.append(torch.cat((head_state, tail_state), dim=-1))
        else:
            relation_states.append(torch.stack((head_state, tail_state)).mean(dim=0))
    relation_states = (
        torch.stack(relation_states).unsqueeze(0) if relation_states else None
    )

    edges = []
    pair_count = 0
    if specs:
        pairs = generator.generate(candidates, [None], specs)
        pair_count = len(pairs)
        if pair_count:
            logits = scorer(
                text_states, relation_states, candidates, pairs
            )
            temperature = head.settings.relation_temperature
            alias = {
                f"{name}: {description}": name
                for name, description in schema.get("relation_descriptions", {}).items()
            }
            text_len = len(words)
            for index, logit in enumerate(logits):
                probability = float(torch.sigmoid(logit / temperature))
                if probability < threshold:
                    continue
                name = pairs.relation_types[index]
                name = alias.get(name, name)
                hs, he = int(pairs.head_start[index]), int(pairs.head_end[index])
                ts, te = int(pairs.tail_start[index]), int(pairs.tail_end[index])
                if not (0 <= hs < he <= text_len and 0 <= ts < te <= text_len):
                    continue
                head = " ".join(words[hs:he]).strip()
                tail = " ".join(words[ts:te]).strip()
                if not head or not tail:
                    continue
                edges.append({
                    "relation": name,
                    "score": probability,
                    "head": head,
                    "head_start": hs,
                    "head_end": he,
                    "tail": tail,
                    "tail_start": ts,
                    "tail_end": te,
                })

    return {
        "text": text,
        "threshold": threshold,
        "input_ids": batch.input_ids[0].tolist(),
        "text_words": words,
        "text_word_first_positions": batch.text_word_indices[0].tolist(),
        "query_marker_indices": batch.query_marker_indices[0].tolist(),
        "query_names": [
            spec["field_name"] for spec in _ext_specs(batch.schema_tokens_list[0])
        ],
        "query_count": query_count,
        "relation_specs": [
            [spec.relation_type, list(spec.head_query_ids), list(spec.tail_query_ids)]
            for spec in specs
        ],
        "pair_count": pair_count,
        "edges": edges,
    }


def _ext_specs(schema_tokens_list) -> list:
    specs = []
    for tokens in schema_tokens_list:
        specs.extend({"field_name": name} for name in tokens[5:-2:2])
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=fixture_dir() / "relations-e2e-golden.json"
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
    settings = head.settings
    scorer = build_relation_scorer(settings, 768)
    generator = TypedRelationPairGenerator(
        RelationProposalSettings(
            heads_per_relation=settings.relation_heads_per_type,
            tails_per_relation=settings.relation_tails_per_type,
            pair_cap=settings.relation_pair_cap,
            argument_threshold=settings.relation_argument_proposal_threshold,
        )
    )

    cases = [
        run_case(processor, encoder, head, scorer, generator, text, schema, threshold)
        for text, schema, threshold in CASES
    ]
    dump_json(
        args.out,
        {
            "relation_schema": _relation_schema(),
            "mixed_schema": _mixed_schema(),
            "threshold_default": 0.5,
            "cases": cases,
        },
    )
    for case in cases:
        print(
            f"  {case['text']!r} @{case['threshold']}: {case['pair_count']} pair(s) -> "
            f"{len(case['edges'])} edge(s) {[(e['relation'], e['head'], e['tail']) for e in case['edges']]}",
            file=sys.stderr,
        )
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
