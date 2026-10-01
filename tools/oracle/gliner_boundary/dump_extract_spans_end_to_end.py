"""End-to-end golden: real text + schema -> final decoded spans.

This is the first oracle in this directory that does not start from synthetic
``text_states``. It wires the reference pipeline end to end:

    SchemaTransformer  ->  input_ids + word/marker routing
    DeBERTa-v3-base    ->  last_hidden_state
    BoundaryHead       ->  pool + pair logits
    decode_candidates  ->  spans

and dumps the decoded spans. The DeBERTa encoder comes from a separate
``microsoft/deberta-v3-base`` download (``models/deberta-v3-base``), not from the
GLiNER checkpoint, so the reference side stays independent of the port: the
Rust tokenizer, prompt builder and encoder are all being checked against
``transformers`` + the reference ``SchemaTransformer``.

Routing mirrors ``BoundaryExtractorModel._encode_core``'s ``fast_routing``
branch (``model.py:1300-1320``) exactly:

    text_states  = hidden[text_word_indices]   * text_word_mask
    query_states = hidden[query_marker_indices] * query_marker_mask

where ``text_word_indices`` are the first-subword positions of each word
(``token_pooling = "first"``) and ``query_marker_indices`` are the
``positions[1:]`` of each schema group — one query per extractive field, since
the group marker itself is not scored.

Span indices are word offsets into the record's word list, which is what the
reference reports.

Regenerate with:
    PYTHONPATH=target/gliner2-oracle \\
        models/.venv/bin/python3 \\
        tools/oracle/gliner_boundary/dump_extract_spans_end_to_end.py

Needs ``models/deberta-v3-base`` (from ModelScope, ``microsoft/deberta-v3-base``).
"""
import argparse
import copy
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import transformers  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from common import build_head, dump_json, fixture_dir  # noqa: E402
from gliner2.processor import SchemaTransformer  # noqa: E402
from gliner2.models.boundary.engine import _resolve_spans  # noqa: E402
from gliner2.models.boundary.model import decode_candidates  # noqa: E402

ENCODER_DIR = REPO_ROOT / "models" / "deberta-v3-base"
# Extractive schema in the shape `_process_entities` expects: the KEYS of
# `entities` are the field names, and each becomes one `[E]` query. Descriptions
# live in a separate top-level `entity_descriptions` map and switch the prompt to
# the "descriptions" example mode, so they are included on purpose. Field order
# is dict insertion order on the Python side and `Vec` order on the Rust side, so
# the query order is part of the contract.
SCHEMA = {
    "entities": {
        "person": ["Ada Lovelace"],
        "organization": ["Babbage Analytical Engine Co"],
        "location": ["London"],
    },
    "entity_descriptions": {
        "person": "an individual human being",
        "organization": "a company or institution",
        "location": "a city, country or other place",
    },
}
# `policy` is base-v1's `boundary_head.overlap_policy`. The last three cases
# drop the threshold so overlapping candidates survive thresholding and the
# resolver actually has to choose between them; at the default threshold the
# model is confident enough that no two spans of one field overlap, which would
# leave the resolution stage untested end to end.
CASES = [
    ("Ada Lovelace worked with Charles Babbage in London.", 0.5, "flat"),
    ("Marie Curie moved to Paris and later to the Curie Institute.", 0.3, "flat"),
    ("nothing here should extract cleanly", 0.5, "flat"),
    ("Marie Curie worked with Pierre Curie in Paris.", 0.02, "flat"),
    ("Dr. John Smith Jr. visited New York City and Boston.", 0.02, "flat"),
    ("Marie Curie worked with Pierre Curie in Paris.", 0.02, "allow"),
]


def _grouped_for_policy(candidates, threshold):
    """`_group_scored_candidates` output: (sample, query) -> [(score, start, end)]."""
    from gliner2.models.boundary.model import _group_scored_candidates

    return _group_scored_candidates(candidates, threshold=threshold)


def run_case(processor, encoder, head, text: str, threshold: float, policy: str) -> dict:
    # `_collate_batch` is the inference entry point (`collate_fn_inference` and
    # `transform_record` both funnel into it) and it calls `_normalize_text`,
    # which appends a "." when the text does not already end in sentence
    # punctuation. `transform_and_format` — which reads like the "main
    # preprocessing entry point" — skips that step, so a text without final
    # punctuation would get one fewer word and every downstream index would
    # shift. Using the entry point the reference's own engine uses.
    normalized = processor._normalize_text(text)
    words = [token for token, _, _ in processor.word_splitter(normalized, lower=True)]
    batch = processor._collate_batch(
        [(text, copy.deepcopy(SCHEMA))],
        max_len=None,
        error_policy="fallback",
        build_targets=False,
    )
    if int(batch.text_word_counts[0]) != len(words):
        raise ValueError(
            f"word count mismatch: {int(batch.text_word_counts[0])} != {len(words)}"
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
    # `decode_candidates` is only the *threshold + sort* half of the engine's
    # decode. `_decode_entities` then runs `_resolve_spans(..., policy)` on the
    # survivors, and for base-v1 the policy is "flat" (canonical "disallow"),
    # i.e. maximum-total-score non-overlapping. Both stages are recorded so a
    # regression in either one is visible.
    decoded = decode_candidates(candidates, threshold=threshold)
    resolved = [
        [
            _resolve_spans([(score, start, end) for score, start, end in scored], policy)
            for scored in sample
        ]
        for sample in _grouped_for_policy(candidates, threshold)
    ]

    return {
        "text": text,
        "normalized_text": normalized,
        "threshold": threshold,
        "input_ids": batch.input_ids[0].tolist(),
        "text_words": words,
        "text_word_first_positions": batch.text_word_indices[0].tolist(),
        "query_marker_indices": batch.query_marker_indices[0].tolist(),
        "query_marker_mask": batch.query_marker_mask[0].tolist(),
        "text_word_mask": batch.text_word_mask[0].tolist(),
        "query_names": [
            spec["field_name"] for spec in _ext_specs(batch.schema_tokens_list[0])
        ],
        "pair_logits": candidates.pair_logits[0].flatten().tolist(),
        "spans": decoded[0],
        "resolved_spans": [
            [[score, start, end] for score, start, end in query]
            for query in resolved[0]
        ],
        "overlap_policy": policy,
    }


def _ext_specs(schema_tokens_list) -> list:
    """One spec per extractive field, in the order the markers were routed.

    ``_encode_core`` flattens ``schema_special_positions[group][1:]`` over the
    non-classification groups, and ``positions`` follows the token layout, so the
    field names are the tokens at the odd indices from 5 of each group.
    """
    specs = []
    for tokens in schema_tokens_list:
        # `["(", "[P]", prompt, "(", "[E]", f0, "[E]", f1, ..., ")", ")"]`
        specs.extend({"field_name": name} for name in tokens[5:-2:2])
    return specs


def load_checkpoint_encoder(encoder) -> None:
    from safetensors.torch import load_file

    state = load_file(str(REPO_ROOT / "models" / "gliner2.5-base-v1" / "model.safetensors"))
    prefix = "encoder."
    mapping = {}
    for key, value in state.items():
        if key.startswith(prefix):
            mapping[key[len(prefix):]] = value
    own = encoder.state_dict()
    missing = sorted(set(own) - set(mapping))
    extra = sorted(set(mapping) - set(own))
    if missing or extra:
        raise ValueError(
            f"encoder key mismatch: {len(missing)} missing (e.g. {missing[:3]}), "
            f"{len(extra)} extra (e.g. {extra[:3]})"
        )
    with torch.no_grad():
        for name, target in own.items():
            value = mapping[name]
            if tuple(target.shape) != tuple(value.shape):
                if name == "embeddings.word_embeddings.weight":
                    # The checkpoint carries 128011 rows (SPM vocab + in-vocab
                    # `[MASK]` + the 10 schema specials); HF pads to 128100.
                    target[: value.shape[0]].copy_(value)
                    continue
                raise ValueError(f"{name}: checkpoint {tuple(value.shape)} != HF {tuple(target.shape)}")
            target.copy_(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=fixture_dir() / "extract-spans-e2e-golden.json")
    args = parser.parse_args()

    processor = SchemaTransformer(
        "models/deberta-v3-base", token_pooling="first", word_splitter=None
    )
    tokenizer = AutoTokenizer.from_pretrained(str(ENCODER_DIR))
    encoder = AutoModel.from_pretrained(str(ENCODER_DIR))
    # The reference loads `microsoft/deberta-v3-base` and then overwrites it with
    # the checkpoint's `encoder.*` tensors during `from_pretrained`. Those are
    # fine-tuned weights: with the stock encoder instead, every pair logit comes
    # out around -15 and the model extracts nothing. The checkpoint's key is the
    # HF name with an `encoder.` prefix (so it nests as `encoder.encoder.layer.*`
    # against HF's `encoder.layer.*`).
    load_checkpoint_encoder(encoder)
    encoder.eval()
    # `SchemaTransformer.__init__` adds the eleven schema specials, which land
    # right after the 128000-entry SPM vocab (plus the in-vocab `[MASK]` at
    # 128000). The GGUF packs the same ids, so assert the agreement instead of
    # assuming it: a mismatch would silently misalign every marker.
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SchemaTransformer.SPECIAL_TOKENS}
    )
    if added != len(SchemaTransformer.SPECIAL_TOKENS):
        raise ValueError(f"added {added} special tokens, expected {len(SchemaTransformer.SPECIAL_TOKENS)}")
    for offset, token in enumerate(SchemaTransformer.SPECIAL_TOKENS):
        index = 128001 + offset
        actual = tokenizer.convert_tokens_to_ids(token)
        if actual != index:
            raise ValueError(f"{token}: tokenizer id {actual} != {index}")

    head = build_head()
    cases = [
        run_case(processor, encoder, head, text, threshold, policy)
        for text, threshold, policy in CASES
    ]
    dump_json(args.out, {"schema": SCHEMA, "threshold_default": 0.5, "cases": cases})
    for case in cases:
        raw = sum(len(row) for row in case["spans"])
        kept = sum(len(row) for row in case["resolved_spans"])
        print(
            f"  {case['text']!r} @{case['threshold']} {case['overlap_policy']}: "
            f"{raw} -> {kept} span(s) across {len(case['query_names'])} queries"
        )


if __name__ == "__main__":
    main()
