"""Golden for the classification head, ``null_projection`` and ``count_head``.

Three heads the extraction path does not touch:

* the **classification head** — the reference's shared ``classifier.0`` + ReLU +
  ``classifier.3`` applied to the ``[C]`` marker states. ``_encode_core`` routes
  those to ``cls_marker_indices``, *not* ``query_marker_indices``
  (``processor.py:712-719``), so they never reach the document pool. The decode
  is ``inference/runtime.py:562``: drop the group's ``[P]`` row, divide by
  ``classification_temperature``, softmax for a single-label group and sigmoid
  for a multi-label one, threshold at ``cls_threshold``.
* ``null_projection`` — one scalar per extractive query. The reference drops a
  whole query's spans when ``sigmoid(null_logits[q]) > abstention_threshold``.
* ``count_head`` — per-query count log-rate, only consumed when
  ``adaptive_threshold`` is on (base-v1 leaves it off, so it is recorded rather
  than applied).

The reference side runs the real ``_encode_core`` routing and the real
``_extract_classification_result``, so the routing indices in the fixture are the
ones inference uses rather than ones this file re-derived.

Regenerate with:
    HF_HUB_OFFLINE=1 PYTHONPATH=target/gliner2-oracle \\
        models/.venv/bin/python3 \\
        tools/oracle/gliner_boundary/dump_classification_and_query_heads.py
"""
import argparse
import copy
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import HIDDEN_SIZE, build_head, dump_json, fixture_dir  # noqa: E402
from dump_extract_spans_end_to_end import load_checkpoint_encoder  # noqa: E402

from transformers import AutoModel, AutoTokenizer  # noqa: E402
from gliner2.processor import SchemaTransformer  # noqa: E402
from gliner2.configuration import BoundaryHeadSettings  # noqa: E402
from gliner2.models.boundary.encoding import BoundaryAttentionBlock  # noqa: E402
from gliner2.models.boundary.heads import BoundaryQueryHead  # noqa: E402
from gliner2.models.boundary.scoring import SparseBoundaryPairScorer  # noqa: E402
from gliner2.models.boundary.content import SpanContentPooler  # noqa: E402
from gliner2.models.boundary.proposal import (  # noqa: E402
    ProposalSettings, SparseBoundaryProposer,
)

ENCODER_DIR = REPO_ROOT / "models" / "deberta-v3-base"

# One extractive group plus one classification group, so the fixture covers both
# routings at once. The classification group is single-label ("auto" ->
# softmax); the second case is multi-label with an explicit threshold.
# `schema["classifications"]` is a *list* of `{"task", "labels", ...}` items —
# `_process_classifications` iterates it and builds one group per entry, with
# `[L]` as the child marker (`processor.py:1129-1188`). The matching decode
# config is looked up by task name in `_resolve_classification_config`.
CASES = [
    (
        "Ada Lovelace worked with Charles Babbage in London.",
        {
            "entities": {"person": ["Ada Lovelace"]},
            "classifications": [
                {
                    "task": "topic",
                    "labels": ["science", "commerce", "art"],
                    "label_descriptions": {
                        "science": "scientific research and discovery",
                        "commerce": "trade, business and industry",
                        "art": "creative and cultural work",
                    },
                }
            ],
        },
        {"task": "topic", "labels": ["science", "commerce", "art"], "class_act": "auto"},
    ),
    (
        "The order shipped late and support was unreachable.",
        {
            "entities": {},
            "classifications": [
                {
                    "task": "sentiment",
                    "labels": ["negative", "neutral", "positive"],
                    "multi_label": True,
                    "cls_threshold": 0.5,
                }
            ],
        },
        {
            "task": "sentiment",
            "labels": ["negative", "neutral", "positive"],
            "class_act": "auto",
            "multi_label": True,
            "cls_threshold": 0.5,
        },
    ),
]


def reference_heads(model_dir: Path):
    """The four modules the classification / query heads read, from the
    checkpoint. Built individually rather than through ``BoundaryHead`` so a
    missing key names the module that wanted it."""
    from safetensors.torch import load_file

    config = __import__("json").loads((model_dir / "config.json").read_text())
    settings = BoundaryHeadSettings(**config["boundary_head"])
    state = load_file(str(model_dir / "model.safetensors"))

    def sub(prefix):
        return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}

    d = settings.boundary_dim
    query = BoundaryQueryHead(HIDDEN_SIZE, d, HIDDEN_SIZE, settings.dropout)
    query.load_state_dict(
        sub("boundary_head.boundary_query_head."), strict=True
    )
    scorer = SparseBoundaryPairScorer(
        d, HIDDEN_SIZE, settings.pair_dim,
        use_inside_evidence=settings.use_inside_evidence,
        dropout=settings.dropout,
        enable_span_content=settings.enable_span_content,
        content_dim=settings.content_dim,
        content_soft_max_pool=settings.content_soft_max_pool,
        enable_rotary_endpoints=settings.enable_rotary_endpoints,
        rotary_base=settings.rotary_base,
        query_conditioned_inside_weight=settings.query_conditioned_inside_weight,
        endpoint_difference_features=settings.endpoint_difference_features,
        reranker_endpoint_compat=settings.reranker_endpoint_compat,
        multihead_pair_compat_heads=settings.multihead_pair_compat_heads,
        content_hidden_size=HIDDEN_SIZE,
    )
    scorer.load_state_dict(sub("boundary_head.pair_scorer."), strict=True)

    class Mlp:
        """`create_mlp(hidden, [2*hidden], 1, ..., activation="relu",
        add_layer_norm=False)` — a Linear, ReLU, dropout, Linear."""

        def __init__(self, weights, biases):
            self.w0, self.b0, self.w3, self.b3 = weights, biases, None, None
            del biases

        def __call__(self, rows):  # pragma: no cover - replaced below
            raise NotImplementedError

    classifier_w0 = state["classifier.0.weight"]
    classifier_b0 = state["classifier.0.bias"]
    classifier_w3 = state["classifier.3.weight"]
    classifier_b3 = state["classifier.3.bias"]
    del Mlp
    return {
        "settings": settings,
        "query_head": query,
        "pair_scorer": scorer,
        "classifier": (classifier_w0, classifier_b0, classifier_w3, classifier_b3),
        "null_projection": state["boundary_head.null_projection.weight"],
        "null_projection_bias": state["boundary_head.null_projection.bias"],
        "count_head": state["boundary_head.count_head.weight"],
        "count_head_bias": state["boundary_head.count_head.bias"],
    }


def classifier_logits(states, weights):
    w0, b0, w3, b3 = weights
    hidden = states @ w0.T + b0
    hidden = hidden.clamp_min(0.0)
    return (hidden @ w3.T + b3).squeeze(-1)


def run_case(processor, encoder, modules, text, schema, cls_config):
    settings = modules["settings"]
    normalized = processor._normalize_text(text)
    words = [token for token, _, _ in processor.word_splitter(normalized, lower=True)]
    batch = processor._collate_batch(
        [(text, copy.deepcopy(schema))], max_len=None,
        error_policy="fallback", build_targets=False,
    )
    with torch.inference_mode():
        hidden = encoder(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask
        ).last_hidden_state
    width = hidden.shape[-1]

    def gather(indices, mask):
        safe = indices.clamp(0, hidden.shape[1] - 1)
        states = hidden.gather(1, safe.unsqueeze(-1).expand(-1, -1, width))
        return states * mask.unsqueeze(-1).to(states.dtype)

    query_states = gather(batch.query_marker_indices, batch.query_marker_mask)
    cls_states = gather(batch.cls_marker_indices, batch.cls_marker_mask)

    temperature = settings.classification_temperature
    with torch.inference_mode():
        logits = classifier_logits(cls_states[0], modules["classifier"]) / temperature
        is_multi = bool(cls_config.get("multi_label", False))
        probs = torch.sigmoid(logits) if is_multi else torch.softmax(logits, dim=-1)
        null_logits = (query_states[0] @ modules["null_projection"].T
                       + modules["null_projection_bias"]).squeeze(-1)
        count_log_rates = (query_states[0] @ modules["count_head"].T
                           + modules["count_head_bias"]).squeeze(-1)

    labels = list(cls_config["labels"])
    if is_multi:
        threshold = cls_config.get("cls_threshold", 0.5)
        chosen = [
            labels[j] for j in range(len(labels)) if float(probs[j]) >= threshold
        ]
        if not chosen:
            chosen = [labels[int(torch.argmax(probs))]]
    else:
        chosen = [labels[int(torch.argmax(probs))]]

    return {
        "text": text,
        "normalized_text": normalized,
        "schema": schema,
        "input_ids": batch.input_ids[0].tolist(),
        "text_words": words,
        "query_marker_indices": batch.query_marker_indices[0].tolist(),
        "query_names": _field_names(batch.schema_tokens_list[0], ("[E]", "[R]")),
        "cls_marker_indices": batch.cls_marker_indices[0].tolist(),
        "cls_labels": labels,
        "cls_logits": logits.tolist(),
        "cls_probabilities": probs.tolist(),
        "cls_chosen": chosen,
        "cls_activation": "sigmoid" if is_multi else "softmax",
        "null_logits": null_logits.tolist(),
        "count_log_rates": count_log_rates.tolist(),
        "abstention_threshold": settings.abstention_threshold,
    }


def _field_names(schema_tokens, markers) -> list:
    names = []
    for tokens in schema_tokens:
        for index in range(len(tokens) - 1):
            if tokens[index] in markers:
                names.append(tokens[index + 1])
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=fixture_dir() / "classification-head-golden.json"
    )
    args = parser.parse_args()

    processor = SchemaTransformer(
        "models/deberta-v3-base", token_pooling="first", word_splitter=None
    )
    AutoTokenizer.from_pretrained(str(ENCODER_DIR))
    encoder = AutoModel.from_pretrained(str(ENCODER_DIR))
    load_checkpoint_encoder(encoder)
    encoder.eval()
    modules = reference_heads(REPO_ROOT / "models" / "gliner2.5-base-v1")

    cases = [run_case(processor, encoder, modules, text, schema, cfg)
             for text, schema, cfg in CASES]
    dump_json(
        args.out,
        {
            "classification_temperature": modules["settings"].classification_temperature,
            "abstention_threshold": modules["settings"].abstention_threshold,
            "cases": cases,
        },
    )


if __name__ == "__main__":
    main()
