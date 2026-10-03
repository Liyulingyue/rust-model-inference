"""Byte-exact golden for `fastino/GLiNER2.5-Decide-1B`.

The reference loads this checkpoint through the same generic path as every other
variant — `SpanExtractorModel._load_encoder` calls
`AutoModel.from_pretrained(model_name, trust_remote_code=True)`, and
`model_name` is `jhu-clsp/ettin-enc-from-dec-1b`, a **ModernBERT**. So there is
no GLiNER-specific encoder code to mirror here: running transformers is running
the reference. That is the reason this script is thin.

What it *does* have to get right is the vocabulary. The tokenizer is a ByteLevel
BPE whose embedding table is 50378 rows while `model.vocab` is 50280, because 98
added tokens extend past the BPE vocabulary. A wrong `vocab_size` here either
fails `load_state_dict` or — if it happens to fit — shifts every id, so the row
count is read off the checkpoint and cross-checked against the tokenizer's
highest added-token id.

The encoder must be the checkpoint's own fine-tuned weights, not the base
model's: the released repo ships a randomly-initialised ModernBERT, and using it
leaves every logit near zero with no error anywhere.

Fixture:
    models/.venv/bin/python tools/oracle/gliner2/dump_ettin_golden.py \
        --model-dir models/GLiNER2.5-Decide-1B \
        --encoder-config target/gliner2-deberta-config \
        --out tests/fixtures/GLiNER2.5-Decide-1B/classify-golden.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "target" / "gliner2-oracle") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from gliner2.inference.schema import Schema  # noqa: E402
from gliner2.processor import SchemaTransformer  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402

# The four fixture cases, same as every other span golden in this family so the
# comparisons are across models rather than across fixtures.
CASES = [
    {
        "text": "My subscription renewed on April 15 for 5,400 after the service was already down. Can I get that charge refunded?",
        "tasks": [{"name": "intent", "labels": [
            {"name": "order_status"}, {"name": "refund_request"},
            {"name": "cancel_subscription"}, {"name": "other"}]}],
    },
    {
        "text": "The treaty was signed in Paris in 1992 and ratified the following year.",
        "tasks": [{"name": "qa", "labels": [{"name": "yes"}, {"name": "no"}]}],
    },
    {
        "text": "Battery dies before lunch, but the keyboard is excellent and the screen is fine.",
        "tasks": [{"name": "aspects", "labels": [
            {"name": "battery"}, {"name": "keyboard"}, {"name": "screen"},
            {"name": "camera"}, {"name": "price"}, {"name": "support"}],
            "multi_label": True, "cls_threshold": 0.4}],
    },
    {
        "text": "Guest in room 1408 says the AC has been out since yesterday and they want to move tonight or leave.",
        "tasks": [{"name": "intent", "labels": [
            {"name": "maintenance"}, {"name": "room_change"}, {"name": "checkout"},
            {"name": "billing"}, {"name": "complaint"}, {"name": "amenity_request"}]}],
    },
    {
        "text": "Please reset the card PIN. The last one I tried was declined twice.",
        "tasks": [{"name": "intent", "labels": [
            {"name": "card_pin_change"}, {"name": "card_lost"},
            {"name": "balance_inquiry"}]}],
    },
    {
        "text": "I finished it in two nights. The second half dragged but the ending landed.",
        "tasks": [{"name": "rating", "labels": [{"name": str(i)} for i in range(11)]}],
    },
]


def build_schema(case: dict, schema_builder) -> dict:
    schema = schema_builder()
    for task in case["tasks"]:
        labels = [label["name"] for label in task["labels"]]
        descriptions = {
            label["name"]: label["description"]
            for label in task["labels"] if "description" in label
        }
        schema.classification(
            task["name"],
            descriptions or labels,
            multi_label=task.get("multi_label", False),
            cls_threshold=task.get("cls_threshold", 0.5),
            **({"prompt": task["prompt"]} if "prompt" in task else {}),
        )
    return schema.schema


def main() -> None:
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument(
        "--model-dir", type=Path, default=REPO_ROOT / "models" / "GLiNER2.5-Decide-1B")
    arguments.add_argument(
        "--base-encoder", type=Path,
        default=REPO_ROOT / "models" / "ettin-enc-from-dec-1b",
        help="the base ModernBERT repo; supplies the encoder config and tokenizer")
    arguments.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "GLiNER2.5-Decide-1B" / "classify-golden.json")
    options = arguments.parse_args()

    model_dir = options.model_dir
    # The tokenizer comes from the checkpoint, not the base encoder: only the
    # checkpoint carries the eleven schema specials. The base encoder's own
    # tokenizer would tokenize the same text but could not resolve `[E]` &c.
    processor = SchemaTransformer(str(model_dir), token_pooling="first")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    # The encoder config has to come from the *base* encoder, not from
    # `model_dir`: the checkpoint's own `config.json` describes the GLiNER task
    # (`model_type: extractor`), which no AutoConfig recognises. Same reason the
    # DeBERTa span oracles take `--base-encoder`.
    config = AutoConfig.from_pretrained(
        str(REPO_ROOT / "models" / "ettin-enc-from-dec-1b"), trust_remote_code=True)
    # The checkpoint's own row count, cross-checked below. Taking it from
    # `tokenizer.json` alone is wrong: 98 added tokens live past the end of the
    # BPE vocabulary.
    state = load_file(str(model_dir / "model.safetensors"))
    rows = state["encoder.embeddings.tok_embeddings.weight"].shape[0]
    fast = json.loads((model_dir / "tokenizer.json").read_text())
    highest = max(int(entry["id"]) for entry in fast.get("added_tokens", []))
    if rows != highest + 1:
        raise ValueError(
            f"embedding table ({rows} rows) does not match the tokenizer's highest "
            f"added-token id + 1 ({highest + 1})"
        )
    if rows < int(config.vocab_size):
        raise ValueError("encoder config vocab_size exceeds the checkpoint's table")
    config.vocab_size = rows
    encoder = AutoModel.from_config(config)
    prefix = "encoder."
    missing, unexpected = encoder.load_state_dict(
        {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)},
        strict=False,
    )
    assert not unexpected, unexpected
    assert [name for name in missing if "position_ids" not in name] == [], missing
    encoder.eval()

    hidden = config.hidden_size
    classifier = torch.nn.Sequential(
        torch.nn.Linear(hidden, hidden * 2), torch.nn.ReLU(), torch.nn.Linear(hidden * 2, 1)
    )
    classifier.load_state_dict({
        "0.weight": state["classifier.0.weight"], "0.bias": state["classifier.0.bias"],
        "2.weight": state["classifier.2.weight"], "2.bias": state["classifier.2.bias"],
    })
    classifier.eval()

    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SchemaTransformer.SPECIAL_TOKENS})
    print(f"added {added} special tokens", file=sys.stderr)

    rows_out = []
    with torch.inference_mode():
        for case in CASES:
            batch = processor.collate_fn_inference(
                [(case["text"], build_schema(case, Schema))])
            out = encoder(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask
            ).last_hidden_state
            _, schema_embs = processor.extract_embeddings_from_batch(
                out, batch.input_ids, batch)
            logits = []
            for t_idx in range(len(batch.task_types[0])):
                label_rows = schema_embs[0][t_idx][1:]
                assert len(label_rows) == len(case["tasks"][t_idx]["labels"])
                produced = classifier(torch.stack(list(label_rows))).squeeze(-1)
                logits.append([float(value) for value in produced])
            rows_out.append({
                "text": case["text"],
                "tasks": case["tasks"],
                "input_ids": [int(value) for value in batch.input_ids[0]],
                "marker_positions": [
                    [int(value) for value in row] for row in batch.schema_special_indices[0]],
                "logits": logits,
            })

    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(json.dumps({"cases": rows_out}, indent=2) + "\n")
    print(f"wrote {options.out} ({len(rows_out)} cases)", file=sys.stderr)


if __name__ == "__main__":
    main()
