"""Golden vectors for GLiNER2.5-Decide: input ids, marker rows and logits.

Runs the reference stack (GLiNER2's `SchemaTransformer` + HF DeBERTa-v3 +
the checkpoint's classifier head) and writes
`tests/fixtures/gliner2-decide/classify-golden.json`, which the Rust parity
test replays through the GGUF.

Needs `transformers==4.48.1`, `sentencepiece` and `torch`, so it lives outside
the repo; regenerate with:

    models/.venv/bin/python /tmp/gliner_ref/dump_golden.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]

CASES = [
    {
        "text": "My subscription renewed on April 15 for 5,400 after the service was already down. "
                "Can I get that charge refunded?",
        "tasks": [
            {
                "name": "intent",
                "labels": [
                    {"name": "order_status"},
                    {"name": "refund_request"},
                    {"name": "cancel_subscription"},
                    {"name": "other"},
                ],
            }
        ],
    },
    {
        "text": "The treaty was signed in Paris in 1992. It entered into force the following year, "
                "after the last signatory ratified it.",
        "tasks": [
            {
                "name": "answer",
                "prompt": "Did the treaty enter into force in 1992?",
                "labels": [{"name": "yes"}, {"name": "no"}],
            }
        ],
    },
    {
        "text": "Battery dies before lunch, but the keyboard and the screen are the best I have used "
                "on a laptop.",
        "tasks": [
            {
                "name": "aspects",
                "multi_label": True,
                "cls_threshold": 0.4,
                "labels": [
                    {"name": "battery"},
                    {"name": "keyboard"},
                    {"name": "screen"},
                    {"name": "camera"},
                    {"name": "price"},
                    {"name": "support"},
                ],
            }
        ],
    },
    {
        "text": "Guest in room 1408 says the AC has been out since yesterday and they want to move "
                "tonight or leave. They also asked for the incidentals hold to be released.",
        "tasks": [
            {
                "name": "intent",
                "labels": [
                    {"name": "maintenance"},
                    {"name": "room_change"},
                    {"name": "checkout"},
                    {"name": "billing"},
                    {"name": "complaint"},
                    {"name": "amenity_request"},
                ],
            },
            {
                "name": "priority",
                "labels": [
                    {"name": "low"},
                    {"name": "normal"},
                    {"name": "high"},
                    {"name": "urgent"},
                ],
            },
            {"name": "needs_human", "labels": [{"name": "yes"}, {"name": "no"}]},
            {
                "name": "topics",
                "multi_label": True,
                "cls_threshold": 0.4,
                "labels": [
                    {"name": "hvac"},
                    {"name": "billing"},
                    {"name": "housekeeping"},
                    {"name": "noise"},
                    {"name": "safety"},
                ],
            },
        ],
    },
    {
        "text": "Please reset the card PIN. The new one never arrived and the old one is locked "
                "after three tries.",
        "tasks": [
            {
                "name": "intent",
                "labels": [
                    {"name": "card_pin_change", "description": "The customer wants a new PIN or the current PIN replaced"},
                    {"name": "card_lost", "description": "The physical card is missing"},
                    {"name": "balance_inquiry", "description": "The customer wants the current balance"},
                ],
            }
        ],
    },
    {
        "text": "I finished it in two nights. The ending is earned, the middle drags, and I would "
                "still hand it to a friend.",
        "tasks": [
            {
                "name": "rating",
                "labels": [{"name": str(value)} for value in range(11)],
            }
        ],
    },
]


def build_schema(case, schema_builder):
    schema = schema_builder()
    for task in case["tasks"]:
        labels = [label["name"] for label in task["labels"]]
        descriptions = {
            label["name"]: label["description"] for label in task["labels"] if "description" in label
        }
        schema.classification(
            task["name"],
            descriptions or labels,
            multi_label=task.get("multi_label", False),
            cls_threshold=task.get("cls_threshold", 0.5),
            **({"prompt": task["prompt"]} if "prompt" in task else {}),
        )
    return schema.schema


def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument(
        "--gliner2",
        type=Path,
        default=Path(os.environ.get("GLINER2_SRC", "target/gliner2-oracle")),
        help="upstream GLiNER2 checkout (the SchemaTransformer reference)",
    )
    arguments.add_argument(
        "--encoder-config",
        type=Path,
        default=REPO_ROOT / "target/gliner2-deberta-config",
        help="directory holding the base deberta-v3-large config.json",
    )
    arguments.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "models/GLiNER2.5-Decide",
    )
    arguments.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "tests/fixtures/gliner2-decide/classify-golden.json",
    )
    options = arguments.parse_args()

    sys.path.insert(0, str(options.gliner2.resolve()))
    # Imported after sys.path so the checkout wins over anything installed.
    from gliner2.inference.schema import Schema  # noqa: E402
    from gliner2.processor import SchemaTransformer  # noqa: E402
    from safetensors.torch import load_file  # noqa: E402
    from transformers import AutoConfig, DebertaV2Model  # noqa: E402

    model_dir = options.model_dir
    processor = SchemaTransformer(str(model_dir), token_pooling="first")
    config = AutoConfig.from_pretrained(str(options.encoder_config))
    config.vocab_size = 128011
    encoder = DebertaV2Model(config)
    state = load_file(str(model_dir / "model.safetensors"))
    missing, unexpected = encoder.load_state_dict(
        {key[len("encoder."):]: value for key, value in state.items() if key.startswith("encoder.")},
        strict=False,
    )
    assert not unexpected, unexpected
    assert [name for name in missing if "position_ids" not in name] == [], missing
    encoder.eval()

    classifier = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048), torch.nn.ReLU(), torch.nn.Linear(2048, 1)
    )
    classifier.load_state_dict(
        {
            "0.weight": state["classifier.0.weight"],
            "0.bias": state["classifier.0.bias"],
            "2.weight": state["classifier.2.weight"],
            "2.bias": state["classifier.2.bias"],
        }
    )
    classifier.eval()

    rows = []
    with torch.inference_mode():
        for case in CASES:
            batch = processor.collate_fn_inference([(case["text"], build_schema(case, Schema))])
            hidden = encoder(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask
            ).last_hidden_state
            _, schema_embs = processor.extract_embeddings_from_batch(
                hidden, batch.input_ids, batch
            )
            logits = []
            for t_idx in range(len(batch.task_types[0])):
                label_rows = schema_embs[0][t_idx][1:]
                assert len(label_rows) == len(case["tasks"][t_idx]["labels"])
                out = classifier(torch.stack(list(label_rows))).squeeze(-1)
                logits.append([float(value) for value in out])
            rows.append(
                {
                    "text": case["text"],
                    "tasks": case["tasks"],
                    "input_ids": [int(value) for value in batch.input_ids[0]],
                    "marker_positions": [
                        [int(value) for value in row]
                        for row in batch.schema_special_indices[0]
                    ],
                    "logits": logits,
                }
            )

    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(json.dumps({"cases": rows}, indent=1, ensure_ascii=False) + "\n")
    print(f"{options.out} ({len(rows)} cases)")


if __name__ == "__main__":
    main()
