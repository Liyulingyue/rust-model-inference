"""Generate a golden for the BoundaryEncoder forward.

Mirrors ``tests/gliner2_5_base_v1_forward.rs``'s smoke input: a single
sample of 6 valid tokens + 2 padding tokens. Loads
``models/gliner2.5-base-v1/model.safetensors`` directly, instantiates the
reference ``BoundaryEncoder`` from ``target/gliner2-oracle``, runs it on a
deterministic ``text_states`` input, and writes the resulting
``(states, mask)`` pair to
``tests/fixtures/gliner2.5-base-v1/boundary-encoder-golden.json``.

The Rust parity test replays the same ``text_states`` through the Rust
``BoundaryEncoder::forward`` and compares ``states`` / ``mask`` bits
against this fixture.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))

from gliner2.models.boundary.encoding import BoundaryEncoder  # noqa: E402

# Base-v1 contract: DeBERTa-v3-base encoder, boundary_dim=128,
# refinement_layers=1, ffn_multiplier=2.0, attention_layers=2,
# attention_heads=4, attention_window=128.
HIDDEN_SIZE = 768
BOUNDARY_DIM = 128
REFINEMENT_LAYERS = 1
FFN_MULTIPLIER = 2.0
ATTENTION_LAYERS = 2
ATTENTION_HEADS = 4
ATTENTION_WINDOW = 128


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "models" / "gliner2.5-base-v1")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "gliner2.5-base-v1" / "boundary-encoder-golden.json",
    )
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()

    state_dict = load_file(str(args.model_dir / "model.safetensors"))
    prefix = "boundary_head.boundary_encoder"
    renames = {
        f"{prefix}.bos_state": "bos_state",
        f"{prefix}.eos_state": "eos_state",
        f"{prefix}.left_projection.weight": "left_projection.weight",
        f"{prefix}.left_projection.bias": "left_projection.bias",
        f"{prefix}.right_projection.weight": "right_projection.weight",
        f"{prefix}.right_projection.bias": "right_projection.bias",
        f"{prefix}.output_projection.weight": "output_projection.weight",
        f"{prefix}.output_projection.bias": "output_projection.bias",
        f"{prefix}.layer_norm.weight": "layer_norm.weight",
        f"{prefix}.layer_norm.bias": "layer_norm.bias",
    }
    for i in range(ATTENTION_LAYERS):
        for j in ["norm.weight", "norm.bias", "qkv_projection.weight",
                  "qkv_projection.bias", "output_projection.weight",
                  "output_projection.bias"]:
            renames[f"{prefix}.attention_blocks.{i}.{j}"] = f"attention_blocks.{i}.{j}"
    for i in range(REFINEMENT_LAYERS):
        for j in ["norm.weight", "norm.bias", "input_projection.weight",
                  "input_projection.bias", "output_projection.weight",
                  "output_projection.bias"]:
            renames[f"{prefix}.refinement_blocks.{i}.{j}"] = f"refinement_blocks.{i}.{j}"

    renamed = {renames[k]: v for k, v in state_dict.items() if k in renames}

    encoder = BoundaryEncoder(
        hidden_size=HIDDEN_SIZE,
        boundary_dim=BOUNDARY_DIM,
        refinement_layers=REFINEMENT_LAYERS,
        ffn_multiplier=FFN_MULTIPLIER,
        attention_layers=ATTENTION_LAYERS,
        attention_heads=ATTENTION_HEADS,
        attention_window=ATTENTION_WINDOW,
    )
    # Reference loads its weights via the standard ``load_state_dict`` path.
    missing, unexpected = encoder.load_state_dict(renamed, strict=False)
    bad_missing = [name for name in missing if "num_batches_tracked" not in name]
    assert not bad_missing, f"missing weights: {bad_missing}"
    assert not unexpected, f"unexpected weights: {unexpected}"
    encoder.eval()

    torch.manual_seed(args.seed)
    seq_len = 8
    text_states = (torch.arange(seq_len * HIDDEN_SIZE, dtype=torch.float32) * 0.011).sin() - 1.0
    text_states = text_states.view(1, seq_len, HIDDEN_SIZE)
    text_mask = torch.tensor([[True] * 6 + [False] * 2])

    with torch.inference_mode():
        encoding = encoder(text_states, text_mask)

    fixture = {
        "config": {
            "hidden_size": HIDDEN_SIZE,
            "boundary_dim": BOUNDARY_DIM,
            "refinement_layers": REFINEMENT_LAYERS,
            "ffn_multiplier": FFN_MULTIPLIER,
            "attention_layers": ATTENTION_LAYERS,
            "attention_heads": ATTENTION_HEADS,
            "attention_window": ATTENTION_WINDOW,
            "seq_len": seq_len,
            "valid_tokens": 6,
        },
        "states": encoding.states.flatten().tolist(),
        "mask": encoding.mask.flatten().tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture) + "\n")
    print(f"{args.out} (states={fixture['states'].__len__()}, mask={fixture['mask'].__len__()})")


if __name__ == "__main__":
    main()