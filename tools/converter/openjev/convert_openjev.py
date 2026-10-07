"""Convert `AlexWortega/openjev` (Qwen3.5 + 3-class NLI head) into a GGUF
the engine can load via the existing `qwen3` cross-encoder / classification
head plumbing (`cls.output.weight` + `pooling_type = 3`).

Inputs:
* base Qwen3.5 GGUF (e.g. `unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-BF16.gguf`).
  The converter copies every backbone tensor verbatim — the only model-level
  change is the addition of `cls.output.weight` and the classification
  metadata.
* HF safetensors checkpoint for `AlexWortega/openjev/qwen3.5-0.8b-nli-v2s-long`
  (or `…/qwen3.5-4b-nli-v5`, `…/qwen3.5-2b-nli-v5`). The classifier head
  is stored as `score.weight` (PyTorch `nn.Linear` semantics: out × in) of
  shape `[num_labels, hidden_size]`. We transpose to `[hidden_size,
  num_labels]` and rewrite as F32 `cls.output.weight`, the llama.cpp
  rerank-packer naming convention that `qwen3` trunk already consumes.
* `config.json` (HF) — for `label2id`, `id2label`, `nli_template`,
  `problem_type`, `pad_token_id`.

Output: a single GGUF (v3) with:
* the original `general.*` metadata + qwen35-specific tensor directory
* new tensor `cls.output.weight` of shape `[hidden_size, num_labels]`,
  dtype F32 (the head is `[3, 1024]` for the 0.8B variant — only 12 KiB,
  no quantisation savings)
* new tensor `cls.output.bias` of shape `[num_labels]`, dtype F32, all
  zeros (the HF head has no bias, so the GGUF must carry the same
  zero-bias contract the llama.cpp rerank packer relies on)
* `qwen35.classifier.*` metadata: `label2id`, `id2label`,
  `nli_template`, `problem_type`, `pad_token_id`

The 0.8B variant is the smallest (1.7 GiB HF + 1.5 GiB BF16 GGUF) and
fastest to verify, so the converter and docs focus on it; the 2B and 4B
variants only differ in `hidden_size` / `num_hidden_layers` /
`intermediate_size` and the head shape, all read from the HF config.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.converter.utils.gguf import (
    GGML_F32,
    GgufWriter,
    read_gguf_directory,
    read_gguf_tensor_bytes,
    open_safetensors,
)


def _transposed_f32(cls_weight_bytes: bytes, n_out: int, n_in: int) -> bytes:
    """Convert a HuggingFace `nn.Linear` weight tensor of shape
    `[n_out, n_in]` (BF16, stored row-major) into a llama.cpp
    `cls.output.weight` of shape `[n_in, n_out]` (F32, column-major).

    The transposed shape is what the existing qwen3 trunk already loads
    under `cls_score: Weight` (see `src/models/qwen3/trunk/weights.rs`).
    """
    if len(cls_weight_bytes) != n_out * n_in * 2:
        raise ValueError(
            f"score.weight payload {len(cls_weight_bytes)} != expected "
            f"{n_out}*{n_in}*2 BF16 bytes"
        )
    # Decode BF16 → F32 then transpose row-major [out, in] → col-major [in, out].
    decoded = bytearray(n_out * n_in * 4)
    for row in range(n_out):
        for col in range(n_in):
            i = (row * n_in + col) * 2
            bf16_bits = cls_weight_bytes[i] | (cls_weight_bytes[i + 1] << 8)
            # BF16 is the upper 16 bits of an IEEE-754 single.
            f32_bits = bf16_bits << 16
            j = (col * n_out + row) * 4
            decoded[j:j + 4] = f32_bits.to_bytes(4, "little")
    return bytes(decoded)


def _read_hf_config(checkpoint_dir: Path) -> dict:
    cfg_path = checkpoint_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing HF config.json at {cfg_path}")
    return json.loads(cfg_path.read_text())


def _read_hf_score_weight(safetensors_path: Path) -> bytes:
    """Return the raw bytes of the `score.weight` tensor from a HF
    safetensors file. The safetensors format is well-defined enough that
    we can read just this single tensor without scanning the whole file.
    """
    st = open_safetensors(safetensors_path)
    if "score.weight" not in st.header:
        raise ValueError(
            f"{safetensors_path}: missing 'score.weight' in safetensors header"
        )
    info = st.header["score.weight"]
    if info.get("dtype") != "BF16":
        raise ValueError(
            f"score.weight dtype expected BF16, got {info.get('dtype')!r}"
        )
    return st.get("score.weight").raw


def _shape_of(dims: tuple) -> tuple:
    """The GGUF writer expects tuple dims; some readers return lists."""
    return tuple(dims)


def convert(
    base_gguf: Path,
    hf_checkpoint: Path,
    output_gguf: Path,
) -> None:
    metadata, tensors = read_gguf_directory(base_gguf)
    print(
        f"[convert] base GGUF: {len(tensors)} tensors, "
        f"{len(metadata)} metadata entries, arch={metadata.get('general.architecture')!r}",
        file=sys.stderr,
    )
    arch = metadata.get("general.architecture")
    if arch != "qwen35":
        raise ValueError(
            f"base GGUF architecture must be qwen35, got {arch!r}; "
            "this converter only handles Qwen3.5-NLI variants"
        )

    hf_config = _read_hf_config(hf_checkpoint)
    text_config = hf_config.get("text_config", {})
    hidden_size = int(text_config["hidden_size"])
    id2label = {int(k): v for k, v in hf_config["id2label"].items()}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)
    nli_template = hf_config.get("nli_template")
    if not nli_template or "{premise}" not in nli_template or "{hypothesis}" not in nli_template:
        raise ValueError(
            f"config.nli_template {nli_template!r} must contain both "
            f"{{premise}} and {{hypothesis}} placeholders"
        )
    pad_token_id = int(text_config.get("pad_token_id", hf_config.get("pad_token_id", 0)))

    # Resolve which safetensors file holds the score.weight tensor.
    candidates = sorted(hf_checkpoint.glob("model*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"no model*.safetensors under {hf_checkpoint}")
    cls_bytes = None
    for shard in candidates:
        try:
            cls_bytes = _read_hf_score_weight(shard)
            break
        except (KeyError, ValueError):
            continue
    if cls_bytes is None:
        raise ValueError(
            f"score.weight not found in any safetensors shard under {hf_checkpoint}"
        )

    out = GgufWriter(output_gguf)

    # Copy metadata, replacing general.name / general.size_label / qwen35.* tuning
    # numbers that the llama.cpp packer would also override when rerank-packing.
    hidden_size_meta = int(metadata.get(f"{arch}.embedding.length", hidden_size))
    pooling_key = f"{arch}.pooling_type"
    for key, value in metadata.items():
        if key in {"general.name", "general.size_label", "general.quantized_by"}:
            continue
        if key.startswith(f"{arch}.classifier."):
            continue
        if key == pooling_key:
            # last-token pooling — the qwen3 rerank path uses 4 (Last) but
            # the existing qwen3/embedding.rs accepts only 1 / 3; use 3 to
            # stay within the supported set ("Last" semantically).
            out.add_meta(key, 3)
            continue
        if key == f"{arch}.embedding.length" and hidden_size_meta != hidden_size:
            out.add_meta(key, hidden_size)
            continue
        out.add_meta(key, value)
    # If the base GGUF had no pooling_type (it is optional), add the
    # last-token value now — the cross-encoder scorer requires it.
    if pooling_key not in metadata:
        out.add_meta(pooling_key, 3)

    # Set new general.* + qwen35.classifier.* metadata.
    out.add_meta("general.name", "openjev-NLI-Qwen3.5")
    out.add_meta("general.size_label", f"{(hidden_size * num_labels // 8) // 1000}k-NLI-{hidden_size}")
    # GGUF arrays carry scalars — use parallel arrays in the order matching
    # `id2label` (label index → string) and a parallel key→int map is
    # captured as `label_strings` + `label_for_class` series, one per
    # class index. The Rust loader reads them as parallel arrays.
    label_strings = [id2label[i] for i in range(num_labels)]
    label_indices = list(range(num_labels))
    out.add_meta(f"{arch}.classifier.label_strings", label_strings)
    out.add_meta(f"{arch}.classifier.label_indices", label_indices)
    out.add_meta(f"{arch}.classifier.nli_template", nli_template)
    out.add_meta(f"{arch}.classifier.problem_type", hf_config.get("problem_type", "single_label_classification"))
    out.add_meta(f"{arch}.classifier.pad_token_id", pad_token_id)

    # Copy every backbone tensor verbatim (same name, same dims, same qtype,
    # same raw bytes).
    n_copied = 0
    for name, (ggml_type, dims, offset) in tensors.items():
        if name == "cls.output.weight" or name == "cls.output.bias":
            raise ValueError(
                f"base GGUF already carries {name}; this converter layers "
                "the head on top of an unrouted base"
            )
        raw = read_gguf_tensor_bytes(base_gguf, name)
        out.add_tensor(name, ggml_type, _shape_of(dims), raw)
        n_copied += 1
    print(
        f"[convert] copied {n_copied} backbone tensors verbatim",
        file=sys.stderr,
    )

    # Append the classification head. Openjev stores it as
    # `score.weight: [num_labels, hidden_size]` in BF16; the engine wants
    # `cls.output.weight: [hidden_size, num_labels]` in F32.
    cls_out = _transposed_f32(cls_bytes, n_out=num_labels, n_in=hidden_size)
    out.add_tensor("cls.output.weight", GGML_F32, (hidden_size, num_labels), cls_out)
    out.add_tensor("cls.output.bias", GGML_F32, (num_labels,), b"\x00" * (num_labels * 4))

    out.write()
    print(
        f"[convert] wrote {output_gguf} "
        f"({output_gguf.stat().st_size // 1024 // 1024} MiB)",
        file=sys.stderr,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert Qwen3.5 + 3-class NLI head (openjev) into GGUF",
    )
    ap.add_argument(
        "--base-gguf",
        required=True,
        type=Path,
        help="Qwen3.5 base GGUF (e.g. unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-BF16.gguf)",
    )
    ap.add_argument(
        "--hf-checkpoint",
        required=True,
        type=Path,
        help="HF checkpoint dir (contains config.json + model*.safetensors)",
    )
    ap.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output GGUF path",
    )
    args = ap.parse_args()
    convert(args.base_gguf, args.hf_checkpoint, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())