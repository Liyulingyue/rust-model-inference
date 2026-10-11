"""
Whistle (Cactus-Compute) HF safetensors → GGUF converter (placeholder).

Outputs F32 GGUF that mirrors the safetensors layout. The converter is
intentionally minimal — it does NOT (yet) replace `cactus-quants` with
GGML Q4_K/Q8_0, and it does NOT lift placeholder ops (ZCRMSNorm,
mhc_phi, taps). It is enough to drive the in-tree parity tests in
`tests/whistle_layer_test.rs` and a future `--asr` CLI mode that uses
the safetensors reader.

Layout convention:
- cact-quants weights are dropped (the safetensors ship F32).
- 2D kernels stay `[num_layers, in, out]` (Whistle's actual layout).
- Per-layer scales stay `[num_layers, hidden]`.
- Tensor names match the safetensors keys verbatim so the Rust
  TensorSource loaders can find them.

Usage:
  python tools/converter/whistle/convert_whistle.py \
      --safetensors checkpoints/whistle.safetensors \
      --config     config.json \
      --out        /path/to/whistle.f32.gguf
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

# Re-use the shared GGUFRs writer.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.gguf import GgufWriter, open_safetensors, GGML_F32


def load_config(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--safetensors", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_config(args.config)
    st = open_safetensors(args.safetensors)
    tensor_names = [k for k in st.header.keys() if k != "__metadata__"]

    # GGUF metadata mirroring the source config so the Rust side can
    # detect this is a whistle checkpoint.
    meta = {
        "general.architecture": "whistle",
        "whistle.engine": cfg.get("engine", "Cactus-Compute/needle3"),
        "whistle.model_type": cfg.get("model_type", "whistle"),
        "whistle.languages": ",".join(cfg.get("languages", [])),
        "whistle.sample_rate": str(cfg.get("sample_rate", 16000)),
        "whistle.max_audio_seconds": str(cfg.get("max_audio_seconds", 30)),
        "whistle.encoder_frame_ms": str(cfg.get("encoder_frame_ms", 80)),
        "whistle.hidden": "512",
        "whistle.num_layers": "8",
        "whistle.num_heads": "8",
        "whistle.num_kv_heads": "2",
        "whistle.text_vocab": "8199",
    }

    writer = GgufWriter(args.out)
    for k, v in meta.items():
        writer.add_meta(k, v)
    for name in tensor_names:
        t = st.get(name)
        assert t.dtype == "F32", f"unexpected dtype {t.dtype} for {name}"
        # Alias token_embd.weight alongside embedding/embedding.
        out_name = "token_embd.weight" if name == "embedding/embedding" else name
        # GGUF rejects zero-rank tensors; promote `[]` → `[1]`.
        shape = t.shape if t.shape else (1,)
        writer.add_tensor(out_name, GGML_F32, shape, t.raw)
    writer.write()


if __name__ == "__main__":
    main()