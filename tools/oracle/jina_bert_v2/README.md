# Jina v2 scalar architecture parity

Scope: `jina-embeddings-v2-base-en-q8_0.gguf`, architecture `jina-bert-v2`,
against llama.cpp `b96806d96061049a5b574269b049bf6241d63d46` on macOS ARM64.
The encoder is bidirectional and has no KV cache, logits or greedy generation.
Jina v3/v5, other weights and accelerated paths are outside this result.

## Fixed model contract

- Source: `ggml-org/jina-embeddings-v2-base-en-Q8_0-GGUF` on ModelScope.
- SHA256: `c4743c5cfcf2b1c6fe0bb548b13d8ecf828cfc9a8d2efd48df926f746e2dcd22`.
- Size: 146,218,816 bytes; 196 tensors, Q8_0 matrices and F32 vectors.
- 12 layers; hidden 768; 12 heads × 64; FFN 3072; context 8192; vocabulary 30528.
- WordPiece (`tokenizer.ggml.model=bert`, `pre=jina-v2-en`), CLS=101,
  SEP=102, UNK=100, PAD=0, MASK=103; no chat template.
- LayerNorm epsilon `1e-12`; noncausal attention; ALiBi maximum bias 8;
  mean pooling including special tokens, followed by L2 normalization.

| Tensor | GGUF dimensions | Type |
|---|---|---|
| `token_embd.weight` | `[768,30528]` | Q8_0 |
| `token_types.weight` | `[768,2]` (row 0) | F32 |
| `token_embd_norm.{weight,bias}` | `[768]` | F32 |
| `blk.{0..11}.attn_{q,k,v,output}.weight` | `[768,768]` | Q8_0 |
| `blk.{0..11}.attn_{q,k,v,output}.bias` | `[768]` | F32 |
| `blk.{0..11}.{attn_output_norm,layer_output_norm}.{weight,bias}` | `[768]` | F32 |
| `blk.{0..11}.ffn_{gate,up}.weight` | `[768,3072]` | Q8_0 |
| `blk.{0..11}.ffn_down.weight` | `[3072,768]` | Q8_0 |
| `blk.{0..11}.ffn_down.bias` | `[768]` | F32 |

## Reproduce

`build.sh` checks the reference commit, clones it into a new work directory,
and adds checkpoint names/capture only. It does not alter reference arithmetic
or the original checkout. Its flags disable NEON, automatic vectorization,
FMA contraction, weight repacking, BLAS, Accelerate, Metal and OpenMP.
Use the installed Clang/CMake toolchain on macOS ARM64.

```sh
LLAMA_CPP=/path/to/llama.cpp # checkout at the pinned commit above
JINA_MODEL=models/jina-embeddings-v2-base-en-Q8_0-GGUF/jina-embeddings-v2-base-en-q8_0.gguf
sh tools/oracle/jina_bert_v2/build.sh "$LLAMA_CPP" target/jina-bert-v2-oracle
RUSTFLAGS='-C no-vectorize-loops -C no-vectorize-slp' \
  cargo build --profile release-fast --features parity-trace --bin rust-model-inference
python3 tools/oracle/jina_bert_v2/verify.py \
  --model "$JINA_MODEL" \
  --oracle target/jina-bert-v2-oracle/build/bin/llama-eval-callback \
  --rust target/release-fast/rust-model-inference \
  --output target/jina-bert-v2-verification
```

The output directory must be new. The verifier checks the model hash, runs both
real CLIs with one thread (`RMI_SCALAR=1` for Rust), and retains logs, JSONL
manifests, complete little-endian F32 arrays and `verification.json`.
It compares token IDs and each checkpoint's canonical order, layer, shape,
occurrence and **every raw F32 bit**, stopping at the first difference.
Canonical order removes only backend scheduling differences.

## Verified coverage

Six fixtures cover English punctuation, whitespace-only input, accents/NFD,
CJK, emoji, `[MASK]`/`[SEP]`, and an 82-token sequence. Each has 76 tensor
checkpoints: input embedding and normalization, Q/K/V, attention projection,
FFN input and output for all 12 layers, mean pooling and final normalization.
Total: **456 checkpoints / 7,056,384 F32 values**, plus identical token IDs.
The recorded run is in [verification.json](verification.json), including model
and binary hashes, the exact prompts and token IDs, and per-fixture counts.
The CLI intentionally rejects a literal empty prompt; whitespace exercises the
two-special-token encoder input, and the tokenizer unit tests cover empty text.

Confirmed fixes: punctuation/CJK splitting, accent normalization and special
token parsing; duplicated attention residual; LayerNorm rounding; attention
dot reductions; mean-pooling multiply/reduce order; L2's F32 products; and
the scalar GELU table's accidental FMA. Ordinary accelerated GELU keeps its
existing branch. No tolerance or cosine criterion is used for parity.

```sh
RMI_SCALAR=1 RUSTFLAGS='-C no-vectorize-loops -C no-vectorize-slp' \
  cargo test --profile release-fast --features parity-trace --lib gelu_ggml_f16_matches_pinned_oracle_bits
RMI_JINA_V2_BASE_EN_MODEL="$JINA_MODEL" \
  RUSTFLAGS='-C no-vectorize-loops -C no-vectorize-slp' \
  cargo test --profile release-fast --features parity-trace --test jina_v2_base_en
```

Regression results after integrating main `d8c41b3` (2026-10-01): tokenizer
26 passed / 2 ignored; BERT family 12 passed; GLiNER 23 passed; scalar GELU
bit test and all 4 real-model Jina integration tests passed. The ordinary
CLI returns a 768-dimensional embedding with tracing disabled.
Repository-wide `cargo fmt --all -- --check` and `git diff --check` pass.

Before integrating main, the full test command stopped at library tests:
1,015 passed, 30 failed and 71 ignored. A rebuilt clean `60882a0` baseline had
the **same 30 failing tests** (1,014 passed, 71 ignored). The full suite was
not repeated after integrating main; the relevant tests above were rerun.
