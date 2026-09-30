---
license: apache-2.0
base_model:
  - Contrastive-LM/CLM-v0.1-8B
  - Qwen/Qwen3-8B
pipeline_tag: text-ranking
language:
  - en
  - zh
tags:
  - gguf
  - clm
  - qwen3
  - contrastive-learning
  - candidate-scoring
---

# CLM-v0.1-8B projection heads in GGUF

This repository contains the **state and action projection heads only** from
[Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B).
CLM scores candidate actions against a state; it does not generate text. A separate,
frozen **Qwen3-8B** encoder is required to turn text into embeddings.

| File | Contents | Size | SHA-256 |
| --- | --- | ---: | --- |
| `clm-v0.1-8B-heads-f32.gguf` | 17 F32 tensors, including both projection heads | 75,552,224 bytes | `d5bbef8f4c9edf0b5cfb38b4a9326a1ac776cc9b479345a56b2efc37de3c39be` |

The source checkpoint is `CLM_v0.1-8B.pt` (SHA-256
`b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5`).
The [converter](https://github.com/Liyulingyue/rust-model-inference/blob/main/tools/converter/clm/convert_clm.py)
preserves the F32 weight and bias bytes. It stores the reference scoring scale
`min(exp(logit_scale), 100) = 100` in the GGUF. The GGUF does **not** contain
the Qwen3-8B encoder, tokenizer, or original `.pt` file.

## Download and run

The verified encoder file is
[`unsloth/Qwen3-8B-GGUF` / `Qwen3-8B-BF16.gguf`](https://huggingface.co/unsloth/Qwen3-8B-GGUF),
SHA-256 `5e416a2020fe63e76ea13c8979be35fc6070aaf3578f7876400c55c2f5c3eb30`.
Use [rust-model-inference](https://github.com/Liyulingyue/rust-model-inference)
at or after the merged [CLM parity change #128](https://github.com/Liyulingyue/rust-model-inference/pull/128),
then download both files:

```bash
git clone https://github.com/Liyulingyue/rust-model-inference.git
cd rust-model-inference

hf download EvoAwaken-Workshop/CLM-v0.1-8B-gguf \
  clm-v0.1-8B-heads-f32.gguf --local-dir models/CLM-v0.1-8B
hf download unsloth/Qwen3-8B-GGUF Qwen3-8B-BF16.gguf \
  --revision a6adef130ffb23ddaf1a62fec9dced968c9bc482 \
  --local-dir models/Qwen3-8B-GGUF
```

CLM uses the existing `--jev` candidate-scoring interface:

```bash
cargo build --profile release-fast --bin rust-model-inference
./target/release-fast/rust-model-inference \
  --model models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf \
  --jev --clm-head models/CLM-v0.1-8B/clm-v0.1-8B-heads-f32.gguf \
  --jev-context 'Customer: my invoice was charged twice.' \
  --jev-question 'Which team should handle this?' \
  --jev-option 'Charges, invoices, refunds' \
  --jev-option 'Bugs and outages' \
  --threads 1
```

The state input is `context + "\n\n" + question`; each candidate is encoded as
written. The encoder uses last-token pooling and L2 normalization before the
two heads. Scores and probabilities are relative to the supplied candidates.

## Conversion and verification

The [official CLM implementation](https://github.com/Contrastive-LM/CLM) defines
the two heads and scoring rule. The GGUF uses `general.architecture=clm` and is
loaded alongside a `qwen3` encoder. The source checkpoint and converter pass a
tensor round-trip check.

On 2026-09-30, two real-text scoring cases matched a fixed scalar llama.cpp
encoder and an independent scalar CLM-head implementation: token IDs, **32,768
encoder F32 values**, and **34,818 projection-head F32 values** were bitwise
identical. Final score bit patterns were `0x415a1e78` and `0x41b29eb4`.
[Hashes, commands, and the verification script](https://github.com/Liyulingyue/rust-model-inference/blob/main/tools/oracle/clm/README.md)
record the exact scope.

This bitwise result applies to the specified BF16 encoder and F32 heads in
scalar mode. The official vLLM service, NEON/FMA/BLAS/Accelerate, the default
accelerated path, and other quantizations have not been compared bitwise.

## Attribution and license

- Original model: [Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)
- Base encoder: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- Official implementation: [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM)
- Model description: [Contrastive Language Models](https://contrastive-lm.notion.site)

The CLM heads and Qwen3-8B base model are released under Apache 2.0. See the
`LICENSE` file in this repository and the original model pages for details.
