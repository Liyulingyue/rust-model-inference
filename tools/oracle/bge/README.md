# BGE verification

BGE conversion and inference cover 20 text repositories and 8 visual repositories
from ModelScope BAAI. Visualized BGE publishes two separate checkpoints.
[Conversion and CLI/HTTP usage](../../converter/README.md#bge-转换与推理)
list the complete scope. [verification.json](verification.json) records the
fixed references, model hashes, real input modes and test coverage.

## Exact operator checks

At the user's request, **all lossy operator alignment is skipped**, including
F16 GELU lookup tables even when the model weights are F32. Quantization,
F16/BF16 activation rounding, F16 KV, approximate SIMD, and FMA are excluded.
No BERT, Gemma2, visual, quantized or reduced-precision whole-model output
parity is claimed. Existing inference implementations remain usable.

`scalar_ops.cpp` calls the actual GGML F32 graph kernels; it does not reimplement
reference formulas. `examples/bge_scalar_ops.rs` calls the repository's actual
operators on the same binary fixtures. `compare_ops.py` compares raw `u32` bits
for matmul, affine LayerNorm/RMSNorm, softmax, exact SiLU, add, and multiply.
The 44 deterministic fixtures cover widths 1/3/16/32/64/256/384/768/1024/2048/4096
and ordinary, constant, small and cancellation inputs: **208,828 F32 values
match exactly**. This is operator evidence, not complete graph parity or
retrieval-quality evaluation.

The reference is a clean `llama.cpp` checkout at
`11fe02151f79c41d0d4af7da708755d73b9c0da6`. On macOS ARM64 it was built with:

```bash
bge_scalar_flags='-U__ARM_NEON -U__ARM_NEON__ -U__ARM_FEATURE_FMA -ffp-contract=off -fno-vectorize -fno-slp-vectorize'
cmake -S /tmp/bge-llama-scalar-11fe -B /tmp/bge-llama-scalar-11fe-build \
  -DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_NATIVE=OFF \
  -DCMAKE_C_FLAGS="$bge_scalar_flags" -DCMAKE_CXX_FLAGS="$bge_scalar_flags"
cmake --build /tmp/bge-llama-scalar-11fe-build --target llama-debug -j4

c++ -std=c++17 -ffp-contract=off -fno-vectorize -fno-slp-vectorize \
  -I/tmp/bge-llama-scalar-11fe/ggml/include \
  tools/oracle/bge/scalar_ops.cpp \
  /tmp/bge-llama-scalar-11fe-build/ggml/src/libggml-cpu.a \
  /tmp/bge-llama-scalar-11fe-build/ggml/src/libggml-base.a \
  -pthread -o target/bge-scalar/ggml-scalar-ops

RUSTFLAGS='-C no-vectorize-loops -C no-vectorize-slp' \
  cargo build --profile release-fast --features parity-trace \
  --example bge_scalar_ops --bin rust-model-inference --target-dir target/bge-scalar
.venv/bin/python tools/oracle/bge/compare_ops.py \
  --oracle target/bge-scalar/ggml-scalar-ops \
  --rust target/bge-scalar/release-fast/examples/bge_scalar_ops \
  --work-dir target/bge-ops-check-new
```

The work directory must be fresh. Each fixture hash is recorded in its
`report.json`; the verified report is included in `verification.json`.
Do not use `-C target-cpu=generic` on Apple ARM: the existing ring dependency
requires the platform's static CPU features.

## Tokenizer checks and optional exact decoder outputs

`compare.py` compares token IDs by default and explicitly skips numeric output
comparison. It uses one CPU thread, `RMI_SCALAR=1`, F32 KV and the same converted
GGUF in both runtimes. Four real fixtures (small-en-v1.5, small-zh,
small-zh-v1.5 and M3) have been checked on these four texts:

```text
hello
Hello, 世界! 123
What is the capital of China?
中国的首都是北京。
```

```bash
.venv/bin/python tools/oracle/bge/compare.py \
  --model models/BAAI-bge-m3/bge-m3-source-F32.gguf \
  --oracle /tmp/bge-llama-scalar-11fe-build/bin/llama-debug \
  --rust target/bge-scalar/release-fast/rust-model-inference \
  --work-dir target/bge-m3-token-check-new
```

`--compare-outputs --pooling last` is an opt-in raw-bit check for wholly F32
`llama`, `qwen2` and `qwen3` text graphs only. It rejects BERT/Gemma2 GELU tables,
visual flags, any non-F32 tensor and quantized file types. Small synthetic
Mistral/Qwen2/Qwen3 fixtures exercise this mode; they do not establish parity
for real large decoder weights or every intermediate layer.

## Visual contracts and validation limits

Visualized BGE follows FlagEmbedding
`fd1a2bdf69488ffebe0327999d4400d8c8058a0b` (EVA02 saved 2D RoPE, SwiGLU and
BERT/XLM-R fusion). BGE-VL CLIP and LLaVA follow the ModelScope model code and
config. Screenshot follows its published Qwen2.5-VL model and transformers
v4.51.3 configuration defaults, including temporal patch pairs, window/full
vision attention, the official prompt templates and three-axis M-RoPE.
Image preprocessing is an inference implementation; resized pixels and visual
whole-model outputs have no bit-level Oracle claim.

Real CLI checks assert expected dimensions, finite values and unit L2 norm for
text/image/fusion inputs. These checks demonstrate execution, not embedding
quality. F32/Q8_0 synthetic fixtures exercise all text and visual families,
context and missing-image errors, padded vocabularies, tensor inventories and
HTTP image embedding without a separate projector. Text decoder checkpoint
indices cover 1,490 tensor mappings; every repository is not downloaded and
run individually. M3 sparse/ColBERT and rerankers are outside the requested
embedding scope.

LLaVA currently prefills image tokens one row at a time with F32 KV. The real
CPU image check took about 12 minutes while sharing the host with builds;
batched embedding prefill and stable throughput measurements remain unimplemented.

The final serial parity-trace library suite has 1,163 passed, 25 failed and
81 ignored. The 25 failures are identical to the unmodified `b6599fd4` baseline
(1,155 passed, 25 failed, 81 ignored). The full suite is not green.
