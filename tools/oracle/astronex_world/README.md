# Astronex-World scalar Oracle

This directory is an independent raw-bit gate for the pinned official graph.
`scalar.c` is compiled at run time with `-ffp-contract=off`,
`-fno-vectorize`, and `-fno-slp-vectorize`; no BLAS, fused attention, CUDA,
approximate operator, or Rust implementation is used to produce Oracle values.
The graph is CPU F32 after loading weights. The only wider arithmetic allowed is
the official F64/complex RoPE and time-frequency path; its result is rounded to
F32 at the graph boundary. Any other floating operator or BF16/F16 cast fails
closed.

The reference checkout must be clean at
`27584bf1a89da01ef35a03b48d83bc818d2adf2b`. If a contract checkout is supplied,
it must be clean at `dec4ec502f7113a1ab07b78ae40a7ac1b88b2eeb`. The runner does
not edit either checkout.

## Commands

```sh
.venv/bin/python tools/oracle/astronex_world/trace_official.py self-test
.venv/bin/python tools/oracle/astronex_world/trace_official.py fixture /tmp/astronex-fixture
.venv/bin/python tools/oracle/astronex_world/trace_official.py trace \
  --weights models/Astronex-World --contract references/astronex-world-contract \
  /tmp/astronex-oracle
.venv/bin/python tools/oracle/astronex_world/trace_official.py compare \
  /tmp/astronex-oracle /tmp/astronex-native
cargo test --profile release-fast --lib astronex_checkpoint_tokenizer -- --ignored
.venv/bin/python tools/oracle/astronex_world/checkpoint_operators.py \
  artifacts/astronex-checkpoint-operators
RMI_SCALAR=1 cargo test --profile release-fast --test astronex_checkpoint_operators \
  -- --ignored --nocapture
```

The ignored checkpoint-tokenizer test reads `astronex.tokenizer.json` from the
actual components GGUF and compares its token IDs and all 512 mask/padding
positions with `fixtures/tokenizer_ids.json`. The seven cases include empty,
English, Chinese, whitespace, Unicode, special tokens and truncation. Expected
IDs were produced by the pinned official `AutoTokenizer.from_pretrained` call
with `padding="max_length"`, `max_length=512`, `truncation=True` and special
tokens enabled. The fixture records the input tokenizer SHA256; this is a real
checkpoint tokenizer check, separate from synthetic floating-operator fixtures.

`checkpoint_operators.py` reads raw BF16/F32 values from the original safetensors
and computes independent scalar C results using the same no-FMA/no-vectorization
flags. The ignored Rust test loads the actual paired GGUFs and compares every
output bit for twelve cases: nine denoiser/UMT5 projections, patch and VAE
convolutions, and six embedding rows including the last vocabulary entry.
On the local checkpoint, all 137,984 F32 output values match. Serialized inputs,
tensor names/shapes, source-byte hashes, pinned reference commit and conversion
contract revision make the cases reproducible; the Rust test checks that revision
against the GGUF metadata. Approximate activations, normalization, solver and
GPU arithmetic are absent. This is real-weight operator parity, not a whole-model
causal or 5B end-to-end parity result.

`fixture` is an operator contract only. It proves sequential reductions,
materialized softmax attention, 3-D convolution, token IDs, trace ordering,
and first-divergence reporting; it is not a model-weight result. `trace` refuses
to start until all indexed denoiser shards, `tokenizer/`, `text_encoder/`, and
`vae/` are present and complete. The real trace uses a fixed prompt, fixed
CPU noise, 32x32 latent geometry, camera matrices, action vectors, an initial
bidirectional-style denoiser call, a cache continuation call, VAE encode/decode,
and the official UniPC sampler. It records the text/tokenizer, non-causal
bidirectional Wan stage, causal cache stage (including camera/action inputs),
VAE encode/decode, and UniPC stages in one ordered stream. The native route now implements the no-control causal KV continuation graph.
Camera/action and `clip_fea/y` records remain diagnostic coverage and are not
claimed as native parity. Causal cache behavior and GPU kernels have separate
behavior/error tests; they are not retroactively treated as raw-bit parity.

## Trace contract

Each line in `trace.jsonl` contains `name`, zero-based `occurrence`, `kind`, and
`shape`. Floating checkpoints contain one little-endian F32 file and an exact
element `count`; token checkpoints contain integer `values`. The comparator
requires both traces to have nonzero equal counts, identical names/order,
occurrences, kinds, shapes, token IDs, and file lengths. It then compares raw
`u32` bits and reports the first divergent checkpoint/index with both hex words.
Missing, empty, duplicate, truncated, or extra checkpoints are errors. Cache
continuations intentionally repeat stable names with occurrence `1` rather
than silently replacing occurrence `0`.

The converter's mapping is preserved: official `model.<suffix>` becomes
`astronex.<suffix>` in the native trace, while the Oracle hooks the official
module names (`astronex.text.*`, `astronex.denoiser.*`, and `astronex.vae.*`).
The native side must normalize those names to this same order before running
`compare`. Approximate GELU, SIMD/FMA and quantized kernels, platform `ln`/
`exp_m1`, UniPC scheduler arithmetic, GPU execution, and the causal-only paths
are excluded from the raw-bit gate; they require separate tolerance or behavior
tests.

The independent small-graph tests cover those same official module classes
without weights. Real-weight execution remains fail-closed: every indexed shard
must be complete and an `.incomplete` shard is deliberately rejected. A real
causal trace can take substantially longer than the supported 32x32 native
smoke, so an interrupted run is not a parity result.
