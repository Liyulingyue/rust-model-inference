# Q4 and K-Quant Bitwise Parity Design

## Goal

Make the existing CPU inference path for these two local models bitwise reproducible against a pinned llama.cpp scalar oracle:

- `models/Qwen3-0.6B/Qwen3-0.6B-Q4_0.gguf`
- `models/Qwen3-0.6B/Qwen3-0.6B-Q4_K_M.gguf`

For every declared runtime checkpoint, comparison covers every element's raw little-endian `f32` bits. Final validation also covers all 151936 logits and generated token IDs. Approximate metrics such as cosine similarity, relative error, and tolerances are not acceptance criteria.

## Oracle Contract

The reference implementation is upstream llama.cpp commit `749f688fcaa4c472ec034b08cb8a907c45cfaa02`.

The parity harness builds a deterministic reference executable with:

- CPU only and one worker thread;
- Metal, Accelerate, BLAS, and architecture-specific SIMD disabled;
- F32 key/value cache;
- greedy sampling;
- warmup disabled;
- the exact same GGUF and prompt bytes used by Rust.

The repository stores the pin, the small oracle patch, and build/run scripts. It does not vendor llama.cpp, generated executables, models, or trace binaries. The harness checks the reference checkout's exact commit before applying the patch or running comparisons.

Bitwise parity is guaranteed only for this pinned scalar contract. Native llama.cpp SIMD and GPU backends are outside the exact-bit contract because their reduction order is platform-specific.

## Trace Format and Coverage

Both implementations write a JSONL manifest plus raw little-endian `f32` sidecars. Each record contains a stable checkpoint name, layer and step when applicable, shape, element count, occurrence, and sidecar path. Token records contain exact unsigned token IDs.

The comparator first checks record names, order, shapes, lengths, and token IDs. It then compares each `f32` word as a `u32` and stops at the first mismatch, reporting model, prompt, step, layer, checkpoint, flat index, Rust bits, llama.cpp bits, and both decoded values.

The trace covers, for every evaluated token and all 28 layers:

1. encoded prompt token IDs and the current input token;
2. token embedding output;
3. attention RMS normalization;
4. Q, K, and V projections;
5. Q/K head normalization;
6. Q/K values after RoPE;
7. attention scores, probabilities, and value aggregation;
8. attention output projection and post-attention residual;
9. FFN RMS normalization;
10. gate and up projections;
11. SiLU-gated activation;
12. down projection and post-FFN residual;
13. final RMS normalization;
14. all 151936 output logits;
15. selected and generated token IDs.

Trace checkpoints are compiled only with the existing `parity-trace` feature and have no cost in normal builds.

## Fixtures

Each model runs these deterministic fixtures:

- `a`, covering the shortest non-empty prompt and first-token logits;
- `17 + 25 =`, covering several ASCII tokens;
- `你好`, covering multi-byte UTF-8 tokenization;
- a two-token generation from `a`, covering KV-cache readback on the second decode step.

The harness rejects a comparison before tensor checks if tokenization differs. All fixtures use one thread, F32 KV cache, greedy decoding, and the same chat-template/thinking configuration.

## Production Changes

Work proceeds strictly from the first mismatch rather than changing several math paths at once.

The already-identified first candidate is Q6_K token embedding lookup. Both target GGUFs store `token_embd.weight` as Q6_K and tie output weights to that tensor. The embedding lookup will reuse the existing, verified Q6_K row decoder instead of keeping a second bit-layout implementation.

K-quant matrix multiplication will use the existing Q8_K activation representation and llama.cpp-compatible Q4_K x Q8_K and Q6_K x Q8_K scalar dot products. The current Q8_0-to-F32 reconstruction path is not part of the parity design because it changes quantization and accumulation order.

Subsequent changes to RMSNorm, RoPE, softmax, attention, FFN, KV-cache conversion, or sampling are permitted only when the trace identifies that operation as the first remaining mismatch. No new abstraction or dependency is introduced unless an observed mismatch cannot be fixed through the existing operation boundary.

## Test Strategy

Every production fix follows red-green-refactor:

1. add the smallest non-uniform unit fixture that fails on the incorrect layout or arithmetic order;
2. record the expected raw bits from the pinned scalar oracle;
3. run the focused test and confirm the expected mismatch;
4. make the smallest production change;
5. rerun the focused test and the end-to-end first-mismatch comparator.

Focused unit tests cover Q6_K embedding rows, Q8_K activation quantization, Q4_K/Q6_K dot products, and any later primitive implicated by the trace. Environment-backed integration tests are ignored by default and require explicit model and oracle paths. They fail loudly on a wrong llama.cpp commit, missing trace record, stale trace, shape difference, or bit mismatch.

## Acceptance Criteria

The work is complete only when:

- both models load and run without panic;
- all four fixtures have identical prompt token IDs;
- every declared checkpoint has identical order, shape, length, and raw `f32` bits;
- every full-vocabulary logit vector has zero bit mismatches;
- generated token IDs match for two decode steps;
- normal builds remain free of parity-trace I/O;
- focused Rust tests, the two environment-backed parity tests, the main binary build, and an x86_64 compile check pass.

Existing unrelated repository-wide formatting or obsolete test failures are reported separately and are not silently reformatted or repaired as part of this work.

## Non-Goals

- Bitwise parity with Metal, GPU, Accelerate, BLAS, or native SIMD llama.cpp.
- New Q4_K/Q6_K SIMD kernels before the scalar parity gate passes.
- Committing GGUFs, trace sidecars, generated binaries, or a llama.cpp source tree.
- Refactoring unrelated model, audio, vision, or diffusion code.
