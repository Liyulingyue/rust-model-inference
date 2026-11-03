# PiVo-1.4M validation

Measured 2026-11-04 on Apple M3 Max, Darwin 27 arm64, Rust 1.98.1,
`--locked --profile release-fast`. Oracle: Python 3.14.7, NumPy 2.5.4,
ONNX 1.23.2, ONNX Runtime 1.31.1 CPU (one thread), PyTorch 2.15.0 for conversion.
This is an implementation check, not a listening study or benchmark.

## Pinned inputs

[sources.json](sources.json) pins all 23 ModelScope files by per-file revision,
SHA256 and byte count. The reference source files were not edited.

| Asset | SHA256 |
| --- | --- |
| `checkpoints/acoustic.pt` | `0b4a2dc993e7e500aac1825465886ff23c11f2e1a2662577f2f5cd27f4efb4eb` |
| `checkpoints/vocoder.pt` | `355d470826817afaa4e729d1b19c8383e985f365af5d2b913f1b9acdf8f7eca4` |
| `pivo/resources/cmudict.dict` | `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22` |
| Converted `pivo-F32.gguf`, 9,252,704 bytes | `e01cad6189b2d3dadee0a3662e7d65388044092b6f00ef66a21cdfd8c6de6d3d` |

Architecture `pivo`, format `pivo-f32-v1`: 109 F32 tensors, 1,401,562 learned
parameters plus 40 normalization-buffer entries. All 109 tensor payloads were
checked byte-for-byte against the original checkpoints. The GGUF includes both
networks, the dictionary, source hashes and retained license/notice texts.
The shared [contract](../../../src/models/pivo/contract.json) fixes every tensor
name, dimension and acoustic configuration value. Loader tests reject missing
components, incorrect shapes, F16, wrong lengths, NaN and unsupported config/format.

## Numerical results

`trace.py` wraps the unchanged official sessions and records encoder outputs,
the official chunked decoder's features, conditioning and recurrent states at
steps 0/1/16/128/256/512/1024 when present. Rust uses a full-utterance causal decoder
with the same 64-frame window. Teacher-forced checks isolate stages; a separate
comparison runs the complete native frontend, acoustic network, vocoder and DSP.

Eight cases cover both voices, speeds 0.5/1/1.25/2, initialisms, currency and
leap-day normalization, Unicode NFKC, pronunciation overrides, a five-frame
utterance, and a 504-frame utterance spanning multiple cache windows. Normalized
text, token IDs, every integer duration and every pitch period match exactly in
CPU SIMD, scalar-dispatch and Auto runs. Floating-point stages use tolerances.

Worst max-absolute / RMSE values across the eight cases:

| Checkpoint | CPU SIMD, two threads | Scalar dispatch, one thread | Acceptance gate |
| --- | --- | --- | --- |
| Encoder | `2.50e-6 / 3.29e-7` | `2.62e-6 / 3.06e-7` | `1e-4 / 1e-5` |
| Log duration | `1.79e-6 / 5.72e-7` | `1.67e-6 / 5.84e-7` | `1e-4 / 1e-5` |
| Expansion/phase | `5.96e-8 / 2.67e-9` | same | `1e-6 / 1e-7` |
| Decoder on official encoder output | `4.39e-5 / 1.38e-6` | `4.39e-5 / 1.38e-6` | `1e-4 / 1e-5` |
| Full native acoustic features | `4.20e-5 / 1.40e-6` | `4.20e-5 / 1.39e-6` | `1e-4 / 1e-5` |
| Conditioning | `1.34e-6 / 1.23e-7` | `9.24e-7 / 1.17e-7` | `2e-5 / 2e-6` |
| Teacher-forced excitation | `1.04e-7 / 5.28e-8` | `1.04e-7 / 4.33e-8` | `2e-5 / 2e-6` |
| Teacher-forced recurrent states | `2.09e-6 / 5.83e-7` | `1.73e-6 / 4.69e-7` | `2e-5 / 2e-6` |
| Vocoder on official features | `1.97e-3 / 2.80e-4` | `3.59e-3 / 5.17e-4` | `5e-3 / 7.5e-4` |
| Complete waveform | `4.22e-3 / 6.11e-4` | `1.93e-3 / 2.74e-4` | `5e-3 / 7.5e-4` |

Auto reports `scope=pivo backend=Cpu`, and its metrics match CPU SIMD. All eight
CLI WAVs are byte-identical between CPU and Auto. Rust uses existing
`ops::dot_f32`, RMS norm, softmax and activation dispatch. No shared operator
was modified and no external acceleration library was added.

The largest waveform difference is for Vo saying "Hello, my name is Vo.".
Before changing the initial RMSE gate of `5e-4`, two causes were separated:

1. An initial graph error applied RoPE to both decoder blocks. The pinned graph
   uses it only in block zero; fixing this reduced decoder error from about 1.34
   to at most `4.39e-5`.
2. FARGAN feedback amplifies rounding. Moving official decoder output one F32 ULP
   toward positive infinity and running the official vocoder produces max error
   `0.0045546442`, RMSE `0.00066176345`. Unperturbed official features reproduce
   the reference PCM exactly. Rust's pitch periods match; teacher-forced states
   remain within `2.09e-6`.

The final waveform gate accommodates this measured sensitivity while keeping
separate, tighter intermediate gates. `trace.py` records the one-ULP probe
separately; the primary reference checkpoints and PCM stay unmodified. This does
not establish perceptual equivalence for arbitrary text.

## Interfaces and review

The real `check_interfaces.py` run passed:

- Eight CLI oracle cases in CPU and Auto: mono 16 kHz PCM16 WAV and exact sample
  counts. The shared encoder truncates while the official writer rounds, allowing
  one PCM16 unit of serialization difference in addition to model error.
- Missing `--tts`, invalid voices/speeds/language, projector/reference, sampling,
  seed, KV/template, resolution/duration and other incompatible controls rejected.
- HTTP server defaults from voice/speed/pronunciations and per-request replacement;
  model identity, WAV and raw PCM checked against CLI bytes.
- Repeated requests and synthesis after concurrent traffic return the same WAV.
  Four simultaneous requests returned one 200 and three 429 responses.
- Invalid input/model/format returns 400; schema errors 422. Chat, embeddings, and
  text/image JEV requests return 404 because these routes are absent.

Self-review and independent review found and fixed ignored controls, ambiguous
case-insensitive override keys and non-finite feature indexing. Regression tests
verify deterministic rejection and that finite overflow-inducing weights return
a numerical error before vocoding. HTTP maps numerical failures to 500. The
integration run also exposed and fixed the missing `--tts` architecture gate.

## Reproduction and regression

See [README.md](README.md) for download/conversion and CLI/HTTP commands. Commands
used here (select a compatible toolchain if 1.98.1 is not installed):

```sh
.venv/bin/python -m unittest tools.converter.pivo.test_convert_pivo
.venv/bin/python tools/oracle/pivo/trace.py models/PiVo-1.4M artifacts/pivo/fixtures.json
PIVO_GGUF=models/PiVo-1.4M/pivo-F32.gguf PIVO_FIXTURES=artifacts/pivo/fixtures.json RAYON_NUM_THREADS=2 \
  cargo +1.98.1 test --locked --profile release-fast --lib models::pivo -- --include-ignored --nocapture
# Repeat the oracle command with RMI_SCALAR=1 RAYON_NUM_THREADS=1 and with PIVO_AUTO=1.
cargo +1.98.1 test --locked --profile release-fast --test pivo_cli
cargo +1.98.1 build --locked --profile release-fast --bin rust-model-inference --bin rust-model-server
.venv/bin/python tools/oracle/pivo/check_interfaces.py models/PiVo-1.4M/pivo-F32.gguf artifacts/pivo/fixtures.json
cargo +1.98.1 check --locked --profile release-fast --features vulkan --lib --bin rust-model-inference --bin rust-model-server
git diff --check
```

Converter: 2 tests passed. PiVo including real oracle: 6 passed in each of CPU
SIMD, scalar and Auto. CLI regressions: 2 passed. Vulkan-feature library/main/server
compilation passed. Strict Vulkan is rejected at runtime by the model's gate,
also covered without a physical GPU.

Full `cargo +1.98.1 test --locked --profile release-fast --lib`: baseline
**1330 passed / 20 failed / 88 ignored**; after **1336 / 20 / 89**. The exact
20 failure names were compared and are unchanged:

```text
models::breeze::tests::main_matrices_require_original_dtype_exact_shape_and_length
models::breeze::tests::main_source_preserves_original_bf16_and_legacy_codec_f32
models::breeze::tests::main_source_rejects_non_bf16_heads_eoi_and_norms
models::clm::tests::matches_reference_golden_vectors
models::clm::tests::rejects_wrong_embedding_width
models::diffusion::z_image::text::tests::qwen_attention_softmax_matches_the_pinned_ggml_neon_reduction
models::diffusion::z_image::text::tests::silu_mul_inplace_matches_pinned_ggml_neon_activation
models::diffusion::z_image::vae::tests::silu_matches_pinned_ggml_neon_vector_path
models::funasr::encoder::parity_tests::fsmn_rounds_product_before_adding_like_ggml
models::funasr::encoder::parity_tests::layernorm_accumulates_variance_in_ggml_neon_groups
models::gemma4::trunk::tests::failed_later_gemma4_chunk_preserves_successful_prefix
models::gemma4::trunk::tests::per_layer_bf16_projection_matches_pinned_scalar_dot_bits
models::gemma4::trunk::tests::per_layer_projection_rejects_non_bf16_weight
models::gemma4::trunk::tests::per_layer_projection_rejects_wrong_bf16_storage_length
ops::argmax::tests::all_negative
ops::argmax::tests::large_random
ops::argmax::tests::middle_max
ops::argmax::tests::tail_handles_non_aligned_lengths
ops::argmax::tests::ties_pick_lowest_index
ops::matmul::neon_tests::neon_softmax_matches_ggml_vector_exp_and_f64_sum
```

`cargo fmt --all -- --check` still fails on eight pre-existing unformatted files:
`app/server/api.rs`, `app/server/prediction.rs`, `app/wevisone_vl.rs`,
`models/audio8/{gpu,mod,text}.rs`, `ops/quant/neon_k.rs` and `vulkan/ops.rs`
(all under `src/`). Each failure was reproduced against its baseline HEAD source.
New files and changed PiVo regions were formatted; unrelated formatting was left
alone. The entire repository test suite and CI are not claimed green.

## Limits

English Pi/Vo, F32 GGUF, CPU/Auto only. JEV, voice cloning, sampling, reduced
precision and streaming HTTP are not implemented. Real-model measurements were
on this arm64 host; x86 CI is configured for lightweight loader/frontend/CLI
regressions, not the downloaded oracle. Long-form quality, listening tests,
physical GPU execution and performance remain unverified.
