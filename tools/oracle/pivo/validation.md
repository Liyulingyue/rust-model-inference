# PiVo-1.4M validation

Measured 2026-11-04 on Apple M3 Max, Darwin 27 arm64, Rust 1.98.1,
`--locked --profile release-fast`. Oracle: Python 3.14.7, NumPy 2.5.4,
ONNX 1.23.2, ONNX Runtime 1.31.1 CPU (one thread), PyTorch 2.15.0 for conversion.
This is an implementation check, not a listening study or benchmark.

Re-measured 2026-11-10 on Linux x86_64 (Intel Core Ultra 5 125H), Rust 1.95.0,
ONNX Runtime 1.31.1 CPU (one thread), NumPy 2.5.4, PyTorch 2.14.1 — see
[Coverage matrix](#coverage-matrix), [Performance](#performance) and
[Q8_0 export](#q80-export-the-audio-is-not-usable). The arm64 numbers below are
the original ones; the x86 column is the same eight oracle cases.

## Coverage matrix

Status columns: implementation, then verification. A cell that was never run
says so; it never says "not applicable" to hide a gap.

| Capability / component | Weight format | Compute / KV precision | Request backend | CLI command and result | HTTP route, request and result | Actual backend, metric, evidence | Implementation / verification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| English TTS, voice Pi | `pivo-f32-v1` | F32 weights, F32 compute | CPU | `--tts --model pivo-F32.gguf --prompt 'NASA is ready for the next mission.' --voice Pi --out pi.wav` — mono 16 kHz PCM16 WAV, 29,920 samples | `POST /v1/audio/speech` `{"input":"...","voice":"Pi"}` — same bytes as the CLI | CPU SIMD; encoder 2.21e-6 / 3.10e-7, waveform 3.87e-3 / 5.67e-4 (x86, worst of eight cases); oracle `official_onnx_checkpoints_and_waveform` | implemented / **passed** |
| English TTS, voice Vo | `pivo-f32-v1` | F32 weights, F32 compute | CPU | `--voice Vo` — verified through the eight oracle cases | `{"voice":"Vo"}` — verified in `check_interfaces.py` | as above | implemented / **passed** |
| Speed 0.5–2 | `pivo-f32-v1` | F32 | CPU | `--speed 0.5 … 2` — 0.5, 1, 1.25 and 2 are oracle cases | `{"speed":1.25}` — verified in `check_interfaces.py` | integer durations match the official graph exactly | implemented / **passed** |
| Pronunciation overrides | `pivo-f32-v1` | F32 | CPU | `--pronunciations '{"zzvo":["Z","IY1","V","OW0"]}'` — oracle case | `{"overrides":{"zzvo":[...]}}` — verified in `check_interfaces.py` | token IDs match the official frontend exactly | implemented / **passed** |
| Frontend normalization | `pivo-f32-v1` | F32 | CPU | numbers, currency, ISO dates, initialisms, NFKC — oracle cases | same code path | normalized text and token IDs match exactly | implemented / **passed** |
| JEV, chat, embeddings | — | — | — | — | 404, the routes are absent | — | **not applicable**: PiVo is a speech synthesis model with no scoring or text semantics (contract has no probability head) |
| Device path (Vulkan) | `pivo-f32-v1` | F32 weights, F32 compute | Vulkan | `--compute vulkan` — the acoustic encoder runs on the device, the trace prints `scope=pivo_encoder backend=Vulkan` | the server takes the same flag and the same path | encoder 4.61e-7 relative against the host, frame durations identical, waveform RMSE 7.41e-5 of full scale (gate 7.5e-4); the encoder itself is 1.03x (short) to 1.12x (long) slower than the host's SIMD path | implemented / **passed** — the encoder only, see [Vulkan device path](#vulkan-device-path) |
| Quantized weights | `pivo-q8_0-v1` | Q8_0 storage, F32 compute | CPU | `--quant q8_0` export — 6,988,192 bytes instead of 9,252,704 | loads, and the server prints the same warning as the CLI | waveform error 0.82 of full scale, pitch period mismatch up to 23.23% | implemented / **failed** — the audio is not usable, see [Q8_0 export](#q80-export-the-audio-is-not-usable) |

Auto is the same code path as CPU: `--compute auto` runs on the CPU, reports
`scope=pivo_encoder backend=Cpu` and produces byte-identical WAVs. The strict
`--compute vulkan` policy moves the acoustic encoder to the device; the rest
of the pipeline stays on the host.

## Pinned inputs

[sources.json](sources.json) pins all 23 ModelScope files by per-file revision,
SHA256 and byte count. The reference source files were not edited.

| Asset | SHA256 |
| --- | --- |
| `checkpoints/acoustic.pt` | `0b4a2dc993e7e500aac1825465886ff23c11f2e1a2662577f2f5cd27f4efb4eb` |
| `checkpoints/vocoder.pt` | `355d470826817afaa4e729d1b19c8383e985f365af5d2b913f1b9acdf8f7eca4` |
| `pivo/resources/cmudict.dict` | `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22` |
| Converted `pivo-F32.gguf`, 9,252,704 bytes | `e01cad6189b2d3dadee0a3662e7d65388044092b6f00ef66a21cdfd8c6de6d3d` |
| Converted `pivo-Q8_0.gguf`, 6,988,192 bytes | `4ddbb8fe7929d788a23e45cb472980b24cde85334da72c19fb40e12ba1a30bc0` |

Architecture `pivo`, format `pivo-f32-v1`: 109 F32 tensors, 1,401,562 learned
parameters plus 40 normalization-buffer entries. All 109 tensor payloads were
checked byte-for-byte against the original checkpoints. The GGUF includes both
networks, the dictionary, source hashes and retained license/notice texts.
The shared [contract](../../../src/models/pivo/contract.json) fixes every tensor
name, dimension and acoustic configuration value. Loader tests reject missing
components, incorrect shapes, F16, wrong lengths, NaN and unsupported config/format.

The Q8_0 export stores 52 of the 109 tensors in Q8_0 and keeps the rest F32.
GGML quantizes a row at a time and a Q8_0 block holds 32 values, so only the
matrices whose row length is a whole number of blocks qualify; the
one-dimensional norms, biases and statistics stay F32 on purpose, and the
convolution kernels keep a kernel axis of 3 or 5.

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

| Checkpoint | CPU SIMD, two threads (arm64) | CPU SIMD, two threads (x86) | Scalar dispatch, one thread (arm64) | Acceptance gate |
| --- | --- | --- | --- | --- |
| Encoder | `2.50e-6 / 3.29e-7` | `2.21e-6 / 3.10e-7` | `2.62e-6 / 3.06e-7` | `1e-4 / 1e-5` |
| Log duration | `1.79e-6 / 5.72e-7` | `2.62e-6 / 7.85e-7` | `1.67e-6 / 5.84e-7` | `1e-4 / 1e-5` |
| Expansion/phase | `5.96e-8 / 2.67e-9` | `5.96e-8 / 2.28e-9` | same | `1e-6 / 1e-7` |
| Decoder on official encoder output | `4.39e-5 / 1.38e-6` | `4.48e-5 / 1.34e-6` | `4.39e-5 / 1.38e-6` | `1e-4 / 1e-5` |
| Full native acoustic features | `4.20e-5 / 1.40e-6` | `4.77e-5 / 1.37e-6` | `4.20e-5 / 1.39e-6` | `1e-4 / 1e-5` |
| Conditioning | `1.34e-6 / 1.23e-7` | `9.24e-7 / 1.07e-7` | `9.24e-7 / 1.17e-7` | `2e-5 / 2e-6` |
| Teacher-forced excitation | `1.04e-7 / 5.28e-8` | `6.71e-8 / 2.99e-8` | `1.04e-7 / 4.33e-8` | `2e-5 / 2e-6` |
| Teacher-forced recurrent states | `2.09e-6 / 5.83e-7` | `2.72e-6 / 4.28e-7` | `1.73e-6 / 4.69e-7` | `2e-5 / 2e-6` |
| Vocoder on official features | `1.97e-3 / 2.80e-4` | `2.19e-3 / 3.33e-4` | `3.59e-3 / 5.17e-4` | `5e-3 / 7.5e-4` |
| Complete waveform | `4.22e-3 / 6.11e-4` | `3.87e-3 / 5.67e-4` | `1.93e-3 / 2.74e-4` | `5e-3 / 7.5e-4` |

The x86 column is the F32 export, worst over the same eight cases. It lands in
the same band as the arm64 column: both sit just under the waveform gate,
because the FARGAN feedback amplifies the transcendental rounding of both
architectures. The Q8_0 export is a different story, see
[Q8_0 export](#q80-export-the-audio-is-not-usable).

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

## Q8_0 export: the audio is not usable

The converter can export the weight matrices in Q8_0
(`--quant q8_0`), and the loader reads that format. **The audio it produces is
not usable, and the format should not be used.** The measurement is here so the
conclusion is reproducible rather than a matter of opinion.

The quantization itself is exact: for all 52 quantized tensors the largest
dequantization error is 0.4997 of the Q8_0 grid step `amax/127`, which is the
half-ULP bound of the round-to-nearest rule. The loss is not in the
quantizer.

The loss is in the model. PiVo's FARGAN vocoder feeds its own output back into
the next step, so a small weight perturbation compounds. Scaling every weight
matrix of the F32 export by a constant gives:

| Weight scaling | Waveform error against the F32 export | Waveform correlation |
| --- | --- | --- |
| × (1 + 1e-5) | 0.12% RMS | 0.999999 |
| × (1 + 1e-4) | 1.20% RMS | 0.999928 |
| × (1 + 1e-3) | 74.97% RMS | 0.718 |
| Q8_0 export (an effective weight change of the same order) | 86.53% RMS | 0.626 |

The Q8_0 export lands past the cliff. The consequences, measured against the
official ONNX graph over the eight oracle cases:

| Checkpoint | max_abs | RMSE |
| --- | --- | --- |
| Encoder | `3.29e-2` | `6.55e-3` |
| Log duration | `3.87e-2` | `1.27e-2` |
| Decoder | `1.75e-1` | `1.16e-2` |
| Conditioning | `6.02e-2` | `6.48e-3` |
| Vocoder on official features | `5.99e-1` | `5.88e-2` |
| Complete waveform | `8.20e-1` | `7.91e-2` |

The waveform error is 0.82 of full scale. The discrete contract breaks too:
up to 7.14% of the frame durations and up to 23.23% of the pitch periods stop
matching the official graph. The loudness envelope survives (frame energy
correlation 0.9993, no time shift), so the speech keeps its rhythm and volume,
but the carrier phase drifts and the timbre changes.

The loader therefore keeps the format readable, the CLI prints
`format=pivo-q8_0-v1` and warns that the audio is not usable, and the oracle
test asserts the measured degradation band instead of the F32 gates. The F32
export is the one to use.

## Performance

[performance.json](performance.json) holds the full record. Same machine, same
weights, same input, measured from a cold process to the finished audio on
both sides (Intel Core Ultra 5 125H, x86_64, `release-fast`):

| Case | Rust `--compute cpu` | Official ONNX Runtime, one thread | RTF (Rust / ONNX) |
| --- | --- | --- | --- |
| "NASA is ready for the next mission." (1.87 s audio) | 192.4 ms | 192.3 ms | 0.103 / 0.103 |
| "The quick brown fox jumps over the lazy dog." (2.57 s audio) | 219.9 ms | 240.2 ms | 0.086 / 0.093 |

The Rust implementation matches the official ONNX Runtime: 1.00x on the short
case and 0.92x on the long one, both far faster than real time. `--threads 1`,
`2` and `4` make no measurable difference — the model is 1.4M parameters, so
the per-frame vocoder loop dominates and the thread pool has nothing to
parallelise.

The Vulkan device path exists for the acoustic encoder. It is measured, and it
is slower than the host's SIMD path, so Auto stays on the CPU and the strict
policy is the only way to reach it:

| Case | `--compute cpu` (HTTP request) | `--compute vulkan` (HTTP request) | Ratio |
| --- | --- | --- | --- |
| "NASA is ready for the next mission." (21 tokens) | 75.8 ms | 78.3 ms | 1.03x |
| Four sentences (about 130 tokens) | 561.7 ms | 630.7 ms | 1.12x |

The reason is structural, not a missing kernel: the encoder is 3-12% of an
utterance (the vocoder's strictly sequential FARGAN loop is the rest), and the
device pays roughly 50 microseconds per dispatch while the encoder needs about
70 dispatches. The numbers are in [performance.json](performance.json).

## Vulkan device path

The strict `--compute vulkan` policy runs the acoustic encoder on the device.
Everything else stays on the host, and the split is not a compromise between
the two: the two parts cannot move.

- **The vocoder stays on the host.** FARGAN feeds its own output back into the
  next step, so the frames of an utterance are strictly sequential. A device
  would add a dispatch per frame and remove no work: 92% of an utterance's
  time is that loop.
- **The decoder stays on the host.** It feeds a pitch period whose rounding
  margin is 1.36e-6, smaller than the difference between the host and the
  device, so a device decoder would flip periods and change the audio.
- **The duration head stays on the host.** It produces the integer frame
  durations and those are rounded. The rounding margin of the shortest oracle
  case is 0.0024 frames, so the device and the host have to agree on the log
  duration far better than that before the frame count can differ.

What runs on the device is therefore the encoder: three attention blocks, three
convolution blocks and the final norm. Seven shaders implement it, all in
PiVo's own layout, because the shared kernels keep a different one and would
need a transpose on the way in or out:

| Shader | Operator |
| --- | --- |
| `pivo_linear.comp` | `out[row][o] = bias[o] + sum_i in[row][i] * weight[o][i]` |
| `pivo_rms_norm.comp` | RMS normalisation, Kahan sum, weight resident in the arena |
| `pivo_rope.comp` | split-half rotary embedding, driven by the host's table |
| `pivo_attention.comp` | grouped-query attention, four query heads over two key heads |
| `pivo_conv1d.comp` | dilated 1D convolution in PiVo's frame-major layout |
| `pivo_swiglu.comp` | `silu(gate) * up` |
| `pivo_silu.comp` | SiLU for the convolution blocks |

The weights are uploaded once and stay resident; the activations stay in the
arena between the operators, and the whole encoder is recorded into one
command buffer.

### The rotary embedding is the host's

A device's `sin` and `cos` are a couple of ULP away from the host's, and the
rotation moves with them. The first version computed the angle on the device
and the encoder drifted by 34% — the encoder's output is the input of the
duration head, whose rounding margin is 0.0024 frames, so that much drift
changes the utterance length. The host now fills a table with its own
`sin_cos` and the device reads it, which makes the rotation the host's
rotation bit for bit.

### What the device path reproduces

| Quantity | Device against the host |
| --- | --- |
| Encoder output, relative error | 4.61e-7 |
| Log duration, absolute error | 4.77e-7 |
| Frame durations | identical, every case |
| Waveform, RMS error of full scale | 7.41e-5 (the oracle's gate is 7.5e-4) |
| Waveform, largest sample difference | 13 of 32767, on one Intel integrated GPU |

`src/vulkan/ops.rs` holds an operator-level check that runs every one of the
seven shaders against a reference that follows the host kernel step by step,
and `src/models/pivo/tests.rs` holds the encoder-level check that compares the
frame durations of the device and the host.

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

The x86 re-run repeated all of it: `check_interfaces.py` reported
`CLI: 8 oracle cases; CPU/Auto byte-identical WAV`,
`CLI: incompatible controls and task routes rejected` and
`HTTP: defaults/overrides, PCM/WAV, reset, 400/404/422/429 passed [200, 429, 429, 429]`.

The Vulkan build adds one line: `CLI: strict Vulkan ran the encoder on the
device and stayed inside the gate`. A build without the Vulkan feature
rejects the flag instead and says so.

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

Converter: 3 tests passed. PiVo including real oracle: 9 passed in each of CPU
SIMD, scalar and Auto, and 9 passed with the Q8_0 export. CLI regressions: 2 passed.
Vulkan-feature library/main/server compilation passed. Strict Vulkan is rejected
at runtime by the model's gate, also covered without a physical GPU.

Full `cargo test --locked --profile release-fast --lib` on 2026-11-10 (x86_64):
baseline **1471 passed / 16 failed / 96 ignored** on `origin/main`; after
**1479 / 16 / 97**. The exact 16 failure names were compared and are unchanged:

```text
app::media::tests::stereo_uploads_are_refused
format::ggufrs::tests::open_does_not_map_sparse_segment_region
models::breeze::tests::main_matrices_require_original_dtype_exact_shape_and_length
models::breeze::tests::main_source_preserves_original_bf16_and_legacy_codec_f32
models::breeze::tests::main_source_rejects_non_bf16_heads_eoi_and_norms
models::clm::tests::matches_reference_golden_vectors
models::clm::tests::rejects_wrong_embedding_width
models::dots::vocoder::tests::q8_lstm_matches_independent_gate_order_bias_and_recurrent_state
models::gemma4::trunk::tests::failed_later_gemma4_chunk_preserves_successful_prefix
models::gemma4::trunk::tests::per_layer_bf16_projection_uses_the_repository_bf16_kernel
models::qwen_drive::tests::planner_scalar_fixture_matches_every_f32_word
models::yue2::tests::nar_noise_matches_torch_2_10_cpu_bits_across_tail_boundaries
ops::dot::tests::dot_bf16_f32_is_bitwise_stable_against_ascending_reference
ops::math::exp::avx2_tests::avx2_exp_stays_finite_above_the_clamp
ops::matmul::neon_tests::rope_neox_inplace_matches_pinned_ggml_recurrence_and_fused_rotation
ops::rope::tests::vision_rope_matches_ggml_libc_trig_bits
```

`cargo fmt --all -- --check` passed on 2026-11-10. The eight pre-existing
unformatted files that this record used to list
(`app/server/api.rs`, `app/server/prediction.rs`, `app/wevisone_vl.rs`,
`models/audio8/{gpu,mod,text}.rs`, `ops/quant/neon_k.rs` and `vulkan/ops.rs`)
were reformatted upstream in the meantime, so that list no longer applies.
The entire repository test suite and CI are not claimed green.

## Limits

English Pi/Vo, F32 GGUF, CPU/Auto by default. JEV, voice cloning, sampling and
streaming HTTP are not implemented. The Q8_0 export exists and loads, but its
audio is not usable, so the F32 export is the only one to use. Real-model
measurements were made on an arm64 host and repeated on x86_64; the CI is
configured for lightweight loader/frontend/CLI regressions, not the downloaded
oracle. Long-form quality, listening tests and performance beyond the cases
above remain unverified. The device path was measured on one Intel
integrated GPU; a discrete GPU was not available, and the device path is
slower than the host's SIMD path there.
