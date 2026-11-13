# Underdog Woof validation

2026-11-14. **Experimental / partial adaptation**: CPU same-weight stage gates
pass for all nine formats; the long Mira free-generation waveform gate and
strict Vulkan probes fail. This is a draft integration, not full acceptance.
Machine-readable measurements, commands and errors are in [results.json](results.json)
and [performance.json](performance.json). Reproduction starts in [README.md](README.md).

## Contract and environment

- Base: `237a67bba3d8255f459e493701b4435c29e3cb11`.
- ModelScope revision: `bfdcdcbb002ce26fd86ca07c13845a78130aaec4`.
- `woof_int8.npz`: 40,595,426 bytes;
  SHA256 `d895fbbe84b8f4f461e3e94ab48b1164353b1988ce8d365f5aa8263d0de1203e`.
- F32 GGUF: 153,003,648 bytes;
  SHA256 `89a63f9c9913d1a833253ff0fcd603535be2498041d9bd1f51ddac88b766a67c`.
- Host: AMD EPYC 9334, Linux 6.8.0-147, AMD Radeon Graphics / RADV NAVI31.
  Rust 1.99.1; Python 3.12.3; NumPy 2.3.4; gguf 0.19.0. Two compute threads.
- Assets were streamed directly into the remote repository under
  `models/Underdog-Woof`. No model weights were downloaded to the Mac.
  Python dependencies live in `.venv`, Rust toolchain/cache in `target/`.
- [contract.json](../../../src/models/underdog_woof/contract.json) pins all source
  hashes, 175 NPZ arrays, configuration, 127 GGUF tensor names/shapes and SEANet
  layout. Explicit architecture is `underdog_woof`; no projector is required.
- Pure English TTS: Juno/Mira, 24 kHz mono. Speech transformer, one-step flow
  head and adapted Mimi decoder are included in the same GGUF.

F32 denotes exact effective values from the published int8/grouped-int8/FP16
model, not unavailable weights preceding upstream quantization. Source and
F32-reloaded official runtimes have **240 exactly matching checkpoints**.
Normalization has 466 pinned golden cases, splitting has 47, and character
IDs are compared exactly. Lossy floating-point paths use numerical tolerances.

## Precision and independent oracle

`export_reference.py --gguf` uses upstream `gguf.quants.dequantize` and reverses
the storage layouts into the pinned official NumPy runtime. Reference arithmetic
does not call Rust kernels or the converter's quantizer. Fixed latent feedback
and fixed decoder input separate implementation error from autoregressive drift.

The following thresholds were declared before the probes and remain unchanged:

| Stored format | Relative RMSE maximum | Cosine minimum | Maximum absolute error |
| --- | ---: | ---: | --- |
| F32 | 0.0002 | 0.999999 | `0.002 * (1 + reference peak)` |
| F16 / BF16 | 0.002 | 0.99999 | `0.02 * (1 + reference peak)` |
| Q8_0 / Q4_0 / Q4_1 / Q4_K / Q5_K / Q6_K | 0.06 | 0.998 | `0.25 * (1 + reference peak)` |

Both cases ran for every format: Juno / seed 0 / `Hi.` / 16 frames, and
Mira / seed 7 / `The quick brown fox.` / 32 frames; temperature 0.5. These
are teacher-forced speech probes plus fixed-input decoder probes. Maxima over
the two cases are below; all three gates passed, including cosine.

| Format | Latent relative RMSE | Latent max abs | Wave relative RMSE | Wave max abs |
| --- | ---: | ---: | ---: | ---: |
| F32 | 7.9965e-7 | 5.9605e-6 | 1.1559e-6 | 1.0952e-6 |
| F16 | 0.00063930 | 0.0058920 | 0.0010841 | 0.0010229 |
| BF16 | 8.9878e-7 | 4.9472e-6 | 9.8974e-7 | 7.1526e-7 |
| Q8_0 | 0.025889 | 0.21427 | 0.026499 | 0.015080 |
| Q4_0 | 0.026369 | 0.32901 | 0.030609 | 0.026479 |
| Q4_1 | 0.025036 | 0.18240 | 0.027833 | 0.017241 |
| Q4_K | 7.9965e-7 | 5.9605e-6 | 0.024795 | 0.025405 |
| Q5_K | 7.9965e-7 | 5.9605e-6 | 0.024236 | 0.015111 |
| Q6_K | 7.9965e-7 | 5.9605e-6 | 0.030012 | 0.019612 |

A further **260-frame** F32 forced probe crosses the decoder's 250-frame
attention window and eight-frame convolution chunks: latent relative RMSE
`7.0058e-7`, waveform `9.8867e-7`; strict F32 gates passed. This probe repeats
official short-case latents as feedback and is a window/state check, not a
260-frame natural speech quality claim.

Normal CPU inference uses shared SIMD, `ComputePool`, prepared F16/BF16 and
native block-quantized projections. Quantized kernels also round activations;
KV, state and accumulation are F32. F16/BF16 each store 84 low-precision and
43 F32 tensors; Q8_0/Q4_0/Q4_1 store 76 quantized and 51 F32 tensors. K formats
store 16 quantized decoder matrices and 111 F32 tensors because the speech
matrix row widths 448/1216/320 are not K-block aligned. K-format latent results
therefore match F32. Storage dtype is not whole-model compute precision.

`results.json.storage_loss_against_original` separately measures export loss
against original effective weights. F32 has zero export loss; further
quantization changes generated audio, especially 4-bit exports. Same-weight
implementation gates do not establish equal audio quality across formats.
F32 is recommended. Subjective listening was not performed.

## Natural generation and the retained failure

Original-weight `Hi.` with natural EOS passes strict F32 gates for both voices:

| Voice | Frames | Latent relative RMSE | Wave relative RMSE |
| --- | ---: | ---: | ---: |
| Juno | 26 | 4.0119e-6 | 3.9060e-5 |
| Mira | 34 | 9.1942e-6 | 0.00010043 |

The official public wrapper exercises normalization, splitting, silence/runaway
scoring, up to four takes and best-take selection. Full WAV/PCM comparison uses
the declared public threshold `relative RMSE <= 0.002, cosine >= 0.99999`:

| Public input / voice | Samples | Wave relative RMSE | Cosine | Result |
| --- | ---: | ---: | ---: | --- |
| `Hi.` / Juno | 24,960 | 0.00037024 | 0.99999994 | Pass |
| `Hi.` / Mira | 28,800 | 0.00029544 | 0.99999996 | Pass |
| Two sentences / Juno | 125,760 | 0.00040454 | 0.99999993 | Pass |
| Two sentences / Mira | 165,120 | **0.04634149** | **0.99892601** | **Fail** |

Two-sentence input: `Your order costs $12.50. It ships on March 3rd.`
Mira's first normalized sentence has 110 natural frames. Teacher forcing the
same sequence passes, and all checkpoint relative errors stay below `1.132e-5`.
Without forcing, the first floating difference is `speech.prefix.0`
(`3.1006e-7` relative RMSE); the first relative gate crossing is
`speech.head.24` (`0.00026383`). This localizes amplification to autoregressive
feedback from small reduction-rounding differences, rather than decoder layout.

An independent official-runtime sensitivity experiment changes flow-head
results with `np.nextafter`, leaving its source files and weights unchanged:

| Official head perturbation | Public waveform relative RMSE |
| --- | ---: |
| +1 ULP / -1 ULP | 0.021614 / 0.008708 |
| +4 ULP / -4 ULP | 0.008306 / 0.013406 |
| +16 ULP / -16 ULP | 0.109457 / 0.038111 |

All retain 165,120 samples. This supports a rounding-sensitivity diagnosis;
it does **not** turn the failed natural gate into a pass or establish perceptual
quality. SIMD remains enabled. The original failed record remains in
`interfaces[0].oracle_failures`; the interface checker exits unsuccessfully
when that gate fails even though the remaining API checks run.

## CLI / HTTP / backend coverage

Each row exercises all TTS components in the same model: text frontend, speech
transformer/flow head and decoder. Implementation and verification are separate.
All rows use F32 state/KV. The GPU design offloads shared linear projections;
normalization, RoPE, attention/state and convolution orchestration stay on host.

`check_interfaces.py` records full per-row CLI/server commands and actual HTTP
requests in `results.json.interfaces`. Both voices use `Hi.`, seed 0,
temperature 0.5. Every successful row checks nonempty WAV, CLI=HTTP bytes,
PCM payload, request reset, invalid inputs and route isolation. Extended F32 CPU
checks cover nondefault seed/temperature, normalize=false, split input and
concurrent 200/429 responses. Unsupported explicit CLI controls are rejected.

| Format | Requested | Implementation | CLI verification | HTTP verification | Actual execution / numeric evidence |
| --- | --- | --- | --- | --- | --- |
| F32 | cpu | Implemented | Pass | Pass | CPU; stage gates pass; long public Mira gate fails |
| F32 | auto | Implemented | Pass | Pass | CPU fallback; short same-weight gates pass; GPU unverified |
| F32 | vulkan | Implemented hybrid | Fail | Fail / 500 | Device lost before projections; no GPU numeric result |
| F16 | cpu | Implemented | Pass | Pass | CPU; both same-weight cases pass |
| F16 | auto | Implemented | Pass | Pass | CPU fallback; short same-weight gates pass; GPU unverified |
| F16 | vulkan | Implemented hybrid | Fail | Fail / 500 | Device lost; no GPU numeric result |
| BF16 | cpu | Implemented | Pass | Pass | CPU; both same-weight cases pass |
| BF16 | auto | Implemented | Pass | Pass | CPU fallback; short same-weight gates pass; GPU unverified |
| BF16 | vulkan | Implemented hybrid | Fail | Fail / 500 | Device lost; no GPU numeric result |
| Q8_0 | cpu | Implemented | Pass | Pass | CPU; both same-weight cases pass |
| Q8_0 | auto | Implemented | Pass | Pass | CPU fallback; short same-weight gates pass; GPU unverified |
| Q8_0 | vulkan | Implemented hybrid | Fail | Fail / 500 | Device lost; no GPU numeric result |
| Q4_0 | cpu | Implemented | Pass | Pass | CPU; both same-weight cases pass |
| Q4_0 | auto | Implemented | Pass | Pass | CPU fallback; short same-weight gates pass; GPU unverified |
| Q4_0 | vulkan | Implemented hybrid | Fail | Fail / 500 | Device lost; no GPU numeric result |
| Q4_1 | cpu | Implemented | Pass | Pass | CPU; both same-weight cases pass |
| Q4_1 | auto | Implemented | Pass | Pass | CPU fallback; short same-weight gates pass; GPU unverified |
| Q4_1 | vulkan | Implemented hybrid | Fail | Fail / 500 | Device lost; no GPU numeric result |
| Q4_K | cpu | Implemented | Pass | Pass | CPU; both same-weight cases pass |
| Q4_K | auto | Implemented | Pass | Pass | CPU fallback; short same-weight gates pass; GPU unverified |
| Q4_K | vulkan | Implemented hybrid | Fail | Fail / 500 | Device lost; no GPU numeric result |
| Q5_K | cpu | Implemented | Pass | Pass | CPU; both same-weight cases pass |
| Q5_K | auto | Implemented | Pass | Pass | CPU fallback; short same-weight gates pass; GPU unverified |
| Q5_K | vulkan | Implemented hybrid | Fail | Fail / 500 | Device lost; no GPU numeric result |
| Q6_K | cpu | Implemented | Pass | Pass | CPU; both same-weight cases pass |
| Q6_K | auto | Implemented | Pass | Pass | CPU fallback; short same-weight gates pass; GPU unverified |
| Q6_K | vulkan | Implemented hybrid | Fail | Fail / 500 | Device lost; no GPU numeric result |

JEV, transcription, chat, voice cloning and HTTP streaming are not applicable
to this fixed TTS contract. The matching routes are absent (404); unknown
speech fields/invalid JSON types return 422, invalid model/voice/text/format
return 400, concurrent synthesis 429. CLI requires `--tts`, rejects irrelevant
controls and server-only explicit prompt/output. Native controls and defaults
are shared through `models::underdog_woof::SpeechRequest`.

## GPU failure and performance

All nine strict Vulkan attempts fail with `Vulkan init failed: The logical
device has been lost` after RADV warmup. All Auto fixture probes record
`gpu_projections=0, cpu_projections=917`. Auto fallback proves availability of
CPU output, not GPU support. The unchanged shared control
`compute::linear::tests::rounded_bf16_prepared_linear_matches_cpu_with_tails`
fails with the same device error, before any Woof-specific operation. No shared
GPU kernel or driver/system configuration was changed in this adaptation.

On the same host, F32/Juno/two-sentence input yields 5.24 seconds of audio:

| Path | Seconds | Scope / outcome |
| --- | ---: | --- |
| Official NumPy median of 3 | 0.7891 | Loaded-model synthesis; construction separately 0.3300 s |
| Rust CPU median of 3 | 1.4359 | Full release CLI process including startup/load; RTF 0.274 |
| Rust Auto, one attempt | 5.2919 | CPU fallback plus failed device setup; no GPU throughput |
| Rust strict Vulkan, one attempt | 4.9122 | Failed; no completed inference |

The CPU scopes differ, so no equal-scope speed ratio is claimed. Rust reloads
and prepares projections for a request and orchestrates causal/window attention
and convolution on host. NumPy is the read-only upstream oracle; no external
acceleration library was added to Rust.

## Engineering checks and self-review

- Release CLI and server build with Vulkan: passed (8m32s).
- `cargo fmt --check`, `git diff --check`: passed.
- Offline `cargo check --profile release-fast --all-targets --features vulkan,parity-trace --locked`:
  passed, including all existing binaries, examples and integration tests.
- Woof targeted tests with Vulkan/parity: **13 passed, 0 failed, 1 ignored**;
  ignored model fixture was exercised separately with remote assets.
- Converter/download Python tests: **13 passed**, plus `py_compile` for the tools.
- Baseline library suite: **1527 passed, 14 failed, 100 ignored**.
  Current: **1540 passed, 14 failed, 101 ignored**. All 14 failure names match
  exactly; the list is retained in `results.json.regression`.
- Cargo metadata confirms the original nine binary target names/paths are
  unchanged. No model-specific executable was added.

Changes are localized to the model/converter/oracle, necessary architecture
registration, existing TTS CLI/server wiring and docs/CI. No other model's
neural implementation or shared SIMD/GPU kernel changed. The HTTP WAV/PCM
serialization was extracted into a common helper because PiVo and Woof use
the same response contract. Real PiVo Pi/Vo `Hello world.` CLI and HTTP outputs
match byte-for-byte after extraction; hashes are in `results.json.regression`.

Self-review corrected the missing exhaustive server backend label arm, checked
request/route isolation and retry semantics, retained upstream code/derivative
licensing, and fixed the interface checker to finish negative/concurrency tests
without hiding the numerical failure. Documentation now names both acceptance
failures. No subjective listening or broad perceptual evaluation was done.

Remaining work before marking complete:

1. Shared compute/Vulkan follow-up: diagnose the failing NAVI31 initialization
   path, then rerun the shared control, all nine strict-device numeric/interface
   rows, Auto device-execution evidence and completed GPU performance timings.
2. Woof model follow-up: establish task-level quality acceptance for long natural
   Mira generation using a broader fixed corpus; preserve the recorded strict
   waveform failure and distinguish reduction sensitivity from audible errors.
3. Check subjective and perceptual quality before recommending low-bit exports.

The upstream non-commercial license covers code and derivatives as well as
weights. [LICENSE](licenses/LICENSE) and [NOTICE](licenses/NOTICE) are retained;
applications/audio require the specified attribution. See the model guide for use.
