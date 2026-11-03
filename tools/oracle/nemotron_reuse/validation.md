# RE-USE validation — 2026-11-04

Host: macOS/Darwin 27.0.0 arm64, Rust 1.98.1, `release-fast`, CPU. Base commit `afed6d48b1886c10ce785b56f10d9427830a0681`. Oracle pins, dependency versions and reproduction commands are in [README.md](README.md) and `reference.py`.

**Result: Experimental.** Model arithmetic agrees on identical spectral features; independent end-to-end waveform equivalence has not passed. The model consumes phase angles directly, so FFT rounding at ±π is a material numerical limitation. The error below is retained as a failed strict-parity result, rather than hidden by an expanded waveform threshold.

## Measured coverage

The fixed-seed inputs are quantized to exact PCM16 before oracle generation. Cases cover seven standard rates (8/16/22.05/24/32/44.1/48 kHz), silence, a real speech excerpt from the pinned sample, stereo and a 16,813-frame (2.101625-second) 8 kHz input that exercises two overlapping chunks. The sample WAV SHA256 is checked by the reference generator. This is functional/numerical coverage, not a speech-quality dataset.

| Area | Evidence | Result |
|---|---|---|
| Tensor/config contract | Real F32 checkpoint; 1,416 tensor names/shapes/types/storage and finite values; config equality; missing/malformed dtype/shape/storage tests | Pass; other dtypes and GGUF are explicitly unsupported |
| CLI | Real default-build CPU/Auto, 11 cases × 2; same WAV metadata/frame count; mixed-task validation integration test | Pass; CPU/Auto WAV bytes equal in all 11 cases |
| HTTP | Real default-build server on loopback, CPU and Auto; mixed-case MIME, response equals CLI; malformed WAV, unsupported rate/channel, short audio, wrong MIME | Pass; valid 200, invalid 400 |
| HTTP limits/isolation | 64 MiB + 1 byte body; two simultaneous requests; chat and four text/image JEV routes | Pass; 413, one 200 + one 429, all unsupported routes 404 |
| Scalar/SIMD | Existing `dot_f32` dispatch, `RMI_SCALAR=1` versus normal ARM SIMD; noisy 8 kHz and speech 44.1 kHz in diagnostic and ordinary builds | Numerical checks pass; floating-point reductions are not required to match bits |
| CPU/Auto/Vulkan | Both CPU and Auto identify backend=Cpu; strict Vulkan rejected before loading | Pass; no GPU execution or GPU acceleration claimed |
| Multiweight/quantization | Published single F32 file only; no tokenizer/mmproj/component | Not applicable; no unprovided format added |
| Shared FFT | Only additive `RealFft::complex`/`inverse` methods; old transform/power/magnitude bodies and struct layout unchanged; FFT/STFT roundtrip at five rates | Roundtrip max error < 2e-6; existing callers retain their original implementation |
| Production network + ISTFT | Ignored real-weight Rust test, **without** `parity-trace`; all 13 chunk/channel groups supplied official STFT features | Pass; worst waveform max 3.48222e-6, worst RMSE 9.05722e-7 |
| Diagnostic network | 36 checkpoints/group, names/shapes/count/finite checks; CPU/Auto all cases, scalar two cases; official unmodified network supplied native STFT features | Pass within the stated component tolerances; conditional on identical frontend features |
| Independent end-to-end | Ordinary build versus official independent STFT/network/ISTFT, 11 cases × CPU/Auto | Measured drift below; strict waveform alignment remains unresolved |

`scalar_mode()` checks the process-wide `RMI_SCALAR` variable, including Rayon workers. Scalar dots use F32 products/F64 accumulation; SIMD dots use the existing NEON/FMA reduction. The scripts remove inherited scalar mode for CPU/Auto. `--waveform-only` rejects a binary that emits parity traces, preventing a traced build from being mislabeled as an ordinary build.

The repository's existing `parity-trace` feature changes shared LayerNorm rounding. Diagnostic checkpoint evidence therefore cannot stand in for production-build evidence. Separate ordinary-build waveform runs, HTTP checks and the spectral-input Rust test cover the normal path.

## Component tolerances and observed errors

No bitwise comparison is imposed on FFT, normalization, reductions, exponentials or trigonometry. Exact checks cover contracts, checkpoint structure, frame counts and CPU/Auto output identity.

- Frontend log-compressed complex spectrum: `max_abs <= 1e-3`; observed maximum 1.67341e-5. Circular phase error and raw ±π branch crossings are recorded separately; circular phase maximum was 0.00578935 radians near low-energy bins.
- Encoder/block/decoder magnitude: `max_abs <= 5e-4 + 3e-4 * reference_peak`, `RMSE <= 1e-4 + 1e-4 * reference_RMS`.
- Decoder phase is recorded as circular error. Its effect is checked through reconstructed real/imaginary spectra (`max_abs <= 5e-4`, `RMSE <= 1e-4`) and the reconstructed waveform (`max_abs <= 5e-4`, `RMSE <= 5e-5`). Raw angles at negligible amplitude are not a useful standalone pass criterion.
- The production spectral-input Rust test uses waveform `max_abs <= 5e-4`, `RMSE <= 5e-5`.

The initial uniform intermediate `max_abs <= 5e-4` check **failed** on silence at `reuse.block.1` (max 0.00200641, RMSE 3.80718e-5; reference peak 6.01654, relative RMS error 4.24578e-5). Low-variance normalization magnifies rounding there. The revised component criterion adds an explicit scale-relative bound; it does not change the independent waveform result.

Across the diagnostic CPU/Auto cases, worst encoder max was 4.32730e-5, worst Mamba-block max 0.00200641, worst magnitude max 1.03713e-5. Worst matched-feature waveform max was 3.48874e-6 / RMSE 9.07504e-7 (silence). On non-silent cases the original seven-rate checkpoint run's worst block max was 3.29018e-5. The reconstructed decoder spectrum's worst component max was 3.53944e-4 (silence).

## Independent waveform drift — ordinary build

Values compare decoded PCM16 CLI output to the clipped official F32 waveform. PCM16 scaling and truncation contribute a small encoding error; they do not explain the larger phase-driven errors. Relative RMSE is divided by reference output RMS, not input RMS. CPU and Auto results are identical.

| Case | max_abs | RMSE | Relative RMSE |
|---|---:|---:|---:|
| noisy_8000 | 0.0120251 | 0.00272179 | 25.43% |
| noisy_16000 | 0.00715289 | 0.00145393 | 15.71% |
| noisy_22050 | 0.00573196 | 0.00182085 | 37.00% |
| noisy_24000 | 0.00465103 | 0.00122062 | 24.36% |
| noisy_32000 | 0.00181361 | 0.000468163 | 12.30% |
| noisy_44100 | 0.00166506 | 0.000370093 | 11.02% |
| noisy_48000 | 0.00127528 | 0.000229436 | 7.72% |
| silence_8000 | 0.0000312041 | 0.0000174521 | 0.25% |
| speech_44100 | 0.000143847 | 0.0000323171 | 2.25% |
| stereo_16000 | 0.00715289 | 0.00108665 | 13.44% |
| chunked_8000 | 0.0705030 | 0.00345145 | 42.07% |

The first failed independent comparison led to inspection of the STFT checkpoints. Very small imaginary residuals near a negative real value select opposite `+π`/`-π` angles. Those angles describe nearly identical complex values, but are very different raw scalar inputs to this network. Supplying identical frontend features removes the large downstream discrepancy. The CPU fixtures contained 1,186 such raw branch crossings across all chunk/channel groups.

Hann generation was corrected to F32 window arithmetic using the existing SLEEF cosine helper; this improved frontend agreement but did not eliminate the branch sensitivity. Canonicalizing phases, adding dither or replacing phase by sine/cosine would change the published input semantics. No such model change or bespoke Torch FFT emulation was introduced. `parity.py` records independent waveform metrics without declaring a pass. The largest remaining error is material and keeps this adaptation Experimental/Draft.

## Regression and review

Executed:

```sh
cargo +1.98.1 test --locked --profile release-fast --lib
cargo +1.98.1 test --locked --profile release-fast --test nemotron_reuse_cli
cargo +1.98.1 build --locked --profile release-fast --bin rust-model-inference --bin rust-model-server
```

- Before: **1330 passed / 20 failed / 88 ignored**.
- After: **1334 passed / 20 failed / 89 ignored**. Failure names were compared as sets: no new failure and no resolved baseline failure.
- CLI integration: **1 passed**, covering explicit enhancement and incompatible task/payload options.
- Opt-in spectral real-weight test: **1 passed**, all cases/chunks/channels.
- Ordinary and diagnostic binaries built; new Rust files passed targeted rustfmt checks; Python scripts passed compilation; diff whitespace checked.

The unchanged baseline failure groups are Breeze dtype/source tests (3), CLM golden/width tests (2), Z-Image NEON softmax/SiLU tests (3), FunASR FSMN/LayerNorm parity tests (2), Gemma4 BF16/chunk tests (4), argmax tests (5), and NEON softmax reduction (1). This is not a green full library suite.

Self-review plus an independent scoped review fixed silent acceptance of incompatible TTS/ASR/JEV payload flags and case-sensitive HTTP MIME matching. Follow-up review found and fixed two validation gaps: ordinary-build checks now reject emitted traces, and HTTP checks clear inherited scalar/trace environment variables. Review of the published graph, chunk schedule and all 40,001 accepted integer-rate shape bounds found no additional structural defect.

Not checked: other hosts/architectures, Vulkan execution, unprovided weight formats, subjective listening, PESQ/STOI or other quality benchmarks, performance/RAM under sustained load, 10-minute or eight-channel real inference. Those limits and the unresolved independent phase sensitivity are not covered by the passing component checks.
