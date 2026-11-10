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
| Multiweight/quantization | Published single 36.8 MiB F32 file; no tokenizer/mmproj/component | F16/BF16, GGUF and Q8_0/Q4_K are unimplemented; unsupported dtypes fail explicitly. This remains a partial adaptation; follow-up layers are described in README. |
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

Not checked in the initial 2026-11-04 run: other hosts/architectures, Vulkan execution, unprovided weight formats, subjective listening, PESQ/STOI or other quality benchmarks, performance/RAM under sustained load, 10-minute or eight-channel real inference. Follow-up evidence is below. Those limits and the unresolved independent phase sensitivity are not covered by the passing component checks.


## Review fixes and main integration — 2026-11-09

Merged main through `b05ee343997564c48a8b84a18563b9df5a8bcc6f`, including #208's shared audio frontend migration. The additive FFT methods now live in `src/ops/audio/frontend.rs`; all three RE-USE imports use it. The merge preserves AuK, System One and other new main routes. Enhancement and System One validation remain separate blocks.

The enhancement parser now checks an explicit allowed set of parsed switches, using main's existing argument-index tracking. Flag-shaped values do not select a mode, unsupported/default-valued options and unknown future switches fail, and mixed System One/enhancement flags fail in either order. The structural validator still protects direct `CliOptions` callers. No binary target was added or removed (Cargo metadata comparison).

Fresh macOS checks:

- `cargo +1.98.1 check --locked --all-targets`: passed, including the latest main test updates.
- `cargo +1.98.1 fmt -- --check`: passed.
- `cargo +1.98.1 check --locked --features vulkan --all-targets`: passed; this checks compilation, not model GPU execution.
- Main baseline `532d051a`: **1501 passed / 13 failed / 96 ignored**. The `src/` trees of `532d051a` and `b05ee343` are identical; the latter adds only integration-test updates.
- Merged branch library: **1506 passed / 13 failed / 97 ignored**. Compared failure-name sets: no added or removed failure.
- Enhancement CLI integration: **2 passed**; existing System One integration: **6 passed**. The existing x86 CI job now runs RE-USE contract/FFT tests and the enhancement CLI integration tests.
- Normal-build real-weight official-spectral test: **1 passed**; FFT/STFT, malformed input/weights and strict >50% suppression boundary tests passed.
- Ordinary scalar-mode checks also ran for noisy 8 kHz and speech 44.1 kHz; the same independent-phase limitation remains.
- Ordinary-build independent waveform recheck: CPU/Auto bytes match for all 11 cases; all 22 WAV files also match the original PR's ordinary-build outputs byte for byte. The original strict-parity failures reproduce, including `chunked_8000` RMSE `0.00345145` / relative `42.07%`.
- HTTP CPU/Auto recheck: response equals CLI; 400/404/413 and 200/429 concurrency cases passed.
- Quality fixture generation unit check: **1 passed**, including deterministic PCM16 pairs, duration and requested SNR.

### Published suppression provenance and impact

The pinned official drivers **both** suppress sweep artifacts before ISTFT. `inference.py` SHA256 is `71ccef1b231bd30c85b1cddd85266ef9cdc3216dc85db65e38f89a670ae28236`; `inference_chunk.py` is `df0fd3e983536f1a49440ab61ec7fb5cea2e878c58ea1aee15e182769646db60`. `reference.py` now verifies both and the README download command includes them. The postprocess is absent from the standalone official ISTFT helper, which explains why checking that helper alone misses it.

Rust now applies this as named model postprocessing, with the same strict `> 0.5` test; generic inverse STFT only performs synthesis. With finite F32 log magnitudes, `log_mag <= 0` is equivalent to a zero `expm1(relu(log_mag))` bin. Exactly half the bins does not suppress a frame.

Ablating this on the stored **official** pre-suppression magnitude/phase traces (raw per-group [results](sweep_ablation.json)) from `oracle-pcm` changes 1 of the 802 frames across the two `chunked_8000` chunks. In its first chunk, removing suppression changes max waveform amplitude by `0.0048201382` and RMSE by `0.0002236692`; its second chunk and every other original fixture are unchanged. These are same-spectrum, single-chunk counterfactual errors, before overlap, not a second end-to-end parity result. Quality-pair ablation results are recorded separately below.

### Paired clean/noisy speech quality

Method, fixed clip hashes, dependencies, encoding and reproduction are in [README.md](README.md) and `quality.py`. PESQ/STOI compare each output to its clean target, rather than treating official enhanced audio as clean speech. The official CPU-reference path computes independent frontend features using `selective_scan_ref`; this is not a CUDA-driver validation. Absolute quality-delta limits were set to PESQ `0.05` and STOI `0.005` before running the evaluation; strict waveform thresholds remain unchanged.


All **12/12** cases passed the predeclared quality-delta checks. CPU/Auto WAV bytes were identical in all 12 cases. Unrounded per-case values, hashes, versions and timings are in [quality_results.json](quality_results.json).

Maximum native-minus-official absolute differences: **PESQ 0.000920534**, **STOI 0.000050456**. Independent waveform relative RMSE ranged from 0.1110% to 0.3076%.

Each score cell lists **noisy / official CPU reference / native**. PESQ is NB at 8 kHz and WB at 16 kHz; STOI is the standard non-extended version.

| Speaker clip | Rate | Input SNR | PESQ | STOI |
|---|---:|---:|---|---|
| 5703-47212-0000 | 8000 | 0 dB | 1.156327 / 1.871443 / 1.870522 | 0.603383 / 0.795782 / 0.795732 |
| 5703-47212-0000 | 8000 | 10 dB | 1.372032 / 2.767314 / 2.767804 | 0.776135 / 0.898066 / 0.898078 |
| 5703-47212-0000 | 16000 | 0 dB | 1.029784 / 1.530286 / 1.530437 | 0.670282 / 0.834694 / 0.834691 |
| 5703-47212-0000 | 16000 | 10 dB | 1.051469 / 2.063978 / 2.064072 | 0.821523 / 0.912565 / 0.912597 |
| 3436-172162-0000 | 8000 | 0 dB | 1.374700 / 2.638001 / 2.637957 | 0.784454 / 0.892242 / 0.892237 |
| 3436-172162-0000 | 8000 | 10 dB | 1.705046 / 3.616263 / 3.616065 | 0.918770 / 0.968520 / 0.968515 |
| 3436-172162-0000 | 16000 | 0 dB | 1.070022 / 1.908485 / 1.907914 | 0.826507 / 0.918169 / 0.918134 |
| 3436-172162-0000 | 16000 | 10 dB | 1.181918 / 3.160806 / 3.160642 | 0.941002 / 0.982915 / 0.982917 |
| 198-209-0000 | 8000 | 0 dB | 1.232476 / 2.287085 / 2.287308 | 0.709167 / 0.888483 / 0.888521 |
| 198-209-0000 | 8000 | 10 dB | 1.508688 / 3.212567 / 3.212857 | 0.860202 / 0.931608 / 0.931624 |
| 198-209-0000 | 16000 | 0 dB | 1.034591 / 1.621385 / 1.620698 | 0.754241 / 0.913353 / 0.913368 |
| 198-209-0000 | 16000 | 10 dB | 1.137330 / 2.536164 / 2.535388 | 0.903023 / 0.959836 / 0.959834 |

The paired speech cases suppress **0 of 9624** official frames. Their enabled/disabled ablation waveforms and quality scores are identical; the synthetic chunk fixture above supplies a triggered example with a measurable effect.

These results support negligible PESQ/STOI differences **on these three speakers and stationary-noise pairs**. They do not demonstrate general perceptual equivalence, independent strict-waveform parity, CUDA parity, other codecs/rates/noise types or listening quality. The original synthetic failures remain visible and the model stays **Experimental / Draft**.


## Linux/x86_64 follow-up — 2026-11-09

The replacement host completed native inference and independent official-reference measurements. Host: Linux 6.8.0-79-generic, glibc 2.39, AMD EPYC 9334; 128 visible CPUs with a shared 16-core cgroup quota and 55 GiB memory limit. Native source snapshot `bfa8564e33a716ed2dc0b08d8f0ef4942adbc503` was checked against all 1,567 tracked file hashes. [Environment and binary hashes](environment_linux.json) pin this run.

The installed Rust 1.94.1 cannot compile main's `if let` guard in `src/prompt/jinja_compat.rs`. Rust 1.98.1 was installed inside this workspace's ignored `target/rustup`; Cargo cache, Python dependencies and build output also remained workspace-local. No source change or compiler bootstrap flag was used to bypass that error. The normal `release-fast` CLI/server build succeeded without `parity-trace`.

Confirmed Linux checks:

- RE-USE model tests: **5 passed / 0 failed / 1 ignored**, including config/weight contracts, WAV validation, the strict suppression boundary and FFT roundtrips.
- `cargo check --offline --locked --all-targets`: passed. Enhancement CLI integration: **2 passed**; existing System One integration: **6 passed**.
- Ordinary-build real-weight official-spectral test: **1 passed**, all 13 chunk/channel groups; worst waveform max absolute error `5.35906e-6`, worst RMSE `1.36356e-6`. This remains conditional on identical frontend features.
- Deterministic paired-speech fixture check: **1 passed**, including PCM16 length, repeatability and requested SNR.
- Independent waveform measurements: **11 cases × CPU/Auto**; output metadata/frame counts and CPU/Auto WAV byte identity passed. The reference generated its own FFT features and official CPU network outputs on this host.
- HTTP CPU/Auto: mixed-case MIME and HTTP/CLI byte equality passed; malformed WAV, rate/channel/length/MIME errors returned 400; unsupported routes returned 404; size cap returned 413; concurrency returned 200/429. [Raw HTTP results](http_results_linux.json).

The unchanged strict waveform gate still does **not** pass. The original chunked fixture has relative RMSE **40.12%** on Linux, versus **42.07%** on the earlier Mac run. Different host FFT/rounding results do not eliminate the raw-phase branch sensitivity. [All 22 ordinary-build records](waveform_results_linux.json):

| Case | max_abs | RMSE | Relative RMSE |
|---|---:|---:|---:|
| noisy_8000 | 0.009835502 | 0.002588931 | 24.24% |
| noisy_16000 | 0.01295383 | 0.002604313 | 29.08% |
| noisy_22050 | 0.006664528 | 0.001799992 | 36.36% |
| noisy_24000 | 0.001728377 | 0.00044908 | 9.26% |
| noisy_32000 | 0.004643316 | 0.001025582 | 26.93% |
| noisy_44100 | 0.001583444 | 0.0003998744 | 12.11% |
| noisy_48000 | 0.001028402 | 0.0002295246 | 7.73% |
| silence_8000 | 3.292406e-05 | 1.748602e-05 | 0.25% |
| speech_44100 | 0.0001646769 | 3.16727e-05 | 2.19% |
| stereo_16000 | 0.01295383 | 0.001899572 | 23.95% |
| chunked_8000 | 0.06798412 | 0.003266162 | 40.12% |

Linux paired-speech quality: **12/12 cases** passed the unchanged absolute delta limits (PESQ `0.05`, STOI `0.005`); CPU/Auto WAV bytes matched for every case. Maximum native-minus-official absolute deltas were **PESQ `0.000941038`** and **STOI `0.0000354976`**. Independent speech waveform relative RMSE was **0.1207%–0.3196%**. All per-case scores, input/source/binary/script hashes, versions and timings are in [quality_results_linux.json](quality_results_linux.json). Suppression triggered **0/9624** official frames, so its enabled/disabled ablation outputs and quality scores were identical for these pairs.

The official CPU path uses Python 3.12.3 / Torch 2.9.1+gitff65f5b. The same `quality.py` recipe ran as six independent speaker/rate groups via repeated `--case` switches, two SNR cases per group, with one Torch thread and four native CLI threads per process. Group reports were combined only after asserting identical metadata/binary hashes, exact 12-case coverage and both modes' fixed quality limits. These jobs shared the cgroup with other work; recorded timings are validation timings, not isolated performance benchmarks. Neither the bounded speech-quality result nor passing same-feature checks removes the independent synthetic-waveform failure.

Cross-host fixture audit: source OGG hashes, NumPy 2.5.4, SciPy 1.17.0, SoundFile 0.13.1 and libsndfile 1.2.2 match, but platform-dependent OGG decode rounding changes a few PCM16 samples after pairing. Across all 12 pairs, clean WAVs differ at **60/432,000** samples and noisy WAVs at **75/432,000**, with maximum difference **1/32768** in either set. Decoded OGG crops already differ by at most `1.78813934e-7`; supplying those remote decoded crops to the local pairing code reproduces all 24 remote PCM payloads exactly. Each host records its own clean/input hashes and compares native/oracle against the same local PCM. Cross-host PCM bit identity is not an acceptance gate for this lossy decode path.

These Linux checks do not establish GPU, CUDA-driver, quantized/other-dtype, sustained-load or general perceptual equivalence. No full Linux library baseline comparison was run in this follow-up; the macOS full-suite comparison and prior CI evidence above remain separate.

## Main integration — 2026-11-10

Merged upstream main `0ea4e3837611325b58f22b7457b039cfeaecf176` into branch head `79b4bd5541bc87ab6edde19cf8d6e4b5741aca33`. Resolutions preserve both enhancement and THX-02 help, dispatch and model-list entries, main's `supplied_flags` tracking, and independent enhancement/decision/System One validation blocks. Direct `CliOptions` callers now also reject enhancement combined with either new decision field; the added regression failed before that fix and passes afterward. RE-USE model and shared FFT arithmetic are unchanged by this merge.

Fresh macOS arm64 checks with Rust 1.98.1 (`RMI_SCALAR` and `RMI_PARITY_TRACE` unset):

- Default and `--features vulkan` `cargo check --locked --all-targets`: passed; Vulkan execution remains unverified.
- `cargo fmt -- --check`: passed.
- `cargo test --locked --profile release-fast --lib`: before merge **1506 passed / 13 failed / 97 ignored**, after merge **1531 passed / 13 failed / 99 ignored**. The exact failure-name sets are identical, with no added or removed failure.
- Enhancement CLI **3 passed**, System One CLI **6 passed**, decision CLI **9 passed** using the same test profile.
- Ordinary-build official-spectral real-weight test: **1 passed**, all 13 chunk/channel groups; worst waveform max absolute error `3.48221511e-6`, worst RMSE `9.05721563e-7`. This still requires identical frontend features.
- Vela3 Python oracle tests (`python3 -m unittest discover -s tools/oracle/vela3 -p 'test_*.py'`): **6 passed**, including main's fixes for the previously reported fixture failures.

Independent waveform/quality pairs, HTTP inference and remote/GPU execution were not rerun for this integration-only change. Their dated evidence and limits above are unchanged; strict independent waveform parity remains unresolved, and this PR stays Draft / Experimental.
