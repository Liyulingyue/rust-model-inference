# ArkTTS Realistic Air

`Audio8-TTS-Realistic-Air` uses the dedicated `arktts` / `arktts_codec` pair. The model is
loaded only by the `--tts` capability and emits mono 44.1 kHz PCM16 WAV or raw PCM. The
main model contains the slow and fast DualAR transformers; the codec contains the encoder,
codebooks, causal convolutions and decoder.

The converter supports F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K. Each pair carries
the same bundle ID. Two dimensional dense projections use the shared `LinearExecutor` with
32-row staging chunks. Vectors, convolution weights, norms, biases and codebooks retain their
lossless source precision. Q4/Q5/Q6_K keep 896-wide rows in BF16 and quantize compatible 4864-wide
FFN down projections; codec resampling ConvNeXt projections remain F32 because activation QDQ
there can amplify small bin changes into waveform drift.
Q4_0 and Q4_1 also retain the codec's `quantizer.post_module.*` decoder transformer in F32.
Their main model and 60 encoder projections remain quantized; this avoids calibrated
waveform-gate failures on x86 CPU and M3 GPU without changing the shared kernels or tolerances.

Vulkan now runs the numerical graph on the device. Native-format projections reuse the shared
prepared matmul kernels. RMS/LayerNorm, BF16-rounded interleaved RoPE, F32 KV, causal/windowed
attention, residual/FFN updates, embedding lookup, top-k/top-p/Gumbel selection, codec causal/grouped
and transposed convolution, Snake/GELU/tanh and residual codebook search all execute on Vulkan.
`arktts-device` and `arktts-operators` logs report the graph, operator counts and transfer bytes;
`ComputeStats` supplies actual device submission counters. Intermediate activations and KV stay
in the arena; parity tracing deliberately adds checkpoint readbacks and reports those separately.

The host retains tokenization, request validation, seeded RNG, generation control and WAV encoding.
Immutable embedding lookup tables are decoded once to F32 during graph setup, while projection
weights retain their native GGUF storage and activation contract. `cpu` uses the existing SIMD
implementation, `auto` attempts each complete device graph and retries that stage on CPU after a
device failure, and strict `vulkan` returns the failure. Device allocations use checked arithmetic;
requests exceeding the device's storage-buffer/workspace or memory limits fail explicitly in strict
mode. This replaces the projection-only implementation described in the historical measurements below.

## Verification

On 2026-11-11, both an Apple M3 Max (ARM64) and an AMD EPYC 9334 / RADV NAVI31 host
(Linux x86_64) ran all nine formats across CPU, Auto and strict Vulkan with four inference threads.
The held-out suite compared 1053 fixed-history checkpoints against independent gguf-py
dequantization and the unmodified official Python model. Every format/backend passed its frozen
precision profile; Auto and strict Vulkan recorded resident generation, encode and decode graphs,
non-linear device operators and actual submissions with zero host-linear fallback. The real-entry suite
also passed CLI plain/clone and HTTP WAV/PCM plain/clone for all 27 combinations, with HTTP bytes
equal to the corresponding CLI output.

The numerical gates are tolerance based. F32/BF16 use relative RMSE `1e-4`, normalized max error
`5e-5`, logits max error `2e-3`, waveform max error `2e-4` and waveform RMSE `2e-5`. F16 uses
`0.01 / 0.01 / 0.25 / 0.001 / 0.0001`; block quantized formats use `0.2 / 0.15 / 5 / 0.006 /
0.001`. Lossy floating point and quantized paths are not expected to be bitwise equal.

Run the focused checks from the repository root:

```sh
MODEL_DIR=/Users/gouzi/Documents/git/rust-model-inference/models/Audio8-TTS-Realistic-Air
cargo test --locked --profile release-fast --features vulkan,parity-trace --lib arktts
.venv/bin/python -m unittest tools.oracle.arktts.test_same_weight \
  tools.oracle.arktts.test_compare tools.converter.arktts.test_convert_arktts
```

The [complete matrix commands](../../tools/oracle/arktts/README.md#完整矩阵复现) include conversion,
the same-weight Oracle, fixed-history traces, free-running greedy, projections, and CLI/HTTP.
Current evidence is kept outside Git under `MODEL_DIR/device-20261111-final-macos/`,
`MODEL_DIR/entries-device-final-macos/` and the corresponding Linux archive in
`MODEL_DIR/device-20261111-remote/`. The corrected Q4_0 pair uses `MODEL_DIR/multiq-device/q4_0/`;
Q4_1 uses `MODEL_DIR/multiq-reviewed/q4_1/`. Weights are intentionally not in Git.
`tools/oracle/arktts/performance.json` separates request-level observations from phase timings
that exclude loading and trace I/O. All timings use `release-fast`, four threads and four frames.
A release throughput benchmark remains separate work.

## Resident-graph same-host observations

Same GGUF and prompts, three four-frame generation cases, three fixed decode frames and 6145 encode samples. No parity trace I/O; Rust phases include graph setup and static uploads. CPU code is unchanged by the final GPU submission bound and retains this round's previous CPU evidence. Same-host Oracle CPU timings reuse the explicitly matched no-trace phase observations from earlier on the same date, except fresh Q4_0/Q4_1 fixtures where available. Resources were shared; these are observations, not release-throughput benchmarks.

### Apple M3 Max

Generation seconds (median of three texts, each four frames):

| Format | CPU | Auto | Vulkan | Same-weight Oracle CPU |
| --- | ---: | ---: | ---: | ---: |
| F32 | 3.178 | 2.861 | 1.477 | 0.427 |
| BF16 | 1.064 | 1.037 | 1.001 | 0.623 |
| F16 | 0.419 | 0.967 | 1.418 | 1.004 |
| Q8_0 | 0.216 | 0.752 | 0.901 | 1.192 |
| Q4_0 | 0.843 | 0.658 | 0.728 | 0.586 |
| Q4_1 | 0.224 | 0.657 | 0.700 | 0.549 |
| Q4_K | 0.833 | 0.750 | 0.772 | 0.457 |
| Q5_K | 0.831 | 0.761 | 0.762 | 0.456 |
| Q6_K | 0.847 | 0.831 | 0.962 | 0.453 |

Codec phase times are recorded in `performance.json`. Projection setup/uploads and short sequences
can dominate GPU time; attention, convolution and other numerical operators no longer fall back to CPU.
The reference here is the official CPU implementation with independent same-weight dequantization/QDQ.

### AMD EPYC 9334 / RADV NAVI31

Generation seconds (median of three texts, each four frames):

| Format | CPU | Auto | Vulkan | Same-weight Oracle CPU |
| --- | ---: | ---: | ---: | ---: |
| F32 | 0.638 | 2.475 | 2.504 | 7.077 |
| BF16 | 0.426 | 1.683 | 1.753 | 7.145 |
| F16 | 0.544 | 2.458 | 2.450 | 7.007 |
| Q8_0 | 0.338 | 1.305 | 1.367 | 7.227 |
| Q4_0 | 0.432 | 1.362 | 1.368 | 6.936 |
| Q4_1 | 0.367 | 1.370 | 1.378 | 7.201 |
| Q4_K | 0.374 | 1.645 | 1.725 | 6.885 |
| Q5_K | 0.359 | 1.714 | 1.752 | 7.324 |
| Q6_K | 0.322 | 1.877 | 1.869 | 7.034 |

Codec phase times are recorded in `performance.json`. Projection setup/uploads and short sequences
can dominate GPU time; attention, convolution and other numerical operators no longer fall back to CPU.
The reference here is the official CPU implementation with independent same-weight dequantization/QDQ.


## Historical projection-only observations

The following tables preserve the `c4b07899e` implementation and do not describe the current graph.

Apple M3 Max generation phase (seconds, median over the three four-frame texts):

| Format | CPU | Auto | Vulkan | Same-weight Oracle CPU |
| --- | ---: | ---: | ---: | ---: |
| F32 | 2.861 | 3.490 | 3.256 | 0.427 |
| BF16 | 1.403 | 1.943 | 2.830 | 0.623 |
| F16 | 0.428 | 0.545 | 0.911 | 1.004 |
| Q8_0 | 0.179 | 0.421 | 0.792 | 1.192 |
| Q4_0 | 0.808 | 0.453 | 0.743 | 0.504 |
| Q4_1 | 0.174 | 0.283 | 0.729 | 0.549 |
| Q4_K | 0.999 | 1.543 | 2.093 | 0.457 |
| Q5_K | 1.023 | 1.547 | 2.091 | 0.456 |
| Q6_K | 1.087 | 1.547 | 2.102 | 0.453 |

These are single-pass observations, not isolated throughput trials. Generation excludes model loading
and trace I/O but includes Rust executor setup/device uploads. The Oracle uses the same GGUF
weights and activation-QDQ contract on CPU; its F16 reduction accumulates more widely.
The JSON also records matched fixed-code decode and reference encode phases. Per-projection
synchronization and host attention/codec work can make this hybrid Vulkan path slower than CPU.
Resources were not isolated; some Oracle timing overlapped compilation.

### Historical Linux same-host observations

AMD EPYC 9334, Linux 6.8.0-79-generic x86_64, AMD Radeon Graphics (RADV NAVI31),
four inference threads, container CPU quota 16 and memory limit 55 GiB. The host was shared
with other jobs; these are single observations during validation, not an isolated benchmark.
The phase boundaries, GGUF weights and inputs match the CPU/Auto/Vulkan/Oracle runs on this host.

| Format | CPU | Auto | Vulkan | Same-weight Oracle CPU |
| --- | ---: | ---: | ---: | ---: |
| F32 | 0.531 | 3.612 | 4.555 | 7.077 |
| BF16 | 0.482 | 3.118 | 4.822 | 7.145 |
| F16 | 0.611 | 3.920 | 4.807 | 7.007 |
| Q8_0 | 0.375 | 1.597 | 2.566 | 7.227 |
| Q4_0 | 0.571 | 2.160 | 2.277 | 7.290 |
| Q4_1 | 0.329 | 2.128 | 2.444 | 7.201 |
| Q4_K | 0.329 | 4.684 | 3.170 | 6.885 |
| Q5_K | 0.327 | 2.404 | 3.391 | 7.324 |
| Q6_K | 0.301 | 2.891 | 3.364 | 7.034 |

Times are seconds for the median of three four-frame generation cases. `performance.json`
also records three-frame codec decode and 6145-sample codec encode for every format/backend.
Rust generation includes executor setup and weight uploads; the same-weight CPU Oracle
uses independent dequantization and activation QDQ. The observed GPU path is slower on these
short cases: dense calls synchronize through host buffers and attention, convolution,
norm/RoPE, codebooks and sampling remain on CPU. No speedup or full-release claim is made.
