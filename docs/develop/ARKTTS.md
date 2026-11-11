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
Q4_1 also retains the codec's `quantizer.post_module.*` decoder transformer in F32. Its main
model and 60 encoder projections remain Q4_1; this avoids a waveform-gate failure on x86.

Vulkan is deliberately hybrid. Dense projections submit to the existing Vulkan matmul kernels;
norms, RoPE, attention/KV state, causal convolutions, codebook search and sampling remain on the
host. `cpu` disables GPU matmul, `auto` uses the device when available and falls back to CPU, and
strict `vulkan` requires a usable device. A strict request never silently falls back. The code does
not claim that the whole ArkTTS graph is on the GPU.

## Verification

On 2026-11-11, both an Apple M3 Max (ARM64) and an AMD EPYC 9334 / RADV NAVI31 host
(Linux x86_64) ran all nine formats across CPU, Auto and strict Vulkan with four inference threads.
The held-out suite compared 1053 fixed-history checkpoints against independent gguf-py
dequantization and the unmodified official Python model. Every format/backend passed its frozen
precision profile; strict Vulkan recorded 4308 dense device calls per run. The real-entry suite
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
Evidence is kept outside Git under `MODEL_DIR/multiq/` and `MODEL_DIR/remote-20261111/`;
the corrected Q4_1 pair uses `MODEL_DIR/multiq-reviewed/q4_1/`. Weights are intentionally not in Git.
`tools/oracle/arktts/performance.json` separates request-level observations from phase timings
that exclude loading and trace I/O. All timings use `release-fast`, four threads and four frames.
A release build and longer-context performance run remain separate work.

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

## Linux same-host observations

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
