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

Vulkan is deliberately hybrid. Dense projections submit to the existing Vulkan matmul kernels;
norms, RoPE, attention/KV state, causal convolutions, codebook search and sampling remain on the
host. `cpu` disables GPU matmul, `auto` uses the device when available and falls back to CPU, and
strict `vulkan` requires a usable device. A strict request never silently falls back. The code does
not claim that the whole ArkTTS graph is on the GPU.

## Verification

On 2026-11-11, an Apple M3 Max with four threads ran all nine formats across CPU, Auto and strict
Vulkan. The held-out suite compared 1053 fixed-history checkpoints against independent gguf-py
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
Evidence is kept outside Git under `MODEL_DIR/multiq/`; weights are intentionally not in Git.
`tools/oracle/arktts/performance.json` separates request-level observations from phase timings
that exclude loading and trace I/O. All timings use `release-fast`, four threads and four frames.
A release build and longer-context performance run remain separate work.

Same-host generation phase (seconds, median over the three four-frame texts):

| Format | CPU | Auto | Vulkan | Official CPU |
| --- | ---: | ---: | ---: | ---: |
| F32 | 2.861 | 3.490 | 3.256 | 0.427 |
| BF16 | 1.403 | 1.943 | 2.830 | 0.623 |
| F16 | 0.428 | 0.545 | 0.911 | 1.004 |
| Q8_0 | 0.179 | 0.421 | 0.792 | 1.192 |
| Q4_0 | 0.808 | 0.453 | 0.743 | 0.504 |
| Q4_1 | 0.209 | 0.303 | 0.771 | 0.580 |
| Q4_K | 0.999 | 1.543 | 2.093 | 0.457 |
| Q5_K | 1.023 | 1.547 | 2.091 | 0.456 |
| Q6_K | 1.087 | 1.547 | 2.102 | 0.453 |

These are single-pass observations, not isolated throughput trials. Generation excludes model loading
and trace I/O but includes Rust executor setup/device uploads. The Oracle uses the same GGUF
weights and activation-QDQ contract on CPU; its F16 reduction accumulates more widely.
The JSON also records matched fixed-code decode and reference encode phases. Per-projection
synchronization and host attention/codec work can make this hybrid Vulkan path slower than CPU.
Resources were not isolated; some Oracle timing overlapped compilation.
