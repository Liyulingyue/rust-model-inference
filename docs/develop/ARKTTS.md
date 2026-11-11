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

Reproduce the matrix from the repository root:

```sh
MODEL_DIR=/Users/gouzi/Documents/git/rust-model-inference/models/Audio8-TTS-Realistic-Air
cargo test --locked --profile release-fast --features vulkan,parity-trace --lib arktts
.venv/bin/python -m unittest tools.oracle.arktts.test_same_weight \
  tools.oracle.arktts.test_compare tools.converter.arktts.test_convert_arktts
```

The complete evidence is kept outside Git under `MODEL_DIR/multiq/`; weights are intentionally not
part of this repository. `tools/oracle/arktts/performance.json` contains the same-host timing
observations. They use `release-fast`, four threads and four frames, include model/server loading,
and are not a release-throughput promise. A release build and longer-context performance run remain
separate work.
