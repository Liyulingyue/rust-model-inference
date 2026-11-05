# Hy-SimulMT

Use the official single-file GGUF from
[Tencent's Hy-SimulMT collection](https://modelscope.cn/collections/Tencent-Hunyuan/Hy-SimulMT).
Both translation-only and Interleave models include the audio encoder; no
separate mmproj is needed.

```sh
model=models/Hy-SimulMT-30B-A3B-GGUF/Hy-SimulMT-30B-A3B-Q4_K_M.gguf
cargo build --release --bin rust-model-inference
target/release/rust-model-inference \
  --model models/Hy-SimulMT-30B-A3B-GGUF/Hy-SimulMT-30B-A3B-Q4_K_M.gguf \
  --audio speech.wav --language zh --temp 0 --max-tokens 512 \
  --threads 4 --compute cpu --kv-format f32 --max-context 8192
```

Input must be mono **16 kHz PCM16 WAV**. `--language` is the **target** language
(default Chinese); `zh`/`Chinese` and `en`/`English` are accepted aliases.
`--prompt` overrides the translation instruction, so its requested direction
should agree with the target language. Audio is processed in the model's 1.6 s
first turn and 0.8 s subsequent turns. The token limit applies to the whole
recording; a full KV context returns an error rather than silently dropping
history.

For Interleave, select `Hy-SimulMT-30B-A3B-Interleave-Q4_K_M.gguf`. Its metadata
selects the source-branch prompt automatically. Generated
`<｜hy_place▁holder▁no▁791｜>` separates source text from translated text.
The API returns token IDs and each turn's tokens in addition to rendered text.
CLI text is printed when the recording completes.

## Compute and architecture

Hy owns its configuration, weights, mixed dense/MoE decoder and KV session.
It does not enter the Qwen3 trunk. Audio frontends, normalization, attention,
linear/conv operations and ISA selection live in `ops`; the original model
paths re-export the moved helpers for compatibility.

`--compute cpu` uses the shared CPU kernels, including runtime AVX2/FMA or
NEON/DotProd dispatch with a scalar fallback. QKV and gate/up projections use
multi-weight groups and share activation preparation within each group.
ARM Q6_K weights also have a runtime-gated NEON DotProd path.

For Standard, build with `--features vulkan` and select `--compute auto` to offload
supported linear groups, audio projections and convolutions to the accelerator. Linear
groups below the shared output-size threshold stay on CPU. Attention,
normalization, routing and KV remain on the host. Auto falls back to CPU when
the device or a kernel is unavailable. `--compute vulkan` requests a complete
Vulkan decoder and is rejected for this hybrid model. Set `RMI_COMPUTE_TRACE=1`
to inspect actual backend choices and cumulative device counters.

Interleave Auto keeps the speech encoder's GPU offload and uses CPU for the
decoder, reporting the numerical-validation reason at startup. Its GPU decoder
logits still exceed the unchanged calibrated cosine bound. The CPU decoder
retains SIMD and multi-weight grouping. JEV is unsupported for both variants.

Auto is experimental. Earlier one-token M3 Max smoke measurements were 10.36 s
Auto / 48.74 s CPU, including setup; these are not controlled throughput results.
The local follow-up verifies complete three-turn outputs and actual device
submissions, without establishing a GPU speedup. Shared Vulkan exit cleanup and
validation-feature lifetimes have been repaired and tested on M3 Max. The
historical NVIDIA 570.153.02 exit-time SIGSEGV has not been retested.

## HTTP

Start the audio backend with the same model and generation options:

```sh
target/release/rust-model-inference --serve --model "$model" --audio speech.wav \
  --host 127.0.0.1 --port 8080 --compute cpu --temp 0 --max-tokens 512
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@speech.wav -F language=en
```

`POST /v1/audio/transcriptions_json` also accepts
`{"input":"<base64 WAV>","language":"en","prompt":"optional instruction"}`.
Both routes return `{"text":"..."}`. The request language is the target language;
omitting it uses the server's default. Only omitted/`json` response formats and
greedy temperature `0` are supported. Empty target languages are rejected.

Numerical evidence and reproducible checks for both Q4_K_M variants are recorded
in [the Oracle guide](../../tools/oracle/hy_simulmt/README.md). TTS, microphone
input, resampling and FP8/other quantizations remain outside this integration.
