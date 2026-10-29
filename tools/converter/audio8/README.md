# Audio8-ASR-Infinite GGUF

The converter packs the supplied merged-v2 safetensors without changing BF16
tensor bytes. It includes the eight semantic VAD tensors even though the ASR
CLI does not use them. The source index has 938 tensors; `--verify` compares
every GGUF tensor byte against its source shard.

```bash
MODEL=models/Audio8-ASR-Infinite
GGUF="$MODEL/Audio8-ASR-Infinite-BF16.gguf"
./.venv/bin/python -m tools.converter.audio8.convert_audio8 "$MODEL" "$GGUF"
./.venv/bin/python -m tools.converter.audio8.convert_audio8 "$MODEL" "$GGUF" --verify
RMI_AUDIO8_GGUF="$GGUF" cargo test --profile release-fast --test audio8_gguf
cargo build --profile release-fast --bin rust-model-inference
ffmpeg -nostdin -y -hide_banner -loglevel error -ss 10 \
  -i "$MODEL/Audio8-Asr-Infinite-Demo.mp4" -t 4 -vn -ac 1 -ar 16000 \
  -c:a pcm_s16le /tmp/audio8-demo-10-14s.wav
target/release-fast/rust-model-inference --model "$GGUF" \
  --audio /tmp/audio8-demo-10-14s.wav --language zh --threads 4 --max-tokens 16
```

For the WAV above, extract seconds 10–14 of the supplied demo video as mono,
16 kHz PCM16 with `ffmpeg`. The Rust CLI and the pinned official implementation
(`c8ba8eea829be0339e8d7757f8ca52dac06e1e32`) both returned
`一个两个半年，`. This is a transcription smoke check, not bit parity.

## Quantized variants

The BF16 file is the lossless reference. The shared quantizer reads it and
writes the smaller formats; the runtime loads all of them through the same
`Weight::from_quantized` path, so no separate loader is involved.

```bash
for fmt in q8_0 q6_k q4_k; do
  ./.venv/bin/python -m tools.converter.utils.quantize_gguf \
    "$MODEL/Audio8-ASR-Infinite-BF16.gguf" \
    "$MODEL/Audio8-ASR-Infinite-$(echo "$fmt" | tr a-z A-Z).gguf" \
    --format "$fmt" --keep frame_len_embedding.weight
done
```

`--keep frame_len_embedding.weight` is required: the runtime reads that
tensor as a float vector, so it cannot be block-quantized. Everything else is
handled by the rules in `tools/converter/utils/quantize_gguf.py` — 1-D tensors
(norms, biases), the 3-D convolution kernels (which the audio tower decodes
by hand, so they stay BF16) and any tensor whose contiguous row width is not
block-aligned keep their source type.

Measured on this repository's runtime, 4 s of the demo clip, GB10 / aarch64,
`--threads 4`, machine otherwise idle:

| Format | Size | Peak RSS | Export | Wall clock | Transcript |
|---|---|---|---|---|---|
| BF16 | 8.18 GB | 8.2 GB | 14 s | 50 s | `就是两个半月，` |
| F16 | 8.18 GB | 8.2 GB | 2.5 min | 44 s | `就是两个半月，` |
| Q8_0 | 4.35 GB | 4.5 GB | 25 s | 20 s | `For two and a half years` |
| Q6_K | 3.37 GB | 3.5 GB | 6 min | 64 s | `For two and a half years` |
| Q4_K | 2.32 GB | 2.5 GB | 15 min | 11 s | `就是一个二十二年` |

F16 keeps the transcript bit-for-bit equivalent to BF16 (the audio embeddings
agree to 1.3e-6 relative RMS, against 7e-2 for Q8_0) and is 15 % faster,
because the BF16 kernel has no NEON path on this host while F16 does. F16 is
the same size as BF16, so it is a speed knob, not a memory knob.

The block formats change the transcript, and they change it in different ways:
Q8_0 and Q6_K answer in English, Q4_K answers in Chinese with the wrong
content. The reason is the runtime's Q8_0 kernel path: it quantizes the
*activations* to Q8_0 and accumulates in int32, the llama.cpp convention. The
Audio8 activations are heavy-tailed, so that rounding is a few percent per
projection, the 32 audio layers compound it, and the greedy decode flips the
first token. The weight rounding alone is not the cause — the Q8_0 weights
dequantize to 0.5 % relative RMS of the BF16 ones, and F16 (which keeps F32
activations) reproduces the BF16 transcript exactly.

The transcripts above are for the WAV this repository's own extraction
produced; the README's reference clip came out of `ffmpeg`, and the two cuts
are not the same samples, so the wording differs from the official run's
`一个两个半年，` without either side being wrong.

| Artifact | SHA-256 |
|---|---|
| `model.safetensors` | `2cf97d69e9f5853855b783b359dffb661c035281a02053782c21fe009e31c7ba` |
| `semantic_vad_heads.safetensors` | `d1ff79e0282ef53aae176a42b1c2491815f8d26ef10a1e453dd2e41832adb7f7` |
| `Audio8-ASR-Infinite-BF16.gguf` | `1cfe353b4aa074cc3951385052b7e3a074152db4b1d20932fbab1e13195e17c4` |

The native C scalar reference in `tools/oracle/audio8/conv_scalar.c` decodes
the original BF16 safetensors and builds with FP contraction and vectorization
disabled. On Darwin arm64, with `RMI_SCALAR=1` and `--features parity-trace`, the first 32 real
normalized Mel frames produce four audio/text groups. The converter's
two convolution outputs, the first two audio frames' 32 layer outputs, four
audio projections, all four text tokens' 36 layer outputs, and all four
151,936-value logits arrays match the scalar reference **bit for bit** (257
compared checkpoints). The four-token check exercises the dedicated Audio8 text
decoder and F32 KV continuation. The generic Qwen3 loader rejects Audio8.
No SIMD, FMA, BLAS, Accelerate, or MPS path is used in this comparison.

To reproduce the four-token check after tracing the WAV CLI in scalar mode:

```bash
cargo build --profile release-fast --features parity-trace --bin rust-model-inference
RMI_SCALAR=1 RMI_PARITY_TRACE=/tmp/audio8-cli.jsonl \
  target/release-fast/rust-model-inference --model "$GGUF" \
  --audio /tmp/audio8-demo-10-14s.wav --language zh --threads 4 --max-tokens 16
./.venv/bin/python - <<'PY'
import json
import numpy as np
records = (json.loads(line) for line in open('/tmp/audio8-cli.jsonl'))
entry = next(record for record in records if record['name'] == 'asr.normalized_mel')
frames = entry['shape'][1]
mel = np.fromfile(entry['binary_path'], dtype='<f4').reshape(128, frames)
mel[:, :32].T.copy().tofile('/tmp/audio8-first32-mel.f32')
PY
RMI_SCALAR=1 RMI_PARITY_TRACE=/tmp/audio8-four-token.jsonl \
  RMI_AUDIO8_GGUF="$GGUF" RMI_AUDIO8_MEL=/tmp/audio8-first32-mel.f32 \
  cargo test --profile release-fast --features parity-trace --lib \
  models::audio8::tests::trace_real_audio_group
./.venv/bin/python tools/oracle/audio8/compare_conv.py "$MODEL" \
  /tmp/audio8-four-token.jsonl --mel-f32 /tmp/audio8-first32-mel.f32
```

The normalized Mel file is the comparison input. The WAV-to-Mel frontend has
not been independently checked for scalar raw-bit parity; the 16-step greedy
WAV transcription is a separate end-to-end check. With `RMI_SCALAR=1` and
`RMI_PARITY_FILTER=audio8.generated_ids`, the dedicated decoder produces IDs
`[151666,14777,18947,151665,151665,151666,77540,18947,99369,151665,151665,151665,151665,151666,7948,3837]`,
identical to the earlier scalar run. The CPU path explicitly
rejects audio beyond 1500 tower frames because rolling RoPE rebasing has not
been implemented.
