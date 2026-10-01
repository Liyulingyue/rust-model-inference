# Audio8-ASR-Infinite BF16 GGUF

The converter packs the supplied merged-v2 safetensors without changing BF16
tensor bytes. It includes the eight semantic VAD tensors even though the ASR
CLI does not use them. The source index has 938 tensors; `--verify` compares
every GGUF tensor byte against its source shard.

```bash
MODEL=/Users/gouzi/Documents/git/rust-model-inference/models/Audio8-ASR-Infinite
GGUF="$MODEL/Audio8-ASR-Infinite-BF16.gguf"
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
