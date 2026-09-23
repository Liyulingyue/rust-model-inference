# Jina v5 Omni Small Retrieval audio 对照

固定 llama.cpp `b96806d96061049a5b574269b049bf6241d63d46`，在独立副本中应用本目录的 `mtmd-audio-projection.patch` 和现有的 `../qwen35/qwen35-scalar-softmax.patch`。参考仓库保持原样。构建时关闭 Accelerate，运行时关闭 Flash Attention；这是逐位对照所用的 CPU 路径。

```sh
git -C "$LLAMA_DIR" apply "$RMI_REPO/tools/oracle/jina_audio/mtmd-audio-projection.patch"
git -C "$LLAMA_DIR" apply "$RMI_REPO/tools/oracle/qwen35/qwen35-scalar-softmax.patch"
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build-rmi-jina-audio" -DGGML_ACCELERATE=OFF -DGGML_METAL=OFF -DCMAKE_CXX_FLAGS=-DRMI_QWEN35_SCALAR_SOFTMAX
cmake --build "$LLAMA_DIR/build-rmi-jina-audio" --target llama-mtmd-cli -j 8
```

`LLAMA_DIR` 是固定 commit 的独立 llama.cpp 副本，`RMI_REPO` 是此仓库路径。测试输入是 16 kHz、单声道、PCM16 的 0.2 秒 440 Hz WAV，可用标准库生成：

```sh
python3 - <<'PY'
import math, struct, wave
with wave.open('/tmp/rmi-jina-440hz.wav', 'wb') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(16000)
    wav.writeframes(b''.join(struct.pack('<h', round(0.2 * 32767 * math.sin(2 * math.pi * 440 * i / 16000))) for i in range(3200)))
PY
```

WAV SHA256：`9540eac6175bf067b2091b4ac02b861def9559044f5da2693e02aedbd2aa4641`。Q8_0 LLM SHA256：`8fd3a363dd158a67bd19708f86b4f678908ef320cb8302fcae102ba7ed7f7d9f`；F16 audio mmproj SHA256：`ff2833fcd945aa8c6e19034d94db469c2560d081142fb41d8cc1b56cc303ae56`。

```sh
RMI_JINA_AUDIO_EMBD=/tmp/rmi-jina-audio-oracle.f32 "$LLAMA_DIR/build-rmi-jina-audio/bin/llama-mtmd-cli" \
  -m "$JINA_MODEL" --mmproj "$JINA_AUDIO_MMPROJ" --audio /tmp/rmi-jina-440hz.wav \
  -p 'Represent this audio for retrieval.' -t 1 -ngl 0 -fa off -n 1

RMI_JINA_AUDIO_MMPROJ="$JINA_AUDIO_MMPROJ" \
RMI_JINA_AUDIO_WAV=/tmp/rmi-jina-440hz.wav \
RMI_JINA_AUDIO_ORACLE_PROJECTED=/tmp/rmi-jina-audio-oracle.f32 \
  cargo test --release --lib jina_audio_projection_matches_llama_cpp_bits -- --ignored

cargo run --release -- --model "$JINA_MODEL" --embedding \
  --mmproj "$JINA_AUDIO_MMPROJ" --audio /tmp/rmi-jina-440hz.wav \
  --prompt 'Represent this audio for retrieval.' --threads 1 --embedding-output summary
```

对照覆盖全部 `750 × 1024` 个投影 F32 原始位。当前仅支持一段不超过 30 秒的 Whisper audio chunk；超过长度会明确报错。上述逐位结果限定为 macOS ARM CPU、单线程、标量 softmax、关闭 Flash Attention 的 Oracle 配置。
