# Fun-ASR-Nano 用法

Fun-ASR-Nano 是 SenseVoice 编码器 + Qwen3-0.6B LLM 的语音识别模型，支持 31 语种。
本仓库通过 SAN-M encoder + Qwen3 LLM trunk 实现 CPU 端零 Python 推理。

## 模型文件

从 HuggingFace 下载 GGUF：

```
models/Fun-ASR-Nano-GGUF/
├── funasr-encoder-f16.gguf   # SenseVoice SAN-M 编码器 + adaptor (F16, ~469 MB)
└── qwen3-0.6b-q8_0.gguf      # Qwen3-0.6B LLM (Q8_0, ~805 MB)
```

## 快速开始

```bash
cargo build --release --bin rust-model-inference

./target/release/rust-model-inference \
  --model models/Fun-ASR-Nano-GGUF/qwen3-0.6b-q8_0.gguf \
  --mmproj models/Fun-ASR-Nano-GGUF/funasr-encoder-f16.gguf \
  --audio audio.wav \
  --threads 8
```

输出：
```
我想问我在滨海新区有房
```

## 参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model` | Qwen3 LLM GGUF 路径 | 必填 |
| `--mmproj` | FunASR encoder GGUF 路径 | 必填 |
| `--audio` | WAV 音频文件路径（16 kHz mono 最佳） | 必填 |
| `--threads N` | 推理线程数 | 自动检测 |
| `--max-tokens N` | 最大生成 token 数 | 512 |
| `--chunk SECONDS` | 分块窗口大小（秒），长音频分段推理 | 不分块 |
| `--srt` | 输出 SRT 字幕格式（带时间戳） | 关闭 |
| `--repetition-penalty α` | 重复抑制（>1 抑制，1.0 禁用） | 1.0 |

## 长音频分块

长音频一次性推理可能触发 LLM 重复输出（OOD 问题）。使用 `--chunk` 分段：

```bash
./target/release/rust-model-inference \
  --model models/Fun-ASR-Nano-GGUF/qwen3-0.6b-q8_0.gguf \
  --mmproj models/Fun-ASR-Nano-GGUF/funasr-encoder-f16.gguf \
  --audio long_audio.wav \
  --threads 8 \
  --chunk 15
```

每个 chunk 独立编码 + 独立 KV cache，输出按顺序拼接。

## 音频格式

- 期望 16 kHz mono PCM16 WAV
- 其他采样率会自动线性重采样到 16 kHz（精度有限，建议预处理）
- 多声道会自动混缩为 mono

## 推理流程

```
WAV (16kHz) → kaldi 80-mel fbank + LFR(7/6) → [T, 560]
  → pre-scale sqrt(512) + sinusoidal position encoding
  → SAN-M encoder (50+20 layers, FSMN attention)
  → adaptor (512→2048→1024 + 2 transformer layers)
  → LFR truncation (~T/8 audio tokens)
  → [prefix tokens | audio embeds | suffix tokens]
  → Qwen3-0.6B LLM (greedy)
  → transcription text
```

## 参考实现

- [FunASR llama.cpp runtime](https://github.com/modelscope/FunASR/tree/main/runtime/llama.cpp) (v0.2.6)
- GGUF 架构名：`funasr-sensevoice-encoder`
- LLM 架构名：`qwen3`（标准 Qwen3 GGUF）

## SRT 字幕输出

使用 `--srt` 输出带时间戳的字幕格式（需配合 `--chunk`）：

```bash
./target/release/rust-model-inference \
  --model models/Fun-ASR-Nano-GGUF/qwen3-0.6b-q8_0.gguf \
  --mmproj models/Fun-ASR-Nano-GGUF/funasr-encoder-f16.gguf \
  --audio long_audio.wav \
  --threads 8 \
  --chunk 15 \
  --srt
```

输出：
```
1
00:00:00,000 --> 00:00:15,000
第一段文字

2
00:00:15,000 --> 00:00:30,000
第二段文字
```

## 重复惩罚

长音频推理可能出现 LLM 重复输出。使用 `--repetition-penalty` 抑制：

```bash
./target/release/rust-model-inference \
  --model models/Fun-ASR-Nano-GGUF/qwen3-0.6b-q8_0.gguf \
  --mmproj models/Fun-ASR-Nano-GGUF/funasr-encoder-f16.gguf \
  --audio long_audio.wav \
  --threads 8 \
  --chunk 15 \
  --repetition-penalty 1.2
```

## 已知限制

- 无 VAD 分段（参考实现支持 `--vad` FSMN-VAD，本仓库尚未实现）
