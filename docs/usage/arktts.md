# Audio8-TTS-Realistic-Air 用法

[Audio8-TTS-Realistic-Air](https://modelscope.cn/models/Edge0/Audio8-TTS-Realistic-Air)
是 ArkTTS DualAR TTS：配对的主模型 GGUF（`arktts`，slow/fast 双 Transformer）
与 codec GGUF（`arktts_codec`，encoder、codebook、因果卷积与 decoder）。
输出单声道 44.1 kHz WAV 或裸 PCM16。实现细节与精度策略见
[ArkTTS 开发说明](../develop/ARKTTS.md)，数值/入口矩阵见
[Oracle 验证](../../tools/oracle/arktts/validation.md)。

> 共用前置：`cargo build --release --bin rust-model-inference`。

## 1. 准备 GGUF

转换器 `tools/converter/arktts/convert_arktts.py` 是 torch-free 之外的最小依赖导出器，
输入官方 `config.json`、`tokenizer.json`、`model.safetensors` 与 `codec.pth`：

```bash
.venv/bin/python -m tools.converter.arktts.convert_arktts "$MODEL_DIR"
```

`--format` / `--codec-format` 支持 `f32`、`f16`、`bf16`、`q8_0`、`q4_0`、`q4_1`、
`q4_k`、`q5_k`、`q6_k`，默认 BF16 主模型 + F32 codec：

```bash
.venv/bin/python -m tools.converter.arktts.convert_arktts "$MODEL_DIR" \
  --out-dir "$MODEL_DIR/multiq/q4_k" --format q4_k --codec-format q4_k
```

主模型 226 个张量、codec 455 个张量；每对文件写入相同的 `arktts.bundle_id`，
两组件必须来自同一次导出，混搭会在加载前报错。

## 2. 普通合成（CLI）

```bash
cargo run --release --bin rust-model-inference -- \
  --model "$MODEL_DIR/Audio8-TTS-Realistic-Air-BF16.gguf" \
  --mmproj "$MODEL_DIR/Audio8-TTS-Realistic-Air-codec-F32.gguf" \
  --tts --prompt "你好，这是一个语音合成测试。" \
  --out plain.wav
```

`--max-tokens` 对 Audio8 是**帧预算**（1..2048，默认 128）；每帧 2048 个采样
（约 46 ms），因此 128 帧约 5.9 秒。`--temperature` 默认 `0.8`，`0` 为 greedy；
`--seed` 默认 `42`。`--compute` 可选 `cpu` / `auto` / `vulkan`。

## 3. 声音克隆（CLI）

```bash
cargo run --release --bin rust-model-inference -- \
  --model "$MODEL_DIR/Audio8-TTS-Realistic-Air-BF16.gguf" \
  --mmproj "$MODEL_DIR/Audio8-TTS-Realistic-Air-codec-F32.gguf" \
  --tts --prompt "再见。" \
  --ref-audio plain.wav --ref-text "你好，这是一个语音合成测试。" \
  --out clone.wav
```

参考 WAV 自动 mix-down 到单声道并重采样到 44.1 kHz；`--ref-audio` 必须与
非空 `--ref-text` 同时给出。

## 4. HTTP `/v1/audio/speech`

启动服务时 `--tts` 的模型/codec 对即为该端点后端：

```bash
{ "model": "Audio8-TTS-Realistic-Air",
  "input": "你好。",
  "response_format": "wav",
  "voice": "file:///data/ref.wav",
  "reference_text": "参考音频的转写",
  "speed": 1.0 }
```

- `response_format`：`wav`（默认）或 `pcm`。
- `voice` 与 `reference_text` 必须同时给出才是克隆；`voice` 接受
  `file://path`（≤16 MiB 的常规文件）或 `data:audio/wav;base64,...`。
- `speed` 只支持 `1.0`。同一时刻只允许一个生成请求（忙时返回 429）。

## 5. 作为多模态后处理（Omni → TTS）

带媒体的多模态请求可以额外指定 TTS 模型，把回复文本再合成为 WAV：

```bash
cargo run --release --bin rust-model-inference -- \
  --model omni.gguf --mmproj omni-mmproj.gguf --image frame.png \
  --prompt "描述这张图" --out reply.wav \
  --tts-model "$MODEL_DIR/Audio8-TTS-Realistic-Air-BF16.gguf" \
  --tts-mmproj "$MODEL_DIR/Audio8-TTS-Realistic-Air-codec-F32.gguf"
```

该路径要求 `--tts-model`/`--tts-mmproj` 成对出现，且请求本身带
`--image`/`--video`/`--audio` 或 `--mmproj`；未显式给 `--temperature` 时
Omni 回复走 greedy，TTS 阶段保留音频模型的默认温度。

## 6. 选项速查

| 选项 | 用途 | 默认 |
|---|---|---|
| `--prompt` | 合成文本 | 必填 |
| `--out` | 输出 WAV 路径（已存在则报错） | 必填 |
| `--mmproj` | codec GGUF（`arktts_codec`） | 必填 |
| `--ref-audio` / `--ref-text` | 参考音色 + 转写（声音克隆） | 成对出现 |
| `--max-tokens` | 帧预算，1..2048 | `128` |
| `--temperature` | 采样温度，`0` = greedy | `0.8` |
| `--top-k` / `--top-p` | top-k / nucleus | `50` / `0.95` |
| `--seed` | 随机数种子 | `42` |
| `--threads` | 线程数 | 物理核数 |
| `--compute` | `cpu` / `auto` / `vulkan` | `cpu` |

## 7. 已确认的限制 / 边界

| 范围 | 行为 |
|---|---|
| 主模型/codec 架构不匹配或 `bundle_id` 不同 | 加载前报错 |
| `--max-tokens` > 2048 | 加载前报错 |
| `--voice` / `--speed` / `--pronunciations` | 加载前报错（PiVo 专属） |
| `--rerank` / `--clm-head` / `--laya-request` / `--tts-model` 与 `--tts` 同用 | 加载前报错（会被静默丢弃） |
| `--edit` / `--language` 以外的 Qwen3-TTS 专属项 | 加载前报错 |
| `--compute vulkan` 但设备缺少所需扩展/内存 | 严格模式显式失败；`auto` 回退 CPU |
| 自由 greedy 码流 | 精度/后端可能分叉；固定历史对照通过不代表主观音质证明 |

## 8. 相关源码索引

- `src/models/arktts/` — 主模型、codec、设备图与契约
- `src/app/tts/mod.rs` — `--tts` 分派、`arktts_sampling`、CLI 入口
- `src/app/server/audio_speech.rs` — `/v1/audio/speech` 后端
- `src/app/text/tts_bridge.rs` — Omni → TTS 后处理
- `tools/converter/arktts/` — 转换器与自检
- `tools/oracle/arktts/` — 对照脚本、shader 与验证记录