# Breeze TTS 2 用法

[Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2)
是 3.5B 多模块 TTS：Qwen3 backbone (1.4B) + T5Gemma2 text_encoder (1B) +
depth_decoder (0.4B) + Mimi codec (0.1B)。支持 Voice Clone / Voice Design
（纯指令）/ Voice Direction（reference + instruction），代码侧入口
`src/app/breeze.rs::run_breeze_tts_cli`。

> 共用前置：构建 `cargo build --release --bin rust-model-inference`。
> 仓库自带导出脚本 `tools/breeze/convert_breeze.py`，是 torch-free 的 GGUF
> 转换器（保留 BF16 / F32 原始字节、张量名）。**不是**基于 llama.cpp 的
> converter 移植。

## 1. 准备 GGUF

```bash
python tools/breeze/convert_breeze.py models/Breeze-TTS-2 --out-dir models/Breeze-TTS-2
```

转换产物（输入路径与 README 同目录，输出落在 `--out-dir`）：

- `models/Breeze-TTS-2/breeze-tts-2-BF16.gguf`（主模型 6.6 GB，arch=breeze）
- `models/Breeze-TTS-2/breeze-tts-2-mmproj-F32.gguf`（音频 codec 651 MB，
  arch=breeze_audio）

主模型 GGUF 包含 TTS 全链路（Qwen3 backbone + T5Gemma2 text_encoder +
depth_decoder + embedding + lm_head），codec GGUF 包含 Mimi 音频编解码。

## 2. Plain TTS（无指令 / 无参考）

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/Breeze-TTS-2/breeze-tts-2-BF16.gguf \
  --mmproj models/Breeze-TTS-2/breeze-tts-2-mmproj-F32.gguf \
  --tts --prompt "你好，这是一个声音合成测试。" \
  --out plain.wav --seed 42
```

## 3. Voice Design（仅指令）

`--instruction` + `--cfg-scale`（默认 3，有指令时启用 CFG；显式给 1 关闭）。
指令语言应与目标文本语言一致。

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/Breeze-TTS-2/breeze-tts-2-BF16.gguf \
  --mmproj models/Breeze-TTS-2/breeze-tts-2-mmproj-F32.gguf \
  --tts --prompt "你好。" --instruction "温柔地说。" \
  --cfg-scale 3 \
  --out instruction.wav --seed 42
```

## 4. Voice Clone（参考音频 + 转写）

`--ref-audio` 必须配套非空 `--ref-text`，否则报错。参考 WAV 自动 mix-down
到 mono、resample 到 24 kHz。

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/Breeze-TTS-2/breeze-tts-2-BF16.gguf \
  --mmproj models/Breeze-TTS-2/breeze-tts-2-mmproj-F32.gguf \
  --tts --prompt "再见。" \
  --ref-audio plain.wav --ref-text "你好，这是一个声音合成测试。" \
  --out clone.wav --seed 42
```

## 5. CLI 选项速查

| 选项 | 用途 | 默认 |
|---|---|---|
| `--prompt` | 合成文本 | 必填 |
| `--out` | 输出 WAV 路径 | 必填 |
| `--mmproj` | 音频 codec GGUF（必填） | — |
| `--instruction` | 音色 / 风格描述（非空时启用 CFG） | — |
| `--cfg-scale` | 指令 CFG 强度 | `3` 有指令 / `1` 无指令 |
| `--ref-audio` / `--ref-text` | 参考音色（声音克隆） | 两者必须同时给 |
| `--max-tokens` | 最大帧数 | `128` |
| `--seed` | 随机数种子 | `42` |
| `--temperature` | 采样温度 | `0.9`；`0` = greedy |
| `--top-k` / `--top-p` | top-k / nucleus | `50` / `1.0` |
| `--threads` | 线程数 | 物理核数 |

采样：`--seed` 给定时温度被强制 `0.9`（仅锁定 RNG），
`--temperature 0` 切换为 greedy。

## 6. 与原始仓库的功能差异

| 原始 README 功能 | 当前 Rust 支持 |
|---|---|
| Voice Clone（ref-audio + ref-text） | 支持 |
| Voice Design（仅 instruction） | 支持 |
| Voice Direction（ref + instruction 同时） | **未实现**（`validate_breeze_options` 未分流） |
| Vocal Events（`(laugh)`、`[笑]` 等内联事件） | **未实现**（text_encoder 走原始 BPE，无事件 token 注入） |
| Streaming API（`/v1/audio/speech`） | **未实现**（当前 CLI 一次性合成） |
| `--fast-all` / CUDA Graph 加速 | **未实现**（纯 CPU eager） |

Voice Design 与 Voice Clone 是两条独立路径；同时给 `--ref-audio` +
`--instruction` 会被视为 voice clone（reference 生效、CFG 仍按指令启用），
未实现 reference-aware 的 CFG 双流。

## 7. 已确认的限制 / 边界

| 范围 | 行为 |
|---|---|
| `--ref-audio` 单独存在 | 推理前报错（要求配对 `--ref-text`） |
| `--cfg-scale != 1` 但无 `--instruction` | 推理前报错 |
| `--edit` / `--steps` / `--language` / `--gpu` | 推理前报错（Breeze 不识别） |
| 输出 WAV 已存在 | 推理前报错，避免覆盖 |
| 参考 WAV 多声道 / 非 24 kHz | 自动 mix-down + resample |

## 8. 相关源码索引

- `src/models/breeze/` — 主模型 + Mimi codec Rust 实现
- `src/app/breeze.rs` — CLI 入口（`run_breeze_tts_cli`）与参数校验
- `src/app/tts.rs` — `--tts` dispatcher；按 mmproj 类型分派
- `tools/breeze/convert_breeze.py` — 仓库自带 GGUF 导出器
- `tools/breeze/compare_breeze_trace.py` — 与原生修订的 F32 逐 bit 对比
- `tools/breeze/README.md` — 仓库转换 / 采样说明
