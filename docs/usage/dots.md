# dots.tts 用法

dots.tts（[studio-dots-ai/dots.tts](https://github.com/studio-dots-ai/dots.tts)）
是 Qwen2 LLM + 纯 Rust 多组件 TTS 链路：PatchEncoder → LLM（codec tokens）→
Flow-Matching DiT → CAM++ speaker → BigVGAN-style AudioVAE。

模型配置：arch-qwen2 LLM GGUF（28×1536）+ clip mmproj（PatchEncoder + DiT +
CAM++ speaker + AudioVAE vocoder）。代码侧入口 `src/app/dots.rs::run_dots_tts_cli`。

> 共用前置：构建 `cargo build --release --bin rust-model-inference`。
> 仓库自带导出脚本 `tools/dots/convert_dots_tts.py`，是 torch-free 的 GGUF
> 转换器（weight_norm 折叠、Kaiser 滤波器、latent_stats）。**不是**基于
> llama.cpp 的 converter 移植。详见 `docs/REFERENCE_IMPLEMENTATIONS.md`：
> studio-dots-ai/dots.tts @ `32407a55228630475c48ecdb2c4e2c0f9c09e030` 用作
> Pinned **Oracle**（不是实现来源）。

## 1. 准备 GGUF

参考仓库的转换脚本（仓库内自带）：

```bash
python tools/dots/convert_dots_tts.py \
  --input path/to/studio-dots-ai-dots.tts-checkpoint \
  --output models/dots.tts
```

转换产物：

- `models/dots.tts/dots-tts-LLM-Q8_0.gguf`（arch=qwen2）
- `models/dots.tts/dots-tts-mmproj-F16.gguf`

LLM gguf 同时也能被 llama.cpp 直接跑（标准 GGUF 格式）；mmproj 是仓库私有布局。

## 2. Base（默认音色）

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/dots.tts/dots-tts-LLM-Q8_0.gguf \
  --mmproj models/dots.tts/dots-tts-mmproj-F16.gguf \
  --tts --prompt "你好，这是一个声音合成测试。" --language cn \
  --out output.wav
```

## 3. Edit（声音克隆 / 局部编辑）

Edit 通过 `--ref-audio` 给出参考音色（提供音频时同时建议给 `--ref-text`），
并可通过 `--source-audio` + `--instruction` 做指令式编辑（替换源音频中某段
文字的发音 / 内容 / 音色）。

```bash
# 仅声音克隆
cargo run --release --bin rust-model-inference -- \
  --model models/dots.tts/dots-tts-LLM-Q8_0.gguf \
  --mmproj models/dots.tts/dots-tts-mmproj-F16.gguf \
  --tts --prompt "你好，这是克隆后的声音。" --language cn \
  --ref-audio reference.wav --ref-text "这是参考文本。" \
  --out cloned.wav

# 指令式编辑
cargo run --release --bin rust-model-inference -- \
  --model models/dots.tts/dots-tts-LLM-Q8_0.gguf \
  --mmproj models/dots.tts/dots-tts-mmproj-F16.gguf \
  --tts --edit \
  --source-audio source.wav \
  --source-text "源音频的逐字文本" \
  --target-text "目标文本" \
  --instruction "把第二句换成更高兴的语气" \
  --ref-audio reference.wav --ref-text "音色参考文本" \
  --out edited.wav
```

约束（来自 `src/app/dots.rs:77-86`）：

- `--tts --edit` 必须同时传 `--source-audio` + 非空 `--instruction`
- `--ref-text` 缺省时使用参考音频的 x-vector only（无 patch prefill）
- `--use-xvector` 显式启用 x-vector 编码

## 4. CLI 选项速查

| 选项 | 用途 | 默认 |
|---|---|---|
| `--prompt` | 合成文本 | 必填 |
| `--language` | 语言标签 | 必填 |
| `--out` | 输出 WAV 路径 | 必填 |
| `--ref-audio` / `--ref-text` | 参考音色（声音克隆） | — |
| `--source-audio` / `--source-text` | Edit 模式源音频 + 文本 | — |
| `--target-text` | Edit 模式目标文本 | — |
| `--instruction` | Edit 模式指令 | — |
| `--use-xvector` | 强制 x-vector | — |
| `--steps` | NFE 步数 | — |
| `--seed` | 随机数种子 | — |
| `--temperature` | 采样温度 | — |

`--seed` 给定时 sampling 温度会被强制设为 `0.9`（`dots.rs:72-74`），仅锁定 RNG。

## 5. 与 Oracle 的对齐

Pinned Oracle：`studio-dots-ai/dots.tts @ 32407a55228630475c48ecdb2c4e2c0f9c09e030`

构建 / 测试入口：

- `tools/dots/build_dots_tts_oracle.sh`
- `tools/dots/run_dots_tts_oracle.py`
- `tools/dots/dots-tts-oracle-trace.patch`
- `tests/dots_tts_reference.rs`

仓库记录的覆盖（commit `16451c2 feat(dots): dots.tts base/edit GGUF+mmproj
export and pure-Rust TTS inference (#37)`）：

- vocoder 张量 bit-identical 到 `vocoder.safetensors`
- decode 链路与上游 torch reference 的 numpy port 逐 stage 对齐
- LLM gguf 能被 llama.cpp 直接加载（标准 GGUF 格式）

## 6. 已确认的限制 / 边界

| 范围 | 行为 |
|---|---|
| Edit 模式缺 `--source-audio` / `--instruction` | 推理前报错 |
| 通用 `--tts` 选项 | 必须 `--prompt` + `--mmproj` + `--out` |
| Edit 在 `--prompt` 缺省时 | 仍允许（从 `--source-text` / `--target-text` 派生） |

## 7. 相关源码索引

- `src/models/dots/` — 完整 Rust TTS 链路（Qwen2 LLM + PatchEncoder + DiT +
  CAM++ + AudioVAE）
- `src/app/dots.rs` — CLI 入口（`run_dots_tts_cli`）
- `src/app/tts.rs` — `--tts` dispatcher；按 mmproj 类型分派 Qwen3-TTS 或 dots
- `tools/dots/convert_dots_tts.py` — 仓库自带 GGUF 导出器
- `tests/dots_tts_reference.rs` — pinned Oracle 对齐
- `docs/REFERENCE_IMPLEMENTATIONS.md` — Oracle pin 与脚本列表