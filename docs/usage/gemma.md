# Gemma 4 E2B 用法

Gemma 4 是本仓库唯一同时原生支持**文本 + 视觉 + 音频**的模型架构，通过同一
mmproj 提供视觉和音频两个 projector。代码侧入口是
`src/models/gemma4/app.rs::run_gemma4(Gemma4Request)`。

> 共用前置：构建 `cargo build --release --bin rust-model-inference`。
> KV cache 默认 F16。
> Gemma 4 当前**仅支持 greedy 解码**（`src/app/text.rs:29`），传 `--temp > 0`
> 会被 CLI 拒绝。

## 1. 模型 / mmproj 约定

- 主模型：`models/gemma-4-e2b/gemma-4-E2B-it-Q8_0.gguf`（GGUF `general.architecture = gemma4`）
- mmproj：`models/gemma-4-e2b/mmproj-F16.gguf`（F16 精度；提供视觉 + 音频双 projector）
  - `clip.vision.projector_type = gemma4v`（`src/models/gemma4/vision/config.rs:22`）
  - `clip.audio.projector_type = gemma4a`（`src/models/gemma4/asr/config.rs:20`）

任一媒体输入都必须**同时传入 mmproj**；文本模式不需要 mmproj。

## 2. 文本

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/gemma-4-e2b/gemma-4-E2B-it-Q8_0.gguf \
  --prompt "Explain the result." --max-tokens 32 --temp 0
```

## 3. 图像 + 文本

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/gemma-4-e2b/gemma-4-E2B-it-Q8_0.gguf \
  --mmproj models/gemma-4-e2b/mmproj-F16.gguf \
  --image path/to/image.png \
  --prompt "Describe this image." --max-tokens 32 --temp 0
```

## 4. 音频 + 文本

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/gemma-4-e2b/gemma-4-E2B-it-Q8_0.gguf \
  --mmproj models/gemma-4-e2b/mmproj-F16.gguf \
  --audio path/to/audio.wav \
  --prompt "Transcribe the audio." --max-tokens 32 --temp 0
```

## 5. 图像 + 音频 + 文本

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/gemma-4-e2b/gemma-4-E2B-it-Q8_0.gguf \
  --mmproj models/gemma-4-e2b/mmproj-F16.gguf \
  --image path/to/image.png \
  --audio path/to/audio.wav \
  --prompt "Describe the image and audio." --max-tokens 32 --temp 0
```

约束（来自 `gemma4/app.rs::build_turn_rows` + `README.md`）：

- 每轮一种媒体最多一份
- 同时提供时，输入顺序固定为**图像 → 音频 → 提示词**
- 音频必须是 16 kHz PCM16 WAV
- 当前仅支持**一个用户轮次**

## 6. CLI 路由速查

| 输入组合 | CLI 分派 | 入口 |
|---|---|---|
| `--model ... --prompt ...`（无媒体） | `run_gemma4` 文本路径 | `src/models/gemma4/app.rs` |
| `--image` / `--audio` 任一 + `--mmproj` | `run_gemma4` 多模态路径 | 同上 |
| `--audio` 但 `arch != gemma4` | 拒绝：`Only gemma4 architecture is supported for multimodal audio, got: ...`（`src/app/text.rs:805-806`） |

## 7. 与 llama.cpp 的对齐

Pinned Oracle：`llama.cpp @ 3173a56471c1753650cd806694145ffd6dcace67`

构建 / 测试入口：

- `tools/gemma4/build_oracle.sh`
- `tests/gemma4_reference.rs`

CPU 对齐覆盖 attention softmax 前的 token IDs，以及
`gemma4.vision.preprocessed` / `gemma4.audio.mel` 的原始 F32 `u32` 位。
attention 之后**不**承诺与 llama.cpp 逐位一致——仓库统一使用准确、稳定的
标量 softmax。`--gpu` 可以运行，但当前不提供 GPU 位级对比保证。

## 8. 服务端模式

Gemma 4 通过同一 server 入口暴露；`--image` 在 server 启动时固定传入：

```bash
cargo run --release --bin server -- \
  --model models/gemma-4-e2b/gemma-4-E2B-it-Q8_0.gguf \
  --mmproj models/gemma-4-e2b/mmproj-F16.gguf \
  --image path/to/image.png \
  --host 0.0.0.0 --port 8080 --threads 4
```

## 9. 已确认的限制 / 边界

| 范围 | 行为 |
|---|---|
| `--temp > 0` | 拒绝（Gemma 4 强制 greedy） |
| `--video` | 拒绝（Gemma 4 不支持视频） |
| 缺 `--mmproj` 但传入 `--image` 或 `--audio` | 推理前拒绝 |
| `--audio` 与非 gemma4 arch | 拒绝（仅 gemma4 支持音频模态） |

## 10. 相关源码索引

- `src/models/gemma4/app.rs` — `run_gemma4` 主入口
- `src/models/gemma4/trunk/` — Gemma 4 LLM trunk
- `src/models/gemma4/vision/` — 视觉 encoder + projector（`gemma4v`）
- `src/models/gemma4/asr/` — 音频 encoder + projector（`gemma4a`）
- `src/app/text.rs:29, 765-803` — Gemma 4 CLI 路由
- `tests/gemma4_reference.rs` — pinned llama.cpp 对齐
- `docs/REFERENCE_IMPLEMENTATIONS.md` — Oracle pin 与构建脚本