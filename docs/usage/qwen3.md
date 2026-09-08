# Qwen3 家族用法

本仓库对 Qwen3 / Qwen3.5 / Qwen3.8 / Qwen3-TTS / Qwen3-ASR 等模型的端到端命令行示例。

> 通用前置：构建 `cargo build --release --bin rust-model-inference`。所有命令均以
> 工作目录为仓库根目录为前提；GGUF / 音频 / 图像路径请按本地调整。
> KV cache 默认 F16；与 llama.cpp 做位级对比时显式传 `--kv-cache f16`。

## 1. 文本（Qwen3 / Qwen3.5 / Qwen3.8）

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --prompt "法国的首都是" --max-tokens 30
```

`--threads N` 默认 `min(available_parallelism, 8)`。`--bench` 分别报告
`BENCH: pp`（prompt 处理）和 `BENCH: tg`（token 生成）。

## 2. Embedding

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/Qwen3-Embedding-0.6B-Q8_0.gguf --embedding
```

回 pinned llama.cpp 向量校验（位级对比）：

```bash
cargo test --test embedding_parity
```

## 3. 视觉语言（Qwen3-VL / Qwen3.5 / Qwen3.8）

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/Qwen3.5-0.8B-Q8_0.gguf \
  --mmproj models/Qwen3.5-0.8B-mmproj-F16.gguf \
  --image path/to/image.jpg \
  --prompt "描述这张图片"
```

限制：

- 当前每种媒体最多一份；同一轮同时给图像和音频时顺序固定为图像、音频、提示词。
- 音频必须是 16 kHz PCM16 WAV。

## 4. ASR（Qwen3-ASR + `qwen3a` mmproj）

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/Qwen3-ASR-0.6B-Q8_0.gguf \
  --mmproj models/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
  --audio models/001_16k.wav \
  --language en
```

约束（来自 SUPPORTED_MODELS.md）：

- ASR 模式下 `--temp` 必须为 0（greedy）。
- ASR 与 `--image` 不能同时使用，CLI 推理前会拒绝。

本仓库对 ASR 的**回归覆盖**：`src/format/ggufrs.rs` 的 GGUF / GGUFRS 张量等价测试。
**没有 llama.cpp 端到端 Oracle pin**（`docs/REFERENCE_IMPLEMENTATIONS.md` 仅
`Reference only`，指向 `hqu-little-boy/asr.cpp` fork）。

## 5. TTS（Qwen3-TTS Base，参考音频声音克隆）

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
  --mmproj models/Qwen3-TTS/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
  --tts --prompt "你好，这是一个声音合成测试。" --language cn \
  --ref-audio reference.wav --out output.wav
```

`--ref-audio` / `--ref-text` 用于声音克隆；不传则使用 Base 模型的默认音色。

Pinned llama.cpp Oracle：`201e50c2076a20adc460c41598593c7cd7b0813`，
通过 `tests/qwen3_tts_reference.rs` 与 `tools/tts/build_qwen3_tts_oracle.sh` 覆盖。

## 6. 与 llama.cpp 的数值对齐

### 通用 scalar 位级对比（Qwen3-0.6B）

```bash
cargo build --release --features parity-trace
cargo run --release --features parity-trace --bin rust-model-inference -- \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --prompt "法国的首都是" \
  --dump-logits
```

### Qwen3.5 / Qwen3.8 / Qwen3-TTS 的固定 Oracle

详见 `docs/REFERENCE_IMPLEMENTATIONS.md`：

- Qwen3.5 / Qwen3.8-27B：`llama.cpp @ b96806d96061049a5b574269b049bf6241d63d46`
- Qwen3-TTS Base：`llama.cpp @ 201e50cc2076a20adc460c41598593c7cd7b0813`

构建 / 测试脚本：

- `tools/parity/build_qwen35_oracle.sh`、`tests/qwen35_reference.rs`
- `tools/tts/build_qwen3_tts_oracle.sh`、`tests/qwen3_tts_reference.rs`

### CPU 性能基准

公平对比：

```bash
# 本仓库
cargo run --release --bin rust-model-inference -- \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --prompt "2 + 3 =" --max-tokens 4 --temp 0 --threads 8 \
  --kv-cache f16 --bench

# llama.cpp（同等 8 线程 / 0 GPU）
llama-bench -ngl 0 -t 8 -m models/Qwen3-0.6B-Q8_0.gguf
```

复现脚本与机器固定方法见 `docs/OPTIMIZATION.md#rust-与-llamacpp-固定机器对比2026-08-10`。

## 7. 服务端模式（OpenAI 兼容）

```bash
# 文本
cargo run --release --bin server -- \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --host 0.0.0.0 --port 8080 --threads 4

# Embedding
cargo run --release --bin server -- \
  --model models/Qwen3-Embedding-0.6B-Q8_0.gguf --embedding

# ASR
cargo run --release --bin server -- \
  --model models/Qwen3-ASR-0.6B-Q8_0.gguf \
  --mmproj models/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
  --audio models/001_16k.wav --language en

# TTS
cargo run --release --bin server -- \
  --model models/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
  --mmproj models/Qwen3-TTS/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
  --tts --language cn
```

## 8. 已确认的限制 / 边界

| 范围 | 行为 |
|------|------|
| ASR + 图像 | 拒绝（CLI 在推理前检查） |
| ASR 非 greedy 解码（`--temp > 0`） | 拒绝 |
| Qwen3-VL 不匹配的维度（除 1024-dim 与 2048-dim 两组白名单外） | 配置阶段拒绝 |
| GPU 后端（`--features vulkan`） | 可跑，但当前不提供 GPU 位级 Oracle 保证 |

## 9. 相关源码索引

- `src/models/qwen3/` — 文本 / Embedding / VL / ASR / TTS trunk
- `src/app/audio.rs` — ASR 路由
- `src/app/text.rs` — 文本、Embedding、多模态 CLI 入口
- `src/app/tts.rs` — TTS CLI 入口
- `src/format/ggufrs.rs` — GGUF / GGUFRS 等价测试
- `docs/REFERENCE_IMPLEMENTATIONS.md` — Pinned Oracle 与构建脚本
- `docs/SUPPORTED_MODELS.md` — 验证状态与量化格式支持