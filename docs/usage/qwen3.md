# Qwen3 家族用法

本仓库对 Qwen3 / Qwen3.5 / Qwen3.8 / Qwen3-TTS / Qwen3-ASR 等模型的端到端命令行示例。

> 通用前置：构建 `cargo build --release --bin rust-model-inference`。所有命令均以
> 工作目录为仓库根目录为前提；GGUF / 音频 / 图像路径请按本地调整。
> KV cache 默认 F16；与 llama.cpp 做位级对比时显式传 `--kv-cache f16`。
> 
> 常用生成参数（适用于所有 Qwen3 路径）：
> 
> - `--max-context N`：KV cache 容量上限，默认 8192。超过此值的输入会触发预
>   警告或分配错误；显式调小可避免大上下文模型（如 Qwen3.5-27B 128k、
>   K2-Horizon-4B/7B 524k）一次性占用 GB~TB 级 KV 内存。
> - `--repetition-penalty α`：logit 级重复抑制，默认 1.0（禁用）；α > 1 抑制
>   重复（与 llama.cpp / Hugging Face `repetition_penalty` 等价）。对低质量量化
>   （如 Q4_K_M）下陷入复读循环的模型尤其有用。

## 1. 文本（Qwen3 / Qwen3.5 / Qwen3.8）

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --prompt "法国的首都是" --max-tokens 30
```

`--threads N` 默认 `min(available_parallelism, 8)`。`--bench` 分别报告
`BENCH: pp`（prompt 处理）和 `BENCH: tg`（token 生成）。

### Chunked prefill

prompt 默认按最多 64 个 token 分块处理；`--prefill-batch-size 1` 是顺序诊断基线。已验证的
具体实物为 Qwen3 Q4_0、Qwen3.5 0.8B BF16，以及 Gemma4 E2B Q8_0（F16 mmproj），
测试设备为 Apple M3 Max；其他尺寸或量化仍受现有 Vulkan eligibility 约束。

```bash
rust-model-inference --model model.gguf --prompt "Hello" --prefill-batch-size 64
rust-model-inference --model model.gguf --prompt "Hello" --prefill-batch-size 1
```

Qwen3 和 Qwen3.5 在已支持的 CPU/Vulkan 路径执行分块 prefill。Gemma4 在 CPU/Vulkan
路径支持分块 prefill；Vulkan 仅加速批量线性投影，attention 与 KV 保持模型控制的 CPU
路径。chunk 状态整块原子提交；Vulkan 失败时从同一已提交状态在 CPU 重算完整 chunk，
不会提交部分结果。decode 仍保持逐 token 行为。

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

### 5.1 TTS 作为多模态回复后处理（`--tts-model` / `--tts-mmproj`）

Qwen2.5-Omni 等多模态模型官方输出包含**文本 + 语音**两路,本仓库在
`Qwen2.5-Omni` GGUF 集合中只包含 Thinker（文本 LLM）+ mmproj（视觉/音频
编码器）,不含独立 Talker。补齐语音输出：将 `Qwen3-TTS-12Hz-1.7B-Base`
作为后处理器,在文本生成完成后自动合成 24 kHz mono WAV。

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/Qwen2.5-Omni-3B-GGUF/Qwen2.5-Omni-3B-Q8_0.gguf \
  --mmproj models/Qwen2.5-Omni-3B-GGUF/mmproj-BF16.gguf \
  --image references/apple.png \
  --prompt "Describe the image briefly." \
  --tts-model models/Qwen3-TTS-12Hz-1.7B-Base-GGUF/Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
  --tts-mmproj models/Qwen3-TTS-12Hz-1.7B-Base-GGUF/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
  --out speech.wav
```

触发条件（全部满足）：
- 入口路径包含 `--mmproj`/`--image`/`--audio`/`--video` 任一（即
  `run_multimodal_with_video` 路径）
- 同时传 `--tts-model` 与 `--tts-mmproj`
- 传 `--out <wav>` 指向可写路径

TTS 帧预算自动按 `max(max_tokens * 4, 128).min(1024)` 计算 — 用户传
`--max-tokens 30` 时 TTS 跑 128 帧（约 1.6 秒音频）,`--max-tokens 200` 时
800 帧（约 10 秒）。

支持所有 Omni 输入模态：
- `--image <file>`：视觉
- `--audio <16kHz WAV>`：语音（听写、转写）
- `--video <mp4>`：视频（需要 `ffmpeg`+`ffprobe`，见 README）

实测 `models/Qwen2.5-Omni-3B-Q8_0.gguf` 在 18 线程下：

| 输入 → 输出 | vision encode | TTS frame_loop | TTS dac_decode | 总耗时 |
|-----------|--------------|---------------|---------------|-------|
| apple.png → text + 24k WAV | ~60s | ~20s | ~20s | ~5min |
| zh.wav → text + 24k WAV | ~5s（encoder） | ~20s | ~20s | ~3min |
| test.mp4 (320×240) → text + 24k WAV | ~5min（4 帧） | ~20s | ~20s | ~10min |

完整输出验证参考 `models/omni_apple_reply.wav` 等。

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
