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

## 10. JEV 决策评分（Choice / Binary / Score / Multi-Q）

`--jev` 是 OpenJEV 风格的 single-forward-pass 决策模式：跳过自回归生成，
在一次 prefill 后直接读最后一层 logits，对候选 label tokens (A/B/C/…) 做
softmax 得到概率分布。架构上和"生成第一个 token"完全一样，只是后半段不
做 embedding lookup 和循环 decode。

> 参考：[`references/openjev/decisionmaking/prompts.py`](../../references/openjev/decisionmaking/prompts.py)
> 的 system prompt + 候选排序 + 单 token label 协议。

### 10.1 协议

每条 `--jev-option` 是一个候选描述，会自动映射成 A/B/C/…，并以如下
system + JSON user 喂给模型：

```
<|im_start|>system
Answer the question using the supplied context and candidate answers.
Select the single best answer. Reply with only its letter label.<|im_end|>
<|im_start|>user
{"context": "...", "question": "...", "candidates": {"A": "...", "B": "...", ...}}<|im_end|>
<|im_start|>assistant
```

模型在最后一个 token 位置输出 A/B/C/... 中某个，我们读 logits 做
softmax 得到候选分布。

### 10.2 三种 mode

模式按 `--jev-option` 的形态自动判定，无需显式声明：

**Choice（默认，K≥2）**

```bash
rust-model-inference --model qwen3.gguf --jev \
  --jev-context "明天下午2点要去机场接人" \
  --jev-question "明天的天气怎么样？" \
  --jev-option "晴天" --jev-option "阴天" --jev-option "雨天"
```

输出：

```
choice: A
probabilities:
  A: 0.8248
  B: 0.1546
  C: 0.0206
```

**Binary（K=2 + `--jev-positive`）**

K=2 且指定 `--jev-positive` 时进入 binary 模式。`--jev-positive` 缺省为 A，
与 OpenJEV Web UI 行为一致（"first option as positive"）。

```bash
rust-model-inference --model qwen3.gguf --jev \
  --jev-context "天空乌云密布，能听到远处雷声" \
  --jev-question "现在在下雨吗？" \
  --jev-option "是的" --jev-option "没有" \
  --jev-positive A
```

输出：

```
choice: B
probability (A): 0.3679
```

**Score（`description:value` 语法）**

任意 `--jev-option` 含 `:` 自动进入 score 模式。每个选项关联一个数值，
最终输出 `Σ p_i × value_i`（期望分数）。

```bash
rust-model-inference --model qwen3.gguf --jev \
  --jev-context "今天股市整体上涨，科技板块表现强劲" \
  --jev-question "市场情绪如何？" \
  --jev-option "极度乐观:5" --jev-option "乐观:4" \
  --jev-option "中性:3" --jev-option "悲观:2" --jev-option "极度悲观:1"
```

输出：

```
score: 4.7372
breakdown:
  A: 0.8117 × 5 = 4.0586
  B: 0.1516 × 4 = 0.6065
  C: 0.0171 × 3 = 0.0513
  D: 0.0013 × 2 = 0.0025
  E: 0.0183 × 1 = 0.0183
```

### 10.3 多 question（顺序 forward pass）

`--jev-question` 可重复，每次开启新问题；选项累积直到下一个
`--jev-question` 或结束。N 个问题 = N 次顺序 prefill（不共享 KV cache，
与 OpenJEV 当前实现一致）。

```bash
rust-model-inference --model qwen3.gguf --jev \
  --jev-context "天空乌云密布，能听到远处雷声" \
  --jev-question "现在在下雨吗？" --jev-option "yes" --jev-option "no" \
  --jev-question "需要带伞吗？" --jev-option "需要" --jev-option "不需要"
```

### 10.4 JSON 输出

`--jev-output json` 输出 newline-delimited JSON，每行一个 question：

```bash
rust-model-inference --model qwen3.gguf --jev \
  --jev-context "..." \
  --jev-question "..." \
  --jev-option "晴天:5" --jev-option "阴天:3" --jev-option "雨天:1" \
  --jev-output json
```

```json
{"mode":"score","question":"...","labels":["A","B","C"],"descriptions":["晴天","阴天","雨天"],"values":[5.0,3.0,1.0],"probabilities":{"A":0.81,"B":0.15,"C":0.04},"score":4.74,"prefill_ms":6470}
```

### 10.5 性能

`models/qwen3-0.6b-gguf/Qwen3-0.6B-IQ4_NL.gguf`，Intel Core Ultra 5 125H，4 threads：

| Mode | Prefill |
|---|---|
| Choice / Binary / Score（单 question） | ~5.7s (cold) |
| Multi-question × N | ~5.7s × N（顺序，不共享 KV） |

注意：cold-start 包含模型加载 + tokenizer 初始化；连续运行时每次 prefill
只算真正的前向传播时间（~200ms 量级）。

### 10.6 与 OpenJEV 的差异

| 维度 | 本实现 | OpenJEV |
|---|---|---|
| Model | 任何本地 Qwen3 GGUF | 锁定 Qwen3-4B-Instruct-2507 |
| Multi-question KV 复用 | ❌ 每次新建 session | ❌ 当前也按顺序重编码（planned） |
| Shared prefix batching | ❌ | 计划中 |
| Score schema (`positive_id`) | ✅ | ✅ |
| Bilingual UI / i18n | ❌ | ✅ |
| HTTP API / JSON export | ❌（CLI 模式） | ✅ |
| 候选 label 单 token 校验 | ✅（硬约束） | ✅（硬约束） |

### 10.7 已知限制

1. **Qwen3-0.6B 是 base 模型**（不是 Instruct）。OpenJEV README 报告
   `Qwen3-0.6B instruction, FP16` 在 development challenge 上准确率仅
   65.7%。要让 JEV 输出真正"对的"答案，需要 Qwen3-0.6B-Instruct GGUF。
   机制本身（logit → softmax → choice）在 base model 上**能跑**，只是
   选出的 label 不一定准确。
2. **label letters 必须单 token**。Qwen3 tokenizer 对 A/B/C/…/Z 是单 token，
   其他现代 BPE tokenizer 一般也满足，但老 tokenizer 可能不满足——会报
   "Label 'A' tokenizes to N tokens"。
3. **不支持 8-bit 以下量化 KV 的精度 drift 修复**。Q4_NL 等量化会引入 ≤
   1 ULP drift，对单 forward pass 影响比生成更大（生成会自我纠正，
   single-shot 选择不会）。可以接受"次于 FP16 baseline 的概率分布但
   argmax 偶尔分叉"的容忍度。

- `docs/SUPPORTED_MODELS.md` — 验证状态与量化格式支持
