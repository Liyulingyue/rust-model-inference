# JEV 决策评分

> **Status (2026-09-20)**：`--jev` CLI 是 OpenJEV 风格的 single-forward-pass
> 决策评分模式。9 个 model trunk 全支持（qwen3 / qwen3.5 / llama 家族 /
> gemma4 / lfm2 / lfm25 / spark2_5 / nemotron_h / hunyuan-dense）。

## 1. 设计动机

标准的 LLM 生成模式（`--prompt "..."`）是自回归循环：prefill →
sample token 1 → 把 token 1 喂回去 → sample token 2 → …。这种模式
不适合两类场景：

1. **约束决策**：候选答案在调用前已确定（如客服路由的 3 个队列），模型
   只需要为每个候选打分，不该自由生成文本。
2. **结构化评分**：需要的是概率分布（或加权分），不是生成的字符串。

`--jev` 跳过自回归循环，**只做一次 prefill，读最后一层 logits**，然后
对候选 label tokens (A/B/C/…) 做 softmax 得到概率分布。

## 2. 协议

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

## 3. 三种 mode（自动检测）

模式按 `--jev-option` 形态自动判定，无需显式声明：

### 3.1 Choice（默认，K≥2）

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

### 3.2 Binary（K=2 + `--jev-positive`）

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

### 3.3 Score（`description:value` 语法）

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

## 4. 多 question（顺序 forward pass）

`--jev-question` 可重复，每次开启新问题；选项累积直到下一个
`--jev-question` 或结束。N 个问题 = N 次顺序 prefill（不共享 KV cache，
与 OpenJEV 当前实现一致）。

```bash
rust-model-inference --model qwen3.gguf --jev \
  --jev-context "天空乌云密布，能听到远处雷声" \
  --jev-question "现在在下雨吗？" --jev-option "yes" --jev-option "no" \
  --jev-question "需要带伞吗？" --jev-option "需要" --jev-option "不需要"
```

## 5. JSON 输出

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

## 6. 架构原理

```
prefill ──► logits [batch, seq_len, vocab]
                      └── 只取 [0, -1] 最后一个 token 的 logits
                           │
                           ├── logits["A"] → softmax → P(A)
                           ├── logits["B"] → softmax → P(B)
                           └── argmax → 选概率最大的选项

没有自回归，没有第二次 forward，没有 embedding lookup
```

和"生成第一个 token"几乎完全一样，只是后半段不做循环 decode。

## 7. 跨 trunk 实现现状

| 架构 | 状态 | 实现方式 |
|---|---|---|
| Qwen3 / Qwen3-VL（含 Qwen3-0.6B-Instruct / IQ4_NL / IQ4_XS / Q2_K…） | ✅ | `Qwen3Session::forward_logits` |
| Qwen3.5 / Qwen3.5-2B 等 dense + hybrid | ✅ | `Qwen35Session::forward_logits` |
| Llama trunk（llama / k2-horizon / granite / nanbeige / qwen2_2） | ✅ | `llama::run_forward_logits_llama`（copy-paste prefill） |
| Gemma4 | ✅ | `Gemma4Session::forward_logits`（thin wrapper over `forward_rows`） |
| LFM2（Liquid Foundation Model 2，attention + shortconv hybrid） | ✅ | `lfm2::run_forward_logits_lfm2`（copy-paste prefill） |
| LFM2.5（1.2B 等） | ✅ | `lfm25::run_forward_logits_lfm25`（copy-paste prefill） |
| Spark 2.5（1.7B / 4B） | ✅ | `SparkSession::forward_logits`（拆 `decode_step` 抽出 `forward_step_logits`） |
| Nemotron-H | ✅ | `nemotron_h::run_forward_logits_nemotron_h`（wrap `NemotronModel::prefill`） |
| Hunyuan-dense（Hy-MT2 1.8B / 7B） | ✅ | 复用 `Qwen3Session::forward_logits` + Hunyuan chat prompt |
| LFM2-MoE | ❌ | 待办（MoE 独立 trunk） |

> 模式分两类：
> 1. **Session API（薄包装）**：当 trunk 已经有 Session/forward_rows/prefill
>    这类结构化入口时，加一个 `forward_logits` 方法即可（qwen3 / qwen35 /
>    gemma4 / nemotron / spark）。
> 2. **Copy-paste prefill**：当 trunk 是 monolith（run_inference 一个
>    函数包揽全部），按当前 `run_inference` 的 setup + prefill 部分
>    拷贝一份独立函数（llama / lfm2 / lfm25）。

Dispatcher 在 `src/app/text.rs::run_jev_decision` 中按 `general.architecture`
自动路由到对应 trunk 实现。

## 8. Per-arch chat template 差异

JEV 按 `general.architecture` 自动选 chat template：

| Arch | Chat template |
|---|---|
| qwen3 / qwen3vl | `<\|im_start\|>system\n...\n<\|im_end\|>\n<\|im_start\|>user\n...\n<\|im_end\|>\n<\|im_start\|>assistant\n` |
| qwen35 | 同 qwen3，但 positions 用 mrope `[t, t, t, 0]` |
| llama / qwen2_2 / minicpm | `system\n{...}\nuser\n{...}\nassistant\n` |
| k2-horizon | `<\|start_of_role\|>system<\|end_of_role\|>...<\|end_of_text\|>\n...<\|start_of_role\|>assistant<\|end_of_role\|>` |
| granite | 同 k2-horizon 格式 |
| nanbeige | base model，无 chat template，直接 `{system}\n\n{payload}\n\nAnswer:` |
| gemma4 | `{system}\n\n{payload}\n\n<turn\|>\n<\|turn>model\n` |
| lfm2 / lfm25 | `{role}\n{content}\n` + `assistant\n`（用 `tokenizer.bos_id()` 前缀） |
| spark2_5 | `<｜start▁of▁sentence｜><\|System\|>\n...\n<｜end▁of▁sentence｜><｜start▁of▁sentence｜><\|User\|>...\n<｜end▁of▁sentence｜><｜start▁of▁sentence｜><\|Bot\|></think>` |
| nemotron_h | base model，没有 chat template，直接 `{system}\n\n{payload}\n\nAnswer:` |
| hunyuan-dense (1.8B) | `<\|hy_User\|>{msg}<\|hy_Assistant\|>`（v1 模板） |
| hunyuan-dense (7B+) | raw 文本，无 chat header（`build_hunyuan_chat_prompt` 自动判别） |

## 9. 性能

`models/qwen3-0.6b-gguf/Qwen3-0.6B-IQ4_NL.gguf`，Intel Core Ultra 5 125H，4 threads：

| Mode | Prefill |
|---|---|
| Choice / Binary / Score（单 question） | ~5.7s (cold) |
| Multi-question × N | ~5.7s × N（顺序，不共享 KV） |

注意：cold-start 包含模型加载 + tokenizer 初始化；连续运行时每次 prefill
只算真正的前向传播时间（~200ms 量级）。

## 10. 端到端使用

```bash
cargo build --release --bin rust-model-inference

# Choice mode（默认）
./target/release/rust-model-inference \
  --model "models/qwen3-0.6b-gguf/Qwen3-0.6B-IQ4_NL.gguf" \
  --jev --jev-context "..." --jev-question "..." \
  --jev-option "晴天" --jev-option "阴天" --jev-option "雨天"

# Binary mode
./target/release/rust-model-inference \
  --model "models/..." \
  --jev --jev-question "..." --jev-option "yes" --jev-option "no" \
  --jev-positive A

# Score mode（description:value）
./target/release/rust-model-inference \
  --model "..." \
  --jev --jev-question "..." \
  --jev-option "极度乐观:5" --jev-option "乐观:4" --jev-option "中性:3"

# Multi-question（同一 context，N 次顺序 forward）
./target/release/rust-model-inference \
  --model "..." --jev --jev-context "..." \
  --jev-question "Q1" --jev-option "a" --jev-option "b" \
  --jev-question "Q2" --jev-option "x" --jev-option "y"

# JSON 输出（newline-delimited，每行一个 question）
./target/release/rust-model-inference \
  --model "..." --jev --jev-output json --jev-context "..." \
  --jev-question "..." --jev-option "晴天:5" --jev-option "阴天:3"
```

## 11. 与 OpenJEV 的差异

| 维度 | 本实现 | OpenJEV |
|---|---|---|
| Model | 任何本地 GGUF（trunk-specific） | 锁定 Qwen3-4B-Instruct-2507 |
| Multi-question KV 复用 | ❌（每个 question 重新 prefill） | ❌（planned） |
| Shared prefix batching | ❌（TODO） | 计划中 |
| Score schema (`positive_id`) | ✅ | ✅ |
| Bilingual UI / i18n | ❌ | ✅ |
| HTTP API / JSON export | ❌（CLI 模式） | ✅ |
| 候选 label 单 token 校验 | ✅（硬约束） | ✅（硬约束） |

## 12. 已知限制

1. **Base model 准确率低**：Qwen3-0.6B 是 base model（不是 Instruct）。
   OpenJEV README 报告 base model 在 development challenge 上准确率仅
   65.7%。要让 JEV 输出真正"对的"答案，需要 instruct GGUF。机制本身
   在 base model 上能跑，只是选出的 label 不一定准确。
2. **label letters 必须单 token**：Qwen3 tokenizer 对 A/B/C/…/Z 是单
   token，其他现代 BPE tokenizer 一般也满足。校验在
   `verify_label_tokens_single` 中。
3. **量化 KV drift**：Q4_NL 等量化会引入 ≤1 ULP drift，对单 forward pass
   影响比生成更大（生成可自我纠正，single-shot 选择不会）。可接受
   "次于 FP16 baseline 的概率分布但 argmax 偶尔分叉"的容忍度。
4. **per-arch chat template 硬编码**：每个 trunk 的 prompt format 是
   写死在 `run_jev_decision_*` 函数里的。如果上游 tokenizer chat
   template 变化，需要同步更新。

## 13. 源码索引

| 路径 | 角色 |
|---|---|
| `src/app/cli.rs` | `--jev*` CLI 解析 |
| `src/app/text.rs::run_jev_decision` | arch dispatcher |
| `src/app/text.rs::run_jev_decision_{qwen3,qwen35,llama,gemma4,lfm2,lfm25,spark,nemotron_h,hunyuan}` | per-arch 实现 |
| `src/app/text.rs::{prepare_jev_questions,verify_label_tokens_single,build_jev_prompt,print_jev_question,compute_jev_result}` | 共享 helper |
| `src/models/*/trunk/forward.rs` 或 `session.rs` | 各 trunk 的 `forward_logits` 实现 |
| `tests/quantized_inference.rs` | 已有 IQ4_NL parity 测试 |

> Reference：上游协议来源 `references/openjev/decisionmaking/prompts.py`
> 的 system prompt + 候选排序 + 单 token label 协议。