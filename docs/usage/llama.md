# llama / Granite / Nanbeige / MiniCPM5 用法

`general.architecture` 为 `llama` / `granite` / `nanbeige` 的 GGUF 全部走
`src/models/llama/` trunk。CLI 路由条件：

```rust
} else if arch == "llama" || arch == "granite" || arch == "nanbeige" {
    crate::models::llama::run_inference(...)
```

arch 区分 chat template、`add_special`、RoPE 公式等行为差异；**forward 实现共用**。

> 共用前置：构建 `cargo build --release --bin rust-model-inference`。
> KV cache 默认 F16；与 llama.cpp 位级对比时显式传 `--kv-cache f16`。
>
> 常用生成参数（适用于所有 llama 家族）：
>
> - `--max-context N`：KV cache 容量上限，默认 8192。
> - `--repetition-penalty α`：logit 级重复抑制，默认 1.0（禁用）；α > 1 抑制
>   重复。

## 1. 通用 Llama / MiniCPM5

`MiniCPM5-1B` 与通用 Llama 都使用 Qwen2 风格 chat template：

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/MiniCPM5-1B-Q8_0.gguf \
  --prompt "法国的首都是" --max-tokens 30
```

对应 chat template（`src/models/llama/trunk/forward.rs:199`）：

```
<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n
```

## 2. Granite

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/granite-XXXX-Q8_0.gguf \
  --prompt "Explain quantum entanglement briefly."
```

Granite 用 Granite 原生 chat template（`forward.rs:192-195`）：

```
<|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|>\n
<|start_of_role|>assistant<|end_of_role|>
```

附加处理（见 `src/app/text.rs:99-110` 注释）：

- `granite.attention.scale` metadata 若存在会覆盖默认的 `1/sqrt(n_embd_head)`
- 已确认 `src/models/llama/trunk/forward.rs` 在 Granite 路径下使用这个 scale
- 文档 [OPTIMIZATION.md](../../OPTIMIZATION.md) 中 Apple Silicon 性能基准
  使用 `Qwen3-0.6B`，不覆盖 Granite

## 3. Nanbeige

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/nanbeige-XXXX.gguf \
  --prompt "请继续。"
```

- Nanbeige 是 base 模型，**没有 chat template**（`forward.rs:196-198`）：
  prompt 直接喂进模型，由 BOS 标记生成起点。
- `add_special = arch == "nanbeige"`（`forward.rs:206`）。
- 状态：`Experimental`。对应 merge commit 标题明确写「未成功」。
- Tokenizer：SPM，仓库通过 `llama` trunk 路由。

## 4. CLI 路由速查

| GGUF `general.architecture` | chat template | BOS | 状态 |
|---|---|---|---|
| `llama`（含 MiniCPM5、Granite 之外的 Llama 家族） | Qwen2 风格 | 视 `add_bos_token` | `Verified`（MiniCPM5）/ 默认 |
| `granite` | Granite 风格 | 视 `add_bos_token` | `Supported` |
| `nanbeige` | 无 | 强制 add_special | `Experimental` |

## 5. 与 llama.cpp 的对齐

`docs/REFERENCE_IMPLEMENTATIONS.md` 中**通用 scalar 位级回归**使用
`llama.cpp @ 749f688fcaa4c472ec034b08cb8a907c45cfaa02`：

```bash
tools/parity/build_llama_oracle.sh
cargo test --test inference_parity
```

Granite / Nanbeige **没有**专属 pinned commit 与 build 脚本，跑对齐只能
临时挑一个 llama.cpp 提交。

## 6. 服务端模式

```bash
cargo run --release --bin server -- \
  --model models/MiniCPM5-1B-Q8_0.gguf \
  --host 0.0.0.0 --port 8080 --threads 4
```

`gemma4 / hunyuan / lfm2 / llama` 等文本架构都按 CLI 选项暴露（`--host` /
`--port` 是 server 端唯一额外参数）。

## 7. 已确认的限制 / 边界

| 范围 | 行为 |
|---|---|
| Granite 非 greedy 解码 | 未限制；但只保证 `add_special` 与模板正确 |
| Nanbeige | Experimental；不保证端到端正确性 |
| Granite `attention.scale` 缺失 | 回退到 `1/sqrt(n_embd_head)` |

## 8. 相关源码索引

- `src/models/llama/trunk/forward.rs` — forward + chat template 分派
- `src/models/llama/trunk/config.rs` — 配置加载
- `src/app/text.rs:99-110` — `llama` / `granite` / `nanbeige` CLI 路由
- `src/prompt.rs` — Hunyuan 与其他 chat template 构造（Granite 由 forward.rs 内联）
- `docs/REFERENCE_IMPLEMENTATIONS.md` — 通用 scalar pinned commit
- `docs/SUPPORTED_MODELS.md` — 验证状态

## 9. JEV 决策评分

`--jev` 是 OpenJEV 风格的 single-forward-pass 决策评分模式：跳过自回归
生成，直接读最后一层 logits 对候选 label tokens (A/B/C/…) 做 softmax。
通用协议、3 种 mode、JSON 输出、已知限制见
[`docs/develop/jev.md`](../develop/jev.md) 和
[`docs/usage/qwen3.md` §10](qwen3.md)。

### 9.1 Arch 路由

| Arch | JEV 路径 |
|---|---|
| `llama` / `qwen2_2` / `minicpm` | `app/text.rs::run_jev_decision_llama` → `llama::run_forward_logits_llama` |
| `granite` / `k2-horizon` | 同上（chat template 不同） |
| `nanbeige` | 同上（base model，无 chat template） |

### 9.2 Chat template 差异（per-arch）

| Arch | Prompt 模板 |
|---|---|
| `llama` / `qwen2_2` | `system\n{sys}\nuser\n{payload}\nassistant\n` |
| `granite` | `<\|start_of_role\|>system<\|end_of_role\|>{sys}<\|end_of_text\|>\n<\|start_of_role\|>user<\|end_of_role\|>{payload}<\|end_of_text\|>\n<\|start_of_role\|>assistant<\|end_of_role\|>` |
| `k2-horizon` | 同 granite |
| `nanbeige` | 无 chat template，直接 `{sys}\n\n{payload}\n\nAnswer:` |

### 9.3 示例

```bash
# MiniCPM5-1B Q8_0 — Choice mode
rust-model-inference --model models/MiniCPM5-1B-Q8_0.gguf \
  --jev --jev-context "用户问的是航空公司的行李规定" \
  --jev-question "这是哪个业务领域？" \
  --jev-option "退款" --jev-option "行李" --jev-option "里程" \
  --threads 4

# Granite — Binary mode
rust-model-inference --model models/granite-Q8_0.gguf \
  --jev --jev-context "天空乌云密布，能听到远处雷声" \
  --jev-question "现在在下雨吗？" \
  --jev-option "是的" --jev-option "没有" --jev-positive A \
  --threads 4

# Nanbeige — Score mode（base model，准确率可能受限）
rust-model-inference --model models/nanbeige-4.2-3B-Q8_0.gguf \
  --jev --jev-context "今天股市整体上涨，科技板块表现强劲" \
  --jev-question "市场情绪如何？" \
  --jev-option "极度乐观:5" --jev-option "乐观:4" --jev-option "中性:3" \
  --threads 4
```

> ⚠️ 注意：Nanbeige 是 base model，JEV 输出概率分布但准确率
> 有限。Granite 是 instruct-tuned，score mode 表现更可靠。