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

## 3. Nanbeige4.2-3B

```bash
cargo run --profile release-fast --bin rust-model-inference -- \
  --model /Users/gouzi/Documents/git/rust-model-inference/models/Nanbeige4.2-3B-GGUF/Nanbeige4.2-3B-Q8_0.gguf \
  --prompt "请只回答：北京是哪个国家的首都？" \
  --max-tokens 32 --temp 0 --threads 4 --max-context 256
```

- GGUF architecture 为 `nanbeige`，SPM tokenizer；22 层物理权重循环两次，44 个逻辑层各自保存 KV，第一轮末尾应用 `output_norm`。
- Q/K/V head dimension 为 128，不能从 `3072 / 48` 推导；Q/output attention width 为 6144，KV width 为 1024。
- 提供的聊天模型使用内嵌 ChatML 默认 system turn。默认关闭 thinking；`--thinking` 使用 `<think>\n` 后缀。没有聊天模板的旧 base GGUF 保留原始 prompt 输入。
- 验证文件：`Nanbeige4.2-3B-Q8_0.gguf`，4,434,787,248 bytes，SHA-256 `76627e550979d8ea5746cb11922ad10352d91546aa540c0e8292522f8dd9c2b5`；201 tensors（156 Q8_0、45 F32）。
- Oracle：llama.cpp `b96806d96061049a5b574269b049bf6241d63d46`，CPU、1 thread、flash attention 关闭；分别验证标量/F32 KV、ARM64 NEON/F32 KV 和默认 NEON/F16 KV。`Hello` 的 IDs 为 `[166100,23877]`，两步 greedy 为 `[152518,324]`；每种模式比较 1733 条记录，含 embedding、44 层各 13 个检查点、轮间归一化和每步 166144 个 F32 logits 的原始位模式，CLI 计算和 Session 输出均逐位一致。
- NEON/F16 KV 另验证 `你好，世界！` 的五个输入 token 和四步 greedy，4618 条记录逐位一致。中文、Unicode、空白、特殊 token 和空输入的六组 Tokenizer IDs 固定在 `tests/nanbeige.rs`。
- F32/F16 KV 下，Rust 单线程单 token 与 4 线程 batch=2 中文 prefill 的完整最终 logits，在标量和 NEON 两种模式分别逐位一致。普通 NEON/F16 CLI 中文聊天通过；其他量化、型号、CPU 架构、JEV 和 server 未纳入本次验证。共享 Q8_0 dispatch 改用现有 ggml NRC1 累加顺序；性能影响未测量。

标量回归（Oracle 构建只操作临时副本）：

```bash
MODEL=/Users/gouzi/Documents/git/rust-model-inference/models/Nanbeige4.2-3B-GGUF/Nanbeige4.2-3B-Q8_0.gguf
ORACLE=$(sh tools/oracle/nanbeige/build_oracle.sh /Users/gouzi/Documents/git/llama.cpp | tail -n 1)
RMI_PARITY_TRACE=/tmp/nanbeige-oracle.jsonl "$ORACLE" -m "$MODEL" -p Hello -n 2
RMI_NANBEIGE_MODEL="$MODEL" RMI_NANBEIGE_ORACLE_TRACE=/tmp/nanbeige-oracle.jsonl RMI_SCALAR=1 \
  cargo test --profile release-fast --features parity-trace --test nanbeige -- --include-ignored --test-threads=1
```

NEON/F16 回归（F32 时删去两处 `RMI_NANBEIGE_F16=1`）：

```bash
ORACLE=$(RMI_ORACLE_SIMD=1 sh tools/oracle/nanbeige/build_oracle.sh /Users/gouzi/Documents/git/llama.cpp | tail -n 1)
RMI_NANBEIGE_F16=1 RMI_PARITY_TRACE=/tmp/nanbeige-neon-f16.jsonl "$ORACLE" -m "$MODEL" -p '你好，世界！' -n 4
RMI_NANBEIGE_F16=1 RMI_NANBEIGE_PROMPT='你好，世界！' \
  RMI_NANBEIGE_MODEL="$MODEL" RMI_NANBEIGE_ORACLE_TRACE=/tmp/nanbeige-neon-f16.jsonl \
  cargo test --profile release-fast --features parity-trace --test nanbeige nanbeige_matches_oracle_bit_for_bit -- --ignored
```

`RMI_SCALAR` 仅在 `parity-trace` 构建生效；普通构建继续使用现有 SIMD。检查保留首个分叉的 trace 文件，全部通过后清理 Rust trace。

## 4. CLI 路由速查

| GGUF `general.architecture` | chat template | BOS | 状态 |
|---|---|---|---|
| `llama`（含 MiniCPM5、Granite 之外的 Llama 家族） | Qwen2 风格 | 视 `add_bos_token` | `Verified`（MiniCPM5）/ 默认 |
| `granite` | Granite 风格 | 视 `add_bos_token` | `Supported` |
| `nanbeige` | Nanbeige4.2 ChatML；无模板时 raw prompt | GGUF add_bos_token | `Verified`（Q8_0 标量/F32、NEON/F32/F16 KV） |

## 5. 与 llama.cpp 的对齐

`docs/REFERENCE_IMPLEMENTATIONS.md` 中**通用 scalar 位级回归**使用
`llama.cpp @ 749f688fcaa4c472ec034b08cb8a907c45cfaa02`：

```bash
tools/oracle/shared/build_llama_oracle.sh
cargo test --test inference_parity
```

Nanbeige4.2 使用第 3 节中的独立固定版本和 Oracle 构建脚本。
Granite 尚无专属固定版本与构建脚本。

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
| Nanbeige4.2-3B Q8_0 | 标量/F32、NEON/F32/F16 KV 逐位验证；其他量化、CPU 架构、JEV 和 server 未验证 |
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
| `llama` / `qwen2_2` / `minicpm` | `app/jev/single/llama.rs::run_jev_decision_llama` → `llama::run_forward_logits_llama` |
| `granite` / `k2-horizon` | 同上（chat template 不同） |
| `nanbeige` | 保留历史路由；本次 SPM 聊天 GGUF 的 JEV 未验证 |

### 9.2 Chat template 差异（per-arch）

| Arch | Prompt 模板 |
|---|---|
| `llama` / `qwen2_2` | `system\n{sys}\nuser\n{payload}\nassistant\n` |
| `granite` | `<\|start_of_role\|>system<\|end_of_role\|>{sys}<\|end_of_text\|>\n<\|start_of_role\|>user<\|end_of_role\|>{payload}<\|end_of_text\|>\n<\|start_of_role\|>assistant<\|end_of_role\|>` |
| `k2-horizon` | 同 granite |
| `nanbeige` | 历史 JEV 模板为 `{sys}\n\n{payload}\n\nAnswer:`；不等同于新版 ChatML |

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

```

> Nanbeige4.2 的 SPM 聊天模型尚未接入 JEV 的 BPE scorer，本次支持范围是文本 CLI 和 LlamaSession。
