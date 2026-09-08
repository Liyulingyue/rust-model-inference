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