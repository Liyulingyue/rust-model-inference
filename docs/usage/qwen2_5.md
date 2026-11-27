# Qwen2.5 1.5B-Instruct 用法

[Qwen2.5-1.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct)
走 `qwen3` arch 入口（GGUF `general.architecture = "qwen2"`，但本仓库
dispatch 统一汇入 `models::qwen3::trunk`）。

> 共用前置：构建 `cargo build --release --bin rust-model-inference`。
> 仓库自带 GGUF：`models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf`
> 来自 `Qwen/Qwen2.5-1.5B-Instruct-GGUF`（ModelScope 上 ggml-org 的官方再量化）。

## 1. 模型规格（contract）

`tests/qwen2_5_1_5b_instruct_q4_k_m.rs` 钉住的 contract（来自
`Qwen/Qwen2.5-1.5B-Instruct-GGUF` 的 GGUF metadata，与官方 re-quantization
匹配，跨量化档稳定）：

| 字段 | 值 |
|---|---|
| `general.architecture` | `qwen2` |
| `qwen2.block_count` | 28 |
| `qwen2.embedding_length` | 1536 |
| `qwen2.attention.head_count` | 12 |
| `qwen2.attention.head_count_kv` | 2（GQA-6：12 Q heads / 2 KV heads） |
| `qwen2.feed_forward_length` | 8960 |
| `qwen2.context_length` | 32768 |
| `head_dim = n_embd / n_head` | 128 |
| RoPE `theta` | 1e6 |
| RMSNorm `eps` | 1e-6 |
| `output.weight` | untied（与 `token_embd.weight` 分离） |
| `vocab_size` | 151,936 |
| ChatML `<|im_start|>` | 151644 |
| ChatML `<|im_end|>` | 151645 |
| Q4_K_M 文件大小 | 1.1 GB |
| 张量数 | 339（28 层 × 12 张量 + 3 head） |

## 2. CLI：文本生成

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --prompt "What is the capital of France?" --max-tokens 30
```

端到端实测（`tests/qwen2_5_1_5b_instruct_q4_k_m.rs::q4_k_m_tokenizer_matches_qwen2_5_chatml`
断言 ChatML 包封）："What is the capital of France?" → "Paris"；
"1+1等于几?" → "1+1=2"。

## 3. 派生模型

`Qwen2.5-1.5B-Instruct` 同时是 `harshatheg/Qwen-2.5-1B-RLCD`（"Parallel
Constrained Decoding" for Apple Silicon MLX）所驱动的底座。本仓库不
直接跑 RLCD，但 Q4_K_M GGUF 的 contract 测试钉住了同一组权重，
因此任何对该模型的扩展（蒸馏、量化、constraint schema 改动）都
可以对照本仓库的 q4_k_m_*_test 来做 sanity check。

## 4. 测试

```bash
RMI_QWEN2_5_1_5B_INSTRUCT_Q4_K_MODEL=\\
  models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf \\
  cargo test --release --test qwen2_5_1_5b_instruct_q4_k_m
```

`q4_k_m_*` 三个测试在 env var 指向的 GGUF 存在时跑出 contract + tensor
inventory + ChatML tokenizer 的 bit-equal assertions；env var 缺失
或文件不存在时静默 no-op（不挂 CI）。
