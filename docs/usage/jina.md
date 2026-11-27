# jina-bert-v2 家族用法

本仓库对 `jina-embeddings-v2-base-en`、`jina-reranker-v1-turbo-en` 等
`arch = "jina-bert-v2"` 模型的端到端命令行示例。

> 通用前置：构建 `cargo build --release --bin rust-model-inference`。
> 路径以仓库根为工作目录为前提；GGUF 路径请按本地调整。

## 1. Embedding（jina-embeddings-v2-base-en）

`jina-bert-v2` arch 与 `bert` arch 走同一张图（共享 `src/models/bert_family/`），
唯一区别是 ALiBi（无条件 `f_max_alibi_bias = 8.0`）+ GEGLU FFN + 必填的
`token_types.weight`。`--embedding` 与其它 BERT 家族变体走完全相同的
入口 `app::run_embedding` → `bert_family::run_embedding`。

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/jina-embeddings-v2-base-en-Q8_0.gguf \
  --embedding --prompt "What is the capital of France?"
```

输出 768 维 L2-归一化向量。`tests/jina_v2_base_en.rs` 覆盖 metadata
契约、`tokenizer.ggml.model="bert"` WPM 派发、语义排序。

## 2. 跨编码器 Rerank（jina-reranker-v1-turbo-en）

38M 的 `jina-bert-v2` 变体，专门为 cross-encoder rerank 训练，比
`jina-embeddings-v2-base-en`（137M / 12 层 / 768 维）小很多 —— 仅
6 层 / 384 维 / 12 heads；GGUF 顶部多 `cls.weight [384]` + `cls.bias [1]`
线性头（不存在于纯 embedding GGUF）。
`tokenizer.ggml.model = "gpt2"`（BPE 而非 WordPiece）走
`BPETokenizer::from_gguf_metadata` + `pre = "jina-v1-en"`。

通过主 binary 的 `--rerank` 入口（不再有独立的 `jina_rerank` 二进制，
`app::run_rerank` 按 arch 自动分发）：

```bash
# CLI：--rerank-doc 重复多次，或 --rerank-documents <newline-separated FILE>
cargo run --release --bin rust-model-inference -- \
  --model models/jina-reranker-v1-turbo-en-GGUF/Jina-Bert-Implementation-38M-F16.gguf \
  --rerank --rerank-query "Machine learning is" \
  --rerank-doc "A machine is a physical system that uses power and control movement." \
  --rerank-doc "Learning is the process of acquiring new understanding and knowledge." \
  --rerank-doc "Machine learning is a field of study in artificial intelligence." \
  --rerank-doc "Paris is the capital and most populous city of France."
```

输出 sigmoid 后的 `[0, 1]` 评分（jina 路径返回原始 CLS logit，
CLI 在 `src/main.rs` 里走 sigmoid 与 qwen3 路径的 `yes_prob` 对齐），
排序后打印 `rank\tidx\trelevance_score` + 摘要片段。Paris vs ML
文档集的相关性差距 ~0.013（sigmoid 压缩后差距小，但顺序稳定）。

服务端模式自动检测 `jina-bert-v2` + `cls.weight` + `cls.bias` 并开
`POST /v1/rerank`（Cohere/Jina 兼容 schema）：

```bash
cargo run --release --bin server -- \
  --model models/jina-reranker-v1-turbo-en-GGUF/Jina-Bert-Implementation-38M-F16.gguf \
  --port 8080 --threads 4

curl -s -X POST http://127.0.0.1:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Machine learning is",
    "documents": [
      "A machine is a physical system that uses power and control movement.",
      "Learning is the process of acquiring new understanding and knowledge.",
      "Machine learning is a field of study in artificial intelligence.",
      "Paris is the capital and most populous city of France."
    ],
    "top_n": 4
  }'
```

返回响应（`relevance_score` 已 sigmoid 到 `[0, 1]`）：

```json
{
  "id": "rerank-xxxx",
  "model": "jina-reranker-v1-turbo-en",
  "results": [
    {"index": 2, "relevance_score": 0.488857},
    {"index": 1, "relevance_score": 0.479220},
    {"index": 0, "relevance_score": 0.475035},
    {"index": 3, "relevance_score": 0.471123}
  ]
}
```

Rerank 序列 `[BOS] query [EOS] [SEP] doc [EOS]` 严格走
`references/llama.cpp/tools/server/server-common.cpp:1817-1830`（jina-v1-en
的 `seperator_token_id == eos_token_id == 2` 让 mid-stream `[SEP]` 与
末尾 `[EOS]` 折叠为同一 id）。CLS 行（row 0，由 `llm_graph_input_cls`
的 `pos < target_pos` 取最低位）投影过 `cls.weight` + `cls.bias`。

`tests/jina_reranker_v1_turbo_en.rs` 6/6 覆盖：metadata 契约（6 层 /
384 维 / 12 heads / eps=1e-12 / `tokenizer.ggml.model="gpt2"` /
`pre="jina-v1-en"`）、102 张量清单（含 `cls.weight`/`cls.bias`）、
BPE BOS/EOS 包封、embedding 维度 + L2 norm + 全有限值、rerank 排序、
超长 prompt 触发 `prompt is too long` 错误。

## 3. 与 llama.cpp 的数值对齐

`jina-embeddings-v2-base-en` 与 `jina-reranker-v1-turbo-en` 都通过
`tools/oracle/jina_bert_v2/verify.py` 做位级 oracle 对照（见
`tools/oracle/jina_bert_v2/README.md`）。rerank 路径在 main 的
`scalar_mode()` 开关下严格走 f32 × f32 + f64 累积，与 llama.cpp
scalar pipeline bit-equal；非 scalar 路径（默认）在 f32 累加下与
oracle 存在 1-2 ULP 偏差，不影响排序顺序。

## 4. 服务端模式（OpenAI 兼容）

jina 家族的两类输出都通过 server auto-detect：

```bash
# Embedding
cargo run --release --bin server -- \
  --model models/jina-embeddings-v2-base-en-Q8_0.gguf --embedding

# Rerank（auto-detect jina-bert-v2 + cls.weight + cls.bias → POST /v1/rerank）
cargo run --release --bin server -- \
  --model models/jina-reranker-v1-turbo-en-GGUF/Jina-Bert-Implementation-38M-F16.gguf \
  --port 8080 --threads 4
```

## 5. 已确认的限制 / 边界

| 范围 | 行为 |
|------|------|
| Embedding 路径忽略 `cls.weight`/`cls.bias` | 设计如此（专用 rerank 头不参与 embedding forward） |
| Rerank 路径忽略 `cls.token_types` 含 row 2（segment B） | jina-v1-en GGUF 不发 segment B；`token_types.weight [n_embd, 2]` 中只用 row 0 |
| ALiBi 缺失（GGUF 没有 `LLM_KV_ATTENTION_MAX_ALIBI_BIAS` key） | 不影响：`jina-bert-v2.cpp:5` 无条件写 `f_max_alibi_bias = 8.0`，本仓库在 `BertVariant::uses_alibi()` 直接走 jina 分支 |
| GPU 后端（`--features vulkan`） | 可跑，但当前不提供 GPU 位级 Oracle 保证 |

## 6. 相关源码索引

- `src/models/bert_family/compute.rs` — `compute_embedding` / `compute_rerank_score` / `forward_bert_layers` / `populate_embeddings`
- `src/models/bert_family/weights.rs` — `cls_weight`/`cls_bias` 张量加载 + `is_rerank()`
- `src/app/server/mod.rs::is_jina_rerank_gguf` — server-side auto-detect
- `src/app/server/rerank.rs::score_one_doc_jina` — HTTP handler 的 jina scoring 路径（与 CLI 共享 `compute_rerank_score`）
- `src/app/mod.rs::run_rerank` — CLI dispatcher，按 arch 分发到 `bert_family::compute_rerank_score` 或 `qwen3::trunk::score_qwen3_rerank`
- `src/core/tokenizer/mod.rs:413` — `BPETokenizer::from_gguf_metadata` 新接受的 `jina-v1-en` pre_type（jina-reranker 必走）
- `tools/oracle/jina_bert_v2/` — jina-bert-v2 scalar parity oracle + verify.py