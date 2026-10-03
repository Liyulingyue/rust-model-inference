//! # BERT encoder family (`arch = "bert"` / `"jina-bert-v2"`)
//!
//! Contract pinned against the local read-only oracle
//! `references/llama.cpp/src/models/bert.cpp` + `jina-bert-v2.cpp`, and the
//! GGUF tensor inventory of `jina-embeddings-v2-base-en-q8_0.gguf`.
//!
//! `llama.cpp` serves 5 architectures from the single `bert.cpp` graph
//! (`bert`, `jina-bert-v2`, `jina-bert-v3`, `nomic-bert`, `nomic-bert-moe`).
//! This module implements the two variants that are reachable today —
//! `bert` and `jina-bert-v2` — because they differ in only three knobs and
//! the extra code is shared, not speculative:
//!
//! | knob | `bert` | `jina-bert-v2` |
//! |---|---|---|
//! | position | absolute `pos_embd` added to the embedding | **none**; ALiBi instead |
//! | segment | `type_embd` optional | **required** (`token_types.weight`) |
//! | FFN | GELU, single `ffn_up` (`LLM_FFN_SEQ`) | GEGLU, `ffn_gate` + `ffn_up` (`LLM_FFN_PAR`) |
//!
//! Everything else is identical and shared: post-norm LayerNorm sandwich,
//! Q/K/V + attention-output biases, bidirectional attention with no KV cache,
//! `1/sqrt(head_dim)` score scale, and a raw transformer output with no
//! LM head.
//!
//! ## Oracle citations
//!
//! - `causal_attn = false` + `build_attn_inp_no_cache()` →
//!   `bert.cpp:7,88` (bidirectional, no KV cache).
//! - Per-layer body (residual → LayerNorm → residual → LayerNorm) →
//!   `bert.cpp:139-197`.
//! - ALiBi mask value `-|p0 - p1|` → `llama-graph.cpp:441`; the per-head slope
//!   multiply happens in `soft_max_ext` → `ggml-cpu/ops.cpp:8944`.
//! - Score scale `1/sqrt(n_embd_head)` → `bert.cpp:147`.
//! - `f_attention_scale` / `f_max_alibi_bias = 8.0f` → `jina-bert-v2.cpp:5`.
//! - Mean pooling semantics → `llama-graph.cpp:3717`; L2 normalize is applied
//!   by the example layer with `embd_normalize = 2` (euclidean) →
//!   `common/common.cpp:1893` and `common/common.h:615`.

pub mod compute;
pub mod weights;

pub use compute::{compute_embedding, compute_rerank_score, run_embedding};
