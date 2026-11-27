//! # Gemma Embedding (EmbeddingGemma-300M)
//!
//! `arch = "gemma-embedding"` — an encoder-style Gemma-3 variant used as a
//! sentence embedder. Contract verified against the local read-only oracle
//! `references/llama.cpp/src/models/gemma-embedding.cpp` (no code changes there)
//! and the GGUF tensor inventory of `embeddinggemma-300M-Q8_0.gguf`.
//!
//! ## What makes it different from the `gemma4` trunk
//!
//! | aspect | gemma4 trunk | gemma-embedding |
//! |---|---|---|
//! | attention | causal + KV cache | **bidirectional, no KV cache** |
//! | sliding window | gateway pattern (some layers) | **symmetric window on every layer but `il % 6 == 5`** |
//! | QK norm | n/a | **present, applied *before* RoPE, per head** |
//! | norms per layer | 2 | **4 (pre/post sandwich around attention and FFN)** |
//! | FFN activation | gelu | gelu (`geglu`), gate & up are separate projections |
//! | output | LM-head logits | `output_norm` → mean pool → `dense_2`/`dense_3` |
//!
//! ## Reference semantics (oracle line numbers)
//!
//! - `causal_attn = false` + `build_attn_inp_no_cache()` → full bidirectional
//!   attention, no KV cache (`gemma-embedding.cpp:7-8`).
//! - `swa_type = LLAMA_SWA_TYPE_SYMMETRIC`, `n_swa = 512` from
//!   `attention.sliding_window`; a key is masked when
//!   `|p1 - p0| > n_swa / 2` (`llama-hparams.h:472-500`).
//! - `load_swa_pattern(ml, 6)` → `is_swa(il) = (il % 6 < 5)`; layers with
//!   `il % 6 == 5` are dense/global and use `rope.freq_base`, the rest use
//!   `rope.freq_base_swa` (`llama-model.cpp:3378-3395`, `llama-model.cpp:2318`).
//! - QK norm is RMSNorm over `n_embd_head_k` applied per head, **before**
//!   RoPE (`gemma-embedding.cpp:105-118`).
//! - `f_attention_scale = 1 / sqrt(n_embd_head_k)` scales Q after RoPE
//!   (`gemma-embedding.cpp:27,122`).
//! - Token embeddings are scaled by `sqrt(n_embd)` before layer 0
//!   (`gemma-embedding.cpp:89`).
//! - FFN is `LLM_FFN_GELU` + `LLM_FFN_PAR` → `ggml_geglu_split` =
//!   `gelu(x) * g` (`llama-graph.cpp:1866-1869`, `vec.h:ggml_vec_geglu_f32`).
//! - Final: `output_norm` RMSNorm → mean pooling → `dense_2` (768→3072) →
//!   `dense_3` (3072→768). `dense_1` is intentionally absent in this GGUF.
//! - `attention.layer_norm_rms_eps` is **0** in this GGUF, so we fall back to
//!   `model_config.norm_eps`-equivalent clamping to avoid 0/0 (see
//!   [`safe_eps`]).

pub mod compute;
pub mod weights;

pub use compute::{compute_embedding, run_embedding};
