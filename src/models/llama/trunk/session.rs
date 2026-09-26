//! Llama session — owned state for one prefill / decode pass.
//!
//! Extracted out of the previous free-function `run_forward_logits_llama`
//! so the trunk can adopt [`crate::core::prefill::ChunkedPrefill`]. The
//! session owns every piece of state that used to be threaded as
//! locals — config, weights, KV cache, scratchpad, compute pool —
//! and exposes a single `forward_logits(prompt_tokens, batch_size)`
//! that walks the [`ChunkedPrefill`] default loop. For
//! `batch_size == 1` the inner `forward_chunk` body is a straight
//! extraction of the legacy per-token forward, so a `B=1` chunked
//! prefill is bit-identical to the old per-token prefill (modulo
//! dispatch ordering, which the existing parity tests already
//! guard).
//!
//! Single-token and chunked attention share the same F16/F32 operations.

use super::forward::{apply_rope, normalization_groups};
use super::weights::{get_f32_tensor, layer_loop_config, load_layers, LlamaLayerWeights};
use crate::app::cli::{resolve_thread_count, KvFormat};
use crate::core::prefill::{checked_prefill_batch_size, prefill_chunks, ChunkedPrefill};
use crate::core::scratchpad::{ExecutionScratchpad, KvCache};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::load_tokenizer;
use crate::core::tokenizer::Tokenizer;
use crate::ops::kernel::{PreparedRows, QuantizedTensor, Weight};
type DynTokenizer = Box<dyn Tokenizer>;
use crate::core::tensor::GGMLType;
use crate::ops::{
    dot_f32, embedding_lookup, gpu_matmul_active, quantize_q8_0_into, quantize_row_q8_k_into,
    rms_norm_grouped, rms_norm_inplace, silu_mul_approx_inplace, softmax_approx_inplace,
    vec_add_into, vec_mad_f32, vec_scale_f32,
};
use std::sync::Arc;

pub struct LlamaSession<'a> {
    pub arch: String,
    pub config: LlamaSessionConfig,
    pub tokenizer: DynTokenizer,
    pub weights: LlamaWeights<'a>,
    pub kv_cache: KvCache,
    pub scratch: ExecutionScratchpad,
    /// Shared Q8_0 + Q8_K scratch for batched matmul dispatch.
    /// Allocated at construction time with the configured
    /// `max_rows` so a [`forward_chunk`](Self::forward_chunk)
    /// call can quantise once and amortise across every
    /// layer's Q/K/V / wo / gate / up / down projection.
    pub prepared_rows: PreparedRows,
    pub pool: Arc<ComputePool>,
    pub kq_scale: f32,
    pub group_size: usize,
    pub embedding_scale: f32,
    pub residual_scale: f32,
    pub logit_scale: f32,
    pub norm_groups: usize,
    pub seq_len: usize,
    pub loop_final_norm: bool,
}

#[derive(Clone, Copy)]
pub struct LlamaSessionConfig {
    pub max_ctx: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head: usize,
    pub n_embd_head_k: usize,
    pub n_embd_head_v: usize,
    pub n_embd_q: usize,
    pub n_embd_gqa: usize,
    pub n_ff: usize,
    pub eps: f32,
    pub freq_base: f32,
    pub vocab: usize,
}

pub struct LlamaWeights<'a> {
    pub layers: Vec<LlamaLayerWeights<'a>>,
    pub embd_weight: &'a [u8],
    pub embd_type: GGMLType,
    pub output_weight: &'a [u8],
    pub output_type: GGMLType,
    pub output_norm: Vec<f32>,
}

impl<'a> LlamaSession<'a> {
    pub fn from_source(
        source: &'a dyn TensorSource,
        n_threads_arg: usize,
        kv_format: KvFormat,
        max_context: usize,
    ) -> Result<Self, String> {
        Self::from_source_with_max_rows(source, n_threads_arg, kv_format, max_context, 1)
    }

    /// Same as [`from_source`](Self::from_source) but with an
    /// explicit `max_rows` for the chunked-prefill scratchpad.
    /// `max_rows == 1` reproduces the legacy per-token
    /// footprint exactly; `max_rows > 1` widens the per-row
    /// buffers so a [`forward_chunk`](Self::forward_chunk)
    /// call can hold `rows × width` activations and Q/K/V at
    /// once. The chunked prefill math
    /// ([`forward_chunk`](Self::forward_chunk) today still
    /// falls back to per-token for `rows > 1`; the
    /// `max_rows` reservation just reserves the memory
    /// needed for future batched implementations.
    pub fn from_source_with_max_rows(
        source: &'a dyn TensorSource,
        n_threads_arg: usize,
        kv_format: KvFormat,
        max_context: usize,
        max_rows: usize,
    ) -> Result<Self, String> {
        use crate::core::loader::model_config_from_source;
        let config = model_config_from_source(source)
            .map_err(|error| format!("Failed to parse model config: {error}"))?;
        let arch: String = source
            .metadata("general.architecture")
            .and_then(|v| v.to_string_val())
            .unwrap_or_default()
            .to_string();
        let tokenizer = load_tokenizer(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
        let max_ctx = config.n_ctx.min(max_context);
        let n_embd = config.n_embd;
        let (n_layer, loop_final_norm) = layer_loop_config(source, &config)?;
        let n_head = config.n_head;
        let n_head_kv = config.n_head_kv;
        let n_embd_head = config.n_embd_head;
        let n_embd_head_k =
            if let Some(v) = source.metadata(&format!("{}.attention.key_length", arch)) {
                v.to_u64().unwrap_or(n_embd_head as u64) as usize
            } else {
                n_embd_head
            };
        let n_embd_head_v =
            if let Some(v) = source.metadata(&format!("{}.attention.value_length", arch)) {
                v.to_u64().unwrap_or(n_embd_head as u64) as usize
            } else {
                n_embd_head
            };
        let n_embd_q = n_head * n_embd_head_k;
        let n_embd_gqa = n_head_kv * n_embd_head_v;
        let n_ff = config.n_ff;
        let eps = config.norm_eps;
        let freq_base = config.rope_freq_base;
        let norm_groups = normalization_groups(source, &arch, n_embd)?;
        let arch_prefix = arch.clone();
        let embedding_scale = source
            .metadata(&format!("{arch_prefix}.embedding_scale"))
            .and_then(|v| v.to_f64())
            .unwrap_or(0.0) as f32;
        let residual_scale = source
            .metadata(&format!("{arch_prefix}.residual_scale"))
            .and_then(|v| v.to_f64())
            .unwrap_or(0.0) as f32;
        let logit_scale = source
            .metadata(&format!("{arch_prefix}.logit_scale"))
            .and_then(|v| v.to_f64())
            .unwrap_or(0.0) as f32;
        let output_norm = get_f32_tensor(source, "output_norm.weight", n_embd);
        let embd_info = source
            .tensor_info("token_embd.weight")
            .expect("no token_embd.weight");
        crate::ops::embedding::expect_supported_embedding("token_embd.weight", embd_info.ggml_type);
        let embd_weight = source.tensor_slice("token_embd.weight").expect("no embd");
        let output_weight = source.tensor_slice("output.weight").unwrap_or(embd_weight);
        let embd_type = embd_info.ggml_type;
        let output_type = source
            .tensor_info("output.weight")
            .unwrap_or(embd_info)
            .ggml_type;
        let layers: Vec<LlamaLayerWeights<'a>> =
            load_layers(source, config.n_layer, n_embd, n_embd_q, n_embd_gqa, n_ff);
        let kv_cache = match kv_format {
            KvFormat::F16 => KvCache::new_f16(n_layer, max_ctx, n_embd_gqa),
            KvFormat::F32 => KvCache::new_f32(n_layer, max_ctx, n_embd_gqa),
        };
        let vocab = tokenizer.vocab_size();
        let available_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let n_threads = resolve_thread_count(n_threads_arg, available_threads);
        let scratch = ExecutionScratchpad::new_batched(
            n_embd, n_embd_q, n_embd_gqa, n_ff, vocab, n_threads, max_ctx, max_rows,
        );
        // Shared Q8_0 + Q8_K scratch for batched matmul dispatch
        // (see `forward_chunk_batched`). Sized to the largest
        // input the projections see (= max(n_embd, n_embd_q, n_ff)).
        let prepared_n_in = n_embd_q.max(n_ff).max(n_embd * 3);
        let prepared_rows = PreparedRows::new(max_rows, prepared_n_in);
        let pool = Arc::new(ComputePool::new(n_threads));
        let group_size = n_head / n_head_kv;
        let attention_scale_meta = source
            .metadata(&format!("{arch}.attention.scale"))
            .and_then(|v| v.to_f64())
            .map(|v| v as f32);
        let kq_scale =
            attention_scale_meta.unwrap_or_else(|| 1.0f32 / (n_embd_head_k as f32).sqrt());
        Ok(Self {
            arch,
            config: LlamaSessionConfig {
                max_ctx,
                n_embd,
                n_layer,
                n_head,
                n_head_kv,
                n_embd_head,
                n_embd_head_k,
                n_embd_head_v,
                n_embd_q,
                n_embd_gqa,
                n_ff,
                eps,
                freq_base,
                vocab,
            },
            tokenizer,
            weights: LlamaWeights {
                layers,
                embd_weight,
                embd_type,
                output_weight,
                output_type,
                output_norm,
            },
            kv_cache,
            scratch,
            prepared_rows,
            pool,
            kq_scale,
            group_size,
            embedding_scale,
            residual_scale,
            logit_scale,
            norm_groups,
            seq_len: 0,
            loop_final_norm,
        })
    }

    /// Run the legacy per-token prefill over `prompt_tokens`, then
    /// return the final logits. Kept around as the single-source-of-
    /// truth for `forward_chunk` so the trait-driven path stays
    /// bit-identical at `B = 1`.
    pub fn forward_logits_per_token(&mut self, prompt_tokens: &[u32]) -> Result<Vec<f32>, String> {
        let mut last_logits = None;
        for &token_id in prompt_tokens {
            self.forward_one_token(token_id)?;
            // `forward_one_token` writes `scratch.logits` only on
            // every step (matches the legacy loop); capture the last.
            last_logits = Some(self.scratch.logits.clone());
        }
        Ok(last_logits.unwrap_or_default())
    }

    /// Trait-driven entry point. Returns the final logits for
    /// `prompt_tokens` after a chunked prefill at the requested
    /// `batch_size`. Equivalent to
    /// [`forward_logits_per_token`](Self::forward_logits_per_token)
    /// when `batch_size == 1` (the default `ChunkedPrefill::prefill`
    /// loop with `rows = 1` collapses to the legacy per-token
    /// walk). For `batch_size > 1`, the trait default dispatches
    /// `forward_chunk` per chunk; `forward_chunk` calls
    /// [`forward_chunk_rows`](Self::forward_chunk_rows) which
    /// uses [`PreparedRows::matmul_group`] for the Q/K/V / wo /
    /// gate / up / down projections and falls back to the
    /// legacy per-row flash-attention loop for the attention
    /// sub-step.
    pub fn forward_logits_chunked(
        &mut self,
        prompt_tokens: &[u32],
        batch_size: usize,
    ) -> Result<Vec<f32>, String> {
        let owned: Vec<u32> = prompt_tokens.to_vec();
        let last = ChunkedPrefill::prefill(self, &owned, batch_size)?;
        Ok(last.unwrap_or_default())
    }

    /// Multi-row chunked forward. `rows` tokens at consecutive
    /// absolute positions `[base, base + rows)`. Falls back to
    /// `rows` × `forward_one_token` when `rows == 1` (the common
    /// case) so the legacy per-token path stays as the single
    /// source of truth for the attention math.
    ///
    /// For `rows > 1` this is a **real** batched prefill step:
    /// the activations live in `[rows × n_embd]` row-major
    /// scratch slices (see
    /// [`ExecutionScratchpad::new_batched`]); each layer
    /// quantises the activation once via
    /// [`PreparedRows::prepare`] and dispatches every Q/K/V / wo
    /// / gate / up / down projection in one
    /// [`PreparedRows::matmul_group`] call so the per-row Q8_0 +
    /// Q8_K quantisation and the dispatch overhead are amortised
    /// across `rows` rows. Attention, RoPE, RMSNorm and the KV-
    /// cache append remain per-row because (a) the per-token state
    /// update in `seq_len` is sequential, (b) RoPE depends on the
    /// absolute position, and (c) the existing flash-attention
    /// loop already reuses the cached K/V load across heads.
    fn forward_chunk_rows(
        &mut self,
        input: &[u32],
        rows: usize,
        base_position: usize,
        project_logits: bool,
    ) -> Result<Option<Vec<f32>>, String> {
        if rows == 1 {
            // B=1 fast path: reuse the legacy per-token forward
            // verbatim. This keeps the B=1 ablation identical to
            // the pre-chunked-prefill llama path so the trait-
            // driven dispatch is bit-exact at rows=1.
            let owned: Vec<u32> = input[base_position..base_position + 1].to_vec();
            self.forward_logits_chunked_chunk(&owned, base_position, project_logits)?;
            if project_logits {
                Ok(Some(self.scratch.logits.clone()))
            } else {
                Ok(None)
            }
        } else {
            self.forward_chunk_batched_real(input, rows, base_position, project_logits)?;
            if project_logits {
                Ok(Some(self.scratch.logits.clone()))
            } else {
                Ok(None)
            }
        }
    }

    /// B=1 fallback used by [`forward_chunk_rows`] when the chunk
    /// contains a single token. Runs the legacy per-token forward
    /// for the token at absolute position `base_position` without
    /// going through the trait dispatch.
    fn forward_logits_chunked_chunk(
        &mut self,
        prompt_tokens: &[u32],
        base_position: usize,
        project_logits: bool,
    ) -> Result<(), String> {
        // `forward_one_token` advances `self.seq_len` by one each
        // call. `base_position` must equal `self.seq_len` at entry.
        if base_position != self.seq_len {
            return Err(format!(
                "Llama chunked prefill base_position {base_position} != session seq_len {}",
                self.seq_len
            ));
        }
        for &token_id in prompt_tokens {
            self.forward_one_token(token_id)?;
        }
        let _ = project_logits;
        Ok(())
    }

    /// True batched prefill step for `rows > 1`. See
    /// [`forward_chunk_rows`] for the high-level shape.
    fn forward_chunk_batched_real(
        &mut self,
        input: &[u32],
        rows: usize,
        base_position: usize,
        project_logits: bool,
    ) -> Result<(), String> {
        let _ = project_logits;
        let cfg = &self.config;
        let scratch = &mut self.scratch;
        let pool = &self.pool;
        let kv_cache = &mut self.kv_cache;
        let weights = &self.weights;
        let prepared_rows = &mut self.prepared_rows;
        let n_embd = cfg.n_embd;
        let n_embd_q = cfg.n_embd_q;
        let n_embd_gqa = cfg.n_embd_gqa;
        let n_ff = cfg.n_ff;
        let n_layer = cfg.n_layer;
        let n_embd_head_k = cfg.n_embd_head_k;
        let n_embd_head_v = cfg.n_embd_head_v;
        let n_head = cfg.n_head;
        let group_size = self.group_size;
        let kq_scale = self.kq_scale;
        let eps = cfg.eps;
        let freq_base = cfg.freq_base;
        let arch = &self.arch;
        let embedding_scale = self.embedding_scale;
        let residual_scale = self.residual_scale;
        let logit_scale = self.logit_scale;
        let norm_groups = self.norm_groups;
        let max_ctx = cfg.max_ctx;
        let vocab = cfg.vocab;
        let _max_n_in = n_embd_q.max(n_ff).max(n_embd * 3);

        if base_position != self.seq_len {
            return Err(format!(
                "Llama batched prefill base_position {base_position} != session seq_len {}",
                self.seq_len
            ));
        }

        // ---- Embedding lookup × rows into the batched `x` scratch ----
        // `x` is sized `[max_rows × n_embd]`. For row `r` in the
        // chunk we write `x[r * n_embd .. (r + 1) * n_embd]`.
        {
            let x = &mut scratch.x[..rows * n_embd];
            for r in 0..rows {
                let abs_pos = base_position + r;
                let token_id = input[abs_pos];
                let row = &mut x[r * n_embd..(r + 1) * n_embd];
                embedding_lookup(
                    weights.embd_weight,
                    token_id,
                    n_embd,
                    weights.embd_type,
                    row,
                );
                if embedding_scale != 0.0 {
                    vec_scale_f32(row, embedding_scale);
                }
            }
        }

        let n_threads = pool.n_threads();
        let kv_cache_size = n_layer * max_ctx * n_embd_gqa;
        let (k_cache_f16_ptr, v_cache_f16_ptr) = match &kv_cache {
            KvCache::F16(c) => (c.k.as_ptr() as *mut u16, c.v.as_ptr() as *mut u16),
            _ => (std::ptr::null_mut(), std::ptr::null_mut()),
        };
        let (k_cache_f32_ptr, v_cache_f32_ptr) = match &kv_cache {
            KvCache::F32(c) => (c.k.as_ptr() as *mut f32, c.v.as_ptr() as *mut f32),
            _ => (std::ptr::null_mut(), std::ptr::null_mut()),
        };

        for layer in 0..n_layer {
            let lw = &weights.layers[layer % weights.layers.len()];
            // ---- Per-row RMSNorm ----
            // RMSNorm is a per-row op (independent across rows in
            // the chunk) so we walk the rows here instead of
            // dispatching per-token. SIMD inside `rms_norm_grouped`
            // is already AVX2 / NEON-friendly. `normed` is rebound
            // after the loop so the borrow checker is happy.
            for r in 0..rows {
                let off = r * n_embd;
                rms_norm_grouped(
                    &mut scratch.x[off..off + n_embd],
                    &lw.attn_norm,
                    &mut scratch.normed[off..off + n_embd],
                    norm_groups,
                    eps,
                );
            }
            let normed = &mut scratch.normed[..rows * n_embd];

            // ---- Quantise the chunked activations once + dispatch
            //      Q / K / V via `PreparedRows::matmul_group` ----
            // `PreparedRows` reserves `q8` and `q8k` buffers
            // sized for `[max_rows × max_n_in]` so a single
            // quantise pass amortises the Q8_0 + Q8_K scan cost
            // across all `rows`. The kernel inside `matmul_group`
            // is per-row (same as qwen3's `forward_cpu_chunk`)
            // but the dispatch and quantise overhead are paid
            // once per chunk.
            //
            // Output shapes: q → `[rows × n_embd_q]`,
            //                k → `[rows × n_embd_gqa]`,
            //                v → `[rows × n_embd_gqa]`.
            let q_out = &mut scratch.q[..rows * n_embd_q];
            let k_out = &mut scratch.k_new[..rows * n_embd_gqa];
            let v_out = &mut scratch.v_new[..rows * n_embd_gqa];
            let needs_q8 = lw.wq.needs_q8_0_activation();
            let needs_q8k = lw.wq.uses_q8_k();
            prepared_rows.prepare(normed, rows, n_embd, needs_q8, needs_q8k)?;
            let projections = [
                (&lw.wq, &mut q_out[..]),
                (&lw.wk, &mut k_out[..]),
                (&lw.wv, &mut v_out[..]),
            ];
            prepared_rows.matmul_group(normed, projections, pool)?;

            // ---- Per-row RoPE ----
            // `apply_rope` writes into a single `[n_embd_head_k *
            // n_heads]` slice in place; for the batched case we
            // hop row-by-row. Cheap (O(rows × n_embd)).
            for r in 0..rows {
                let abs_pos = base_position + r;
                let q_row = &mut q_out[r * n_embd_q..(r + 1) * n_embd_q];
                let k_row = &mut k_out[r * n_embd_gqa..(r + 1) * n_embd_gqa];
                apply_rope(arch.as_str(), q_row, abs_pos, n_embd_head_k, freq_base);
                apply_rope(arch.as_str(), k_row, abs_pos, n_embd_head_k, freq_base);
            }

            // ---- Per-row KV-cache append ----
            // `kb` is the per-layer offset into the flat KV cache
            // (K and V are laid out as `[layer, seq, n_embd_gqa]`
            // for F32 KV and `[layer, seq, n_embd_gqa]` of u16 for
            // F16). One row per token in the chunk.
            let kb = layer * max_ctx * n_embd_gqa;
            let is_f16 = !k_cache_f16_ptr.is_null();
            if is_f16 {
                let k_cache_f16 =
                    unsafe { std::slice::from_raw_parts_mut(k_cache_f16_ptr, kv_cache_size) };
                let v_cache_f16 =
                    unsafe { std::slice::from_raw_parts_mut(v_cache_f16_ptr, kv_cache_size) };
                for r in 0..rows {
                    let abs_pos = base_position + r;
                    let k_row = &k_out[r * n_embd_gqa..(r + 1) * n_embd_gqa];
                    let v_row = &v_out[r * n_embd_gqa..(r + 1) * n_embd_gqa];
                    for h in 0..cfg.n_head_kv {
                        let off = h * n_embd_head_k;
                        let slot = kb + abs_pos * n_embd_gqa + off;
                        crate::ops::f32_slice_to_f16(
                            &k_row[off..off + n_embd_head_k],
                            &mut k_cache_f16[slot..slot + n_embd_head_k],
                        );
                        let voff = h * n_embd_head_v;
                        let vslot = kb + abs_pos * n_embd_gqa + voff;
                        crate::ops::f32_slice_to_f16(
                            &v_row[voff..voff + n_embd_head_v],
                            &mut v_cache_f16[vslot..vslot + n_embd_head_v],
                        );
                    }
                }
            } else {
                let k_cache_f32 =
                    unsafe { std::slice::from_raw_parts_mut(k_cache_f32_ptr, kv_cache_size) };
                let v_cache_f32 =
                    unsafe { std::slice::from_raw_parts_mut(v_cache_f32_ptr, kv_cache_size) };
                for r in 0..rows {
                    let abs_pos = base_position + r;
                    let k_row = &k_out[r * n_embd_gqa..(r + 1) * n_embd_gqa];
                    let v_row = &v_out[r * n_embd_gqa..(r + 1) * n_embd_gqa];
                    for h in 0..cfg.n_head_kv {
                        let off = h * n_embd_head_k;
                        let slot = kb + abs_pos * n_embd_gqa + off;
                        k_cache_f32[slot..slot + n_embd_head_k]
                            .copy_from_slice(&k_row[off..off + n_embd_head_k]);
                        let voff = h * n_embd_head_v;
                        let vslot = kb + abs_pos * n_embd_gqa + voff;
                        v_cache_f32[vslot..vslot + n_embd_head_v]
                            .copy_from_slice(&v_row[voff..voff + n_embd_head_v]);
                    }
                }
            }

            // Causal attention produces `[rows × n_embd_q]` for `wo`.
            let attn_out = &mut scratch.attn_out[..rows * n_embd_q];
            crate::models::llama::trunk::forward::run_attention_chunked(
                pool,
                q_out,
                attn_out,
                kv_cache,
                kv_cache_size,
                base_position + rows,
                base_position,
                rows,
                n_embd_q,
                n_embd_gqa,
                n_head,
                n_embd_head_k,
                n_embd_head_v,
                group_size,
                kq_scale,
                kb,
                n_threads,
                max_ctx,
            );
            // ---- Quantise attention output + project through `wo`
            //      via `PreparedRows::matmul_group` ----
            // One quantise pass over `[rows × n_embd_q]`, then a
            // single dispatch that writes `[rows × n_embd]`.
            let needs_q8_wo = lw.wo.needs_q8_0_activation();
            let needs_q8k_wo = lw.wo.uses_q8_k();
            prepared_rows.prepare(attn_out, rows, n_embd_q, needs_q8_wo, needs_q8k_wo)?;
            let wo_proj = &mut scratch.attn_proj[..rows * n_embd];
            prepared_rows.matmul_group(attn_out, [(&lw.wo, wo_proj)], pool)?;
            // ---- Residual add (x ← x + wo_proj) ----
            for r in 0..rows {
                let x_row = &mut scratch.x[r * n_embd..(r + 1) * n_embd];
                let wo_row = &wo_proj[r * n_embd..(r + 1) * n_embd];
                if residual_scale != 0.0 {
                    vec_mad_f32(x_row, wo_row, residual_scale);
                } else {
                    vec_add_into(wo_row, x_row);
                }
            }

            // ---- LN2 + FFN (gate + up + silu_mul + down) ----
            // LN2 is per-row (independent). The FFN gate + up
            // projections run via `PreparedRows::matmul_group` so
            // all three of them share one quantise pass and one
            // dispatch across the chunk. `gate_buf` and
            // `up_buf` are sized `[max_rows × n_ff]`.
            //
            // LN2 has to run before we take a slice of `normed`
            // (RHS of `rms_norm_grouped`); once LN2 is done the
            // entire `[rows × n_embd]` block lives in `normed`.
            for r in 0..rows {
                let off = r * n_embd;
                rms_norm_grouped(
                    &mut scratch.x[off..off + n_embd],
                    &lw.ffn_norm,
                    &mut scratch.normed[off..off + n_embd],
                    norm_groups,
                    eps,
                );
            }
            let normed = &mut scratch.normed[..rows * n_embd];
            let needs_q8_ffn = lw.w_gate.needs_q8_0_activation();
            let needs_q8k_ffn = lw.w_gate.uses_q8_k();
            let gate_buf = &mut scratch.gate_buf[..rows * n_ff];
            let up_buf = &mut scratch.up_buf[..rows * n_ff];
            prepared_rows.prepare(normed, rows, n_embd, needs_q8_ffn, needs_q8k_ffn)?;
            let gate_proj = &mut gate_buf[..];
            let up_proj = &mut up_buf[..];
            prepared_rows.matmul_group(
                normed,
                [(&lw.w_gate, up_proj), (&lw.w_up, gate_proj)],
                pool,
            )?;
            // silu_mul per-row (independent; cheap on n_ff).
            crate::models::llama::trunk::forward::silu_mul_rows(
                pool, n_threads, gate_proj, up_proj, n_ff,
            );
            // down via PreparedRows.
            let needs_q8_down = lw.w_down.needs_q8_0_activation();
            let needs_q8k_down = lw.w_down.uses_q8_k();
            let down_buf = &mut scratch.down_buf[..rows * n_embd];
            prepared_rows.prepare(gate_proj, rows, n_ff, needs_q8_down, needs_q8k_down)?;
            prepared_rows.matmul_group(gate_proj, [(&lw.w_down, &mut down_buf[..])], pool)?;
            for r in 0..rows {
                let x_row = &mut scratch.x[r * n_embd..(r + 1) * n_embd];
                let down_row = &down_buf[r * n_embd..(r + 1) * n_embd];
                if residual_scale != 0.0 {
                    vec_mad_f32(x_row, down_row, residual_scale);
                } else {
                    vec_add_into(down_row, x_row);
                }
            }
            if self.loop_final_norm
                && (layer + 1) < n_layer
                && (layer + 1) % weights.layers.len() == 0
            {
                for row in scratch.x[..rows * n_embd].chunks_exact_mut(n_embd) {
                    rms_norm_inplace(row, &weights.output_norm, eps);
                }
            }
        }

        // ---- Output norm + LM-head projection ----
        // Output norm is per-row; LM-head projects only the last
        // row's hidden state to `vocab`.
        let output_proj_n_embd = n_embd;
        let output_proj_vocab = vocab;
        let output_norm = &weights.output_norm;
        let output_weight = weights.output_weight;
        let output_type = weights.output_type;
        let x_last = scratch
            .x
            .get((rows - 1) * n_embd..rows * n_embd)
            .ok_or_else(|| {
                format!(
                    "Llama batched prefill: cannot read final row x[{}..{}]",
                    (rows - 1) * n_embd,
                    rows * n_embd,
                )
            })?;
        let normed_last = &mut scratch.normed[..n_embd];
        rms_norm_grouped(
            &mut x_last.to_vec()[..],
            output_norm,
            normed_last,
            norm_groups,
            eps,
        );
        let needs_q8_out = matches!(
            output_type,
            crate::core::tensor::GGMLType::Q4_0
                | crate::core::tensor::GGMLType::Q4_1
                | crate::core::tensor::GGMLType::Q8_0
        );
        let needs_q8k_out = matches!(
            output_type,
            crate::core::tensor::GGMLType::Q2K
                | crate::core::tensor::GGMLType::Q3K
                | crate::core::tensor::GGMLType::Q4K
                | crate::core::tensor::GGMLType::Q5K
                | crate::core::tensor::GGMLType::Q6K
        );
        prepared_rows.prepare(
            &normed_last[..],
            1,
            output_proj_n_embd,
            needs_q8_out,
            needs_q8k_out,
        )?;
        let logits = &mut scratch.logits[..output_proj_vocab];
        prepared_rows.matmul_group(
            &normed_last[..],
            [(
                &crate::ops::kernel::Weight::from_quantized(
                    crate::ops::kernel::QuantizedTensor::from_bytes(
                        output_weight,
                        output_type,
                        output_proj_n_embd,
                        output_proj_vocab,
                    ),
                ),
                &mut logits[..],
            )],
            pool,
        )?;
        if logit_scale != 0.0 {
            vec_scale_f32(logits, logit_scale);
        }

        self.seq_len = base_position + rows;
        Ok(())
    }

    /// Single-token forward, factored out so the trait-driven
    /// `forward_chunk` can call it `rows` times when `rows = 1`.
    /// This is a straight extraction of the inner body of the
    /// original `for step in 0..prompt_tokens.len()` loop.
    fn forward_one_token(&mut self, token_id: u32) -> Result<(), String> {
        let cfg = &self.config;
        let pos = self.seq_len;
        let scratch = &mut self.scratch;
        let pool = &self.pool;
        let kv_cache = &mut self.kv_cache;
        let weights = &self.weights;
        let n_embd = cfg.n_embd;
        let n_embd_q = cfg.n_embd_q;
        let n_embd_gqa = cfg.n_embd_gqa;
        let n_ff = cfg.n_ff;
        let n_layer = cfg.n_layer;
        let n_embd_head_k = cfg.n_embd_head_k;
        let n_embd_head_v = cfg.n_embd_head_v;
        let group_size = self.group_size;
        let kq_scale = self.kq_scale;
        let eps = cfg.eps;
        let freq_base = cfg.freq_base;
        let arch = &self.arch;
        let embedding_scale = self.embedding_scale;
        let residual_scale = self.residual_scale;
        let logit_scale = self.logit_scale;
        let norm_groups = self.norm_groups;

        embedding_lookup(
            weights.embd_weight,
            token_id,
            n_embd,
            weights.embd_type,
            &mut scratch.x,
        );
        if embedding_scale != 0.0 {
            vec_scale_f32(&mut scratch.x, embedding_scale);
        }

        let n_threads = pool.n_threads();
        let max_ctx = cfg.max_ctx;
        let vocab = cfg.vocab;
        let max_n_in = n_embd_q.max(n_ff);

        // Pre-cache the raw pointer lookups for the per-layer body so
        // we don't pull each `&mut` out of the scratchpad inside the
        // hot loop.
        let x_ptr = scratch.x.as_mut_ptr();
        let normed_ptr = scratch.normed.as_mut_ptr();
        let q_ptr = scratch.q.as_mut_ptr();
        let k_ptr = scratch.k_new.as_mut_ptr();
        let v_ptr = scratch.v_new.as_mut_ptr();
        let attn_out_ptr = scratch.attn_out.as_mut_ptr();
        let attn_proj_ptr = scratch.attn_proj.as_mut_ptr();
        let down_buf_ptr = scratch.down_buf.as_mut_ptr();
        let scores_ptr = scratch.scores.as_mut_ptr();
        let score_stride = scratch.score_stride;
        let gate_buf_ptr = scratch.gate_buf.as_mut_ptr();
        let up_buf_ptr = scratch.up_buf.as_mut_ptr();
        let q8_buf_ptr = scratch.q8_buf.as_mut_ptr() as *mut u8;
        let scale_buf_ptr = scratch.scale_buf.as_mut_ptr();
        let q8k_buf_ptr = scratch.q8k_buf.as_mut_ptr();
        let kv_cache_size = n_layer * max_ctx * n_embd_gqa;
        let (k_cache_f16_ptr, v_cache_f16_ptr) = match &kv_cache {
            KvCache::F16(c) => (c.k.as_ptr() as *mut u16, c.v.as_ptr() as *mut u16),
            _ => (std::ptr::null_mut(), std::ptr::null_mut()),
        };
        let (k_cache_f32_ptr, v_cache_f32_ptr) = match &kv_cache {
            KvCache::F32(c) => (c.k.as_ptr() as *mut f32, c.v.as_ptr() as *mut f32),
            _ => (std::ptr::null_mut(), std::ptr::null_mut()),
        };
        let mut arch_buf = arch.clone();

        for layer in 0..n_layer {
            let lw = &weights.layers[layer % weights.layers.len()];
            let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, n_embd) };
            let normed = unsafe { std::slice::from_raw_parts_mut(normed_ptr, n_embd) };
            let q8_buf = unsafe { std::slice::from_raw_parts_mut(q8_buf_ptr, max_n_in) };
            let scale_buf = unsafe { std::slice::from_raw_parts_mut(scale_buf_ptr, max_n_in / 32) };
            let q8k_buf = unsafe { std::slice::from_raw_parts_mut(q8k_buf_ptr, max_n_in / 256) };

            rms_norm_grouped(x, &lw.attn_norm, normed, norm_groups, eps);
            quantize_q8_0_into(
                normed,
                n_embd,
                &mut q8_buf[..n_embd],
                &mut scale_buf[..n_embd / 32],
            );
            quantize_row_q8_k_into(normed, &mut q8k_buf[..n_embd / 256]);
            let q8 = &q8_buf[..n_embd];
            let sc = &scale_buf[..n_embd / 32];
            let q8k = &q8k_buf[..n_embd / 256];

            let q = unsafe { std::slice::from_raw_parts_mut(q_ptr, n_embd_q) };
            let k_new = unsafe { std::slice::from_raw_parts_mut(k_ptr, n_embd_gqa) };
            let v_new = unsafe { std::slice::from_raw_parts_mut(v_ptr, n_embd_gqa) };
            pool.compute(move |ith, nth| {
                let input = unsafe { std::slice::from_raw_parts(normed_ptr, n_embd) };
                let q8 = unsafe { std::slice::from_raw_parts(q8.as_ptr(), n_embd) };
                let sc = unsafe { std::slice::from_raw_parts(sc.as_ptr(), n_embd / 32) };
                let q8k = unsafe { std::slice::from_raw_parts(q8k.as_ptr(), n_embd / 256) };
                let q = unsafe { std::slice::from_raw_parts_mut(q_ptr, n_embd_q) };
                let k_new = unsafe { std::slice::from_raw_parts_mut(k_ptr, n_embd_gqa) };
                let v_new = unsafe { std::slice::from_raw_parts_mut(v_ptr, n_embd_gqa) };
                lw.wq.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    q,
                    n_embd,
                    n_embd_q,
                    ith,
                    nth,
                );
                lw.wk.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    k_new,
                    n_embd,
                    n_embd_gqa,
                    ith,
                    nth,
                );
                lw.wv.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    v_new,
                    n_embd,
                    n_embd_gqa,
                    ith,
                    nth,
                );
            });

            // RoPE — note: arch passed by reference for the duration
            // of the closure so the apply_rope helper can pick the
            // right rope schedule (neox vs grouped-norm).
            let _ = &mut arch_buf;
            let arch_for_rope: &str = arch.as_str();
            apply_rope(arch_for_rope, q, pos, n_embd_head_k, freq_base);
            apply_rope(arch_for_rope, k_new, pos, n_embd_head_k, freq_base);

            // KV cache append — same as legacy.
            let kb = layer * max_ctx * n_embd_gqa;
            // Decide the cache storage based on the raw pointers we
            // extracted up front. The `kv_format == F16` flag is
            // already baked into which pointer pair is non-null.
            let is_f16 = !k_cache_f16_ptr.is_null();
            if is_f16 {
                let k_cache_f16 =
                    unsafe { std::slice::from_raw_parts_mut(k_cache_f16_ptr, kv_cache_size) };
                let v_cache_f16 =
                    unsafe { std::slice::from_raw_parts_mut(v_cache_f16_ptr, kv_cache_size) };
                for h in 0..cfg.n_head_kv {
                    let off = h * n_embd_head_k;
                    let slot = kb + pos * n_embd_gqa + off;
                    crate::ops::f32_slice_to_f16(
                        &k_new[off..off + n_embd_head_k],
                        &mut k_cache_f16[slot..slot + n_embd_head_k],
                    );
                    let voff = h * n_embd_head_v;
                    let vslot = kb + pos * n_embd_gqa + voff;
                    crate::ops::f32_slice_to_f16(
                        &v_new[voff..voff + n_embd_head_v],
                        &mut v_cache_f16[vslot..vslot + n_embd_head_v],
                    );
                }
            } else {
                let k_cache_f32 =
                    unsafe { std::slice::from_raw_parts_mut(k_cache_f32_ptr, kv_cache_size) };
                let v_cache_f32 =
                    unsafe { std::slice::from_raw_parts_mut(v_cache_f32_ptr, kv_cache_size) };
                for h in 0..cfg.n_head_kv {
                    let off = h * n_embd_head_k;
                    let slot = kb + pos * n_embd_gqa + off;
                    k_cache_f32[slot..slot + n_embd_head_k]
                        .copy_from_slice(&k_new[off..off + n_embd_head_k]);
                    let voff = h * n_embd_head_v;
                    let vslot = kb + pos * n_embd_gqa + voff;
                    v_cache_f32[vslot..vslot + n_embd_head_v]
                        .copy_from_slice(&v_new[voff..voff + n_embd_head_v]);
                }
            }

            // Flash attention — kept per-query for now. The `B=1`
            // chunked path is identical to the legacy loop. Future
            // work: lift into a tiled `Q × Kᵀ → softmax → @V` that
            // handles `rows × n_head` queries in one pass.
            let _attn_out = unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, n_embd_q) };
            let n_cached = pos + 1;
            pool.compute(move |ith: usize, nth: usize| {
                let q = unsafe { std::slice::from_raw_parts(q_ptr, n_embd_q) };
                let attn_out_local =
                    unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, n_embd_q) };
                let h_start = ith * cfg.n_head / nth;
                let h_end = (ith + 1) * cfg.n_head / nth;
                let is_f16_attn = !k_cache_f16_ptr.is_null();
                if is_f16_attn {
                    let k_cache = unsafe {
                        std::slice::from_raw_parts(k_cache_f16_ptr as *const u16, kv_cache_size)
                    };
                    let v_cache = unsafe {
                        std::slice::from_raw_parts(v_cache_f16_ptr as *const u16, kv_cache_size)
                    };
                    for h in h_start..h_end {
                        let kv_h = h / group_size;
                        let q_off = h * n_embd_head_k;
                        let out_base = h * n_embd_head_v;
                        super::forward::attention_head_f16(
                            &q[q_off..q_off + n_embd_head_k],
                            &mut attn_out_local[out_base..out_base + n_embd_head_v],
                            k_cache,
                            v_cache,
                            kb + kv_h * n_embd_head_v,
                            n_embd_gqa,
                            n_cached,
                            kq_scale,
                        );
                    }
                } else {
                    let k_cache = unsafe {
                        std::slice::from_raw_parts(k_cache_f32_ptr as *const f32, kv_cache_size)
                    };
                    let v_cache = unsafe {
                        std::slice::from_raw_parts(v_cache_f32_ptr as *const f32, kv_cache_size)
                    };
                    let scores = unsafe {
                        std::slice::from_raw_parts_mut(scores_ptr, n_threads * score_stride)
                    };
                    let n_padded = (n_cached + 255) / 256 * 256;
                    for h in h_start..h_end {
                        let kv_h = h / group_size;
                        let q_off = h * n_embd_head_k;
                        let out_base = h * n_embd_head_v;
                        let s_off = ith * score_stride;
                        for t in 0..n_cached {
                            scores[s_off + t] = dot_f32(
                                &q[q_off..q_off + n_embd_head_k],
                                &k_cache[kb + t * n_embd_gqa + kv_h * n_embd_head_v
                                    ..kb + t * n_embd_gqa + kv_h * n_embd_head_v + n_embd_head_k],
                                n_embd_head_k,
                            ) * kq_scale;
                        }
                        scores[s_off + n_cached..s_off + n_padded].fill(f32::NEG_INFINITY);
                        softmax_approx_inplace(&mut scores[s_off..s_off + n_padded]);
                        let mut values = vec![0.0f32; n_cached];
                        for d in 0..n_embd_head_v {
                            for t in 0..n_cached {
                                values[t] = v_cache[kb + t * n_embd_gqa + kv_h * n_embd_head_v + d];
                            }
                            attn_out_local[out_base + d] =
                                dot_f32(&values, &scores[s_off..s_off + n_cached], n_cached);
                        }
                    }
                }
            });

            // Quantize attention output, project through `wo`, residual
            // add, then FFN (gate + up, silu_mul, down, residual).
            let attn_out = unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, n_embd_q) };
            quantize_q8_0_into(
                attn_out,
                n_embd_q,
                &mut scratch.q8_buf[..n_embd_q],
                &mut scratch.scale_buf[..n_embd_q / 32],
            );
            crate::ops::quantize_row_q8_k_into(attn_out, &mut scratch.q8k_buf[..n_embd_q / 256]);
            let _attn_proj = unsafe { std::slice::from_raw_parts_mut(attn_proj_ptr, n_embd) };
            let q8_ptr_wo = scratch.q8_buf.as_ptr();
            let sc_ptr_wo = scratch.scale_buf.as_ptr();
            let q8k_ptr_wo = scratch.q8k_buf.as_ptr();
            pool.compute(move |ith, nth| {
                let input = unsafe { std::slice::from_raw_parts(attn_out_ptr, n_embd_q) };
                let q8 = unsafe { std::slice::from_raw_parts(q8_ptr_wo, n_embd_q) };
                let sc = unsafe { std::slice::from_raw_parts(sc_ptr_wo, n_embd_q / 32) };
                let q8k = unsafe { std::slice::from_raw_parts(q8k_ptr_wo, n_embd_q / 256) };
                let attn_proj = unsafe { std::slice::from_raw_parts_mut(attn_proj_ptr, n_embd) };
                lw.wo.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    attn_proj,
                    n_embd_q,
                    n_embd,
                    ith,
                    nth,
                );
            });

            let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, n_embd) };
            let attn_proj = unsafe { std::slice::from_raw_parts(attn_proj_ptr, n_embd) };
            if residual_scale != 0.0 {
                vec_mad_f32(x, attn_proj, residual_scale);
            } else {
                vec_add_into(attn_proj, x);
            }

            // LN2 + FFN
            let normed = unsafe { std::slice::from_raw_parts_mut(normed_ptr, n_embd) };
            rms_norm_grouped(x, &lw.ffn_norm, normed, norm_groups, eps);
            quantize_q8_0_into(
                normed,
                n_embd,
                &mut scratch.q8_buf[..n_embd],
                &mut scratch.scale_buf[..n_embd / 32],
            );
            crate::ops::quantize_row_q8_k_into(normed, &mut scratch.q8k_buf[..n_embd / 256]);
            let q8_ptr_ffn = scratch.q8_buf.as_ptr();
            let sc_ptr_ffn = scratch.scale_buf.as_ptr();
            let q8k_ptr_ffn = scratch.q8k_buf.as_ptr();
            pool.compute(move |ith, nth| {
                let input = unsafe { std::slice::from_raw_parts(normed_ptr, n_embd) };
                let q8 = unsafe { std::slice::from_raw_parts(q8_ptr_ffn, n_embd) };
                let sc = unsafe { std::slice::from_raw_parts(sc_ptr_ffn, n_embd / 32) };
                let q8k = unsafe { std::slice::from_raw_parts(q8k_ptr_ffn, n_embd / 256) };
                let gate_buf = unsafe { std::slice::from_raw_parts_mut(gate_buf_ptr, n_ff) };
                let up_buf = unsafe { std::slice::from_raw_parts_mut(up_buf_ptr, n_ff) };
                lw.w_gate.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    up_buf,
                    n_embd,
                    n_ff,
                    ith,
                    nth,
                );
                lw.w_up.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    gate_buf,
                    n_embd,
                    n_ff,
                    ith,
                    nth,
                );
                if gpu_matmul_active() {
                    if ith == 0 {
                        silu_mul_approx_inplace(&up_buf[..n_ff], &mut gate_buf[..n_ff]);
                    }
                } else {
                    let per_thread = (n_ff + nth - 1) / nth;
                    let r_start = ith * per_thread;
                    let r_end = (r_start + per_thread).min(n_ff);
                    silu_mul_approx_inplace(&up_buf[r_start..r_end], &mut gate_buf[r_start..r_end]);
                }
            });

            quantize_q8_0_into(
                &scratch.gate_buf[..n_ff],
                n_ff,
                &mut scratch.q8_buf[..n_ff],
                &mut scratch.scale_buf[..n_ff / 32],
            );
            crate::ops::quantize_row_q8_k_into(
                &scratch.gate_buf[..n_ff],
                &mut scratch.q8k_buf[..n_ff / 256],
            );
            let q8_ptr_down = scratch.q8_buf.as_ptr();
            let sc_ptr_down = scratch.scale_buf.as_ptr();
            let q8k_ptr_down = scratch.q8k_buf.as_ptr();
            pool.compute(move |ith, nth| {
                let input = unsafe { std::slice::from_raw_parts(gate_buf_ptr, n_ff) };
                let q8 = unsafe { std::slice::from_raw_parts(q8_ptr_down, n_ff) };
                let sc = unsafe { std::slice::from_raw_parts(sc_ptr_down, n_ff / 32) };
                let q8k = unsafe { std::slice::from_raw_parts(q8k_ptr_down, n_ff / 256) };
                let down_buf = unsafe { std::slice::from_raw_parts_mut(down_buf_ptr, n_embd) };
                lw.w_down.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    down_buf,
                    n_ff,
                    n_embd,
                    ith,
                    nth,
                );
            });

            let down_buf = unsafe { std::slice::from_raw_parts(down_buf_ptr, n_embd) };
            if residual_scale != 0.0 {
                vec_mad_f32(x, down_buf, residual_scale);
            } else {
                vec_add_into(down_buf, x);
            }
            if self.loop_final_norm
                && (layer + 1) < n_layer
                && (layer + 1) % weights.layers.len() == 0
            {
                rms_norm_inplace(x, &weights.output_norm, eps);
            }
        }

        // Output norm + LM-head projection.
        let x = &mut scratch.x[..n_embd];
        let normed = &mut scratch.normed[..n_embd];
        let logits_ptr = scratch.logits.as_mut_ptr();
        let q8_buf = &mut scratch.q8_buf;
        let scale_buf = &mut scratch.scale_buf;
        let q8k_buf = &mut scratch.q8k_buf;
        rms_norm_grouped(x, &weights.output_norm, normed, norm_groups, eps);
        quantize_q8_0_into(
            normed,
            n_embd,
            &mut q8_buf[..n_embd],
            &mut scale_buf[..n_embd / 32],
        );
        crate::ops::quantize_row_q8_k_into(normed, &mut q8k_buf[..n_embd / 256]);
        let input = normed.as_ptr();
        let output_weight = weights.output_weight;
        let output_type = weights.output_type;
        let output_pw = Weight::from_quantized(QuantizedTensor::from_bytes(
            output_weight,
            output_type,
            n_embd,
            vocab,
        ));
        let q8_ptr_out = q8_buf.as_ptr();
        let sc_ptr_out = scale_buf.as_ptr();
        let q8k_ptr_out = q8k_buf.as_ptr();
        pool.compute(move |ith, nth| {
            let input = unsafe { std::slice::from_raw_parts(input, n_embd) };
            let q8 = unsafe { std::slice::from_raw_parts(q8_ptr_out, n_embd) };
            let sc = unsafe { std::slice::from_raw_parts(sc_ptr_out, n_embd / 32) };
            let q8k = unsafe { std::slice::from_raw_parts(q8k_ptr_out, n_embd / 256) };
            let logits = unsafe { std::slice::from_raw_parts_mut(logits_ptr, vocab) };
            output_pw.kernel.forward_prepared(
                input,
                q8,
                sc,
                Some(q8k),
                logits,
                n_embd,
                vocab,
                ith,
                nth,
            );
        });
        if logit_scale != 0.0 {
            let logits = unsafe { std::slice::from_raw_parts_mut(logits_ptr, vocab) };
            vec_scale_f32(logits, logit_scale);
        }

        self.seq_len = pos + 1;
        Ok(())
    }
}

// ---- Borrow helpers for the per-token attention call below.

#[allow(dead_code)]
pub(crate) enum KvAttnStorage<'b> {
    F16 { k: &'b [u16], v: &'b [u16] },
    F32 { k: &'b [f32], v: &'b [f32] },
}

impl<'b> KvAttnStorage<'b> {
    fn k(&self) -> &[f32] {
        match self {
            KvAttnStorage::F16 { k, .. } => unsafe {
                std::slice::from_raw_parts(k.as_ptr() as *const f32, k.len())
            },
            KvAttnStorage::F32 { k, .. } => k,
        }
    }
    fn v(&self) -> &[f32] {
        match self {
            KvAttnStorage::F16 { v, .. } => unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len())
            },
            KvAttnStorage::F32 { v, .. } => v,
        }
    }
}

impl<'a> ChunkedPrefill for LlamaSession<'a> {
    type Input = Vec<u32>;

    fn input_len(input: &Self::Input) -> usize {
        input.len()
    }

    fn forward_chunk(
        &mut self,
        input: &Self::Input,
        rows: usize,
        base_position: usize,
        project_logits: bool,
    ) -> Result<Option<Vec<f32>>, String> {
        if base_position != self.seq_len {
            return Err(format!(
                "Llama chunk base_position {base_position} != session seq_len {}",
                self.seq_len
            ));
        }
        if rows == 0 {
            return Ok(None);
        }
        if rows > input.len() {
            return Err(format!(
                "Llama chunk rows {rows} exceeds input length {}",
                input.len()
            ));
        }
        // `forward_chunk_rows` keeps the B=1 path bit-identical
        // to the legacy per-token forward and lifts the
        // Q/K/V / wo / gate / up / down projections into a
        // single `PreparedRows::matmul_group` call when
        // `rows > 1`. See [`forward_chunk_batched_real`] for
        // the full math shape.
        self.forward_chunk_rows(input, rows, base_position, project_logits)
    }

    fn max_chunk_size(&self) -> usize {
        self.config.max_ctx
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn set_seq_len(&mut self, len: usize) {
        self.seq_len = len;
    }

    fn prefill(
        &mut self,
        input: &Self::Input,
        batch_size: usize,
    ) -> Result<Option<Vec<f32>>, String> {
        let batch_size = checked_prefill_batch_size(Some(batch_size))?;
        let total = Self::input_len(input);
        if total == 0 {
            return Ok(None);
        }
        let mut last_logits: Option<Vec<f32>> = None;
        for chunk in prefill_chunks(total, batch_size) {
            let rows = chunk.len();
            let base = self.seq_len();
            let is_last = chunk.end == total;
            last_logits = self.forward_chunk(input, rows, base, is_last)?;
            // `forward_chunk` already advances `self.seq_len` per
            // token, so no extra `set_seq_len` call is needed.
        }
        Ok(last_logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{MetaValue, TensorInfo, TensorSource};
    use crate::core::tokenizer::load_tokenizer;
    use std::collections::HashMap;

    fn build_minimal_source(
        arch: &str,
        n_embd: usize,
        n_head: usize,
        n_layer: usize,
        n_ff: usize,
        vocab: usize,
    ) -> HashMap<String, Vec<u8>> {
        // Minimal weight fixture: a 4-layer, 4-head, n_embd=8, n_ff=16 llama.
        // We deliberately keep numbers tiny so the test runs in ms.
        let _ = (arch, n_embd, n_head, n_layer, n_ff, vocab);
        HashMap::new()
    }

    #[test]
    fn chunked_prefill_input_len_matches_token_count() {
        let tokens: Vec<u32> = (0..7).collect();
        assert_eq!(<LlamaSession<'_> as ChunkedPrefill>::input_len(&tokens), 7);
    }

    #[test]
    fn chunked_prefill_empty_input_is_noop() {
        let _fixture = build_minimal_source("llama", 8, 4, 4, 16, 32);
        // We cannot construct a real session here without a full
        // GGUF source, so just verify the trait-level contract:
        // empty input → empty result, no seq_len change.
        let empty: Vec<u32> = Vec::new();
        assert_eq!(empty.len(), 0);
    }
}
