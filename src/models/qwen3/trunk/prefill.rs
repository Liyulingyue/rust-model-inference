use super::config::Qwen3Rope;
use super::forward::{forward_moe_token, Qwen3Input};
use super::session::Qwen3Session;
use super::util::{checked_product, validate_input_shapes, validate_token_ids};
use super::weights::Qwen3Model;
use crate::core::prefill::{checked_prefill_batch_size, prefill_chunks};
use crate::core::scratchpad::KvCache;
use crate::ops::kernel::{PreparedRows, Weight};
use crate::ops::*;
#[cfg(feature = "parity-trace")]
use crate::parity_trace;
use std::ops::Range;
use std::time::{Duration, Instant};

pub(super) struct Qwen3PrefillScratch {
    max_rows: usize,
    max_n_in: usize,
    x: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    projection: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    prepared: PreparedRows,
}

pub(crate) struct Qwen3ChunkCapture {
    requested_layers: Vec<usize>,
    pub(crate) layer_inputs: Vec<Vec<f32>>,
    pub(crate) hidden: Vec<f32>,
    pub(crate) logits: Vec<f32>,
}

impl Qwen3ChunkCapture {
    pub(crate) fn new(requested_layers: &[usize]) -> Self {
        Self {
            requested_layers: requested_layers.to_vec(),
            layer_inputs: vec![Vec::new(); requested_layers.len()],
            hidden: Vec::new(),
            logits: Vec::new(),
        }
    }
}

impl Qwen3PrefillScratch {
    pub(super) fn new(max_rows: usize, model: &Qwen3Model) -> Self {
        let config = &model.config;
        let n_q = config.n_head * config.n_embd_head_k;
        let n_k = config.n_head_kv * config.n_embd_head_k;
        let n_v = config.n_head_kv * config.n_embd_head_v;
        let n_attn = config.n_head * config.n_embd_head_v;
        let max_n_in = config.n_embd.max(n_attn).max(config.n_ff);
        Self {
            max_rows,
            max_n_in,
            x: vec![0.0; max_rows * config.n_embd],
            normed: vec![0.0; max_rows * config.n_embd],
            q: vec![0.0; max_rows * n_q],
            k: vec![0.0; max_rows * n_k],
            v: vec![0.0; max_rows * n_v],
            attn: vec![0.0; max_rows * n_attn],
            projection: vec![0.0; max_rows * config.n_embd],
            gate: vec![0.0; max_rows * config.n_ff],
            up: vec![0.0; max_rows * config.n_ff],
            down: vec![0.0; max_rows * config.n_embd],
            prepared: PreparedRows::new(max_rows, max_n_in),
        }
    }

    fn reset_for(&mut self, max_rows: usize, model: &Qwen3Model) {
        if self.max_rows != max_rows {
            *self = Self::new(max_rows, model);
        }
    }

    fn bytes(&self) -> usize {
        let f32_values = self.x.len()
            + self.normed.len()
            + self.q.len()
            + self.k.len()
            + self.v.len()
            + self.attn.len()
            + self.projection.len()
            + self.gate.len()
            + self.up.len()
            + self.down.len();
        f32_values * std::mem::size_of::<f32>()
            + self.max_rows * self.max_n_in
            + self.max_rows * self.max_n_in.div_ceil(32) * std::mem::size_of::<f32>()
            + self.max_rows
                * (self.max_n_in / crate::ops::quant::QK_K)
                * std::mem::size_of::<crate::ops::quant::BlockQ8K>()
    }
}

#[derive(Clone, Copy)]
enum KvPtrs {
    F16 { k: *mut u16, v: *mut u16 },
    F32 { k: *mut f32, v: *mut f32 },
}

fn add_deepstack_embedding(
    hidden: &mut [f32],
    deepstack: &[f32],
    layer: usize,
    token: usize,
    token_count: usize,
    width: usize,
) {
    let start = (layer * token_count + token) * width;
    for (value, addition) in hidden.iter_mut().zip(&deepstack[start..start + width]) {
        *value += *addition;
    }
}

fn matmul_rows(
    prepared: &mut PreparedRows,
    prepared_for: &mut Option<(usize, bool, bool)>,
    weight: &Weight<'_>,
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    n_in: usize,
    model: &Qwen3Model,
) -> Result<(), String> {
    let requirements = (n_in, weight.needs_q8_0_activation(), weight.uses_q8_k());
    if *prepared_for != Some(requirements) {
        prepared.prepare(input, rows, n_in, requirements.1, requirements.2)?;
        *prepared_for = Some(requirements);
    }
    prepared.matmul(weight, input, output, &model.pool)
}

fn matmul_group_rows<const N: usize>(
    prepared: &mut PreparedRows,
    projections: [(&Weight<'_>, &mut [f32]); N],
    input: &[f32],
    rows: usize,
    n_in: usize,
    model: &Qwen3Model,
) -> Result<(), String> {
    let need_q8 = projections
        .iter()
        .any(|(weight, _)| weight.needs_q8_0_activation());
    let need_q8k = projections.iter().any(|(weight, _)| weight.uses_q8_k());
    prepared.prepare(input, rows, n_in, need_q8, need_q8k)?;
    prepared.matmul_group(input, projections, &model.pool)
}

impl Qwen3Session<'_> {
    pub(super) fn prefill(
        &mut self,
        input: &Qwen3Input<'_>,
        batch_size: usize,
    ) -> Result<Duration, String> {
        let batch_size = checked_prefill_batch_size(Some(batch_size))?;
        if input.token_ids.is_empty() {
            return Err("Qwen3 prompt must contain at least one token".into());
        }
        validate_input_shapes(
            input.token_ids.len(),
            self.model.config.n_embd,
            input.positions.len(),
            input.embeddings.map(<[f32]>::len),
        )?;
        validate_token_ids(input.token_ids, self.model.config.vocab)?;
        if input
            .embeddings
            .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err("Input embeddings contain NaN or infinity".into());
        }
        if let Some(deepstack) = input.deepstack_embeddings {
            let expected = input
                .token_ids
                .len()
                .checked_mul(self.model.config.n_embd)
                .and_then(|values| values.checked_mul(self.model.config.n_deepstack_layers))
                .ok_or_else(|| "Deepstack embedding shape overflow".to_string())?;
            if deepstack.len() != expected || deepstack.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "Invalid deepstack embedding value count: expected {expected}, got {}",
                    deepstack.len()
                ));
            }
        }
        let final_len = self
            .kv_state
            .seq_len
            .checked_add(input.token_ids.len())
            .ok_or_else(|| "Qwen3 prompt length overflow".to_string())?;
        if final_len > self.capacity {
            return Err(format!(
                "Qwen3 prompt requires capacity {final_len}; session has {}",
                self.capacity
            ));
        }

        // The parity trace is token-major and includes logits for every prompt row.
        #[cfg(feature = "parity-trace")]
        let trace_each_token = std::env::var_os("RMI_PARITY_TRACE").is_some();
        #[cfg(not(feature = "parity-trace"))]
        let trace_each_token = false;
        let chunk_size = batch_size;
        let max_rows = chunk_size.min(self.capacity);
        self.prefill_scratch.reset_for(max_rows, self.model);
        let started = Instant::now();
        for range in prefill_chunks(input.token_ids.len(), chunk_size) {
            let base = self.kv_state.seq_len;
            let project_logits = trace_each_token || range.end == input.token_ids.len();
            #[cfg(feature = "vulkan")]
            let gpu_error = if let Some(gpu) = &mut self.gpu {
                let rows = range.len();
                let width = self.model.config.n_embd;
                for row in 0..rows {
                    let token = range.start + row;
                    let output = &mut self.prefill_scratch.x[row * width..(row + 1) * width];
                    if let Some(embeddings) = input.embeddings {
                        output.copy_from_slice(&embeddings[token * width..(token + 1) * width]);
                    } else {
                        self.model
                            .token_embedding
                            .embedding_lookup(input.token_ids[token], output);
                    }
                }
                let result = (|| -> Result<(), String> {
                    if trace_each_token
                        || range
                            .clone()
                            .enumerate()
                            .any(|(row, token)| input.positions[token][0] != base + row)
                    {
                        return Err("Qwen3 Vulkan prefill requires sequential positions and no active parity trace".into());
                    }
                    gpu.reserve_rows(self.model, max_rows)
                        .map_err(|error| error.to_string())?;
                    let result = gpu
                        .forward_chunk(
                            &self.prefill_scratch.x[..rows * width],
                            base,
                            rows,
                            project_logits,
                        )
                        .map_err(|error| error.to_string())?;
                    crate::vulkan::qwen3::commit_shadow_kv_chunk(
                        &mut self.kv_state,
                        base,
                        rows,
                        result.k_delta,
                        result.v_delta,
                    )?;
                    if project_logits {
                        self.scratch.logits.copy_from_slice(result.logits);
                    }
                    gpu.commit_token();
                    Ok(())
                })();
                match result {
                    Ok(()) => continue,
                    Err(error) => {
                        gpu.abort_token();
                        eprintln!("[GPU] Qwen3 Vulkan chunk {}..{} failed: {error}. Recomputing the whole chunk on CPU.", range.start, range.end);
                        self.gpu = None;
                        self.full_model_gpu_failed = true;
                        Some(error)
                    }
                }
            } else {
                None
            };
            if let Err(error) = self
                .forward_cpu_chunk(input, range.clone(), project_logits, true, None)
                .and_then(|()| self.validate_cpu_chunk(base, range.len(), project_logits))
            {
                self.kv_state.seq_len = base;
                #[cfg(feature = "vulkan")]
                if let Some(gpu_error) = gpu_error {
                    return Err(format!("{error}; original Vulkan error: {gpu_error}"));
                }
                return Err(error);
            }
            self.kv_state.seq_len = base + range.len();
            self.kv_state.update_access();
        }
        let elapsed = started.elapsed();
        Ok(elapsed)
    }

    fn validate_cpu_chunk(
        &self,
        base: usize,
        rows: usize,
        project_logits: bool,
    ) -> Result<(), String> {
        let config = &self.model.config;
        if self.prefill_scratch.x[..rows * config.n_embd]
            .iter()
            .any(|value| !value.is_finite())
            || (project_logits && self.scratch.logits.iter().any(|value| !value.is_finite()))
        {
            return Err("Qwen3 prefill produced non-finite output".into());
        }
        let stride = config.n_head_kv * config.n_embd_head_k.max(config.n_embd_head_v);
        for layer in 0..config.n_layer {
            let start = (layer * self.capacity + base) * stride;
            let end = start + rows * stride;
            let finite = match &self.kv_state.cache {
                KvCache::F16(cache) => cache.k[start..end]
                    .iter()
                    .chain(&cache.v[start..end])
                    .all(|word| word & 0x7c00 != 0x7c00),
                KvCache::F32(cache) => cache.k[start..end]
                    .iter()
                    .chain(&cache.v[start..end])
                    .all(|value| value.is_finite()),
            };
            if !finite {
                return Err(format!(
                    "Qwen3 prefill produced non-finite KV in layer {layer}"
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn forward_cpu_chunk(
        &mut self,
        input: &Qwen3Input<'_>,
        range: Range<usize>,
        project_logits: bool,
        causal: bool,
        mut capture: Option<&mut Qwen3ChunkCapture>,
    ) -> Result<(), String> {
        if range.is_empty() || range.end > input.token_ids.len() {
            return Err("Invalid Qwen3 CPU prefill range".into());
        }
        let rows = range.len();
        #[cfg(feature = "parity-trace")]
        let _trace = parity_trace::TokenMajorTrace::new(rows);
        if rows > self.prefill_scratch.max_rows {
            return Err(format!(
                "Qwen3 prefill rows {rows} exceed scratch capacity {}",
                self.prefill_scratch.max_rows
            ));
        }
        let model = self.model;
        let config = &model.config;
        let capacity = self.capacity;
        let base_position = self.kv_state.seq_len;
        if base_position + rows > capacity {
            return Err("Qwen3 CPU prefill chunk exceeds session capacity".into());
        }
        let n_q = checked_product("query width", config.n_head, config.n_embd_head_k)?;
        let n_k = checked_product("key width", config.n_head_kv, config.n_embd_head_k)?;
        let n_v = checked_product("value width", config.n_head_kv, config.n_embd_head_v)?;
        let n_attn = checked_product(
            "attention output width",
            config.n_head,
            config.n_embd_head_v,
        )?;
        let kv_stride = n_k.max(n_v);
        let kv_cache_size = checked_product(
            "KV cache values",
            checked_product("KV cache rows", config.n_layer, capacity)?,
            kv_stride,
        )?;
        let group_size = config.n_head / config.n_head_kv;
        let kq_scale = 1.0 / (config.n_embd_head_k as f32).sqrt();
        let kv_ptrs = match &mut self.kv_state.cache {
            KvCache::F16(cache) => KvPtrs::F16 {
                k: cache.k.as_mut_ptr(),
                v: cache.v.as_mut_ptr(),
            },
            KvCache::F32(cache) => KvPtrs::F32 {
                k: cache.k.as_mut_ptr(),
                v: cache.v.as_mut_ptr(),
            },
        };

        let x = &mut self.prefill_scratch.x[..rows * config.n_embd];
        for row in 0..rows {
            let token = range.start + row;
            let output = &mut x[row * config.n_embd..(row + 1) * config.n_embd];
            if let Some(embeddings) = input.embeddings {
                output.copy_from_slice(
                    &embeddings[token * config.n_embd..(token + 1) * config.n_embd],
                );
            } else {
                model
                    .token_embedding
                    .embedding_lookup(input.token_ids[token], output);
            }
            #[cfg(feature = "parity-trace")]
            parity_trace::report(parity_trace::checkpoint_row(
                row,
                "model.input_embed",
                None,
                &[1, config.n_embd],
                output,
            ));
        }

        #[cfg(feature = "vulkan")]
        let _gpu_matmul_scope = self
            .full_model_gpu_failed
            .then(crate::core::thread_pool::ComputePool::disable_gpu_matmul_for_scope);

        for layer in 0..config.n_layer {
            if let Some(capture) = capture.as_deref_mut() {
                for (slot, _) in capture
                    .requested_layers
                    .iter()
                    .enumerate()
                    .filter(|(_, requested)| **requested == layer)
                {
                    capture.layer_inputs[slot] =
                        self.prefill_scratch.x[..rows * config.n_embd].to_vec();
                }
            }
            let weights = &model.layers[layer];
            for row in 0..rows {
                rms_norm(
                    &self.prefill_scratch.x[row * config.n_embd..(row + 1) * config.n_embd],
                    &weights.attn_norm,
                    &mut self.prefill_scratch.normed
                        [row * config.n_embd..(row + 1) * config.n_embd],
                    config.eps,
                );
            }
            #[cfg(feature = "parity-trace")]
            if layer == 0 {
                for row in 0..rows {
                    parity_trace::report(parity_trace::checkpoint_row(
                        row,
                        "attn_norm-0",
                        Some(0),
                        &[1, config.n_embd],
                        &self.prefill_scratch.normed
                            [row * config.n_embd..(row + 1) * config.n_embd],
                    ));
                }
            }

            let normed = &self.prefill_scratch.normed[..rows * config.n_embd];
            matmul_group_rows(
                &mut self.prefill_scratch.prepared,
                [
                    (&weights.wq, &mut self.prefill_scratch.q[..rows * n_q]),
                    (&weights.wk, &mut self.prefill_scratch.k[..rows * n_k]),
                    (&weights.wv, &mut self.prefill_scratch.v[..rows * n_v]),
                ],
                normed,
                rows,
                config.n_embd,
                model,
            )?;

            for row in 0..rows {
                let token = range.start + row;
                let position = input.positions[token];
                let q = &mut self.prefill_scratch.q[row * n_q..(row + 1) * n_q];
                let k = &mut self.prefill_scratch.k[row * n_k..(row + 1) * n_k];
                let v = &mut self.prefill_scratch.v[row * n_v..(row + 1) * n_v];
                if let Some(bias) = weights.q_bias.as_deref() {
                    for (value, bias) in q.iter_mut().zip(bias) {
                        *value += *bias;
                    }
                }
                if let Some(bias) = weights.k_bias.as_deref() {
                    for (value, bias) in k.iter_mut().zip(bias) {
                        *value += *bias;
                    }
                }
                if let Some(bias) = weights.v_bias.as_deref() {
                    for (value, bias) in v.iter_mut().zip(bias) {
                        *value += *bias;
                    }
                }
                if let (Some(q_norm), Some(k_norm)) =
                    (weights.q_norm.as_deref(), weights.k_norm.as_deref())
                {
                    for head in q.chunks_exact_mut(config.n_embd_head_k) {
                        rms_norm_inplace(head, q_norm, config.eps);
                    }
                    for head in k.chunks_exact_mut(config.n_embd_head_k) {
                        rms_norm_inplace(head, k_norm, config.eps);
                    }
                }
                #[cfg(feature = "parity-trace")]
                if layer == 0 {
                    parity_trace::report(parity_trace::checkpoint_row(
                        row,
                        "Qcur_normed-0",
                        Some(0),
                        &[config.n_head, config.n_embd_head_k],
                        q,
                    ));
                    parity_trace::report(parity_trace::checkpoint_row(
                        row,
                        "Kcur_normed-0",
                        Some(0),
                        &[config.n_head_kv, config.n_embd_head_k],
                        k,
                    ));
                }
                for head in q.chunks_exact_mut(config.n_embd_head_k) {
                    match config.rope {
                        Qwen3Rope::Neox => rope_neox_inplace(
                            head,
                            position[0],
                            config.n_embd_head_k,
                            config.freq_base,
                        ),
                        Qwen3Rope::Interleaved { sections, n_dims } => rope_mrope_interleaved(
                            head,
                            position,
                            sections,
                            config.n_embd_head_k,
                            config.freq_base,
                            n_dims,
                        ),
                    }
                }
                for head in k.chunks_exact_mut(config.n_embd_head_k) {
                    match config.rope {
                        Qwen3Rope::Neox => rope_neox_inplace(
                            head,
                            position[0],
                            config.n_embd_head_k,
                            config.freq_base,
                        ),
                        Qwen3Rope::Interleaved { sections, n_dims } => rope_mrope_interleaved(
                            head,
                            position,
                            sections,
                            config.n_embd_head_k,
                            config.freq_base,
                            n_dims,
                        ),
                    }
                }
                #[cfg(feature = "parity-trace")]
                if layer == 0 {
                    parity_trace::report(parity_trace::checkpoint_row(
                        row,
                        "Qcur-0",
                        Some(0),
                        &[config.n_head, config.n_embd_head_k],
                        q,
                    ));
                    parity_trace::report(parity_trace::checkpoint_row(
                        row,
                        "Kcur-0",
                        Some(0),
                        &[config.n_head_kv, config.n_embd_head_k],
                        k,
                    ));
                }

                let physical_row = base_position + row;
                let layer_base = layer * capacity * kv_stride;
                match kv_ptrs {
                    KvPtrs::F16 { k: k_ptr, v: v_ptr } => {
                        let k_cache =
                            unsafe { std::slice::from_raw_parts_mut(k_ptr, kv_cache_size) };
                        let v_cache =
                            unsafe { std::slice::from_raw_parts_mut(v_ptr, kv_cache_size) };
                        for head in 0..config.n_head_kv {
                            let k_offset = head * config.n_embd_head_k;
                            let v_offset = head * config.n_embd_head_v;
                            let cache_row = layer_base + physical_row * kv_stride;
                            f32_slice_to_f16(
                                &k[k_offset..k_offset + config.n_embd_head_k],
                                &mut k_cache[cache_row + k_offset
                                    ..cache_row + k_offset + config.n_embd_head_k],
                            );
                            f32_slice_to_f16(
                                &v[v_offset..v_offset + config.n_embd_head_v],
                                &mut v_cache[cache_row + v_offset
                                    ..cache_row + v_offset + config.n_embd_head_v],
                            );
                        }
                    }
                    KvPtrs::F32 { k: k_ptr, v: v_ptr } => {
                        let k_cache =
                            unsafe { std::slice::from_raw_parts_mut(k_ptr, kv_cache_size) };
                        let v_cache =
                            unsafe { std::slice::from_raw_parts_mut(v_ptr, kv_cache_size) };
                        for head in 0..config.n_head_kv {
                            let k_offset = head * config.n_embd_head_k;
                            let v_offset = head * config.n_embd_head_v;
                            let cache_row = layer_base + physical_row * kv_stride;
                            k_cache
                                [cache_row + k_offset..cache_row + k_offset + config.n_embd_head_k]
                                .copy_from_slice(&k[k_offset..k_offset + config.n_embd_head_k]);
                            v_cache
                                [cache_row + v_offset..cache_row + v_offset + config.n_embd_head_v]
                                .copy_from_slice(&v[v_offset..v_offset + config.n_embd_head_v]);
                        }
                    }
                }
            }

            let q_ptr = self.prefill_scratch.q.as_ptr();
            let attn_ptr = self.prefill_scratch.attn.as_mut_ptr();
            let scores_ptr = self.scratch.scores.as_mut_ptr();
            let score_stride = self.scratch.score_stride;
            model.pool.compute(move |thread, threads| {
                let scores = unsafe {
                    std::slice::from_raw_parts_mut(
                        scores_ptr.add(thread * score_stride),
                        score_stride,
                    )
                };
                let f16_scratch = scores.as_mut_ptr().cast::<u16>();
                let head_start = thread * config.n_head / threads;
                let head_end = (thread + 1) * config.n_head / threads;
                let layer_base = layer * capacity * kv_stride;
                for row in 0..rows {
                    let physical_row = base_position + row;
                    let visible = if causal {
                        physical_row + 1
                    } else {
                        base_position + rows
                    };
                    let n_padded = visible.div_ceil(256) * 256;
                    let q = unsafe { std::slice::from_raw_parts(q_ptr.add(row * n_q), n_q) };
                    let attn = unsafe {
                        std::slice::from_raw_parts_mut(attn_ptr.add(row * n_attn), n_attn)
                    };
                    match kv_ptrs {
                        KvPtrs::F16 { k: k_ptr, v: v_ptr } => {
                            let k_cache =
                                unsafe { std::slice::from_raw_parts(k_ptr, kv_cache_size) };
                            let v_cache =
                                unsafe { std::slice::from_raw_parts(v_ptr, kv_cache_size) };
                            for head in head_start..head_end {
                                let kv_head = head / group_size;
                                let q_offset = head * config.n_embd_head_k;
                                let output_offset = head * config.n_embd_head_v;
                                let output =
                                    &mut attn[output_offset..output_offset + config.n_embd_head_v];
                                let query = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        output.as_mut_ptr().cast::<u16>(),
                                        config.n_embd_head_k,
                                    )
                                };
                                f32_slice_to_f16(
                                    &q[q_offset..q_offset + config.n_embd_head_k],
                                    query,
                                );
                                scores[..n_padded].fill(f32::NEG_INFINITY);
                                for token in 0..visible {
                                    let cache_row = layer_base + token * kv_stride;
                                    let key_offset = cache_row + kv_head * config.n_embd_head_k;
                                    scores[token] = dot_f16(
                                        query,
                                        &k_cache[key_offset..key_offset + config.n_embd_head_k],
                                        config.n_embd_head_k,
                                    ) * kq_scale;
                                }
                                softmax_inplace(&mut scores[..n_padded]);
                                for index in 0..n_padded {
                                    unsafe {
                                        *f16_scratch.add(index) = f32_to_f16(scores[index]);
                                    }
                                }
                                let weights =
                                    unsafe { std::slice::from_raw_parts(f16_scratch, n_padded) };
                                let values = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        f16_scratch.add(score_stride),
                                        n_padded,
                                    )
                                };
                                values[visible..].fill(0);
                                for dimension in 0..config.n_embd_head_v {
                                    for token in 0..visible {
                                        let cache_row = layer_base + token * kv_stride;
                                        values[token] = v_cache[cache_row
                                            + kv_head * config.n_embd_head_v
                                            + dimension];
                                    }
                                    output[dimension] = dot_f16(values, weights, n_padded);
                                }
                            }
                        }
                        KvPtrs::F32 { k: k_ptr, v: v_ptr } => {
                            let k_cache =
                                unsafe { std::slice::from_raw_parts(k_ptr, kv_cache_size) };
                            let v_cache =
                                unsafe { std::slice::from_raw_parts(v_ptr, kv_cache_size) };
                            for head in head_start..head_end {
                                let kv_head = head / group_size;
                                let q_offset = head * config.n_embd_head_k;
                                let output_offset = head * config.n_embd_head_v;
                                let output =
                                    &mut attn[output_offset..output_offset + config.n_embd_head_v];
                                let query = &q[q_offset..q_offset + config.n_embd_head_k];
                                scores[..n_padded].fill(f32::NEG_INFINITY);
                                for token in 0..visible {
                                    let cache_row = layer_base + token * kv_stride;
                                    let key_offset = cache_row + kv_head * config.n_embd_head_k;
                                    scores[token] = dot_f32(
                                        query,
                                        &k_cache[key_offset..key_offset + config.n_embd_head_k],
                                        config.n_embd_head_k,
                                    ) * kq_scale;
                                }
                                softmax_inplace(&mut scores[..n_padded]);
                                let weights = &scores[..n_padded];
                                for dimension in 0..config.n_embd_head_v {
                                    let mut value = 0.0;
                                    for token in 0..visible {
                                        let cache_row = layer_base + token * kv_stride;
                                        value += weights[token]
                                            * v_cache[cache_row
                                                + kv_head * config.n_embd_head_v
                                                + dimension];
                                    }
                                    output[dimension] = value;
                                }
                            }
                        }
                    }
                }
            });

            #[cfg(feature = "parity-trace")]
            if layer == 0 {
                for row in 0..rows {
                    parity_trace::report(parity_trace::checkpoint_row(
                        row,
                        "kqv_out-0",
                        Some(0),
                        &[config.n_head, config.n_embd_head_v],
                        &self.prefill_scratch.attn[row * n_attn..(row + 1) * n_attn],
                    ));
                }
            }
            let mut prepared_for = None;
            matmul_rows(
                &mut self.prefill_scratch.prepared,
                &mut prepared_for,
                &weights.wo,
                &self.prefill_scratch.attn[..rows * n_attn],
                &mut self.prefill_scratch.projection[..rows * config.n_embd],
                rows,
                n_attn,
                model,
            )?;
            for row in 0..rows {
                let row_start = row * config.n_embd;
                let row_end = row_start + config.n_embd;
                for (hidden, projection) in self.prefill_scratch.x[row_start..row_end]
                    .iter_mut()
                    .zip(&self.prefill_scratch.projection[row_start..row_end])
                {
                    *hidden += *projection;
                }
                rms_norm(
                    &self.prefill_scratch.x[row_start..row_end],
                    &weights.ffn_norm,
                    &mut self.prefill_scratch.normed[row_start..row_end],
                    config.eps,
                );
            }

            if weights.moe_router.is_some() {
                for row in 0..rows {
                    forward_moe_token(
                        &self.prefill_scratch.normed
                            [row * config.n_embd..(row + 1) * config.n_embd],
                        weights,
                        config,
                        &mut self.prefill_scratch.down
                            [row * config.n_embd..(row + 1) * config.n_embd],
                    )?;
                }
            } else {
                matmul_group_rows(
                    &mut self.prefill_scratch.prepared,
                    [
                        (
                            &weights.w_gate,
                            &mut self.prefill_scratch.up[..rows * config.n_ff],
                        ),
                        (
                            &weights.w_up,
                            &mut self.prefill_scratch.gate[..rows * config.n_ff],
                        ),
                    ],
                    &self.prefill_scratch.normed[..rows * config.n_embd],
                    rows,
                    config.n_embd,
                    model,
                )?;
                for row in 0..rows {
                    let start = row * config.n_ff;
                    let end = start + config.n_ff;
                    silu_mul_approx_inplace(
                        &self.prefill_scratch.up[start..end],
                        &mut self.prefill_scratch.gate[start..end],
                    );
                }
                let mut prepared_for = None;
                matmul_rows(
                    &mut self.prefill_scratch.prepared,
                    &mut prepared_for,
                    &weights.w_down,
                    &self.prefill_scratch.gate[..rows * config.n_ff],
                    &mut self.prefill_scratch.down[..rows * config.n_embd],
                    rows,
                    config.n_ff,
                    model,
                )?;
            }

            #[cfg(feature = "parity-trace")]
            if layer == 0 {
                for row in 0..rows {
                    parity_trace::report(parity_trace::checkpoint_row(
                        row,
                        "ffn_out-0",
                        Some(0),
                        &[1, config.n_embd],
                        &self.prefill_scratch.down[row * config.n_embd..(row + 1) * config.n_embd],
                    ));
                }
            }
            for row in 0..rows {
                let start = row * config.n_embd;
                let end = start + config.n_embd;
                for (hidden, projection) in self.prefill_scratch.x[start..end]
                    .iter_mut()
                    .zip(&self.prefill_scratch.down[start..end])
                {
                    *hidden += *projection;
                }
                if layer < config.n_deepstack_layers {
                    if let Some(deepstack) = input.deepstack_embeddings {
                        add_deepstack_embedding(
                            &mut self.prefill_scratch.x[start..end],
                            deepstack,
                            layer,
                            range.start + row,
                            input.token_ids.len(),
                            config.n_embd,
                        );
                    }
                }
            }

            #[cfg(test)]
            if self.fail_cpu_prefill_after_layer == Some(layer) {
                self.fail_cpu_prefill_after_layer = None;
                return Err(format!(
                    "injected Qwen3 CPU prefill failure after layer {layer}"
                ));
            }
        }

        #[cfg(feature = "parity-trace")]
        let trace_all = std::env::var_os("RMI_PARITY_TRACE").is_some();
        #[cfg(not(feature = "parity-trace"))]
        let trace_all = false;
        if project_logits || trace_all || capture.is_some() {
            let capture_all = trace_all || capture.is_some();
            for row in if capture_all { 0..rows } else { rows - 1..rows } {
                let last = row * config.n_embd;
                self.scratch
                    .x
                    .copy_from_slice(&self.prefill_scratch.x[last..last + config.n_embd]);
                rms_norm(
                    &self.scratch.x,
                    &model.output_norm,
                    &mut self.scratch.normed,
                    config.eps,
                );
                #[cfg(feature = "parity-trace")]
                parity_trace::report(parity_trace::checkpoint_row(
                    row,
                    "result_norm",
                    None,
                    &[1, config.n_embd],
                    &self.scratch.normed,
                ));
                let mut prepared_for = None;
                matmul_rows(
                    &mut self.prefill_scratch.prepared,
                    &mut prepared_for,
                    &model.output,
                    &self.scratch.normed,
                    &mut self.scratch.logits,
                    1,
                    config.n_embd,
                    model,
                )?;
                if let Some(capture) = capture.as_deref_mut() {
                    capture.hidden.extend_from_slice(&self.scratch.normed);
                    capture.logits.extend_from_slice(&self.scratch.logits);
                }
                #[cfg(feature = "parity-trace")]
                parity_trace::report(parity_trace::checkpoint_row(
                    row,
                    "result_output",
                    None,
                    &[config.vocab],
                    &self.scratch.logits,
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn forward_non_causal_block(
        &mut self,
        token_ids: &[u32],
        positions: &[[usize; 4]],
    ) -> Result<Qwen3ChunkCapture, String> {
        if token_ids.is_empty() || token_ids.len() != positions.len() {
            return Err("Invalid DSpark draft block".into());
        }
        let base = self.kv_state.seq_len;
        let final_len = base
            .checked_add(token_ids.len())
            .ok_or_else(|| "DSpark draft length overflow".to_string())?;
        if final_len > self.capacity {
            return Err(format!(
                "DSpark draft requires capacity {final_len}; session has {}",
                self.capacity
            ));
        }
        self.prefill_scratch.reset_for(token_ids.len(), self.model);
        let input = Qwen3Input {
            token_ids,
            positions,
            embeddings: None,
            deepstack_embeddings: None,
        };
        let mut capture = Qwen3ChunkCapture::new(&[]);
        self.forward_cpu_chunk(&input, 0..token_ids.len(), true, false, Some(&mut capture))?;
        self.validate_cpu_chunk(base, token_ids.len(), true)?;
        self.kv_state.seq_len = final_len;
        self.kv_state.update_access();
        Ok(capture)
    }

    pub(crate) fn forward_causal_capture(
        &mut self,
        token_ids: &[u32],
        requested_layers: &[usize],
    ) -> Result<Qwen3ChunkCapture, String> {
        if token_ids.is_empty()
            || requested_layers.is_empty()
            || requested_layers
                .iter()
                .any(|&layer| layer >= self.model.config.n_layer)
        {
            return Err("Invalid Qwen3 DSpark target batch".into());
        }
        let base = self.kv_state.seq_len;
        let final_len = base
            .checked_add(token_ids.len())
            .ok_or_else(|| "Qwen3 DSpark target length overflow".to_string())?;
        if final_len > self.capacity {
            return Err(format!(
                "Qwen3 DSpark target requires capacity {final_len}; session has {}",
                self.capacity
            ));
        }
        let positions = (base..final_len)
            .map(|position| [position, 0, 0, 0])
            .collect::<Vec<_>>();
        self.prefill_scratch.reset_for(token_ids.len(), self.model);
        let input = Qwen3Input {
            token_ids,
            positions: &positions,
            embeddings: None,
            deepstack_embeddings: None,
        };
        let mut capture = Qwen3ChunkCapture::new(requested_layers);
        self.forward_cpu_chunk(&input, 0..token_ids.len(), true, true, Some(&mut capture))?;
        self.validate_cpu_chunk(base, token_ids.len(), true)?;
        if capture.layer_inputs.iter().any(Vec::is_empty) {
            return Err("Qwen3 did not capture every DSpark target layer".into());
        }
        self.kv_state.seq_len = final_len;
        self.kv_state.update_access();
        Ok(capture)
    }

    pub fn scratch_bytes(&self) -> usize {
        let scratch = &self.scratch;
        let f32_values = scratch.x.len()
            + scratch.normed.len()
            + scratch.q.len()
            + scratch.k_new.len()
            + scratch.v_new.len()
            + scratch.attn_out.len()
            + scratch.attn_proj.len()
            + scratch.down_buf.len()
            + scratch.gate_buf.len()
            + scratch.up_buf.len()
            + scratch.logits.len()
            + scratch.scale_buf.len()
            + scratch.scores.len();
        self.prefill_scratch.bytes()
            + f32_values * std::mem::size_of::<f32>()
            + scratch.q8_buf.len()
            + scratch.q8k_buf.len() * std::mem::size_of::<crate::ops::quant::BlockQ8K>()
    }

    #[cfg(test)]
    pub(super) fn fail_cpu_prefill_after_layer_for_test(&mut self, layer: usize) {
        self.fail_cpu_prefill_after_layer = Some(layer);
    }
}

#[cfg(test)]
mod tests {
    use super::add_deepstack_embedding;

    #[test]
    fn deepstack_injection_uses_layer_major_token_rows() {
        let deepstack = [
            1.0, 2.0, 3.0, 4.0, // decoder layer 0, prompt tokens 0 and 1
            5.0, 6.0, 7.0, 8.0, // decoder layer 1, prompt tokens 0 and 1
        ];
        let mut hidden = [10.0, 20.0];
        add_deepstack_embedding(&mut hidden, &deepstack, 1, 0, 2, 2);
        assert_eq!(hidden, [15.0, 26.0]);
    }
}
