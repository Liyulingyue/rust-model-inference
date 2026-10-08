use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::core::scratchpad::{KvArch, KvCache, KvFormat, KvState};
use crate::core::tensor::{load_f32_tensor, GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::ops::kernel::{Kernel, QuantizedTensor, Weight};
use crate::ops::quant::BlockQ8K;

use super::config::YuE2Config;
use super::nar::Mt19937;
use super::protocol::{
    SamplingConfig, YuE2Protocol, ABC_END, CODEC_OFFSET, CODEC_SIZE, EOD, MUSIC_END,
};

pub(crate) struct YuE2Weight {
    fast: Weight<'static>,
    bf16: Option<&'static [u8]>,
    #[cfg(feature = "vulkan")]
    gpu_bf16: crate::ops::kernel::vulkan::GpuLinear,
    n_in: usize,
    n_out: usize,
}

impl YuE2Weight {
    /// The on-disk type, so a device uploader can pick a matching layout
    /// instead of guessing from the shape.
    pub(crate) fn ggml_type(&self) -> GGMLType {
        self.fast.ggml_type
    }
}

/// GGML types the YuE2 converter can emit for a 2-D projection.
///
/// `bf16` keeps the raw bytes so `matmul` can use the widening NEON dot.
/// Everything else goes through the block-quantized `Weight` path, which needs
/// `n_in` to be a whole number of blocks. The converter only ever quantizes
/// matrices whose `n_in` is one of the architecture widths (2048 / 6144), so
/// this is checked here rather than trusted.
///
/// `Q4K` / `Q6K` are the k-quants the `q4_k_m` / `q6_k` and `ar_q4_k_m` modes
/// emit. They share the Q8_K activation staging that `Weight` already owns, and
/// the Q8_K scratch here is sized in 256-element super-blocks to match.
const MATRIX_TYPES: [GGMLType; 7] = [
    GGMLType::BF16,
    GGMLType::F16,
    GGMLType::F32,
    GGMLType::Q8_0,
    GGMLType::Q4_0,
    GGMLType::Q4K,
    GGMLType::Q6K,
];

/// Reusable per-thread staging buffers for [`YuE2Weight::matmul_rows`].
///
/// One instance is owned by the NAR session, so the Q8_0 activation buffer is
/// allocated once per generation instead of once per projection per step.
pub(super) struct RowScratch {
    /// One Q8_0 activation buffer per pool thread. Sharing a single buffer
    /// across threads is a data race: a thread quantizes its row and then
    /// consumes the payload, and another thread can overwrite it in between.
    slots: Vec<(Vec<u8>, Vec<f32>)>,
    threads: usize,
}

impl RowScratch {
    pub(super) fn new() -> Self {
        Self {
            slots: Vec::new(),
            threads: 0,
        }
    }

    /// Ensure at least `threads` slots, each large enough for the widest input
    /// seen so far. The NAR reuses one scratch across matrices of differing
    /// `n_in` (2048 / 1024 / 6144), so the buffers must grow, not just fill in.
    fn reserve_threads(&mut self, threads: usize, q8_len: usize, scale_len: usize) {
        self.threads = threads;
        if self.slots.len() < threads {
            self.slots
                .resize_with(threads, || (vec![0u8; q8_len], vec![0.0f32; scale_len]));
        }
        for (q8, scales) in &mut self.slots {
            if q8.len() < q8_len {
                q8.resize(q8_len, 0);
            }
            if scales.len() < scale_len {
                scales.resize(scale_len, 0.0);
            }
        }
    }
}

impl YuE2Weight {
    fn load(
        source: &dyn TensorSource,
        name: &str,
        n_in: usize,
        n_out: usize,
    ) -> Result<Self, String> {
        let info = source
            .tensor_info(name)
            .ok_or_else(|| format!("Missing tensor info: {name}"))?;
        if !MATRIX_TYPES.contains(&info.ggml_type) {
            return Err(format!(
                "YuE2 weight {name} has unsupported type {:?}; expected one of the \
                 converter-emitted matrix types",
                info.ggml_type
            ));
        }
        if matches!(
            info.ggml_type,
            GGMLType::Q8_0 | GGMLType::Q4_0 | GGMLType::Q4K | GGMLType::Q6K
        ) && n_in % info.ggml_type.type_traits().0 != 0
        {
            return Err(format!(
                "YuE2 weight {name} has n_in={n_in}, which is not a whole number of \
                 {:?} blocks; the converter must not quantize this matrix",
                info.ggml_type
            ));
        }
        let bytes = source
            .tensor_slice(name)
            .ok_or_else(|| format!("Missing tensor data: {name}"))?;
        let bytes: &'static [u8] = unsafe { std::mem::transmute(bytes) };
        Ok(Self {
            fast: Weight::from_quantized(QuantizedTensor::from_bytes(
                bytes,
                info.ggml_type,
                n_in,
                n_out,
            )),
            bf16: (info.ggml_type == GGMLType::BF16).then_some(bytes),
            #[cfg(feature = "vulkan")]
            gpu_bf16: Default::default(),
            n_in,
            n_out,
        })
    }

    /// Test-only accessor for the underlying kernel, so parity tests can call
    /// the same kernel entry the batched path uses.
    #[cfg(test)]
    pub(super) fn kernel(&self) -> &crate::ops::kernel::Weight<'static> {
        &self.fast
    }

    /// Test-only constructor for a quantized weight, so parity tests can
    /// exercise the block-quantized `matmul_rows` path rather than the BF16 one.
    #[cfg(test)]
    pub(super) fn from_quantized_bytes(
        bytes: &'static [u8],
        ggml_type: crate::core::tensor::GGMLType,
        n_in: usize,
        n_out: usize,
    ) -> Self {
        Self {
            fast: Weight::from_quantized(QuantizedTensor::from_bytes(
                bytes, ggml_type, n_in, n_out,
            )),
            bf16: None,
            #[cfg(feature = "vulkan")]
            gpu_bf16: Default::default(),
            n_in,
            n_out,
        }
    }

    #[cfg(test)]
    pub(super) fn from_f32(values: Vec<f32>, n_in: usize, n_out: usize) -> Self {
        Self {
            fast: Weight::from_quantized(QuantizedTensor::F32 {
                data: values,
                n_in,
                n_out,
            }),
            bf16: None,
            #[cfg(feature = "vulkan")]
            gpu_bf16: Default::default(),
            n_in,
            n_out,
        }
    }

    pub(super) fn embedding_lookup(&self, token_id: u32, output: &mut [f32]) {
        if let Some(bytes) = self.bf16 {
            let start = token_id as usize * self.n_in * 2;
            for (index, value) in output.iter_mut().enumerate() {
                let offset = start + index * 2;
                *value =
                    crate::ops::bf16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]));
            }
            return;
        }
        self.fast.embedding_lookup(token_id, output);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn matmul(
        &self,
        input: &[f32],
        output: &mut [f32],
        pool: &ComputePool,
        q8: &mut [u8],
        scales: &mut [f32],
        q8k: &mut [BlockQ8K],
    ) {
        debug_assert_eq!(input.len(), self.n_in);
        debug_assert_eq!(output.len(), self.n_out);
        if self.bf16.is_some() {
            self.matmul_bf16(input, None, output, pool);
            return;
        }
        self.fast
            .quantize_and_matmul_with_scratch(input, q8k, q8, scales, output, pool);
    }

    /// Batched `n_rows x n_in @ n_in x n_out` used by the NAR passes.
    ///
    /// The single-row `matmul` quantizes the activation on the calling thread and
    /// then hands the *output* rows to `pool.compute`.  The NAR evaluates all
    /// ~258 latent positions at once, so calling that per row meant 258
    /// single-threaded quantizations and 258 pool barriers for every projection,
    /// which dominated the NAR solve.  Here the whole batch is quantized once
    /// (also inside the pool) and each thread owns a contiguous slice of the
    /// *input* rows, so the weight tile it streams stays hot in cache.
    pub(super) fn matmul_rows(
        &self,
        input: &[f32],
        output: &mut [f32],
        pool: &ComputePool,
        scratch: &mut RowScratch,
        bias: Option<&[f32]>,
    ) -> Result<(), String> {
        let n_rows = input.len() / self.n_in;
        if input.len() % self.n_in != 0 || output.len() != n_rows * self.n_out || n_rows == 0 {
            return Err("YuE2 batched matmul got inconsistent row counts".into());
        }
        if let Some(bias) = bias {
            if bias.len() != self.n_out {
                return Err("YuE2 batched matmul bias width mismatch".into());
            }
        }
        if self.bf16.is_some() {
            self.matmul_bf16(input, bias, output, pool);
            return Ok(());
        }
        #[cfg(feature = "vulkan")]
        if self.fast.ggml_type != GGMLType::F16 && self.fast.try_vulkan_rows(input, output, n_rows)
        {
            if let Some(bias) = bias {
                for row in output.chunks_exact_mut(self.n_out) {
                    for (value, offset) in row.iter_mut().zip(bias) {
                        *value += offset;
                    }
                }
            }
            return Ok(());
        }
        // NAR F16 consumes Q8-prequantized input; the generic F16 GPU path
        // rounds raw input to F16, so this contract stays on CPU.
        #[cfg(feature = "vulkan")]
        let _cpu_scope = ComputePool::disable_gpu_matmul_for_scope();
        if self.fast.ggml_type == crate::core::tensor::GGMLType::F32 {
            // F32 weights consume raw activations; handing them Q8_0 bytes would
            // silently produce zeros because the F32 kernel's `forward_prepared`
            // only takes the f32 path when `input_f32` is populated.
            for row in 0..n_rows {
                self.fast.kernel.forward_prepared(
                    &input[row * self.n_in..(row + 1) * self.n_in],
                    &[],
                    &[],
                    None,
                    &mut output[row * self.n_out..(row + 1) * self.n_out],
                    self.n_in,
                    self.n_out,
                    0,
                    1,
                );
                if let Some(bias) = bias {
                    for (value, &offset) in output[row * self.n_out..(row + 1) * self.n_out]
                        .iter_mut()
                        .zip(bias)
                    {
                        *value += offset;
                    }
                }
            }
            return Ok(());
        }

        // Per-thread Q8_0 staging. This MUST be one buffer per pool thread: a
        // thread quantizes its activation and then immediately consumes it, so a
        // shared buffer lets a second thread overwrite the payload in between and
        // silently produces a different result (observed as ~12% error).
        let q8_stride = self.n_in;
        let scale_stride = self.n_in.div_ceil(32);
        scratch.reserve_threads(pool.n_threads().max(1), q8_stride, scale_stride);
        let weight = &self.fast;
        let n_in = self.n_in;
        let n_out = self.n_out;
        let input_ptr = input.as_ptr();
        let output_ptr = output.as_mut_ptr();
        let bias_ptr = bias.map(|bias| bias.as_ptr());
        let slots_ptr = scratch.slots.as_mut_ptr();
        let slot_count = scratch.slots.len();
        pool.compute(|ith, nth| {
            let (start, end) = crate::ops::kernel::bf16::BF16Kernel::row_range(n_rows, ith, nth);
            if start == end {
                return;
            }
            // SAFETY: slot `ith` belongs to exactly this pool thread, and
            // `start..end` is disjoint from every other thread's rows.
            let (q8, scales) = unsafe {
                let (q8, scales) = &mut *slots_ptr.add(ith.min(slot_count - 1));
                (q8.as_mut_slice(), scales.as_mut_slice())
            };
            for row in start..end {
                let activation =
                    unsafe { std::slice::from_raw_parts(input_ptr.add(row * n_in), n_in) };
                crate::ops::quantize_q8_0_into(activation, n_in, q8, scales);
                let out =
                    unsafe { std::slice::from_raw_parts_mut(output_ptr.add(row * n_out), n_out) };
                weight
                    .kernel
                    .forward_prequantized(q8, scales, out, n_in, n_out, 0, 1);
                if let Some(bias_ptr) = bias_ptr {
                    for (value, &offset) in out
                        .iter_mut()
                        .zip(unsafe { std::slice::from_raw_parts(bias_ptr, n_out) })
                    {
                        *value += offset;
                    }
                }
            }
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn matmul_bias(
        &self,
        input: &[f32],
        bias: &[f32],
        output: &mut [f32],
        pool: &ComputePool,
        q8: &mut [u8],
        scales: &mut [f32],
        q8k: &mut [BlockQ8K],
    ) {
        debug_assert_eq!(bias.len(), self.n_out);
        if self.bf16.is_some() {
            self.matmul_bf16(input, Some(bias), output, pool);
        } else {
            self.matmul(input, output, pool, q8, scales, q8k);
            for (value, &bias) in output.iter_mut().zip(bias) {
                *value += bias;
            }
        }
    }

    fn matmul_bf16(
        &self,
        input: &[f32],
        bias: Option<&[f32]>,
        output: &mut [f32],
        pool: &ComputePool,
    ) {
        let bytes = self.bf16.unwrap();
        let n_rows = input.len() / self.n_in;
        #[cfg(feature = "vulkan")]
        if self.gpu_bf16.try_matmul(
            bytes,
            crate::vulkan::ops::GpuWeightFormat::BF16Dot,
            input,
            output,
            self.n_in,
            self.n_out,
            n_rows,
        ) {
            for row in output.chunks_exact_mut(self.n_out) {
                for (column, value) in row.iter_mut().enumerate() {
                    let sum = bias.map_or(*value, |bias| *value + bias[column]);
                    *value = half::bf16::from_f32(sum).to_f32();
                }
            }
            return;
        }
        let batched_rows = n_rows / 4 * 4;
        let output_ptr = output.as_mut_ptr();
        pool.compute(|thread, threads| {
            let (start, end) =
                crate::ops::kernel::bf16::BF16Kernel::row_range(self.n_out, thread, threads);
            if start == end {
                return;
            }
            for row in (0..batched_rows).step_by(4) {
                let inputs = &input[row * self.n_in..(row + 4) * self.n_in];
                for column in start..end {
                    let weight = &bytes[column * self.n_in * 2..(column + 1) * self.n_in * 2];
                    let sums = crate::ops::dot_bf16_f32_4(inputs, weight, self.n_in);
                    for (offset, sum) in sums.into_iter().enumerate() {
                        let sum = bias.map_or(sum, |bias| sum + bias[column]);
                        unsafe {
                            output_ptr
                                .add((row + offset) * self.n_out + column)
                                .write(half::bf16::from_f32(sum).to_f32());
                        }
                    }
                }
            }
            for row in batched_rows..n_rows {
                let output = unsafe {
                    std::slice::from_raw_parts_mut(
                        output_ptr.add(row * self.n_out + start),
                        end - start,
                    )
                };
                torch_bf16_matmul_rows(
                    bytes,
                    &input[row * self.n_in..(row + 1) * self.n_in],
                    bias,
                    output,
                    self.n_in,
                    start,
                );
            }
        });
    }
}

pub(super) fn torch_bf16_matmul_rows(
    bytes: &[u8],
    input: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
) {
    for (offset_row, value) in output.iter_mut().enumerate() {
        let row = row_start + offset_row;
        let byte_start = row * n_in * 2;
        let sum = crate::ops::dot_bf16_f32(input, &bytes[byte_start..byte_start + n_in * 2], n_in);
        let sum = bias.map_or(sum, |bias| sum + bias[row]);
        *value = half::bf16::from_f32(sum).to_f32();
    }
}

pub(crate) struct YuE2AttentionWeights {
    pub(crate) norm: Vec<f32>,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) q: YuE2Weight,
    pub(crate) k: YuE2Weight,
    pub(crate) v: YuE2Weight,
    pub(crate) output: YuE2Weight,
}

pub(crate) struct YuE2MlpWeights {
    pub(crate) norm: Vec<f32>,
    pub(crate) gate: YuE2Weight,
    pub(crate) up: YuE2Weight,
    pub(crate) down: YuE2Weight,
}

pub(crate) struct YuE2LayerWeights {
    pub(crate) ar_attention: YuE2AttentionWeights,
    pub(crate) ar_mlp: YuE2MlpWeights,
    pub(crate) nar_attention: YuE2AttentionWeights,
    pub(crate) nar_mlp: YuE2MlpWeights,
}

pub(super) struct YuE2AuxWeights {
    pub(super) llm2vae: YuE2Weight,
    pub(super) llm2vae_bias: Vec<f32>,
    pub(super) vae2llm: YuE2Weight,
    pub(super) vae2llm_bias: Vec<f32>,
    pub(super) time_in: YuE2Weight,
    pub(super) time_in_bias: Vec<f32>,
    pub(super) time_out: YuE2Weight,
    pub(super) time_out_bias: Vec<f32>,
    pub(super) latent_position: YuE2Weight,
}

pub struct YuE2Model {
    source: Option<Arc<dyn TensorSource>>,
    tokenizer: Arc<BPETokenizer>,
    pub(super) pool: Arc<ComputePool>,
    pub(super) config: YuE2Config,
    pub(super) layers: Vec<YuE2LayerWeights>,
    pub(super) token_embedding: YuE2Weight,
    pub(super) final_norm: Vec<f32>,
    lm_head: YuE2Weight,
    pub(super) aux: YuE2AuxWeights,
}

impl fmt::Debug for YuE2Model {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("YuE2Model")
            .field("config", &self.config)
            .field("layers", &self.layers.len())
            .finish_non_exhaustive()
    }
}

impl YuE2Model {
    /// The tensor source, for consumers that upload weights to another device
    /// (the Vulkan session does).
    pub fn tensor_source(&self) -> Option<&Arc<dyn TensorSource>> {
        self.source.as_ref()
    }

    /// The AR half of the per-layer weights. The NAR half is deliberately not
    /// exposed for device upload: it is a diffusion solve that amplifies weight
    /// noise every step, so its weights have to stay BF16.
    pub(crate) fn ar_layers(&self) -> &[YuE2LayerWeights] {
        &self.layers
    }

    pub(crate) fn ar_final_norm(&self) -> &[f32] {
        &self.final_norm
    }

    /// `lm_head.weight`, the only AR projection outside the layer stack.
    pub(crate) fn ar_lm_head(&self) -> &YuE2Weight {
        &self.lm_head
    }

    pub fn from_source(
        source: Arc<dyn TensorSource>,
        tokenizer: Arc<BPETokenizer>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let config = YuE2Config::from_source(source.as_ref())?;
        YuE2Protocol::from_source(source.as_ref())?;
        if tokenizer.vocab_size() != config.vocab {
            return Err(format!(
                "YuE2 vocabulary size {} does not match tokenizer vocab {}",
                config.vocab,
                tokenizer.vocab_size()
            ));
        }
        preflight(source.as_ref(), &config)?;

        let q_width = config.q_heads * config.head_dim;
        let kv_width = config.kv_heads * config.head_dim;
        let mut layers = Vec::with_capacity(config.layers);
        for layer in 0..config.layers {
            let base = format!("model.layers.{layer}");
            layers.push(YuE2LayerWeights {
                ar_attention: load_attention(
                    source.as_ref(),
                    &base,
                    "input_layernorm",
                    "self_attn",
                    &config,
                )?,
                ar_mlp: load_mlp(
                    source.as_ref(),
                    &base,
                    "post_attention_layernorm",
                    "mlp",
                    &config,
                )?,
                nar_attention: load_attention(
                    source.as_ref(),
                    &base,
                    "nar_input_layernorm",
                    "nar_self_attn",
                    &config,
                )?,
                nar_mlp: load_mlp(
                    source.as_ref(),
                    &base,
                    "nar_pre_mlp_layernorm",
                    "nar_mlp",
                    &config,
                )?,
            });
        }

        let model = Self {
            token_embedding: YuE2Weight::load(
                source.as_ref(),
                "model.embed_tokens.weight",
                config.hidden,
                config.vocab,
            )?,
            final_norm: load_f32_tensor(
                source.as_ref(),
                "model.norm.weight",
                &[config.hidden as u64],
            )?,
            lm_head: YuE2Weight::load(
                source.as_ref(),
                "lm_head.weight",
                config.hidden,
                config.vocab,
            )?,
            aux: YuE2AuxWeights {
                llm2vae: YuE2Weight::load(
                    source.as_ref(),
                    "llm2vae.weight",
                    config.hidden,
                    config.latent_channels,
                )?,
                llm2vae_bias: load_f32_tensor(
                    source.as_ref(),
                    "llm2vae.bias",
                    &[config.latent_channels as u64],
                )?,
                vae2llm: YuE2Weight::load(
                    source.as_ref(),
                    "vae2llm.weight",
                    config.latent_channels,
                    config.hidden,
                )?,
                vae2llm_bias: load_f32_tensor(
                    source.as_ref(),
                    "vae2llm.bias",
                    &[config.hidden as u64],
                )?,
                time_in: YuE2Weight::load(
                    source.as_ref(),
                    "time_embedder.mlp.0.weight",
                    256,
                    config.hidden,
                )?,
                time_in_bias: load_f32_tensor(
                    source.as_ref(),
                    "time_embedder.mlp.0.bias",
                    &[config.hidden as u64],
                )?,
                time_out: YuE2Weight::load(
                    source.as_ref(),
                    "time_embedder.mlp.2.weight",
                    config.hidden,
                    config.hidden,
                )?,
                time_out_bias: load_f32_tensor(
                    source.as_ref(),
                    "time_embedder.mlp.2.bias",
                    &[config.hidden as u64],
                )?,
                latent_position: YuE2Weight::load(
                    source.as_ref(),
                    "latent_pos_embed.pe",
                    config.hidden,
                    config.context,
                )?,
            },
            source: Some(Arc::clone(&source)),
            tokenizer,
            pool,
            config,
            layers,
        };
        debug_assert_eq!(model.layers[0].ar_attention.q.n_out, q_width);
        debug_assert_eq!(model.layers[0].ar_attention.k.n_out, kv_width);
        Ok(model)
    }

    pub fn config(&self) -> &YuE2Config {
        &self.config
    }

    pub fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }

    pub fn generate_abc(
        &self,
        prefix: &[u32],
        sampling: SamplingConfig,
        seed: u64,
    ) -> Result<Vec<u32>, String> {
        self.generate_phase(prefix, sampling, seed, Phase::Abc)
    }

    pub fn generate_semantic(
        &self,
        prefix: &[u32],
        sampling: SamplingConfig,
        seed: u64,
    ) -> Result<Vec<u32>, String> {
        self.generate_phase(prefix, sampling, seed, Phase::Semantic)
    }

    fn generate_phase(
        &self,
        prefix: &[u32],
        sampling: SamplingConfig,
        seed: u64,
        phase: Phase,
    ) -> Result<Vec<u32>, String> {
        sampling.validate()?;
        let capacity = prefix
            .len()
            .checked_add(sampling.max_tokens)
            .ok_or("YuE2 generation length overflow")?;
        if prefix.is_empty() || capacity > self.config.context {
            return Err(format!(
                "YuE2 prefix plus generation requires {capacity} positions, maximum is {}",
                self.config.context
            ));
        }
        let mut session = YuE2ArSession::new(self, capacity)?;
        let mut logits = session.prefill(prefix)?.to_vec();
        let mut rng = Mt19937::new(seed);
        let mut output = Vec::with_capacity(sampling.max_tokens);
        let sampling_started = std::time::Instant::now();
        for step in 0..sampling.max_tokens {
            let token = sample_phase_token(&logits, &output, sampling, step, phase, &mut rng)?;
            if token == phase.end_token() {
                break;
            }
            output.push(token);
            if step + 1 < sampling.max_tokens {
                logits = session.prefill(&[token])?.to_vec();
            }
            // Single-token decode over a 3B AR pass is seconds per step on
            // CPU, so emit a periodic rate log instead of a silent stall.
            if output.len() % 16 == 0 || step + 1 == sampling.max_tokens {
                let elapsed = sampling_started.elapsed().as_secs_f64();
                eprintln!(
                    "[yue2:{}] {} tokens in {:.1}s ({:.2} tok/s)",
                    phase.short_name(),
                    output.len(),
                    elapsed,
                    output.len() as f64 / elapsed.max(f64::MIN_POSITIVE),
                );
            }
        }
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::token_ids(phase.trace_name(), &output));
        Ok(output)
    }

    #[cfg(test)]
    pub(super) fn tiny_for_test(tokenizer: Arc<BPETokenizer>, pool: Arc<ComputePool>) -> Self {
        let config = YuE2Config {
            hidden: 4,
            layers: 1,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            ffn: 8,
            vocab: 16,
            context: 8,
            rms_eps: 0.000001,
            rope_base: 1_000_000.0,
            latent_channels: 2,
            timestep_shift: 1.0,
        };
        fn weight(n_in: usize, n_out: usize, salt: usize) -> YuE2Weight {
            let values = (0..n_in * n_out)
                .map(|index| (((index + salt) % 11) as f32 - 5.0) * 0.025)
                .collect();
            YuE2Weight::from_f32(values, n_in, n_out)
        }
        fn attention(config: &YuE2Config, salt: usize) -> YuE2AttentionWeights {
            YuE2AttentionWeights {
                norm: vec![1.0; config.hidden],
                q_norm: vec![1.0; config.head_dim],
                k_norm: vec![1.0; config.head_dim],
                q: weight(config.hidden, config.q_heads * config.head_dim, salt),
                k: weight(config.hidden, config.kv_heads * config.head_dim, salt + 1),
                v: weight(config.hidden, config.kv_heads * config.head_dim, salt + 2),
                output: weight(config.q_heads * config.head_dim, config.hidden, salt + 3),
            }
        }
        fn mlp(config: &YuE2Config, salt: usize) -> YuE2MlpWeights {
            YuE2MlpWeights {
                norm: vec![1.0; config.hidden],
                gate: weight(config.hidden, config.ffn, salt),
                up: weight(config.hidden, config.ffn, salt + 1),
                down: weight(config.ffn, config.hidden, salt + 2),
            }
        }
        let layer = YuE2LayerWeights {
            ar_attention: attention(&config, 1),
            ar_mlp: mlp(&config, 5),
            nar_attention: attention(&config, 9),
            nar_mlp: mlp(&config, 13),
        };
        Self {
            source: None,
            tokenizer,
            pool,
            token_embedding: weight(config.hidden, config.vocab, 17),
            final_norm: vec![1.0; config.hidden],
            lm_head: weight(config.hidden, config.vocab, 19),
            aux: YuE2AuxWeights {
                llm2vae: weight(config.hidden, config.latent_channels, 21),
                llm2vae_bias: vec![0.0; config.latent_channels],
                vae2llm: weight(config.latent_channels, config.hidden, 22),
                vae2llm_bias: vec![0.0; config.hidden],
                time_in: weight(256, config.hidden, 23),
                time_in_bias: vec![0.0; config.hidden],
                time_out: weight(config.hidden, config.hidden, 24),
                time_out_bias: vec![0.0; config.hidden],
                latent_position: weight(config.hidden, config.context, 25),
            },
            config,
            layers: vec![layer],
        }
    }
}

pub struct YuE2ArSession<'model> {
    model: &'model YuE2Model,
    kv: KvState,
    x: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attention: Vec<f32>,
    projected: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    logits: Vec<f32>,
    scores: Vec<f32>,
    q8: Vec<u8>,
    scales: Vec<f32>,
    q8k: Vec<BlockQ8K>,
    /// Experimental whole-AR mirror. Normal sessions keep CPU state operations
    /// and offload projections: the mirror lacks YuE2's BF16 rounding contract.
    /// Outer None requests initialization; Some(None) permanently keeps fallback.
    #[cfg(feature = "vulkan")]
    gpu: Option<Option<crate::vulkan::yue2::YuE2VulkanSession<'model>>>,
}

/// Run one device chunk for `token_ids`, mirror the K/V deltas into the CPU
/// shadow cache, and copy the logits back.
///
/// Deliberately a free function rather than a method: the caller already holds
/// a `&mut` to the GPU session, and taking `&mut self` as well would alias.
#[cfg(feature = "vulkan")]
fn prefill_on_gpu(
    model: &YuE2Model,
    gpu: &mut crate::vulkan::yue2::YuE2VulkanSession<'_>,
    kv: &mut KvState,
    logits: &mut [f32],
    token_ids: &[u32],
    base: usize,
) -> Result<(), String> {
    let config = &model.config;
    let mut input = vec![0.0f32; token_ids.len() * config.hidden];
    for (row, &token) in token_ids.iter().enumerate() {
        model.token_embedding.embedding_lookup(
            token,
            &mut input[row * config.hidden..(row + 1) * config.hidden],
        );
    }
    let kv_stride = config.kv_heads * config.head_dim;
    let result = gpu
        .forward_chunk(&input, base, token_ids.len())
        .map_err(|error| error.to_string())?;
    crate::vulkan::yue2::commit_kv_shadow(
        &mut kv.cache,
        base,
        token_ids.len(),
        kv.capacity,
        kv_stride,
        config.layers,
        result.k_delta,
        result.v_delta,
    )?;
    if result.logits.len() != logits.len() {
        return Err(format!(
            "YuE2 GPU returned {} logits, expected {}",
            result.logits.len(),
            logits.len()
        ));
    }
    logits.copy_from_slice(result.logits);
    Ok(())
}

impl<'model> YuE2ArSession<'model> {
    pub fn new(model: &'model YuE2Model, capacity: usize) -> Result<Self, String> {
        if capacity == 0 || capacity > model.config.context {
            return Err(format!(
                "YuE2 session capacity {capacity} must be within 1..={}",
                model.config.context
            ));
        }
        let config = &model.config;
        let q_width = checked_product(config.q_heads, config.head_dim, "YuE2 query width")?;
        let kv_width = checked_product(config.kv_heads, config.head_dim, "YuE2 KV width")?;
        let _ = checked_product(
            checked_product(config.layers, capacity, "YuE2 KV rows")?,
            kv_width,
            "YuE2 KV values",
        )?;
        let max_input = config.ffn.max(config.hidden).max(q_width);
        let arch = Arc::new(KvArch::new(
            config.layers,
            config.kv_heads,
            config.head_dim,
            config.head_dim,
            config.context,
        ));
        Ok(Self {
            model,
            kv: KvState::new(arch, KvFormat::F32, capacity),
            x: vec![0.0; config.hidden],
            normed: vec![0.0; config.hidden],
            q: vec![0.0; q_width],
            k: vec![0.0; kv_width],
            v: vec![0.0; kv_width],
            attention: vec![0.0; q_width],
            projected: vec![0.0; config.hidden],
            gate: vec![0.0; config.ffn],
            up: vec![0.0; config.ffn],
            down: vec![0.0; config.hidden],
            logits: vec![0.0; config.vocab],
            scores: vec![0.0; capacity],
            q8: vec![0; max_input],
            scales: vec![0.0; max_input.div_ceil(32)],
            q8k: vec![
                BlockQ8K {
                    d: 0.0,
                    qs: [0; 256],
                    bsums: [0; 16],
                };
                max_input.div_ceil(256)
            ],
            #[cfg(feature = "vulkan")]
            // ponytail: reuse projection offload until the whole-AR BF16 ops,
            // KV round trip and long-prefix submissions pass real-model checks.
            gpu: Some(None),
        })
    }

    pub fn position(&self) -> usize {
        self.kv.seq_len
    }

    pub fn prefill(&mut self, token_ids: &[u32]) -> Result<&[f32], String> {
        if token_ids.is_empty() {
            return Err("YuE2 prefill requires at least one token".into());
        }
        let end = self
            .kv
            .seq_len
            .checked_add(token_ids.len())
            .ok_or("YuE2 prefill length overflow")?;
        if end > self.kv.capacity {
            return Err(format!(
                "YuE2 prefill requires {end} positions; session capacity is {}",
                self.kv.capacity
            ));
        }
        validate_token_ids(token_ids, self.model.config.vocab)?;
        let base = self.kv.seq_len;

        #[cfg(feature = "vulkan")]
        {
            // Initialize before the prefix so decode sees its device KV.
            // Some(None) remembers fallback and must never retry with empty KV.
            if self.gpu.is_none() {
                self.gpu = Some(match crate::ops::get_vulkan_context() {
                    Some(context) => crate::vulkan::yue2::YuE2VulkanSession::try_new(
                        self.model,
                        self.kv.capacity,
                        context,
                    )
                    .map_err(|error| error.to_string())?,
                    None => None,
                });
            }
            if !matches!(self.gpu, Some(None)) {
                let attempt = self.gpu.get_or_insert(None);
                if let Some(gpu) = attempt {
                    let model = self.model;
                    let kv = &mut self.kv;
                    let logits = &mut self.logits;
                    match prefill_on_gpu(model, gpu, kv, logits, token_ids, base) {
                        Ok(()) => {
                            self.kv.seq_len = end;
                            self.kv.update_access();
                            return Ok(&self.logits);
                        }
                        Err(error) => {
                            eprintln!(
                                "[GPU] YuE2 AR forward failed ({error}); falling back to CPU."
                            );
                            // The device may hold a partial KV shadow, so drop
                            // it and let the CPU path re-derive every position.
                            *attempt = None;
                            self.kv.seq_len = base;
                        }
                    }
                }
            }
        }

        for (index, &token) in token_ids.iter().enumerate() {
            let position = self.kv.seq_len;
            self.forward_token(token, position, index + 1 == token_ids.len())?;
            self.kv.seq_len += 1;
        }
        self.kv.update_access();
        Ok(&self.logits)
    }

    fn forward_token(
        &mut self,
        token_id: u32,
        position: usize,
        project_logits: bool,
    ) -> Result<(), String> {
        let config = &self.model.config;
        self.model
            .token_embedding
            .embedding_lookup(token_id, &mut self.x);
        trace(
            "yue2.ar.embedding",
            None,
            position,
            &[config.hidden],
            &self.x,
        );

        let q_width = config.q_heads * config.head_dim;
        let kv_width = config.kv_heads * config.head_dim;
        let group_size = config.q_heads / config.kv_heads;
        let scale = (config.head_dim as f32).sqrt().recip();
        for (layer_index, layer) in self.model.layers.iter().enumerate() {
            rms_norm(
                &self.x,
                &layer.ar_attention.norm,
                &mut self.normed,
                config.rms_eps,
            );
            trace(
                "yue2.ar.attn_norm",
                Some(layer_index),
                position,
                &[config.hidden],
                &self.normed,
            );
            layer.ar_attention.q.matmul(
                &self.normed,
                &mut self.q,
                &self.model.pool,
                &mut self.q8,
                &mut self.scales,
                &mut self.q8k,
            );
            layer.ar_attention.k.matmul(
                &self.normed,
                &mut self.k,
                &self.model.pool,
                &mut self.q8,
                &mut self.scales,
                &mut self.q8k,
            );
            layer.ar_attention.v.matmul(
                &self.normed,
                &mut self.v,
                &self.model.pool,
                &mut self.q8,
                &mut self.scales,
                &mut self.q8k,
            );
            rms_norm_heads(
                &mut self.q,
                &layer.ar_attention.q_norm,
                config.head_dim,
                config.rms_eps,
            );
            rms_norm_heads(
                &mut self.k,
                &layer.ar_attention.k_norm,
                config.head_dim,
                config.rms_eps,
            );
            trace(
                "yue2.ar.q",
                Some(layer_index),
                position,
                &[config.q_heads, config.head_dim],
                &self.q,
            );
            trace(
                "yue2.ar.k",
                Some(layer_index),
                position,
                &[config.kv_heads, config.head_dim],
                &self.k,
            );
            trace(
                "yue2.ar.v",
                Some(layer_index),
                position,
                &[config.kv_heads, config.head_dim],
                &self.v,
            );
            rope(&mut self.q, position, config.head_dim, config.rope_base);
            rope(&mut self.k, position, config.head_dim, config.rope_base);
            trace(
                "yue2.ar.rope_q",
                Some(layer_index),
                position,
                &[config.q_heads, config.head_dim],
                &self.q,
            );
            trace(
                "yue2.ar.rope_k",
                Some(layer_index),
                position,
                &[config.kv_heads, config.head_dim],
                &self.k,
            );

            let (key_cache, value_cache) = match &mut self.kv.cache {
                KvCache::F32(cache) => (&mut cache.k, &mut cache.v),
                KvCache::F16(_) => return Err("YuE2 AR requires an F32 KV cache".into()),
            };
            let layer_base = layer_index * self.kv.capacity * kv_width;
            let row_base = layer_base + position * kv_width;
            key_cache[row_base..row_base + kv_width].copy_from_slice(&self.k);
            value_cache[row_base..row_base + kv_width].copy_from_slice(&self.v);
            self.attention.fill(0.0);
            for head in 0..config.q_heads {
                let kv_head = head / group_size;
                let q_start = head * config.head_dim;
                let kv_start = kv_head * config.head_dim;
                attention_head(
                    &self.q[q_start..q_start + config.head_dim],
                    &key_cache[layer_base..],
                    &value_cache[layer_base..],
                    kv_width,
                    kv_start,
                    &mut self.scores[..=position],
                    &mut self.attention[q_start..q_start + config.head_dim],
                    scale,
                );
            }
            trace(
                "yue2.ar.attn",
                Some(layer_index),
                position,
                &[q_width],
                &self.attention,
            );
            layer.ar_attention.output.matmul(
                &self.attention,
                &mut self.projected,
                &self.model.pool,
                &mut self.q8,
                &mut self.scales,
                &mut self.q8k,
            );
            trace(
                "yue2.ar.attn_output",
                Some(layer_index),
                position,
                &[config.hidden],
                &self.projected,
            );
            add_in_place(&mut self.x, &self.projected);
            trace(
                "yue2.ar.attn_residual",
                Some(layer_index),
                position,
                &[config.hidden],
                &self.x,
            );

            rms_norm(
                &self.x,
                &layer.ar_mlp.norm,
                &mut self.normed,
                config.rms_eps,
            );
            trace(
                "yue2.ar.ffn_norm",
                Some(layer_index),
                position,
                &[config.hidden],
                &self.normed,
            );
            layer.ar_mlp.gate.matmul(
                &self.normed,
                &mut self.gate,
                &self.model.pool,
                &mut self.q8,
                &mut self.scales,
                &mut self.q8k,
            );
            trace(
                "yue2.ar.ffn_gate",
                Some(layer_index),
                position,
                &[config.ffn],
                &self.gate,
            );
            layer.ar_mlp.up.matmul(
                &self.normed,
                &mut self.up,
                &self.model.pool,
                &mut self.q8,
                &mut self.scales,
                &mut self.q8k,
            );
            trace(
                "yue2.ar.ffn_up",
                Some(layer_index),
                position,
                &[config.ffn],
                &self.up,
            );
            for (gate, &up) in self.gate.iter_mut().zip(&self.up) {
                let activated = half::bf16::from_f32(silu(*gate)).to_f32();
                *gate = half::bf16::from_f32(activated * up).to_f32();
            }
            layer.ar_mlp.down.matmul(
                &self.gate,
                &mut self.down,
                &self.model.pool,
                &mut self.q8,
                &mut self.scales,
                &mut self.q8k,
            );
            trace(
                "yue2.ar.ffn_down",
                Some(layer_index),
                position,
                &[config.hidden],
                &self.down,
            );
            add_in_place(&mut self.x, &self.down);
            trace(
                "yue2.ar.ffn_residual",
                Some(layer_index),
                position,
                &[config.hidden],
                &self.x,
            );
        }

        if !project_logits && !cfg!(feature = "parity-trace") {
            if self.x.iter().any(|value| !value.is_finite()) {
                return Err("YuE2 AR produced non-finite hidden state".into());
            }
            return Ok(());
        }
        rms_norm(
            &self.x,
            &self.model.final_norm,
            &mut self.normed,
            config.rms_eps,
        );
        trace(
            "yue2.ar.final_norm",
            None,
            position,
            &[config.hidden],
            &self.normed,
        );
        self.model.lm_head.matmul(
            &self.normed,
            &mut self.logits,
            &self.model.pool,
            &mut self.q8,
            &mut self.scales,
            &mut self.q8k,
        );
        trace(
            "yue2.ar.logits",
            None,
            position,
            &[config.vocab],
            &self.logits,
        );
        if self.logits.iter().any(|value| !value.is_finite()) {
            return Err("YuE2 AR produced non-finite logits".into());
        }
        Ok(())
    }
}

fn preflight(source: &dyn TensorSource, config: &YuE2Config) -> Result<(), String> {
    for (name, dims) in [
        (
            "model.embed_tokens.weight",
            vec![config.hidden, config.vocab],
        ),
        ("model.norm.weight", vec![config.hidden]),
        ("lm_head.weight", vec![config.hidden, config.vocab]),
        (
            "llm2vae.weight",
            vec![config.hidden, config.latent_channels],
        ),
        ("llm2vae.bias", vec![config.latent_channels]),
        (
            "vae2llm.weight",
            vec![config.latent_channels, config.hidden],
        ),
        ("vae2llm.bias", vec![config.hidden]),
        ("time_embedder.mlp.0.weight", vec![256, config.hidden]),
        ("time_embedder.mlp.0.bias", vec![config.hidden]),
        (
            "time_embedder.mlp.2.weight",
            vec![config.hidden, config.hidden],
        ),
        ("time_embedder.mlp.2.bias", vec![config.hidden]),
        ("latent_pos_embed.pe", vec![config.hidden, config.context]),
    ] {
        // Both time-embedder matrices are quantized by the converter; the rest
        // of this table (vocab embeddings, bridge projections, position table,
        // and every 1-D norm/bias) must stay BF16.
        if name.starts_with("time_embedder.mlp.") && name.ends_with(".weight") {
            require_quantizable(source, name, &dims)?;
        } else {
            require_bf16(source, name, &dims)?;
        }
    }
    let q_width = config.q_heads * config.head_dim;
    let kv_width = config.kv_heads * config.head_dim;
    for layer in 0..config.layers {
        let base = format!("model.layers.{layer}");
        for (suffix, dims) in [
            ("input_layernorm.weight", vec![config.hidden]),
            ("self_attn.q_proj.weight", vec![config.hidden, q_width]),
            ("self_attn.k_proj.weight", vec![config.hidden, kv_width]),
            ("self_attn.v_proj.weight", vec![config.hidden, kv_width]),
            ("self_attn.o_proj.weight", vec![q_width, config.hidden]),
            ("self_attn.q_norm.weight", vec![config.head_dim]),
            ("self_attn.k_norm.weight", vec![config.head_dim]),
            ("post_attention_layernorm.weight", vec![config.hidden]),
            ("mlp.gate_proj.weight", vec![config.hidden, config.ffn]),
            ("mlp.up_proj.weight", vec![config.hidden, config.ffn]),
            ("mlp.down_proj.weight", vec![config.ffn, config.hidden]),
            ("nar_input_layernorm.weight", vec![config.hidden]),
            ("nar_self_attn.q_proj.weight", vec![config.hidden, q_width]),
            ("nar_self_attn.k_proj.weight", vec![config.hidden, kv_width]),
            ("nar_self_attn.v_proj.weight", vec![config.hidden, kv_width]),
            ("nar_self_attn.o_proj.weight", vec![q_width, config.hidden]),
            ("nar_self_attn.q_norm.weight", vec![config.head_dim]),
            ("nar_self_attn.k_norm.weight", vec![config.head_dim]),
            ("nar_pre_mlp_layernorm.weight", vec![config.hidden]),
            ("nar_mlp.gate_proj.weight", vec![config.hidden, config.ffn]),
            ("nar_mlp.up_proj.weight", vec![config.hidden, config.ffn]),
            ("nar_mlp.down_proj.weight", vec![config.ffn, config.hidden]),
        ] {
            // The projections may carry a quantized type; the norms may not.
            if is_matrix_suffix(suffix) {
                require_quantizable(source, &format!("{base}.{suffix}"), &dims)?;
            } else {
                require_bf16(source, &format!("{base}.{suffix}"), &dims)?;
            }
        }
    }
    Ok(())
}

/// True for the per-layer 2-D projections the converter is allowed to quantize.
fn is_matrix_suffix(suffix: &str) -> bool {
    matches!(
        suffix,
        "self_attn.q_proj.weight"
            | "self_attn.k_proj.weight"
            | "self_attn.v_proj.weight"
            | "self_attn.o_proj.weight"
            | "mlp.gate_proj.weight"
            | "mlp.up_proj.weight"
            | "mlp.down_proj.weight"
            | "nar_self_attn.q_proj.weight"
            | "nar_self_attn.k_proj.weight"
            | "nar_self_attn.v_proj.weight"
            | "nar_self_attn.o_proj.weight"
            | "nar_mlp.gate_proj.weight"
            | "nar_mlp.up_proj.weight"
            | "nar_mlp.down_proj.weight"
    )
}

fn require_bf16(source: &dyn TensorSource, name: &str, dims: &[usize]) -> Result<(), String> {
    require_type(source, name, dims, &[GGMLType::BF16])
}

/// Shape check that also accepts a quantized matrix type.
///
/// The converter only re-encodes the transformer projections; everything else
/// (1-D norms, the vocab embeddings, the position/bridge tables) must still be
/// BF16 because the loader reads those through `load_f32_tensor`. `QUANTIZABLE`
/// is the same tensor set the converter applies its `--quant` modes to, so a
/// checkpoint that has, say, a Q8_0 `mlp.gate_proj.weight` loads while a Q8_0
/// `model.norm.weight` is still rejected.
fn require_quantizable(
    source: &dyn TensorSource,
    name: &str,
    dims: &[usize],
) -> Result<(), String> {
    let mut allowed = vec![GGMLType::BF16];
    allowed.extend_from_slice(&MATRIX_TYPES);
    require_type(source, name, dims, &allowed)
}

fn require_type(
    source: &dyn TensorSource,
    name: &str,
    dims: &[usize],
    allowed: &[GGMLType],
) -> Result<(), String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let expected = dims.iter().map(|&value| value as u64).collect::<Vec<_>>();
    if info.dims != expected {
        return Err(format!(
            "Invalid tensor {name} shape {:?}; expected {expected:?}",
            info.dims
        ));
    }
    if !allowed.contains(&info.ggml_type) {
        return Err(format!(
            "Invalid tensor {name} type {:?}; expected one of {allowed:?}",
            info.ggml_type
        ));
    }
    info.checked_nbytes()
        .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?;
    Ok(())
}

fn load_attention(
    source: &dyn TensorSource,
    base: &str,
    norm: &str,
    attention: &str,
    config: &YuE2Config,
) -> Result<YuE2AttentionWeights, String> {
    let q_width = config.q_heads * config.head_dim;
    let kv_width = config.kv_heads * config.head_dim;
    Ok(YuE2AttentionWeights {
        norm: load_f32_tensor(
            source,
            &format!("{base}.{norm}.weight"),
            &[config.hidden as u64],
        )?,
        q_norm: load_f32_tensor(
            source,
            &format!("{base}.{attention}.q_norm.weight"),
            &[config.head_dim as u64],
        )?,
        k_norm: load_f32_tensor(
            source,
            &format!("{base}.{attention}.k_norm.weight"),
            &[config.head_dim as u64],
        )?,
        q: YuE2Weight::load(
            source,
            &format!("{base}.{attention}.q_proj.weight"),
            config.hidden,
            q_width,
        )?,
        k: YuE2Weight::load(
            source,
            &format!("{base}.{attention}.k_proj.weight"),
            config.hidden,
            kv_width,
        )?,
        v: YuE2Weight::load(
            source,
            &format!("{base}.{attention}.v_proj.weight"),
            config.hidden,
            kv_width,
        )?,
        output: YuE2Weight::load(
            source,
            &format!("{base}.{attention}.o_proj.weight"),
            q_width,
            config.hidden,
        )?,
    })
}

fn load_mlp(
    source: &dyn TensorSource,
    base: &str,
    norm: &str,
    mlp: &str,
    config: &YuE2Config,
) -> Result<YuE2MlpWeights, String> {
    Ok(YuE2MlpWeights {
        norm: load_f32_tensor(
            source,
            &format!("{base}.{norm}.weight"),
            &[config.hidden as u64],
        )?,
        gate: YuE2Weight::load(
            source,
            &format!("{base}.{mlp}.gate_proj.weight"),
            config.hidden,
            config.ffn,
        )?,
        up: YuE2Weight::load(
            source,
            &format!("{base}.{mlp}.up_proj.weight"),
            config.hidden,
            config.ffn,
        )?,
        down: YuE2Weight::load(
            source,
            &format!("{base}.{mlp}.down_proj.weight"),
            config.ffn,
            config.hidden,
        )?,
    })
}

#[derive(Clone, Copy)]
pub(super) enum Phase {
    Abc,
    Semantic,
}

impl Phase {
    fn end_token(self) -> u32 {
        match self {
            Self::Abc => ABC_END,
            Self::Semantic => MUSIC_END,
        }
    }

    #[cfg(feature = "parity-trace")]
    fn trace_name(self) -> &'static str {
        match self {
            Self::Abc => "yue2.abc.generated_ids",
            Self::Semantic => "yue2.semantic.generated_ids",
        }
    }

    fn short_name(self) -> &'static str {
        match self {
            Self::Abc => "abc",
            Self::Semantic => "semantic",
        }
    }

    fn allows(self, token: usize) -> bool {
        match self {
            Self::Abc => token < EOD as usize || token == ABC_END as usize,
            Self::Semantic => {
                (CODEC_OFFSET as usize..CODEC_OFFSET as usize + CODEC_SIZE).contains(&token)
                    || token == MUSIC_END as usize
            }
        }
    }
}

pub(super) fn sample_phase_token(
    logits: &[f32],
    history: &[u32],
    sampling: SamplingConfig,
    step: usize,
    phase: Phase,
    rng: &mut Mt19937,
) -> Result<u32, String> {
    if logits.iter().any(|value| !value.is_finite()) {
        return Err("YuE2 sampling received non-finite logits".into());
    }
    let mut scores = logits.to_vec();
    for (token, score) in scores.iter_mut().enumerate() {
        if !phase.allows(token)
            || (step < sampling.min_tokens && token == phase.end_token() as usize)
        {
            *score = f32::NEG_INFINITY;
        }
    }
    let recent = &history[history.len().saturating_sub(sampling.penalty_window)..];
    let mut counts = HashMap::new();
    for &token in recent {
        *counts.entry(token).or_insert(0) += 1;
    }
    crate::ops::sampling::apply_repetition_penalty(
        &mut scores,
        &counts,
        sampling.repetition_penalty,
    );
    if sampling.temperature == 0.0 {
        return argmax_finite(&scores).map(|token| token as u32);
    }
    if sampling.temperature != 1.0 {
        for score in &mut scores {
            *score /= sampling.temperature;
        }
    }
    let mut candidates = scores
        .into_iter()
        .enumerate()
        .filter(|(_, score)| score.is_finite())
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
    candidates.truncate(sampling.top_k.min(candidates.len()));
    if candidates.is_empty() {
        return Err("YuE2 sampling has no allowed tokens".into());
    }
    let max = candidates[0].1;
    let mut total = 0.0f32;
    for (_, value) in &mut candidates {
        *value = (*value - max).exp();
        total += *value;
    }
    if sampling.top_p < 1.0 {
        let mut cumulative = 0.0f32;
        let mut keep = 0;
        for &(_, probability) in &candidates {
            cumulative += probability / total;
            keep += 1;
            if cumulative >= sampling.top_p {
                break;
            }
        }
        candidates.truncate(keep.max(1));
        total = candidates.iter().map(|(_, probability)| probability).sum();
    }
    Ok(torch_multinomial(&mut candidates, total, logits.len(), rng) as u32)
}

fn torch_multinomial(
    candidates: &mut [(usize, f32)],
    total: f32,
    vocabulary: usize,
    rng: &mut Mt19937,
) -> usize {
    candidates.sort_unstable_by_key(|&(token, _)| token);
    let mut candidate = 0;
    let mut best = (0, f32::NEG_INFINITY);
    for token in 0..vocabulary {
        let random = (u64::from(rng.next()) << 32) | u64::from(rng.next());
        if candidate == candidates.len() || candidates[candidate].0 != token {
            continue;
        }
        let uniform = (random & ((1u64 << 53) - 1)) as f64 * (1.0 / (1u64 << 53) as f64);
        let exponential = -(-uniform).ln_1p() as f32;
        let score = (candidates[candidate].1 / total) / exponential;
        if score > best.1 {
            best = (token, score);
        }
        candidate += 1;
    }
    best.0
}

fn argmax_finite(values: &[f32]) -> Result<usize, String> {
    values
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|left, right| left.1.total_cmp(right.1).then(right.0.cmp(&left.0)))
        .map(|(index, _)| index)
        .ok_or_else(|| "YuE2 sampling has no allowed tokens".into())
}

fn checked_product(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{name} overflow"))
}

fn validate_token_ids(token_ids: &[u32], vocab: usize) -> Result<(), String> {
    if let Some(token) = token_ids
        .iter()
        .copied()
        .find(|&token| token as usize >= vocab)
    {
        return Err(format!(
            "YuE2 token ID {token} is outside vocabulary {vocab}"
        ));
    }
    Ok(())
}

pub(super) fn rms_norm(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    let sum = input.iter().map(|v| v * v).sum::<f32>();
    let scale = (sum / input.len() as f32 + eps).sqrt().recip();
    let scale = half::bf16::from_f32(scale).to_f32();
    for ((output, &input), &weight) in output.iter_mut().zip(input).zip(weight) {
        let scaled = half::bf16::from_f32(input * scale).to_f32();
        *output = half::bf16::from_f32(scaled * weight).to_f32();
    }
}

pub(super) fn rms_norm_heads(values: &mut [f32], weight: &[f32], head_dim: usize, eps: f32) {
    let mut normalized = vec![0.0; head_dim];
    for head in values.chunks_exact_mut(head_dim) {
        rms_norm(head, weight, &mut normalized, eps);
        head.copy_from_slice(&normalized);
    }
}

/// Cached sin/cos tables for a single (head_dim, base) pair, extended lazily.
///
/// `rope` is called twice per layer per token and each call used to rebuild the
/// whole table from scratch: `head_dim / 2` `powf` calls plus `head_dim` scalar
/// SLEEF sin/cos evaluations and two heap allocations. Positions only ever grow
/// during a generation, so keeping the tables and appending is exactly equivalent
/// -- every cached entry is the value the original call would have produced,
/// since the values are still computed by the same function.
struct RopeTable {
    head_dim: usize,
    base: f32,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl RopeTable {
    fn new(head_dim: usize, base: f32) -> Self {
        Self {
            head_dim,
            base,
            cos: Vec::new(),
            sin: Vec::new(),
        }
    }

    /// Ensure rows `0..=position` are present and return their range.
    fn range(&mut self, position: usize, head_dim: usize, base: f32) -> (usize, usize) {
        if self.head_dim != head_dim || self.base != base {
            *self = Self::new(head_dim, base);
        }
        let stride = self.head_dim;
        let have = self.cos.len() / stride.max(1);
        if position < have {
            return (position * stride, (position + 1) * stride);
        }
        // Grow to the requested position in one batch so a long generation does
        // not re-extend on every token.
        let want = position + 1;
        let (mut cos, mut sin) = crate::ops::rope::rope_sin_cos_sleef_table_with_threads(
            &(have..want).collect::<Vec<usize>>(),
            self.head_dim,
            self.base,
            1,
        );
        // The YuE2 contract rounds the table through bf16 before the rotation.
        for value in cos.iter_mut().chain(&mut sin) {
            *value = half::bf16::from_f32(*value).to_f32();
        }
        self.cos.append(&mut cos);
        self.sin.append(&mut sin);
        (position * stride, (position + 1) * stride)
    }
}

pub(super) fn rope(values: &mut [f32], position: usize, head_dim: usize, base: f32) {
    use std::sync::{Mutex, OnceLock};
    static TABLE: OnceLock<Mutex<RopeTable>> = OnceLock::new();
    let cell = TABLE.get_or_init(|| Mutex::new(RopeTable::new(head_dim, base)));
    // Hold the lock for the whole call: the rotation reads the table slices
    // while this guard owns them, so a concurrent `rope` on another thread
    // cannot reallocate the vectors out from under the reader.
    let mut guard = cell.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    // `range` is a no-op for a matching (head_dim, base); a different
    // architecture in the same process rebuilds instead of silently returning
    // a table for the wrong width or base.
    let (start, end) = guard.range(position, head_dim, base);
    crate::ops::rope::rope_neox_inplace_with_table(
        values,
        head_dim,
        &guard.cos[start..end],
        &guard.sin[start..end],
    );
}
pub(super) fn dot(left: &[f32], right: &[f32]) -> f32 {
    crate::ops::dot_f32(left, right, left.len())
}

/// Bit-exact vectorized value reduction shared by the NAR attention kernels and
/// the AR single-token decode. Lives here rather than in `nar.rs` because
/// `attention_head` in this file needs it too and `nar` already depends on `ar`.
///
/// Two-segment variant: tokens `0..n_first` come from `values_first`, the rest
/// from `values_second`, but both land in the *same* accumulator.
///
/// The prefix KV and the latent KV are separate allocations, so a 512-wide KV
/// block can straddle the boundary between them. Reducing each segment into its
/// own accumulator and adding both to the output associates the additions as
/// `(out + head) + tail`, while the original single loop produces
/// `out + (head + tail)`. Those round differently, which is enough to change a
/// bf16 rounding decision downstream. One accumulator reproduces the original.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn value_reduce_block2(
    values_first: &[f32],
    base_first: usize,
    values_second: &[f32],
    base_second: usize,
    n_first: usize,
    scores: &[f32],
    output: &mut [f32],
    row_stride: usize,
    head_offset: usize,
    n_tokens: usize,
    head_width: usize,
) {
    debug_assert!(n_tokens > 0);
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if crate::ops::has_neon() && head_width >= 4 {
        // SAFETY: NEON availability is checked, and both segments are bounded
        // by the caller to stay inside their slices.
        unsafe {
            value_reduce_block2_neon(
                values_first,
                base_first,
                values_second,
                base_second,
                n_first,
                scores,
                output,
                row_stride,
                head_offset,
                n_tokens,
                head_width,
            );
        }
        return;
    }
    value_reduce_block2_scalar(
        values_first,
        base_first,
        values_second,
        base_second,
        n_first,
        scores,
        output,
        row_stride,
        head_offset,
        n_tokens,
        head_width,
    );
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn value_reduce_block2_neon(
    values_first: &[f32],
    base_first: usize,
    values_second: &[f32],
    base_second: usize,
    n_first: usize,
    scores: &[f32],
    output: &mut [f32],
    row_stride: usize,
    head_offset: usize,
    n_tokens: usize,
    head_width: usize,
) {
    use std::arch::aarch64::*;
    let mut dim = 0;
    while dim + 4 <= head_width {
        let mut acc = vdupq_n_f32(0.0);
        for token in 0..n_tokens {
            let (values, start) = if token < n_first {
                let row = base_first + token;
                (values_first, row * row_stride + head_offset + dim)
            } else {
                let row = base_second + (token - n_first);
                (values_second, row * row_stride + head_offset + dim)
            };
            let weight = vdupq_n_f32(scores[token]);
            let v = vld1q_f32(values.as_ptr().add(start));
            acc = vaddq_f32(acc, vmulq_f32(weight, v));
        }
        let out = vld1q_f32(output.as_ptr().add(dim));
        vst1q_f32(output.as_mut_ptr().add(dim), vaddq_f32(out, acc));
        dim += 4;
    }
    while dim < head_width {
        let mut sum = 0.0f32;
        for token in 0..n_tokens {
            let (values, start) = if token < n_first {
                let row = base_first + token;
                (values_first, row * row_stride + head_offset + dim)
            } else {
                let row = base_second + (token - n_first);
                (values_second, row * row_stride + head_offset + dim)
            };
            sum += scores[token] * *values.get_unchecked(start);
        }
        *output.get_unchecked_mut(dim) += sum;
        dim += 1;
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn value_reduce_block2_scalar(
    values_first: &[f32],
    base_first: usize,
    values_second: &[f32],
    base_second: usize,
    n_first: usize,
    scores: &[f32],
    output: &mut [f32],
    row_stride: usize,
    head_offset: usize,
    n_tokens: usize,
    head_width: usize,
) {
    for dim in 0..head_width {
        let mut sum = 0.0f32;
        for token in 0..n_tokens {
            let (values, start) = if token < n_first {
                let row = base_first + token;
                (values_first, row * row_stride + head_offset + dim)
            } else {
                let row = base_second + (token - n_first);
                (values_second, row * row_stride + head_offset + dim)
            };
            sum += scores[token] * values[start];
        }
        output[dim] += sum;
    }
}

#[inline]
/// `base` is an **element** offset into `values` (the kernel computes
/// `base + token * row_stride + head_offset + dim`), not a row index. Passing a
/// row number silently reads the wrong elements, which is why the NAR call sites
/// pass `start * kv_width`.
pub(super) fn value_reduce_block(
    values: &[f32],
    scores: &[f32],
    output: &mut [f32],
    base: usize,
    row_stride: usize,
    head_offset: usize,
    n_tokens: usize,
    head_width: usize,
) {
    debug_assert!(n_tokens > 0);
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    {
        if crate::ops::has_neon() && head_width >= 4 && n_tokens > 0 {
            // SAFETY: NEON is available per `has_neon`, and every load below is
            // bounded by `base + n_tokens * row_stride + head_offset + head_width`
            // which the caller guarantees is within `values`.
            unsafe {
                value_reduce_block_neon(
                    values,
                    scores,
                    output,
                    base,
                    row_stride,
                    head_offset,
                    n_tokens,
                    head_width,
                );
            }
            return;
        }
    }
    value_reduce_block_scalar(
        values,
        scores,
        output,
        base,
        row_stride,
        head_offset,
        n_tokens,
        head_width,
    );
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
pub(super) unsafe fn value_reduce_block_neon(
    values: &[f32],
    scores: &[f32],
    output: &mut [f32],
    base: usize,
    row_stride: usize,
    head_offset: usize,
    n_tokens: usize,
    head_width: usize,
) {
    use std::arch::aarch64::*;
    let mut dim = 0;
    while dim + 4 <= head_width {
        // Fresh accumulator per block, and a separate multiply so the product is
        // rounded before the add. Both are load-bearing for bit-exactness.
        let mut acc = vdupq_n_f32(0.0);
        for token in 0..n_tokens {
            let start = base + token * row_stride + head_offset + dim;
            let weight = vdupq_n_f32(scores[token]);
            let v = vld1q_f32(values.as_ptr().add(start));
            acc = vaddq_f32(acc, vmulq_f32(weight, v));
        }
        let out = vld1q_f32(output.as_ptr().add(dim));
        vst1q_f32(output.as_mut_ptr().add(dim), vaddq_f32(out, acc));
        dim += 4;
    }
    while dim < head_width {
        let mut sum = 0.0f32;
        for token in 0..n_tokens {
            let start = base + token * row_stride + head_offset + dim;
            sum += scores[token] * *values.get_unchecked(start);
        }
        *output.get_unchecked_mut(dim) += sum;
        dim += 1;
    }
}

#[inline]
pub(super) fn value_reduce_block_scalar(
    values: &[f32],
    scores: &[f32],
    output: &mut [f32],
    base: usize,
    row_stride: usize,
    head_offset: usize,
    n_tokens: usize,
    head_width: usize,
) {
    for dim in 0..head_width {
        let mut sum = 0.0f32;
        for token in 0..n_tokens {
            let start = base + token * row_stride + head_offset + dim;
            sum += scores[token] * values[start];
        }
        output[dim] += sum;
    }
}

pub(super) fn attention_head(
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    kv_width: usize,
    kv_start: usize,
    scores: &mut [f32],
    output: &mut [f32],
    scale: f32,
) {
    for (cached_position, score) in scores.iter_mut().enumerate() {
        let key_start = cached_position * kv_width + kv_start;
        *score = dot(query, &key_cache[key_start..key_start + query.len()]) * scale;
    }
    if scores.len() <= 512 {
        let inverse_sum = softmax(scores);
        // The scalar form iterated dimension-outer, so consecutive reads of
        // `value_cache` were `kv_width` floats apart and every access landed on a
        // fresh cache line. `value_reduce_block` walks tokens outer and the head
        // dimension inner, which is the contiguous order, and it reproduces the
        // original arithmetic exactly: a fresh zeroed accumulator per call, a
        // separate multiply and add rather than an FMA, and one add into
        // `output` at the end. This is the same kernel the NAR attention uses
        // and it is pinned bit-exact by
        // `nar::attention_parity_tests::optimized_attention_matches_legacy_bitwise`.
        output.fill(0.0);
        value_reduce_block(
            value_cache,
            scores,
            output,
            0,
            kv_width,
            kv_start,
            scores.len(),
            output.len(),
        );
        for value in output.iter_mut() {
            *value = half::bf16::from_f32(*value * inverse_sum).to_f32();
        }
        return;
    }

    output.fill(0.0);
    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0f32;
    for start in (0..scores.len()).step_by(512) {
        let end = scores.len().min(start + 512);
        let block = &mut scores[start..end];
        let block_max = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let next_max = running_max.max(block_max);
        let block_sum = softmax_exp_sum(block, next_max);
        let rescale = (running_max - next_max).exp();
        running_sum = rescale.mul_add(running_sum, block_sum);
        if start > 0 {
            for value in output.iter_mut() {
                *value *= rescale;
            }
        }
        // Same contiguous, bit-exact reduction as the short path above, applied
        // per KV block so the streaming max/rescale arithmetic is untouched.
        // `base` is an element offset, not a row index.
        value_reduce_block(
            value_cache,
            block,
            output,
            start * kv_width,
            kv_width,
            kv_start,
            block.len(),
            output.len(),
        );
        running_max = next_max;
    }
    let inverse_sum = running_sum.recip();
    for value in output.iter_mut() {
        *value = half::bf16::from_f32(*value * inverse_sum).to_f32();
    }
}

pub(super) fn softmax(values: &mut [f32]) -> f32 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    softmax_exp_sum(values, max).recip()
}

fn softmax_exp_sum(values: &mut [f32], max: f32) -> f32 {
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        let exponential = (*value - max).exp();
        sum += exponential;
        *value = half::bf16::from_f32(exponential).to_f32();
    }
    sum
}

pub(super) fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

pub(super) fn add_in_place(target: &mut [f32], update: &[f32]) {
    for (target, &update) in target.iter_mut().zip(update) {
        *target = half::bf16::from_f32(*target + update).to_f32();
    }
}

fn trace(name: &str, layer: Option<usize>, step: usize, shape: &[usize], values: &[f32]) {
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint_at(
        name,
        layer,
        Some(step),
        shape,
        values,
    ));
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, layer, step, shape, values);
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "requires a Vulkan device and YUE2_MODEL_GGUF pointing to a BF16 GGUF"]
    fn vulkan_ar_initializes_before_prefix_and_reuses_session() {
        let path =
            std::env::var("YUE2_MODEL_GGUF").expect("set YUE2_MODEL_GGUF to a BF16 YuE2 GGUF");
        let source: Arc<dyn TensorSource> = Arc::from(
            crate::open_model_source(std::path::Path::new(&path), crate::ComponentRole::Llm)
                .unwrap(),
        );
        let tokenizer = Arc::new(
            BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).unwrap(),
        );
        let model =
            YuE2Model::from_source(source, tokenizer, Arc::new(ComputePool::new(4))).unwrap();
        crate::ops::enable_gpu();
        let context = crate::ops::get_vulkan_context().expect("Vulkan required");
        let before = context.submission_count();
        let mut session = YuE2ArSession::new(&model, 8).unwrap();
        assert!(matches!(session.gpu, Some(None)));
        // Exercise the isolated executor's lifecycle; normal sessions use
        // projection offload until its numerical contract is validated.
        session.gpu = None;
        assert!(session
            .prefill(&[1, 2, 3, 4])
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
        assert!(
            session.gpu.as_ref().is_some_and(Option::is_some),
            "the prefix must populate the device KV before incremental decode"
        );
        assert_eq!(session.position(), 4);
        assert_eq!(context.submission_count(), before + 1);
        assert!(session.prefill(&[5]).unwrap().iter().all(|v| v.is_finite()));
        assert_eq!(session.position(), 5);
        assert_eq!(context.submission_count(), before + 2);
    }

    struct CountingKernel {
        inner: Box<dyn Kernel>,
        calls: Arc<AtomicUsize>,
    }

    impl Kernel for CountingKernel {
        fn forward_prequantized(
            &self,
            input: &[u8],
            scales: &[f32],
            output: &mut [f32],
            n_in: usize,
            n_out: usize,
            thread: usize,
            threads: usize,
        ) {
            self.inner
                .forward_prequantized(input, scales, output, n_in, n_out, thread, threads);
        }

        fn forward_prepared(
            &self,
            input: &[f32],
            quantized: &[u8],
            scales: &[f32],
            q8k: Option<&[BlockQ8K]>,
            output: &mut [f32],
            n_in: usize,
            n_out: usize,
            thread: usize,
            threads: usize,
        ) {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.inner.forward_prepared(
                input, quantized, scales, q8k, output, n_in, n_out, thread, threads,
            );
        }
    }

    #[test]
    fn prefill_projects_only_returned_logits_and_preserves_decode_bits() {
        let mut model = super::super::tests::tiny_yue2_model();
        let mut sequential = YuE2ArSession::new(&model, 8).unwrap();
        let mut expected = Vec::new();
        for token in [1, 2, 3, 4] {
            expected = sequential.prefill(&[token]).unwrap().to_vec();
        }
        let expected_decode = sequential.prefill(&[5]).unwrap().to_vec();
        drop(sequential);

        let calls = Arc::new(AtomicUsize::new(0));
        model.lm_head.fast.kernel = Box::new(CountingKernel {
            inner: model.lm_head.fast.kernel,
            calls: Arc::clone(&calls),
        });
        let mut batched = YuE2ArSession::new(&model, 8).unwrap();
        let actual = batched.prefill(&[1, 2, 3, 4]).unwrap();
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        let expected_calls = if cfg!(feature = "parity-trace") { 4 } else { 1 };
        assert_eq!(calls.load(Ordering::Relaxed), expected_calls);
        assert_eq!(batched.position(), 4);
        assert_eq!(
            batched
                .prefill(&[5])
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected_decode
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(calls.load(Ordering::Relaxed), expected_calls + 1);
    }

    #[test]
    fn bf16_batches_match_single_row_bits_with_bias_tails_and_partitions() {
        for (n_in, n_out) in [(3, 2), (259, 17), (2048, 9), (6144, 5)] {
            let bytes: &'static [u8] = Box::leak(
                (0..n_in * n_out)
                    .flat_map(|index| {
                        half::bf16::from_f32(((index * 17 % 101) as f32 - 50.0) * 0.013)
                            .to_bits()
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            );
            let weight = YuE2Weight {
                fast: Weight::from_quantized(QuantizedTensor::from_bytes(
                    bytes,
                    GGMLType::BF16,
                    n_in,
                    n_out,
                )),
                bf16: Some(bytes),
                #[cfg(feature = "vulkan")]
                gpu_bf16: Default::default(),
                n_in,
                n_out,
            };
            let bias: Vec<f32> = (0..n_out)
                .map(|index| (index as f32 - 4.0) * 0.017)
                .collect();
            for threads in [1, 4] {
                let pool = ComputePool::new(threads);
                for n_rows in [1, 3, 4, 5, 9] {
                    let input: Vec<f32> = (0..n_rows * n_in)
                        .map(|index| ((index * 29 % 73) as f32 - 36.0) * 0.017)
                        .collect();
                    for bias in [None, Some(bias.as_slice())] {
                        let mut expected = vec![0.0; n_rows * n_out];
                        for row in 0..n_rows {
                            torch_bf16_matmul_rows(
                                bytes,
                                &input[row * n_in..(row + 1) * n_in],
                                bias,
                                &mut expected[row * n_out..(row + 1) * n_out],
                                n_in,
                                0,
                            );
                        }
                        let mut actual = vec![f32::NAN; expected.len()];
                        weight
                            .matmul_rows(&input, &mut actual, &pool, &mut RowScratch::new(), bias)
                            .unwrap();
                        assert_eq!(
                            actual
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                            expected
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                            "n_in={n_in} n_out={n_out} n_rows={n_rows} threads={threads} bias={}",
                            bias.is_some()
                        );
                    }
                }
            }
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "requires a Vulkan device; fails if offload is unavailable"]
    fn vulkan_yue2_bf16_rounds_after_bias_across_tiles() {
        crate::ops::enable_gpu();
        let context = crate::ops::get_vulkan_context().expect("Vulkan device required");
        let bytes: &'static [u8] = Box::leak(
            [1.0, 0.0, 0.0, 1.0]
                .into_iter()
                .flat_map(|value| half::bf16::from_f32(value).to_bits().to_le_bytes())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        let weight = YuE2Weight {
            fast: Weight::from_quantized(QuantizedTensor::from_bytes(bytes, GGMLType::BF16, 2, 2)),
            bf16: Some(bytes),
            gpu_bf16: Default::default(),
            n_in: 2,
            n_out: 2,
        };
        let input: Vec<_> = (0..65)
            .flat_map(|row| [1.0029296875 + row as f32 * 0.0625, row as f32 * 0.25])
            .collect();
        let bias = [0.001953125, -0.001953125];
        let mut actual = vec![f32::NAN; 130];
        assert!(
            weight.gpu_bf16.try_matmul(
                bytes,
                crate::vulkan::ops::GpuWeightFormat::BF16Dot,
                &input,
                &mut actual,
                2,
                2,
                65
            ),
            "GPU BF16 projection declined"
        );
        let before = context.submission_count();
        weight.matmul_bf16(&input, Some(&bias), &mut actual, &ComputePool::new(2));
        assert_eq!(context.submission_count(), before + 2);
        assert_eq!(actual[0], 1.0078125); // Rounding before bias would yield 1.0.
        for row in 0..65 {
            let mut expected = [0.0; 2];
            torch_bf16_matmul_rows(
                bytes,
                &input[row * 2..row * 2 + 2],
                Some(&bias),
                &mut expected,
                2,
                0,
            );
            assert_eq!(actual[row * 2..row * 2 + 2], expected);
        }
        // BF16 rounding magnifies a reassociated reduction. CPU SIMD keeps
        // independent FMA streams, while an ascending scalar sum loses a unit term.
        for (width, n_out, token_rows) in [(1024, 2, 65), (1027, 65, 65), (1027, 65, 69)] {
            let bytes: &'static [u8] =
                Box::leak([0x80, 0x3f].repeat(width * n_out).into_boxed_slice());
            let weight = YuE2Weight {
                fast: Weight::from_quantized(QuantizedTensor::from_bytes(
                    bytes,
                    GGMLType::BF16,
                    width,
                    n_out,
                )),
                bf16: Some(bytes),
                gpu_bf16: Default::default(),
                n_in: width,
                n_out,
            };
            let bias: Vec<_> = (0..n_out).map(|column| bias[column % 2]).collect();
            let mut actual = vec![f32::NAN; token_rows * n_out];
            let mut input = vec![0.0; token_rows * width];
            for row in input.chunks_exact_mut(width) {
                row[..4].copy_from_slice(&[33554432.0, 1.0, -33554432.0, 1.0]);
                row[width - 1] = 0.25;
            }
            assert!(weight.gpu_bf16.try_matmul(
                bytes,
                crate::vulkan::ops::GpuWeightFormat::BF16Dot,
                &input,
                &mut actual,
                width,
                n_out,
                token_rows
            ));
            let before = context.submission_count();
            weight.matmul_bf16(&input, Some(&bias), &mut actual, &ComputePool::new(2));
            assert_eq!(context.submission_count(), before + 2);
            for (row, input) in input.chunks_exact(width).enumerate() {
                let mut expected = vec![0.0; n_out];
                torch_bf16_matmul_rows(bytes, input, Some(&bias), &mut expected, width, 0);
                assert_eq!(
                    actual[row * n_out..(row + 1) * n_out],
                    expected,
                    "width={width} row={row}"
                );
            }
        }
    }
}
