//! Per-request Qwen3 generation state and prompt/decode orchestration.

use super::forward::{Qwen3GenerateOptions, Qwen3Generation, Qwen3Input};
use super::prefill::Qwen3PrefillScratch;
use super::util::{
    check_allocation, checked_generated_position, checked_product, checked_session_capacity,
    sample_token, validate_generation,
};
use super::weights::Qwen3Model;
use crate::core::scratchpad::{
    ExecutionScratchpad, KvArch, KvCache, KvFormat, KvLifecycle, KvState,
};
#[cfg(feature = "parity-trace")]
use crate::parity_trace;
#[cfg(feature = "vulkan")]
use crate::vulkan::qwen3::{commit_shadow_kv, Qwen3VulkanSession};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct Qwen3Session<'model> {
    pub(crate) model: &'model Qwen3Model,
    pub(crate) kv_state: KvState,
    pub(crate) scratch: ExecutionScratchpad,
    pub(super) prefill_scratch: Qwen3PrefillScratch,
    pub(crate) capacity: usize,
    #[cfg(feature = "vulkan")]
    pub(crate) gpu: Option<Qwen3VulkanSession>,
    #[cfg(feature = "vulkan")]
    pub(crate) full_model_gpu_failed: bool,
    #[cfg(test)]
    pub(super) fail_cpu_prefill_after_layer: Option<usize>,
}

impl<'model> Qwen3Session<'model> {
    pub fn new(model: &'model Qwen3Model, capacity: usize) -> Result<Self, String> {
        Self::new_with_kv_state(model, capacity, KvFormat::F16, KvLifecycle::Ephemeral)
    }

    pub fn new_with_kv_state(
        model: &'model Qwen3Model,
        capacity: usize,
        kv_format: KvFormat,
        lifecycle: KvLifecycle,
    ) -> Result<Self, String> {
        if capacity == 0 || capacity > model.config.n_ctx {
            return Err(format!(
                "Session capacity {capacity} must be within 1..={}",
                model.config.n_ctx
            ));
        }
        let config = &model.config;
        let n_embd_q = checked_product("query width", config.n_head, config.n_embd_head_k)?;
        let n_embd_k = checked_product("key width", config.n_head_kv, config.n_embd_head_k)?;
        let n_embd_v = checked_product("value width", config.n_head_kv, config.n_embd_head_v)?;
        let n_attn = checked_product(
            "attention output width",
            config.n_head,
            config.n_embd_head_v,
        )?;
        let kv_stride = n_embd_k.max(n_embd_v);
        let kv_size = checked_product(
            "KV cache values",
            checked_product("KV cache rows", config.n_layer, capacity)?,
            kv_stride,
        )?;
        let kv_bytes = match kv_format {
            KvFormat::F16 => check_allocation("KV cache", kv_size, std::mem::size_of::<u16>())?,
            KvFormat::F32 => check_allocation("KV cache", kv_size, std::mem::size_of::<f32>())?,
        };
        let max_n_in = n_embd_q.max(n_attn).max(config.n_ff);
        let score_stride = capacity
            .checked_add(255)
            .map(|value| value / 256 * 256)
            .ok_or_else(|| "Attention score stride overflow".to_string())?;
        let score_values =
            checked_product("attention scores", model.pool.n_threads(), score_stride)?;
        for (name, len, bytes) in [
            ("hidden state", config.n_embd, std::mem::size_of::<f32>()),
            (
                "normalized state",
                config.n_embd,
                std::mem::size_of::<f32>(),
            ),
            ("queries", n_embd_q, std::mem::size_of::<f32>()),
            ("keys", kv_stride, std::mem::size_of::<f32>()),
            ("values", kv_stride, std::mem::size_of::<f32>()),
            ("attention output", n_attn, std::mem::size_of::<f32>()),
            (
                "attention projection",
                config.n_embd,
                std::mem::size_of::<f32>(),
            ),
            ("down projection", config.n_embd, std::mem::size_of::<f32>()),
            ("gate projection", config.n_ff, std::mem::size_of::<f32>()),
            ("up projection", config.n_ff, std::mem::size_of::<f32>()),
            ("logits", config.vocab, std::mem::size_of::<f32>()),
            ("quantized activations", max_n_in, std::mem::size_of::<u8>()),
            (
                "quantization scales",
                max_n_in / 32,
                std::mem::size_of::<f32>(),
            ),
            ("attention scores", score_values, std::mem::size_of::<f32>()),
        ] {
            check_allocation(name, len, bytes)?;
        }

        let arch = Arc::new(KvArch::new(
            config.n_layer,
            config.n_head_kv,
            config.n_embd_head_k,
            config.n_embd_head_v,
            config.n_ctx,
        ));
        let mut kv_state = KvState::new(arch, kv_format, capacity).with_lifecycle(lifecycle);
        match (kv_format, &mut kv_state.cache) {
            (KvFormat::F16, KvCache::F16(cache)) => {
                cache.k = vec![0; kv_size];
                cache.v = vec![0; kv_size];
            }
            (KvFormat::F32, KvCache::F32(cache)) => {
                cache.k = vec![0.0; kv_size];
                cache.v = vec![0.0; kv_size];
            }
            _ => unreachable!("format and cache variant mismatch"),
        }
        let _ = kv_bytes;

        #[cfg(feature = "vulkan")]
        let (gpu, full_model_gpu_failed) = match crate::ops::get_vulkan_context() {
            Some(context) => match Qwen3VulkanSession::try_new(model, capacity, context) {
                Ok(gpu) => (gpu, false),
                Err(error) => {
                    eprintln!(
                        "[GPU] Qwen3 Vulkan session unavailable: {error}. Falling back to CPU."
                    );
                    (None, true)
                }
            },
            None => (None, false),
        };

        Ok(Self {
            model,
            kv_state,
            scratch: ExecutionScratchpad {
                x: vec![0.0; config.n_embd],
                normed: vec![0.0; config.n_embd],
                q: vec![0.0; n_embd_q],
                k_new: vec![0.0; kv_stride],
                v_new: vec![0.0; kv_stride],
                attn_out: vec![0.0; n_attn],
                attn_proj: vec![0.0; config.n_embd],
                down_buf: vec![0.0; config.n_embd],
                gate_buf: vec![0.0; config.n_ff],
                up_buf: vec![0.0; config.n_ff],
                logits: vec![0.0; config.vocab],
                q8_buf: vec![0; max_n_in],
                scale_buf: vec![0.0; max_n_in / 32],
                q8k_buf: vec![
                    crate::ops::quant::BlockQ8K {
                        d: 0.0,
                        qs: [0; 256],
                        bsums: [0; 16],
                    };
                    max_n_in / 256
                ],
                score_stride,
                scores: vec![0.0; score_values],
            },
            prefill_scratch: Qwen3PrefillScratch::new(1, model),
            capacity,
            #[cfg(feature = "vulkan")]
            gpu,
            #[cfg(feature = "vulkan")]
            full_model_gpu_failed,
            #[cfg(test)]
            fail_cpu_prefill_after_layer: None,
        })
    }

    pub fn kv_state(&self) -> &KvState {
        &self.kv_state
    }

    pub fn last_logits(&self) -> &[f32] {
        &self.scratch.logits
    }

    pub fn reset_kv(&mut self) {
        self.kv_state.reset();
        #[cfg(feature = "vulkan")]
        if let Some(gpu) = &mut self.gpu {
            gpu.reset();
        }
    }

    pub fn generate(
        &mut self,
        input: Qwen3Input<'_>,
        options: Qwen3GenerateOptions,
    ) -> Result<Qwen3Generation, String> {
        self.generate_with_asr_trace(input, options, false)
    }

    pub fn generate_streaming(
        &mut self,
        input: Qwen3Input<'_>,
        options: Qwen3GenerateOptions,
        mut on_token: impl FnMut(&str),
    ) -> Result<Qwen3Generation, String> {
        self.generate_streaming_until(input, options, |text| {
            if !text.is_empty() {
                on_token(text);
            }
            true
        })
    }

    /// Return false from the callback to stop generation. Empty text callbacks
    /// still allow cancellation when a token has not completed a UTF-8 character.
    pub fn generate_streaming_until(
        &mut self,
        input: Qwen3Input<'_>,
        options: Qwen3GenerateOptions,
        mut on_token: impl FnMut(&str) -> bool,
    ) -> Result<Qwen3Generation, String> {
        validate_generation(self.model, &input, &options)?;
        let required = checked_session_capacity(
            input.token_ids.len(),
            options.max_new_tokens,
            self.model.config.n_ctx,
        )?;
        if required > self.capacity {
            return Err(format!(
                "Generation requires capacity {required}; session has {}",
                self.capacity
            ));
        }
        self.generate_inner(input, options, false, Some(&mut on_token))
    }

    pub(crate) fn generate_with_asr_trace(
        &mut self,
        input: Qwen3Input<'_>,
        options: Qwen3GenerateOptions,
        asr_trace: bool,
    ) -> Result<Qwen3Generation, String> {
        validate_generation(self.model, &input, &options)?;
        let required = checked_session_capacity(
            input.token_ids.len(),
            options.max_new_tokens,
            self.model.config.n_ctx,
        )?;
        if required > self.capacity {
            return Err(format!(
                "Generation requires capacity {required}; session has {}",
                self.capacity
            ));
        }
        self.generate_inner(input, options, asr_trace, None)
    }

    fn generate_inner(
        &mut self,
        input: Qwen3Input<'_>,
        options: Qwen3GenerateOptions,
        asr_trace: bool,
        mut on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<Qwen3Generation, String> {
        let model = self.model;
        let config = &model.config;
        let n_prompt = input.token_ids.len();

        #[cfg(feature = "parity-trace")]
        {
            if asr_trace {
                parity_trace::report(parity_trace::token_ids("asr.prompt_ids", input.token_ids));
                let mut positions = Vec::with_capacity(input.positions.len() * 4);
                for position in input.positions {
                    positions.extend_from_slice(position);
                }
                parity_trace::report(parity_trace::usize_values(
                    "asr.positions",
                    &[input.positions.len(), 4],
                    &positions,
                ));
            } else {
                parity_trace::report(parity_trace::token_ids("prompt_ids", input.token_ids));
                let positions = input
                    .positions
                    .iter()
                    .map(|position| position[0])
                    .collect::<Vec<_>>();
                parity_trace::report(parity_trace::usize_values(
                    "qwen3.positions",
                    &[positions.len()],
                    &positions,
                ));
            }
        }
        #[cfg(not(feature = "parity-trace"))]
        let _ = asr_trace;

        #[cfg(feature = "vulkan")]
        let submission_count =
            || crate::ops::get_vulkan_context().map_or(0, |ctx| ctx.submission_count());
        #[cfg(feature = "vulkan")]
        let before_prompt = submission_count();
        let prompt_duration = self.prefill(&input, options.prefill_batch_size)?;
        #[cfg(feature = "vulkan")]
        let after_prompt = submission_count();
        #[cfg(feature = "parity-trace")]
        if asr_trace {
            parity_trace::report(parity_trace::checkpoint(
                "asr.decoder_first_logits",
                None,
                &[config.vocab],
                &self.scratch.logits,
            ));
        }

        let mut generated_tokens = Vec::new();
        generated_tokens
            .try_reserve_exact(options.max_new_tokens)
            .map_err(|error| format!("Failed to allocate generated tokens: {error}"))?;
        let mut rendered_tokens = Vec::new();
        rendered_tokens
            .try_reserve_exact(options.max_new_tokens)
            .map_err(|error| format!("Failed to allocate rendered tokens: {error}"))?;
        let mut decoder = model.tokenizer.streaming_decoder(false);
        let mut decode_duration = Duration::ZERO;

        while generated_tokens.len() < options.max_new_tokens {
            let token_id = sample_token(&self.scratch.logits, options.temperature)?;
            if model.tokenizer.eos_id() == Some(token_id)
                || model.tokenizer.special_token_id("im_end") == Some(token_id)
            {
                break;
            }
            let text = decoder.push(token_id);
            let keep_going = on_token.as_mut().is_none_or(|callback| callback(&text));
            if !text.is_empty() {
                rendered_tokens.push(text);
            }
            generated_tokens.push(token_id);
            if !keep_going || generated_tokens.len() == options.max_new_tokens {
                break;
            }

            let position = checked_generated_position(input.positions, generated_tokens.len() - 1)?;
            let eval_started = Instant::now();
            let token_ids = [token_id];
            let positions = [position];
            let decode_input = Qwen3Input {
                token_ids: &token_ids,
                positions: &positions,
                embeddings: None,
                deepstack_embeddings: None,
            };

            #[cfg(feature = "vulkan")]
            let used_vulkan = {
                model
                    .token_embedding
                    .embedding_lookup(token_id, &mut self.scratch.x);
                let mut disable_reason = None;
                let gpu_position = self.kv_state.seq_len;
                let used = match self.gpu.as_mut() {
                    Some(gpu) => match gpu.forward_token(&self.scratch.x, gpu_position) {
                        Ok(result) => {
                            if let Err(error) = commit_shadow_kv(
                                &mut self.kv_state,
                                gpu_position,
                                result.k_delta,
                                result.v_delta,
                            ) {
                                gpu.abort_token();
                                disable_reason = Some(error);
                                false
                            } else {
                                self.scratch.logits.copy_from_slice(result.logits);
                                gpu.commit_token();
                                true
                            }
                        }
                        Err(error) => {
                            disable_reason = Some(error.to_string());
                            false
                        }
                    },
                    None => false,
                };
                if let Some(reason) = disable_reason {
                    eprintln!(
                        "[GPU] Qwen3 Vulkan session disabled after error: {reason}. Falling back to CPU."
                    );
                    self.gpu = None;
                    self.full_model_gpu_failed = true;
                }
                used
            };
            #[cfg(not(feature = "vulkan"))]
            let used_vulkan = false;

            if !used_vulkan {
                let base = self.kv_state.seq_len;
                self.forward_cpu_chunk(&decode_input, 0..1, true)?;
                self.kv_state.seq_len = base + 1;
                self.kv_state.update_access();
            }
            decode_duration += eval_started.elapsed();

            #[cfg(all(feature = "parity-trace", feature = "vulkan"))]
            if used_vulkan {
                parity_trace::report(parity_trace::checkpoint(
                    "result_output",
                    None,
                    &[config.vocab],
                    &self.scratch.logits,
                ));
            }
        }

        #[cfg(feature = "parity-trace")]
        parity_trace::report(parity_trace::token_ids(
            if asr_trace {
                "asr.generated_ids"
            } else {
                "generated_ids"
            },
            &generated_tokens,
        ));
        let tail = decoder.finish();
        if !tail.is_empty() {
            if let Some(callback) = on_token.as_mut() {
                callback(&tail);
            }
            rendered_tokens.push(tail);
        }
        Ok(Qwen3Generation {
            text: rendered_tokens.concat(),
            rendered_tokens,
            token_ids: generated_tokens,
            prompt_tokens: n_prompt,
            prompt_duration,
            decode_duration,
            #[cfg(feature = "vulkan")]
            prompt_submissions: after_prompt - before_prompt,
            #[cfg(feature = "vulkan")]
            decode_submissions: submission_count() - after_prompt,
        })
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod vulkan_tests {
    use super::*;
    use crate::core::tensor::{GGMLType, MetaValue, TensorInfo, TensorSource};
    use crate::ops::kernel::{QuantizedTensor, Weight};
    use crate::vulkan::VulkanContext;
    use std::collections::HashMap;

    struct FixtureSource(HashMap<String, (TensorInfo, &'static [u8])>);

    impl TensorSource for FixtureSource {
        fn metadata(&self, _: &str) -> Option<&MetaValue> {
            None
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.0.get(name).map(|value| &value.0)
        }
        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            self.0.get(name).map(|value| value.1)
        }
    }

    fn fixture_model() -> Qwen3Model {
        let mut model = super::super::tests::deterministic_session_model(160);
        let mut tensors = HashMap::new();
        let layer = &mut model.layers[0];
        for (name, weight, seed) in [
            ("blk.0.attn_q.weight", &mut layer.wq, 1),
            ("blk.0.attn_k.weight", &mut layer.wk, 2),
            ("blk.0.attn_v.weight", &mut layer.wv, 3),
            ("blk.0.attn_output.weight", &mut layer.wo, 4),
            ("blk.0.ffn_gate.weight", &mut layer.w_gate, 5),
            ("blk.0.ffn_up.weight", &mut layer.w_up, 6),
            ("blk.0.ffn_down.weight", &mut layer.w_down, 7),
            ("token_embd.weight", &mut model.token_embedding, 8),
            ("output.weight", &mut model.output, 9),
        ] {
            let (n_in, n_out) = (weight.n_in, weight.n_out);
            let bytes = (0..n_out)
                .flat_map(|row| {
                    (0..n_in).flat_map(move |column| {
                        let value = ((row * 7 + column * 13 + seed) % 23) as f32 / 64.0 - 0.171875;
                        crate::ops::f32_to_f16(value).to_le_bytes()
                    })
                })
                .collect::<Vec<_>>();
            let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
            *weight = Weight::from_quantized(QuantizedTensor::from_bytes(
                bytes,
                GGMLType::F16,
                n_in,
                n_out,
            ));
            weight.n_in = n_in;
            weight.n_out = n_out;
            tensors.insert(
                name.into(),
                (
                    TensorInfo {
                        name: name.into(),
                        dims: vec![n_in as u64, n_out as u64],
                        ggml_type: GGMLType::F16,
                        offset: 0,
                    },
                    bytes,
                ),
            );
        }
        model.source = Arc::new(FixtureSource(tensors));
        model
    }

    fn snapshot(state: &KvState) -> (usize, Vec<u32>) {
        let mut words = Vec::new();
        let stride = state.arch.n_head_kv * state.arch.n_embd_head_k;
        for layer in 0..state.arch.n_layer {
            let start = layer * state.capacity * stride;
            let end = start + state.seq_len * stride;
            match &state.cache {
                KvCache::F16(cache) => words.extend(
                    cache.k[start..end]
                        .iter()
                        .chain(&cache.v[start..end])
                        .map(|&word| word as u32),
                ),
                KvCache::F32(cache) => words.extend(
                    cache.k[start..end]
                        .iter()
                        .chain(&cache.v[start..end])
                        .map(|word| word.to_bits()),
                ),
            }
        }
        (state.seq_len, words)
    }

    fn run_failure_fixture(
        prompt_len: usize,
        failure: Option<usize>,
    ) -> (Vec<u32>, (usize, Vec<u32>), Vec<u32>) {
        let model = fixture_model();
        let mut session = Qwen3Session::new(&model, prompt_len + 3).unwrap();
        session.gpu = None;
        session.full_model_gpu_failed = true;
        let mut failure_context = None;
        if let Some(row) = failure {
            let context: &'static VulkanContext =
                Box::leak(Box::new(VulkanContext::new().unwrap()));
            let mut gpu = Qwen3VulkanSession::try_new(&model, session.capacity, context)
                .unwrap()
                .unwrap();
            gpu.fail_after_row = Some(row);
            session.gpu = Some(gpu);
            session.full_model_gpu_failed = false;
            failure_context = Some(context);
        }
        let token_ids: Vec<_> = (0..prompt_len).map(|row| (row % 7) as u32).collect();
        let positions: Vec<_> = (0..prompt_len).map(|row| [row, 0, 0, 0]).collect();
        let generation = session
            .generate(
                Qwen3Input {
                    token_ids: &token_ids,
                    positions: &positions,
                    embeddings: None,
                    deepstack_embeddings: None,
                },
                Qwen3GenerateOptions {
                    max_new_tokens: 3,
                    temperature: 0.0,
                    prefill_batch_size: 4,
                },
            )
            .unwrap();
        assert!(session.gpu.is_none());
        if let Some(context) = failure_context {
            assert_eq!(
                context.submission_count(),
                1,
                "failure must occur after GPU row work"
            );
        }
        (
            session.last_logits().iter().map(|v| v.to_bits()).collect(),
            snapshot(session.kv_state()),
            generation.token_ids,
        )
    }

    fn run_qwen3_forced_gpu_failure(
        prompt_len: usize,
        fail_after_row: usize,
    ) -> (Vec<u32>, (usize, Vec<u32>), Vec<u32>) {
        run_failure_fixture(prompt_len, Some(fail_after_row))
    }

    fn run_qwen3_cpu_only(prompt_len: usize) -> (Vec<u32>, (usize, Vec<u32>), Vec<u32>) {
        run_failure_fixture(prompt_len, None)
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn qwen3_gpu_chunk_failure_recomputes_the_whole_chunk_on_cpu() {
        assert_eq!(run_qwen3_forced_gpu_failure(4, 1), run_qwen3_cpu_only(4));
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn qwen3_gpu_chunk_failure_preserves_earlier_commit_and_cpu_error_context() {
        let model = fixture_model();
        let context = Box::leak(Box::new(VulkanContext::new().unwrap()));
        let mut actual = Qwen3Session::new(&model, 9).unwrap();
        actual.gpu = Qwen3VulkanSession::try_new(&model, 9, context).unwrap();
        let prefix = Qwen3Input {
            token_ids: &[0, 1],
            positions: &[[0, 0, 0, 0], [1, 0, 0, 0]],
            embeddings: None,
            deepstack_embeddings: None,
        };
        actual.prefill(&prefix, 4).unwrap();
        assert_eq!(context.submission_count(), 1);
        let committed = snapshot(actual.kv_state());
        let mut expected = Qwen3Session::new(&model, 9).unwrap();
        expected.gpu = None;
        expected.full_model_gpu_failed = true;
        let (KvCache::F16(target), KvCache::F16(source)) =
            (&mut expected.kv_state.cache, &actual.kv_state.cache)
        else {
            panic!("F16 fixture")
        };
        target.k.copy_from_slice(&source.k);
        target.v.copy_from_slice(&source.v);
        expected.kv_state.seq_len = actual.kv_state.seq_len;
        actual.gpu.as_mut().unwrap().fail_after_row = Some(1);
        actual.fail_cpu_prefill_after_layer = Some(0);
        let input = Qwen3Input {
            token_ids: &[2, 3, 4, 5],
            positions: &[[2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]],
            embeddings: None,
            deepstack_embeddings: None,
        };
        let error = actual.prefill(&input, 4).unwrap_err();
        assert!(
            error.contains("CPU prefill failure after layer 0"),
            "{error}"
        );
        assert!(error.contains("GPU failure after row 1"), "{error}");
        assert_eq!(snapshot(actual.kv_state()), committed);
        assert_eq!(context.submission_count(), 2);
        assert!(actual.gpu.is_none());
        let options = Qwen3GenerateOptions {
            max_new_tokens: 3,
            temperature: 0.0,
            prefill_batch_size: 4,
        };
        let actual_tokens = actual.generate(input, options).unwrap().token_ids;
        let expected_tokens = expected
            .generate(
                Qwen3Input {
                    token_ids: &[2, 3, 4, 5],
                    positions: &[[2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]],
                    embeddings: None,
                    deepstack_embeddings: None,
                },
                Qwen3GenerateOptions {
                    max_new_tokens: 3,
                    temperature: 0.0,
                    prefill_batch_size: 4,
                },
            )
            .unwrap()
            .token_ids;
        assert_eq!(actual_tokens, expected_tokens);
        assert_eq!(snapshot(actual.kv_state()), snapshot(expected.kv_state()));
        assert_eq!(
            actual
                .last_logits()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
                .last_logits()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn qwen3_gpu_second_chunk_failure_retries_original_nonzero_range() {
        let model = fixture_model();
        let context = Box::leak(Box::new(VulkanContext::new().unwrap()));
        let tokens = [0, 1, 2, 3, 4, 5, 6, 0];
        let positions: Vec<_> = (0..8).map(|row| [row, 0, 0, 0]).collect();
        let mut expected = Qwen3Session::new(&model, 11).unwrap();
        expected.gpu = Qwen3VulkanSession::try_new(&model, 11, context).unwrap();
        expected
            .prefill(
                &Qwen3Input {
                    token_ids: &tokens[..4],
                    positions: &positions[..4],
                    embeddings: None,
                    deepstack_embeddings: None,
                },
                4,
            )
            .unwrap();
        assert_eq!(context.submission_count(), 1);
        expected.gpu = None;
        expected.full_model_gpu_failed = true;
        let options = Qwen3GenerateOptions {
            max_new_tokens: 3,
            temperature: 0.0,
            prefill_batch_size: 4,
        };
        let expected_tokens = expected
            .generate(
                Qwen3Input {
                    token_ids: &tokens[4..],
                    positions: &positions[4..],
                    embeddings: None,
                    deepstack_embeddings: None,
                },
                options.clone(),
            )
            .unwrap()
            .token_ids;
        let mut actual = Qwen3Session::new(&model, 11).unwrap();
        actual.gpu = Qwen3VulkanSession::try_new(&model, 11, context).unwrap();
        actual.gpu.as_mut().unwrap().fail_after_row = Some(5);
        let actual_tokens = actual
            .generate(
                Qwen3Input {
                    token_ids: &tokens,
                    positions: &positions,
                    embeddings: None,
                    deepstack_embeddings: None,
                },
                options,
            )
            .unwrap()
            .token_ids;
        assert!(
            actual.gpu.is_none(),
            "failure must target row one of the second chunk"
        );
        assert_eq!(context.submission_count(), 3);
        assert_eq!(actual_tokens, expected_tokens);
        assert_eq!(snapshot(actual.kv_state()), snapshot(expected.kv_state()));
        assert_eq!(
            actual
                .last_logits()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
                .last_logits()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn qwen3_vulkan_prefill_batches_match_bits_kv_and_submissions() {
        let model = fixture_model();
        let context = Box::leak(Box::new(VulkanContext::new().unwrap()));
        let tokens: Vec<_> = (0..65).map(|row| (row % 7) as u32).collect();
        let positions: Vec<_> = (0..65).map(|row| [row, 0, 0, 0]).collect();
        let mut baseline = None;
        for batch in [1, 4, 64, 128] {
            let mut session = Qwen3Session::new(&model, 68).unwrap();
            session.gpu = Qwen3VulkanSession::try_new(&model, 68, context).unwrap();
            let before = context.submission_count();
            let input = Qwen3Input {
                token_ids: &tokens,
                positions: &positions,
                embeddings: None,
                deepstack_embeddings: None,
            };
            session.prefill(&input, batch).unwrap();
            assert!(session.gpu.is_some(), "batch {batch} fell back");
            assert_eq!(
                context.submission_count() - before,
                65_usize.div_ceil(batch) as u64
            );
            let result = (
                session
                    .last_logits()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                snapshot(session.kv_state()),
            );
            if let Some(expected) = &baseline {
                assert_eq!(&result, expected, "batch {batch}");
            } else {
                baseline = Some(result);
            }
            let before = context.submission_count();
            let next = Qwen3Input {
                token_ids: &[0],
                positions: &[[65, 0, 0, 0]],
                embeddings: None,
                deepstack_embeddings: None,
            };
            session
                .generate(
                    next,
                    Qwen3GenerateOptions {
                        max_new_tokens: 3,
                        temperature: 0.0,
                        prefill_batch_size: batch,
                    },
                )
                .unwrap();
            assert_eq!(
                context.submission_count() - before,
                3,
                "decode must remain one submission per token"
            );
        }
    }
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;

    #[test]
    fn controlled_stream_stops_after_first_generated_token() {
        let model = super::super::tests::deterministic_session_model(16);
        let mut session = Qwen3Session::new(&model, 16).unwrap();
        let mut callbacks = 0;
        let generation = session
            .generate_streaming_until(
                Qwen3Input {
                    token_ids: &[1],
                    positions: &[[0, 0, 0, 0]],
                    embeddings: None,
                    deepstack_embeddings: None,
                },
                Qwen3GenerateOptions {
                    max_new_tokens: 8,
                    temperature: 0.0,
                    prefill_batch_size: 1,
                },
                |_| {
                    callbacks += 1;
                    false
                },
            )
            .unwrap();
        assert_eq!(callbacks, 1);
        assert_eq!(generation.token_ids.len(), 1);
    }
}
