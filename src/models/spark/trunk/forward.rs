//! Spark 2.5 forward inference loop.
//!
//! - Fused QKV projection (`attn_qkv` → Q/K/V splits)
//! - Per-layer RoPE: full-attn layers use freq_base_full/n_rot_full;
//!   SWA layers use freq_base_swa/n_rot_swa.
//! - Sliding-window attention: SWA layers only see last `sliding_window`
//!   positions; full-attn layers see the entire prefix.
//! - Per-head gating: `sigmoid(attn_gate @ x_inp)` × attention output.
//! - GeGLU FFN: `down(gelu(gate(x)) * up(x))`.
//! - Tied embeddings: `output.weight` absent → reuse `token_embd.weight`.

use half::f16;

use super::config::SparkConfig;
use super::weights::{load_layers, SparkLayerWeights};

use crate::app::cli::KvFormat;
use crate::core::scratchpad::{KvArch, KvCache, KvLifecycle, KvState};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::ops::kernel::{Kernel, QuantizedTensor, Weight};
use crate::ops::{dot_f32, gelu_inplace, rms_norm, rope_neox_partial, softmax_inplace};

use std::io::{self, Write};
use std::sync::Arc;
use std::time::Instant;

pub(crate) fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Loaded Spark 2.5 model.
pub struct SparkModel {
    pub config: SparkConfig,
    pub layers: Vec<SparkLayerWeights<'static>>,
    pub tok_embd: Weight<'static>,
    pub output_norm: Vec<f32>,
    pub output: Option<Weight<'static>>,
}

impl SparkModel {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let config = SparkConfig::from_source(source)?;

        let output_norm = crate::core::tensor::load_f32_tensor(
            source,
            "output_norm.weight",
            &[config.n_embd as u64],
        )?;

        let tok_embd_bytes = source
            .tensor_slice("token_embd.weight")
            .ok_or_else(|| "Missing token_embd.weight".to_string())?;
        let tok_embd_info = source
            .tensor_info("token_embd.weight")
            .ok_or_else(|| "Missing token_embd.weight info".to_string())?;
        let tok_embd_bytes_static: &'static [u8] = unsafe { std::mem::transmute(tok_embd_bytes) };
        let tok_embd = Weight::from_quantized(QuantizedTensor::from_bytes(
            tok_embd_bytes_static,
            tok_embd_info.ggml_type,
            config.n_embd,
            config.vocab,
        ));

        let output = if source.tensor_info("output.weight").is_some() {
            let bytes = source.tensor_slice("output.weight").unwrap();
            let info = source.tensor_info("output.weight").unwrap();
            let bytes_static: &'static [u8] = unsafe { std::mem::transmute(bytes) };
            Some(Weight::from_quantized(QuantizedTensor::from_bytes(
                bytes_static,
                info.ggml_type,
                config.n_embd,
                config.vocab,
            )))
        } else {
            None
        };

        let layers = load_layers(
            config.n_layer,
            config.n_embd,
            config.n_embd_qkv(),
            config.n_head,
            config.n_ff,
            source,
        );

        Ok(Self {
            config,
            layers,
            tok_embd,
            output_norm,
            output,
        })
    }

    pub fn output_weight(&self) -> &Weight<'static> {
        self.output.as_ref().unwrap_or(&self.tok_embd)
    }
}

pub struct SparkSession {
    pub(crate) config: SparkConfig,
    pub(crate) layers: Vec<SparkLayerWeights<'static>>,
    pub(crate) tok_embd: Weight<'static>,
    pub(crate) output_norm: Vec<f32>,
    pub(crate) output: Option<Weight<'static>>,
    pub(crate) kv_state: KvState,
}

impl SparkSession {
    pub fn new(source: &dyn TensorSource, max_ctx: usize) -> Result<Self, String> {
        let model = SparkModel::from_source(source)?;
        let cfg = &model.config;
        let arch = Arc::new(KvArch::new(
            cfg.n_layer,
            cfg.n_head_kv,
            cfg.n_embd_head_k,
            cfg.n_embd_head_v,
            max_ctx,
        ));
        let kv_state =
            KvState::new(arch, KvFormat::F16, max_ctx).with_lifecycle(KvLifecycle::Ephemeral);
        Ok(Self {
            config: model.config,
            layers: model.layers,
            tok_embd: model.tok_embd,
            output_norm: model.output_norm,
            output: model.output,
            kv_state,
        })
    }

    fn output_weight(&self) -> &Weight<'static> {
        self.output.as_ref().unwrap_or(&self.tok_embd)
    }

    /// Decode one token at position `pos`. Returns the next sampled token id.
    pub fn decode_step(
        &mut self,
        token_id: u32,
        pos: usize,
        temperature: f32,
    ) -> Result<u32, String> {
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let n_head = cfg.n_head;
        let n_head_kv = cfg.n_head_kv;
        let n_embd_head = cfg.n_embd_head_k;
        let n_embd_q = cfg.n_embd_q();
        let n_embd_kv = cfg.n_embd_kv();
        let n_ff = cfg.n_ff;
        let group_size = n_head / n_head_kv;

        let mut hidden = vec![0.0f32; n_embd];
        self.tok_embd.embedding_lookup(token_id, &mut hidden);

        for (il, lw) in self.layers.iter().enumerate() {
            let freq_base = cfg.freq_base_for(il);
            let n_rot = cfg.n_rot_for(il);
            let is_swa = cfg.is_swa[il];
            let sliding_window = cfg.sliding_window;

            // Pre-attention norm
            let mut normed = vec![0.0f32; n_embd];
            rms_norm(&hidden, &lw.attn_norm, &mut normed, cfg.eps);

            // Fused QKV
            let mut qkv = vec![0.0f32; cfg.n_embd_qkv()];
            lw.attn_qkv
                .kernel
                .forward(&normed, &mut qkv, n_embd, cfg.n_embd_qkv());

            let (q, mid) = qkv.split_at_mut(n_embd_q);
            let (k_full, v_full) = mid.split_at_mut(n_embd_kv);
            let k_full = &mut k_full[..n_embd_kv];
            let v_full = &mut v_full[..n_embd_kv];

            rope_neox_partial(q, pos, n_embd_head, n_rot, freq_base);
            rope_neox_partial(k_full, pos, n_embd_head, n_rot, freq_base);

            // Write KV cache for this layer at this position
            {
                let KvCache::F16(cache) = &mut self.kv_state.cache else {
                    panic!("only F16 KV cache supported");
                };
                let kv_stride = n_embd_kv;
                let layer_off = il * (self.kv_state.capacity * kv_stride) + pos * kv_stride;
                for (i, &v) in k_full.iter().enumerate() {
                    cache.k[layer_off + i] = f16::from_f32(v).to_bits();
                }
                for (i, &v) in v_full.iter().enumerate() {
                    cache.v[layer_off + i] = f16::from_f32(v).to_bits();
                }
            }

            // Attention: q[h] @ k[0..pos+1] for each head, softmax, weighted v sum
            let scale = 1.0f32 / (n_embd_head as f32).sqrt();
            let mut attn_out = vec![0.0f32; n_embd_q];
            {
                let KvCache::F16(cache) = &self.kv_state.cache else {
                    panic!("only F16 KV cache supported");
                };
                let kv_stride = n_embd_kv;
                let layer_off_base = il * (self.kv_state.capacity * kv_stride);
                for h in 0..n_head {
                    let q_h = &q[h * n_embd_head..(h + 1) * n_embd_head];
                    let kv_h = h / group_size;
                    let mut scores = vec![0.0f32; pos + 1];
                    for t in 0..=pos {
                        let k_off = layer_off_base + t * n_embd_kv + kv_h * n_embd_head;
                        let k_row_f32: Vec<f32> = cache.k[k_off..k_off + n_embd_head]
                            .iter()
                            .map(|&bits| f16::from_bits(bits).to_f32())
                            .collect();
                        scores[t] = dot_f32(q_h, &k_row_f32, n_embd_head) * scale;
                    }
                    // Sliding-window mask (pre-softmax): -inf outside window so
                    // softmax excludes them from the denominator.
                    if is_swa && sliding_window > 0 && pos + 1 > sliding_window {
                        let start = (pos + 1) - sliding_window;
                        for t in 0..start {
                            scores[t] = f32::NEG_INFINITY;
                        }
                    }
                    softmax_inplace(&mut scores);
                    for t in 0..=pos {
                        if scores[t] == 0.0 {
                            continue;
                        }
                        let v_off = layer_off_base + t * n_embd_kv + kv_h * n_embd_head;
                        let v_row_f32: Vec<f32> = cache.v[v_off..v_off + n_embd_head]
                            .iter()
                            .map(|&bits| f16::from_bits(bits).to_f32())
                            .collect();
                        for d in 0..n_embd_head {
                            attn_out[h * n_embd_head + d] += scores[t] * v_row_f32[d];
                        }
                    }
                }
            }

            // Per-head gating: sigmoid(attn_gate @ x_inp)
            let mut gate_raw = vec![0.0f32; n_head];
            lw.attn_gate
                .kernel
                .forward(&normed, &mut gate_raw, n_embd, n_head);
            let gate: Vec<f32> = gate_raw.iter().map(|&g| sigmoid_f32(g)).collect();
            for h in 0..n_head {
                for d in 0..n_embd_head {
                    attn_out[h * n_embd_head + d] *= gate[h];
                }
            }

            // Output projection
            let mut attn_proj = vec![0.0f32; n_embd];
            lw.attn_output
                .kernel
                .forward(&attn_out, &mut attn_proj, n_embd_q, n_embd);

            // Residual
            for i in 0..n_embd {
                hidden[i] += attn_proj[i];
            }

            // FFN (GeGLU)
            let mut normed2 = vec![0.0f32; n_embd];
            rms_norm(&hidden, &lw.ffn_norm, &mut normed2, cfg.eps);

            let mut gate_proj = vec![0.0f32; n_ff];
            lw.ffn_gate
                .kernel
                .forward(&normed2, &mut gate_proj, n_embd, n_ff);
            gelu_inplace(&mut gate_proj);
            let mut up_proj = vec![0.0f32; n_ff];
            lw.ffn_up
                .kernel
                .forward(&normed2, &mut up_proj, n_embd, n_ff);
            let mut ffn_hidden: Vec<f32> = gate_proj
                .iter()
                .zip(up_proj.iter())
                .map(|(g, u)| g * u)
                .collect();
            let mut ffn_out = vec![0.0f32; n_embd];
            lw.ffn_down
                .kernel
                .forward(&ffn_hidden, &mut ffn_out, n_ff, n_embd);

            // Residual
            for i in 0..n_embd {
                hidden[i] += ffn_out[i];
            }
        }

        // Final norm + logits (tied embeddings)
        let mut final_normed = vec![0.0f32; n_embd];
        rms_norm(&hidden, &self.output_norm, &mut final_normed, cfg.eps);
        let mut logits = vec![0.0f32; cfg.vocab];
        self.output_weight()
            .kernel
            .forward(&final_normed, &mut logits, n_embd, cfg.vocab);

        sample_token(&logits, temperature)
    }
}

fn sample_token(logits: &[f32], temperature: f32) -> Result<u32, String> {
    if temperature == 0.0 {
        let mut best_id = 0u32;
        let mut best = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best {
                best = v;
                best_id = i as u32;
            }
        }
        Ok(best_id)
    } else {
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let mut probs = vec![0.0f32; logits.len()];
        for (i, &v) in logits.iter().enumerate() {
            let p = ((v - max_logit) / temperature).exp();
            probs[i] = p;
            sum += p;
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }
        let target = rand::random::<f32>();
        let mut cum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= target {
                return Ok(i as u32);
            }
        }
        Ok((logits.len() - 1) as u32)
    }
}

pub fn run_inference(
    source: &dyn TensorSource,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    enable_thinking: bool,
    _bench: bool,
    _profile: bool,
    _kv_format: KvFormat,
) -> Result<(), String> {
    let started = Instant::now();
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;

    // Spark 2.5 chat template (from `tokenizer.chat_template` in GGUF).
    // `enable_thinking` (--thinking flag) controls whether the model
    // produces a reasoning block before its reply.
    //
    //   System:    <｜start▁of▁sentence｜><|System|>\nyou are a helpful assistant.<｜end▁of▁sentence｜>
    //   User:      <｜start▁of▁sentence｜><|User|>{content}<｜end▁of▁sentence｜>
    //   Assistant: <｜start▁of▁sentence｜><|Bot|><think>{reasoning}</think>{content}<｜end▁of▁sentence｜>
    //   Gen-prompt:<｜start▁of▁sentence｜><|Bot|><think>  (thinking=true)
    //   Gen-prompt:<｜start▁of▁sentence｜><|Bot|></think>  (thinking=false)
    let sos = "<｜start▁of▁sentence｜>";
    let eos = "<｜end▁of▁sentence｜>";
    let bot_suffix = if enable_thinking {
        "<think>"
    } else {
        "</think>"
    };
    // Match the GGUF `tokenizer.chat_template` Jinja output exactly. The
    // template uses `{{- ... }}` to strip adjacent whitespace, so each
    // role block concatenates directly without `\n` separators between
    // them. The only embedded `\n` is right after `<|System|>`.
    let prompt_text = format!(
        "{sos}<|System|>\nyou are a helpful assistant.{eos}\
         {sos}<|User|>{prompt}{eos}\
         {sos}<|Bot|>{bot_suffix}",
        sos = sos,
        eos = eos,
        bot_suffix = bot_suffix,
    );
    let mut prompt_tokens = tokenizer.encode(
        &prompt_text,
        EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    );
    // Only prepend BOS if `tokenizer.ggml.add_bos_token` is set. Spark2.5
    // sets this to 0; the chat template already emits `<｜start▁of▁sentence｜>`.
    if tokenizer.add_bos() {
        if let Some(bos) = tokenizer.bos_id() {
            prompt_tokens.insert(0, bos);
        }
    }

    let available_threads = std::thread::available_parallelism()
        .map(|threads| threads.get())
        .unwrap_or(4);
    use crate::app::cli::resolve_thread_count;
    let _pool = ComputePool::new(resolve_thread_count(n_threads_arg, available_threads));

    let config = SparkConfig::from_source(source)?;
    println!(
        "Model: spark2_5 | n_embd={} n_layer={} n_head={} n_head_kv={} n_ff={} | loaded in {}ms",
        config.n_embd,
        config.n_layer,
        config.n_head,
        config.n_head_kv,
        config.n_ff,
        started.elapsed().as_millis()
    );
    eprintln!("compute pool: {} threads", _pool.n_threads());
    println!("Prompt: {} ({} tokens)", prompt_text, prompt_tokens.len());

    let mut session = SparkSession::new(source, prompt_tokens.len() + max_tokens)?;

    print!("Output: ");
    io::stdout().flush().map_err(|error| error.to_string())?;

    let inference_started = Instant::now();
    let mut last_token = prompt_tokens[0];
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        let next = session.decode_step(tok, pos, temperature)?;
        last_token = next;
        let _ = tokenizer.decode(&[tok], true);
    }
    let mut generated: Vec<u32> = Vec::new();
    for _ in 0..max_tokens {
        let next = session.decode_step(
            last_token,
            prompt_tokens.len() + generated.len(),
            temperature,
        )?;
        let piece = tokenizer.decode(&[next], true);
        print!("{}", piece);
        io::stdout().flush().map_err(|error| error.to_string())?;
        generated.push(next);
        // Stop on EOS token (matches `tokenizer.ggml.eos_token_id`) or
        // on the trailing `<｜end▁of▁sentence｜>` (the assistant turn
        // delimiter that the model emits at the end of its reply).
        if let Some(eos) = tokenizer.eos_id() {
            if next == eos {
                break;
            }
        }
        if piece.contains("<｜end▁of▁sentence｜>") {
            break;
        }
        last_token = next;
    }

    let elapsed_ms = inference_started.elapsed().as_millis();
    let tps = if elapsed_ms > 0 {
        generated.len() as f64 / elapsed_ms as f64 * 1000.0
    } else {
        0.0
    };
    println!();
    println!(
        "[end-to-end: {} output tokens in {}ms | {:.1} tok/s]",
        generated.len(),
        elapsed_ms,
        tps
    );
    Ok(())
}
