//! Paraformer ASR: fbank → CMVN → SAN-M encoder → CIF predictor → SAN-M decoder.
//!
//! Non-autoregressive speech recognition: the CIF predictor determines the
//! number of output tokens, and the decoder produces all token logits in one
//! pass (no token-by-token generation).

use crate::core::tensor::{MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::models::funasr::config::FunAsrConfig;
use crate::models::funasr::encoder::{
    add_residual, fsmn_shift_accumulate, kqv_dot_f32, layernorm_fwd, load_f32_vec,
    load_layernorm, load_linear, load_linear_opt_bias, linear_fwd, relu_inplace, LayerNorm, Linear,
    SanmEncoder, SharedMut,
};
use crate::models::funasr::fbank;
use crate::ops::{dot_f32, sigmoid_inplace, softmax_inplace};
use std::sync::Arc;

pub const ARCH: &str = "paraformer";

pub fn is_paraformer(source: &dyn TensorSource) -> bool {
    source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        .is_some_and(|arch| arch == ARCH)
}

// ======================= Decoder layer structs =======================

struct DecoderFfn {
    w1: Linear,
    norm: LayerNorm,
    w2: Linear,
}

struct DecoderLayer {
    norm1: LayerNorm,
    ff: DecoderFfn,
    fsmn: Vec<f32>,
    linear_q: Linear,
    linear_kv: Linear,
    linear_out: Linear,
}

struct FfnOnlyLayer {
    norm1: LayerNorm,
    ff: DecoderFfn,
}

pub struct ParaformerModel {
    encoder: SanmEncoder,
    pool: Arc<ComputePool>,
    cmvn_shift: Vec<f32>,
    cmvn_scale: Vec<f32>,
    cif_conv1d_w: Vec<f32>,
    cif_conv1d_b: Vec<f32>,
    cif_out: Linear,
    decoder_layers: Vec<DecoderLayer>,
    ffn_only: FfnOnlyLayer,
    after_norm: LayerNorm,
    output_layer: Linear,
    vocab: Vec<String>,
    d_model: usize,
    input_size: usize,
    n_head: usize,
    dk: usize,
    kernel_size: usize,
}

impl ParaformerModel {
    pub fn new(source: Arc<dyn TensorSource>, pool: Arc<ComputePool>) -> Result<Self, String> {
        let config = FunAsrConfig {
            tp_blocks: 0,
            ..Default::default()
        };

        let encoder = SanmEncoder::new(source.as_ref(), Arc::clone(&pool), "encoder.", config)?;

        let cmvn_shift = load_f32_vec(source.as_ref(), "cmvn.shift")?;
        let cmvn_scale = load_f32_vec(source.as_ref(), "cmvn.scale")?;

        let cif_conv1d_w = load_f32_vec(source.as_ref(), "predictor.cif_conv1d.weight")?;
        let cif_conv1d_b = load_f32_vec(source.as_ref(), "predictor.cif_conv1d.bias")?;
        let cif_out = load_linear(source.as_ref(), "predictor.cif_output.")?;

        let n_dec_layers = 16usize;
        let mut decoder_layers = Vec::with_capacity(n_dec_layers);
        for i in 0..n_dec_layers {
            decoder_layers.push(load_decoder_layer(source.as_ref(), i)?);
        }
        let ffn_only = load_ffn_only_layer(source.as_ref())?;

        let after_norm = load_layernorm(source.as_ref(), "decoder.after_norm.")?;
        let output_layer = load_linear(source.as_ref(), "decoder.output_layer.")?;
        let vocab = load_vocab(source.as_ref(), "pf.vocab")?;

        let n_head = config.attention_heads;
        let d_model = config.output_size;
        let dk = d_model / n_head;

        Ok(Self {
            encoder,
            pool,
            cmvn_shift,
            cmvn_scale,
            cif_conv1d_w,
            cif_conv1d_b,
            cif_out,
            decoder_layers,
            ffn_only,
            after_norm,
            output_layer,
            vocab,
            d_model,
            input_size: config.input_size,
            n_head,
            dk,
            kernel_size: config.kernel_size,
        })
    }

    pub fn transcribe(&self, samples: &[f32]) -> Result<String, String> {
        let (fbank_data, t_fbank) = fbank::compute_fbank(samples);
        if t_fbank == 0 {
            return Ok(String::new());
        }

        let dim = self.input_size;
        let mut x = fbank_data;
        for i in 0..dim.min(self.cmvn_shift.len()).min(self.cmvn_scale.len()) {
            for t in 0..t_fbank {
                x[t * dim + i] = (x[t * dim + i] + self.cmvn_shift[i]) * self.cmvn_scale[i];
            }
        }

        let scale = (self.d_model as f32).sqrt();
        for v in &mut x {
            *v *= scale;
        }
        fbank::add_position_encoding(&mut x, t_fbank, self.input_size);

        let enc_out = self.encoder.encode(&x, t_fbank)?;
        let t_enc = t_fbank;
        let d_model = self.d_model;

        let alpha = self.cif_predict(&enc_out, t_enc, d_model)?;
        let cif_embeds = cif_integrate_fire(&enc_out, &alpha, t_enc, d_model);
        let n_tokens = cif_embeds.len() / d_model;
        if n_tokens == 0 {
            return Ok(String::new());
        }

        let dec_out = self.decode(&cif_embeds, n_tokens, &enc_out, t_enc)?;
        let logits = linear_fwd(&self.output_layer, &dec_out, n_tokens, &self.pool);
        let vocab_size = self.output_layer.out_dim;

        let token_ids: Vec<usize> = (0..n_tokens)
            .map(|i| {
                let row = &logits[i * vocab_size..(i + 1) * vocab_size];
                let mut best = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                for (j, &v) in row.iter().enumerate() {
                    if v > best_val {
                        best_val = v;
                        best = j;
                    }
                }
                best
            })
            .collect();

        Ok(detokenize(&token_ids, &self.vocab))
    }

    fn cif_predict(&self, enc_out: &[f32], t: usize, dim: usize) -> Result<Vec<f32>, String> {
        let conv_out = conv1d_fwd(
            enc_out,
            &self.cif_conv1d_w,
            &self.cif_conv1d_b,
            t,
            dim,
            dim,
            3,
            &self.pool,
        );

        let mut residual = conv_out;
        for i in 0..t * dim {
            residual[i] += enc_out[i];
        }
        relu_inplace(&mut residual);

        let alpha = linear_fwd(&self.cif_out, &residual, t, &self.pool);
        let mut alpha = alpha;
        sigmoid_inplace(&mut alpha);
        Ok(alpha)
    }

    fn decode(
        &self,
        cif_embeds: &[f32],
        n: usize,
        enc_out: &[f32],
        t_enc: usize,
    ) -> Result<Vec<f32>, String> {
        let dim = self.d_model;
        let mut x = cif_embeds.to_vec();

        for layer in &self.decoder_layers {
            x = self.decoder_layer_fwd(layer, &x, n, enc_out, t_enc, dim);
        }

        x = self.ffn_only_fwd(&x, n, dim);
        x = layernorm_fwd(&self.after_norm, &x, n, &self.pool);
        Ok(x)
    }

    #[inline]
    fn decoder_layer_fwd(
        &self,
        layer: &DecoderLayer,
        x: &[f32],
        n: usize,
        enc_out: &[f32],
        t_enc: usize,
        dim: usize,
    ) -> Vec<f32> {
        let normed1 = layernorm_fwd(&layer.norm1, x, n, &self.pool);
        let ff_out = self.ffn_fwd(&layer.ff, &normed1, n, dim);
        let mut h = add_residual(x, &ff_out, n * dim);

        let fsmn = fsmn_shift_accumulate(&h, n, dim, self.kernel_size, &layer.fsmn, &self.pool);
        h = add_residual(&h, &fsmn, n * dim);

        let q = linear_fwd(&layer.linear_q, &h, n, &self.pool);
        let kv = linear_fwd(&layer.linear_kv, enc_out, t_enc, &self.pool);
        let (k, v) = split_kv(&kv, t_enc, dim);
        let attn = cross_attention(
            &q, &k, &v, n, t_enc, dim, self.n_head, self.dk, &self.pool,
        );
        let o = linear_fwd(&layer.linear_out, &attn, n, &self.pool);
        h = add_residual(&h, &o, n * dim);
        h
    }

    #[inline]
    fn ffn_fwd(&self, ff: &DecoderFfn, x: &[f32], n: usize, _dim: usize) -> Vec<f32> {
        let h1 = linear_fwd(&ff.w1, x, n, &self.pool);
        let mut h1_relu = h1;
        relu_inplace(&mut h1_relu);
        let normed = layernorm_fwd(&ff.norm, &h1_relu, n, &self.pool);
        linear_fwd(&ff.w2, &normed, n, &self.pool)
    }

    #[inline]
    fn ffn_only_fwd(&self, x: &[f32], n: usize, dim: usize) -> Vec<f32> {
        let normed1 = layernorm_fwd(&self.ffn_only.norm1, x, n, &self.pool);
        let ff_out = self.ffn_fwd(&self.ffn_only.ff, &normed1, n, dim);
        add_residual(x, &ff_out, n * dim)
    }
}

// ======================= Weight loading =======================

fn load_decoder_ffn(source: &dyn TensorSource, prefix: &str) -> Result<DecoderFfn, String> {
    Ok(DecoderFfn {
        w1: load_linear(source, &format!("{prefix}feed_forward.w_1."))?,
        norm: load_layernorm(source, &format!("{prefix}feed_forward.norm."))?,
        w2: load_linear_opt_bias(source, &format!("{prefix}feed_forward.w_2."))?,
    })
}

fn load_decoder_layer(source: &dyn TensorSource, idx: usize) -> Result<DecoderLayer, String> {
    let p = format!("decoder.decoders.{idx}.");
    let sa = format!("{p}self_attn.");
    let ca = format!("{p}src_attn.");
    Ok(DecoderLayer {
        norm1: load_layernorm(source, &format!("{p}norm1."))?,
        ff: load_decoder_ffn(source, &p)?,
        fsmn: load_f32_vec(source, &format!("{sa}fsmn_block.weight"))?,
        linear_q: load_linear(source, &format!("{ca}linear_q."))?,
        linear_kv: load_linear(source, &format!("{ca}linear_k_v."))?,
        linear_out: load_linear(source, &format!("{ca}linear_out."))?,
    })
}

fn load_ffn_only_layer(source: &dyn TensorSource) -> Result<FfnOnlyLayer, String> {
    let p = "decoder.decoders3.0.";
    Ok(FfnOnlyLayer {
        norm1: load_layernorm(source, &format!("{p}norm1."))?,
        ff: load_decoder_ffn(source, &p)?,
    })
}

fn load_vocab(source: &dyn TensorSource, key: &str) -> Result<Vec<String>, String> {
    let arr = source
        .metadata(key)
        .and_then(MetaValue::to_arr)
        .ok_or_else(|| format!("missing metadata: {key}"))?;
    Ok(arr
        .iter()
        .map(|v| v.to_string_val().unwrap_or("").to_string())
        .collect())
}

// ======================= Forward primitives =======================

fn conv1d_fwd(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    t: usize,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    pool: &ComputePool,
) -> Vec<f32> {
    let pad = (kernel - 1) / 2;
    let mut padded = vec![0.0f32; (t + 2 * pad) * in_ch];
    for i in 0..t {
        padded[(i + pad) * in_ch..(i + pad + 1) * in_ch]
            .copy_from_slice(&input[i * in_ch..(i + 1) * in_ch]);
    }

    let mut out = vec![0.0f32; t * out_ch];
    let out_ptr = SharedMut(out.as_mut_ptr());
    let stride = in_ch * kernel;

    pool.compute(move |ith, nth| {
        let per = t.div_ceil(nth);
        let start = ith * per;
        let end = (start + per).min(t);
        for t_idx in start..end {
            let out_row = unsafe { out_ptr.slice(t_idx * out_ch, out_ch) };
            for c in 0..out_ch {
                let mut sum = bias[c];
                let w_off = c * stride;
                for k in 0..kernel {
                    let pad_idx = t_idx + k;
                    for c2 in 0..in_ch {
                        sum += weight[w_off + c2 * kernel + k] * padded[pad_idx * in_ch + c2];
                    }
                }
                out_row[c] = sum;
            }
        }
    });

    out
}

fn split_kv(kv: &[f32], t: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut k = vec![0.0f32; t * dim];
    let mut v = vec![0.0f32; t * dim];
    for i in 0..t {
        let row = &kv[i * 2 * dim..(i + 1) * 2 * dim];
        k[i * dim..(i + 1) * dim].copy_from_slice(&row[..dim]);
        v[i * dim..(i + 1) * dim].copy_from_slice(&row[dim..]);
    }
    (k, v)
}

fn cross_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    t: usize,
    dim: usize,
    n_head: usize,
    dk: usize,
    pool: &ComputePool,
) -> Vec<f32> {
    let scale = 1.0 / (dk as f32).sqrt();
    let mut out = vec![0.0f32; n * dim];
    let out_ptr = SharedMut(out.as_mut_ptr());

    pool.compute(move |ith, nth| {
        let per = n_head.div_ceil(nth);
        let h_start = ith * per;
        let h_end = (h_start + per).min(n_head);
        if h_start >= h_end {
            return;
        }
        let mut scores = vec![0.0f32; n * t];
        for h in h_start..h_end {
            let off = h * dk;
            for i in 0..n {
                let q_row = &q[i * dim + off..i * dim + off + dk];
                for j in 0..t {
                    let k_row = &k[j * dim + off..j * dim + off + dk];
                    scores[i * t + j] = dot_f32(q_row, k_row, dk) * scale;
                }
            }
            for i in 0..n {
                let row = &mut scores[i * t..(i + 1) * t];
                softmax_inplace(row);
            }
            for i in 0..n {
                let out_row = unsafe { out_ptr.slice(i * dim + off, dk) };
                kqv_dot_f32(out_row, v, &scores[i * t..(i + 1) * t], t, dim, off);
            }
        }
    });

    out
}

fn cif_integrate_fire(
    enc_out: &[f32],
    alpha: &[f32],
    t: usize,
    dim: usize,
) -> Vec<f32> {
    let threshold = 1.0f32;
    let tail_threshold = 0.45f32;

    let mut embeddings: Vec<f32> = Vec::new();
    let mut acc = 0.0f32;
    let mut cache = vec![0.0f32; dim];

    for t_idx in 0..t {
        let w = alpha[t_idx];
        acc += w;
        for d in 0..dim {
            cache[d] += w * enc_out[t_idx * dim + d];
        }

        if acc >= threshold {
            let fire: Vec<f32> = cache.iter().map(|&v| v / acc).collect();
            embeddings.extend_from_slice(&fire);
            let residual = acc - threshold;
            cache = fire.iter().map(|&v| v * residual).collect();
            acc = residual;
        }
    }

    if acc >= tail_threshold {
        let fire: Vec<f32> = cache.iter().map(|&v| v / acc).collect();
        embeddings.extend_from_slice(&fire);
    }

    embeddings
}

fn detokenize(token_ids: &[usize], vocab: &[String]) -> String {
    let mut text = String::new();
    for &id in token_ids {
        if id == 1 || id == 2 {
            continue;
        }
        if let Some(word) = vocab.get(id) {
            text.push_str(word);
        }
    }
    let text = text.replace("@@", "");
    let text = text.replace('\u{2581}', " ").replace('▁', " ");
    text.trim().to_string()
}
