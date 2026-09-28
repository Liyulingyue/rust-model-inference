//! Xing4.0-29B-A4B layer weights.
//!
//! Per layer the tensor set is the MLA trio (`attn_q_a/b`,
//! `attn_kv_a_mqa`, `attn_k_b`, `attn_v_b`), the hyper-connection mixer
//! (`hc_{attn,ffn}_{fn,base,scale}`), and either the dense FFN
//! (`ffn_{gate,up,down}`) or the MoE (`ffn_gate_inp`,
//! `ffn_{gate,up,down}_exps`, `exp_probs_b.bias` plus the shared expert
//! `ffn_{gate,up,down}_shexp`).

use super::config::Xing4Config;
use crate::core::tensor::{load_f32_tensor, TensorSource};
use crate::ops::kernel::{QuantizedTensor, Weight};

pub struct Xing4LayerWeights<'a> {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,

    // MLA
    pub wq_a: Weight<'a>,
    pub wq_b: Weight<'a>,
    pub wkv_a_mqa: Weight<'a>,
    pub attn_q_a_norm: Vec<f32>,
    pub attn_kv_a_norm: Vec<f32>,
    /// `[n_head]` entries, each `[nope, kv_lora_rank]` — one matrix per
    /// head, used to absorb the query's nope part into the shared latent
    /// space.
    pub wk_b: Vec<Weight<'a>>,
    /// `[n_head]` entries, each `[kv_lora_rank, v_dim]` — expands the
    /// attended latent back into per-head values.
    pub wv_b: Vec<Weight<'a>>,
    pub wo: Weight<'a>,

    // mHC
    pub hc_attn_fn: Weight<'a>,
    pub hc_attn_base: Vec<f32>,
    pub hc_attn_scale: Vec<f32>,
    pub hc_ffn_fn: Weight<'a>,
    pub hc_ffn_base: Vec<f32>,
    pub hc_ffn_scale: Vec<f32>,

    // FFN (dense or MoE)
    pub dense: DenseFfn<'a>,
    pub moe: Option<MoeFfn<'a>>,
}

pub struct DenseFfn<'a> {
    pub w_gate: Weight<'a>,
    pub w_up: Weight<'a>,
    pub w_down: Weight<'a>,
}

pub struct MoeFfn<'a> {
    pub router: Vec<f32>,
    pub exp_probs_b: Vec<f32>,
    pub gate_exps: Vec<Weight<'a>>,
    pub up_exps: Vec<Weight<'a>>,
    pub down_exps: Vec<Weight<'a>>,
    pub shared_gate: Weight<'a>,
    pub shared_up: Weight<'a>,
    pub shared_down: Weight<'a>,
}

pub struct Xing4Weights<'a> {
    pub tok_embd: Weight<'a>,
    pub output_norm: Vec<f32>,
    pub output: Weight<'a>,
    /// MTP block (`nextn_predict_layers` == 1): `eh_proj` fuses the
    /// embedded next token with the trunk's hidden state.
    pub nextn_eh_proj: Option<Weight<'a>>,
    pub nextn_embed_tokens: Option<Weight<'a>>,
    pub nextn_enorm: Option<Vec<f32>>,
    pub nextn_hnorm: Option<Vec<f32>>,
    pub nextn_shared_head_head: Option<Weight<'a>>,
    pub nextn_shared_head_norm: Option<Vec<f32>>,
}

fn load_f32<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    expected: usize,
) -> Result<Vec<f32>, String> {
    load_f32_tensor(source, name, &[expected as u64])
        .map_err(|e| format!("{name}: {e}"))
}

/// Load an F32/BF16 tensor with any shape, returning it flattened. Used
/// for `ffn_gate_inp` (`[n_embd, n_expert]`) where the 2-D layout does
/// not carry meaning for the router dot products — row-major order gives
/// expert `e` the contiguous span `[e * n_embd, (e + 1) * n_embd)`, which
/// is what the forward reads.
fn load_f32_flat<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    expected: usize,
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("tensor {name} not found"))?;
    if !matches!(info.ggml_type, crate::core::tensor::GGMLType::F32 | crate::core::tensor::GGMLType::BF16) {
        return Err(format!("{name}: expected F32/BF16, found {:?}", info.ggml_type));
    }
    let total: u64 = info.dims.iter().product();
    if total as usize != expected {
        return Err(format!(
            "{name}: element count {total} != expected {expected}"
        ));
    }
    load_f32_tensor(source, name, &info.dims).map_err(|e| format!("{name}: {e}"))
}

fn load_weight<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Weight<'a>, String> {
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("tensor {name} not found"))?;
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("tensor info {name} not found"))?;
    Ok(Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes, info.ggml_type, n_in, n_out,
    )))
}

/// Split a 3-D expert tensor into per-expert 2-D weights. GGML flattens
/// ne[0] fastest, so each expert is one contiguous span.
fn expert_weights<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    n_expert: usize,
    n_in: usize,
    n_ff: usize,
) -> Result<Vec<Weight<'a>>, String> {
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("tensor {name} not found"))?;
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("tensor info {name} not found"))?;
    if bytes.len() % n_expert != 0 {
        return Err(format!("{name}: byte size not divisible by {n_expert} experts"));
    }
    let per = bytes.len() / n_expert;
    Ok((0..n_expert)
        .map(|e| {
            Weight::from_quantized(QuantizedTensor::from_bytes(
                &bytes[e * per..(e + 1) * per],
                info.ggml_type,
                n_in,
                n_ff,
            ))
        })
        .collect())
}

pub fn load_layers<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    cfg: &Xing4Config,
) -> Result<Vec<Xing4LayerWeights<'a>>, String> {
    let n_embd = cfg.n_embd;
    let q_out = cfg.n_head * cfg.n_embd_head_qk();
    let hc_dim = cfg.hc_flat();
    let mix_dim = cfg.hc_mix_dim();

    (0..cfg.n_layer)
        .map(|l| {
            let is_dense = cfg.is_dense_ffn(l);
            let moe = if is_dense {
                None
            } else {
                Some(MoeFfn {
                    router: load_f32_flat(
                        source,
                        &format!("blk.{l}.ffn_gate_inp.weight"),
                        cfg.n_expert * n_embd,
                    )?,
                    exp_probs_b: load_f32_vec(
                        source,
                        &format!("blk.{l}.exp_probs_b.bias"),
                        cfg.n_expert,
                    )?,
                    gate_exps: expert_weights(
                        source,
                        &format!("blk.{l}.ffn_gate_exps.weight"),
                        cfg.n_expert,
                        n_embd,
                        cfg.n_ff_exp,
                    )?,
                    up_exps: expert_weights(
                        source,
                        &format!("blk.{l}.ffn_up_exps.weight"),
                        cfg.n_expert,
                        n_embd,
                        cfg.n_ff_exp,
                    )?,
                    down_exps: expert_weights(
                        source,
                        &format!("blk.{l}.ffn_down_exps.weight"),
                        cfg.n_expert,
                        cfg.n_ff_exp,
                        n_embd,
                    )?,
                    shared_gate: load_weight(
                        source,
                        &format!("blk.{l}.ffn_gate_shexp.weight"),
                        n_embd,
                        cfg.n_ff_exp,
                    )?,
                    shared_up: load_weight(
                        source,
                        &format!("blk.{l}.ffn_up_shexp.weight"),
                        n_embd,
                        cfg.n_ff_exp,
                    )?,
                    shared_down: load_weight(
                        source,
                        &format!("blk.{l}.ffn_down_shexp.weight"),
                        cfg.n_ff_exp,
                        n_embd,
                    )?,
                })
            };

            Ok(Xing4LayerWeights {
                attn_norm: load_f32(source, &format!("blk.{l}.attn_norm.weight"), n_embd)?,
                ffn_norm: load_f32(source, &format!("blk.{l}.ffn_norm.weight"), n_embd)?,

                wq_a: load_weight(
                    source,
                    &format!("blk.{l}.attn_q_a.weight"),
                    n_embd,
                    cfg.q_lora_rank,
                )?,
                wq_b: load_weight(
                    source,
                    &format!("blk.{l}.attn_q_b.weight"),
                    cfg.q_lora_rank,
                    q_out,
                )?,
                wkv_a_mqa: load_weight(
                    source,
                    &format!("blk.{l}.attn_kv_a_mqa.weight"),
                    n_embd,
                    cfg.kv_cache_width(),
                )?,
                attn_q_a_norm: load_f32(
                    source,
                    &format!("blk.{l}.attn_q_a_norm.weight"),
                    cfg.q_lora_rank,
                )?,
                attn_kv_a_norm: load_f32(
                    source,
                    &format!("blk.{l}.attn_kv_a_norm.weight"),
                    cfg.kv_lora_rank,
                )?,
                // GGML stores per-head matrices as [in, out] (ne0=in),
                // so `wk_b[h] @ q_nope` (in=nope, out=kv_lora) and
                // `wv_b[h] @ o_latent` (in=kv_lora, out=v_dim) both
                // work with the shared `from_bytes(n_in, n_out)` helper.
                wk_b: per_head_tensors(
                    source,
                    &format!("blk.{l}.attn_k_b.weight"),
                    cfg.n_head,
                    cfg.n_embd_head_k_nope(),
                    cfg.kv_lora_rank,
                )?,
                wv_b: per_head_tensors(
                    source,
                    &format!("blk.{l}.attn_v_b.weight"),
                    cfg.n_head,
                    cfg.kv_lora_rank,
                    cfg.n_embd_head_v,
                )?,
                wo: load_weight(
                    source,
                    &format!("blk.{l}.attn_output.weight"),
                    cfg.n_head * cfg.n_embd_head_v,
                    n_embd,
                )?,

                hc_attn_fn: load_weight(
                    source,
                    &format!("blk.{l}.hc_attn_fn.weight"),
                    hc_dim,
                    mix_dim,
                )?,
                hc_attn_base: load_f32(source, &format!("blk.{l}.hc_attn_base.weight"), mix_dim)?,
                hc_attn_scale: load_f32(source, &format!("blk.{l}.hc_attn_scale.weight"), 3)?,
                hc_ffn_fn: load_weight(
                    source,
                    &format!("blk.{l}.hc_ffn_fn.weight"),
                    hc_dim,
                    mix_dim,
                )?,
                hc_ffn_base: load_f32(source, &format!("blk.{l}.hc_ffn_base.weight"), mix_dim)?,
                hc_ffn_scale: load_f32(source, &format!("blk.{l}.hc_ffn_scale.weight"), 3)?,

                // Dense weights are only present on the leading
                // `n_layer_dense_lead` layers; the rest expose the MoE
                // tensors instead. Placeholders for the absent branch keep
                // the struct non-optional: the forward never touches them.
                dense: if is_dense {
                    DenseFfn {
                        w_gate: load_weight(
                            source,
                            &format!("blk.{l}.ffn_gate.weight"),
                            n_embd,
                            cfg.n_ff,
                        )?,
                        w_up: load_weight(
                            source,
                            &format!("blk.{l}.ffn_up.weight"),
                            n_embd,
                            cfg.n_ff,
                        )?,
                        w_down: load_weight(
                            source,
                            &format!("blk.{l}.ffn_down.weight"),
                            cfg.n_ff,
                            n_embd,
                        )?,
                    }
                } else {
                    // Borrow the first shared-expert projection as a
                    // stand-in; it is never invoked on MoE layers.
                    let shexp_gate = load_weight(
                        source,
                        &format!("blk.{l}.ffn_gate_shexp.weight"),
                        n_embd,
                        cfg.n_ff_exp,
                    )?;
                    DenseFfn {
                        w_gate: shexp_gate,
                        w_up: load_weight(
                            source,
                            &format!("blk.{l}.ffn_up_shexp.weight"),
                            n_embd,
                            cfg.n_ff_exp,
                        )?,
                        w_down: load_weight(
                            source,
                            &format!("blk.{l}.ffn_down_shexp.weight"),
                            cfg.n_ff_exp,
                            n_embd,
                        )?,
                    }
                },
                moe,
            })
        })
        .collect()
}

fn load_f32_vec<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    expected: usize,
) -> Result<Vec<f32>, String> {
    if source.tensor_info(name).is_none() {
        return Ok(vec![0.0; expected]);
    }
    load_f32(source, name, expected)
}

/// Split a 3-D quantized tensor `[d0, d1, d2]` (GGML: d0 fastest) into
/// `d2` per-slice 2-D weights of shape `[d0, d1]`, so a head's matrix is
/// addressable directly.
fn per_head_tensors<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    d0: usize,
    d1: usize,
    d2: usize,
) -> Result<Vec<Weight<'a>>, String> {
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("tensor {name} not found"))?;
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("tensor info {name} not found"))?;
    if bytes.len() % d2 != 0 {
        return Err(format!("{name}: byte size not divisible by {d2} heads"));
    }
    let per = bytes.len() / d2;
    Ok((0..d2)
        .map(|h| {
            Weight::from_quantized(QuantizedTensor::from_bytes(
                &bytes[h * per..(h + 1) * per],
                info.ggml_type,
                d0,
                d1,
            ))
        })
        .collect())
}

pub fn load_global<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    cfg: &Xing4Config,
) -> Result<Xing4Weights<'a>, String> {
    let n_embd = cfg.n_embd;
    // GGML stores embeddings as [n_embd, n_vocab] (ne0 = n_embd = in), so
    // token `t`'s vector is the contiguous span `[t*n_embd, (t+1)*n_embd)`.
    // Getting (n_in, n_out) backwards makes the embedding read a strided
    // slice instead, which silently produces a garbage residual stream.
    let embd = load_weight(source, "token_embd.weight", n_embd, cfg.n_vocab)?;
    let out = load_weight(source, "output.weight", n_embd, cfg.n_vocab)?;
    let base = cfg.n_layer;

    let nextn = |name: &str| Option::<String>::Some(format!("blk.{base}.nextn.{name}.weight"));

    Ok(Xing4Weights {
        tok_embd: embd,
        output_norm: load_f32(source, "output_norm.weight", n_embd)?,
        output: out,
        nextn_eh_proj: nextn("eh_proj")
            .and_then(|n| load_weight(source, &n, 2 * n_embd, n_embd).ok()),
        nextn_embed_tokens: nextn("embed_tokens")
            .and_then(|n| load_weight(source, &n, n_embd, cfg.n_vocab).ok()),
        nextn_enorm: nextn("enorm").and_then(|n| load_f32(source, &n, n_embd).ok()),
        nextn_hnorm: nextn("hnorm").and_then(|n| load_f32(source, &n, n_embd).ok()),
        nextn_shared_head_head: nextn("shared_head_head")
            .and_then(|n| load_weight(source, &n, n_embd, cfg.n_vocab).ok()),
        nextn_shared_head_norm: nextn("shared_head_norm")
            .and_then(|n| load_f32(source, &n, n_embd).ok()),
    })
}
