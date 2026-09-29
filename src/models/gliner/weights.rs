//! Weight loading for `arch = "gliner2"`.
//!
//! Tensor names are the converter's: `token_embd`, `rel_embeddings` +
//! `rel_norm` (DeBERTa's relative-position table and its
//! `norm_rel_ebd = "layer_norm"`), the `blk.{i}.attn_*` / `ffn_*` trunk, and
//! the two `classifier.*` linears. `span_rep`, `count_embed` and `count_pred`
//! are not converted — they only run on the NER path.

use crate::core::tensor::TensorSource;
use crate::ops::kernel::{QuantizedTensor, Weight};

pub struct LayerWeights<'a> {
    pub attn_q: Weight<'a>,
    pub attn_q_bias: Vec<f32>,
    pub attn_k: Weight<'a>,
    pub attn_k_bias: Vec<f32>,
    pub attn_v: Weight<'a>,
    pub attn_v_bias: Vec<f32>,
    pub attn_output: Weight<'a>,
    pub attn_output_bias: Vec<f32>,
    pub attn_out_norm: Norm,
    pub ffn_up: Weight<'a>,
    pub ffn_up_bias: Vec<f32>,
    pub ffn_down: Weight<'a>,
    pub ffn_down_bias: Vec<f32>,
    pub output_norm: Norm,
}

#[derive(Debug, Clone)]
pub struct Norm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

pub struct ModelWeights<'a> {
    pub token_embd: &'a [u8],
    pub token_embd_type: crate::core::tensor::GGMLType,
    pub tok_norm: Norm,
    /// `rel_embeddings.weight` [n_embd, pos_ebd_size]; LayerNorm is applied to
    /// the whole table before any projection, so the loader keeps it raw.
    pub rel_embeddings: &'a [u8],
    pub rel_embeddings_type: crate::core::tensor::GGMLType,
    pub rel_norm: Norm,
    pub layers: Vec<LayerWeights<'a>>,
    pub classifier_0: Weight<'a>,
    pub classifier_0_bias: Vec<f32>,
    pub classifier_2: Weight<'a>,
    pub classifier_2_bias: Vec<f32>,
}

fn load_vec<S: TensorSource + ?Sized>(source: &S, name: &str, len: usize) -> Result<Vec<f32>, String> {
    crate::core::tensor::load_f32_tensor(source, name, &[len as u64])
        .map_err(|error| format!("{name}: {error}"))
}

fn load_norm<S: TensorSource + ?Sized>(source: &S, name: &str, width: usize) -> Result<Norm, String> {
    Ok(Norm {
        weight: load_vec(source, &format!("{name}.weight"), width)?,
        bias: load_vec(source, &format!("{name}.bias"), width)?,
    })
}

fn load_weight<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Weight<'a>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    if info.dims != [n_in as u64, n_out as u64] {
        return Err(format!(
            "tensor {name} has dims {:?}, expected [{n_in}, {n_out}]",
            info.dims
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("missing tensor data {name}"))?;
    Ok(Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes,
        info.ggml_type,
        n_in,
        n_out,
    )))
}

pub struct LoadSpec {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_ff: usize,
    pub pos_ebd_size: usize,
    pub classifier_intermediate: usize,
}

pub fn load_weights<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    spec: &LoadSpec,
) -> Result<ModelWeights<'a>, String> {
    let d = spec.n_embd;
    let embd_info = source
        .tensor_info("token_embd.weight")
        .ok_or("missing token_embd.weight")?;
    if embd_info.dims[0] as usize != d {
        return Err(format!(
            "token_embd.weight ne0 is {}, expected {d}",
            embd_info.dims[0]
        ));
    }
    let token_embd = source
        .tensor_slice("token_embd.weight")
        .ok_or("missing token_embd.weight data")?;

    let rel_info = source
        .tensor_info("rel_embeddings.weight")
        .ok_or("missing rel_embeddings.weight")?;
    if rel_info.dims != [d as u64, spec.pos_ebd_size as u64] {
        return Err(format!(
            "rel_embeddings.weight has dims {:?}, expected [{d}, {}]",
            rel_info.dims, spec.pos_ebd_size
        ));
    }
    let rel_embeddings = source
        .tensor_slice("rel_embeddings.weight")
        .ok_or("missing rel_embeddings.weight data")?;

    let mut layers = Vec::with_capacity(spec.n_layer);
    for index in 0..spec.n_layer {
        let name = |suffix: &str| format!("blk.{index}.{suffix}");
        layers.push(LayerWeights {
            attn_q: load_weight(source, &name("attn_q.weight"), d, d)?,
            attn_q_bias: load_vec(source, &name("attn_q.bias"), d)?,
            attn_k: load_weight(source, &name("attn_k.weight"), d, d)?,
            attn_k_bias: load_vec(source, &name("attn_k.bias"), d)?,
            attn_v: load_weight(source, &name("attn_v.weight"), d, d)?,
            attn_v_bias: load_vec(source, &name("attn_v.bias"), d)?,
            attn_output: load_weight(source, &name("attn_output.weight"), d, d)?,
            attn_output_bias: load_vec(source, &name("attn_output.bias"), d)?,
            attn_out_norm: load_norm(source, &name("attn_out_norm"), d)?,
            ffn_up: load_weight(source, &name("ffn_up.weight"), d, spec.n_ff)?,
            ffn_up_bias: load_vec(source, &name("ffn_up.bias"), spec.n_ff)?,
            ffn_down: load_weight(source, &name("ffn_down.weight"), spec.n_ff, d)?,
            ffn_down_bias: load_vec(source, &name("ffn_down.bias"), d)?,
            output_norm: load_norm(source, &name("output_norm"), d)?,
        });
    }

    let classifier_0 = load_weight(
        source,
        "classifier.0.weight",
        d,
        spec.classifier_intermediate,
    )?;
    let classifier_2 = load_weight(source, "classifier.2.weight", spec.classifier_intermediate, 1)?;

    Ok(ModelWeights {
        token_embd,
        token_embd_type: embd_info.ggml_type,
        tok_norm: load_norm(source, "tok_norm", d)?,
        rel_embeddings,
        rel_embeddings_type: rel_info.ggml_type,
        rel_norm: load_norm(source, "rel_norm", d)?,
        layers,
        classifier_0,
        classifier_0_bias: load_vec(source, "classifier.0.bias", spec.classifier_intermediate)?,
        classifier_2,
        classifier_2_bias: load_vec(source, "classifier.2.bias", 1)?,
    })
}
