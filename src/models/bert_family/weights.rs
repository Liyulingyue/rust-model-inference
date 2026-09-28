//! Weight loading for the BERT encoder family.
//!
//! Tensor names are the ones `bert.cpp` requests via `LLM_TENSOR_*`
//! (`references/llama.cpp/src/models/bert.cpp:22-62`), which in GGUF land is
//! the `blk.{l}.attn_*` / `blk.{l}.ffn_*` / `blk.{l}.*_norm` vocabulary.

use crate::core::tensor::{GGMLType, TensorSource};
use crate::ops::kernel::{QuantizedTensor, Weight};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BertVariant {
    /// `arch = "bert"` — absolute `pos_embd`, GELU FFN over a single `ffn_up`.
    Bert,
    /// `arch = "jina-bert-v2"` — ALiBi, required `token_types`, GEGLU FFN.
    JinaBertV2,
}

impl BertVariant {
    pub fn arch_name(self) -> &'static str {
        match self {
            BertVariant::Bert => "bert",
            BertVariant::JinaBertV2 => "jina-bert-v2",
        }
    }

    /// `f_max_alibi_bias > 0` ⇒ ALiBi (`jina-bert-v2.cpp:5`, `llama-model.cpp:1483`).
    pub fn uses_alibi(self) -> bool {
        matches!(self, BertVariant::JinaBertV2)
    }

    /// GEGLU (`ffn_gate` + `ffn_up`) vs plain GELU (`ffn_up` only).
    pub fn uses_gelu_gate(self) -> bool {
        matches!(self, BertVariant::JinaBertV2)
    }

    pub fn from_arch(arch: &str) -> Option<Self> {
        match arch {
            "bert" => Some(BertVariant::Bert),
            "jina-bert-v2" => Some(BertVariant::JinaBertV2),
            _ => None,
        }
    }
}

/// ALiBi `f_max_alibi_bias` for jina-bert-v2 (`jina-bert-v2.cpp:5`).
pub const MAX_ALIBI_BIAS_JINA_V2: f32 = 8.0;

pub struct BertLayerWeights<'a> {
    pub wq: Weight<'a>,
    pub wq_bias: Bias,
    pub wk: Weight<'a>,
    pub wk_bias: Bias,
    pub wv: Weight<'a>,
    pub wv_bias: Bias,
    pub wo: Weight<'a>,
    pub wo_bias: Bias,
    pub attn_out_norm: NormWithBias,
    pub layer_out_norm: NormWithBias,
    pub ffn_up: Weight<'a>,
    pub ffn_gate: Option<Weight<'a>>,
    pub ffn_up_bias: Bias,
    pub ffn_gate_bias: Bias,
    pub ffn_down: Weight<'a>,
    pub ffn_down_bias: Bias,
}

/// A projection bias. Empty when the GGUF omits it (BERT tensors are
/// `TENSOR_NOT_REQUIRED` for biases; jina-bert-v2 ships none).
#[derive(Default)]
pub struct Bias {
    pub values: Option<Vec<f32>>,
}

impl Bias {
    pub fn add_to(&self, output: &mut [f32]) {
        if let Some(values) = &self.values {
            debug_assert_eq!(values.len(), output.len());
            for (slot, value) in output.iter_mut().zip(values) {
                *slot += *value;
            }
        }
    }
}

#[derive(Default)]
pub struct NormWithBias {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

pub struct BertWeights<'a> {
    pub variant: BertVariant,
    pub n_layer: usize,
    pub token_embd: &'a [u8],
    pub token_embd_ggml_type: GGMLType,
    /// `token_embd_norm.weight` — LayerNorm affine scale over `n_embd`.
    pub tok_norm: NormWithBias,
    /// `token_types.weight` [n_embd, n_token_types]; row 0 ("sentence A")
    /// is added to the embeddings. Present for jina-bert-v2, absent for some
    /// `bert` packers (`bert.cpp:28` marks it NOT_REQUIRED).
    pub token_types: Option<(&'a [u8], GGMLType)>,
    /// `pos_embd.weight` [n_embd, n_ctx_train] — `bert` only.
    pub pos_embd: Option<(&'a [u8], GGMLType)>,
    pub layers: Vec<BertLayerWeights<'a>>,
}

/// Decode an F32 row at `index` from a plain (non-quantized) tensor.
///
/// `token_types` and `pos_embd` are F32 `[n_embd, n]` tables rather than
/// quantized embedding matrices, so they bypass the `embedding_lookup`
/// dispatch entirely.
pub fn decode_f32_row_public(bytes: &[u8], expected_len: usize) -> Option<Vec<f32>> {
    decode_f32_row(GGMLType::F32, bytes, expected_len)
}

fn decode_f32_row(ggml_type: GGMLType, bytes: &[u8], expected_len: usize) -> Option<Vec<f32>> {
    match ggml_type {
        GGMLType::F32 => {
            if bytes.len() / 4 < expected_len {
                return None;
            }
            Some(
                (0..expected_len)
                    .map(|i| {
                        f32::from_bits(u32::from_le_bytes([
                            bytes[i * 4],
                            bytes[i * 4 + 1],
                            bytes[i * 4 + 2],
                            bytes[i * 4 + 3],
                        ]))
                    })
                    .collect(),
            )
        }
        GGMLType::F16 => {
            if bytes.len() / 2 < expected_len {
                return None;
            }
            Some(
                (0..expected_len)
                    .map(|i| {
                        crate::ops::f16_to_f32(u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]))
                    })
                    .collect(),
            )
        }
        _ => None,
    }
}

fn f32_tensor<S: TensorSource + ?Sized>(source: &S, name: &str, expected_len: usize) -> Vec<f32> {
    let info = source
        .tensor_info(name)
        .unwrap_or_else(|| panic!("missing tensor {name}"));
    let bytes = source
        .tensor_slice(name)
        .unwrap_or_else(|| panic!("missing tensor data for {name}"));
    decode_f32_row(info.ggml_type, bytes, expected_len)
        .unwrap_or_else(|| panic!("tensor {name} is not a decodable f32 row"))
}

fn optional_f32_tensor<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    expected_len: usize,
) -> Option<Vec<f32>> {
    let info = source.tensor_info(name)?;
    let bytes = source.tensor_slice(name)?;
    decode_f32_row(info.ggml_type, bytes, expected_len)
}

fn load_weight<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Weight<'a> {
    let info = source
        .tensor_info(name)
        .unwrap_or_else(|| panic!("missing tensor {name}"));
    let bytes = source
        .tensor_slice(name)
        .unwrap_or_else(|| panic!("missing tensor data for {name}"));
    Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes,
        info.ggml_type,
        n_in,
        n_out,
    ))
}

/// Load a projection, cross-checking both dims against the GGUF inventory.
fn load_sized_weight<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    expect_in: usize,
    expect_out: usize,
) -> Weight<'a> {
    let info = source
        .tensor_info(name)
        .unwrap_or_else(|| panic!("missing tensor {name}"));
    let (n_in, n_out) = (info.dims[0] as usize, info.dims[1] as usize);
    assert_eq!(
        n_in, expect_in,
        "{name}: n_in {n_in} != expected {expect_in}"
    );
    assert_eq!(
        n_out, expect_out,
        "{name}: n_out {n_out} != expected {expect_out}"
    );
    load_weight(source, name, n_in, n_out)
}

#[allow(clippy::too_many_arguments)]
pub fn load_weights<S: TensorSource + ?Sized>(
    source: &S,
    variant: BertVariant,
    n_layer: usize,
    n_embd: usize,
    n_embd_q: usize,
    n_embd_gqa: usize,
    n_embd_head: usize,
    n_ff: usize,
) -> BertWeights<'_> {
    let embd_info = source
        .tensor_info("token_embd.weight")
        .expect("missing token_embd.weight");
    let token_embd_ggml_type = embd_info.ggml_type;
    assert_eq!(
        embd_info.dims[0] as usize, n_embd,
        "token_embd n_in mismatch"
    );
    let token_embd = source
        .tensor_slice("token_embd.weight")
        .expect("missing token_embd.weight data");

    let tok_norm = NormWithBias {
        weight: f32_tensor(source, "token_embd_norm.weight", n_embd),
        bias: f32_tensor(source, "token_embd_norm.bias", n_embd),
    };

    // `token_types` is required for jina-bert-v2, optional otherwise.
    let token_types = match source.tensor_info("token_types.weight") {
        Some(_) => {
            let bytes = source
                .tensor_slice("token_types.weight")
                .expect("missing token_types.weight data");
            Some((
                bytes,
                source.tensor_info("token_types.weight").unwrap().ggml_type,
            ))
        }
        None => None,
    };
    if variant.uses_gelu_gate() {
        assert!(
            token_types.is_some(),
            "jina-bert-v2 requires token_types.weight"
        );
    }

    let pos_embd = source.tensor_info("pos_embd.weight").map(|info| {
        (
            source
                .tensor_slice("pos_embd.weight")
                .expect("missing pos_embd.weight data"),
            info.ggml_type,
        )
    });
    if variant == BertVariant::Bert {
        assert!(pos_embd.is_some(), "bert requires pos_embd.weight");
    }

    let layers = (0..n_layer)
        .map(|l| {
            let name_of = |suffix: &str| format!("blk.{l}.{suffix}");
            let bias_of = |suffix: &str| Bias {
                values: optional_f32_tensor(source, &name_of(suffix), n_embd),
            };
            BertLayerWeights {
                wq: load_sized_weight(source, &name_of("attn_q.weight"), n_embd, n_embd_q),
                wq_bias: Bias {
                    values: optional_f32_tensor(source, &name_of("attn_q.bias"), n_embd_q),
                },
                wk: load_sized_weight(source, &name_of("attn_k.weight"), n_embd, n_embd_gqa),
                wk_bias: Bias {
                    values: optional_f32_tensor(source, &name_of("attn_k.bias"), n_embd_gqa),
                },
                wv: load_sized_weight(source, &name_of("attn_v.weight"), n_embd, n_embd_gqa),
                wv_bias: Bias {
                    values: optional_f32_tensor(source, &name_of("attn_v.bias"), n_embd_gqa),
                },
                wo: load_sized_weight(source, &name_of("attn_output.weight"), n_embd_q, n_embd),
                wo_bias: Bias {
                    values: optional_f32_tensor(source, &name_of("attn_output.bias"), n_embd),
                },
                attn_out_norm: NormWithBias {
                    weight: f32_tensor(source, &name_of("attn_output_norm.weight"), n_embd),
                    bias: f32_tensor(source, &name_of("attn_output_norm.bias"), n_embd),
                },
                layer_out_norm: NormWithBias {
                    weight: f32_tensor(source, &name_of("layer_output_norm.weight"), n_embd),
                    bias: f32_tensor(source, &name_of("layer_output_norm.bias"), n_embd),
                },
                ffn_up: load_sized_weight(source, &name_of("ffn_up.weight"), n_embd, n_ff),
                ffn_gate: match variant.uses_gelu_gate() {
                    true => Some(load_sized_weight(
                        source,
                        &name_of("ffn_gate.weight"),
                        n_embd,
                        n_ff,
                    )),
                    false => None,
                },
                ffn_up_bias: Bias {
                    values: optional_f32_tensor(source, &name_of("ffn_up.bias"), n_ff),
                },
                ffn_gate_bias: Bias {
                    values: optional_f32_tensor(source, &name_of("ffn_gate.bias"), n_ff),
                },
                ffn_down: load_sized_weight(source, &name_of("ffn_down.weight"), n_ff, n_embd),
                ffn_down_bias: Bias {
                    values: optional_f32_tensor(source, &name_of("ffn_down.bias"), n_embd),
                },
            }
        })
        .collect();

    BertWeights {
        variant,
        n_layer,
        token_embd,
        token_embd_ggml_type,
        tok_norm,
        token_types,
        pos_embd,
        layers,
    }
}
