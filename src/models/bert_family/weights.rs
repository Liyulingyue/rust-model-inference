//! Weight loading for the BERT encoder family.
//!
//! Tensor names are the ones `bert.cpp` requests via `LLM_TENSOR_*`
//! (`references/llama.cpp/src/models/bert.cpp:22-62`), which in GGUF land is
//! the `blk.{l}.attn_*` / `blk.{l}.ffn_*` / `blk.{l}.*_norm` vocabulary.

use crate::core::tensor::{GGMLType, MetaValue, TensorSource};
use crate::ops::kernel::{QuantizedTensor, Weight};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BertVariant {
    /// `arch = "bert"` — absolute `position_embd`, GELU FFN over a single `ffn_up`.
    Bert,
    /// `arch = "jina-bert-v2"` — ALiBi, required `token_types`, GEGLU FFN.
    JinaBertV2,
    /// `arch = "nomic-bert"` — RoPE, fused QKV, SwiGLU FFN, no projection bias.
    NomicBert,
    /// `arch = "nomic-bert-moe"` — fused QKV, RoPE; FFN alternates between
    /// GELU MoE (`expert_count=8`, top-2 softmax) and dense GELU per
    /// `moe_every_n_layers` (`nomic-bert-moe.cpp:36-44`,
    /// `bert.cpp:165-178`).
    NomicBertMoe,
}

impl std::fmt::Display for BertVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.arch_name())
    }
}

impl BertVariant {
    pub fn arch_name(self) -> &'static str {
        match self {
            BertVariant::Bert => "bert",
            BertVariant::JinaBertV2 => "jina-bert-v2",
            BertVariant::NomicBert => "nomic-bert",
            BertVariant::NomicBertMoe => "nomic-bert-moe",
        }
    }

    /// ALiBi is applied for this variant, with bias
    /// [`MAX_ALIBI_BIAS_JINA_V2`]. See that constant for why the GGUF key is
    /// irrelevant: `jina-bert-v2.cpp:5` sets `f_max_alibi_bias = 8.0f`
    /// unconditionally, and `llama-model.cpp:1483` derives `use_alibi` from it.
    pub fn uses_alibi(self) -> bool {
        matches!(self, BertVariant::JinaBertV2)
    }

    /// GEGLU (`ffn_gate` + `ffn_up`) vs plain GELU (`ffn_up` only).
    pub fn uses_gelu_gate(self) -> bool {
        matches!(self, BertVariant::JinaBertV2)
    }

    /// SwiGLU (`ffn_gate` + `ffn_up`, `silu`), `nomic-bert.cpp:196-203` — the
    /// `bert.cpp` fall-through branch, which NOMIC_BERT reaches because it is
    /// listed in neither the `BERT || NOMIC_BERT_MOE || JINA_BERT_V3` GELU arm
    /// (`bert.cpp:179`) nor the `JINA_BERT_V2` GEGLU arm (`bert.cpp:187`).
    pub fn uses_silu_gate(self) -> bool {
        matches!(self, BertVariant::NomicBert)
    }

    /// RoPE applied to Q and K. `bert.cpp:120-133` ropes only NOMIC_BERT /
    /// NOMIC_BERT_MOE / JINA_BERT_V3; BERT and jina-bert-v2 fall through.
    pub fn uses_rope(self) -> bool {
        matches!(self, BertVariant::NomicBert | BertVariant::NomicBertMoe)
    }

    /// Whether the variant has any MoE layers at all (currently only
    /// `nomic-bert-moe`).
    pub fn uses_moe(self) -> bool {
        matches!(self, BertVariant::NomicBertMoe)
    }

    pub fn from_arch(arch: &str) -> Option<Self> {
        match arch {
            "bert" => Some(BertVariant::Bert),
            "jina-bert-v2" => Some(BertVariant::JinaBertV2),
            "nomic-bert" => Some(BertVariant::NomicBert),
            "nomic-bert-moe" => Some(BertVariant::NomicBertMoe),
            _ => None,
        }
    }
}

/// ALiBi `f_max_alibi_bias` for jina-bert-v2, hardcoded exactly as
/// `references/llama.cpp/src/models/jina-bert-v2.cpp:5` does it:
///
/// ```c++
/// hparams.f_max_alibi_bias = 8.0f;   // unconditional, not read from the GGUF
/// ```
///
/// **The GGUF key being absent does not disable ALiBi here.**
/// `LLM_KV_ATTENTION_MAX_ALIBI_BIAS` is never read in `llama-model.cpp`'s
/// `load_hparams`; only a few other arches (`mpt.cpp:6`, `jais.cpp:5`) consult
/// it with a "do not overwrite the default" flag. Both jina GGUFs available
/// locally omit the key, yet llama.cpp still applies an 8.0 ALiBi because the
/// arch-level hparams loader sets it unconditionally, and
/// `llama-model.cpp:1483` then derives `use_alibi = (f_max_alibi_bias > 0.0f)`.
///
/// Do not "fix" this into a GGUF-derived value defaulting to 0.0: that would
/// silently turn ALiBi off for every jina-bert-v2 model.
pub const MAX_ALIBI_BIAS_JINA_V2: f32 = 8.0;

/// Bundle for a 3D expert tensor of shape `[rows, cols, n_expert]` (per
/// GGUF dim order). The bytes are laid out as `n_expert` concatenated
/// `[rows, cols]` quantized matrices, each stored in the same Q8_0
/// block-stride as a stand-alone tensor of the same shape.
#[derive(Clone)]
pub struct MoeExperts<'a> {
    pub bytes: &'a [u8],
    pub ggml_type: GGMLType,
    pub rows: usize,
    pub cols: usize,
    pub n_expert: usize,
}

impl<'a> MoeExperts<'a> {
    pub fn per_expert_bytes(&self, expert: usize) -> &'a [u8] {
        // Q8_0 block = 2-byte f16 scale + 32 int8 quants per 32 elements.
        // Q4K = 144, Q6K = 210; if you need other quantizations here,
        // add the matching block size.
        let block_size = match self.ggml_type {
            GGMLType::Q8_0 => 34,
            GGMLType::Q4K => 144,
            GGMLType::Q6K => 210,
            _ => panic!("unsupported MoE expert ggml type"),
        };
        let blocks_per_expert = (self.rows * self.cols) / 32;
        let stride = blocks_per_expert * block_size;
        let start = expert * stride;
        let end = start + stride;
        &self.bytes[start..end]
    }
}

fn load_moe_experts<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    rows: usize,
    cols: usize,
    n_expert: usize,
) -> Option<MoeExperts<'a>> {
    let info = source.tensor_info(name)?;
    let bytes = source.tensor_slice(name)?;
    if n_expert == 0 || bytes.is_empty() {
        return None;
    }
    Some(MoeExperts {
        bytes,
        ggml_type: info.ggml_type,
        rows,
        cols,
        n_expert,
    })
}

pub struct BertLayerWeights<'a> {
    pub wq: Option<Weight<'a>>,
    pub wq_bias: Bias,
    pub wk: Option<Weight<'a>>,
    pub wk_bias: Bias,
    pub wv: Option<Weight<'a>>,
    pub wv_bias: Bias,
    /// Fused `attn_qkv.weight` [n_embd, n_embd_q + n_embd_kv + n_embd_kv], the
    /// packing nomic-bert ships. Rows are `[Q | K | V]`, the llama.cpp fused
    /// QKV order. `None` when the GGUF splits q/k/v.
    pub wqkv: Option<Weight<'a>>,
    pub wo: Weight<'a>,
    pub wo_bias: Bias,
    pub attn_out_norm: NormWithBias,
    pub layer_out_norm: NormWithBias,
    /// `true` if this layer is an MoE layer (only `nomic-bert-moe` ever
    /// sets this). When true, `ffn_gate_inp`/`ffn_up_exps`/`ffn_down_exps`
    /// carry the expert weights and `ffn_up`/`ffn_down` are absent.
    pub is_moe_layer: bool,
    /// Dense GELU branch (BERT / jina / nomic-bert / nomic-bert-moe
    /// non-MoE layers). Required when `is_moe_layer == false`.
    pub ffn_up: Option<Weight<'a>>,
    pub ffn_down: Option<Weight<'a>>,
    pub ffn_up_bias: Bias,
    pub ffn_down_bias: Bias,
    /// Gated-FFN projection. Required for `jina-bert-v2` (GEGLU) and
    /// `nomic-bert` (SwiGLU); absent for plain `bert` and MoE layers.
    pub ffn_gate: Option<Weight<'a>>,
    pub ffn_gate_bias: Bias,
    /// Router logits projection [n_embd, n_expert], F32. Only present on
    /// MoE layers.
    pub ffn_gate_inp: Option<Weight<'a>>,
    /// Per-expert up projection (3D Q8_0). Stored as a `MoeExperts`
    /// because each expert slice must be presented as a 2D `[rows, cols]`
    /// matrix to the Q8_0 matmul kernel.
    pub ffn_up_exps: Option<MoeExperts<'a>>,
    /// Per-expert down projection (3D Q8_0).
    pub ffn_down_exps: Option<MoeExperts<'a>>,
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
    /// `position_embd.weight` [n_embd, n_ctx_train] — `bert` only.
    /// The GGUF spells it `position_embd`, not `pos_embd`
    /// (`llama-arch.cpp:480`); the shorter form does not exist in any GGUF.
    pub pos_embd: Option<(&'a [u8], GGMLType)>,
    /// `moe_every_n_layers` (`LLM_KV_MOE_EVERY_N_LAYERS`, default 0 = no
    /// MoE). Layer `l` is an MoE layer when `moe_every_n_layers > 0` and
    /// `l % moe_every_n_layers == 1` (`bert.cpp:165-178`).
    pub moe_every_n_layers: usize,
    /// Number of experts per MoE layer (`LLM_KV_EXPERT_COUNT`).
    pub expert_count: usize,
    /// Number of experts used per token (`LLM_KV_EXPERT_USED_COUNT`,
    /// top-k for the router).
    pub expert_used_count: usize,
    /// `expert_weights_scale` (`LLM_KV_EXPERT_WEIGHTS_SCALE`); default 1.0.
    pub expert_weights_scale: f32,
    pub layers: Vec<BertLayerWeights<'a>>,
}

/// Decode an F32 row at `index` from a plain (non-quantized) tensor.
///
/// `token_types` and `position_embd` are F32 `[n_embd, n]` tables rather than
/// quantized embedding matrices, so they bypass the `embedding_lookup`
/// dispatch entirely.
pub fn decode_f32_row_public(bytes: &[u8], expected_len: usize) -> Option<Vec<f32>> {
    decode_f32_row(GGMLType::F32, bytes, expected_len)
}

/// Decode F32 row `row` from a plain (non-quantized) `[width, rows]` table.
///
/// `token_types` and `position_embd` are F32 tables indexed per element
/// position (`bert.cpp:83-89` reads `token_types` row 0 and `position_embd` row
/// `pos`), so both go through here. `token_types` passes row 0.
pub fn decode_f32_row_at_public(bytes: &[u8], row: usize, expected_len: usize) -> Option<Vec<f32>> {
    decode_f32_row_at(GGMLType::F32, bytes, row, expected_len)
}

fn decode_f32_row(ggml_type: GGMLType, bytes: &[u8], expected_len: usize) -> Option<Vec<f32>> {
    decode_f32_row_at(ggml_type, bytes, 0, expected_len)
}

fn decode_f32_row_at(
    ggml_type: GGMLType,
    bytes: &[u8],
    row: usize,
    expected_len: usize,
) -> Option<Vec<f32>> {
    let stride: usize = match ggml_type {
        GGMLType::F32 => 4,
        GGMLType::F16 => 2,
        _ => return None,
    };
    let start = row.checked_mul(stride.checked_mul(expected_len)?)?;
    let bytes = bytes.get(start..start + stride * expected_len)?;
    Some(
        (0..expected_len)
            .map(|i| match ggml_type {
                GGMLType::F32 => f32::from_bits(u32::from_le_bytes([
                    bytes[i * 4],
                    bytes[i * 4 + 1],
                    bytes[i * 4 + 2],
                    bytes[i * 4 + 3],
                ])),
                _ => crate::ops::f16_to_f32(u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]])),
            })
            .collect(),
    )
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
    // MoE hyperparameters — read unconditionally so per-layer loading
    // can dispatch even when the variant isn't MoE (the fields default
    // to zero/empty and the dispatch `l % moe_every_n_layers == 1` is a
    // no-op when moe_every_n_layers == 0).
    let arch = variant.arch_name();
    let moe_every_n_layers = source
        .metadata(format!("{arch}.moe_every_n_layers").as_str())
        .and_then(MetaValue::to_u64)
        .map(|v| v as usize)
        .unwrap_or(0);
    let expert_count = source
        .metadata(format!("{arch}.expert_count").as_str())
        .and_then(MetaValue::to_u64)
        .map(|v| v as usize)
        .unwrap_or(0);
    let expert_used_count = source
        .metadata(format!("{arch}.expert_used_count").as_str())
        .and_then(MetaValue::to_u64)
        .map(|v| v as usize)
        .unwrap_or(0);
    let expert_weights_scale = source
        .metadata(format!("{arch}.expert_weights_scale").as_str())
        .and_then(|v| v.to_f64())
        .map(|v| v as f32)
        .unwrap_or(1.0);

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

    // `llama-arch.cpp:480` spells this `position_embd`, and it is a
    // `LLM_TENSOR_LAYER_INPUT` tensor, so it carries no `blk.{l}.` prefix.
    // `bert.cpp:32` creates it with flag 0, i.e. required for `bert` only.
    let pos_embd = source.tensor_info("position_embd.weight").map(|info| {
        (
            source
                .tensor_slice("position_embd.weight")
                .expect("missing position_embd.weight data"),
            info.ggml_type,
        )
    });
    if variant == BertVariant::Bert {
        assert!(
            pos_embd.is_some(),
            "bert requires position_embd.weight (llama-arch.cpp:480, bert.cpp:32)"
        );
    }

    // Per-layer first pass: attention tensors + LayerNorms. FFN tensors
    // are filled in a second pass below because the field layout differs
    // between MoE and dense.
    let mut layers: Vec<BertLayerWeights> = (0..n_layer)
        .map(|l| {
            let name_of = |suffix: &str| format!("blk.{l}.{suffix}");
            // nomic-bert ships one fused [n_embd, n_embd_q + 2*n_embd_gqa]
            // projection (`create_tensor_qkv` accepts a fused tensor first).
            // The others split q/k/v.
            let wqkv = source.tensor_info(&name_of("attn_qkv.weight")).map(|_| {
                load_sized_weight(
                    source,
                    &name_of("attn_qkv.weight"),
                    n_embd,
                    n_embd_q + 2 * n_embd_gqa,
                )
            });
            let wq = match wqkv {
                Some(_) => None,
                None => Some(load_sized_weight(
                    source,
                    &name_of("attn_q.weight"),
                    n_embd,
                    n_embd_q,
                )),
            };
            let wk = match wqkv {
                Some(_) => None,
                None => Some(load_sized_weight(
                    source,
                    &name_of("attn_k.weight"),
                    n_embd,
                    n_embd_gqa,
                )),
            };
            let wv = match wqkv {
                Some(_) => None,
                None => Some(load_sized_weight(
                    source,
                    &name_of("attn_v.weight"),
                    n_embd,
                    n_embd_gqa,
                )),
            };
            let is_moe =
                variant.uses_moe() && moe_every_n_layers > 0 && (l % moe_every_n_layers == 1);
            BertLayerWeights {
                wq,
                wq_bias: Bias {
                    values: optional_f32_tensor(source, &name_of("attn_q.bias"), n_embd_q),
                },
                wk,
                wk_bias: Bias {
                    values: optional_f32_tensor(source, &name_of("attn_k.bias"), n_embd_gqa),
                },
                wv,
                wv_bias: Bias {
                    values: optional_f32_tensor(source, &name_of("attn_v.bias"), n_embd_gqa),
                },
                wqkv,
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
                is_moe_layer: is_moe,
                ffn_up: None,
                ffn_down: None,
                ffn_up_bias: Bias::default(),
                ffn_down_bias: Bias::default(),
                ffn_gate: None,
                ffn_gate_bias: Bias::default(),
                ffn_gate_inp: None,
                ffn_up_exps: None,
                ffn_down_exps: None,
            }
        })
        .collect();

    // ---- second pass: FFN tensors ----
    for l in 0..n_layer {
        let name_of = |suffix: &str| format!("blk.{l}.{suffix}");
        if layers[l].is_moe_layer {
            // MoE branch. `ffn_gate_inp` is a plain F32 router matrix
            // `[n_embd, n_expert]`; `ffn_up_exps` and `ffn_down_exps` are
            // 3D `[n_ff, n_embd, n_expert]` / `[n_embd, n_ff, n_expert]`
            // Q8_0 tensors.
            let gate_info = source
                .tensor_info(&name_of("ffn_gate_inp.weight"))
                .map(|info| {
                    (
                        source
                            .tensor_slice(&name_of("ffn_gate_inp.weight"))
                            .expect("missing ffn_gate_inp.weight data"),
                        info.ggml_type,
                    )
                });
            let ffn_gate_inp = gate_info.as_ref().map(|(bytes, ty)| {
                Weight::from_quantized(QuantizedTensor::from_bytes(
                    bytes,
                    *ty,
                    n_embd,
                    expert_count,
                ))
            });
            // GGUF stores both experts as `[ne0 × ne1]` row-major, where
            // `ne0` (innermost) is the column count. For up: ne0 = n_ff;
            // for down: ne0 = n_embd. We pass them through `load_moe_experts`
            // which stores the raw bytes plus row/col metadata; the
            // forward path slices a 2D window per expert for matmul.
            let ffn_up_exps = load_moe_experts(
                source,
                &name_of("ffn_up_exps.weight"),
                n_ff,
                n_embd,
                expert_count,
            );
            let ffn_down_exps = load_moe_experts(
                source,
                &name_of("ffn_down_exps.weight"),
                n_embd,
                n_ff,
                expert_count,
            );
            assert!(ffn_gate_inp.is_some(), "moe layer missing ffn_gate_inp");
            assert!(ffn_up_exps.is_some(), "moe layer missing ffn_up_exps");
            assert!(ffn_down_exps.is_some(), "moe layer missing ffn_down_exps");
            layers[l].ffn_gate_inp = ffn_gate_inp;
            layers[l].ffn_up_exps = ffn_up_exps;
            layers[l].ffn_down_exps = ffn_down_exps;
        } else {
            // Dense FFN branch — same as `nomic-bert-moe.cpp:36-39` for
            // nomic-bert-moe, or `bert.cpp:179-186` for the others.
            layers[l].ffn_up = Some(load_sized_weight(
                source,
                &name_of("ffn_up.weight"),
                n_embd,
                n_ff,
            ));
            layers[l].ffn_down = Some(load_sized_weight(
                source,
                &name_of("ffn_down.weight"),
                n_ff,
                n_embd,
            ));
            layers[l].ffn_up_bias = Bias {
                values: optional_f32_tensor(source, &name_of("ffn_up.bias"), n_ff),
            };
            layers[l].ffn_down_bias = Bias {
                values: optional_f32_tensor(source, &name_of("ffn_down.bias"), n_embd),
            };
            // jina-bert-v2 has GEGLU on every layer (`bert.cpp:187-194`).
            if variant.uses_gelu_gate() {
                layers[l].ffn_gate = Some(load_sized_weight(
                    source,
                    &name_of("ffn_gate.weight"),
                    n_embd,
                    n_ff,
                ));
                layers[l].ffn_gate_bias = Bias {
                    values: optional_f32_tensor(source, &name_of("ffn_gate.bias"), n_ff),
                };
            }
            // nomic-bert (v1.5) has SwiGLU on every layer (`nomic-bert.cpp:41`).
            if variant.uses_silu_gate() {
                layers[l].ffn_gate = Some(load_sized_weight(
                    source,
                    &name_of("ffn_gate.weight"),
                    n_embd,
                    n_ff,
                ));
            }
        }
    }

    BertWeights {
        variant,
        n_layer,
        token_embd,
        token_embd_ggml_type,
        tok_norm,
        token_types,
        pos_embd,
        moe_every_n_layers,
        expert_count,
        expert_used_count,
        expert_weights_scale,
        layers,
    }
}
