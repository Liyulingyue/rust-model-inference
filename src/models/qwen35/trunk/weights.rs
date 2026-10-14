//! GGUF → shared hybrid trunk weight loading.
//!
//! `Qwen35Model::from_source` accepts Qwen3.5 only. `Edge0Model` loads the
//! same attention/SSM trunk and keeps its MoE weights in `models::edge0`.
//! The shared loader reads a GGUF `TensorSource` and builds
//! `Qwen35LayerWeights` rows, one per layer. Recurrent (Mamba) layers only
//! fill the SSM field group; dense (attention) layers only fill the
//! attention field group — see `config.is_recurrent`.
//!
//! `load_weight` and `load_weight_f32` are the per-tensor helpers used
//! by `from_source`. They are `pub(crate)` because they are only useful
//! inside this module's loading path.

use super::config::Qwen35Config;
use super::util::f16_at;
use crate::core::tensor::GGMLType;
use crate::core::tensor::TensorSource;
use crate::models::edge0::weights::load_affine;
use crate::ops::kernel::{QuantizedTensor, Weight};
#[cfg(feature = "vulkan")]
use crate::vulkan::qwen35::Qwen35VulkanSession;

// =============================================================================
// Model + Layer-weight structs
// =============================================================================

/// All weights for a single Qwen3.5 layer.
///
/// The `Option` fields distinguish dense-attention layers (which fill
/// `wq`/`wk`/`wv`/`wo`/`attn_q_norm`/`attn_k_norm`) from recurrent (Mamba
/// SSM) layers (which fill `wqkv`/`wqkv_gate`/`ssm_*`). `config.is_recurrent`
/// selects which group is active.
pub struct Qwen35LayerWeights<'a> {
    pub attn_norm: Vec<f32>,
    pub attn_post_norm: Vec<f32>,
    pub wq: Option<Weight<'a>>,
    pub wk: Option<Weight<'a>>,
    pub wv: Option<Weight<'a>>,
    pub wo: Option<Weight<'a>>,
    pub attn_q_norm: Option<Vec<f32>>,
    pub attn_k_norm: Option<Vec<f32>>,
    pub wqkv: Option<Weight<'a>>,
    pub wqkv_gate: Option<Weight<'a>>,
    pub ssm_conv1d: Option<Vec<f32>>,
    pub ssm_dt: Option<Vec<f32>>,
    pub ssm_a: Option<Vec<f32>>,
    pub ssm_beta: Option<Weight<'a>>,
    pub ssm_alpha: Option<Weight<'a>>,
    pub ssm_norm: Option<Vec<f32>>,
    pub ssm_out: Option<Weight<'a>>,
    /// Dense SwiGLU FFN. Absent on MoE-only towers, where every layer routes
    /// through experts instead and the weights are unused.
    pub ffn_gate: Option<Weight<'a>>,
    pub ffn_up: Option<Weight<'a>>,
    pub ffn_down: Option<Weight<'a>>,
}

/// Shared Qwen3.5/Edge0 attention and SSM weights + parsed config.
///
/// Qwen3.5 exposes this as `Qwen35Model`; Edge0 composes it with MoE weights.
pub struct HybridTrunk<'a> {
    pub config: Qwen35Config,
    pub tok_embd: Weight<'a>,
    pub output_norm: Vec<f32>,
    pub output_weight: Weight<'a>,
    pub layers: Vec<Qwen35LayerWeights<'a>>,
    /// Optional classification head (cross-encoder / NLI), loaded when the
    /// GGUF carries `cls.output.weight` (and optionally `cls.output.bias`).
    /// The shape is `[n_embd, num_labels]` so the same `Weight` matmul used
    /// by `qwen3::trunk::weights::score_logits` works here; the bias, when
    /// present, is a separate `Vec<f32>` of length `num_labels`.
    pub cls_score: Option<Weight<'a>>,
    pub cls_score_bias: Vec<f32>,
    /// Lazily-initialized Vulkan session. Built on the first decode token
    /// after `--gpu` enables the global Vulkan context. `None` means CPU
    /// fallback (no eligible GPU or Vulkan init failed).
    #[cfg(feature = "vulkan")]
    pub(crate) gpu: Option<Qwen35VulkanSession>,
}

/// Qwen3.5 uses the shared hybrid trunk without architecture-specific weights.
pub type Qwen35Model<'a> = HybridTrunk<'a>;

// Convenience alias so that `impl Qwen35Model { fn from_source(...) }` in
// `weights.rs` and `impl Qwen35Model { fn forward(...) }` in `forward.rs`
// can refer to a common TensorSource without redundant imports.
pub(crate) type Source<'a> = &'a dyn TensorSource;

/// Load a quantized weight (F32/F16/Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K) into a
/// `Weight` borrowing the GGUF bytes. Returns `None` if the tensor is
/// missing or the dtype is not supported (with a stderr warning).
pub(crate) fn load_weight<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
) -> Option<Weight<'a>> {
    let ti = source.tensor_info(name)?;
    let data = source.tensor_slice(name)?;
    let n_cols = ti.dims[0] as usize;
    let n_rows = if ti.dims.len() >= 2 {
        ti.dims[1] as usize
    } else {
        1
    };

    match ti.ggml_type {
        GGMLType::I32 => match load_affine(source, name, None) {
            Ok(weight) => Some(weight),
            Err(error) => {
                eprintln!("WARNING: {error}");
                None
            }
        },
        GGMLType::F32
        | GGMLType::F16
        | GGMLType::BF16
        | GGMLType::Q8_0
        | GGMLType::Q4_0
        | GGMLType::Q4_1
        | GGMLType::Q4K
        | GGMLType::Q5K
        | GGMLType::Q6K => {
            let mut weight = Weight::from_quantized(QuantizedTensor::from_bytes(
                data,
                ti.ggml_type,
                n_cols,
                n_rows,
            ));
            weight.n_in = n_cols;
            weight.n_out = n_rows;
            if ti.ggml_type == GGMLType::BF16 {
                weight.kernel =
                    Box::new(crate::ops::kernel::bf16::BF16Kernel::with_bf16_input(data));
            }
            Some(weight)
        }
        _ => {
            eprintln!(
                "WARNING: unsupported quant type {:?} for tensor {}",
                ti.ggml_type, name
            );
            None
        }
    }
}

/// Load an F32-or-F16 norm/bias tensor into an owned `Vec<f32>`. Used for
/// non-matmul tensors (norms, biases, conv1d, dt.bias, A-log).
pub(crate) fn load_weight_f32<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
) -> Option<Vec<f32>> {
    let ti = source.tensor_info(name)?;
    let data = source.tensor_slice(name)?;
    let n_el = ti.n_elements();
    match ti.ggml_type {
        GGMLType::F32 => {
            let mut out = Vec::with_capacity(n_el);
            for i in 0..n_el {
                let off = i * 4;
                if off + 4 <= data.len() {
                    out.push(f32::from_le_bytes([
                        data[off],
                        data[off + 1],
                        data[off + 2],
                        data[off + 3],
                    ]));
                } else {
                    out.push(0.0);
                }
            }
            Some(out)
        }
        GGMLType::F16 => {
            let mut out = Vec::with_capacity(n_el);
            for i in 0..n_el {
                out.push(f16_at(data, i));
            }
            Some(out)
        }
        GGMLType::BF16 => {
            let mut out = Vec::with_capacity(n_el);
            for i in 0..n_el {
                let off = i * 2;
                if off + 2 <= data.len() {
                    out.push(crate::ops::bf16_to_f32(u16::from_le_bytes([
                        data[off],
                        data[off + 1],
                    ])));
                } else {
                    out.push(0.0);
                }
            }
            Some(out)
        }
        _ => None,
    }
}

impl<'a> HybridTrunk<'a> {
    pub fn from_source(source: &'a dyn TensorSource) -> Result<Self, String> {
        let arch = source
            .metadata("general.architecture")
            .and_then(|value| value.to_string_val());
        if arch == Some("edge0") {
            return Err("Edge0 architecture requires Edge0Model::from_source".into());
        }
        // qwen35moe shares every attention and SSM tensor with qwen35; only the
        // FFN differs, and OccamyModel supplies the MoE half.
        if arch != Some("qwen35") && arch != Some("qwen35moe") {
            return Err(format!(
                "Qwen35Model requires general.architecture=qwen35, got {arch:?}"
            ));
        }
        let config = Qwen35Config::from_source_for_arch(source, "qwen35moe")?;
        Self::load_trunk(source, config, true)
    }

    pub(crate) fn load_trunk(
        source: &'a dyn TensorSource,
        config: Qwen35Config,
        edge0: bool,
    ) -> Result<Self, String> {
        let token_info = source
            .tensor_info("token_embd.weight")
            .ok_or("Missing token_embd.weight")?;
        let actual = token_info
            .dims
            .iter()
            .map(|value| *value as usize)
            .collect::<Vec<_>>();
        // The lossless export stores packed 4-bit codes, so the embedding is
        // one eighth as wide; the expanded modes store whole values instead.
        let expanded = crate::models::edge0::weights::is_expanded(source);
        let expected = vec![
            if edge0 && !expanded {
                config.n_embd / 8
            } else {
                config.n_embd
            },
            config.vocab_size,
        ];
        if actual != expected {
            return Err(format!(
                "token_embd.weight shape mismatch: expected {expected:?}, got {actual:?}, dtype={:?}",
                token_info.ggml_type
            ));
        }
        let tok_embd = load_weight(source, "token_embd.weight").ok_or_else(|| {
            format!(
                "Unsupported token_embd.weight dtype: {:?}",
                token_info.ggml_type
            )
        })?;

        let output_norm =
            load_weight_f32(source, "output_norm.weight").ok_or("Missing output_norm.weight")?;

        let output_weight = {
            let name = if source.tensor_info("output.weight").is_some() {
                "output.weight"
            } else {
                "token_embd.weight"
            };
            load_weight(source, name).ok_or("Missing output weight")?
        };
        if edge0
            && !expanded
            && (
                tok_embd.n_in,
                tok_embd.n_out,
                output_weight.n_in,
                output_weight.n_out,
            ) != (
                config.n_embd,
                config.vocab_size,
                config.n_embd,
                config.vocab_size,
            )
        {
            return Err("Edge0 embedding or output shape mismatch".into());
        }
        // Every mode, packed or expanded, must present the same logical matrix,
        // so the expanded path checks the widths the kernel will actually use.
        if edge0
            && expanded
            && (
                tok_embd.n_in,
                tok_embd.n_out,
                output_weight.n_in,
                output_weight.n_out,
            ) != (
                config.n_embd,
                config.vocab_size,
                config.n_embd,
                config.vocab_size,
            )
        {
            return Err("Edge0 expanded embedding or output shape mismatch".into());
        }

        let n_layers_impl = config.n_layer_impl();
        let mut layers = Vec::with_capacity(n_layers_impl);
        for i in 0..n_layers_impl {
            let attn_norm = load_weight_f32(source, &format!("blk.{}.attn_norm.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.attn_norm.weight", i))?;
            let attn_post_norm =
                load_weight_f32(source, &format!("blk.{}.post_attention_norm.weight", i))
                    .ok_or_else(|| format!("Missing blk.{}.post_attention_norm.weight", i))?;
            let is_recr = config.is_recurrent[i];

            let (wq, wk, wv, wo, attn_q_norm, attn_k_norm) = if !is_recr {
                (
                    load_weight(source, &format!("blk.{}.attn_q.weight", i)),
                    load_weight(source, &format!("blk.{}.attn_k.weight", i)),
                    load_weight(source, &format!("blk.{}.attn_v.weight", i)),
                    load_weight(source, &format!("blk.{}.attn_output.weight", i)),
                    load_weight_f32(source, &format!("blk.{}.attn_q_norm.weight", i)),
                    load_weight_f32(source, &format!("blk.{}.attn_k_norm.weight", i)),
                )
            } else {
                (None, None, None, None, None, None)
            };

            let (
                wqkv,
                wqkv_gate,
                ssm_conv1d,
                ssm_dt,
                ssm_a,
                ssm_beta,
                ssm_alpha,
                ssm_norm,
                ssm_out,
            ) = if is_recr {
                (
                    load_weight(source, &format!("blk.{}.attn_qkv.weight", i)),
                    load_weight(source, &format!("blk.{}.attn_gate.weight", i)),
                    load_weight_f32(source, &format!("blk.{}.ssm_conv1d.weight", i)),
                    load_weight_f32(source, &format!("blk.{}.ssm_dt.bias", i)),
                    load_weight_f32(source, &format!("blk.{}.ssm_a", i)),
                    load_weight(source, &format!("blk.{}.ssm_beta.weight", i)),
                    load_weight(source, &format!("blk.{}.ssm_alpha.weight", i)),
                    load_weight_f32(source, &format!("blk.{}.ssm_norm.weight", i)),
                    load_weight(source, &format!("blk.{}.ssm_out.weight", i)),
                )
            } else {
                (None, None, None, None, None, None, None, None, None)
            };
            let mut ssm_a = ssm_a;
            if edge0 {
                if let Some(values) = &mut ssm_a {
                    for value in values {
                        *value = -value.exp();
                    }
                }
            }

            let ffn_gate = load_weight(source, &format!("blk.{}.ffn_gate.weight", i));
            let ffn_up = load_weight(source, &format!("blk.{}.ffn_up.weight", i));
            let ffn_down = load_weight(source, &format!("blk.{}.ffn_down.weight", i));
            if ffn_gate.is_none() && ffn_up.is_none() && ffn_down.is_none() {
                // MoE-only tower; the expert weights live in a separate loader.
            } else if ffn_gate.is_none() || ffn_up.is_none() || ffn_down.is_none() {
                return Err(format!(
                    "blk.{i} has a partial dense FFN; all three of ffn_gate/ffn_up/ffn_down are required"
                ));
            }
            layers.push(Qwen35LayerWeights {
                attn_norm,
                attn_post_norm,
                wq,
                wk,
                wv,
                wo,
                attn_q_norm,
                attn_k_norm,
                wqkv,
                wqkv_gate,
                ssm_conv1d,
                ssm_dt,
                ssm_a,
                ssm_beta,
                ssm_alpha,
                ssm_norm,
                ssm_out,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        Ok(Self {
            config,
            tok_embd,
            output_norm,
            output_weight,
            layers,
            cls_score: load_weight(source, "cls.output.weight"),
            cls_score_bias: load_weight_f32(source, "cls.output.bias").unwrap_or_default(),
            #[cfg(feature = "vulkan")]
            gpu: None,
        })
    }

    pub fn embed_tokens(&self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        if let Some(&token_id) = token_ids
            .iter()
            .find(|&&token_id| token_id as usize >= self.tok_embd.n_out)
        {
            return Err(format!(
                "Qwen3.5 token id {token_id} out of range (vocab={})",
                self.tok_embd.n_out
            ));
        }
        let len = token_ids
            .len()
            .checked_mul(self.config.n_embd)
            .ok_or("Qwen3.5 token embedding length overflow")?;
        let mut embeddings = vec![0.0; len];
        for (row, &token_id) in embeddings
            .chunks_exact_mut(self.config.n_embd)
            .zip(token_ids)
        {
            self.tok_embd.embedding_lookup(token_id, row);
        }
        Ok(embeddings)
    }

    /// True when the GGUF carried a `cls.output.weight` head. Mirrors
    /// `Qwen3Model::is_rerank` so callers (server / rerank CLI) can probe
    /// the model the same way for both backbones.
    pub fn is_classifier(&self) -> bool {
        self.cls_score.is_some()
    }

    /// Project `last_hidden` (length `n_embd`) through the classification
    /// head, returning one logit per class plus the optional bias. Returns
    /// an error if the head wasn't loaded.
    ///
    /// Same contract as `Qwen3Model::score_logits` but with a `cls_score_bias`
    /// added on top of the matmul output. The HF `score.weight` is stored
    /// as `[num_labels, n_embd]` in safetensors; the converter transposes
    /// it to `[n_embd, num_labels]` for the llama.cpp rerank-packer layout,
    /// and we add a zero bias when the HF head had no bias (`openjev` does
    /// not have one; `jina-bert-v2` does).
    pub fn score_logits(&self, last_hidden: &[f32]) -> Result<Vec<f32>, String> {
        let weight = self
            .cls_score
            .as_ref()
            .ok_or_else(|| "Qwen3.5 model has no cls.output.weight head".to_string())?;
        if last_hidden.len() != self.config.n_embd {
            return Err(format!(
                "score_logits: hidden size {} does not match n_embd {}",
                last_hidden.len(),
                self.config.n_embd
            ));
        }
        let n_in = self.config.n_embd;
        let n_cls = weight.n_out;
        let mut out = vec![0.0f32; n_cls];
        weight
            .kernel
            .forward_prepared(last_hidden, &[], &[], None, &mut out, n_in, n_cls, 0, 1);
        if !self.cls_score_bias.is_empty() {
            if self.cls_score_bias.len() != n_cls {
                return Err(format!(
                    "cls.output.bias length {} does not match num_labels {}",
                    self.cls_score_bias.len(),
                    n_cls
                ));
            }
            for (slot, bias) in out.iter_mut().zip(self.cls_score_bias.iter()) {
                *slot += *bias;
            }
        }
        Ok(out)
    }
}
