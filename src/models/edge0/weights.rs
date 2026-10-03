//! Edge0 model weights and architecture-specific validation.

use crate::core::tensor::{GGMLType, TensorInfo, TensorSource};
use crate::models::qwen35::trunk::{HybridTrunk, Qwen35Config};
use crate::ops::kernel::mlx_affine::MlxAffineKernel;
use crate::ops::kernel::{QuantizedTensor, Weight};

/// Bytes one `n_in x n_out` matrix occupies for a GGML type.
///
/// `GGMLType::nbytes` already knows each type's block geometry, including the
/// 256-value super-blocks of the k-quants, so this is deliberately not a
/// hand-rolled per-type table: the Q4_K_M export writes Q4_K, Q6_K, Q8_0, F16
/// and BF16 in the same file, and a table that predates the k-quants panics on
/// the first expert matrix it meets.
fn quantized_stride(ggml_type: GGMLType, n_in: usize, n_out: usize) -> usize {
    ggml_type.nbytes(n_in * n_out)
}

/// GGML types the expanded exports write.
///
/// The lossless export keeps the packed U32 words as I32 plus BF16 `scales` and
/// `biases` companions, which only `MlxAffineKernel` can read.  Every other
/// mode folds the affine groups into the matrix itself and re-encodes it, so
/// the result is an ordinary GGML tensor and goes through `load_weight`.
///
/// `q4_k_m` is in this list because it is a mix: it writes Q4_K for the bulk of
/// the linears, Q6_K where the mixed-precision rule asks for more, and leaves
/// small or sensitive tensors at F16 / Q8_0.  All of those are already expanded,
/// so treating any of them as affine codes sends the loader looking for
/// `.scales` companions that the export never wrote.
const EXPANDED_TYPES: [GGMLType; 6] = [
    GGMLType::F32,
    GGMLType::F16,
    GGMLType::Q8_0,
    GGMLType::Q4_0,
    GGMLType::Q4K,
    GGMLType::Q6K,
];

/// True when the checkpoint stores pre-expanded matrices instead of affine codes.
///
/// `edge0.quant.mode` is written by `tools/converter/edge0/convert_edge0.py`;
/// a checkpoint without the key predates the flag and is lossless.
pub(crate) fn is_expanded(source: &dyn TensorSource) -> bool {
    match source.metadata("edge0.quant.mode") {
        Some(value) => value.to_string_val().as_deref() != Some("lossless"),
        None => false,
    }
}

/// Load one Edge0 MoE matrix, accepting either the affine triplet or an
/// already-expanded GGML tensor.
///
/// `expert` selects a slice out of a 3D expert-stacked tensor.  The expanded
/// export keeps the expert axis last, so the same stride arithmetic applies to
/// both layouts once the per-expert element count is known.
pub(crate) fn load_affine<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    expert: Option<usize>,
) -> Result<Weight<'a>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("missing {name}"))?;
    if EXPANDED_TYPES.contains(&info.ggml_type) {
        return load_expanded(source, name, info, expert);
    }
    load_packed(source, name, info, expert)
}

/// Load a matrix whose affine groups have already been folded in, so the tensor
/// is a plain GGML matrix and the generic quantized kernels can run it.
fn load_expanded<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    info: &TensorInfo,
    expert: Option<usize>,
) -> Result<Weight<'a>, String> {
    let experts = info.dims.get(2).copied().unwrap_or(1) as usize;
    if expert.is_some() != (info.dims.len() == 3) || expert.unwrap_or(0) >= experts {
        return Err(format!("invalid Edge0 expert index for {name}"));
    }
    let n_in = info.dims[0] as usize;
    let n_out = info.dims[1] as usize;
    if n_in == 0 || n_out == 0 {
        return Err(format!("invalid Edge0 matrix dimensions for {name}"));
    }
    // `from_quantized` addresses a whole matrix, so hand it this expert's slice
    // rather than teaching the shared kernel about a leading expert stride.
    let data = source
        .tensor_slice(name)
        .ok_or_else(|| format!("missing {name} data"))?;
    let stride = quantized_stride(info.ggml_type, n_in, n_out);
    let index = expert.unwrap_or(0);
    let slice = data
        .get(index * stride..(index + 1) * stride)
        .ok_or_else(|| format!("truncated {name}"))?;
    let mut weight = Weight::from_quantized(QuantizedTensor::from_bytes(
        slice,
        info.ggml_type,
        n_in,
        n_out,
    ));
    weight.n_in = n_in;
    weight.n_out = n_out;
    if info.ggml_type == GGMLType::BF16 {
        weight.kernel = Box::new(crate::ops::kernel::bf16::BF16Kernel::with_bf16_input(slice));
    }
    Ok(weight)
}

/// Load the lossless layout: I32 packed codes plus BF16 scale/bias companions,
/// dequantized on the fly by `MlxAffineKernel`.
fn load_packed<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    info: &TensorInfo,
    expert: Option<usize>,
) -> Result<Weight<'a>, String> {
    let scale_name = name.replace(".weight", ".scales");
    let bias_name = name.replace(".weight", ".biases");
    let scale_info = source
        .tensor_info(&scale_name)
        .ok_or_else(|| format!("missing {scale_name}"))?;
    let bias_info = source
        .tensor_info(&bias_name)
        .ok_or_else(|| format!("missing {bias_name}"))?;
    if info.ggml_type != GGMLType::I32
        || scale_info.ggml_type != GGMLType::BF16
        || bias_info.ggml_type != GGMLType::BF16
        || scale_info.dims != bias_info.dims
        || info.dims.len() != scale_info.dims.len()
        || !matches!(info.dims.len(), 2 | 3)
        || info.dims[1..] != scale_info.dims[1..]
    {
        return Err(format!("invalid Edge0 affine tensor contract: {name}"));
    }
    let experts = info.dims.get(2).copied().unwrap_or(1) as usize;
    if expert.is_some() != (info.dims.len() == 3) || expert.unwrap_or(0) >= experts {
        return Err(format!("invalid Edge0 expert index for {name}"));
    }
    let n_out = info.dims[1] as usize;
    let groups = scale_info.dims[0] as usize;
    let n_in = groups.checked_mul(64).ok_or("Edge0 input width overflow")?;
    let packed_cols = info.dims[0] as usize;
    if n_in == 0 || packed_cols == 0 || packed_cols * 32 % n_in != 0 {
        return Err(format!("invalid Edge0 packed width for {name}"));
    }
    let bits = packed_cols * 32 / n_in;
    let packed = source
        .tensor_slice(name)
        .ok_or_else(|| format!("missing {name} data"))?;
    let scales = source
        .tensor_slice(&scale_name)
        .ok_or_else(|| format!("missing {scale_name} data"))?;
    let biases = source
        .tensor_slice(&bias_name)
        .ok_or_else(|| format!("missing {bias_name} data"))?;
    let packed_stride = packed_cols * n_out * 4;
    let scale_stride = groups * n_out * 2;
    let index = expert.unwrap_or(0);
    let packed = packed
        .get(index * packed_stride..(index + 1) * packed_stride)
        .ok_or_else(|| format!("truncated {name}"))?;
    let scales = scales
        .get(index * scale_stride..(index + 1) * scale_stride)
        .ok_or_else(|| format!("truncated {scale_name}"))?;
    let biases = biases
        .get(index * scale_stride..(index + 1) * scale_stride)
        .ok_or_else(|| format!("truncated {bias_name}"))?;
    let lora_a_name = name.replace(".weight", ".lora_A");
    let lora_b_name = name.replace(".weight", ".lora_B");
    let lora = match (
        source.tensor_info(&lora_a_name),
        source.tensor_info(&lora_b_name),
    ) {
        (None, None) => None,
        (Some(a), Some(b))
            if expert.is_none()
                && a.ggml_type == GGMLType::F16
                && b.ggml_type == GGMLType::F16
                && a.dims.len() == 2
                && b.dims.len() == 2
                && a.dims[0] as usize == n_in
                && b.dims[0] == a.dims[1]
                && b.dims[1] as usize == n_out =>
        {
            let scale = source
                .metadata("edge0.lora.scale")
                .and_then(|v| v.to_f64())
                .ok_or("missing edge0.lora.scale")? as f32;
            Some((
                source
                    .tensor_slice(&lora_a_name)
                    .ok_or("missing LoRA A data")?,
                source
                    .tensor_slice(&lora_b_name)
                    .ok_or("missing LoRA B data")?,
                a.dims[1] as usize,
                scale,
            ))
        }
        _ => return Err(format!("invalid Edge0 LoRA pair for {name}")),
    };
    let kernel = MlxAffineKernel::new(packed, scales, biases, n_in, n_out, bits, lora)?;
    Ok(Weight {
        kernel: Box::new(kernel),
        ggml_type: GGMLType::I32,
        n_in,
        n_out,
    })
}

pub struct Edge0Model<'a> {
    pub(crate) trunk: HybridTrunk<'a>,
    pub(crate) moe: Vec<Edge0MoeWeights<'a>>,
}

impl<'a> Edge0Model<'a> {
    pub fn from_source(source: &'a dyn TensorSource) -> Result<Self, String> {
        let arch = source
            .metadata("general.architecture")
            .and_then(|value| value.to_string_val());
        if arch != Some("edge0") {
            return Err(format!(
                "Edge0Model requires general.architecture=edge0, got {arch:?}"
            ));
        }
        let config = Qwen35Config::from_source_for_arch(source, "edge0")?;
        let trunk = HybridTrunk::load_trunk(source, config, true)?;
        validate_edge0_trunk(&trunk)?;
        let moe = load_edge0_moe(source, &trunk.config)?;
        Ok(Self { trunk, moe })
    }
}

pub struct Edge0MoeWeights<'a> {
    pub router: Weight<'a>,
    pub shared_gate: Weight<'a>,
    pub gate: Vec<Weight<'a>>,
    pub up: Vec<Weight<'a>>,
    pub down: Vec<Weight<'a>>,
    pub used: usize,
}

fn load_edge0_moe<'a>(
    source: &'a dyn TensorSource,
    config: &Qwen35Config,
) -> Result<Vec<Edge0MoeWeights<'a>>, String> {
    let count = source
        .metadata("edge0.expert_count")
        .and_then(|v| v.to_u64())
        .unwrap_or(0) as usize;
    let used = source
        .metadata("edge0.expert_used_count")
        .and_then(|v| v.to_u64())
        .unwrap_or(0) as usize;
    let width = source
        .metadata("edge0.expert_feed_forward_length")
        .and_then(|v| v.to_u64())
        .unwrap_or(0) as usize;
    if count != 256 || used != 4 || width != 512 || config.n_ff != 512 {
        return Err("unsupported Edge0 MoE contract".into());
    }
    (0..config.n_layer_impl())
        .map(|layer| {
            let prefix = format!("blk.{layer}");
            let router = load_affine(source, &format!("{prefix}.ffn_gate_inp.weight"), None)?;
            let shared_gate =
                load_affine(source, &format!("{prefix}.ffn_shared_gate.weight"), None)?;
            if (
                router.n_in,
                router.n_out,
                shared_gate.n_in,
                shared_gate.n_out,
            ) != (config.n_embd, count, config.n_embd, 1)
            {
                return Err(format!("Edge0 router shape mismatch at layer {layer}"));
            }
            let load_experts =
                |kind: &str, input: usize, output: usize| -> Result<Vec<Weight<'a>>, String> {
                    let name = format!("{prefix}.ffn_{kind}_exps.weight");
                    let info = source
                        .tensor_info(&name)
                        .ok_or_else(|| format!("missing {name}"))?;
                    if info.dims.len() != 3 || info.dims[2] as usize != count {
                        return Err(format!("Edge0 expert count mismatch for {name}"));
                    }
                    (0..count)
                        .map(|expert| {
                            let weight = load_affine(source, &name, Some(expert))?;
                            if (weight.n_in, weight.n_out) != (input, output) {
                                return Err(format!("Edge0 expert shape mismatch for {name}"));
                            }
                            Ok(weight)
                        })
                        .collect()
                };
            Ok(Edge0MoeWeights {
                router,
                shared_gate,
                gate: load_experts("gate", config.n_embd, width)?,
                up: load_experts("up", config.n_embd, width)?,
                down: load_experts("down", width, config.n_embd)?,
                used,
            })
        })
        .collect()
}

fn validate_edge0_trunk(trunk: &HybridTrunk<'_>) -> Result<(), String> {
    let require =
        |name: &str, weight: Option<&Weight<'_>>, input: usize, output: usize| match weight {
            Some(weight) if (weight.n_in, weight.n_out) == (input, output) => Ok(()),
            _ => Err(format!("Edge0 {name} shape mismatch or missing")),
        };
    for (index, layer) in trunk.layers.iter().enumerate() {
        let prefix = format!("layer {index}");
        require(
            &format!("{prefix} shared gate"),
            Some(&layer.ffn_gate),
            trunk.config.n_embd,
            trunk.config.n_ff,
        )?;
        require(
            &format!("{prefix} shared up"),
            Some(&layer.ffn_up),
            trunk.config.n_embd,
            trunk.config.n_ff,
        )?;
        require(
            &format!("{prefix} shared down"),
            Some(&layer.ffn_down),
            trunk.config.n_ff,
            trunk.config.n_embd,
        )?;
        if trunk.config.is_recurrent[index] {
            require(
                &format!("{prefix} qkv"),
                layer.wqkv.as_ref(),
                trunk.config.n_embd,
                8192,
            )?;
            require(
                &format!("{prefix} z"),
                layer.wqkv_gate.as_ref(),
                trunk.config.n_embd,
                4096,
            )?;
            require(
                &format!("{prefix} alpha"),
                layer.ssm_alpha.as_ref(),
                trunk.config.n_embd,
                32,
            )?;
            require(
                &format!("{prefix} beta"),
                layer.ssm_beta.as_ref(),
                trunk.config.n_embd,
                32,
            )?;
            require(
                &format!("{prefix} out"),
                layer.ssm_out.as_ref(),
                4096,
                trunk.config.n_embd,
            )?;
            if layer.ssm_conv1d.as_ref().map(Vec::len) != Some(8192 * 4)
                || layer.ssm_dt.as_ref().map(Vec::len) != Some(32)
                || layer.ssm_a.as_ref().map(Vec::len) != Some(32)
                || layer.ssm_norm.as_ref().map(Vec::len) != Some(128)
            {
                return Err(format!("Edge0 {prefix} recurrent tensor shape mismatch"));
            }
        } else {
            require(
                &format!("{prefix} q"),
                layer.wq.as_ref(),
                trunk.config.n_embd,
                8192,
            )?;
            require(
                &format!("{prefix} k"),
                layer.wk.as_ref(),
                trunk.config.n_embd,
                512,
            )?;
            require(
                &format!("{prefix} v"),
                layer.wv.as_ref(),
                trunk.config.n_embd,
                512,
            )?;
            require(
                &format!("{prefix} o"),
                layer.wo.as_ref(),
                4096,
                trunk.config.n_embd,
            )?;
            if layer.attn_q_norm.as_ref().map(Vec::len) != Some(256)
                || layer.attn_k_norm.as_ref().map(Vec::len) != Some(256)
            {
                return Err(format!("Edge0 {prefix} attention norm shape mismatch"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_stride_matches_the_ggml_block_layouts() {
        // 64 inputs by 3 rows: two 32-value blocks per row.
        assert_eq!(quantized_stride(GGMLType::Q8_0, 64, 3), 2 * 34 * 3);
        assert_eq!(quantized_stride(GGMLType::Q4_0, 64, 3), 2 * 18 * 3);
        assert_eq!(quantized_stride(GGMLType::F16, 64, 3), 64 * 3 * 2);
        assert_eq!(quantized_stride(GGMLType::BF16, 64, 3), 64 * 3 * 2);
        assert_eq!(quantized_stride(GGMLType::F32, 64, 3), 64 * 3 * 4);
    }

    #[test]
    fn expanded_types_cover_the_modes_the_converter_can_write() {
        // Everything the converter emits for --quant f32/f16/q8_0/q4_0.
        for ty in [GGMLType::F32, GGMLType::F16, GGMLType::Q8_0, GGMLType::Q4_0] {
            assert!(
                EXPANDED_TYPES.contains(&ty),
                "{ty:?} must route to the generic kernels"
            );
        }
        // The lossless layout must keep going through MlxAffineKernel.
        assert!(!EXPANDED_TYPES.contains(&GGMLType::I32));
    }
}
