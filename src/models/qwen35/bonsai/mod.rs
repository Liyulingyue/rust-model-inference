//! Prism's v1 signed, blockwise Hadamard representation of dense Qwen3.5.
//! Buffer layout and activation basis are independent: repacking into Q4_0
//! preserves every ternary code and FP16 scale, but its input still needs H·D.
use super::Qwen35Config;
use crate::core::tensor::{GGMLType, MetaValue, TensorSource};
use crate::ops::kernel::{q4_0::Q4_0Kernel, Kernel, Weight};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

mod packing;
use packing::repack;
#[cfg(test)]
mod tests;

pub(crate) const BLOCK: usize = 1024;

fn hadamard(values: &mut [f32]) {
    debug_assert!(values.len().is_multiple_of(BLOCK));
    for block in values.chunks_exact_mut(BLOCK) {
        let mut half = 1;
        while half < BLOCK {
            for pair in block.chunks_exact_mut(2 * half) {
                let (left, right) = pair.split_at_mut(half);
                for (a, b) in left.iter_mut().zip(right) {
                    (*a, *b) = (*a + *b, *a - *b);
                }
            }
            half *= 2;
        }
        for v in block {
            *v *= 1.0 / 32.0;
        }
    }
}

/// The gather fuses the sign diagonal and the optional GDN value-head transpose.
fn signed_map(signs: &[i32], heads: Option<(usize, usize)>) -> Result<Arc<[i32]>, String> {
    let (groups, repeats, head_dim) = match heads {
        Some((groups, heads))
            if groups != 0
                && heads != 0
                && heads.is_multiple_of(groups)
                && signs.len().is_multiple_of(heads) =>
        {
            (groups, heads / groups, signs.len() / heads)
        }
        Some(_) => return Err("invalid prism.hadamard GDN value head geometry".into()),
        None => (1, 1, signs.len()),
    };
    Ok(signs
        .iter()
        .enumerate()
        .map(|(i, sign)| {
            let head = i / head_dim;
            let src = ((head % repeats) * groups + head / repeats) * head_dim + i % head_dim;
            sign * (src as i32 + 1)
        })
        .collect())
}

struct FoldedMatrix {
    q4: Vec<u8>,
    map: Arc<[i32]>,
    // PQ2_0's CPU reference uses one Q8_K activation scale per 256 elements.
    q8k: bool,
}

impl Kernel for FoldedMatrix {
    fn weight_bytes(&self) -> Option<&[u8]> {
        Some(&self.q4)
    }
    #[cfg(feature = "vulkan")]
    fn hadamard_input_map(&self) -> Option<&[i32]> {
        Some(&self.map)
    }

    fn forward_prepared(
        &self,
        input: &[f32],
        _: &[u8],
        _: &[f32],
        _: Option<&[crate::ops::quant::BlockQ8K]>,
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        let mut rotated: Vec<f32> = self
            .map
            .iter()
            .map(|&idx| {
                let x = input[idx.unsigned_abs() as usize - 1];
                if idx < 0 {
                    -x
                } else {
                    x
                }
            })
            .collect();
        hadamard(&mut rotated);
        let mut q8 = vec![0; n_in];
        let mut scales = vec![0.0; n_in / 32];
        if self.q8k {
            let blocks = crate::ops::quant::quantize_row_q8_k(&rotated);
            for ((block, dst), ds) in blocks
                .iter()
                .zip(q8.chunks_exact_mut(256))
                .zip(scales.chunks_exact_mut(8))
            {
                dst.copy_from_slice(bytemuck::cast_slice(&block.qs));
                ds.fill(block.d);
            }
        } else {
            crate::ops::quantize_q8_0_into(&rotated, n_in, &mut q8, &mut scales);
        }
        Q4_0Kernel::new(&self.q4, n_in, n_out)
            .forward_prequantized(&q8, &scales, output, n_in, n_out, ith, nth);
    }

    fn forward_prequantized(
        &self,
        q8: &[u8],
        scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        let input: Vec<f32> = q8
            .iter()
            .enumerate()
            .map(|(i, &q)| q as i8 as f32 * scales[i / 32])
            .collect();
        self.forward_prepared(&input, &[], &[], None, output, n_in, n_out, ith, nth);
    }

    fn embedding_lookup(&self, token: u32, width: usize, out: &mut [f32]) {
        let row = &mut out[..width];
        Q4_0Kernel::new(&self.q4, width, 0).embedding_lookup(token, width, row);
        // Embeddings use the inverse basis D·H, without the GDN permutation.
        hadamard(row);
        for (value, &idx) in row.iter_mut().zip(self.map.iter()) {
            if idx < 0 {
                *value = -*value;
            }
        }
    }
}

fn has_folded_layout(source: &dyn TensorSource) -> Result<bool, String> {
    let meta = |key: &str| source.metadata(&format!("prism.hadamard.{key}"));
    if meta("version").is_none() {
        if source.tensor_info("token_embd.weight").is_some_and(|t| {
            matches!(
                t.ggml_type,
                GGMLType::PTQ1_0 | GGMLType::PQ2_0 | GGMLType::Q2_0
            )
        }) {
            return Err("ternary Qwen3.5 embedding requires prism.hadamard.version".into());
        }
        return Ok(false);
    }
    if !matches!(meta("version"), Some(MetaValue::Uint32(1)))
        || !matches!(meta("block_size"), Some(MetaValue::Uint32(n)) if *n as usize == BLOCK)
        || meta("transform").and_then(MetaValue::to_string_val)
            != Some("normalized-sylvester-walsh-hadamard")
        || meta("axis").and_then(MetaValue::to_string_val) != Some("input-last-dimension")
        || meta("sign_mode").and_then(MetaValue::to_string_val) != Some("explicit")
        || source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            != Some("qwen35")
    {
        return Err(
            "unsupported prism.hadamard transform, block size, sign mode or architecture".into(),
        );
    }
    if !matches!(meta("tied_output"), None | Some(MetaValue::Bool(false))) {
        return Err("prism.hadamard version 1 requires an untied output.weight".into());
    }
    Ok(true)
}

/// Validate the whole folded tensor list before allocating the Q4_0 matrices.
pub(crate) fn load(
    source: &dyn TensorSource,
    cfg: &Qwen35Config,
) -> Result<HashMap<String, Weight<'static>>, String> {
    if !has_folded_layout(source)? {
        return Ok(HashMap::new());
    }
    if cfg.key_length != cfg.value_length {
        return Err("Bonsai attention requires equal key and value head widths".into());
    }
    let meta = |key: &str| source.metadata(&format!("prism.hadamard.{key}"));
    let grouped = match meta("gdn_v_grouped") {
        Some(MetaValue::Bool(v)) => *v,
        None => false,
        _ => return Err("invalid prism.hadamard.gdn_v_grouped".into()),
    };
    let array = |key: &str| {
        meta(key)
            .and_then(MetaValue::to_arr)
            .ok_or_else(|| format!("missing prism.hadamard.{key} array"))
    };
    let mut values = array("sign_values")?.iter();
    let mut maps = HashMap::new();
    for raw in array("sign_widths")? {
        let width = match raw {
            MetaValue::Int32(n) if *n > 0 => *n as usize,
            _ => return Err("invalid prism.hadamard sign width".into()),
        };
        if !width.is_multiple_of(BLOCK) || width > values.len() {
            return Err("invalid prism.hadamard sign width".into());
        }
        let signs: Vec<i32> = values
            .by_ref()
            .take(width)
            .map(|v| match v {
                MetaValue::Int32(s @ (-1 | 1)) => Ok(*s),
                _ => Err("prism.hadamard signs must be -1 or 1"),
            })
            .collect::<Result<_, _>>()?;
        let plain = signed_map(&signs, None)?;
        let recurrent = if grouped && width == cfg.ssm_d_inner {
            signed_map(&signs, Some((cfg.ssm_n_group, cfg.ssm_dt_rank)))?
        } else {
            Arc::clone(&plain)
        };
        if maps.insert(width, (plain, recurrent)).is_some() {
            return Err("duplicate prism.hadamard sign width".into());
        }
    }
    if values.len() != 0 {
        return Err("extra prism.hadamard signs".into());
    }
    let mut names = HashSet::new();
    for name in array("weight_names")? {
        let name = name
            .to_string_val()
            .ok_or("expected prism.hadamard weight name")?;
        if !names.insert(name) {
            return Err(format!("duplicate prism.hadamard weight: {name}"));
        }
    }
    if !matches!(array("inverse_weight_names")?.as_slice(), [MetaValue::String(s)] if s == "token_embd.weight")
        || !names.remove("output.weight")
    {
        return Err("expected folded output.weight and inverse token_embd.weight".into());
    }

    let k_dim = cfg
        .ssm_d_state
        .checked_mul(cfg.ssm_n_group)
        .ok_or("GDN width overflow")?;
    let qkv_dim = k_dim
        .checked_mul(2)
        .and_then(|k| k.checked_add(cfg.ssm_d_inner))
        .ok_or("GDN width overflow")?;
    let mut matrices = vec![
        (
            "token_embd.weight".to_string(),
            cfg.n_embd,
            cfg.vocab_size,
            false,
        ),
        (
            "output.weight".to_string(),
            cfg.n_embd,
            cfg.vocab_size,
            false,
        ),
    ];
    for i in 0..cfg.n_layer_impl() {
        let mut add = |suffix, n_in, n_out, permute| -> Result<(), String> {
            let name = format!("blk.{i}.{suffix}.weight");
            if !names.remove(name.as_str()) {
                return Err(format!("missing prism.hadamard transform for {name}"));
            }
            matrices.push((name, n_in, n_out, permute));
            Ok(())
        };
        if cfg.is_recurrent[i] {
            add("attn_qkv", cfg.n_embd, qkv_dim, false)?;
            add("attn_gate", cfg.n_embd, cfg.ssm_d_inner, false)?;
            add("ssm_out", cfg.ssm_d_inner, cfg.n_embd, true)?;
        } else {
            let q = cfg
                .n_head
                .checked_mul(cfg.key_length)
                .ok_or("attention width overflow")?;
            let kv = cfg
                .n_head_kv
                .checked_mul(cfg.key_length)
                .ok_or("attention width overflow")?;
            add(
                "attn_q",
                cfg.n_embd,
                q.checked_mul(2).ok_or("attention gate width overflow")?,
                false,
            )?;
            add("attn_k", cfg.n_embd, kv, false)?;
            add("attn_v", cfg.n_embd, kv, false)?;
            add("attn_output", q, cfg.n_embd, false)?;
        }
        add("ffn_gate", cfg.n_embd, cfg.n_ff, false)?;
        add("ffn_up", cfg.n_embd, cfg.n_ff, false)?;
        add("ffn_down", cfg.n_ff, cfg.n_embd, false)?;
    }
    if let Some(extra) = names.into_iter().next() {
        return Err(format!("unsupported prism.hadamard weight: {extra}"));
    }
    let mut validated = Vec::with_capacity(matrices.len());
    for (name, n_in, n_out, permute) in matrices {
        let ty = crate::core::tensor::require_tensor_type(
            source,
            &name,
            &[n_in as u64, n_out as u64],
            &[GGMLType::PTQ1_0, GGMLType::PQ2_0, GGMLType::Q2_0],
        )?;
        let data = source
            .tensor_slice(&name)
            .ok_or_else(|| format!("invalid ternary tensor payload: {name}"))?;
        let (plain, recurrent) = maps
            .get(&n_in)
            .ok_or_else(|| format!("missing sign vector of width {n_in} for {name}"))?;
        let map = Arc::clone(if permute { recurrent } else { plain });
        validated.push((name, n_in, n_out, ty, data, map));
    }
    let mut weights = HashMap::with_capacity(validated.len());
    for (name, n_in, n_out, ty, data, map) in validated {
        let kernel = FoldedMatrix {
            q4: repack(ty, data)?,
            map,
            q8k: ty == GGMLType::PQ2_0,
        };
        weights.insert(
            name,
            Weight {
                kernel: Box::new(kernel),
                ggml_type: GGMLType::Q4_0,
                n_in,
                n_out,
            },
        );
    }
    Ok(weights)
}
