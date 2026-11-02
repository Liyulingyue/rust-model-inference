//! Shared physical Qwen3.5 weights with independent caches for every Loop visit.
#[cfg(test)]
mod tests;

use crate::core::scratchpad::KvCache;
use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::models::qwen35::trunk::weights::{load_weight, load_weight_f32};
use crate::models::qwen35::trunk::{HybridTrunk, HybridTrunkModel, Qwen35Config, Qwen35Scratchpad};
use crate::ops::kernel::Weight;
use serde::Deserialize;

pub const ARCH: &str = "sensenova_loop";
pub const SOURCE_CONFIG_SHA256: &str =
    "3069b497e0a9ba70b1f74561831cbab0df0ccc4247f3ae106054c1632360ef51";

pub fn validate_projector(source: &dyn TensorSource) -> Result<(), String> {
    if source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        != Some("clip")
        || source
            .metadata("clip.projector_type")
            .and_then(|v| v.to_string_val())
            != Some("qwen3vl_merger")
        || source
            .metadata("sensenova_loop.source_config_sha256")
            .and_then(|v| v.to_string_val())
            != Some(SOURCE_CONFIG_SHA256)
        || source
            .metadata("clip.vision.projection_dim")
            .and_then(|v| v.to_u64())
            != Some(4096)
    {
        return Err("SenseNova Loop requires its matching vision projector".into());
    }
    Ok(())
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct MhcSettings {
    start: usize,
    end: usize,
    streams: usize,
    iterations: usize,
    fixed_identity: bool,
    #[serde(default = "one")]
    pre_scale: f32,
}
fn one() -> f32 {
    1.0
}
#[derive(Deserialize)]
struct LoopSettings {
    start: usize,
    end: usize,
    eval_count: usize,
    counts: Vec<usize>,
}
#[derive(Deserialize)]
struct TextSettings {
    num_hidden_layers: usize,
    layer_types: Vec<String>,
}
#[derive(Deserialize)]
struct Contract {
    text_config: TextSettings,
    loop_settings: LoopSettings,
    mhc: MhcSettings,
}

fn parse_contract(raw: &str, physical_layers: usize) -> Result<(Vec<usize>, MhcSettings), String> {
    let c: Contract = serde_json::from_str(raw).map_err(|e| format!("SenseNova config: {e}"))?;
    let l = c.loop_settings;
    let n = c.text_config.num_hidden_layers;
    if n != physical_layers
        || !(l.start < l.end && l.end <= n)
        || l.eval_count == 0
        || l.counts != [l.eval_count]
    {
        return Err("Invalid SenseNova Loop interval or count".into());
    }
    let visits = (l.end - l.start)
        .checked_mul(l.eval_count - 1)
        .and_then(|extra| n.checked_add(extra))
        .filter(|&v| v <= 4096)
        .ok_or("Oversized SenseNova execution plan")?;
    let types = &c.text_config.layer_types;
    if types.len() != n
        || !types.iter().any(|s| s == "full_attention")
        || types
            .iter()
            .any(|s| s != "full_attention" && s != "linear_attention")
    {
        return Err("Invalid SenseNova physical layer types".into());
    }
    let m = c.mhc;
    let phases = [(0, l.start), (l.start, l.end), (l.end, n)];
    if !(m.start < m.end && m.end <= n)
        || !matches!(m.streams, 2 | 4)
        || !(1..=1024).contains(&m.iterations)
        || !m.pre_scale.is_finite()
        || m.pre_scale <= 0.0
        || !phases.iter().any(|&(a, b)| a <= m.start && m.end <= b)
    {
        return Err("Invalid SenseNova MHC settings or phase boundary".into());
    }
    let mut plan = Vec::with_capacity(visits);
    plan.extend(0..l.start);
    for _ in 0..l.eval_count {
        plan.extend(l.start..l.end);
    }
    plan.extend(l.end..n);
    Ok((plan, m))
}

struct Router<'a> {
    weight: Weight<'a>,
    bias: Vec<f32>,
}
pub struct SenseNovaLoopModel<'a> {
    pub trunk: HybridTrunk<'a>,
    plan: Vec<usize>,
    settings: MhcSettings,
    routers: Vec<(Router<'a>, Router<'a>)>,
}

fn check_tensor(source: &dyn TensorSource, name: &str, dims: &[usize]) -> Result<(), String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing SenseNova tensor {name}"))?;
    let expected: Vec<u64> = dims.iter().map(|&v| v as u64).collect();
    if info.dims != expected || info.ggml_type != GGMLType::F32 {
        return Err(format!(
            "Invalid SenseNova tensor {name}: {:?} {:?}, expected {expected:?} F32",
            info.dims, info.ggml_type,
        ));
    }
    if source.tensor_slice(name).map(|b| b.len() as u64) != info.checked_nbytes() {
        return Err(format!("Invalid SenseNova tensor data: {name}"));
    }
    Ok(())
}

impl<'a> SenseNovaLoopModel<'a> {
    pub fn from_source(source: &'a dyn TensorSource) -> Result<Self, String> {
        if source
            .metadata("general.architecture")
            .and_then(|v| v.to_string_val())
            != Some(ARCH)
        {
            return Err("SenseNovaLoopModel requires sensenova_loop architecture".into());
        }
        let mut cfg = Qwen35Config::from_source_for_arch(source, ARCH)?;
        let raw = source
            .metadata("sensenova_loop.config")
            .and_then(|v| v.to_string_val())
            .ok_or("Missing sensenova_loop.config")?;
        let (plan, settings) = parse_contract(raw, cfg.n_layer)?;
        let json: serde_json::Value = serde_json::from_str(raw).map_err(|e| e.to_string())?;
        let text = &json["text_config"];
        for (key, value) in [
            ("hidden_size", cfg.n_embd),
            ("intermediate_size", cfg.n_ff),
            ("num_attention_heads", cfg.n_head),
            ("num_key_value_heads", cfg.n_head_kv),
            ("head_dim", cfg.key_length),
            ("vocab_size", cfg.vocab_size),
            ("max_position_embeddings", cfg.n_ctx),
            ("linear_conv_kernel_dim", cfg.ssm_d_conv),
            ("linear_key_head_dim", cfg.ssm_d_state),
            ("linear_value_head_dim", cfg.ssm_d_inner / cfg.ssm_dt_rank),
            ("linear_num_key_heads", cfg.ssm_n_group),
            ("linear_num_value_heads", cfg.ssm_dt_rank),
        ] {
            if text[key].as_u64() != Some(value as u64) {
                return Err(format!(
                    "SenseNova text_config.{key} disagrees with GGUF metadata"
                ));
            }
        }
        let rope = &text["rope_parameters"];
        let sections = serde_json::json!(&cfg.rope_dimension_sections[..3]);
        if text["hidden_act"] != "silu"
            || text["attention_bias"] != false
            || text["attn_output_gate"] != true
            || text["tie_word_embeddings"] != false
            || rope["rope_type"] != "default"
            || rope["mrope_interleaved"] != true
            || rope["mrope_section"] != sections
            || cfg.rope_dimension_sections[3] != 0
            || rope["rope_theta"].as_f64().map(|v| v as f32) != Some(cfg.rope_freq_base)
            || text["rms_norm_eps"].as_f64().map(|v| v as f32) != Some(cfg.norm_eps)
            || text["partial_rotary_factor"]
                .as_f64()
                .map(|v| v * cfg.key_length as f64)
                != Some(cfg.rope_dimension_count as f64)
            || rope["partial_rotary_factor"] != text["partial_rotary_factor"]
        {
            return Err("Unsupported or inconsistent SenseNova backbone/RoPE configuration".into());
        }
        let contract: Contract = serde_json::from_str(raw).map_err(|e| e.to_string())?;
        if cfg.n_nextn != 0
            || cfg.moe.is_some()
            || cfg
                .is_recurrent
                .iter()
                .zip(contract.text_config.layer_types)
                .any(|(&r, t)| r != (t == "linear_attention"))
        {
            return Err(
                "SenseNova backbone metadata disagrees with physical execution contract".into(),
            );
        }
        let h = cfg.n_embd;
        let ff = cfg.n_ff;
        let q = cfg.n_head * cfg.key_length;
        let kv = cfg.n_head_kv * cfg.key_length;
        for (name, dims) in [
            ("token_embd.weight", vec![h, cfg.vocab_size]),
            ("output.weight", vec![h, cfg.vocab_size]),
            ("output_norm.weight", vec![h]),
        ] {
            check_tensor(source, name, &dims)?;
        }
        for i in 0..cfg.n_layer {
            let mut shapes = vec![
                ("attn_norm.weight", vec![h]),
                ("post_attention_norm.weight", vec![h]),
                ("ffn_gate.weight", vec![h, ff]),
                ("ffn_up.weight", vec![h, ff]),
                ("ffn_down.weight", vec![ff, h]),
            ];
            if cfg.is_recurrent[i] {
                let conv = 2 * cfg.ssm_n_group * cfg.ssm_d_state + cfg.ssm_d_inner;
                shapes.extend([
                    ("attn_qkv.weight", vec![h, conv]),
                    ("attn_gate.weight", vec![h, cfg.ssm_d_inner]),
                    ("ssm_alpha.weight", vec![h, cfg.ssm_dt_rank]),
                    ("ssm_beta.weight", vec![h, cfg.ssm_dt_rank]),
                    ("ssm_conv1d.weight", vec![cfg.ssm_d_conv, conv]),
                    ("ssm_dt.bias", vec![cfg.ssm_dt_rank]),
                    ("ssm_a", vec![cfg.ssm_dt_rank]),
                    ("ssm_norm.weight", vec![cfg.ssm_d_inner / cfg.ssm_dt_rank]),
                    ("ssm_out.weight", vec![cfg.ssm_d_inner, h]),
                ]);
            } else {
                shapes.extend([
                    ("attn_q.weight", vec![h, 2 * q]),
                    ("attn_k.weight", vec![h, kv]),
                    ("attn_v.weight", vec![h, kv]),
                    ("attn_output.weight", vec![q, h]),
                    ("attn_q_norm.weight", vec![cfg.key_length]),
                    ("attn_k_norm.weight", vec![cfg.key_length]),
                ]);
            }
            for (suffix, dims) in shapes {
                check_tensor(source, &format!("blk.{i}.{suffix}"), &dims)?;
            }
        }
        let s = settings.streams;
        let out = 2 * s + if settings.fixed_identity { 0 } else { s * s };
        let mut routers = Vec::new();
        for i in settings.start..settings.end {
            let load = |kind: &str| -> Result<Router<'a>, String> {
                let stem = format!("blk.{i}.{kind}");
                check_tensor(source, &format!("{stem}.weight"), &[s * h, out])?;
                check_tensor(source, &format!("{stem}.bias"), &[out])?;
                let bias =
                    load_weight_f32(source, &format!("{stem}.bias")).ok_or("Missing mHC bias")?;
                if bias.iter().any(|v| !v.is_finite()) {
                    return Err("Non-finite mHC bias".into());
                }
                Ok(Router {
                    weight: load_weight(source, &format!("{stem}.weight"))
                        .ok_or("Missing mHC weight")?,
                    bias,
                })
            };
            routers.push((load("mhc_attn")?, load("mhc_ffn")?));
        }
        let mut trunk = HybridTrunk::load_trunk(source, cfg.clone(), false)?;
        cfg.is_recurrent = plan.iter().map(|&p| cfg.is_recurrent[p]).collect();
        cfg.n_layer = plan.len();
        trunk.config = cfg;
        Ok(Self {
            trunk,
            plan,
            settings,
            routers,
        })
    }
}

fn sinkhorn(mut logits: Vec<f32>, streams: usize, iterations: usize) -> Vec<f32> {
    for _ in 0..iterations {
        for axis in 0..2 {
            for i in 0..streams {
                let at = |j| {
                    if axis == 0 {
                        i * streams + j
                    } else {
                        j * streams + i
                    }
                };
                let max = (0..streams)
                    .map(|j| logits[at(j)])
                    .fold(f32::NEG_INFINITY, f32::max);
                let sum = (0..streams)
                    .map(|j| (logits[at(j)] - max).exp())
                    .sum::<f32>();
                let norm = max + sum.ln();
                for j in 0..streams {
                    logits[at(j)] -= norm;
                }
            }
        }
    }
    logits.iter_mut().for_each(|v| *v = v.exp());
    logits
}

impl Router<'_> {
    fn route(
        &self,
        residual: &[f32],
        h: usize,
        m: &MhcSettings,
        _pool: &ComputePool,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let s = m.streams;
        let rows = residual.len() / (h * s);
        let mut normalized = residual.to_vec();
        let ones = vec![1.; h * s];
        for row in normalized.chunks_exact_mut(h * s) {
            crate::ops::rms_norm_inplace(row, &ones, 1e-6);
        }
        let mut logits: Vec<f32> = normalized
            .chunks_exact(h * s)
            .flat_map(|row| self.weight.matmul(row))
            .collect();
        let mut branch = vec![0.; rows * h];
        let mut post = vec![0.; rows * s];
        let mut mixes = Vec::with_capacity(rows * s * s);
        for (row, logits) in logits.chunks_exact_mut(self.bias.len()).enumerate() {
            for (l, b) in logits.iter_mut().zip(&self.bias) {
                *l += b;
            }
            for stream in 0..s {
                let pre = 1. / (1. + (-logits[stream]).exp());
                post[row * s + stream] = 2. / (1. + (-logits[s + stream]).exp());
                for k in 0..h {
                    branch[row * h + k] += pre * residual[(row * s + stream) * h + k];
                }
            }
            if m.pre_scale != 1. {
                for v in &mut branch[row * h..(row + 1) * h] {
                    *v *= m.pre_scale;
                }
            }
            mixes.extend(if m.fixed_identity {
                (0..s * s)
                    .map(|i| if i / s == i % s { 1. } else { 0. })
                    .collect()
            } else {
                sinkhorn(logits[2 * s..].to_vec(), s, m.iterations)
            });
        }
        (branch, post, mixes)
    }
}

fn merge(
    residual: &[f32],
    branch: &[f32],
    post: &[f32],
    mix: &[f32],
    s: usize,
    h: usize,
) -> Vec<f32> {
    let rows = branch.len() / h;
    let mut out = vec![0.; residual.len()];
    for row in 0..rows {
        for i in 0..s {
            for k in 0..h {
                let value = (0..s)
                    .map(|j| mix[(row * s + i) * s + j] * residual[(row * s + j) * h + k])
                    .sum::<f32>();
                out[(row * s + i) * h + k] = value + post[row * s + i] * branch[row * h + k];
            }
        }
    }
    out
}

fn trace(name: &str, rows: usize, width: usize, values: &[f32]) {
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint_rows(
        name,
        None,
        &[rows, width],
        values,
    ));
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, rows, width, values);
}

impl<'m> HybridTrunkModel<'m> for SenseNovaLoopModel<'m> {
    fn trunk(&self) -> &HybridTrunk<'m> {
        &self.trunk
    }
    #[cfg(feature = "vulkan")]
    fn supports_vulkan(&self) -> bool {
        false
    }
    fn forward_at(
        &mut self,
        rows: usize,
        base: usize,
        kv: &mut KvCache,
        scratch: &mut Qwen35Scratchpad,
        pool: &ComputePool,
        positions: &[[usize; 4]],
    ) -> Result<Vec<f32>, String> {
        #[cfg(feature = "vulkan")]
        let _cpu = ComputePool::disable_gpu_matmul_for_scope();
        let t = &self.trunk;
        let h = t.config.n_embd;
        let m = &self.settings;
        if rows == 0 || positions.len() != rows {
            return Err("Invalid SenseNova forward rows/positions".into());
        }
        let mut hidden = scratch.x[..rows * h].to_vec();
        trace("loop.embedding", rows, h, &hidden);
        for (visit, &physical) in self.plan.iter().enumerate() {
            let layer = &t.layers[physical];
            let routed = (m.start..m.end).contains(&physical);
            if routed && physical == m.start {
                hidden = hidden
                    .chunks_exact(h)
                    .flat_map(|row| (0..m.streams).flat_map(move |_| row.iter().copied()))
                    .collect();
            }
            for is_ffn in [false, true] {
                let (mut branch, post, mix) = if routed {
                    let routers = &self.routers[physical - m.start];
                    (if is_ffn { &routers.1 } else { &routers.0 }).route(&hidden, h, m, pool)
                } else {
                    (hidden.clone(), Vec::new(), Vec::new())
                };
                let norm = if is_ffn {
                    &layer.attn_post_norm
                } else {
                    &layer.attn_norm
                };
                for row in branch.chunks_exact_mut(h) {
                    crate::ops::rms_norm_inplace(row, norm, t.config.norm_eps);
                }
                let output = if is_ffn {
                    t.forward_ffn_parallel(layer, &branch, rows, scratch, pool, false);
                    scratch.buf[..rows * h].to_vec()
                } else if t.config.is_recurrent[visit] {
                    t.forward_recurrent_layer(
                        physical,
                        visit,
                        &branch,
                        rows,
                        scratch,
                        pool,
                        false,
                        true,
                        #[cfg(feature = "parity-trace")]
                        false,
                    )?
                } else {
                    t.forward_dense_attn_layer(
                        physical,
                        visit,
                        &branch,
                        rows,
                        kv,
                        scratch,
                        pool,
                        positions,
                        Some(base),
                        true,
                        #[cfg(feature = "parity-trace")]
                        false,
                    )
                };
                if routed {
                    hidden = merge(&hidden, &output, &post, &mix, m.streams, h);
                } else {
                    crate::ops::vec_add_into(&output, &mut hidden);
                }
            }
            if routed && physical + 1 == m.end {
                hidden = hidden
                    .chunks_exact(h * m.streams)
                    .flat_map(|row| {
                        (0..h).map(move |k| {
                            (0..m.streams).map(|s| row[s * h + k]).sum::<f32>() / m.streams as f32
                        })
                    })
                    .collect();
            }
            trace(
                &format!("loop.visit.{visit}"),
                rows,
                hidden.len() / rows,
                &hidden,
            );
        }
        scratch.x[..rows * h].copy_from_slice(&hidden);
        for row in hidden.chunks_exact_mut(h) {
            crate::ops::rms_norm_inplace(row, &t.output_norm, t.config.norm_eps);
        }
        scratch.normed_buf[..rows * h].copy_from_slice(&hidden);
        trace("loop.norm", rows, h, &hidden);
        let mut logits = vec![0.; t.config.vocab_size];
        t.output_weight.quantize_and_matmul_with_scratch(
            &hidden[(rows - 1) * h..],
            &mut scratch.q8k_buf,
            &mut scratch.q8_buf,
            &mut scratch.scale_buf,
            &mut logits,
            pool,
        );
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint_row(
            rows - 1,
            "loop.logits",
            None,
            &[1, logits.len()],
            &logits,
        ));
        Ok(logits)
    }
}
