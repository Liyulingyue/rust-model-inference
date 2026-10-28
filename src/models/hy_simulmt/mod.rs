//! Tencent Hy-SimulMT's bundled speech encoder and mixed dense/MoE decoder.
mod audio;
pub mod moe;
mod runtime;

pub use audio::HyAudioEncoder;
pub use runtime::{HySimulMt, Translation};

use crate::core::tensor::{GGMLType, TensorSource};
use crate::models::qwen3::Qwen3Config;

pub(crate) fn check_tensor(
    source: &dyn TensorSource,
    name: &str,
    dims: &[u64],
    float: bool,
) -> Result<(), String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims {
        return Err(format!(
            "Invalid {name} shape {:?}; expected {dims:?}",
            info.dims
        ));
    }
    let supported = if float {
        info.ggml_type == GGMLType::F32
    } else {
        matches!(
            info.ggml_type,
            GGMLType::F32 | GGMLType::BF16 | GGMLType::Q4K | GGMLType::Q6K | GGMLType::Q8_0
        )
    };
    if !supported {
        return Err(format!(
            "Unsupported Hy-SimulMT {name} type {:?}",
            info.ggml_type
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if info.checked_nbytes().and_then(|n| usize::try_from(n).ok()) != Some(bytes.len()) {
        return Err(format!("Invalid {name} byte length"));
    }
    Ok(())
}

pub(crate) fn validate_decoder(s: &dyn TensorSource, c: &Qwen3Config) -> Result<(), String> {
    if c.n_layer < 2
        || c.n_head == 0
        || c.n_head_kv == 0
        || c.n_head % c.n_head_kv != 0
        || c.n_embd_head_k != c.n_embd_head_v
        || c.n_embd_head_k % 2 != 0
    {
        return Err("Invalid Hy-SimulMT decoder dimensions".into());
    }
    let m = c.moe.ok_or("missing Hy-SimulMT MoE config")?;
    let h = c.n_embd as u64;
    let q = c
        .n_head
        .checked_mul(c.n_embd_head_k)
        .ok_or("Hy-SimulMT query width overflow")? as u64;
    let kv = c
        .n_head_kv
        .checked_mul(c.n_embd_head_k)
        .ok_or("Hy-SimulMT KV width overflow")? as u64;
    check_tensor(s, "token_embd.weight", &[h, c.vocab as u64], false)?;
    check_tensor(s, "output.weight", &[h, c.vocab as u64], false)?;
    check_tensor(s, "output_norm.weight", &[h], true)?;
    for l in 0..c.n_layer {
        for (name, dims, float) in [
            ("attn_norm.weight", vec![h], true),
            ("ffn_norm.weight", vec![h], true),
            ("attn_q_norm.weight", vec![c.n_embd_head_k as u64], true),
            ("attn_k_norm.weight", vec![c.n_embd_head_k as u64], true),
            ("attn_q.weight", vec![h, q], false),
            ("attn_k.weight", vec![h, kv], false),
            ("attn_v.weight", vec![h, kv], false),
            ("attn_output.weight", vec![q, h], false),
        ] {
            check_tensor(s, &format!("blk.{l}.{name}"), &dims, float)?;
        }
        if l == 0 {
            for (name, dims) in [
                ("gate", vec![h, c.n_ff as u64]),
                ("up", vec![h, c.n_ff as u64]),
                ("down", vec![c.n_ff as u64, h]),
            ] {
                check_tensor(s, &format!("blk.{l}.ffn_{name}.weight"), &dims, false)?;
            }
        } else {
            check_tensor(
                s,
                &format!("blk.{l}.ffn_gate_inp.weight"),
                &[h, m.expert_count as u64],
                true,
            )?;
            check_tensor(
                s,
                &format!("blk.{l}.exp_probs_b"),
                &[m.expert_count as u64],
                true,
            )?;
            for (name, a, b) in [
                ("gate", h, m.expert_ffn as u64),
                ("up", h, m.expert_ffn as u64),
                ("down", m.expert_ffn as u64, h),
            ] {
                check_tensor(
                    s,
                    &format!("blk.{l}.ffn_{name}_exps.weight"),
                    &[a, b, m.expert_count as u64],
                    false,
                )?;
            }
            for (name, a, b) in [
                ("gate", h, m.shared_expert_ffn as u64),
                ("up", h, m.shared_expert_ffn as u64),
                ("down", m.shared_expert_ffn as u64, h),
            ] {
                check_tensor(
                    s,
                    &format!("blk.{l}.ffn_{name}_shexp.weight"),
                    &[a, b],
                    false,
                )?;
            }
        }
    }
    Ok(())
}

pub(crate) fn trace(name: &str, shape: &[usize], values: &[f32]) {
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint(name, None, shape, values));
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, shape, values);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{MetaValue, TensorInfo};
    struct Source {
        info: TensorInfo,
        bytes: Vec<u8>,
    }
    impl TensorSource for Source {
        fn metadata(&self, _: &str) -> Option<&MetaValue> {
            None
        }
        fn tensor_info(&self, n: &str) -> Option<&TensorInfo> {
            (n == self.info.name).then_some(&self.info)
        }
        fn tensor_slice(&self, n: &str) -> Option<&[u8]> {
            (n == self.info.name).then_some(&self.bytes)
        }
    }
    #[test]
    fn hy_tensor_boundary_rejects_missing_shape_type_and_truncation() {
        let mut s = Source {
            info: TensorInfo {
                name: "w".into(),
                dims: vec![2, 3],
                ggml_type: GGMLType::F32,
                offset: 0,
            },
            bytes: vec![0; 24],
        };
        assert!(check_tensor(&s, "w", &[2, 3], true).is_ok());
        assert!(check_tensor(&s, "missing", &[2, 3], true).is_err());
        assert!(check_tensor(&s, "w", &[3, 2], true).is_err());
        s.bytes.pop();
        assert!(check_tensor(&s, "w", &[2, 3], true).is_err());
        s.info.ggml_type = GGMLType::I32;
        assert!(check_tensor(&s, "w", &[2, 3], false).is_err());
    }
}
