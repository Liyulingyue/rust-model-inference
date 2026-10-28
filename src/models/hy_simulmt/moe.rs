use crate::core::thread_pool::ComputePool;
use crate::models::qwen3::{Qwen3Config, Qwen3LayerWeights};
use crate::ops::kernel::{PreparedRows, Weight};

pub struct SharedExpert<'a> {
    pub(crate) bias: Vec<f32>,
    pub(crate) scale: f32,
    pub(crate) gate: Weight<'a>,
    pub(crate) up: Weight<'a>,
    pub(crate) down: Weight<'a>,
}

fn route(
    logits: &[f32],
    bias: &[f32],
    used: usize,
    scale: f32,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    if logits.is_empty()
        || logits.len() != bias.len()
        || used == 0
        || used > logits.len()
        || !scale.is_finite()
        || scale <= 0.0
        || logits.iter().chain(bias).any(|v| !v.is_finite())
    {
        return Err("invalid Hy-SimulMT router dimensions, values or scale".into());
    }
    let probs: Vec<f32> = logits.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    let mut ids: Vec<usize> = (0..logits.len()).collect();
    ids.sort_by(|&a, &b| {
        (probs[b] + bias[b])
            .total_cmp(&(probs[a] + bias[a]))
            .then(a.cmp(&b))
    });
    ids.truncate(used);
    let sum = (ids.iter().map(|&i| f64::from(probs[i])).sum::<f64>() as f32).max(6.103515625e-5);
    let weights = ids.iter().map(|&i| (probs[i] / sum) * scale).collect();
    Ok((ids, weights))
}

pub(crate) fn forward(
    input: &[f32],
    layer: &Qwen3LayerWeights<'_>,
    cfg: &Qwen3Config,
    pool: &ComputePool,
    output: &mut [f32],
) -> Result<(), String> {
    let shared = layer
        .hy_shared
        .as_ref()
        .ok_or("missing Hy-SimulMT shared expert")?;
    let router = layer
        .moe_router
        .as_deref()
        .ok_or("missing Hy-SimulMT router")?;
    let cfg = cfg.moe.ok_or("missing Hy-SimulMT expert config")?;
    let gate = layer.moe_gate.as_deref().ok_or("missing gate experts")?;
    let up = layer.moe_up.as_deref().ok_or("missing up experts")?;
    let down = layer.moe_down.as_deref().ok_or("missing down experts")?;
    let logits: Vec<f32> = router
        .chunks_exact(input.len())
        .map(|row| crate::ops::dot_f32(input, row, input.len()))
        .collect();
    let (ids, weights) = route(&logits, &shared.bias, cfg.expert_used_count, shared.scale)?;
    // Keep the input quantized once across every routed and shared projection.
    let mut prepared = PreparedRows::new(1, input.len());
    let mut intermediate = PreparedRows::new(1, cfg.expert_ffn.max(cfg.shared_expert_ffn));
    let all = gate.iter().chain(up).chain([&shared.gate, &shared.up]);
    let need_q8 = all.clone().any(Weight::needs_q8_0_activation);
    let need_q8k = all.clone().any(Weight::uses_q8_k);
    prepared.prepare(input, 1, input.len(), need_q8, need_q8k)?;
    output.fill(0.0);
    for (id, probability) in ids
        .into_iter()
        .zip(weights)
        .map(|(id, p)| (Some(id), p))
        .chain(std::iter::once((None, 1.0)))
    {
        let (g, u, d) = match id {
            Some(i) => (&gate[i], &up[i], &down[i]),
            None => (&shared.gate, &shared.up, &shared.down),
        };
        let mut a = vec![0.0; g.n_out];
        let mut b = vec![0.0; u.n_out];
        prepared.matmul_group(input, [(g, &mut a), (u, &mut b)], pool)?;
        crate::ops::activation::silu_inplace(&mut a);
        for (a, b) in a.iter_mut().zip(b) {
            *a *= b;
        }
        intermediate.prepare(&a, 1, a.len(), d.needs_q8_0_activation(), d.uses_q8_k())?;
        let mut value = vec![0.0; output.len()];
        intermediate.matmul(d, &a, &mut value, pool)?;
        for (out, v) in output.iter_mut().zip(value) {
            *out += probability * v;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::route;
    #[test]
    fn hy_router_bias_only_selects_experts() {
        let (ids, weights) = route(
            &[0.0, 3.0f32.ln(), -3.0f32.ln()],
            &[0.0, 0.0, 1.0],
            2,
            2.826,
        )
        .unwrap();
        assert_eq!(ids, [2, 1]);
        for (actual, expected) in weights.iter().zip([0.7065, 2.1195]) {
            assert!((actual - expected).abs() < 3e-7, "{actual} != {expected}");
        }
        assert_eq!(
            route(&[0.0; 4], &[0.0; 4], 2, 1.0).unwrap(),
            (vec![0, 1], vec![0.5, 0.5])
        );
        assert_eq!(route(&[-1000.0], &[0.0], 1, 2.826).unwrap().1, [0.0]);
    }
    #[test]
    fn hy_routed_plus_ungated_shared_expert() {
        use super::*;
        use crate::ops::kernel::QuantizedTensor;
        let w = |data: Vec<f32>, n_in, n_out| {
            Weight::from_quantized(QuantizedTensor::F32 { data, n_in, n_out })
        };
        let layer = Qwen3LayerWeights {
            attn_norm: vec![],
            ffn_norm: vec![],
            q_norm: None,
            k_norm: None,
            q_bias: None,
            k_bias: None,
            v_bias: None,
            wq: w(vec![0.; 4], 2, 2),
            wk: w(vec![0.; 4], 2, 2),
            wv: w(vec![0.; 4], 2, 2),
            wo: w(vec![0.; 4], 2, 2),
            w_gate: w(vec![0.; 2], 2, 1),
            w_up: w(vec![0.; 2], 2, 1),
            w_down: w(vec![0.; 2], 1, 2),
            moe_router: Some(vec![0.; 6]),
            moe_gate: Some((0..3).map(|_| w(vec![1., 0.], 2, 1)).collect()),
            moe_up: Some((0..3).map(|_| w(vec![1., 0.], 2, 1)).collect()),
            moe_down: Some(
                (0..3)
                    .map(|i| w(vec![(i + 1) as f32, (i + 2) as f32], 1, 2))
                    .collect(),
            ),
            hy_shared: Some(SharedExpert {
                bias: vec![0., 0., 1.],
                scale: 2.826,
                gate: w(vec![0., 1.], 2, 1),
                up: w(vec![1., 0.], 2, 1),
                down: w(vec![1., -1.], 1, 2),
            }),
        };
        let cfg = Qwen3Config {
            architecture: "hy_v3".into(),
            n_embd: 2,
            n_layer: 2,
            n_head: 1,
            n_head_kv: 1,
            n_embd_head_k: 2,
            n_embd_head_v: 2,
            n_ff: 1,
            vocab: 1,
            n_ctx: 8,
            eps: 1e-5,
            freq_base: 10000.,
            has_qk_norm: true,
            has_qkv_bias: false,
            n_deepstack_layers: 0,
            rope: crate::models::qwen3::Qwen3Rope::Neox,
            moe: Some(crate::core::loader::Qwen3MoeConfig {
                expert_count: 3,
                expert_used_count: 2,
                expert_ffn: 1,
                shared_expert_ffn: 1,
            }),
        };
        let mut out = [0.; 2];
        forward(&[1., 2.], &layer, &cfg, &ComputePool::new(2), &mut out).unwrap();
        let sigmoid = 1.0 / (1.0 + (-1.0f32).exp());
        let shared = 2.0 / (1.0 + (-2.0f32).exp());
        assert!((out[0] - (5.652 * sigmoid + shared)).abs() < 2e-6);
        assert!((out[1] - (8.478 * sigmoid - shared)).abs() < 2e-6);
    }
    #[test]
    fn hy_router_rejects_invalid_dimensions_and_nonfinite_values() {
        for (logits, bias, used, scale) in [
            (&[][..], &[][..], 1, 1.0),
            (&[0.0][..], &[][..], 1, 1.0),
            (&[0.0][..], &[0.0][..], 2, 1.0),
            (&[f32::NAN][..], &[0.0][..], 1, 1.0),
            (&[0.0][..], &[f32::INFINITY][..], 1, 1.0),
            (&[0.0][..], &[0.0][..], 1, f32::NAN),
        ] {
            assert!(route(logits, bias, used, scale).is_err());
        }
    }
}
