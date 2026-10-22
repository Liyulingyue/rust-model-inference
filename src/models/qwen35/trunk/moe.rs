//! `qwen35moe` routed + shared experts.
//!
//! Follows the oracle's `graph::build_layer_ffn`
//! (`src/models/qwen35moe.cpp`, pin `69f201a2`):
//!
//! ```text
//! moe_out   = build_moe_ffn(cur, gate_inp, up_exps, gate_exps, down_exps,
//!                          n_expert, n_expert_used, LLM_FFN_SILU, norm=true,
//!                          expert_weights_scale, SOFTMAX)
//! ffn_shexp = build_ffn(cur, up_shexp, gate_shexp, down_shexp,
//!                       LLM_FFN_SILU, LLM_FFN_PAR)      // already cur + shared
//! cur       = moe_out + ffn_shexp * sigmoid(gate_inp_shexp . cur)
//! ```
//!
//! `LLM_FFN_PAR` here is llama.cpp's *gate mode*, not a residual: it makes the
//! gate projection read the block input instead of the up-projection output,
//! so the shared expert is `down(silu(gate @ x) * (up @ x))` rather than a
//! textbook SwiGLU. `build_ffn` adds no residual of its own, which is why the
//! outer `moe_out + ...` is the only add in the graph.
//!
//! Routing is a softmax over all experts followed by top-k with the selected
//! weights re-normalised (`norm = true`) and scaled by `expert_weights_scale`.
//! Ties sort by ascending index, matching `ggml_argsort_top_k`, which does not
//! swap the top two on a tie.

use super::config::Qwen35Moe;
use super::scratch::Qwen35Scratchpad;
use super::trace;
use super::weights::Qwen35MoeWeights;
use crate::core::thread_pool::ComputePool;
use crate::ops::activation::silu_inplace;
use crate::ops::kernel::PreparedRows;

/// Routed + shared expert output for one token, written into `out`.
///
/// `layer` and `step` key the `RMI_PARITY_TRACE` checkpoints. They are
/// positional: without them every record lands in the same append-mode files
/// and none can be attributed, which is how the earlier attempt at MoE parity
/// ended up unfalsifiable.
pub(crate) fn forward_qwen35moe_token(
    moe: &Qwen35MoeWeights<'_>,
    cfg: &Qwen35Moe,
    weights_scale: f32,
    input: &[f32],
    out: &mut [f32],
    prepared: &mut PreparedRows,
    pool: &ComputePool,
    layer: usize,
    step: usize,
) -> Result<(), String> {
    let n_embd = input.len();
    let mut logits = vec![0.0f32; moe.n_expert];
    let mut shared_gate = [0.0f32];
    let shared_gate_weight =
        crate::ops::kernel::Weight::from_quantized(crate::ops::kernel::QuantizedTensor::F32 {
            data: moe.gate_inp_shexp.clone().unwrap_or_default(),
            n_in: n_embd,
            n_out: 1,
        });
    prepared.prepare(
        input,
        1,
        n_embd,
        moe.gate_inp.needs_q8_0_activation() || shared_gate_weight.needs_q8_0_activation(),
        moe.gate_inp.uses_q8_k() || shared_gate_weight.uses_q8_k(),
    )?;
    prepared.matmul_group(
        input,
        [
            (&moe.gate_inp, &mut logits[..]),
            (&shared_gate_weight, &mut shared_gate[..]),
        ],
        pool,
    )?;
    let (chosen, scaled) = route_experts(&logits, cfg.n_expert_used, weights_scale, layer, step)?;

    // Rust-only, no oracle counterpart: the block input both expert paths read.
    // The compare script reports it as unjoinable rather than failing on it.
    trace::layer_1d("ffn_moe_input", layer, step, input);

    let mut routed = vec![0.0f32; n_embd];
    for (slot, &expert) in chosen.iter().enumerate() {
        let gate_w = moe.gate(expert);
        let up_w = moe.up(expert);
        let down_w = moe.down(expert);
        let mut gate = vec![0.0f32; moe.n_ff_exp];
        let mut up = vec![0.0f32; moe.n_ff_exp];
        prepared.prepare(
            input,
            1,
            n_embd,
            gate_w.needs_q8_0_activation() || up_w.needs_q8_0_activation(),
            gate_w.uses_q8_k() || up_w.uses_q8_k(),
        )?;
        prepared.matmul_group(
            input,
            [(&gate_w, &mut gate[..]), (&up_w, &mut up[..])],
            pool,
        )?;
        silu_inplace(&mut gate);
        for (g, u) in gate.iter_mut().zip(up) {
            *g *= u;
        }
        let mut down = vec![0.0f32; n_embd];
        prepared.prepare(
            &gate,
            1,
            moe.n_ff_exp,
            down_w.needs_q8_0_activation(),
            down_w.uses_q8_k(),
        )?;
        prepared.matmul(&down_w, &gate, &mut down, pool)?;
        let score = scaled[slot];
        for (o, v) in routed.iter_mut().zip(down) {
            *o += score * v;
        }
    }
    // `ffn_moe_out` is the routed term alone, before the shared expert is added.
    trace::layer_1d("ffn_moe_out", layer, step, &routed);

    // Shared expert. With LLM_FFN_PAR the gate projection reads `input`
    // directly, so this is `silu(gate @ input) * (up @ input)`.
    out.copy_from_slice(&routed);
    let mut shared_acc = vec![0.0f32; n_embd];
    let (Some(up_shexp), Some(gate_shexp), Some(down_shexp)) = (
        moe.up_shexp.as_ref(),
        moe.gate_shexp.as_ref(),
        moe.down_shexp.as_ref(),
    ) else {
        // A tower with routed experts but no shared expert. `ffn_moe_out` is
        // already checkpointed above, so the routed path stays fully observable;
        // only the shexp checkpoints are unavailable, and saying so in the trace
        // beats letting them silently go missing.
        out.copy_from_slice(&routed);
        trace::layer_1d("ffn_shexp_gated", layer, step, &[]);
        return Ok(());
    };
    // The SwiGLU intermediate is `n_ff_shexp`, not the projection's output
    // width: `ffn_{up,gate}_shexp` are [n_embd, n_ff_shexp] and only
    // `ffn_down_shexp` is [n_ff_shexp, n_embd].
    let n_ff_shexp = cfg.n_ff_shexp;
    let mut gate = vec![0.0f32; n_ff_shexp];
    let mut up = vec![0.0f32; n_ff_shexp];
    prepared.prepare(
        input,
        1,
        n_embd,
        gate_shexp.needs_q8_0_activation() || up_shexp.needs_q8_0_activation(),
        gate_shexp.uses_q8_k() || up_shexp.uses_q8_k(),
    )?;
    prepared.matmul_group(
        input,
        [(&gate_shexp, &mut gate[..]), (&up_shexp, &mut up[..])],
        pool,
    )?;
    silu_inplace(&mut gate);
    for (g, u) in gate.iter_mut().zip(up) {
        *g *= u;
    }
    prepared.prepare(
        &gate,
        1,
        n_ff_shexp,
        down_shexp.needs_q8_0_activation(),
        down_shexp.uses_q8_k(),
    )?;
    prepared.matmul(down_shexp, &gate, &mut shared_acc, pool)?;
    // `ffn_shexp` is the shared FFN *before* the gate multiplies it.
    trace::layer_1d("ffn_shexp", layer, step, &shared_acc);

    // The oracle's gate is `build_lora_mm(ffn_gate_inp_shexp, cur)` then a
    // sigmoid, emitted as two nodes. `sigmoid(x)` here is the algebraically
    // equal `1/(1+exp(-x))`; keep it as ggml writes it so the bits match.
    trace::layer_1d("shared_expert_gate", layer, step, &shared_gate);
    let scale = 1.0 / (1.0 + (-shared_gate[0]).exp());
    trace::layer_1d("shared_expert_gate_sigmoid", layer, step, &[scale]);

    // cur = moe_out + ffn_shexp * sigmoid(gate_inp_shexp . cur).
    for (o, shared) in out.iter_mut().zip(&shared_acc) {
        *o += scale * shared;
    }
    // The oracle names the gated product `ffn_shexp_gated` and the final sum
    // `ffn_out`. This code used to dump the *pre*-gate accumulator under the
    // first name, which would have compared two different tensors.
    let gated: Vec<f32> = shared_acc.iter().map(|s| scale * s).collect();
    trace::layer_1d("ffn_shexp_gated", layer, step, &gated);
    trace::layer_1d("ffn_out", layer, step, out);
    Ok(())
}

/// Shared host routing for CPU and Vulkan expert projections.
pub(crate) fn route_experts(
    logits: &[f32],
    used: usize,
    weights_scale: f32,
    layer: usize,
    step: usize,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    if logits.is_empty() || used == 0 || used > logits.len() || !weights_scale.is_finite() {
        return Err("invalid qwen35moe routing dimensions or scale".into());
    }
    if logits.iter().any(|v| !v.is_finite()) {
        return Err("qwen35moe router produced non-finite logits".into());
    }
    trace::layer_1d("ffn_moe_logits", layer, step, &logits);

    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|v| (*v - max).exp()).collect();
    let total = exp.iter().sum::<f32>();
    let probabilities: Vec<f32> = exp.iter().map(|v| v / total).collect();
    trace::layer_1d("ffn_moe_probs", layer, step, &probabilities);

    // Descending by probability, ties by ascending index: ggml_argsort_top_k
    // leaves equal keys in index order.
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| {
        probabilities[b]
            .total_cmp(&probabilities[a])
            .then_with(|| a.cmp(&b))
    });
    order.truncate(used.min(logits.len()));
    let chosen = order;

    // The routing decision itself, as integers. This is the checkpoint that can
    // falsify a claim: "the same experts were selected" has to mean the same
    // ids in the same order, not merely a plausible-looking output.
    let chosen_ids: Vec<u32> = chosen.iter().map(|&i| i as u32).collect();
    trace::ids("ffn_moe_topk", layer, step, &chosen_ids);

    // The oracle's weight pipeline is four separate nodes and this code has to
    // reproduce each one, because `ffn_moe_weights_norm` is unscaled there:
    //   weights             = probs gathered at the chosen ids
    //   weights_sum         = row sum of those
    //   weights_sum_clamped = weights_sum clamped below at 2^-14
    //   weights_norm        = weights / weights_sum_clamped
    //   weights_scaled      = weights_norm * expert_weights_scale
    // Folding the scale into the divide, as this code used to, is algebraically
    // the same but not bit-identical: (s*p)/sum and (p/sum)*s differ in the
    // last bits, and that is exactly the magnitude that moves a router argmax.
    let gathered: Vec<f32> = chosen.iter().map(|&i| probabilities[i]).collect();
    trace::layer_1d("ffn_moe_weights", layer, step, &gathered);

    let sum: f32 = gathered.iter().sum();
    trace::layer_1d("ffn_moe_weights_sum", layer, step, &[sum]);

    // `ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY)` -- 2^-14. Not
    // cosmetic: without it a token whose chosen weights sum to ~0 divides by
    // ~0 and the routed output becomes non-finite.
    const SUM_FLOOR: f32 = 6.103515625e-5;
    let sum_clamped = sum.max(SUM_FLOOR);
    trace::layer_1d("ffn_moe_weights_sum_clamped", layer, step, &[sum_clamped]);

    let normed: Vec<f32> = gathered.iter().map(|w| w / sum_clamped).collect();
    trace::layer_1d("ffn_moe_weights_norm", layer, step, &normed);

    // ggml uses zero as the metadata sentinel for no additional scaling.
    let scale = if weights_scale == 0.0 {
        1.0
    } else {
        weights_scale
    };
    let scaled: Vec<f32> = normed.iter().map(|w| w * scale).collect();
    trace::layer_1d("ffn_moe_weights_scaled", layer, step, &scaled);

    Ok((chosen, scaled))
}

#[cfg(test)]
mod routing_tests {
    use super::route_experts;

    #[test]
    fn routing_preserves_tie_order_and_normalizes_selected_weights() {
        let (ids, weights) = route_experts(&[0.0, 1.0, 1.0, -2.0], 2, 1.5, 0, 0).unwrap();
        assert_eq!(ids, [1, 2]);
        assert_eq!(weights, [0.75, 0.75]);
        let (_, weights) = route_experts(&[0.0, 1.0, 1.0, -2.0], 2, 0.0, 0, 0).unwrap();
        assert_eq!(weights, [0.5, 0.5]);
        let (ids, weights) = route_experts(&[1000.0, 999.0, -1000.0], 2, 1.0, 0, 0).unwrap();
        assert_eq!(ids, [0, 1]);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        for (values, used) in [
            (&[][..], 1),
            (&[f32::NAN][..], 1),
            (&[0.0][..], 0),
            (&[0.0][..], 2),
        ] {
            assert!(route_experts(values, used, 1.0, 0, 0).is_err());
        }
    }
}
