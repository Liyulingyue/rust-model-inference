//! `nemotron_h_moe` routed + shared experts.
//!
//! Follows the upstream reference
//! (`src/models/nemotron-h.cpp::build_ffn_layer`, llama.cpp master):
//!
//! ```text
//! router_logits = ffn_gate_inp @ cur
//! moe_out       = build_moe_ffn(cur, ..., LLM_FFN_RELU_SQR,
//!                              norm=expert_weights_norm,
//!                              scale=expert_weights_scale, SIGMOID)
//! ffn_shexp     = build_ffn(cur, ffn_up_shexp, ffn_down_shexp,
//!                           LLM_FFN_RELU_SQR, LLM_FFN_PAR)
//! cur           = moe_out + ffn_shexp
//! ```
//!
//! Three things separate this from the Qwen3.5 MoE the repository already
//! has:
//!
//! 1. **The routing is a sigmoid, not a softmax**, and the selection bias
//!    (`exp_probs_b.bias`) takes part in *ranking* only. The routing weights
//!    are the **unbiased** sigmoid probabilities.
//! 2. **The experts are not gated.** There is no `ffn_gate_exps`, so the
//!    expert activation is `relu(x)^2` on the up projection alone
//!    (`LLM_FFN_RELU_SQR`).
//! 3. **The shared expert is not gated either.** There is no
//!    `ffn_gate_inp_shexp`, so the shared expert is added directly:
//!    `cur = moe_out + ffn_shexp`.
//!
//! The 30B has no latent projection (`moe_latent_size` is absent), so the
//! router, the experts and the shared expert all read the same normalized
//! hidden state.

use super::config::NemotronMoeConfig;
use super::forward::NemotronModel;
use super::trace;
use super::weights::NemotronMoeWeights;

/// `ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY)` — 2^-14, the
/// smallest number representable in F16. Not cosmetic: without it a token
/// whose chosen weights sum to ~0 divides by ~0 and the routed output
/// becomes non-finite.
const SUM_FLOOR: f32 = 6.103515625e-5;

/// Route one token to `used` experts.
///
/// Returns the chosen expert ids and the routing weight of each, already
/// normalized and scaled.
///
/// The routing decision itself, as integers, is what a claim about "the
/// same experts were selected" has to mean: the same ids, not merely a
/// plausible-looking output.
///
/// `layer` and `step` key the `RMI_PARITY_TRACE` checkpoints. They are
/// positional: without them every record lands in the same append-mode files
/// and none can be attributed.
pub(crate) fn route_experts(
    logits: &[f32],
    selection_bias: &[f32],
    used: usize,
    weights_norm: bool,
    weights_scale: f32,
    layer: usize,
    step: usize,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    if logits.is_empty() || used == 0 || used > logits.len() {
        return Err("invalid nemotron_h_moe routing dimensions".into());
    }
    if selection_bias.len() != logits.len() {
        return Err("nemotron_h_moe selection bias length does not match experts".into());
    }
    if logits.iter().any(|v| !v.is_finite()) {
        return Err("nemotron_h_moe router produced non-finite logits".into());
    }
    if !weights_scale.is_finite() {
        return Err("nemotron_h_moe expert weights scale is not finite".into());
    }
    trace::layer_1d("ffn_moe_logits", layer, step, logits);

    // `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`: probs = sigmoid(logits).
    let probs: Vec<f32> = logits
        .iter()
        .map(|&logit| 1.0 / (1.0 + (-logit).exp()))
        .collect();
    trace::layer_1d("ffn_moe_probs", layer, step, &probs);

    // The selection bias only takes part in the ranking; the weights
    // themselves stay unbiased.
    let selection: Vec<f32> = probs
        .iter()
        .zip(selection_bias)
        .map(|(&prob, &bias)| prob + bias)
        .collect();

    // Descending by selection score, ties by ascending index:
    // `ggml_argsort_top_k` leaves equal keys in index order.
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| {
        selection[b]
            .total_cmp(&selection[a])
            .then_with(|| a.cmp(&b))
    });
    order.truncate(used.min(logits.len()));
    let chosen = order;

    // The routing decision itself, as integers. This is the checkpoint that can
    // falsify a claim: "the same experts were selected" has to mean the same
    // ids in the same order, not merely a plausible-looking output.
    let chosen_ids: Vec<u32> = chosen.iter().map(|&i| i as u32).collect();
    trace::ids("ffn_moe_topk", layer, step, &chosen_ids);

    // The routing weights are the *unbiased* probabilities of the chosen
    // experts.
    let gathered: Vec<f32> = chosen.iter().map(|&i| probs[i]).collect();
    trace::layer_1d("ffn_moe_weights", layer, step, &gathered);

    // `ggml_sum_rows` rounds its F64 accumulator to F32.
    let sum = gathered.iter().map(|&v| f64::from(v)).sum::<f64>() as f32;
    trace::layer_1d("ffn_moe_weights_sum", layer, step, &[sum]);
    let sum_clamped = sum.max(SUM_FLOOR);
    trace::layer_1d(
        "ffn_moe_weights_sum_clamped",
        layer,
        step,
        &[sum_clamped],
    );

    let normed: Vec<f32> = if weights_norm {
        gathered.iter().map(|w| w / sum_clamped).collect()
    } else {
        gathered
    };
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

/// Routed + shared expert output for one token, written into `out`.
///
/// `input` is the normalized hidden state: llama.cpp's
/// `build_ffn_layer` receives the block input *after* the layer's
/// `attn_norm`, and both the router and the experts read that same
/// tensor (`inp_emb` and `inp_latent` are the same value when there is
/// no latent projection).
pub(crate) fn forward_moe_token(
    model: &NemotronModel,
    moe: &NemotronMoeWeights<'_>,
    cfg: &NemotronMoeConfig,
    input: &[f32],
    out: &mut [f32],
    q8_buf: &mut [u8],
    scale_buf: &mut [f32],
    layer: usize,
    step: usize,
) -> Result<(), String> {
    let n_embd = input.len();
    // The normalized hidden state the router and the experts read. The
    // oracle has no callback node for it, so it is Rust-only; it is kept
    // because it is the one checkpoint that separates "the router saw a
    // different input" from "the router arithmetic is wrong".
    trace::layer_1d("ffn_in", layer, step, input);
    // The router, the experts and the shared expert all read the same
    // activation, so it is quantized once.
    let blocks = n_embd.div_ceil(32);
    crate::ops::quantize_q8_0_into(
        input,
        n_embd,
        &mut q8_buf[..n_embd],
        &mut scale_buf[..blocks],
    );

    // The quantized block input lives in `q8_buf[..n_embd]`. Each expert
    // quantizes its own intermediate activation *after* that region: the
    // 30B's expert activation (1856) is narrower than the block input
    // (2688), so reusing the same bytes would clobber the input the next
    // expert still needs to read. `split_at_mut` keeps the two regions
    // disjoint in the type system, not just by convention.
    let (input_q8, interm_q8) = q8_buf.split_at_mut(n_embd);
    let (input_sc, interm_sc) = scale_buf.split_at_mut(blocks);

    let mut logits = vec![0.0f32; moe.n_expert];
    {
        model.run_matmul(
            &*moe.gate_inp.kernel,
            input,
            input_q8,
            input_sc,
            &mut logits,
            n_embd,
            moe.n_expert,
        );
    }

    let (chosen, scaled) = route_experts(
        &logits,
        &moe.exp_probs_b,
        cfg.n_expert_used,
        cfg.expert_weights_norm,
        cfg.expert_weights_scale,
        layer,
        step,
    )?;

    // The routed term alone, before the shared expert is added.
    let mut routed = vec![0.0f32; n_embd];
    let mut up = vec![0.0f32; moe.n_ff_exp];
    let mut down = vec![0.0f32; n_embd];
    let up_blocks = moe.n_ff_exp.div_ceil(32);
    for (slot, &expert) in chosen.iter().enumerate() {
        let up_w = moe.up(expert);
        let down_w = moe.down(expert);
        {
            model.run_matmul(
                &*up_w.kernel,
                input,
                input_q8,
                input_sc,
                &mut up,
                n_embd,
                moe.n_ff_exp,
            );
        }
        // `LLM_FFN_RELU_SQR`: relu then square. Not gated — there is no
        // `ffn_gate_exps` in this architecture.
        for value in up.iter_mut() {
            let x = *value;
            *value = if x > 0.0 { x * x } else { 0.0 };
        }
        crate::ops::quantize_q8_0_into(
            &up,
            moe.n_ff_exp,
            &mut interm_q8[..moe.n_ff_exp],
            &mut interm_sc[..up_blocks],
        );
        {
            model.run_matmul(
                &*down_w.kernel,
                &up,
                &interm_q8[..moe.n_ff_exp],
                &interm_sc[..up_blocks],
                &mut down,
                moe.n_ff_exp,
                n_embd,
            );
        }
        let score = scaled[slot];
        for (o, v) in routed.iter_mut().zip(&down) {
            *o += score * v;
        }
    }

    // The routed term alone, before the shared expert is added. This is the
    // checkpoint that separates a routing bug from an expert-projection bug:
    // if the top-k ids and weights match but this does not, the experts are
    // wrong; if this matches but `ffn_out` does not, the shared expert is.
    trace::layer_1d("ffn_moe_out", layer, step, &routed);

    // The shared expert. `LLM_FFN_PAR` here means "the gate projection
    // reads the block input", but this architecture has no gate at all,
    // so the shared expert is `down(relu(up @ x)^2)`.
    out.copy_from_slice(&routed);
    let mut shexp_up = vec![0.0f32; cfg.n_ff_shexp];
    let mut shared = vec![0.0f32; n_embd];
    {
        model.run_matmul(
            &*moe.up_shexp.kernel,
            input,
            input_q8,
            input_sc,
            &mut shexp_up,
            n_embd,
            cfg.n_ff_shexp,
        );
    }
    // The shared expert's up projection, before the ReLU. Rust-only (the
    // oracle has no callback node for it); kept because the ReLU is
    // `x*x` for `x > 0` and a wrong up projection is otherwise only
    // visible through the down projection's quantized input.
    trace::layer_1d("ffn_shexp_up", layer, step, &shexp_up);
    for value in shexp_up.iter_mut() {
        let x = *value;
        *value = if x > 0.0 { x * x } else { 0.0 };
    }
    trace::layer_1d("ffn_shexp_act", layer, step, &shexp_up);
    let shexp_blocks = cfg.n_ff_shexp.div_ceil(32);
    crate::ops::quantize_q8_0_into(
        &shexp_up,
        cfg.n_ff_shexp,
        &mut interm_q8[..cfg.n_ff_shexp],
        &mut interm_sc[..shexp_blocks],
    );
    {
        model.run_matmul(
            &*moe.down_shexp.kernel,
            &shexp_up,
            &interm_q8[..cfg.n_ff_shexp],
            &interm_sc[..shexp_blocks],
            &mut shared,
            cfg.n_ff_shexp,
            n_embd,
        );
    }
    trace::layer_1d("ffn_shexp", layer, step, &shared);
    for (o, v) in out.iter_mut().zip(&shared) {
        *o += v;
    }
    trace::layer_1d("ffn_out", layer, step, out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn routing_uses_the_unbiased_probabilities() {
        // Two experts with clearly different logits, and a bias that
        // would flip the ranking if it were applied to the weights.
        let logits = [2.0f32, 0.0];
        let bias = [-5.0f32, 0.0];
        let (chosen, weights) = route_experts(&logits, &bias, 1, true, 1.0, 0, 0).unwrap();
        // sigmoid(2) = 0.8808, sigmoid(0) = 0.5. With the bias the
        // second expert wins the ranking (0.5 > 0.8808 - 5).
        assert_eq!(chosen, vec![1]);
        // The weight, however, is the unbiased sigmoid of the chosen
        // expert: 0.5, normalized by itself.
        assert!(approx(weights[0], 1.0), "{weights:?}");
    }

    #[test]
    fn routing_normalizes_and_scales() {
        // sigmoid values are ~0.731 and ~0.881.
        let logits = [1.0f32, 2.0];
        let bias = [0.0f32, 0.0];
        let (chosen, weights) = route_experts(&logits, &bias, 2, true, 2.5, 0, 0).unwrap();
        assert_eq!(chosen, vec![1, 0]);
        let sum: f32 = weights.iter().sum();
        assert!(approx(sum, 2.5), "weights {weights:?} sum {sum}");
    }

    #[test]
    fn routing_clamps_the_weight_sum() {
        // Two experts with hugely negative logits: the sigmoid
        // probabilities underflow to ~0, so the sum would divide by ~0.
        let logits = [-100.0f32, -101.0];
        let bias = [0.0f32, 0.0];
        let (_, weights) = route_experts(&logits, &bias, 2, true, 1.0, 0, 0).unwrap();
        assert!(weights.iter().all(|w| w.is_finite()), "{weights:?}");
    }

    #[test]
    fn routing_ties_break_by_ascending_index() {
        let logits = [1.0f32; 4];
        let bias = [0.0f32; 4];
        let (chosen, _) = route_experts(&logits, &bias, 2, true, 1.0, 0, 0).unwrap();
        assert_eq!(chosen, vec![0, 1]);
    }

    #[test]
    fn routing_rejects_a_mismatched_bias() {
        assert!(route_experts(&[0.0, 0.0], &[0.0], 1, true, 1.0, 0, 0).is_err());
    }
}
