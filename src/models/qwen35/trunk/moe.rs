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
use super::weights::Qwen35MoeWeights;
use crate::core::thread_pool::ComputePool;
use crate::ops::activation::silu_inplace;
use crate::ops::kernel::PreparedRows;

/// Routed + shared expert output for one token, written into `out`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_qwen35moe_token(
    moe: &Qwen35MoeWeights<'_>,
    cfg: &Qwen35Moe,
    weights_scale: f32,
    input: &[f32],
    out: &mut [f32],
    prepared: &mut PreparedRows,
    pool: &ComputePool,
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
    if logits.iter().any(|v| !v.is_finite()) {
        return Err("qwen35moe router produced non-finite logits".into());
    }

    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|v| (*v - max).exp()).collect();
    let total = exp.iter().sum::<f32>();
    let probabilities: Vec<f32> = exp.iter().map(|v| v / total).collect();

    // Descending by probability, ties by ascending index: ggml_argsort_top_k
    // leaves equal keys in index order.
    let mut order: Vec<usize> = (0..moe.n_expert).collect();
    order.sort_by(|&a, &b| {
        probabilities[b]
            .total_cmp(&probabilities[a])
            .then_with(|| a.cmp(&b))
    });
    let chosen = &order[..cfg.n_expert_used.min(moe.n_expert)];
    let chosen_total: f32 = chosen.iter().map(|&i| probabilities[i]).sum();

    // RMI_PARITY: dump the router / routed / shared terms for layer-0 token 0.
    let dump = std::env::var("RMI_MOE_DUMP").ok();

    let mut routed = vec![0.0f32; n_embd];
    for &expert in chosen {
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
        let score = weights_scale * probabilities[expert] / chosen_total;
        for (o, v) in routed.iter_mut().zip(down) {
            *o += score * v;
        }
    }

    // Shared expert. With LLM_FFN_PAR the gate projection reads `input`
    // directly, so this is `silu(gate @ input) * (up @ input)`.
    out.copy_from_slice(&routed);
    let mut shared_acc = vec![0.0f32; n_embd];
    let (Some(up_shexp), Some(gate_shexp), Some(down_shexp)) = (
        moe.up_shexp.as_ref(),
        moe.gate_shexp.as_ref(),
        moe.down_shexp.as_ref(),
    ) else {
        out.copy_from_slice(&routed);
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

    let scale = 1.0 / (1.0 + (-shared_gate[0]).exp());
    // cur = moe_out + ffn_shexp * sigmoid(gate_inp_shexp . cur).
    for (o, shared) in out.iter_mut().zip(&shared_acc) {
        *o += scale * shared;
    }
    if let Some(dir) = &dump {
        if cfg.n_expert == 256 && cfg.n_expert_used == 8 {
            let w = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
            use std::io::Write as _;
            let mut open = |name: &str| {
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(format!("{dir}/{name}"))
            };
            if let Ok(mut f) = open("router.bin") {
                let _ = f.write_all(&w(&logits));
            }
            if let Ok(mut f) = open("routed.bin") {
                let _ = f.write_all(&w(&routed));
            }
            if let Ok(mut f) = open("shared.bin") {
                let _ = f.write_all(&w(&shared_acc));
            }
            if let Ok(mut f) = open("out.bin") {
                let _ = f.write_all(&w(out));
            }
            if let Ok(mut f) = open("scale.bin") {
                let _ = f.write_all(&w(&[scale]));
            }
        }
    }
    Ok(())
}
