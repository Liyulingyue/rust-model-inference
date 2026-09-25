use super::config::{Gemma4Config, VOCAB};
use crate::ops::kernel::PreparedRows;

pub(super) struct Gemma4Scratch {
    pub(super) x: Vec<f32>,
    pub(super) normed: Vec<f32>,
    pub(super) q: Vec<f32>,
    pub(super) k: Vec<f32>,
    pub(super) v: Vec<f32>,
    pub(super) attn: Vec<f32>,
    pub(super) projected: Vec<f32>,
    pub(super) gate: Vec<f32>,
    pub(super) up: Vec<f32>,
    pub(super) down: Vec<f32>,
    pub(super) per_layer: Vec<f32>,
    pub(super) per_layer_projected: Vec<f32>,
    pub(super) per_layer_gate: Vec<f32>,
    pub(super) q8: Vec<u8>,
    pub(super) scales: Vec<f32>,
    pub(super) scores: Vec<f32>,
    pub(super) attention_values: Vec<f32>,
    pub(super) v_norm_weight: Vec<f32>,
    pub(super) logits: Vec<f32>,
    pub(super) prepared: PreparedRows,
    // MoE scratch (empty for dense models).
    pub(super) moe_gate_up: Vec<f32>,
    pub(super) moe_down: Vec<f32>,
    pub(super) moe_logits: Vec<f32>,
    pub(super) moe_normed: Vec<f32>,
    pub(super) moe_q8: Vec<u8>,
    pub(super) moe_scales: Vec<f32>,
}

impl Gemma4Scratch {
    pub(super) fn new(cfg: &Gemma4Config, max_rows: usize) -> Self {
        let embd = cfg.embd;
        let max_ffn = cfg.max_ffn();
        let per_layer_all = cfg.per_layer_all();
        let max_kv_width = cfg.max_kv_width();
        let max_q_width = cfg.max_q_width();
        let max_input = max_ffn.max(max_q_width);
        // Score / value buffers are bounded by the sliding-window length
        // (512 for E2B/E4B, 1024 for 12B) when SWA is on, or full KV
        // context otherwise. Allocate per the model's sliding window so
        // per-decode `resize` is a no-op and we don't grow during
        // prefill (where padded length steps up to 256).
        let max_attn_buffer = cfg.sliding_window.next_power_of_two();
        let per_layer_gate_width = if cfg.use_per_layer_projection() {
            cfg.per_layer_width
        } else {
            0
        };
        Self {
            x: vec![0.0; max_rows * embd],
            normed: vec![0.0; max_rows * embd],
            q: vec![0.0; max_rows * max_q_width],
            k: vec![0.0; max_rows * max_kv_width],
            v: vec![0.0; max_rows * max_kv_width],
            attn: vec![0.0; max_rows * max_q_width],
            projected: vec![0.0; max_rows * embd],
            gate: vec![0.0; max_rows * max_ffn],
            up: vec![0.0; max_rows * max_ffn],
            down: vec![0.0; max_rows * embd],
            per_layer: vec![0.0; max_rows * per_layer_all],
            per_layer_projected: vec![0.0; max_rows * per_layer_all],
            per_layer_gate: vec![0.0; max_rows * per_layer_gate_width],
            q8: vec![0; max_input],
            scales: vec![0.0; max_input.div_ceil(32)],
            scores: vec![f32::NEG_INFINITY; max_attn_buffer],
            attention_values: vec![0.0; max_attn_buffer],
            v_norm_weight: vec![1.0; cfg.full_head_dim],
            logits: vec![0.0; VOCAB],
            prepared: PreparedRows::new(max_rows, max_input.max(embd)),
            moe_gate_up: if cfg.is_moe() {
                vec![0.0; max_rows * cfg.n_expert_used * cfg.n_ff_exp * 2]
            } else {
                Vec::new()
            },
            moe_down: if cfg.is_moe() {
                vec![0.0; max_rows * embd]
            } else {
                Vec::new()
            },
            moe_logits: if cfg.is_moe() {
                vec![0.0; cfg.n_expert]
            } else {
                Vec::new()
            },
            moe_normed: if cfg.is_moe() {
                vec![0.0; max_rows * embd]
            } else {
                Vec::new()
            },
            moe_q8: if cfg.is_moe() {
                vec![0; embd.max(cfg.n_ff_exp * 2)]
            } else {
                Vec::new()
            },
            moe_scales: if cfg.is_moe() {
                vec![0.0; (embd.max(cfg.n_ff_exp * 2)).div_ceil(32)]
            } else {
                Vec::new()
            },
        }
    }

    pub(super) fn bytes(&self) -> usize {
        let f32_values = self.x.len()
            + self.normed.len()
            + self.q.len()
            + self.k.len()
            + self.v.len()
            + self.attn.len()
            + self.projected.len()
            + self.gate.len()
            + self.up.len()
            + self.down.len()
            + self.per_layer.len()
            + self.per_layer_projected.len()
            + self.per_layer_gate.len()
            + self.scales.len()
            + self.scores.capacity()
            + self.attention_values.capacity()
            + self.v_norm_weight.len()
            + self.logits.len()
            + self.moe_gate_up.len()
            + self.moe_down.len()
            + self.moe_logits.len()
            + self.moe_normed.len()
            + self.moe_scales.len();
        f32_values * std::mem::size_of::<f32>()
            + self.q8.len()
            + self.prepared.bytes()
            + self.moe_q8.len()
    }
}
