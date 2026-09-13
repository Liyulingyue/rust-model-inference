use super::config::{Gemma4Config, FULL_HEAD_DIM, HEADS, PER_LAYER, VOCAB};
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
}

impl Gemma4Scratch {
    pub(super) fn new(cfg: &Gemma4Config, max_rows: usize) -> Self {
        let embd = cfg.embd;
        let max_ffn = cfg.max_ffn();
        let per_layer_all = cfg.per_layer_all();
        let max_kv_width = cfg.kv_heads * FULL_HEAD_DIM;
        let max_input = max_ffn.max(HEADS * FULL_HEAD_DIM);
        // Score / value buffers are bounded by the sliding-window length
        // (512 for Gemma4) when SWA is on, or full KV context otherwise.
        // Allocate 512 once so per-decode `resize` is a no-op and we don't
        // grow during prefill (where padded length steps up to 256).
        let max_attn_buffer = 512usize.next_power_of_two();
        Self {
            x: vec![0.0; max_rows * embd],
            normed: vec![0.0; max_rows * embd],
            q: vec![0.0; max_rows * HEADS * FULL_HEAD_DIM],
            k: vec![0.0; max_rows * max_kv_width],
            v: vec![0.0; max_rows * max_kv_width],
            attn: vec![0.0; max_rows * HEADS * FULL_HEAD_DIM],
            projected: vec![0.0; max_rows * embd],
            gate: vec![0.0; max_rows * max_ffn],
            up: vec![0.0; max_rows * max_ffn],
            down: vec![0.0; max_rows * embd],
            per_layer: vec![0.0; max_rows * per_layer_all],
            per_layer_projected: vec![0.0; max_rows * per_layer_all],
            per_layer_gate: vec![0.0; max_rows * PER_LAYER],
            q8: vec![0; max_input],
            scales: vec![0.0; max_input.div_ceil(32)],
            scores: vec![f32::NEG_INFINITY; max_attn_buffer],
            attention_values: vec![0.0; max_attn_buffer],
            v_norm_weight: vec![1.0; FULL_HEAD_DIM],
            logits: vec![0.0; VOCAB],
            prepared: PreparedRows::new(max_rows, max_input.max(embd)),
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
            + self.scores.len()
            + self.attention_values.len()
            + self.v_norm_weight.len()
            + self.logits.len();
        f32_values * std::mem::size_of::<f32>() + self.q8.len() + self.prepared.bytes()
    }
}
