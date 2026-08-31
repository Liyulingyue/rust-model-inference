use super::config::{EMBED, FULL_HEAD_DIM, HEADS, MAX_FFN, PER_LAYER, PER_LAYER_ALL, VOCAB};

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
}

impl Gemma4Scratch {
    pub(super) fn new() -> Self {
        let max_input = MAX_FFN.max(HEADS * FULL_HEAD_DIM);
        Self {
            x: vec![0.0; EMBED],
            normed: vec![0.0; EMBED],
            q: vec![0.0; HEADS * FULL_HEAD_DIM],
            k: vec![0.0; FULL_HEAD_DIM],
            v: vec![0.0; FULL_HEAD_DIM],
            attn: vec![0.0; HEADS * FULL_HEAD_DIM],
            projected: vec![0.0; EMBED],
            gate: vec![0.0; MAX_FFN],
            up: vec![0.0; MAX_FFN],
            down: vec![0.0; EMBED],
            per_layer: vec![0.0; PER_LAYER_ALL],
            per_layer_projected: vec![0.0; PER_LAYER_ALL],
            per_layer_gate: vec![0.0; PER_LAYER],
            q8: vec![0; max_input],
            scales: vec![0.0; max_input.div_ceil(32)],
            scores: Vec::new(),
            attention_values: Vec::new(),
            v_norm_weight: vec![1.0; FULL_HEAD_DIM],
            logits: vec![0.0; VOCAB],
        }
    }
}
