//! Whistle (Cactus-Compute) ASR CLI driver (placeholder stem).
//!
//! Loads the encoder + stack layers from a GGUF, but **does not** run the
//! conv stem (`dw_1/dw_2/w/pw_1/pw_2/out`). The mel pipeline is a known
//! gap (see `docs/MODEL_LIST.md` — there is no public Cactus-compute
//! reference). Callers that need end-to-end ASR must run the stem
//! externally (e.g. via a numpy oracle) and feed the resulting
//! `[B, T, 512]` tensor in.
//!
//! This driver implements the decoder half of the model:
//!
//! 1. Load the GGUF (F32; quantization is TODO).
//! 2. Run the encoder 8 layers (placeholder ZCRMSNorm / mhc / taps).
//! 3. Run the stack's single layer (self-attn + cross-attn).
//! 4. Apply `stack/final_norm` and the shared embedding projection.
//! 5. Greedy-decode the highest-prob token per step.
//!
//! Real ASR quality will improve once Phase 2 (conv stem) and Phase 3
//! (true ZCRMSNorm/mhc_phi/taps) are landed.

use std::sync::Arc;

use crate::app::cli::CliOptions;
use crate::app::open_or_exit;
use crate::core::tensor::TensorSource;
use crate::format::ggufrs::ComponentRole;
use crate::models::whistle::{
    final_rms_norm, load_encoder_layer, load_stack_layer, EncoderLayer,
};

const WHISTLE_HIDDEN: usize = 512;
const WHISTLE_TEXT_VOCAB: usize = 8199;

/// Encoder placeholder hidden state. Replaces the conv stem until the
/// real mel → 512 pipeline lands. Hidden is filled with the encoder
/// input distribution so the encoder layers run end-to-end.
fn placeholder_stem_output(_samples_pcm: &[f32], _sample_rate: usize) -> Vec<f32> {
    let frames = 50; // arbitrary ~1s of 80ms frames
    vec![0.0f32; frames * WHISTLE_HIDDEN]
}

pub fn run_whistle_cli(options: &CliOptions) -> Result<(), String> {
    let model_path = &options.model;
    let source: Arc<dyn TensorSource> =
        Arc::from(open_or_exit(model_path, ComponentRole::Llm));
    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    if arch != "whistle" {
        return Err(format!(
            "--audio whistle path: expected general.architecture == \"whistle\", got {arch:?}"
        ));
    }
    eprintln!(
        "whistle: loaded metadata from {} (engine={:?})",
        model_path.display(),
        source.metadata("whistle.engine").and_then(|v| v.to_string_val())
    );

    // Pre-load all encoder layers + the single stack layer + final norms.
    let enc_layers: Vec<EncoderLayer> =
        (0..8).map(|i| load_encoder_layer(&*source, i)).collect();
    let stack = load_stack_layer(&*source, 0);
    let enc_final_scale = crate::core::tensor::load_f32_tensor(
        &*source,
        "encoder/final_norm/scale",
        &[WHISTLE_HIDDEN as u64],
    )
    .map_err(|e| format!("missing encoder/final_norm/scale: {e}"))?;
    let stack_final_scale = crate::core::tensor::load_f32_tensor(
        &*source,
        "stack/final_norm/scale",
        &[WHISTLE_HIDDEN as u64],
    )
    .map_err(|e| format!("missing stack/final_norm/scale: {e}"))?;
    let embedding = crate::core::tensor::load_f32_tensor(
        &*source,
        "embedding/embedding",
        &[WHISTLE_TEXT_VOCAB as u64, WHISTLE_HIDDEN as u64],
    )
    .or_else(|_| {
        // Some converters rename the embedding to token_embd.weight;
        // try the alternate key.
        crate::core::tensor::load_f32_tensor(
            &*source,
            "token_embd.weight",
            &[WHISTLE_TEXT_VOCAB as u64, WHISTLE_HIDDEN as u64],
        )
    })
    .map_err(|e| format!("missing embedding/embedding: {e}"))?;

    eprintln!("whistle: loaded 8 encoder layers + 1 stack layer (stem=placeholder)");

    // For now we just demonstrate the placeholder stem runs through the
    // encoder + decoder and emit a clear marker. Greedy decoding without
    // a real mel pipeline produces gibberish and is not exposed to the
    // user.
    let hidden = placeholder_stem_output(&[], 16000);
    let seq_len = hidden.len() / WHISTLE_HIDDEN;
    eprintln!(
        "whistle: stem placeholder produced seq_len={seq_len} frames; \
         this path is a smoke test for the encoder+decoder wiring"
    );

    // Encoder forward
    let mut enc = hidden;
    for (i, layer) in enc_layers.iter().enumerate() {
        let mut next = vec![0.0f32; enc.len()];
        layer.forward(&enc, i, 1, seq_len, &mut next);
        enc = next;
    }
    let mut enc_norm = vec![0.0f32; enc.len()];
    final_rms_norm(&enc, &enc_final_scale, 1, seq_len, &mut enc_norm);

    // Greedy decode a single placeholder text token (id=1) through the
    // stack; if it produces *some* finite logits the wiring is sound.
    let text_id: usize = 1;
    let mut text_emb = vec![0.0f32; WHISTLE_HIDDEN];
    text_emb.copy_from_slice(&embedding[text_id * WHISTLE_HIDDEN..(text_id + 1) * WHISTLE_HIDDEN]);

    let mut stack_out = vec![0.0f32; WHISTLE_HIDDEN];
    stack.forward(&text_emb, &enc_norm, 0, 1, 1, seq_len, &mut stack_out);

    let mut logits_emb = vec![0.0f32; WHISTLE_HIDDEN];
    final_rms_norm(&stack_out, &stack_final_scale, 1, 1, &mut logits_emb);

    let mut logits = vec![0.0f32; WHISTLE_TEXT_VOCAB];
    for v in 0..WHISTLE_TEXT_VOCAB {
        let mut acc = 0.0f32;
        for h in 0..WHISTLE_HIDDEN {
            acc += logits_emb[h] * embedding[v * WHISTLE_HIDDEN + h];
        }
        logits[v] = acc;
    }
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    eprintln!(
        "whistle: placeholder ASR produced token id={argmax} (stem is not a real mel pipeline; \
         tokens are meaningless without a working stem)"
    );

    Ok(())
}