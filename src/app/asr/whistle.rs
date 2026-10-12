//! Whistle CLI driver (pure-Rust port).
//!
//! Runs the encoder + decoder logits we have lifted to
//! `models::whistle` (ZCRMSNorm + HadamardMLP + multi-lane MHC at 1 ULP
//! parity with the in-tree Python Oracle). The conv stem is still a
//! placeholder, so this path is a smoke test for the encoder+decoder
//! wiring rather than a real ASR — output is a placeholder token id, not
//! a transcript.
//!
//! For a real transcript today, the closed Cactus runtime
//! (`libneedle3.so`) has to be invoked from a separate dev-only tool
//! at `tools/needle_ffi/main.rs` (not part of this crate). Build it
//! once with `rustc` and pipe the JSON through `jq .text`. See
//! `tools/needle_ffi/README.md` for the workflow.

use std::sync::Arc;

use crate::app::open_or_exit;
use crate::core::tensor::TensorSource;
use crate::format::ggufrs::ComponentRole;
use crate::models::whistle::{
    final_rms_norm, load_encoder_layer, load_mhc_layer, EncoderLayerWeights, MHC_LANES,
};

use crate::app::cli::CliOptions;

const WHISTLE_HIDDEN: usize = 512;

pub fn run_whistle_cli(options: &CliOptions) -> Result<(), String> {
    // The flag wiring lives in main.rs; the dispatcher in src/app/asr/mod.rs
    // hands us a `CliOptions` already decided. We re-check the
    // architecture just to give a clear error if the user mis-routed.
    let model_path = &options.model;
    let source: Arc<dyn TensorSource> = Arc::from(open_or_exit(model_path, ComponentRole::Llm));
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
        "whistle (pure Rust): loading metadata from {} (engine={:?})",
        model_path.display(),
        source
            .metadata("whistle.engine")
            .and_then(|v| v.to_string_val())
    );

    let enc_layers: Vec<_> = (0..8).map(|i| load_encoder_layer(&*source, i)).collect();
    let mhc = load_mhc_layer(&*source);
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
        &[8199, WHISTLE_HIDDEN as u64],
    )
    .or_else(|_| {
        crate::core::tensor::load_f32_tensor(
            &*source,
            "token_embd.weight",
            &[8199, WHISTLE_HIDDEN as u64],
        )
    })
    .map_err(|e| format!("missing embedding/embedding: {e}"))?;

    eprintln!(
        "whistle: loaded 8 encoder layers + 1 stack layer (stem=placeholder); this path is a smoke test for the encoder+decoder wiring"
    );

    let hidden = vec![0.0f32; 50 * WHISTLE_HIDDEN];
    let seq_len = hidden.len() / WHISTLE_HIDDEN;
    eprintln!(
        "whistle: stem placeholder produced seq_len={seq_len} frames"
    );

    // Cumulative multi-lane stream
    let plane_4 = seq_len * MHC_LANES * WHISTLE_HIDDEN;
    let mut stream: Vec<f32> = hidden
        .chunks(WHISTLE_HIDDEN)
        .flat_map(|chunk| (0..MHC_LANES).map(move |_| chunk.to_vec()).flatten())
        .collect();
    if stream.len() != plane_4 {
        return Err(format!(
            "stream init mismatch: got {} expected {plane_4}",
            stream.len()
        ));
    }
    for i in 0..8 {
        let layer = &enc_layers[i];
        let layer_w = EncoderLayerWeights {
            zc_in_scale: &layer.zc_in_scale,
            post_attn_scale: &layer.post_attn_scale,
            q_proj: &layer.q_proj,
            k_proj: &layer.k_proj,
            v_proj: &layer.v_proj,
            out_proj: &layer.out_proj,
            q_norm_scale: &layer.q_norm_scale,
            k_norm_scale: &layer.k_norm_scale,
            attn_gate: &layer.attn_gate,
            hadamard: layer.hadamard.clone(),
        };
        let mut next_stream = vec![0.0f32; plane_4];
        crate::models::whistle::mhc::forward_one_stream(
            &layer_w,
            &mhc,
            &stream,
            i,
            1,
            seq_len,
            &mut next_stream,
        );
        stream = next_stream;
    }
    let mut enc = vec![0.0f32; seq_len * WHISTLE_HIDDEN];
    for bt in 0..seq_len {
        for h in 0..WHISTLE_HIDDEN {
            let mut sum = 0.0f32;
            for lane in 0..MHC_LANES {
                sum += stream[bt * MHC_LANES * WHISTLE_HIDDEN + lane * WHISTLE_HIDDEN + h];
            }
            enc[bt * WHISTLE_HIDDEN + h] = sum / MHC_LANES as f32;
        }
    }

    let mut enc_norm = vec![0.0f32; enc.len()];
    final_rms_norm(&enc, &enc_final_scale, 1, seq_len, &mut enc_norm);

    // Pick the BOS token (id=1, used as prompt for the decoder).
    let mut text_emb = vec![0.0f32; WHISTLE_HIDDEN];
    text_emb.copy_from_slice(&embedding[1 * WHISTLE_HIDDEN..(1 + 1) * WHISTLE_HIDDEN]);
    let stack = crate::models::whistle::load_stack_layer(&*source, 0);
    let mut stack_out = vec![0.0f32; WHISTLE_HIDDEN];
    stack.forward(&text_emb, &enc_norm, 0, 1, 1, seq_len, &mut stack_out);

    let mut logits_emb = vec![0.0f32; WHISTLE_HIDDEN];
    final_rms_norm(&stack_out, &stack_final_scale, 1, 1, &mut logits_emb);

    let mut logits = vec![0.0f32; 8199];
    for v in 0..8199 {
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
        "whistle (pure Rust): placeholder ASR produced token id={argmax} (stem is not a real mel pipeline; see tools/needle_ffi/ for the closed-runtime dev path)"
    );

    Ok(())
}
