use crate::app::cli::resolve_thread_count;
use crate::app::media::{decode_image, normalize_resized_image};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::models::qwen35::vision::{
    qwen_smart_resize, VisionEncoder, VisionGrid, VisionScratchpad,
};
use crate::models::qwen35::Qwen35Model;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

pub(crate) fn encode_qwen35_image(
    mmproj_source: &dyn TensorSource,
    image_path: &Path,
    n_threads_arg: usize,
) -> Result<(VisionGrid, Vec<f32>), String> {
    let image = crate::app::media::decode_image(image_path)?;
    encode_qwen35_image_dynamic(mmproj_source, &image, n_threads_arg)
}

/// Encode one image with the RADIO projector (ZDTaichu5.0-9B).
///
/// The projector is a RADIOv4-H tower plus a ReLU-squared MLP head, so the
/// input is the model's own `image_size` square regardless of the aspect
/// ratio, and the output is a `grid x grid` run of projected tokens: the
/// patches are merged by `scale_factor` before the head runs, so the grid
/// that reaches the language tower is already the merged one.
pub(crate) fn encode_nemotron_v2_vl_image(
    mmproj_source: &dyn TensorSource,
    image: &image::DynamicImage,
    n_threads_arg: usize,
) -> Result<(VisionGrid, Vec<f32>), String> {
    let encoder = crate::models::zdt::vision::ZdtVisionEncoder::from_source(
        mmproj_source,
        Arc::new(ComputePool::new(n_threads_arg.max(1))),
    )
    .map_err(|error| format!("Failed to parse ZDT vision encoder: {error}"))?;
    let size = encoder.config.image_size;
    let normalized = normalize_resized_image(
        image,
        size,
        size,
        &encoder.config.image_mean,
        &encoder.config.image_std,
    )?;
    let projected = encoder.encode(&normalized, size, size)?;
    // After the shuffle the grid is already downsampled by the scale factor,
    // so report the post-merge extents; otherwise `token_count` would promise
    // 32x32 placeholders while the projector emits 16x16.
    let grid = size / (encoder.config.patch_size * encoder.config.scale_factor);
    let vision_grid = VisionGrid {
        grid_t: 1,
        grid_h: grid,
        grid_w: grid,
        patch_size: encoder.config.patch_size,
        merge_size: 1,
    };
    let per_token = encoder.config.projection_dim.max(1);
    if projected.len() % per_token != 0 {
        return Err(format!(
            "ZDT projector output {} is not a multiple of {per_token}",
            projected.len()
        ));
    }
    if projected.len() / per_token != vision_grid.token_count() {
        return Err(format!(
            "ZDT projector emitted {} tokens, expected {}",
            projected.len() / per_token,
            vision_grid.token_count()
        ));
    }
    Ok((vision_grid, projected))
}

/// Encode an already-decoded image. Split out of [`encode_qwen35_image`] so
/// callers that hold bytes (the HTTP image-content blocks: base64 / data URI /
/// in-memory uploads) do not have to round-trip through a temp file.
pub(crate) fn encode_qwen35_image_dynamic(
    mmproj_source: &dyn TensorSource,
    image: &image::DynamicImage,
    n_threads_arg: usize,
) -> Result<(VisionGrid, Vec<f32>), String> {
    let start = Instant::now();
    let mut encoder = VisionEncoder::from_source(
        mmproj_source,
        std::sync::Arc::new(crate::core::thread_pool::ComputePool::new(
            n_threads_arg.max(1),
        )),
    )
    .map_err(|error| format!("Failed to parse vision encoder: {error}"))?;
    encoder.precompute();
    eprintln!(
        "Vision encoder loaded: {} layers, n_embd={}, image_size={}, patch_size={}, merge={}",
        encoder.config.n_layer,
        encoder.config.n_embd,
        encoder.config.image_size,
        encoder.config.patch_size,
        encoder.config.spatial_merge_size
    );
    let original_w =
        usize::try_from(image.width()).map_err(|_| "Original image width does not fit usize")?;
    let original_h =
        usize::try_from(image.height()).map_err(|_| "Original image height does not fit usize")?;
    let grid = qwen_smart_resize(original_w, original_h, &encoder.config)?;
    let preprocess_start = Instant::now();
    let pixels = normalize_resized_image(
        &image,
        grid.image_width(),
        grid.image_height(),
        &encoder.config.image_mean,
        &encoder.config.image_std,
    )?;
    let preprocess_time = preprocess_start.elapsed();
    eprintln!(
        "Image resized to {}x{} ({} vision tokens)",
        grid.image_width(),
        grid.image_height(),
        grid.token_count()
    );
    let n_threads = resolve_thread_count(
        n_threads_arg,
        std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1),
    );
    let pool = Arc::new(ComputePool::new(n_threads));
    let mut scratch = VisionScratchpad::new(&encoder.config);
    eprintln!("Encoding image...");
    let encode_start = Instant::now();
    let encoded_grid = encoder.encode_image(
        &pixels,
        grid.image_width(),
        grid.image_height(),
        &mut scratch,
        &pool,
    )?;
    let encode_time = encode_start.elapsed();
    if encoded_grid != grid {
        return Err(format!(
            "Vision grid mismatch: preprocess={grid:?}, encoder={encoded_grid:?}"
        ));
    }
    let projected_len = grid
        .token_count()
        .checked_mul(encoder.config.projection_dim)
        .ok_or("Projected vision length overflow")?;
    if scratch.projected.len() != projected_len {
        return Err(format!(
            "Projected vision length mismatch: expected {projected_len}, got {}",
            scratch.projected.len()
        ));
    }
    let total = start.elapsed();
    eprintln!(
        "[pipeline-timing] image_total={:.3}s  preprocess={:.3}s ({:.0}%)  vision_encode={:.3}s ({:.0}%)",
        total.as_secs_f64(),
        preprocess_time.as_secs_f64(),
        preprocess_time.as_secs_f64() / total.as_secs_f64() * 100.0,
        encode_time.as_secs_f64(),
        encode_time.as_secs_f64() / total.as_secs_f64() * 100.0,
    );
    eprintln!(
        "Vision tokens: {} (dim={})",
        grid.token_count(),
        encoder.config.projection_dim
    );
    Ok((grid, scratch.projected))
}

pub fn inject_vision_embeddings(
    llm: &Qwen35Model,
    tokens: &[i32],
    image_token_id: Option<i32>,
    vis_embd: &[f32],
    n_vis_tokens: usize,
    proj_dim: usize,
) -> Result<Vec<f32>, String> {
    let n_embd = llm.config.n_embd;
    let expected_vis_len = n_vis_tokens
        .checked_mul(proj_dim)
        .ok_or("Qwen3.5 vision embedding length overflow")?;
    let placeholders = tokens
        .iter()
        .filter(|&&token| image_token_id == Some(token))
        .count();
    if proj_dim != n_embd || vis_embd.len() != expected_vis_len || placeholders != n_vis_tokens {
        return Err(format!(
            "Qwen3.5 vision embedding mismatch: placeholders={placeholders}, rows={n_vis_tokens}, projection_dim={proj_dim}, model_dim={n_embd}, values={}",
            vis_embd.len()
        ));
    }
    let token_ids = tokens
        .iter()
        .copied()
        .filter(|&token| image_token_id != Some(token))
        .map(|token| u32::try_from(token).map_err(|_| format!("invalid negative token id {token}")))
        .collect::<Result<Vec<_>, _>>()?;
    let text_embeddings = llm.embed_tokens(&token_ids)?;
    let mut embeddings = vec![0.0; tokens.len() * n_embd];
    let (mut text_idx, mut vis_idx) = (0, 0);
    for (token, row) in tokens.iter().zip(embeddings.chunks_exact_mut(n_embd)) {
        if image_token_id == Some(*token) {
            row.copy_from_slice(&vis_embd[vis_idx * n_embd..(vis_idx + 1) * n_embd]);
            vis_idx += 1;
        } else {
            row.copy_from_slice(&text_embeddings[text_idx * n_embd..(text_idx + 1) * n_embd]);
            text_idx += 1;
        }
    }
    Ok(embeddings)
}
pub use crate::models::qwen3::vision::media::{
    build_qwen3_media_positions, inject_qwen_media_embeddings,
};
pub(super) fn validate_single_qwen_media(
    image: bool,
    video: bool,
    audio: bool,
) -> Result<(), String> {
    if usize::from(image) + usize::from(video) + usize::from(audio) != 1 {
        return Err(
            "Qwen multimodal generation requires exactly one of --image, --video, or --audio"
                .into(),
        );
    }
    Ok(())
}

/// Encoded image for the qwen3vl (`Qwen3VlMerger`) projector family.
///
/// The Qwen3-VL projector emits not only the projected embeddings but also
/// per-layer *deepstack* features that must be injected at specific decoder
/// layers, so both are returned. `grid_shapes` feeds the media positions.
pub struct Qwen3VlImage {
    /// Projected embeddings, `rows * n_embd` floats.
    pub media: Vec<f32>,
    /// Per-layer deepstack features (empty when the projector has none).
    pub deepstack_layers: Vec<Vec<f32>>,
    /// `(grid_h, grid_w)` per encoded frame pair.
    pub grid_shapes: Vec<(usize, usize)>,
}

impl Qwen3VlImage {
    /// Number of placeholder tokens this image occupies.
    pub fn rows(&self, n_embd: usize) -> usize {
        if n_embd == 0 {
            0
        } else {
            self.media.len() / n_embd
        }
    }
}

/// Encode an already-decoded image for the qwen3vl projector.
///
/// Split out of `app::text::multimodal::run_qwen3_family_multimodal` so the
/// HTTP layer can feed in-memory bytes (base64 / uploads) through the same
/// code path. A single image is encoded as the `(0, 0)` frame pair, exactly
/// like the CLI's non-video path.
pub(crate) fn encode_qwen3vl_image_dynamic(
    mmproj_source: &dyn TensorSource,
    image: &image::DynamicImage,
    n_threads_arg: usize,
) -> Result<Qwen3VlImage, String> {
    use crate::app::media::{normalize_resized_image, validate_mmproj_capabilities, MediaKind};
    use crate::models::qwen3::vision::{
        qwen_smart_resize as qwen3vl_smart_resize, VisionEncoder as VisionEncoder3vl,
        VisionScratchpad as VisionScratchpad3vl,
    };

    // Must be a Qwen3-VL projector (not the Qwen2.5-Omni one).
    if !matches!(
        validate_mmproj_capabilities("qwen3vl", mmproj_source, MediaKind::Image)?,
        crate::app::media::ProjectorFamily::Qwen3VlMerger
    ) {
        return Err("mmproj is not a Qwen3-VL (merger) projector".into());
    }

    let mut encoder = VisionEncoder3vl::from_source(
        mmproj_source,
        std::sync::Arc::new(crate::core::thread_pool::ComputePool::new(
            n_threads_arg.max(1),
        )),
    )
    .map_err(|error| format!("Failed to parse vision encoder: {error}"))?;
    encoder.precompute();
    let original_w =
        usize::try_from(image.width()).map_err(|_| "image width does not fit usize")?;
    let original_h =
        usize::try_from(image.height()).map_err(|_| "image height does not fit usize")?;
    let grid = qwen3vl_smart_resize(original_w, original_h, &encoder.config)?;
    let mean = encoder.config.image_mean.clone();
    let std = encoder.config.image_std.clone();
    let normalized =
        normalize_resized_image(image, grid.image_width(), grid.image_height(), &mean, &std)?;

    let mut scratch = VisionScratchpad3vl::new(&encoder.config);
    // A single image is the (0, 0) pair — the same frame twice, matching the
    // CLI's `pairs = vec![(0, 0)]` for non-video input.
    encoder.encode_pair(
        &normalized,
        &normalized,
        grid.image_width(),
        grid.image_height(),
        &mut scratch,
    )?;

    let media = scratch.projected.clone();
    let width = encoder.config.projection_dim;
    if width == 0 || media.len() % width != 0 {
        return Err("Vision output is not aligned to its projected width".into());
    }
    let rows = media.len() / width;
    let deepstack_layers = encoder
        .config
        .has_deepstack_layers
        .iter()
        .filter(|enabled| **enabled)
        .count();
    let mut deepstack_layers_out: Vec<Vec<f32>> = Vec::new();
    if deepstack_layers > 0 {
        if scratch.deepstack.len() % deepstack_layers != 0 {
            return Err("Vision deepstack output is not layer aligned".into());
        }
        deepstack_layers_out.resize_with(deepstack_layers, Vec::new);
        let per_layer = scratch.deepstack.len() / deepstack_layers;
        for (layer, output) in deepstack_layers_out.iter_mut().enumerate() {
            output
                .extend_from_slice(&scratch.deepstack[layer * per_layer..(layer + 1) * per_layer]);
        }
    }
    // The projected width must match the LLM's embedding width; the caller
    // checks that, but a zero-width projector is always wrong.
    if rows == 0 {
        return Err("Vision encoder produced zero image tokens".into());
    }
    let _ = n_threads_arg; // encode_pair is single-threaded; pool not needed
    Ok(Qwen3VlImage {
        media,
        deepstack_layers: deepstack_layers_out,
        grid_shapes: vec![(grid.grid_h, grid.grid_w)],
    })
}

#[cfg(test)]
mod qwen3vl_image_tests {
    #[test]
    #[ignore = "requires the remote pinned Sophon vision GGUF"]
    fn real_projected_width_preserves_all_image_and_deepstack_rows() {
        let _scope = crate::compute::ComputePolicy::Cpu.enter_legacy_scope();
        let source = crate::core::loader::GGUFLoader::from_file(
            &std::env::var("RMI_SOPHON_VISION").unwrap(),
        )
        .unwrap();
        let config =
            crate::models::qwen3::vision::clip_config::ClipVisionConfig::from_source(&source)
                .unwrap();
        assert_ne!(config.n_embd, config.projection_dim);
        let expected_rows = crate::models::qwen3::vision::qwen_smart_resize(32, 32, &config)
            .unwrap()
            .token_count();
        let encoded =
            super::encode_qwen3vl_image_dynamic(&source, &image::DynamicImage::new_rgb8(32, 32), 4)
                .unwrap();
        assert_eq!(encoded.media.len(), expected_rows * config.projection_dim);
        assert_eq!(
            encoded.deepstack_layers.len(),
            config.has_deepstack_layers.iter().filter(|&&v| v).count()
        );
        for layer in &encoded.deepstack_layers {
            assert_eq!(layer.len(), encoded.media.len());
        }
        assert!(encoded.media.iter().all(|v| v.is_finite()));
    }
}

/// Splice a whole span of media embeddings over a **single** marker token.
///
/// [`inject_vision_embeddings`] requires one `<|image_pad|>` per media row,
/// which cannot express MiniCPM-V 4.7: its tile separators (`</image>`,
/// `<slice>`, `</slice>`, newline) are embeddings with no token id, so the
/// marker count is one short of the row count by design. llama.cpp models the
/// same shape with `MTMD_POS_TYPE_CANVAS`, where the chunk contributes no
/// tokens at all and every row takes its own KV slot.
///
/// Returns `(embeddings, marker_index)`: `marker_index` is the row in the
/// returned buffer where the media span starts, so the caller can lay out
/// positions per row.
pub fn inject_media_span(
    llm: &Qwen35Model,
    tokens: &[i32],
    marker_token: i32,
    vis_embd: &[f32],
    n_rows: usize,
    proj_dim: usize,
) -> Result<(Vec<f32>, usize), String> {
    let n_embd = llm.config.n_embd;
    if proj_dim != n_embd {
        return Err(format!(
            "Media projection dim {proj_dim} does not match model width {n_embd}"
        ));
    }
    let expected = n_rows
        .checked_mul(proj_dim)
        .ok_or("Media embedding length overflow")?;
    if vis_embd.len() != expected {
        return Err(format!(
            "Media embedding mismatch: rows={n_rows}, values={}",
            vis_embd.len()
        ));
    }
    let markers = tokens.iter().filter(|&&t| t == marker_token).count();
    if markers != 1 {
        return Err(format!(
            "Media span needs exactly one marker token, found {markers}"
        ));
    }
    let text_ids: Vec<u32> = tokens
        .iter()
        .filter(|&&t| t != marker_token)
        .map(|&t| u32::try_from(t).map_err(|_| format!("invalid negative token id {t}")))
        .collect::<Result<Vec<_>, _>>()?;
    let text = llm.embed_tokens(&text_ids)?;

    let marker_row = tokens
        .iter()
        .position(|&t| t == marker_token)
        .expect("marker count checked above");
    let total_rows = tokens.len() + n_rows - 1;
    let mut out = vec![0.0f32; total_rows * n_embd];
    let mut text_row = 0usize;
    let mut out_row = 0usize;
    let mut span_start = 0usize;
    for (index, &token) in tokens.iter().enumerate() {
        if index == marker_row {
            span_start = out_row;
            out[out_row * n_embd..(out_row + n_rows) * n_embd].copy_from_slice(vis_embd);
            out_row += n_rows;
        } else {
            out[out_row * n_embd..(out_row + 1) * n_embd]
                .copy_from_slice(&text[text_row * n_embd..(text_row + 1) * n_embd]);
            text_row += 1;
            out_row += 1;
        }
    }
    Ok((out, span_start))
}

/// Positions for a prompt whose image is a token-free span.
///
/// Text tokens take the usual sequential positions. The media span is replaced
/// by [`crate::models::minicpmv::canvas_positions`], whose entries are indexed
/// per embedding **row**, matching what `inject_media_span` returns.
///
/// Returns `(positions, next_text_position)` where `next_text_position` is
/// where decoding resumes — one past the last position used, so it stays
/// monotonic even though the canvas reuses `base` for many rows.
pub fn minicpmv47_prompt_positions(
    tokens: &[i32],
    marker_token: i32,
    span_start: usize,
    canvas: &[[usize; 4]],
) -> (Vec<[usize; 4]>, usize) {
    let mut out = Vec::with_capacity(tokens.len() + canvas.len());
    let mut next = 0usize;
    for &token in tokens {
        if token == marker_token {
            out.extend_from_slice(canvas);
        } else {
            out.push([next, next, next, 0]);
            next += 1;
        }
    }
    let _ = span_start;
    // The canvas slots are shared, so resume past the largest one it used.
    let canvas_max = canvas
        .iter()
        .flat_map(|slot| slot.iter().copied())
        .max()
        .unwrap_or(next);
    (out, canvas_max.max(next) + 1)
}
