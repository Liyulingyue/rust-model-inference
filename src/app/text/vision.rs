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

pub(crate) fn build_qwen3_media_positions(
    token_ids: &[u32],
    placeholder_id: u32,
    grid_shapes: &[(usize, usize)],
) -> Result<Vec<[usize; 4]>, String> {
    let mut positions = Vec::with_capacity(token_ids.len());
    let mut next = 0usize;
    let mut token = 0usize;
    let mut grid_index = 0usize;
    while token < token_ids.len() {
        if token_ids[token] != placeholder_id {
            positions.push([next, next, next, 0]);
            next = next.checked_add(1).ok_or("Qwen media position overflow")?;
            token += 1;
            continue;
        }
        if grid_shapes.is_empty() {
            positions.push([next, next, next, 0]);
            next = next.checked_add(1).ok_or("Qwen audio position overflow")?;
            token += 1;
            continue;
        }
        let (grid_h, grid_w) = *grid_shapes
            .get(grid_index)
            .ok_or("Media placeholder has no matching vision grid")?;
        let count = grid_h
            .checked_mul(grid_w)
            .ok_or("Qwen media grid token count overflow")?;
        let end = token
            .checked_add(count)
            .ok_or("Qwen media placeholder range overflow")?;
        if count == 0
            || end > token_ids.len()
            || token_ids[token..end].iter().any(|id| *id != placeholder_id)
        {
            return Err(format!(
                "Vision grid {grid_index} requires {count} contiguous placeholders"
            ));
        }
        let base = next;
        for index in 0..count {
            let row = index / grid_w;
            let column = index % grid_w;
            positions.push([
                base,
                base.checked_add(row)
                    .ok_or("Qwen media row position overflow")?,
                base.checked_add(column)
                    .ok_or("Qwen media column position overflow")?,
                0,
            ]);
        }
        next = base
            .checked_add(grid_h.max(grid_w))
            .ok_or("Qwen media logical position overflow")?;
        token = end;
        grid_index += 1;
    }
    if !grid_shapes.is_empty() && grid_index != grid_shapes.len() {
        return Err(format!(
            "Unused vision grids: consumed {grid_index}, provided {}",
            grid_shapes.len()
        ));
    }
    Ok(positions)
}

pub(super) fn inject_qwen_media_embeddings(
    token_ids: &[u32],
    pad: u32,
    embeddings: &mut [f32],
    media: &[f32],
    media_deepstack: &[f32],
    width: usize,
) -> Result<Vec<f32>, String> {
    if width == 0 || embeddings.len() != token_ids.len().saturating_mul(width) {
        return Err("Prompt embedding shape mismatch".into());
    }
    if media.len() % width != 0 {
        return Err("Media embeddings are not row aligned".into());
    }
    let media_rows = media.len() / width;
    let per_layer = media.len();
    if !media_deepstack.is_empty() && (per_layer == 0 || media_deepstack.len() % per_layer != 0) {
        return Err("Media deepstack embeddings are not layer aligned".into());
    }
    let deepstack_layers = if per_layer == 0 {
        0
    } else {
        media_deepstack.len() / per_layer
    };
    let mut deepstack = vec![0.0; deepstack_layers * embeddings.len()];
    let mut media_row = 0;
    for (token_index, (&token, row)) in token_ids
        .iter()
        .zip(embeddings.chunks_exact_mut(width))
        .enumerate()
    {
        if token != pad {
            continue;
        }
        if media_row >= media_rows {
            return Err("Media placeholder count exceeds projector rows".into());
        }
        row.copy_from_slice(&media[media_row * width..(media_row + 1) * width]);
        for layer in 0..deepstack_layers {
            let src = (layer * media_rows + media_row) * width;
            let dst = (layer * token_ids.len() + token_index) * width;
            deepstack[dst..dst + width].copy_from_slice(&media_deepstack[src..src + width]);
        }
        media_row += 1;
    }
    if media_row != media_rows {
        return Err(format!(
            "Media placeholder count mismatch: placeholders={media_row}, rows={media_rows}"
        ));
    }
    Ok(deepstack)
}

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

    let mut media = scratch.projected.clone();
    let n_embd = encoder.config.n_embd;
    let rows = if n_embd == 0 { 0 } else { media.len() / n_embd };
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
    media.truncate(rows * n_embd);
    let _ = n_threads_arg; // encode_pair is single-threaded; pool not needed
    Ok(Qwen3VlImage {
        media,
        deepstack_layers: deepstack_layers_out,
        grid_shapes: vec![(grid.grid_h, grid.grid_w)],
    })
}
