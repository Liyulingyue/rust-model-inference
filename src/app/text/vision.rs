use crate::models::qwen35::Qwen35Model;
use std::path::Path;

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

pub(super) fn validate_single_qwen_media(image: bool, video: bool, audio: bool) -> Result<(), String> {
    if usize::from(image) + usize::from(video) + usize::from(audio) != 1 {
        return Err(
            "Qwen multimodal generation requires exactly one of --image, --video, or --audio"
                .into(),
        );
    }
    Ok(())
}
