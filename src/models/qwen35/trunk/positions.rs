//! mRoPE position builder for Qwen3.5 multimodal inputs.
//!
//! Walks the prompt token stream, expanding image-placeholder runs into one
//! mrope position per patch (using `VisionGrid`), and assigning each text
//! token a `[t, t, t, 0]` position. Returns the per-token positions and
//! the next free logical position (for incremental decode).

use crate::models::qwen35::vision::VisionGrid;

pub fn build_qwen35_positions(
    token_ids: &[u32],
    image_token_id: Option<u32>,
    image_grids: &[VisionGrid],
) -> Result<(Vec<[usize; 4]>, usize), String> {
    let mut positions = Vec::with_capacity(token_ids.len());
    let mut next = 0usize;
    let mut token = 0usize;
    let mut grid_index = 0usize;

    while token < token_ids.len() {
        if image_token_id == Some(token_ids[token]) {
            let grid = *image_grids
                .get(grid_index)
                .ok_or("Image placeholder has no matching vision grid")?;
            let count = grid.checked_token_count()?;
            let end = token
                .checked_add(count)
                .ok_or("Image placeholder range overflow")?;
            if end > token_ids.len()
                || token_ids[token..end]
                    .iter()
                    .any(|id| Some(*id) != image_token_id)
            {
                return Err(format!(
                    "Image grid {grid_index} requires {count} contiguous placeholders"
                ));
            }
            let base = next;
            for image_index in 0..count {
                let row = image_index / grid.grid_w;
                let column = image_index % grid.grid_w;
                positions.push([base, base + row, base + column, 0]);
            }
            next = next
                .checked_add(grid.position_span())
                .ok_or("Qwen3.5 logical position overflow")?;
            token = end;
            grid_index += 1;
        } else {
            positions.push([next, next, next, 0]);
            next = next
                .checked_add(1)
                .ok_or("Qwen3.5 logical position overflow")?;
            token += 1;
        }
    }

    if grid_index != image_grids.len() {
        return Err(format!(
            "Unused vision grids: consumed {grid_index}, provided {}",
            image_grids.len()
        ));
    }
    Ok((positions, next))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen35::vision::VisionGrid;

    #[test]
    fn qwen35_positions_use_time_row_column_order() {
        let grid = VisionGrid {
            grid_t: 1,
            grid_h: 2,
            grid_w: 3,
            patch_size: 16,
            merge_size: 2,
        };
        let tokens = [10, 99, 99, 99, 99, 99, 99, 11];
        let (positions, next) = build_qwen35_positions(&tokens, Some(99), &[grid]).unwrap();
        assert_eq!(positions[1], [1, 1, 1, 0]);
        assert_eq!(positions[6], [1, 2, 3, 0]);
        assert_eq!(positions[7], [4, 4, 4, 0]);
        assert_eq!(next, 5);
    }

    #[test]
    fn qwen35_placeholder_count_must_equal_grid_tokens() {
        let grid = VisionGrid {
            grid_t: 1,
            grid_h: 2,
            grid_w: 3,
            patch_size: 16,
            merge_size: 2,
        };
        assert!(build_qwen35_positions(&[10, 99, 99, 11], Some(99), &[grid]).is_err());
    }

    #[test]
    fn qwen35_positions_reject_public_grid_token_overflow() {
        let grid = VisionGrid {
            grid_t: 1,
            grid_h: usize::MAX,
            grid_w: 2,
            patch_size: 1,
            merge_size: 1,
        };
        assert!(build_qwen35_positions(&[99], Some(99), &[grid]).is_err());
    }
}