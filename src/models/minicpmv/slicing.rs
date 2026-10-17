//! MiniCPM-V 4.7 image slicing.
//!
//! Port of `mtmd_image_preprocessor_llava_uhd::get_slice_instructions` plus
//! the `minicpmv` overrides, at the pinned oracle
//! `69f201a2051ea9b9e9b50c3cc56afd9e2ae64414` (`tools/mtmd/mtmd-image.cpp`,
//! `tools/mtmd/mtmd-image.h`).
//!
//! One image becomes a 1x1 "overview" tile plus an `n_col x n_row` grid of
//! slice tiles, all of which the projector then lays out on a single 2D canvas.

use super::config::MiniCpmV47Config;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSize {
    pub width: usize,
    pub height: usize,
}

impl ImageSize {
    pub const ZERO: Self = Self {
        width: 0,
        height: 0,
    };
}

/// One slice of the refined image, in refined-image pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceBox {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceInstructions {
    pub overview_size: ImageSize,
    pub refined_size: ImageSize,
    pub grid_size: ImageSize,
    pub slices: Vec<SliceBox>,
}

impl SliceInstructions {
    /// Small images get the overview only; nothing is sliced.
    pub fn has_slices(&self) -> bool {
        !self.slices.is_empty()
    }
}

/// Python's `round()`: half-to-even. `std::round` rounds half away from zero,
/// which the reference deliberately does not use here.
pub fn align_round(value: f64) -> i64 {
    let floor = value.floor();
    let diff = value - floor;
    if diff > 0.5 {
        return floor as i64 + 1;
    }
    if diff < 0.5 {
        return floor as i64;
    }
    let lo = floor as i64;
    if lo % 2 == 0 {
        lo
    } else {
        lo + 1
    }
}

fn ensure_divide(length: usize, divisor: usize) -> usize {
    align_round(length as f64 / divisor as f64).max(1) as usize * divisor
}

fn get_best_resize(
    original: ImageSize,
    scale_resolution: usize,
    patch_size: usize,
    allow_upscale: bool,
) -> ImageSize {
    let mut width = original.width;
    let mut height = original.height;
    if width * height > scale_resolution * scale_resolution || allow_upscale {
        let r = width as f32 / height as f32;
        height = (scale_resolution as f32 / r.sqrt()) as usize;
        width = (height as f32 * r) as usize;
    }
    ImageSize {
        width: ensure_divide(width, patch_size),
        height: ensure_divide(height, patch_size),
    }
}

fn get_best_grid(max_slice_nums: usize, multiple: usize, log_ratio: f32) -> ImageSize {
    let mut candidates = Vec::new();
    for i in [multiple.saturating_sub(1), multiple, multiple + 1] {
        if i == 1 || i > max_slice_nums {
            continue;
        }
        candidates.push(i);
    }
    let mut grids = Vec::new();
    for n in candidates {
        let mut m = 1usize;
        while m <= n {
            if n % m == 0 {
                grids.push(ImageSize {
                    width: m,
                    height: n / m,
                });
            }
            m += 1;
        }
    }
    let mut best = ImageSize {
        width: 1,
        height: 1,
    };
    let mut min_error = f32::INFINITY;
    for grid in grids {
        let error = (log_ratio - (grid.width as f32 / grid.height as f32).ln()).abs();
        if error < min_error {
            best = grid;
            min_error = error;
        }
    }
    best
}

fn get_refine_size(
    original: ImageSize,
    grid: ImageSize,
    scale_resolution: usize,
    patch_size: usize,
    allow_upscale: bool,
) -> ImageSize {
    let refine_width = ensure_divide(original.width, grid.width);
    let refine_height = ensure_divide(original.height, grid.height);
    let grid_size = ImageSize {
        width: refine_width / grid.width,
        height: refine_height / grid.height,
    };
    let best = get_best_resize(grid_size, scale_resolution, patch_size, allow_upscale);
    ImageSize {
        width: best.width * grid.width,
        height: best.height * grid.height,
    }
}

/// `mtmd_image_preprocessor_minicpmv::get_slice_instructions`.
///
/// The only behavioural difference from the generic llava-uhd path is that a
/// small image (area <= one tile) takes the overview only, rather than being
/// sliced once one side exceeds the tile size.
pub fn get_slice_instructions(cfg: &MiniCpmV47Config, original: ImageSize) -> SliceInstructions {
    let slice_size = cfg.image_size;
    let area = original.width as f32 * original.height as f32;
    let ratio = area / (slice_size * slice_size) as f32;
    if ratio <= 1.0 {
        return SliceInstructions {
            overview_size: get_best_resize(original, slice_size, cfg.slice_align(), true),
            refined_size: ImageSize::ZERO,
            grid_size: ImageSize::ZERO,
            slices: Vec::new(),
        };
    }

    let align = cfg.slice_align();
    let overview_size = get_best_resize(original, slice_size, align, false);
    let log_ratio = (original.width as f32 / original.height as f32).ln();
    let multiple = (ratio.ceil() as usize).min(cfg.max_slice_nums);
    let best_grid = get_best_grid(cfg.max_slice_nums, multiple, log_ratio);
    let refined_size = get_refine_size(original, best_grid, slice_size, align, true);

    let grid_x = refined_size.width / best_grid.width;
    let grid_y = refined_size.height / best_grid.height;
    let mut slices = Vec::with_capacity(best_grid.width * best_grid.height);
    let mut patches_y = 0usize;
    let mut ic = 0usize;
    while patches_y < refined_size.height && ic < best_grid.height {
        let mut patches_x = 0usize;
        let mut jc = 0usize;
        while patches_x < refined_size.width && jc < best_grid.width {
            slices.push(SliceBox {
                x: patches_x,
                y: patches_y,
                width: grid_x,
                height: grid_y,
            });
            patches_x += grid_x;
            jc += 1;
        }
        patches_y += grid_y;
        ic += 1;
    }

    SliceInstructions {
        overview_size,
        refined_size,
        grid_size: best_grid,
        slices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MiniCpmV47Config {
        MiniCpmV47Config {
            image_size: 448,
            patch_size: 14,
            n_embd: 1152,
            n_ff: 4304,
            n_layer: 27,
            n_head: 16,
            n_embd_text: 1024,
            eps: 1e-6,
            n_merge: 4,
            max_slice_nums: 9,
            insert_layer_id: 6,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            d_head: 72,
            kq_scale: 1.0 / 72.0f32.sqrt(),
        }
    }

    #[test]
    fn align_round_breaks_ties_to_even() {
        assert_eq!(align_round(0.5), 0);
        assert_eq!(align_round(1.5), 2);
        assert_eq!(align_round(2.5), 2);
        assert_eq!(align_round(3.5), 4);
        assert_eq!(align_round(4.4), 4);
        assert_eq!(align_round(4.6), 5);
    }

    #[test]
    fn square_tile_image_is_overview_only() {
        let inst = get_slice_instructions(
            &cfg(),
            ImageSize {
                width: 448,
                height: 448,
            },
        );
        assert!(!inst.has_slices());
        assert_eq!(
            inst.overview_size,
            ImageSize {
                width: 448,
                height: 448
            }
        );
        assert_eq!(inst.grid_size, ImageSize::ZERO);
    }

    #[test]
    fn small_image_is_resized_onto_one_tile() {
        let inst = get_slice_instructions(
            &cfg(),
            ImageSize {
                width: 200,
                height: 100,
            },
        );
        assert!(!inst.has_slices());
        // Both sides land on the 56px slice-align grid and the aspect is kept
        // within one rounding step.
        assert_eq!(inst.overview_size.width % 56, 0);
        assert_eq!(inst.overview_size.height % 56, 0);
        assert!(inst.overview_size.width >= inst.overview_size.height);
    }

    #[test]
    fn large_image_slices_into_a_bounded_grid() {
        let inst = get_slice_instructions(
            &cfg(),
            ImageSize {
                width: 1344,
                height: 1344,
            },
        );
        assert!(inst.has_slices());
        assert_eq!(
            inst.slices.len(),
            inst.grid_size.width * inst.grid_size.height
        );
        assert!(inst.slices.len() <= cfg().max_slice_nums);
        // Every slice covers the refined image exactly once.
        let covered: usize = inst.slices.iter().map(|s| s.width * s.height).sum();
        assert_eq!(covered, inst.refined_size.width * inst.refined_size.height);
    }

    #[test]
    fn wide_image_prefers_a_wide_grid() {
        let inst = get_slice_instructions(
            &cfg(),
            ImageSize {
                width: 2000,
                height: 500,
            },
        );
        assert!(inst.grid_size.width >= inst.grid_size.height);
    }

    #[test]
    fn slice_sizes_are_aligned_to_the_patch_grid() {
        let inst = get_slice_instructions(
            &cfg(),
            ImageSize {
                width: 1600,
                height: 1200,
            },
        );
        let align = cfg().slice_align();
        for slice in &inst.slices {
            assert_eq!(slice.width % align, 0, "{slice:?}");
            assert_eq!(slice.height % align, 0, "{slice:?}");
        }
    }
}
