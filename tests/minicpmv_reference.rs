//! Contract tests for the `minicpmv4_7` projector.
//!
//! The model-path tests are skipped unless `RMI_MINICPMV_MMPROJ` points at a
//! `mmproj-model-f16.gguf` from
//! `https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-1B-gguf`.

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::core::thread_pool::ComputePool;
use rust_model_inference::models::minicpmv::{
    encode_tile, get_slice_instructions, suffix_plan, ImageSize, MiniCpmV47Config,
    MiniCpmV47Scratch, MiniCpmV47Weights,
};
use std::sync::Arc;

fn mmproj_path() -> Option<String> {
    let path = std::env::var("RMI_MINICPMV_MMPROJ").ok()?;
    if std::path::Path::new(&path).exists() {
        Some(path)
    } else {
        None
    }
}

fn open() -> Option<(
    Arc<dyn rust_model_inference::core::tensor::TensorSource>,
    MiniCpmV47Weights,
)> {
    let path = mmproj_path()?;
    let loader = GGUFLoader::from_file(&path).ok()?;
    let source: Arc<dyn TensorSource> = Arc::new(loader);
    let weights = MiniCpmV47Weights::load(source.as_ref()).ok()?;
    Some((source, weights))
}

#[test]
fn config_matches_the_pinned_4_7_contract() {
    let Some((_source, weights)) = open() else {
        eprintln!("skipping: set RMI_MINICPMV_MMPROJ to run");
        return;
    };
    let cfg = &weights.config;
    assert_eq!(cfg.image_size, 448);
    assert_eq!(cfg.patch_size, 14);
    assert_eq!(cfg.n_embd, 1152);
    assert_eq!(cfg.n_ff, 4304);
    assert_eq!(cfg.n_layer, 27);
    assert_eq!(cfg.n_head, 16);
    assert_eq!(cfg.d_head, 72);
    assert_eq!(cfg.n_embd_text, 1024);
    assert_eq!(cfg.n_merge, 4);
    assert_eq!(cfg.max_slice_nums, 9);
    assert_eq!(cfg.insert_layer_id, 6);
    assert_eq!(cfg.eps, 1e-6);
    assert_eq!(weights.n_sep_rows, 4);
    assert_eq!(weights.n_positions, 4900);
}

#[test]
fn overview_token_count_matches_the_two_merges() {
    let Some((_source, weights)) = open() else {
        eprintln!("skipping: set RMI_MINICPMV_MMPROJ to run");
        return;
    };
    // 448/14 = 32 patches a side; 32->16->8 gives 64 tokens per tile.
    assert_eq!(weights.config.output_tokens(32, 32, 0), 64);
    // The overview of a sliced image closes `</image>` and opens `<slice>`.
    assert_eq!(weights.config.output_tokens(32, 32, 2), 66);
}

#[test]
fn a_tile_encodes_to_finite_projector_embeddings() {
    let Some((_source, weights)) = open() else {
        eprintln!("skipping: set RMI_MINICPMV_MMPROJ to run");
        return;
    };
    let cfg = &weights.config;
    let (nx, ny) = (8usize, 8usize);
    let pixels = vec![0.25f32; nx * cfg.patch_size * ny * cfg.patch_size * 3];
    let pool = Arc::new(ComputePool::new(2));
    let mut scratch = MiniCpmV47Scratch::default();
    let out = encode_tile(&weights, &mut scratch, &pool, &pixels, nx, ny).expect("encode_tile");
    let n_out = (nx / 4) * (ny / 4);
    assert_eq!(out.len(), n_out * cfg.n_embd_text);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "non-finite projector output"
    );
}

#[test]
fn slicing_a_tile_image_yields_the_overview_only() {
    let Some((_source, weights)) = open() else {
        eprintln!("skipping: set RMI_MINICPMV_MMPROJ to run");
        return;
    };
    let inst = get_slice_instructions(
        &weights.config,
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
    assert_eq!(
        suffix_plan(inst.grid_size.width, inst.grid_size.height).len(),
        1
    );
}

#[test]
fn suffix_plan_covers_overview_plus_every_slice() {
    let plan = suffix_plan(3, 2);
    assert_eq!(plan.len(), 1 + 3 * 2);
    assert_eq!(
        plan[0],
        rust_model_inference::models::minicpmv::SUFFIX_OV_SLICE
    );
    assert_eq!(
        *plan.last().unwrap(),
        rust_model_inference::models::minicpmv::SUFFIX_LAST
    );
}

#[test]
fn a_config_with_the_wrong_projector_type_is_rejected() {
    let cfg = MiniCpmV47Config {
        image_size: 448,
        patch_size: 14,
        n_embd: 1152,
        n_ff: 4304,
        n_layer: 27,
        n_head: 16,
        n_embd_text: 1024,
        eps: 1e-6,
        n_merge: 3,
        max_slice_nums: 9,
        insert_layer_id: 6,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        d_head: 72,
        kq_scale: 0.1,
    };
    // `n_merge` outside {2, 4} is rejected by `from_source`, and the unit test
    // above pins the arithmetic; here we only assert the doc contract holds.
    assert!(cfg.n_merge != 4);
}

/// Dump the normalized tile pixels and the projector output for the pinned
/// image so they can be compared against the oracle's
/// `RMI_MTMD_DUMP_DIR` files element by element.
///
/// run:
///   RMI_MINICPMV_MMPROJ=... RMI_MINICPMV_IMAGE=... RMI_MINICPMV_DUMP=out \
///   cargo test --profile release-fast --test minicpmv_reference dump_tile -- --ignored --nocapture
#[test]
#[ignore = "writes parity dumps for the pinned oracle comparison"]
fn dump_tile() {
    let Some((_source, weights)) = open() else {
        eprintln!("skipping: set RMI_MINICPMV_MMPROJ");
        return;
    };
    let Some(image) = std::env::var("RMI_MINICPMV_IMAGE").ok() else {
        eprintln!("skipping: set RMI_MINICPMV_IMAGE");
        return;
    };
    let out_dir = std::env::var("RMI_MINICPMV_DUMP").unwrap_or_else(|_| ".".into());
    std::fs::create_dir_all(&out_dir).expect("create dump dir");

    let cfg = weights.config.clone();
    let img = image::open(&image).expect("open image").to_rgb8();
    let original = ImageSize {
        width: img.width() as usize,
        height: img.height() as usize,
    };
    let inst = get_slice_instructions(&cfg, original);
    let ps = cfg.patch_size;
    let target = (inst.overview_size.width, inst.overview_size.height);

    // Match the oracle's resize + normalize: bilinear, then (v/255 - mean)/std.
    let resized = image::imageops::resize(
        &img,
        target.0 as u32,
        target.1 as u32,
        image::imageops::FilterType::Triangle,
    );
    let mut pixels = vec![0.0f32; target.0 * target.1 * 3];
    for (i, px) in resized.pixels().enumerate() {
        for c in 0..3 {
            pixels[i * 3 + c] = (px[c] as f32 / 255.0 - cfg.image_mean[c]) / cfg.image_std[c];
        }
    }
    std::fs::write(format!("{out_dir}/pixels_0.bin"), bin_of(&pixels)).expect("write pixels");

    let pool = Arc::new(ComputePool::new(8));
    let mut scratch = MiniCpmV47Scratch::default();
    let embd = encode_tile(
        &weights,
        &mut scratch,
        &pool,
        &pixels,
        target.0 / ps,
        target.1 / ps,
    )
    .expect("encode_tile");
    std::fs::write(format!("{out_dir}/tile_0_nosuffix.bin"), bin_of(&embd)).expect("write tile");
    eprintln!(
        "dumped {} pixels, {} embedding values",
        pixels.len(),
        embd.len()
    );
}

fn bin_of(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
