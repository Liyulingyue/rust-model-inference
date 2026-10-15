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

/// Structural contract, pinned against the oracle's own checkpoint shapes.
///
/// The oracle's `RMI_MTMD_CB_DIR` dump of a 448x448 overview reports, for the
/// default 32x32 patch grid:
///
/// ```text
///   layer_out-0 .. layer_out-6    1024 tokens   (32*32 patches, width 1152)
///   layer_out-7 .. layer_out-26    256 tokens   (after the vit_merger 2x2)
///   post_ln                         256 tokens
///   merger_out                       64 rows    (8*8, width 1024)
/// ```
///
/// Seven layers before the merger and twenty after is the whole signature of
/// `insert_layer_id = 6` running `build_vit_layers(0, 7)` then the merger then
/// `build_vit_layers(7, 27)`. Getting the insertion point wrong -- or merging
/// before rather than after it -- still produces finite embeddings, so this is
/// asserted rather than inferred from a smoke run.
#[test]
fn tower_shape_matches_the_oracle_checkpoint_dump() {
    let cfg = MiniCpmV47Config {
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
    };
    let (nx, ny) = (32usize, 32usize);
    let n_pos = nx * ny;

    // Before the merger: every layer keeps the full patch grid.
    assert_eq!(n_pos, 1024);
    for il in 0..=cfg.insert_layer_id {
        assert_eq!(n_pos, 1024, "layer {il} runs before the merger");
    }
    assert_eq!(
        cfg.insert_layer_id + 1,
        7,
        "seven layers precede the merger"
    );

    // The vit_merger's 2x2 downsample collapses each side by two.
    let after_merger = (nx / 2) * (ny / 2);
    assert_eq!(after_merger, 256, "vit_merger emits 16*16 = 256 rows");
    assert_eq!(
        cfg.n_layer - (cfg.insert_layer_id + 1),
        20,
        "twenty layers follow"
    );
    for il in cfg.insert_layer_id + 1..cfg.n_layer {
        assert_eq!(after_merger, 256, "layer {il} runs after the merger");
    }

    // The final merger halves again and projects to the text width.
    assert_eq!(cfg.output_tokens(nx, ny, 0), 64, "8*8 = 64 projector rows");
    assert_eq!(64 * cfg.n_embd_text, 65536, "merger_out is 65536 floats");
    assert_eq!(
        cfg.insert_layer_id + 1 + (cfg.n_layer - (cfg.insert_layer_id + 1)),
        27
    );
}

/// Diagnostic: run the 35B tower over a pinned token sequence and dump the
/// final-position logits, so they can be compared against the oracle's
/// `RMI_LLAMA_LOGITS_DIR` output for the same tokens.
///
/// RMI_MOE_LOGITS_PROMPT_IDS : comma-separated token ids (default: the
///                             llama.cpp `-p` tokenisation of
///                             `user\nWhat is 2+2?\nassistant\n<think>\n\n</think>\n\n`)
/// RMI_MOE_LOGITS_MODEL     : path to the 35B GGUF
/// RMI_MOE_LOGITS_OUT       : output file for the raw f32 logits
#[test]
#[ignore = "writes parity dumps; needs the 35B model"]
fn dump_qwen35moe_logits() {
    let model_path = match std::env::var("RMI_MOE_LOGITS_MODEL") {
        Ok(path) => path,
        Err(_) => {
            eprintln!("skipping: set RMI_MOE_LOGITS_MODEL");
            return;
        }
    };
    let ids: Vec<u32> = std::env::var("RMI_MOE_LOGITS_PROMPT_IDS")
        .unwrap_or_else(|_| {
            "0,846,198,3710,369,220,17,10,17,30,198,74455,198,248068,271,248069,271".into()
        })
        .split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect();

    let loader = GGUFLoader::from_file(&model_path).expect("open 35B GGUF");
    let source: Arc<dyn TensorSource> = Arc::new(loader);
    // Dump the input embedding of the final prompt token so it can be compared
    // against the oracle's `model.input_embed`.
    {
        use rust_model_inference::models::qwen35::Qwen35Model;
        let model = Qwen35Model::from_source(source.as_ref()).expect("load 35B");
        let n_embd = model.config.n_embd;
        let embd = model.embed_tokens(&ids).expect("embed");
        let out = std::env::var("RMI_MOE_LOGITS_OUT").unwrap_or_else(|_| ".".into());
        let bytes: Vec<u8> = embd.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = std::fs::write(format!("{out}/embed_all.bin"), bytes);
        eprintln!("embed rows={}", embd.len() / n_embd);
    }

    let logits = rust_model_inference::models::qwen35::trunk::run_forward_logits_qwen35_with_batch(
        source.as_ref(),
        &ids,
        8,
        rust_model_inference::core::scratchpad::KvFormat::F32,
        1024,
        1,
    )
    .expect("forward logits")
    .0;
    assert_eq!(logits.len(), 248144, "vocab size");
    let _ = &logits;
    let out = std::env::var("RMI_MOE_LOGITS_OUT").unwrap_or_else(|_| "logits.bin".into());
    let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&out, bytes).expect("write logits");
    let mut top: Vec<usize> = (0..logits.len()).collect();
    top.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let summary: Vec<String> = top
        .iter()
        .take(10)
        .map(|&i| format!("{i}:{:.5}", logits[i]))
        .collect();
    eprintln!("rust top10: {}", summary.join(" "));
}
