use crate::app::cli::DreamXCliOptions;
use crate::core::tensor::TensorSource;
use crate::models::diffusion::dreamx::{
    ensure_memory, gib, resize_to_token_budget, DreamXEstimate, DreamXPipeline, DreamXRequest,
};
use std::sync::Arc;
use std::time::Instant;

pub fn run_dreamx_cli(
    main: Arc<dyn TensorSource>,
    mmproj: Arc<dyn TensorSource>,
    options: DreamXCliOptions,
    n_threads: usize,
) -> Result<(), String> {
    let started = Instant::now();
    let input = image::open(&options.image)
        .map_err(|error| {
            format!(
                "Load DreamX input image {}: {error}",
                options.image.display()
            )
        })?
        .into_rgb8();
    let image = resize_to_token_budget(&input, options.options.target_spatial_tokens)?;
    let request = DreamXRequest {
        image,
        prompt: options.prompt,
        negative_prompt: options.negative_prompt.unwrap_or_default(),
        output: options.out,
        options: options.options,
        overwrite: options.overwrite,
        allow_memory_overcommit: options.allow_memory_overcommit,
    };
    let pipeline = DreamXPipeline::load(main, mmproj, n_threads)?;
    let estimate = pipeline.estimate(&request)?;
    print_estimate(&estimate, request.options.refine);
    ensure_memory(&estimate, request.allow_memory_overcommit)?;
    if options.dry_run {
        println!("DreamX dry-run complete; no tensors were loaded");
        return Ok(());
    }

    let artifacts = pipeline.generate(&request)?;
    println!(
        "DreamX base video: {}\nDreamX audio: {}\nDreamX base mux: {}",
        artifacts.base_video.display(),
        artifacts.audio.display(),
        artifacts.base_muxed.display(),
    );
    if let (Some(video), Some(muxed)) = (artifacts.refined_video, artifacts.refined_muxed) {
        println!(
            "DreamX refined video: {}\nDreamX refined mux: {}",
            video.display(),
            muxed.display(),
        );
    }
    println!(
        "DreamX generation completed in {}ms",
        started.elapsed().as_millis()
    );
    Ok(())
}

fn print_estimate(estimate: &DreamXEstimate, refine: bool) {
    println!(
        "DreamX input: {}x{}, {} spatial tokens, {} output frames",
        estimate.width, estimate.height, estimate.spatial_tokens, estimate.output_frames
    );
    println!(
        "DreamX text stage: {:.2} GiB weights, {:.2} GiB scratch",
        gib(estimate.text_weight_bytes),
        gib(estimate.text_scratch_bytes),
    );
    println!(
        "DreamX first-frame/base VAE stage: {:.2} GiB weights",
        gib(estimate.video_vae_weight_bytes),
    );
    println!(
        "DreamX Creator stage: {:.2} GiB weights, {:.2} GiB scratch",
        gib(estimate.creator_weight_bytes),
        gib(estimate.creator_scratch_bytes),
    );
    println!(
        "DreamX audio stage: {:.2} GiB weights, {} Hz output",
        gib(estimate.audio_vae_weight_bytes),
        estimate.audio_sample_rate,
    );
    if refine {
        println!(
            "DreamX refiner stage: {:.2} GiB weights, {:.2} GiB scratch, {:.2} GiB KV",
            gib(estimate.refiner_weight_bytes),
            gib(estimate.refiner_scratch_bytes),
            gib(estimate.refiner_kv_bytes),
        );
    } else {
        println!("DreamX refiner stage: disabled");
    }
    match estimate.physical_memory_bytes {
        Some(physical) => println!(
            "DreamX estimated peak: {:.2} GiB / {:.2} GiB physical memory",
            gib(estimate.peak_bytes),
            gib(physical),
        ),
        None => println!(
            "DreamX estimated peak: {:.2} GiB; physical memory unavailable",
            gib(estimate.peak_bytes),
        ),
    }
}
