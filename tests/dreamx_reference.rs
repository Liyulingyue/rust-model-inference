use image::RgbImage;
use rust_model_inference::models::diffusion::dreamx::{
    resize_to_token_budget, resolve_spatial_size, DreamXOptions, DreamXPipeline, DreamXRequest,
};
use rust_model_inference::{
    open_model_source, ComponentRole, MetaValue, MetaValueType, TensorInfo, TensorSource,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

struct MetadataOnlySource {
    metadata: HashMap<String, MetaValue>,
}

impl TensorSource for MetadataOnlySource {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
        panic!("DreamX dry-run read tensor metadata")
    }

    fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
        panic!("DreamX dry-run read tensor data")
    }
}

fn strings(values: &[&str]) -> MetaValue {
    MetaValue::Array(
        MetaValueType::String,
        values
            .iter()
            .map(|value| MetaValue::String((*value).into()))
            .collect(),
    )
}

fn integers(values: impl IntoIterator<Item = u64>) -> MetaValue {
    MetaValue::Array(
        MetaValueType::Uint64,
        values.into_iter().map(MetaValue::Uint64).collect(),
    )
}

fn source(role: &str) -> MetadataOnlySource {
    let components = [
        "audio_vae",
        "creator.audio",
        "creator.joint",
        "creator.video",
        "refiner.dit",
        "refiner.lightvae",
        "refiner.upsampler.causal2d",
        "refiner.upsampler.flash",
        "text",
        "video_vae",
    ];
    let mut metadata = HashMap::from([
        (
            "general.architecture".into(),
            MetaValue::String(if role == "main" { "dreamx" } else { "clip" }.into()),
        ),
        ("dreamx.pair_id".into(), MetaValue::String("a".repeat(64))),
        ("dreamx.file_role".into(), MetaValue::String(role.into())),
        (
            "dreamx.outtype".into(),
            MetaValue::String(if role == "main" { "q8_0" } else { "bf16" }.into()),
        ),
        (
            "dreamx.source_model".into(),
            MetaValue::String("GD-ML/DreamX-Creator".into()),
        ),
        ("dreamx.exporter_version".into(), MetaValue::Uint64(1)),
        ("dreamx.components".into(), strings(&components)),
        (
            "dreamx.file_components".into(),
            strings(if role == "main" {
                &[
                    "creator.audio",
                    "creator.joint",
                    "creator.video",
                    "refiner.dit",
                ]
            } else {
                &[
                    "audio_vae",
                    "refiner.lightvae",
                    "refiner.upsampler.causal2d",
                    "refiner.upsampler.flash",
                    "text",
                    "video_vae",
                ]
            }),
        ),
        (
            "dreamx.joint_layers".into(),
            integers((15..=29).map(|value| value as u64)),
        ),
        (
            "dreamx.tokenizer.sha256".into(),
            MetaValue::String("b".repeat(64)),
        ),
    ]);
    if role == "mmproj" {
        metadata.insert("general.type".into(), MetaValue::String("mmproj".into()));
        metadata.insert(
            "clip.projector_type".into(),
            MetaValue::String("dreamx_creator".into()),
        );
    }
    for component in components {
        metadata.insert(
            format!("dreamx.has_component.{component}"),
            MetaValue::Bool(true),
        );
        metadata.insert(
            format!("dreamx.component.{component}.tensor_count"),
            MetaValue::Uint64(1),
        );
    }
    for (key, value) in [
        ("dreamx.video.embedding_length", 3072),
        ("dreamx.video.feed_forward_length", 14336),
        ("dreamx.video.attention.head_count", 24),
        ("dreamx.video.block_count", 30),
        ("dreamx.video.in_channels", 48),
        ("dreamx.audio.embedding_length", 1536),
        ("dreamx.audio.feed_forward_length", 8960),
        ("dreamx.audio.attention.head_count", 12),
        ("dreamx.audio.block_count", 30),
        ("dreamx.audio.in_channels", 128),
        ("dreamx.text.context_length", 512),
        ("dreamx.text.embedding_length", 4096),
        ("dreamx.text.feed_forward_length", 10240),
        ("dreamx.text.attention.head_count", 64),
        ("dreamx.text.block_count", 24),
        ("dreamx.text.vocab_size", 256384),
    ] {
        metadata.insert(key.into(), MetaValue::Uint64(value));
    }
    MetadataOnlySource { metadata }
}

#[test]
fn dry_run_reports_each_stage_without_tensor_compute() {
    let pipeline =
        DreamXPipeline::load(Arc::new(source("main")), Arc::new(source("mmproj")), 2).unwrap();
    let request = DreamXRequest {
        image: RgbImage::new(128, 64),
        prompt: "scene".into(),
        negative_prompt: String::new(),
        output: PathBuf::from("scene.mp4"),
        options: DreamXOptions {
            duration_seconds: 0.2,
            fps: 5,
            steps: 1,
            target_spatial_tokens: 8,
            ..DreamXOptions::default()
        },
        overwrite: false,
        allow_memory_overcommit: false,
    };

    let estimate = pipeline.estimate(&request).unwrap();
    assert!(estimate.text_scratch_bytes > 0);
    assert!(estimate.creator_scratch_bytes > 0);
    assert!(estimate.refiner_kv_bytes > 0);
    assert!(estimate.peak_bytes > estimate.frame_bytes);
    assert_eq!(estimate.width, 128);
    assert_eq!(estimate.height, 64);
    assert_eq!(estimate.output_frames, 1);
    assert_eq!(estimate.audio_sample_rate, 48_000);
}

#[test]
fn spatial_budget_matches_upstream_integer_grid_search() {
    assert_eq!(
        resolve_spatial_size(360, 640, 880).unwrap(),
        (704, 1248, 858)
    );
    assert_eq!(resolve_spatial_size(360, 640, 4).unwrap(), (64, 64, 4));
}

#[test]
#[ignore = "requires exported DreamX-Creator GGUF pair"]
fn dreamx_real_pair_preflight_and_dry_run() {
    let root = std::env::var_os("DREAMX_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/Users/gouzi/Documents/git/rust-model-inference/models/DreamX-Creator")
        });
    let main: Arc<dyn TensorSource> = Arc::from(
        open_model_source(&root.join("DreamX-Creator-Q8_0.gguf"), ComponentRole::Llm).unwrap(),
    );
    let mmproj: Arc<dyn TensorSource> = Arc::from(
        open_model_source(
            &root.join("mmproj-DreamX-Creator-BF16.gguf"),
            ComponentRole::Mmproj,
        )
        .unwrap(),
    );
    let image = image::open(root.join("dreamx-creator_teaser.png"))
        .unwrap()
        .into_rgb8();
    let image = resize_to_token_budget(&image, 4).unwrap();
    let pipeline = DreamXPipeline::load(main, mmproj, 2).unwrap();
    let estimate = pipeline
        .estimate(&DreamXRequest {
            image,
            prompt: "A man speaking while seated on a yellow couch.".into(),
            negative_prompt: String::new(),
            output: std::env::temp_dir().join("dreamx-preflight.mp4"),
            options: DreamXOptions {
                duration_seconds: 0.2,
                fps: 5,
                steps: 1,
                target_spatial_tokens: 4,
                ..DreamXOptions::default()
            },
            overwrite: true,
            allow_memory_overcommit: true,
        })
        .unwrap();
    assert_eq!(estimate.spatial_tokens, 4);
    assert_eq!(estimate.audio_sample_rate, 48_000);
}
