//! Omni-modal embedding: image/video/audio → dense embedding vector.
//!
//! Used by Jina Embeddings v5 Omni retrieval models. Shared media types
//! and decoding utilities live in `app/media.rs`.

use crate::app::cli::EmbeddingOutput;
use crate::app::media::{
    append_media_markers, decode_audio, decode_video, frame_pairs, marker_names, MediaKind,
};
use crate::core::tensor::{MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::format::ggufrs::{open_model_source, ComponentRole};
use crate::models::qwen3::embedding::{print_embedding, run_embedding_tokens, MediaEmbeddings};
use crate::models::qwen3::omni::encode_audio;
use crate::models::qwen35::vision::{qwen_smart_resize, VisionEncoder, VisionScratchpad};
use std::path::Path;
use std::sync::Arc;

fn encode_vision(
    mmproj_path: &Path,
    image_path: Option<&Path>,
    video_path: Option<&Path>,
) -> Result<(MediaKind, Vec<f32>, Vec<usize>), String> {
    let mmproj = open_model_source(mmproj_path, ComponentRole::Mmproj)
        .map_err(|error| format!("Failed to load mmproj {}: {error}", mmproj_path.display()))?;
    let mut encoder = VisionEncoder::from_source(mmproj.as_ref())
        .map_err(|error| format!("Failed to load vision encoder: {error}"))?;
    encoder.precompute();
    let mut frames = if let Some(path) = image_path {
        vec![crate::app::media::decode_image(path)?]
    } else {
        decode_video(video_path.ok_or("Missing image or video input")?)?
    };
    let is_video = video_path.is_some();
    if is_video {
        encoder.config.image_min_pixels = encoder.config.video_min_pixels;
        encoder.config.image_max_pixels = encoder.config.video_max_pixels;
    }
    let first = frames.first().ok_or("Media produced no frames")?;
    let grid = qwen_smart_resize(
        first.width() as usize,
        first.height() as usize,
        &encoder.config,
    )?;
    let normalized = frames
        .drain(..)
        .map(|frame| {
            crate::app::media::normalize_resized_image(
                &frame,
                grid.image_width(),
                grid.image_height(),
                &encoder.config.image_mean,
                &encoder.config.image_std,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    #[cfg(feature = "parity-trace")]
    for frame in &normalized {
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "omni.vision.normalized",
            None,
            &[grid.image_height(), grid.image_width(), 3],
            frame,
        ));
    }
    let pairs = if is_video {
        frame_pairs(normalized.len())
    } else {
        vec![(0, 0)]
    };
    let mut values = Vec::new();
    let mut block_rows = Vec::with_capacity(pairs.len());
    let mut scratch = VisionScratchpad::new(&encoder.config);
    let pool = Arc::new(ComputePool::new(
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    ));
    for (a, b) in pairs {
        let encoded_grid = encoder.encode_pair(
            &normalized[a],
            &normalized[b],
            grid.image_width(),
            grid.image_height(),
            &mut scratch,
            &pool,
        )?;
        if encoded_grid != grid {
            return Err("Vision grid changed during encoding".into());
        }
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "omni.vision.projected",
            None,
            &[grid.token_count(), encoder.config.projection_dim],
            &scratch.projected,
        ));
        values.extend_from_slice(&scratch.projected);
        block_rows.push(grid.token_count());
    }
    Ok((
        if is_video {
            MediaKind::Video
        } else {
            MediaKind::Image
        },
        values,
        block_rows,
    ))
}

fn encode_audio_file(
    mmproj_path: &Path,
    audio_path: &Path,
    threads: usize,
) -> Result<(MediaKind, Vec<f32>), String> {
    let source: Arc<dyn TensorSource> = Arc::from(
        open_model_source(mmproj_path, ComponentRole::Mmproj)
            .map_err(|error| format!("Failed to load mmproj {}: {error}", mmproj_path.display()))?,
    );
    let samples = decode_audio(audio_path)?;
    Ok((MediaKind::Audio, encode_audio(source, &samples, threads)?))
}

pub fn run_omni_embedding(
    source: &dyn TensorSource,
    mmproj_path: &Path,
    image_path: Option<&Path>,
    video_path: Option<&Path>,
    audio_path: Option<&Path>,
    prompt: &str,
    threads: usize,
    output: EmbeddingOutput,
) -> Result<(), String> {
    let started = std::time::Instant::now();
    let arch = source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        .unwrap_or_default();
    if source.metadata(&format!("{arch}.pooling_type")).is_none() {
        return Err(format!(
            "embedding mode requires a Jina embedding model (got architecture {arch})"
        ));
    }
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
    let media_count = usize::from(image_path.is_some())
        + usize::from(video_path.is_some())
        + usize::from(audio_path.is_some());
    if media_count != 1 {
        return Err("Omni embedding requires exactly one image, video, or audio input".into());
    }
    let (kind, media, block_rows) = if let Some(path) = audio_path {
        let (kind, media) = encode_audio_file(mmproj_path, path, threads)?;
        let rows = media.len() / 1024;
        (kind, media, vec![rows])
    } else {
        encode_vision(mmproj_path, image_path, video_path)?
    };
    let rows = media.len() / 1024;
    if rows == 0 || media.len() % 1024 != 0 {
        return Err("Media projector output is not 1024-wide".into());
    }
    if block_rows.iter().sum::<usize>() != rows || block_rows.contains(&0) {
        return Err("Media block rows do not match projector output".into());
    }
    let mut tokens = tokenizer.encode(
        prompt,
        EncodeOptions {
            add_special: true,
            parse_special: false,
        },
    );
    let (start_name, pad_name, end_name) = marker_names(kind);
    let start = tokenizer
        .special_token_id(start_name)
        .ok_or_else(|| format!("Required special token missing: {start_name}"))?;
    let placeholder_id = tokenizer
        .special_token_id(pad_name)
        .ok_or_else(|| format!("Required special token missing: {pad_name}"))?;
    let end = tokenizer
        .special_token_id(end_name)
        .ok_or_else(|| format!("Required special token missing: {end_name}"))?;
    append_media_markers(
        &mut tokens,
        &tokenizer,
        kind,
        start,
        placeholder_id,
        end,
        &block_rows,
    );
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::token_ids("omni.tokens", &tokens));
    let embedding = run_embedding_tokens(
        source,
        &tokens,
        Some(MediaEmbeddings {
            placeholder_id,
            values: &media,
        }),
        threads,
    )?;
    print_embedding(&embedding, output, started.elapsed().as_millis());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::media::{
        append_media_markers, frame_pairs, marker_names, vision_markers, MediaKind,
    };

    #[test]
    fn video_frames_are_paired_in_order_and_duplicate_an_odd_tail() {
        assert_eq!(frame_pairs(4), vec![(0, 1), (2, 3)]);
        assert_eq!(frame_pairs(5), vec![(0, 1), (2, 3), (4, 4)]);
    }

    #[test]
    fn vision_markers_wrap_the_modality_specific_pad() {
        assert_eq!(vision_markers(10, 20, 30, 2), vec![10, 20, 20, 30]);
        assert_eq!(vision_markers(10, 21, 30, 1), vec![10, 21, 30]);
    }

    #[test]
    fn video_marker_assembly_keeps_each_pair_separate_and_adds_timestamps() {
        let tokenizer = BPETokenizer::from_qwen3_embedded_merges().unwrap();
        let mut tokens = vec![999];

        append_media_markers(
            &mut tokens,
            &tokenizer,
            MediaKind::Video,
            151_652,
            151_656,
            151_653,
            &[2, 1],
        );

        assert_eq!(
            tokens,
            [
                999, 27, 15, 13, 17, 6486, 29, 151_652, 151_656, 151_656, 151_653, 27, 16, 13, 17,
                6486, 29, 151_652, 151_656, 151_653,
            ]
        );
    }

    #[test]
    fn audio_uses_its_own_token_family() {
        assert_eq!(
            marker_names(MediaKind::Audio),
            ("audio_start", "audio_pad", "audio_end")
        );
        assert_eq!(
            marker_names(MediaKind::Image),
            ("vision_start", "image_pad", "vision_end")
        );
        assert_eq!(
            marker_names(MediaKind::Video),
            ("vision_start", "video_pad", "vision_end")
        );
    }

    #[test]
    #[ignore = "requires Jina v5 Omni retrieval GGUF"]
    fn jina_audio_markers_use_the_models_real_token_ids() {
        let model = std::env::var("RMI_JINA_OMNI_MODEL").unwrap();
        let source = crate::GGUFLoader::from_file(Path::new(&model)).unwrap();
        let tokenizer =
            BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).unwrap();
        let (start_name, pad_name, end_name) = marker_names(MediaKind::Audio);
        let start = tokenizer.special_token_id(start_name).unwrap();
        let pad = tokenizer.special_token_id(pad_name).unwrap();
        let end = tokenizer.special_token_id(end_name).unwrap();
        assert_eq!((start, pad, end), (151_670, 151_669, 151_671));

        let mut tokens = tokenizer.encode(
            "Represent this audio for retrieval.",
            EncodeOptions {
                add_special: true,
                parse_special: false,
            },
        );
        let prompt_len = tokens.len();
        append_media_markers(
            &mut tokens,
            &tokenizer,
            MediaKind::Audio,
            start,
            pad,
            end,
            &[750],
        );
        assert_eq!(tokens.len(), prompt_len + 752);
        assert_eq!(tokens[prompt_len], start);
        assert!(tokens[prompt_len + 1..prompt_len + 751]
            .iter()
            .all(|&token| token == pad));
        assert_eq!(tokens[prompt_len + 751], end);
    }

    #[test]
    fn generative_qwen_architecture_rejects_omni_embedding_mode() {
        use crate::core::tensor::{MetaValue, TensorInfo, TensorSource};
        use std::collections::HashMap;

        struct Source {
            metadata: HashMap<String, MetaValue>,
        }

        impl TensorSource for Source {
            fn metadata(&self, key: &str) -> Option<&MetaValue> {
                self.metadata.get(key)
            }
            fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
                None
            }
            fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
                None
            }
        }

        let source = Source {
            metadata: HashMap::from([(
                "general.architecture".into(),
                MetaValue::String("qwen2vl".into()),
            )]),
        };
        let error = run_omni_embedding(
            &source,
            Path::new("unused.mmproj"),
            Some(Path::new("image.png")),
            None,
            None,
            "prompt",
            1,
            EmbeddingOutput::Summary,
        )
        .unwrap_err();
        assert!(error.contains("Jina embedding model"), "{error}");
    }

    #[test]
    fn multimodal_capabilities_match_qwen_projector_families() {
        use crate::app::media::validate_mmproj_capabilities;
        use crate::app::media::ProjectorFamily;
        use crate::core::tensor::{MetaValue, TensorInfo, TensorSource};
        use std::collections::HashMap;

        struct Source {
            metadata: HashMap<String, MetaValue>,
        }

        impl TensorSource for Source {
            fn metadata(&self, key: &str) -> Option<&MetaValue> {
                self.metadata.get(key)
            }
            fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
                None
            }
            fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
                None
            }
        }

        let qwen25 = Source {
            metadata: HashMap::from([
                (
                    "clip.projector_type".into(),
                    MetaValue::String("qwen2.5o".into()),
                ),
                ("clip.has_vision_encoder".into(), MetaValue::Bool(true)),
            ]),
        };
        assert_eq!(
            validate_mmproj_capabilities("qwen2vl", &qwen25, MediaKind::Video),
            Ok(ProjectorFamily::Qwen25Omni)
        );

        let qwen3 = Source {
            metadata: HashMap::from([
                (
                    "clip.projector_type".into(),
                    MetaValue::String("qwen3vl_merger".into()),
                ),
                ("clip.has_audio_encoder".into(), MetaValue::Bool(true)),
            ]),
        };
        assert_eq!(
            validate_mmproj_capabilities("qwen3vlmoe", &qwen3, MediaKind::Audio),
            Ok(ProjectorFamily::Qwen3VlMerger)
        );
    }
}
