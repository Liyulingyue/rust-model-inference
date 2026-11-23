use super::*;
use crate::core::tensor::{MetaValue, TensorInfo, TensorSource};

#[test]
fn f32_projections_borrow_mapped_weights_and_reject_invalid_storage() {
    use crate::core::tensor::GGMLType;
    struct Source {
        info: TensorInfo,
        data: Vec<f32>,
    }
    impl TensorSource for Source {
        fn metadata(&self, _: &str) -> Option<&MetaValue> {
            None
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            (name == "weight").then_some(&self.info)
        }
        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            (name == "weight").then(|| bytemuck::cast_slice(self.data.as_slice()))
        }
    }
    let mut source = Source {
        info: TensorInfo {
            name: "weight".into(),
            dims: vec![2, 2],
            ggml_type: GGMLType::F32,
            offset: 0,
        },
        data: vec![1., 2., 3., 4.],
    };
    {
        let projection = weights::load_linear(&source, "weight", None, 2, 2).unwrap();
        if cfg!(target_endian = "little") {
            assert_eq!(
                projection
                    .binding()
                    .weight
                    .kernel
                    .weight_bytes()
                    .unwrap()
                    .as_ptr(),
                source.data.as_ptr().cast::<u8>()
            );
        }
        let mut row = [0.; 2];
        projection.embedding_lookup(1, &mut row).unwrap();
        assert_eq!(row, [3., 4.]);
    }
    assert!(weights::load_linear(&source, "missing", None, 2, 2).is_err());
    assert!(weights::load_linear(&source, "weight", None, 1, 4).is_err());
    source.data.pop();
    assert!(weights::load_linear(&source, "weight", None, 2, 2).is_err());
}

#[test]
#[ignore = "requires the pinned tokenizer on the model host"]
fn pinned_tokenizer_matches_official_unicode_cleaning_and_truncation() {
    struct TokenizerSource(MetaValue);
    impl TensorSource for TokenizerSource {
        fn metadata(&self, name: &str) -> Option<&MetaValue> {
            (name == "fastframes_l.tokenizer_json").then_some(&self.0)
        }
        fn tensor_info(&self, _: &str) -> Option<&TensorInfo> {
            None
        }
        fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
            None
        }
    }
    let path = std::env::var("RMI_FASTFRAMES_TOKENIZER").expect("set pinned tokenizer path");
    let source = TokenizerSource(MetaValue::String(std::fs::read_to_string(path).unwrap()));
    let tokenizer = text::tokenizer(&source).unwrap();
    let fixtures: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../tools/oracle/fastframes_l/tokenizer-fixtures.json"
    ))
    .unwrap();
    for case in fixtures["cases"].as_array().unwrap() {
        let prompt = case["prompt"].as_str().unwrap();
        assert_eq!(text::clean(prompt), case["cleaned"].as_str().unwrap());
        let expected: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
        assert_eq!(text::ids(&tokenizer, prompt).unwrap(), expected, "{prompt}");
    }
}

#[test]
fn released_geometry_and_trimmed_video_lengths() {
    let options = Options::default();
    assert!(options.validate().is_ok());
    assert_eq!(options.latent_frames(), 21);
    for (frames, expected) in [(1, 1), (2, 2), (4, 2), (5, 2), (80, 21), (81, 21)] {
        let options = Options {
            frames,
            ..Options::default()
        };
        assert!(options.validate().is_ok());
        assert_eq!(options.latent_frames(), expected);
    }
}

#[test]
fn invalid_geometry_is_rejected_before_allocating() {
    for options in [
        Options {
            width: 0,
            ..Options::default()
        },
        Options {
            width: 17,
            ..Options::default()
        },
        Options {
            width: usize::MAX,
            ..Options::default()
        },
        Options {
            height: 496,
            ..Options::default()
        },
        Options {
            frames: 0,
            ..Options::default()
        },
        Options {
            frames: 82,
            ..Options::default()
        },
        Options {
            fps: 0,
            ..Options::default()
        },
    ] {
        assert!(options.validate().is_err(), "{options:?}");
    }
}

#[test]
fn flow_schedule_uses_nearest_shifted_training_entry() {
    // The released integer 833 is nearer to base sigma .499 than to .500.
    assert!((sigma(833.) - 2.495f32 / 2.996).abs() < 2e-7);
    assert!((sigma(937.) - 3.74f32 / 3.992).abs() < 2e-7);
    assert_eq!(sigma(1000.), 1.);
    assert_eq!(sigma(625.), 0.625);
}

#[test]
fn patch_embedding_and_flow_head_have_distinct_channel_orders() {
    let latent: Vec<_> = (0..16 * 4 * 4).map(|i| i as f32).collect();
    let patches = patchify(&latent, 4, 4);
    assert_eq!(patches.len(), latent.len());
    assert_eq!(&patches[..8], &[0., 1., 4., 5., 16., 17., 20., 21.]);
    let tokens: Vec<_> = (0..4 * 64).map(|i| i as f32).collect();
    let flow = unpatchify(&tokens, 4, 4);
    assert_eq!(&flow[..8], &[0., 16., 64., 80., 32., 48., 96., 112.]);
    assert_eq!(&flow[16..20], &[1., 17., 65., 81.]);
}

#[test]
fn key_repair_preserves_each_heads_norm() {
    let mut k = vec![3., 4., 0., 2.];
    refiner::repair_keys(&mut k, &[1., -2., 2., 0.], 2);
    assert!((k[0].hypot(k[1]) - 5.).abs() < 3e-6);
    assert!((k[2].hypot(k[3]) - 2.).abs() < 3e-6);
}

#[test]
fn rope_uses_frame_offset_and_independent_spatial_axes() {
    let mut x = vec![0.; 128];
    for pair in x.chunks_exact_mut(2) {
        pair[0] = 1.;
    }
    math::rope(&mut x, 1, 1, 1, 1);
    assert!((x[0] - 1f32.cos()).abs() < 1e-6);
    assert!((x[1] - 1f32.sin()).abs() < 1e-6);
    assert_eq!(x[44], 1.);
    assert_eq!(x[45], 0.);
    assert_eq!(x[86], 1.);
    assert_eq!(x[87], 0.);
}

#[test]
fn depthwise_decoder_never_reads_a_future_frame() {
    let pool = ComputePool::new(1);
    let mut weight = vec![0.; 27];
    weight[4] = 1.;
    weight[13] = 10.;
    weight[22] = 100.;
    let y = decoder::causal_depthwise(&pool, &[1., 2., 9.], 1, 3, 1, 1, &weight, &[0.]).unwrap();
    assert_eq!(y, [100., 210., 921.]);
}

#[test]
fn decoder_drops_leading_frames_and_uses_ties_even() {
    let mut pixels = vec![-1.; 4 * 16 * 16 * 3];
    pixels[3 * 16 * 16 * 3] = 0.;
    let rgb = decoder::unpatchify_pixels(&pixels, 1, 1, 1, 1);
    assert_eq!(rgb.len(), 16 * 16 * 3);
    assert_eq!(&rgb[..3], &[128, 0, 0]);
}

#[test]
fn umt5_buckets_are_bidirectional_and_saturate() {
    for n in 1..8 {
        assert_eq!(text::bucket(n), 16 + n as usize);
        assert_eq!(text::bucket(-n), n as usize);
    }
    assert_eq!(text::bucket(0), 0);
    assert_eq!(text::bucket(128), 31);
    assert_eq!(text::bucket(-1000), 15);
}

#[test]
fn prompt_cleaning_matches_unicode_whitespace_entities_and_encoding_fixtures() {
    for (input, expected) in [
        ("café 漢字 😀", "café 漢字 😀"),
        ("cafÃ©", "café"),
        ("ＡＢＣ", "ABC"),
        ("“hello”", "\"hello\""),
        ("A&amp;amp;B", "A&B"),
        ("a\u{A0}\tb", "a b"),
        ("ﬁ ﬂ", "fi fl"),
        ("e\u{301}", "é"),
    ] {
        assert_eq!(text::clean(input), expected);
    }
}

#[test]
fn contract_expands_all_components_and_grouped_shapes() {
    let shapes = weights::shapes(&weights::Variant::L);
    assert_eq!(shapes["refiner.heads.39.dK.weight"], [5120, 768]);
    assert_eq!(shapes["dit.patch_embedding.weight"], [5120, 64]);
    assert_eq!(
        shapes["decoder.pixel_head.conv_low.0.dw.weight"],
        [1536, 1, 3, 3, 3]
    );
    assert_eq!(shapes["text.blocks.23.attn.q.weight"], [4096, 4096]);
    assert_eq!(shapes.len(), 1095 + 32 + 242 + 49 + 40 * 17);
}

#[test]
fn s_contract_expands_the_1_3b_dit_and_variant_specs() {
    let shapes = weights::shapes(&weights::Variant::S);
    assert_eq!(shapes["refiner.heads.29.dK.weight"], [1536, 768]);
    assert_eq!(shapes["dit.patch_embedding.weight"], [1536, 64]);
    assert_eq!(shapes["text.blocks.23.attn.q.weight"], [4096, 4096]);
    assert_eq!(shapes["dit.head.head.weight"], [64, 1536]);
    assert_eq!(shapes["dit.blocks.29.ffn.0.weight"], [8960, 1536]);
    assert!(!shapes.contains_key("decoder.bridge.weight"));
    // The FF-S recipe: fractional first-frame timesteps, sequential noise.
    let spec = weights::Variant::S.spec();
    assert_eq!(spec.dims.layers, 30);
    assert_eq!(spec.dims.heads, 12);
    assert_eq!(spec.head_dim(), 128);
    assert!(matches!(spec.recipe.noise, weights::NoiseMode::Sequential));
    assert_eq!(spec.recipe.first_frame[1], 937.5);
    assert_eq!(spec.recipe.timesteps[1], 833.3333129882812);
    // FF-L keeps its whole-frame pregenerated transition layout.
    let l = weights::Variant::L.spec();
    assert!(matches!(l.recipe.noise, weights::NoiseMode::Pregenerated));
    assert_eq!(l.dims.layers, 40);
}

#[test]
fn prompt_cleaning_repairs_cp1252_and_html_numeric_references() {
    for (input, expected) in [
        ("doesnâ€™t", "doesn't"),
        ("&#128;", "€"),
        ("a&#11;b", "ab"),
        ("café and cafÃ© 漢字", "café and café 漢字"),
        ("\u{1b}[31mred\u{1b}[0m", "red"),
        ("&ldquo;hello&rdquo;", "\"hello\""),
        ("e&#769;", "é"),
        ("ﬀ ﬃ ﬄ ﬅ ﬆ", "ff ffi ffl st st"),
    ] {
        assert_eq!(text::clean(input), expected);
    }
}
