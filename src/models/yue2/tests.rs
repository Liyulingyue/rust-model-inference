use std::collections::HashMap;
use std::sync::Arc;

use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;

use super::ar::{attention_head, sample_phase_token, torch_bf16_matmul_rows, Phase};
use super::config::YuE2VaeConfig;
use super::nar::{
    midpoint_times, song_chunks, torch_randn, visibility_mask, Mt19937, YuE2Chunk, YuE2NarSession,
};
use super::protocol::{
    YuE2Protocol, YuE2Request, ABC_END, ABC_START, CODEC_OFFSET, CONTEXT, EOD, MUSIC_END,
    MUSIC_START, VOCAB_SIZE,
};
use super::vae::{materialize_weight_norm, YuE2Vae};
use super::{SamplingConfig, YuE2Config, YuE2Model};

fn synthetic_vae_source() -> MapTensorSource {
    let mut source = MapTensorSource::default();
    for (key, value) in [
        ("general.architecture", MetaValue::String("yue2_vae".into())),
        (
            "yue2_vae.release_variant",
            MetaValue::String("standard".into()),
        ),
        ("yue2_vae.latent_channels", MetaValue::Uint64(64)),
        ("yue2_vae.output_channels", MetaValue::Uint64(2)),
        ("yue2_vae.sample_rate", MetaValue::Uint64(48_000)),
        ("yue2_vae.downsampling_ratio", MetaValue::Uint64(1920)),
        ("yue2_vae.decode_core_frames", MetaValue::Uint64(1024)),
        ("yue2_vae.decode_halo_frames", MetaValue::Uint64(16)),
        ("yue2_vae.tensor_count", MetaValue::Uint64(217)),
        (
            "yue2_vae.strides",
            MetaValue::Array(
                MetaValueType::Uint32,
                [2, 2, 4, 4, 5, 6]
                    .into_iter()
                    .map(MetaValue::Uint32)
                    .collect(),
            ),
        ),
    ] {
        source.insert_meta(key, value);
    }
    source
}

#[test]
fn vae_requires_standard_f32_decoder_contract() {
    let config = YuE2VaeConfig::from_source(&synthetic_vae_source()).unwrap();
    assert_eq!(config.strides, [2, 2, 4, 4, 5, 6]);
    assert_eq!(
        (
            config.latent_channels,
            config.output_channels,
            config.sample_rate
        ),
        (64, 2, 48_000)
    );
    let mut legacy = synthetic_vae_source();
    legacy.insert_meta(
        "yue2_vae.release_variant",
        MetaValue::String("legacy".into()),
    );
    assert!(YuE2VaeConfig::from_source(&legacy)
        .unwrap_err()
        .contains("release_variant"));
    let mut wrong_strides = synthetic_vae_source();
    wrong_strides.insert_meta(
        "yue2_vae.strides",
        MetaValue::Array(
            MetaValueType::Uint64,
            [2, 2, 4, 4, 5, 6]
                .into_iter()
                .map(MetaValue::Uint64)
                .collect(),
        ),
    );
    assert!(YuE2VaeConfig::from_source(&wrong_strides)
        .unwrap_err()
        .contains("strides"));
}

#[test]
fn vae_weight_norm_materialization_uses_channel_norms() {
    let filter = materialize_weight_norm(&[2.0], &[3.0, 4.0], 1, 2).unwrap();
    assert_eq!(filter, vec![1.2, 1.6]);
    assert!(materialize_weight_norm(&[1.0], &[0.0, 0.0], 1, 2).is_err());
    let values = (0..448)
        .map(|index| ((index % 11) as f32 - 5.0) * 0.13)
        .collect::<Vec<_>>();
    assert_eq!(
        materialize_weight_norm(&[2.0], &values, 1, 448).unwrap()[0].to_bits(),
        0xbe1954fa
    );
}

#[test]
fn vae_full_and_tiled_decode_have_natural_length_and_bits() {
    let vae = YuE2Vae::tiny_for_test();
    let latents: Vec<f32> = (0..64 * 33)
        .map(|index| (index % 11) as f32 * 0.01)
        .collect();
    let full = vae.decode(&latents, 33).unwrap();
    let tiled = vae.decode_tiled(&latents, 33, 16, 16).unwrap();
    assert_eq!(full.len(), 2 * (1920 * 33 - 64));
    assert_eq!(
        full.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        tiled.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn pipeline_runs_in_order_and_interleaves_stereo_once() {
    let mut stages = Vec::new();
    let generation = run_tiny_pipeline(|stage| stages.push(stage)).unwrap();
    assert_eq!(stages, ["abc", "semantic", "nar", "vae"]);
    assert_eq!(
        super::interleave_stereo(&[1.0, 2.0, 10.0, 20.0], 2).unwrap(),
        vec![1.0, 10.0, 2.0, 20.0],
    );
    assert_eq!(generation.samples_per_channel, 2);
}

fn run_tiny_pipeline(on_stage: impl FnMut(&'static str)) -> Result<super::YuE2Generation, String> {
    super::run_pipeline(
        || Ok(vec![1, 2]),
        |_| Ok(vec![CODEC_OFFSET]),
        |_, _| Ok((vec![0.0; 64], 1)),
        |_, _| Ok(vec![1.0, 2.0, 10.0, 20.0]),
        on_stage,
    )
}

#[derive(Default)]
pub(crate) struct MapTensorSource {
    metadata: HashMap<String, MetaValue>,
    tensors: HashMap<String, TensorInfo>,
}

impl MapTensorSource {
    pub(crate) fn insert_meta(&mut self, key: impl Into<String>, value: MetaValue) {
        self.metadata.insert(key.into(), value);
    }

    pub(crate) fn remove_tensor(&mut self, name: &str) {
        self.tensors.remove(name);
    }

    pub(crate) fn replace_tensor_shape(&mut self, name: &str, dims: &[u64]) {
        self.tensors.get_mut(name).unwrap().dims = dims.to_vec();
    }
}

impl TensorSource for MapTensorSource {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
        None
    }
}

pub(crate) fn protocol_source() -> MapTensorSource {
    let mut source = MapTensorSource::default();
    for (key, value) in [
        ("general.architecture", MetaValue::String("yue2".into())),
        (
            "yue2.protocol_version",
            MetaValue::String("yue2-native-v1".into()),
        ),
        ("yue2.context_length", MetaValue::Uint32(24_576)),
        ("yue2.vocab_size", MetaValue::Uint32(184_704)),
        ("yue2.eod_token_id", MetaValue::Uint32(151_643)),
        ("yue2.abc_start_token_id", MetaValue::Uint32(151_847)),
        ("yue2.abc_end_token_id", MetaValue::Uint32(151_848)),
        ("yue2.music_start_token_id", MetaValue::Uint32(151_851)),
        ("yue2.music_end_token_id", MetaValue::Uint32(151_852)),
        ("yue2.codec_offset", MetaValue::Uint32(151_853)),
        ("yue2.codec_size", MetaValue::Uint32(32_768)),
        ("yue2.latent_start_token_id", MetaValue::Uint32(184_621)),
        ("yue2.latent_end_token_id", MetaValue::Uint32(184_622)),
        ("yue2.latent_pad_token_id", MetaValue::Uint32(184_623)),
        ("yue2.abc.temperature", MetaValue::Float64(0.7)),
        ("yue2.abc.top_p", MetaValue::Float64(0.9)),
        ("yue2.abc.top_k", MetaValue::Uint32(30)),
        ("yue2.abc.repetition_penalty", MetaValue::Float64(1.005)),
        ("yue2.abc.penalty_window", MetaValue::Uint32(100)),
        ("yue2.abc.min_tokens", MetaValue::Uint32(32)),
        ("yue2.abc.max_tokens", MetaValue::Uint32(4096)),
        ("yue2.semantic.temperature", MetaValue::Float64(1.0)),
        ("yue2.semantic.top_p", MetaValue::Float64(0.95)),
        ("yue2.semantic.top_k", MetaValue::Uint32(100)),
        ("yue2.semantic.repetition_penalty", MetaValue::Float64(1.2)),
        ("yue2.semantic.penalty_window", MetaValue::Uint32(50)),
        ("yue2.semantic.min_tokens", MetaValue::Uint32(200)),
        ("yue2.semantic.max_tokens", MetaValue::Uint32(9000)),
    ] {
        source.insert_meta(key, value);
    }
    source
}

fn byte_encoder() -> Vec<String> {
    let mut visible: Vec<u16> = (b'!'..=b'~').map(u16::from).collect();
    visible.extend((0xA1u16)..=0xAC);
    visible.extend((0xAEu16)..=0xFF);
    let extra: Vec<u16> = (0u16..=255)
        .filter(|byte| !visible.contains(byte))
        .collect();
    let mut bytes = visible.clone();
    bytes.extend(&extra);
    let mut chars = visible;
    chars.extend((0..extra.len()).map(|index| 256 + index as u16));
    let mut table = vec![String::new(); 256];
    for (byte, character) in bytes.into_iter().zip(chars) {
        table[byte as usize] = char::from_u32(character as u32).unwrap().to_string();
    }
    table
}

pub(crate) fn tokenizer_fixture() -> BPETokenizer {
    let mut tokens: Vec<String> = (0..VOCAB_SIZE)
        .map(|id| format!("<yue2_test_unused_{id}>"))
        .collect();
    let bytes = byte_encoder();
    tokens[..bytes.len()].clone_from_slice(&bytes);
    tokens[EOD as usize] = "<|endoftext|>".into();
    tokens[ABC_START as usize] = "<abc>".into();
    tokens[ABC_END as usize] = "</abc>".into();
    tokens[MUSIC_START as usize] = "<music>".into();

    let mut types = vec![MetaValue::Uint32(5); VOCAB_SIZE];
    types[..bytes.len()].fill(MetaValue::Uint32(1));
    for id in [EOD, ABC_START, ABC_END] {
        types[id as usize] = MetaValue::Uint32(4);
    }
    let metadata = HashMap::from([
        (
            "tokenizer.ggml.model".to_string(),
            MetaValue::String("gpt2".into()),
        ),
        (
            "tokenizer.ggml.pre".to_string(),
            MetaValue::String("qwen2".into()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            MetaValue::Array(
                MetaValueType::String,
                tokens.into_iter().map(MetaValue::String).collect(),
            ),
        ),
        (
            "tokenizer.ggml.token_type".to_string(),
            MetaValue::Array(MetaValueType::Uint32, types),
        ),
        (
            "tokenizer.ggml.merges".to_string(),
            MetaValue::Array(MetaValueType::String, Vec::new()),
        ),
        (
            "tokenizer.ggml.add_bos_token".to_string(),
            MetaValue::Bool(false),
        ),
        (
            "tokenizer.ggml.add_eos_token".to_string(),
            MetaValue::Bool(false),
        ),
        (
            "tokenizer.ggml.normalizer.nfc".to_string(),
            MetaValue::Bool(true),
        ),
    ]);
    BPETokenizer::from_gguf_metadata(|key| metadata.get(key).cloned()).unwrap()
}

#[test]
fn full_cot_prompt_and_prefix_are_checkpoint_native() {
    let source = protocol_source();
    let protocol = YuE2Protocol::from_source(&source).unwrap();
    let tokenizer = tokenizer_fixture();
    let request = YuE2Request::new("jazz, warm", "[Verse]\n你好", 831001).unwrap();
    assert_eq!(
        protocol.prompt_text(&request),
        "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.\n[Tags]\njazz, warm\n[Lyrics]\n[Verse]\n你好\n"
    );
    let prefix = protocol.abc_prefix(&tokenizer, &request).unwrap();
    assert_eq!(prefix[0], EOD);
    assert_eq!(prefix.last(), Some(&ABC_START));
}

#[test]
fn protocol_rejects_empty_text_and_context_overflow() {
    let protocol = YuE2Protocol::from_source(&protocol_source()).unwrap();
    assert!(YuE2Request::new(" ", "lyrics", 1)
        .unwrap_err()
        .contains("style"));
    assert!(YuE2Request::new("style", "\n", 1)
        .unwrap_err()
        .contains("lyrics"));
    assert!(protocol
        .validate_generation(CONTEXT - 1, 2)
        .unwrap_err()
        .contains("24576"));
}

#[test]
fn protocol_rejects_special_id_drift() {
    let mut source = protocol_source();
    source.insert_meta("yue2.music_start_token_id", MetaValue::Uint32(1));
    assert!(YuE2Protocol::from_source(&source)
        .unwrap_err()
        .contains("yue2.music_start_token_id"));
}

#[test]
fn semantic_prefix_rejects_special_abc_ids_and_closes_once() {
    let protocol = YuE2Protocol::from_source(&protocol_source()).unwrap();
    let tokenizer = tokenizer_fixture();
    let request = YuE2Request::new("jazz", "lyrics", 1).unwrap();
    assert!(protocol
        .semantic_prefix(&tokenizer, &request, &[1, EOD])
        .unwrap_err()
        .contains("ABC"));
    let prefix = protocol
        .semantic_prefix(&tokenizer, &request, &[1, 2])
        .unwrap();
    assert_eq!(prefix[prefix.len() - 2..], [ABC_END, MUSIC_START]);
    assert_eq!(prefix.iter().filter(|&&id| id == ABC_END).count(), 1);
    assert_eq!(prefix.iter().filter(|&&id| id == MUSIC_START).count(), 1);
}

#[allow(dead_code)]
fn add_test_tensor(source: &mut MapTensorSource, name: &str, dims: &[u64]) {
    source.tensors.insert(
        name.into(),
        TensorInfo {
            name: name.into(),
            dims: dims.to_vec(),
            ggml_type: GGMLType::BF16,
            offset: 0,
        },
    );
}

pub(crate) fn synthetic_yue2_source() -> MapTensorSource {
    let mut source = protocol_source();
    for (key, value) in [
        ("yue2.embedding_length", MetaValue::Uint32(2048)),
        ("yue2.block_count", MetaValue::Uint32(28)),
        ("yue2.attention.head_count", MetaValue::Uint32(16)),
        ("yue2.attention.head_count_kv", MetaValue::Uint32(8)),
        ("yue2.attention.head_dim", MetaValue::Uint32(128)),
        ("yue2.feed_forward_length", MetaValue::Uint32(6144)),
        ("yue2.rms_norm_eps", MetaValue::Float64(0.000001)),
        ("yue2.rope.freq_base", MetaValue::Uint32(1_000_000)),
        ("yue2.latent_channels", MetaValue::Uint32(64)),
        ("yue2.timestep_shift", MetaValue::Float64(1.0)),
        ("yue2.tensor_count", MetaValue::Uint32(628)),
    ] {
        source.insert_meta(key, value);
    }

    for (name, dims) in [
        ("model.embed_tokens.weight", &[2048, 184704][..]),
        ("model.norm.weight", &[2048]),
        ("lm_head.weight", &[2048, 184704]),
        ("llm2vae.weight", &[2048, 64]),
        ("llm2vae.bias", &[64]),
        ("vae2llm.weight", &[64, 2048]),
        ("vae2llm.bias", &[2048]),
        ("time_embedder.mlp.0.weight", &[256, 2048]),
        ("time_embedder.mlp.0.bias", &[2048]),
        ("time_embedder.mlp.2.weight", &[2048, 2048]),
        ("time_embedder.mlp.2.bias", &[2048]),
        ("latent_pos_embed.pe", &[2048, 24576]),
    ] {
        add_test_tensor(&mut source, name, dims);
    }

    for layer in 0..28 {
        for (suffix, dims) in [
            ("input_layernorm.weight", &[2048][..]),
            ("self_attn.q_proj.weight", &[2048, 2048]),
            ("self_attn.k_proj.weight", &[2048, 1024]),
            ("self_attn.v_proj.weight", &[2048, 1024]),
            ("self_attn.o_proj.weight", &[2048, 2048]),
            ("self_attn.q_norm.weight", &[128]),
            ("self_attn.k_norm.weight", &[128]),
            ("post_attention_layernorm.weight", &[2048]),
            ("mlp.gate_proj.weight", &[2048, 6144]),
            ("mlp.up_proj.weight", &[2048, 6144]),
            ("mlp.down_proj.weight", &[6144, 2048]),
            ("nar_input_layernorm.weight", &[2048]),
            ("nar_self_attn.q_proj.weight", &[2048, 2048]),
            ("nar_self_attn.k_proj.weight", &[2048, 1024]),
            ("nar_self_attn.v_proj.weight", &[2048, 1024]),
            ("nar_self_attn.o_proj.weight", &[2048, 2048]),
            ("nar_self_attn.q_norm.weight", &[128]),
            ("nar_self_attn.k_norm.weight", &[128]),
            ("nar_pre_mlp_layernorm.weight", &[2048]),
            ("nar_mlp.gate_proj.weight", &[2048, 6144]),
            ("nar_mlp.up_proj.weight", &[2048, 6144]),
            ("nar_mlp.down_proj.weight", &[6144, 2048]),
        ] {
            add_test_tensor(&mut source, &format!("model.layers.{layer}.{suffix}"), dims);
        }
    }
    source
}

pub(crate) fn load_for_test(source: MapTensorSource) -> Result<YuE2Model, String> {
    YuE2Model::from_source(
        Arc::new(source),
        Arc::new(tokenizer_fixture()),
        Arc::new(ComputePool::new(1)),
    )
}

pub(crate) fn tiny_yue2_model() -> YuE2Model {
    YuE2Model::tiny_for_test(Arc::new(tokenizer_fixture()), Arc::new(ComputePool::new(1)))
}

fn noise_bits(chunks: &[YuE2Chunk]) -> Vec<u32> {
    chunks
        .iter()
        .flat_map(|chunk| chunk.noise.iter().copied().map(f32::to_bits))
        .collect()
}

fn tiny_nar_session() -> YuE2NarSession<'static> {
    let model = Box::leak(Box::new(tiny_yue2_model()));
    let chunk = YuE2Chunk {
        ar_tokens: vec![1, 2],
        noise: vec![0.0; model.config().latent_channels * 1],
        context_start: 0,
        context_end: 1,
    };
    YuE2NarSession::new(model, chunk).unwrap()
}

fn finite_state(frames: usize) -> Vec<f32> {
    vec![0.25; frames * 2]
}

#[test]
fn nar_draws_full_song_noise_once_then_slices_original_chunks() {
    let one = song_chunks(&[1, 2], &[3, 4, 5, 6], 831001, 12).unwrap();
    let again = song_chunks(&[1, 2], &[3, 4, 5, 6], 831001, 12).unwrap();
    assert_eq!(noise_bits(&one), noise_bits(&again));
    assert_eq!(
        one.iter()
            .map(|chunk| chunk.noise.len() / 64)
            .sum::<usize>(),
        4
    );
    assert!(one
        .iter()
        .all(|chunk| chunk.ar_tokens.last() == Some(&MUSIC_END)));
}

#[test]
fn nar_noise_matches_torch_2_10_cpu_bits_across_tail_boundaries() {
    let cases = [
        (0, 64, 0xcd80_56be_6268_7bb5, 0xbf90_1b85, 0xbf0d_30fc),
        (0, 65, 0xe48b_aed7_7a2e_75f0, 0xbf90_1b85, 0x3ea7_bd10),
        (0, 64 * 513, 0xa922_b89a_6912_adb9, 0xbf90_1b85, 0xbf4b_7856),
        (831001, 64, 0x2609_2c6d_7c1a_6227, 0x3f06_35ac, 0x3f30_9e22),
        (831001, 65, 0xc379_60d6_9182_9020, 0x3f06_35ac, 0x3fa8_3a6f),
        (
            831001,
            64 * 513,
            0x5a26_5990_41d3_e90a,
            0x3f06_35ac,
            0x3fa9_d434,
        ),
        (
            u32::MAX as u64,
            64,
            0x7c0f_d70e_fc81_3c82,
            0xc02a_83da,
            0xbdf3_764a,
        ),
        (
            u32::MAX as u64,
            65,
            0x2ffe_5559_fa6d_2bce,
            0xc02a_83da,
            0xbf4a_ca2b,
        ),
        (
            u32::MAX as u64,
            64 * 513,
            0xf378_4f8d_ab32_f4c5,
            0xc02a_83da,
            0xbf5b_90b7,
        ),
    ];
    for (seed, count, expected_hash, expected_first, expected_last) in cases {
        let values = torch_randn(seed, count);
        let hash = values
            .iter()
            .fold(0xcbf2_9ce4_8422_2325u64, |mut hash, value| {
                for byte in value.to_bits().to_le_bytes() {
                    hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
                }
                hash
            });
        assert_eq!(
            values.first().unwrap().to_bits(),
            expected_first,
            "seed={seed} count={count}"
        );
        assert_eq!(
            values.last().unwrap().to_bits(),
            expected_last,
            "seed={seed} count={count}"
        );
        assert_eq!(hash, expected_hash, "seed={seed} count={count}");
    }
}

#[test]
fn nar_visibility_and_midpoint_order_are_fixed() {
    let mask = visibility_mask(3, 4);
    assert!(mask.visible(0, 0));
    assert!(!mask.visible(0, 3));
    assert!(mask.visible(3, 0));
    assert!(mask.visible(3, 6));
    assert_eq!(midpoint_times(2).unwrap(), vec![(1.0, 0.75), (0.5, 0.25)]);
}

#[test]
fn nar_rejects_zero_steps_context_overflow_and_non_finite_state() {
    let session = tiny_nar_session();
    assert!(session.solve(0).unwrap_err().contains("steps"));
    assert!(song_chunks(&vec![1; CONTEXT], &[0], 1, CONTEXT).is_err());
    assert!(session
        .velocity(&[f32::NAN; 2], 0.0)
        .unwrap_err()
        .contains("non-finite"));
    assert!(session.velocity(&finite_state(1), 0.0).is_ok());
}

#[test]
fn nar_velocity_uses_timestep_and_midpoint_updates_state() {
    let session = tiny_nar_session();
    let state = finite_state(1);
    let early = session.velocity(&state, 20.0).unwrap();
    let late = session.velocity(&state, 0.0).unwrap();
    assert_eq!(early.len(), state.len());
    assert!(early.iter().all(|value| value.is_finite()));
    assert!(early.iter().any(|&value| value != 0.0));
    assert_ne!(
        noise_bits_from_values(&early),
        noise_bits_from_values(&late)
    );
    let solved = session.solve(2).unwrap();
    assert_eq!(solved.len(), state.len());
    assert_ne!(
        noise_bits_from_values(&solved),
        noise_bits_from_values(&[0.0; 2])
    );
}

fn noise_bits_from_values(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn main_config_requires_exact_release_contract() {
    let config = YuE2Config::from_source(&synthetic_yue2_source()).unwrap();
    assert_eq!(
        (
            config.hidden,
            config.layers,
            config.q_heads,
            config.kv_heads
        ),
        (2048, 28, 16, 8)
    );
    assert_eq!(
        (config.head_dim, config.ffn, config.vocab, config.context),
        (128, 6144, 184704, 24576)
    );
}

#[test]
fn main_loader_rejects_missing_nar_weight_and_wrong_bf16_shape() {
    let mut missing = synthetic_yue2_source();
    missing.remove_tensor("model.layers.0.nar_self_attn.q_proj.weight");
    assert!(load_for_test(missing)
        .unwrap_err()
        .contains("model.layers.0.nar_self_attn.q_proj.weight"));
    let mut wrong = synthetic_yue2_source();
    wrong.replace_tensor_shape("model.layers.0.self_attn.q_proj.weight", &[2047, 2048]);
    assert!(load_for_test(wrong).unwrap_err().contains("[2048, 2048]"));
}

#[test]
fn ar_prefill_and_one_token_decode_reuse_kv() {
    let model = tiny_yue2_model();
    let mut session = model.new_ar_session(8).unwrap();
    let prefill = session.prefill(&[1, 2, 3]).unwrap().to_vec();
    assert_eq!(session.position(), 3);
    let decode = session.decode(4).unwrap().to_vec();
    assert_eq!(session.position(), 4);
    assert_ne!(prefill, decode);
}

#[test]
fn ar_attention_matches_torch_cpu_flash_across_512_key_blocks() {
    fn draw(state: &mut u32, divisor: f32) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        half::bf16::from_f32((((*state >> 8) & 0xffff) as i32 - 32768) as f32 / divisor).to_f32()
    }
    let mut state = 831001u32;
    let query = (0..128)
        .map(|_| draw(&mut state, 16384.0))
        .collect::<Vec<_>>();
    let mut keys = (0..1025 * 128)
        .map(|_| draw(&mut state, 16384.0))
        .collect::<Vec<_>>();
    let values = (0..1025 * 128)
        .map(|_| draw(&mut state, 32768.0))
        .collect::<Vec<_>>();
    // Key 512 is a new score maximum, so CPU Flash must merge a second KV block.
    keys.copy_within(332 * 128..333 * 128, 512 * 128);
    keys[512 * 128] = half::bf16::from_f32(keys[512 * 128] - 1.0).to_f32();
    for (len, expected_hash) in [
        (512, 0x6939_072a_c615_c3a4u64),
        (513, 0xf6b0_46d0_5d48_1f91),
        (1025, 0x01c0_f5e5_ddb4_572b),
    ] {
        let mut scores = vec![0.0; len];
        let mut output = vec![0.0; 128];
        attention_head(
            &query,
            &keys[..len * 128],
            &values[..len * 128],
            128,
            0,
            &mut scores,
            &mut output,
            (128.0f32).sqrt().recip(),
        );
        let hash = output
            .iter()
            .fold(0xcbf2_9ce4_8422_2325u64, |mut hash, value| {
                for byte in value.to_bits().to_le_bytes() {
                    hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
                }
                hash
            });
        assert_eq!(hash, expected_hash, "KV length {len}");
    }
}

#[test]
#[ignore = "requires pinned full YuE2 Oracle trace"]
fn ar_attention_matches_real_flash_tail_at_530_keys() {
    let trace = std::path::PathBuf::from(std::env::var("YUE2_E2E_ORACLE_TRACE").unwrap());
    let root = trace.parent().unwrap();
    let read = |name: &str, occurrence: usize| {
        std::fs::read(root.join(format!("{name}.{occurrence}.f32")))
            .unwrap()
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let occurrence = 529 * 28 + 25;
    let query = read("yue2.ar.rope_q", occurrence);
    let expected = read("yue2.ar.attn", occurrence);
    let mut keys = Vec::with_capacity(530 * 128);
    let mut values = Vec::with_capacity(530 * 128);
    for position in 0..530 {
        let key = read("yue2.ar.rope_k", position * 28 + 25);
        let value = read("yue2.ar.v", position * 28 + 25);
        keys.extend_from_slice(&key[..128]);
        values.extend_from_slice(&value[..128]);
    }
    let mut scores = vec![0.0; 530];
    let mut actual = vec![0.0; 128];
    attention_head(
        &query[128..256],
        &keys,
        &values,
        128,
        0,
        &mut scores,
        &mut actual,
        (128.0f32).sqrt().recip(),
    );
    for (dimension, (&actual, &expected)) in actual.iter().zip(&expected[128..256]).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "head 1 dimension {dimension}"
        );
    }
}

#[test]
fn ar_sampling_masks_phase_vocab_and_minimum_end() {
    let mut sampling = SamplingConfig::abc();
    sampling.temperature = 0.0;
    sampling.min_tokens = 1;
    sampling.max_tokens = 2;
    let mut rng = Mt19937::new(7);

    let mut abc = vec![0.0; VOCAB_SIZE];
    abc[CODEC_OFFSET as usize] = 20.0;
    abc[ABC_END as usize] = 19.0;
    abc[7] = 18.0;
    assert_eq!(
        sample_phase_token(&abc, &[], sampling, 0, Phase::Abc, &mut rng).unwrap(),
        7
    );
    assert_eq!(
        sample_phase_token(&abc, &[], sampling, 1, Phase::Abc, &mut rng).unwrap(),
        ABC_END
    );

    sampling = SamplingConfig::semantic();
    sampling.temperature = 0.0;
    sampling.min_tokens = 1;
    sampling.max_tokens = 2;
    let mut semantic = vec![0.0; VOCAB_SIZE];
    semantic[7] = 20.0;
    semantic[MUSIC_END as usize] = 19.0;
    semantic[CODEC_OFFSET as usize] = 18.0;
    assert_eq!(
        sample_phase_token(&semantic, &[], sampling, 0, Phase::Semantic, &mut rng).unwrap(),
        CODEC_OFFSET
    );
    assert_eq!(
        sample_phase_token(&semantic, &[], sampling, 1, Phase::Semantic, &mut rng).unwrap(),
        MUSIC_END
    );
}

#[test]
fn ar_sampling_matches_torch_cpu_multinomial() {
    let sampling = SamplingConfig {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 3,
        repetition_penalty: 1.0,
        penalty_window: 100,
        min_tokens: 0,
        max_tokens: 10,
    };
    let mut logits = vec![-100.0; VOCAB_SIZE];
    logits[16] = 0.0;
    logits[17] = 1.0;
    logits[18] = 0.5;
    let mut rng = Mt19937::new(831001);
    let mut actual = Vec::new();
    for step in 0..10 {
        let token =
            sample_phase_token(&logits, &actual, sampling, step, Phase::Abc, &mut rng).unwrap();
        actual.push(token);
    }
    assert_eq!(actual, [17, 17, 16, 17, 18, 18, 17, 17, 16, 16]);
}

#[test]
fn bf16_matmul_rows_keep_bits_across_partitions() {
    let (n_in, n_out) = (259, 17);
    let input = (0..n_in)
        .map(|index| ((index % 29) as f32 - 14.0) * 0.03125)
        .collect::<Vec<_>>();
    let weights = (0..n_in * n_out)
        .flat_map(|index| {
            crate::ops::f32_to_bf16(((index % 37) as f32 - 18.0) * 0.015625).to_le_bytes()
        })
        .collect::<Vec<_>>();
    let bias = (0..n_out)
        .map(|row| (row as f32 - 8.0) * 0.0078125)
        .collect::<Vec<_>>();

    let mut single = vec![0.0f32; n_out];
    torch_bf16_matmul_rows(&weights, &input, Some(&bias), &mut single, n_in, 0);

    let mut partitioned = vec![f32::NAN; n_out];
    for thread in 0..4 {
        let (start, end) = crate::ops::kernel::bf16::BF16Kernel::row_range(n_out, thread, 4);
        torch_bf16_matmul_rows(
            &weights,
            &input,
            Some(&bias),
            &mut partitioned[start..end],
            n_in,
            start,
        );
    }

    assert_eq!(
        single
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        partitioned
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "requires the fixed YuE2 end-to-end Oracle trace"]
fn ar_sampling_matches_e2e_oracle_tokens() {
    use std::io::{BufRead, BufReader};
    use std::path::{Path, PathBuf};

    fn records(path: &Path) -> impl Iterator<Item = serde_json::Value> {
        BufReader::new(std::fs::File::open(path).unwrap())
            .lines()
            .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
    }

    fn replay(
        trace: &Path,
        generated_name: &str,
        first_logit_step: u64,
        sampling: SamplingConfig,
        phase: Phase,
    ) {
        let expected = records(trace)
            .find(|record| record["name"] == generated_name)
            .unwrap()["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|token| token.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let mut rng = Mt19937::new(831001);
        let mut history = Vec::with_capacity(expected.len());
        for record in records(trace) {
            if record["name"] == generated_name {
                break;
            }
            if record["name"] != "yue2.ar.logits"
                || record["step"].as_u64().unwrap() < first_logit_step
            {
                continue;
            }
            let sidecar = trace
                .parent()
                .unwrap()
                .join(record["path"].as_str().unwrap());
            let bytes = std::fs::read(sidecar).unwrap();
            let logits = bytes
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>();
            let step = history.len();
            let token =
                sample_phase_token(&logits, &history, sampling, step, phase, &mut rng).unwrap();
            assert_eq!(token, expected[step], "{generated_name} step {step}");
            history.push(token);
            if history.len() == expected.len() {
                break;
            }
        }
        assert_eq!(history.len(), expected.len());
    }

    let trace = PathBuf::from(std::env::var_os("YUE2_E2E_ORACLE_TRACE").unwrap());
    replay(
        &trace,
        "yue2.abc.generated_ids",
        46,
        SamplingConfig::abc(),
        Phase::Abc,
    );
    replay(
        &trace,
        "yue2.semantic.generated_ids",
        1630,
        SamplingConfig::semantic(),
        Phase::Semantic,
    );
}
