mod generation;
mod multimodal;
mod tts_bridge;
mod vision;

pub use generation::*;
pub use multimodal::*;
pub use tts_bridge::*;
pub use vision::*;

#[cfg(test)]
mod tests {
    use super::generation::uses_llama_trunk;
    use super::multimodal::run_multimodal;
    use super::vision::{
        build_qwen3_media_positions, inject_qwen_media_embeddings, inject_vision_embeddings,
        validate_single_qwen_media,
    };
    use crate::app::cli::CliOptions;
    use crate::core::tensor::{MetaValue, TensorInfo, TensorSource};
    use crate::models::qwen35::{Qwen35Config, Qwen35Model};
    use crate::ops::kernel::{QuantizedTensor, Weight};
    use std::path::Path;

    fn qwen35_embedding_model() -> Qwen35Model<'static> {
        let mut tok_embd = Weight::from_quantized(QuantizedTensor::F32 {
            data: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            n_in: 0,
            n_out: 0,
        });
        tok_embd.n_in = 2;
        tok_embd.n_out = 3;
        let mut output_weight = Weight::from_quantized(QuantizedTensor::F32 {
            data: Vec::new(),
            n_in: 0,
            n_out: 0,
        });
        output_weight.n_in = 2;
        output_weight.n_out = 3;
        Qwen35Model {
            config: Qwen35Config {
                n_nextn: 0,
                n_embd: 2,
                n_layer: 0,
                n_head: 1,
                n_head_kv: 1,
                n_ff: 2,
                n_ctx: 4,
                vocab_size: 3,
                rope_freq_base: 1.0,
                norm_eps: 0.0,
                rope_dimension_count: 2,
                rope_dimension_sections: [0; 4],
                ssm_d_conv: 1,
                ssm_d_state: 1,
                ssm_n_group: 1,
                ssm_dt_rank: 1,
                ssm_d_inner: 1,
                full_attention_interval: 1,
                is_recurrent: Vec::new(),
                key_length: 2,
                value_length: 2,
            },
            tok_embd,
            output_norm: vec![1.0; 2],
            output_weight,
            layers: Vec::new(),
            #[cfg(feature = "vulkan")]
            gpu: None,
        }
    }

    struct ArchSource(MetaValue);

    impl TensorSource for ArchSource {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            (key == "general.architecture").then_some(&self.0)
        }

        fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
            None
        }

        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            None
        }
    }

    #[test]
    fn k2_horizon_uses_llama_trunk() {
        assert!(uses_llama_trunk("k2-horizon"));
    }

    #[test]
    fn gemma4_cli_rejects_nonzero_temperature() {
        let source = ArchSource(MetaValue::String("gemma4".into()));
        let error = run_multimodal(
            &source,
            Path::new("missing.gguf"),
            None,
            None,
            None,
            "hello",
            1,
            0.1,
            1,
            crate::core::prefill::DEFAULT_PREFILL_BATCH_SIZE,
            CliOptions::DEFAULT_MAX_CONTEXT,
            1.0,
        )
        .unwrap_err();
        assert!(error.contains("--temp"), "{error}");
    }

    #[test]
    fn qwen_media_positions_expand_grid_rows_and_audio_rows() {
        assert_eq!(
            build_qwen3_media_positions(&[10, 99, 99, 99, 99, 11], 99, &[(2, 2)]).unwrap(),
            vec![
                [0, 0, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 2, 0],
                [1, 2, 1, 0],
                [1, 2, 2, 0],
                [3, 3, 3, 0]
            ]
        );
        assert_eq!(
            build_qwen3_media_positions(&[10, 99, 99, 11], 99, &[]).unwrap(),
            vec![[0, 0, 0, 0], [1, 1, 1, 0], [2, 2, 2, 0], [3, 3, 3, 0]]
        );
        assert!(build_qwen3_media_positions(&[99, 99], 99, &[(1, 1)]).is_err());
    }

    #[test]
    fn qwen_multimodal_generation_rejects_combined_media() {
        assert!(validate_single_qwen_media(true, false, false).is_ok());
        assert!(validate_single_qwen_media(false, true, true).is_err());
    }

    #[test]
    fn qwen_media_injection_preserves_deepstack_layer_and_prompt_order() {
        let tokens = [1, 99, 2, 99];
        let mut embeddings = vec![0.0; tokens.len() * 2];
        let media = [10.0, 11.0, 20.0, 21.0];
        let media_deepstack = [
            100.0, 101.0, 200.0, 201.0, // layer 0 media rows
            300.0, 301.0, 400.0, 401.0, // layer 1 media rows
        ];

        let deepstack =
            inject_qwen_media_embeddings(&tokens, 99, &mut embeddings, &media, &media_deepstack, 2)
                .unwrap();

        assert_eq!(embeddings, [0.0, 0.0, 10.0, 11.0, 0.0, 0.0, 20.0, 21.0]);
        assert_eq!(
            deepstack,
            [
                0.0, 0.0, 100.0, 101.0, 0.0, 0.0, 200.0, 201.0, // layer 0
                0.0, 0.0, 300.0, 301.0, 0.0, 0.0, 400.0, 401.0, // layer 1
            ]
        );
    }

    #[test]
    fn inject_vision_embeddings_preserves_text_and_image_row_order() {
        let model = qwen35_embedding_model();

        assert_eq!(
            inject_vision_embeddings(&model, &[0, 99, 1], Some(99), &[9.0, 8.0], 1, 2).unwrap(),
            [0.0, 1.0, 9.0, 8.0, 2.0, 3.0]
        );
    }

    #[test]
    fn inject_vision_embeddings_rejects_invalid_rows() {
        let model = qwen35_embedding_model();

        assert!(inject_vision_embeddings(&model, &[-1], None, &[], 0, 2).is_err());
        assert!(inject_vision_embeddings(&model, &[99], Some(99), &[1.0], 1, 1).is_err());
    }
}
