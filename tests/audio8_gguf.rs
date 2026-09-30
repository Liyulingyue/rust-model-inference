use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::core::tokenizer::{BPETokenizer, EncodeOptions};
use rust_model_inference::models::audio8::text::Audio8TextDecoder;
use rust_model_inference::models::audio8::Audio8Encoder;
use std::sync::Arc;

#[test]
fn real_audio8_gguf_loads_and_matches_official_token_ids() {
    let Some(path) = std::env::var_os("RMI_AUDIO8_GGUF") else {
        return;
    };
    let source: Arc<dyn TensorSource> = Arc::new(GGUFLoader::from_file(path).unwrap());
    assert_eq!(
        source.model_config().unwrap_err(),
        "Unsupported architecture: audio8_asr_infinite"
    );
    let tokenizer =
        Arc::new(BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).unwrap());
    for (text, expected) in [
        ("hello world", vec![14990, 1879]),
        ("你好，世界！", vec![108386, 3837, 99489, 6313]),
        ("[LANGUAGE_ZH][STREAMING_PAD]", vec![151667, 151665]),
    ] {
        assert_eq!(
            tokenizer.encode(
                text,
                EncodeOptions {
                    parse_special: true,
                    ..Default::default()
                }
            ),
            expected
        );
    }
    let encoder = Audio8Encoder::from_source(Arc::clone(&source)).unwrap();
    assert_eq!(encoder.frame_embedding.len(), 3 * 2048);
    let decoder = Audio8TextDecoder::from_source(source).unwrap();
    assert_eq!(decoder.context(), 32_768);
}
