use rust_model_inference::{BPETokenizer, EncodeOptions, GGUFLoader};

#[test]
#[ignore = "set EDGE0_GGUF to a converted Edge0 model or tokenizer-only GGUF"]
fn edge0_token_ids_match_hf_tokenizers_0_22_1() {
    let path = std::env::var("EDGE0_GGUF").expect("EDGE0_GGUF is required");
    let loader = GGUFLoader::from_file(&path).unwrap();
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| loader.metadata(key).cloned()).unwrap();
    let cases: &[(&str, &[u32], bool)] = &[
        ("Hello, world!", &[9419, 11, 1814, 0], false),
        ("你好，世界\n", &[109266, 3709, 96748, 198], false),
        (
            "<|im_start|>user\nHi<|im_end|>\n",
            &[248045, 846, 198, 12675, 248046, 198],
            true,
        ),
        ("\tfoo  bar", &[197, 7724, 220, 3498], false),
    ];
    for &(text, ids, parse_special) in cases {
        assert_eq!(
            tokenizer.encode(
                text,
                EncodeOptions {
                    add_special: false,
                    parse_special
                }
            ),
            ids
        );
    }
}
