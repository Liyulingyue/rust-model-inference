use super::*;

#[test]
#[ignore = "requires TESSERA_GGUF and its matching official TESSERA_FIXTURE_DIR"]
fn tessera_clip_oracle() {
    use crate::format::ggufrs::{open_model_source, ComponentRole};
    use std::{env, fs, path::PathBuf};
    let gguf = PathBuf::from(env::var_os("TESSERA_GGUF").expect("set TESSERA_GGUF"));
    let dir = PathBuf::from(env::var_os("TESSERA_FIXTURE_DIR").expect("set TESSERA_FIXTURE_DIR"));
    let source = open_model_source(&gguf, ComponentRole::Llm).unwrap();
    let encoder = TextEncoder::load(source.as_ref()).unwrap();
    let prompt = "golden crown with blue gems";
    let ids = fs::read(dir.join("ids.u32")).unwrap();
    assert_eq!(ids.len(), CONTEXT * 4);
    let ids: Vec<_> = ids
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    assert_eq!(encoder.tokenize(prompt).unwrap(), ids);

    // Independently encoded by Hugging Face CLIPTokenizer for the pinned tokenizer revision.
    for (prompt, mut expected) in [
        ("", vec![BOS, EOS]),
        ("  Café\t猫 😀 ", vec![BOS, 15304, 163, 234, 360, 7334, EOS]),
        ("a<|endoftext|>b", vec![BOS, 320, EOS, 321, EOS]),
    ] {
        expected.resize(CONTEXT, EOS);
        assert_eq!(encoder.tokenize(prompt).unwrap(), expected);
    }
    let mut expected = vec![343; CONTEXT];
    expected[0] = BOS;
    expected[CONTEXT - 1] = EOS;
    assert_eq!(encoder.tokenize(&"x ".repeat(50)).unwrap(), expected);

    let policy = match env::var("TESSERA_COMPUTE").as_deref().unwrap_or("cpu") {
        "cpu" => ComputePolicy::Cpu,
        "vulkan" => ComputePolicy::Vulkan,
        "auto" => ComputePolicy::Auto,
        s => panic!("unknown TESSERA_COMPUTE: {s}"),
    };
    let (states, pooled) = encoder
        .encode(prompt, policy, Arc::new(ComputePool::new(8)))
        .unwrap();
    let ty = source
        .tensor_info("text_model.encoder.layers.0.self_attn.q_proj.weight")
        .unwrap()
        .ggml_type;
    let (rrmse_limit, cosine_floor) = match ty {
        GGMLType::F32 => (5e-4, 0.999999),
        GGMLType::BF16 | GGMLType::F16 => (0.025, 0.9995),
        _ => (0.075, 0.997),
    };
    for (name, output) in [("states.f32", states), ("pooled.f32", pooled)] {
        let bytes = fs::read(dir.join(name)).unwrap();
        assert_eq!(bytes.len(), output.len() * 4);
        let (mut max, mut sq_err, mut dot, mut aa, mut bb) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (&a, b) in output.iter().zip(bytes.chunks_exact(4)) {
            let a = f64::from(a);
            let b = f64::from(f32::from_le_bytes(b.try_into().unwrap()));
            max = max.max((a - b).abs());
            sq_err += (a - b).powi(2);
            dot += a * b;
            aa += a * a;
            bb += b * b;
        }
        let relative_rmse = (sq_err / bb).sqrt();
        let cosine = dot / (aa * bb).sqrt();
        eprintln!("Tessera CLIP {name} {ty:?} {policy:?}: max_abs={max:.7} relative_rmse={relative_rmse:.7} cosine={cosine:.9}");
        assert!(
            relative_rmse <= rrmse_limit && cosine >= cosine_floor,
            "{name} {ty:?} {policy:?}"
        );
    }
}
