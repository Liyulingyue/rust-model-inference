use super::*;

#[test]
fn merge_uses_row_major_pixels_within_each_patch() {
    let input: Vec<_> = (0..PIXELS).flat_map(|p| vec![p as f32; D]).collect();
    let packed = pack(&input);
    assert_eq!(
        [packed[0], packed[D], packed[2 * D], packed[3 * D]],
        [0., 1., 16., 17.]
    );
    assert_eq!([packed[4 * D], packed[5 * D]], [2., 3.]);
    assert_eq!(unpack(&packed), input);
}

#[test]
fn axial_rotation_uses_patch_centres_and_leaves_prefix_unrotated() {
    let rope = Rope::new(GRID / 2, 2);
    let mut hidden = vec![1.; (PREFIX + 64) * D];
    rope.normalize_and_rotate(&mut hidden, &[1.; HEAD_DIM]);
    assert!((hidden[0] - 1.).abs() < 1e-6);
    let first = &hidden[PREFIX * D..];
    let (sin, cos) = 0.5f32.sin_cos();
    assert!((first[0] - (cos - sin)).abs() < 1e-6);
    assert!((first[HEAD_DIM / 2] - (sin + cos)).abs() < 1e-6);
    assert!((first[HEAD_DIM / 4] - first[0]).abs() < 1e-6);
    let second = &hidden[(PREFIX + 1) * D..];
    let (sin, cos) = 2.5f32.sin_cos();
    assert!((second[HEAD_DIM / 4] - (cos - sin)).abs() < 1e-6);
}

#[test]
fn transparent_color_has_no_raw_components_and_keeps_cosine_features() {
    let feats = color_features(&[0.7, 0.2, -0.1, 0.5, 0.4, 0.0]);
    assert_eq!(feats.len(), 80);
    assert_eq!(feats[..3], [0.; 3]);
    assert_eq!(feats[3..9], [0.; 6]);
    assert_eq!(feats[9..15], [1.; 6]);
    assert_eq!((feats[39], feats[79]), (0., 1.));
    assert_eq!(feats[40..43], [0., 1., 0.]);
}

/// Compare both CFG branches before and after self-conditioning against the
/// independent PyTorch forward loaded with this GGUF's dequantized weights.
#[test]
#[ignore = "requires TESSERA_GGUF and its official TESSERA_FIXTURE_DIR"]
fn tessera_denoiser_oracle() {
    use crate::format::ggufrs::{open_model_source, ComponentRole};
    use std::{env, fs, path::PathBuf};
    let gguf = PathBuf::from(env::var_os("TESSERA_GGUF").expect("set TESSERA_GGUF"));
    let dir = PathBuf::from(env::var_os("TESSERA_FIXTURE_DIR").expect("set TESSERA_FIXTURE_DIR"));
    let read_f32 = |name: &str| -> Vec<f32> {
        let bytes = fs::read(dir.join(name)).unwrap();
        assert_eq!(bytes.len() % 4, 0);
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    let source = open_model_source(&gguf, ComponentRole::Llm).unwrap();
    // The real projection storage decides the budget, independent of the fixture directory name.
    let ty = source
        .tensor_info("blocks_in.0.attn.qkv.weight")
        .unwrap()
        .ggml_type;
    let quantized = !matches!(
        ty,
        crate::GGMLType::F32 | crate::GGMLType::F16 | crate::GGMLType::BF16
    );
    let (rmse_limit, cosine_min) = if quantized {
        (0.15, 0.995)
    } else {
        (0.005, 0.9999)
    };
    let policy = match env::var("TESSERA_COMPUTE").as_deref().unwrap_or("cpu") {
        "cpu" => ComputePolicy::Cpu,
        "vulkan" => ComputePolicy::Vulkan,
        "auto" => ComputePolicy::Auto,
        s => panic!("unknown TESSERA_COMPUTE: {s}"),
    };
    let weights = Weights::load(source.as_ref()).unwrap();
    let mut net = Denoiser::new(&weights, Arc::new(ComputePool::new(8)), policy).unwrap();
    let states = read_f32("states.f32");
    let pooled = read_f32("pooled.f32");
    let mut ids = vec![VOCAB as u16; PIXELS];
    let mut sc = None;
    for suffix in ["", "_2"] {
        if suffix == "_2" {
            let bytes = fs::read(dir.join("greedy-step-1.u16")).unwrap();
            assert_eq!(bytes.len(), 2 * PIXELS);
            ids = bytes
                .chunks_exact(2)
                .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
                .collect();
            sc = Some(read_f32("sc-step-2.f32"));
        }
        for (branch, dropped) in [("cond", false), ("uncond", true)] {
            let expected = read_f32(&format!("logits_{branch}{suffix}.f32"));
            let actual = net
                .forward(&ids, &states, &pooled, sc.as_deref(), dropped)
                .unwrap();
            assert_eq!(expected.len(), PIXELS * VOCAB);
            assert_eq!(actual.len(), expected.len());
            let (mut err, mut dot, mut aa, mut bb, mut max) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for (&a, &b) in actual.iter().zip(&expected) {
                let (a, b) = (f64::from(a), f64::from(b));
                err += (a - b).powi(2);
                dot += a * b;
                aa += a * a;
                bb += b * b;
                max = max.max((a - b).abs());
            }
            let rmse = (err / actual.len() as f64).sqrt();
            let cosine = dot / (aa * bb).sqrt();
            eprintln!("Tessera logits_{branch}{suffix} {ty:?} {policy:?}: max_abs={max:.6} rmse={rmse:.6} cosine={cosine:.8}");
            assert!(rmse <= rmse_limit && cosine >= cosine_min);
        }
    }
    let (cpu, gpu) = net.counts();
    eprintln!("Tessera denoiser projections: cpu={cpu} vulkan={gpu}");
    match policy {
        ComputePolicy::Cpu => assert!(cpu > 0 && gpu == 0),
        ComputePolicy::Vulkan => assert!(gpu > 0 && cpu == 0),
        ComputePolicy::Auto => assert!(cpu + gpu > 0),
    }
}
