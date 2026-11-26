use super::*;
use crate::core::tensor::TensorInfo;

#[test]
fn folded_metadata_is_required_for_packed_embeddings_and_rejects_tied_v1() {
    struct Header(HashMap<&'static str, MetaValue>, TensorInfo);
    impl TensorSource for Header {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            self.0.get(key)
        }
        fn tensor_info(&self, key: &str) -> Option<&TensorInfo> {
            (key == "token_embd.weight").then_some(&self.1)
        }
        fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
            panic!("layout validation must precede tensor reads")
        }
    }
    let mut source = Header(
        HashMap::new(),
        TensorInfo {
            name: "token_embd.weight".into(),
            dims: vec![1024, 1],
            ggml_type: GGMLType::BF16,
            offset: 0,
        },
    );
    assert!(!has_folded_layout(&source).unwrap());
    source.1.ggml_type = GGMLType::PTQ1_0;
    assert!(has_folded_layout(&source)
        .unwrap_err()
        .contains("requires prism.hadamard.version"));
    source.0.extend([
        ("general.architecture", MetaValue::String("qwen35".into())),
        ("prism.hadamard.version", MetaValue::Uint32(1)),
        ("prism.hadamard.block_size", MetaValue::Uint32(1024)),
        (
            "prism.hadamard.transform",
            MetaValue::String("normalized-sylvester-walsh-hadamard".into()),
        ),
        (
            "prism.hadamard.axis",
            MetaValue::String("input-last-dimension".into()),
        ),
        (
            "prism.hadamard.sign_mode",
            MetaValue::String("explicit".into()),
        ),
    ]);
    assert!(has_folded_layout(&source).unwrap());
    source
        .0
        .insert("prism.hadamard.tied_output", MetaValue::Bool(true));
    assert!(has_folded_layout(&source)
        .unwrap_err()
        .contains("untied output.weight"));
}

#[test]
fn signed_gdn_map_transposes_heads_and_preserves_channels() {
    let signs = [1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1];
    let map = signed_map(&signs, Some((2, 6))).unwrap();
    assert_eq!(&*map, &[1, -2, -5, 6, 9, 10, -3, -4, 7, -8, 11, 12]);
    assert!(signed_map(&signs, Some((4, 6))).is_err());
    assert!(signed_map(&signs, Some((2, 8))).is_err());
}

#[test]
fn inverse_embedding_and_forward_projection_use_the_same_signed_basis() {
    // A single coefficient picks one known column of the Walsh matrix.
    let coefficient: usize = 173;
    let signs: Vec<_> = (0..BLOCK)
        .map(|i| if i % 3 == 0 { -1 } else { 1 })
        .collect();
    for ty in [GGMLType::Q2_0, GGMLType::PQ2_0] {
        let (elements, bytes) = ty.type_traits();
        let row_bytes = BLOCK / elements * bytes;
        let mut rows = vec![0x55; 2 * row_bytes];
        for block in rows.chunks_exact_mut(bytes) {
            block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        }
        let byte = row_bytes + coefficient / elements * bytes + 2 + coefficient % elements / 4;
        rows[byte] ^= 3 << (2 * (coefficient % 4));
        let kernel = FoldedMatrix {
            packed: PackedRows::new(ty, &rows).unwrap(),
            cpu_q4: OnceLock::new(),
            map: signed_map(&signs, None).unwrap(),
        };
        let mut embedding = vec![0.0; BLOCK];
        kernel.embedding_lookup(1, BLOCK, &mut embedding);
        #[cfg(feature = "vulkan")]
        assert!(matches!(
            kernel.device_weight_bytes(),
            Some(std::borrow::Cow::Owned(_))
        ));
        assert!(kernel.cpu_q4.get().is_none());
        for i in [0usize, 17, 53, 1023] {
            let parity = if (i & coefficient).count_ones() % 2 == 0 {
                1.0
            } else {
                -1.0
            };
            let expected = parity * signs[i] as f32 / 32.0;
            let mut input = vec![0.0; BLOCK];
            input[i] = 1.0;
            let mut output = [0.0; 2];
            kernel.forward(&input, &mut output, BLOCK, 2);
            assert!((embedding[i] - expected).abs() < 1e-6);
            // One Q8 code at this scale spans ~0.00025 after rotation.
            assert_eq!(output[0], 0.0);
            assert!((output[1] - expected).abs() < 0.0003);
        }
        #[cfg(feature = "vulkan")]
        assert!(matches!(
            kernel.device_weight_bytes(),
            Some(std::borrow::Cow::Borrowed(_))
        ));
    }
}

#[test]
fn packed_codes_and_scale_survive_repacking() {
    for (ty, size) in [
        (GGMLType::Q2_0, 64),
        (GGMLType::PQ2_0, 128),
        (GGMLType::PTQ1_0, 128),
    ] {
        let values: Vec<u8> = (0..size)
            .map(|i| ((i * 17 + i / 7) % if ty == GGMLType::PTQ1_0 { 3 } else { 4 }) as u8)
            .collect();
        let scale = [0x37, 0x38];
        let mut input = vec![];
        if ty == GGMLType::PTQ1_0 {
            let mut cursor = 0;
            for (stride, digits) in [(16, 5), (8, 5), (2, 4)] {
                for lane in 0..stride {
                    let code = (0..digits).fold(0u16, |code, digit| {
                        code * 3 + u16::from(values[cursor + lane + digit * stride])
                    });
                    let code = if digits == 4 { code * 3 } else { code };
                    input.push((code * 256).div_ceil(243) as u8);
                }
                cursor += stride * digits;
            }
            input.extend(scale);
        } else {
            input.extend(scale);
            input.extend(
                values
                    .chunks_exact(4)
                    .map(|q| q[0] | q[1] << 2 | q[2] << 4 | q[3] << 6),
            );
        }
        let packed = PackedRows::new(ty, &input).unwrap().repack();
        for (block, values) in packed.chunks_exact(18).zip(values.chunks_exact(32)) {
            assert_eq!(&block[..2], &scale);
            for i in 0..16 {
                assert_eq!(block[2 + i] & 15, values[i] + 7);
                assert_eq!(block[2 + i] >> 4, values[i + 16] + 7);
            }
        }
    }
}

#[test]
fn walsh_transform_matches_dense_matrix() {
    let input: Vec<f32> = (0..1024)
        .map(|i| ((37 * i % 119) as f32 - 51.0) / 128.0)
        .collect();
    let mut actual = input.clone();
    hadamard(&mut actual);
    for (i, &result) in actual.iter().enumerate() {
        let expected = input
            .iter()
            .enumerate()
            .map(|(j, x)| {
                if (i & j).count_ones() % 2 == 0 {
                    *x as f64
                } else {
                    -(*x as f64)
                }
            })
            .sum::<f64>()
            / 32.0;
        assert!((result as f64 - expected).abs() < 1e-5);
    }
}
