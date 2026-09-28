//! Bit-exact parity against the pinned stable-diffusion.cpp oracle for the
//! Qwen-Image-2.1 diffusion transformer.
//!
//! Produce the two traces first (see tools/oracle/qwen_image_2_1/parity.sh,
//! which builds the oracle via build_oracle.sh and runs both sides once on the
//! deterministic synthetic inputs: 16x16 latent, 128-token context, timestep
//! 500), then run:
//!   QWEN_IMAGE_2_1_ORACLE_TRACE=<oracle.jsonl> \
//!   QWEN_IMAGE_2_1_RUST_TRACE=<rust.jsonl> \
//!   cargo test --profile release-fast --test qwen_image_2_1_reference
//!
//! Every traced checkpoint — inputs, pe, timestep embed, modulation, txt_in,
//! joint, all 32 blocks, final layer, velocity — must match the oracle bit for bit.

use serde_json::Value;
use std::path::{Path, PathBuf};

const CHECKPOINTS: &[&str] = &[
    "qwen.pe",
    "qwen.time_embed",
    "qwen.modulation",
    "qwen.txt_in",
    "qwen.joint",
    "qwen.joint_final",
    "qwen.scale",
    "qwen.norm_out",
    "qwen.out",
    "qwen.input.x",
    "qwen.input.context",
    "qwen.input.timesteps",
    "qwen.output",
];

fn validated_file_env(name: &str) -> PathBuf {
    let raw = std::env::var(name).unwrap_or_else(|_| panic!("missing {name}"));
    let path = PathBuf::from(&raw);
    let canonical = path
        .canonicalize()
        .unwrap_or_else(|error| panic!("{name} path {} is not usable: {error}", path.display()));
    assert!(
        canonical.is_file(),
        "{} must be a file",
        canonical.display()
    );
    canonical
}

fn records(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn named<'a>(records: &'a [Value], name: &str) -> &'a Value {
    records
        .iter()
        .find(|record| record["name"] == name)
        .unwrap_or_else(|| panic!("missing parity checkpoint {name}"))
}

fn validated_values(record: &Value) -> Vec<f32> {
    let len = record["len"].as_u64().expect("checkpoint len") as usize;
    let shape_len = record["shape"]
        .as_array()
        .expect("checkpoint shape")
        .iter()
        .map(|value| value.as_u64().unwrap() as usize)
        .product::<usize>();
    assert_eq!(len, shape_len, "shape product must match len");
    let sidecar = Path::new(record["binary_path"].as_str().expect("binary_path"));
    assert!(
        sidecar.is_file(),
        "sidecar {} must be a file",
        sidecar.display()
    );
    let bytes = std::fs::read(sidecar).unwrap();
    assert_eq!(bytes.len() % 4, 0, "{}", sidecar.display());
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    assert_eq!(values.len(), len, "sidecar length");
    values
}

#[test]
#[ignore = "requires QWEN_IMAGE_2_1_ORACLE_TRACE and QWEN_IMAGE_2_1_RUST_TRACE (see tools/oracle/qwen_image_2_1/parity.sh)"]
fn qwen_image_2_1_matches_pinned_oracle_bit_for_bit() {
    let oracle_trace = validated_file_env("QWEN_IMAGE_2_1_ORACLE_TRACE");
    let rust_trace = validated_file_env("QWEN_IMAGE_2_1_RUST_TRACE");
    let oracle_records = records(&oracle_trace);
    let rust_records = records(&rust_trace);

    let mut names: Vec<String> = CHECKPOINTS.iter().map(|name| name.to_string()).collect();
    names.extend((0..32).map(|layer| format!("qwen.block.{layer}")));
    let selected = |records: &[Value]| -> Vec<String> {
        records
            .iter()
            .filter_map(|record| record["name"].as_str())
            .filter(|name| names.iter().any(|selected| selected == name))
            .map(str::to_owned)
            .collect()
    };
    let oracle_order = selected(&oracle_records);
    assert_eq!(oracle_order.len(), names.len(), "oracle checkpoint count");
    assert_eq!(
        selected(&rust_records),
        oracle_order,
        "checkpoint order and count"
    );
    for name in names {
        assert_eq!(
            named(&oracle_records, &name)["shape"],
            named(&rust_records, &name)["shape"],
            "{name} shape"
        );
        let oracle = validated_values(named(&oracle_records, &name));
        let rust = validated_values(named(&rust_records, &name));
        assert_eq!(oracle.len(), rust.len(), "{name} element count");
        let diverged = oracle
            .iter()
            .zip(&rust)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            diverged,
            0,
            "{name}: {diverged} of {} values differ",
            oracle.len()
        );
    }
}
