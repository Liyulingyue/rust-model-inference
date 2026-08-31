use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorSource};
#[cfg(not(test))]
const GEMMA4_TOKEN_TABLE_DIGEST: u128 = 0xda70_6e66_3141_68dc_84fd_09da_fee1_4466;
#[cfg(test)]
const GEMMA4_TOKEN_TABLE_DIGEST: u128 = 0x2055_26be_4c3c_b1e6_cf1a_4a8a_00b7_7831;
pub(super) fn require_clip(source: &dyn TensorSource) -> Result<(), String> {
    require_string(source, "general.architecture", "clip")?;
    require_string(source, "general.type", "mmproj")
}

pub(super) fn require_string(
    source: &dyn TensorSource,
    key: &str,
    expected: &str,
) -> Result<(), String> {
    match source.metadata(key) {
        Some(MetaValue::String(value)) if value == expected => Ok(()),
        Some(value) => Err(format!(
            "Invalid metadata {key}: expected string {expected:?}, got {value:?}"
        )),
        None => Err(format!("Missing metadata: {key}")),
    }
}

pub(super) fn require_bool(
    source: &dyn TensorSource,
    key: &str,
    expected: bool,
) -> Result<(), String> {
    match source.metadata(key) {
        Some(MetaValue::Bool(value)) if *value == expected => Ok(()),
        Some(value) => Err(format!(
            "Invalid metadata {key}: expected bool {expected}, got {value:?}"
        )),
        None => Err(format!("Missing metadata: {key}")),
    }
}

pub(super) fn require_u32(
    source: &dyn TensorSource,
    key: &str,
    expected: u32,
) -> Result<(), String> {
    match source.metadata(key) {
        Some(MetaValue::Uint32(value)) if *value == expected => Ok(()),
        Some(value) => Err(format!(
            "Invalid metadata {key}: expected uint32 {expected}, got {value:?}"
        )),
        None => Err(format!("Missing metadata: {key}")),
    }
}

pub(super) fn require_f32(
    source: &dyn TensorSource,
    key: &str,
    expected: f32,
) -> Result<(), String> {
    match source.metadata(key) {
        Some(MetaValue::Float32(value)) if value.to_bits() == expected.to_bits() => Ok(()),
        Some(value) => Err(format!(
            "Invalid metadata {key}: expected float32 {expected}, got {value:?}"
        )),
        None => Err(format!("Missing metadata: {key}")),
    }
}

pub(super) fn require_array(
    source: &dyn TensorSource,
    key: &str,
    expected_type: MetaValueType,
    expected: &[MetaValue],
) -> Result<(), String> {
    match source.metadata(key) {
        Some(MetaValue::Array(value_type, values))
            if *value_type == expected_type && values == expected =>
        {
            Ok(())
        }
        Some(value) => Err(format!(
            "Invalid metadata {key}: expected exact {expected_type:?} array, got {value:?}"
        )),
        None => Err(format!("Missing metadata: {key}")),
    }
}

pub(super) fn require_gemma4_token_table(source: &dyn TensorSource) -> Result<(), String> {
    match source.metadata("tokenizer.ggml.tokens") {
        Some(MetaValue::Array(MetaValueType::String, tokens)) if tokens.len() == 262_144 => {
            let actual = gemma4_token_table_digest(tokens)?;
            if actual != GEMMA4_TOKEN_TABLE_DIGEST {
                return Err(format!(
                    "Invalid metadata tokenizer.ggml.tokens digest: expected {GEMMA4_TOKEN_TABLE_DIGEST:#034x}, got {actual:#034x}"
                ));
            }
            Ok(())
        }
        Some(value) => Err(format!(
            "Invalid metadata tokenizer.ggml.tokens: expected String array length 262144, got {value:?}"
        )),
        None => Err("Missing metadata: tokenizer.ggml.tokens".into()),
    }
}

pub(super) fn gemma4_token_table_digest(tokens: &[MetaValue]) -> Result<u128, String> {
    const OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;

    let mut digest = OFFSET;
    let mut write = |bytes: &[u8]| {
        for &byte in bytes {
            digest = (digest ^ u128::from(byte)).wrapping_mul(PRIME);
        }
    };
    write(&u64::try_from(tokens.len()).unwrap().to_le_bytes());
    for (id, token) in tokens.iter().enumerate() {
        let MetaValue::String(token) = token else {
            return Err(format!(
                "Invalid metadata tokenizer.ggml.tokens[{id}]: expected string, got {token:?}"
            ));
        };
        write(&u64::try_from(id).unwrap().to_le_bytes());
        write(&u64::try_from(token.len()).unwrap().to_le_bytes());
        write(token.as_bytes());
    }
    Ok(digest)
}

pub(super) fn require_tensor(
    source: &dyn TensorSource,
    name: &str,
    expected_dims: &[u64],
    expected_type: GGMLType,
) -> Result<(), String> {
    let tensor = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if tensor.dims != expected_dims || tensor.ggml_type != expected_type {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected {:?} {:?}",
            tensor.dims, tensor.ggml_type, expected_dims, expected_type
        ));
    }
    Ok(())
}

pub(super) fn require_clippable(
    source: &dyn TensorSource,
    prefix: &str,
    dims: &[u64],
) -> Result<(), String> {
    for suffix in ["input_max", "input_min", "output_max", "output_min"] {
        require_tensor(source, &format!("{prefix}.{suffix}"), &[1], GGMLType::F32)?;
    }
    require_tensor(source, &format!("{prefix}.weight"), dims, GGMLType::F16)
}
