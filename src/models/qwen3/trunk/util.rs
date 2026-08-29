//! Helpers shared by [`Qwen3Config`](super::config::Qwen3Config),
//! [`Qwen3Model`](super::weights::Qwen3Model), and
//! [`Qwen3Session`](super::session::Qwen3Session).
//!
//! All functions here are pure: they do not mutate model state and have no
//! hidden dependencies.

use crate::core::tensor::{GGMLType, TensorInfo, TensorSource};
pub use crate::core::tensor::load_f32_tensor;

pub fn optional_usize(source: &dyn TensorSource, key: &str) -> Result<Option<usize>, String> {
    let Some(value) = source.metadata(key) else {
        return Ok(None);
    };
    let value = value
        .to_u64()
        .ok_or_else(|| format!("Invalid metadata: {key}"))?;
    usize::try_from(value)
        .map(Some)
        .map_err(|_| format!("{key} does not fit usize"))
}

pub fn checked_session_capacity(
    prompt: usize,
    generation: usize,
    context: usize,
) -> Result<usize, String> {
    let capacity = prompt
        .checked_add(generation)
        .ok_or_else(|| "Session capacity overflow".to_string())?;
    if capacity > context {
        return Err(format!(
            "Session capacity {capacity} exceeds model context {context}"
        ));
    }
    Ok(capacity)
}

pub fn checked_decoder_steps(
    prompt: usize,
    generation: usize,
    context: usize,
) -> Result<usize, String> {
    checked_session_capacity(prompt, generation, context)?
        .checked_sub(1)
        .ok_or_else(|| "Decoder step count underflow".to_string())
}

pub fn checked_generated_position(
    prompt_positions: &[[usize; 4]],
    generated_index: usize,
) -> Result<[usize; 4], String> {
    let last_prompt_position = prompt_positions
        .last()
        .ok_or_else(|| "Cannot generate a position without prompt positions".to_string())?[0];
    let position = last_prompt_position
        .checked_add(1)
        .and_then(|position| position.checked_add(generated_index))
        .ok_or_else(|| "Generated position overflow".to_string())?;
    Ok([position, position, position, 0])
}

pub fn validate_input_shapes(
    token_count: usize,
    embedding_dim: usize,
    position_count: usize,
    embedding_values: Option<usize>,
) -> Result<(), String> {
    if position_count != token_count {
        return Err(format!(
            "Position count {position_count} does not match token count {token_count}"
        ));
    }
    if let Some(values) = embedding_values {
        let expected = token_count
            .checked_mul(embedding_dim)
            .ok_or_else(|| "Input embedding shape overflow".to_string())?;
        if values != expected {
            return Err(format!(
                "Embedding value count {values} does not match expected {expected}"
            ));
        }
    }
    Ok(())
}

pub fn greedy_token(logits: &[f32]) -> Result<u32, String> {
    let (&first, rest) = logits
        .split_first()
        .ok_or_else(|| "Cannot sample empty logits".to_string())?;
    if !first.is_finite() {
        return Err("Cannot sample non-finite logits".into());
    }
    let mut best_id = 0usize;
    let mut best = first;
    for (index, &logit) in rest.iter().enumerate() {
        if !logit.is_finite() {
            return Err("Cannot sample non-finite logits".into());
        }
        if logit > best {
            best = logit;
            best_id = index + 1;
        }
    }
    u32::try_from(best_id).map_err(|_| "Token ID does not fit u32".into())
}

pub fn validate_generation(
    model: &super::weights::Qwen3Model,
    input: &super::forward::Qwen3Input<'_>,
    options: &super::forward::Qwen3GenerateOptions,
) -> Result<(), String> {
    if input.token_ids.is_empty() {
        return Err("Qwen3 prompt must contain at least one token".into());
    }
    if options.max_new_tokens == 0 {
        return Err("Qwen3 generation must request at least one token".into());
    }
    if !options.temperature.is_finite() || options.temperature < 0.0 {
        return Err(format!(
            "Invalid generation temperature: {}",
            options.temperature
        ));
    }
    validate_input_shapes(
        input.token_ids.len(),
        model.config.n_embd,
        input.positions.len(),
        input.embeddings.map(<[f32]>::len),
    )?;
    validate_token_ids(input.token_ids, model.config.vocab)?;
    if input
        .embeddings
        .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
    {
        return Err("Input embeddings contain NaN or infinity".into());
    }
    checked_session_capacity(
        input.token_ids.len(),
        options.max_new_tokens,
        model.config.n_ctx,
    )?;
    Ok(())
}

pub(crate) fn validate_token_ids(token_ids: &[u32], vocab: usize) -> Result<(), String> {
    for &token_id in token_ids {
        let token =
            usize::try_from(token_id).map_err(|_| format!("Invalid token ID {token_id}"))?;
        if token >= vocab {
            return Err(format!("Token ID {token_id} exceeds vocabulary {vocab}"));
        }
    }
    Ok(())
}

pub(crate) fn sample_token(logits: &[f32], temperature: f32) -> Result<u32, String> {
    if temperature == 0.0 {
        return greedy_token(logits);
    }
    let mut max_logit = f32::NEG_INFINITY;
    for &logit in logits {
        if !logit.is_finite() {
            return Err("Cannot sample non-finite logits".into());
        }
        max_logit = max_logit.max(logit);
    }
    if logits.is_empty() {
        return Err("Cannot sample empty logits".into());
    }
    let sum: f32 = logits
        .iter()
        .map(|logit| ((logit - max_logit) / temperature).exp())
        .sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err("Sampling probability sum is not finite and positive".into());
    }
    let target = rand::random::<f32>() * sum;
    let mut cumulative = 0.0;
    for (index, &logit) in logits.iter().enumerate() {
        cumulative += ((logit - max_logit) / temperature).exp();
        if cumulative >= target {
            return u32::try_from(index).map_err(|_| "Token ID does not fit u32".into());
        }
    }
    u32::try_from(logits.len() - 1).map_err(|_| "Token ID does not fit u32".into())
}

pub fn static_q8_matrix(
    source: &dyn TensorSource,
    name: &str,
    columns: usize,
    rows: usize,
) -> Result<&'static [u8], String> {
    static_q8_tensor(
        source,
        name,
        &[
            usize_to_u64(columns, "matrix columns")?,
            usize_to_u64(rows, "matrix rows")?,
        ],
    )
}

pub fn static_q8_tensor(
    source: &dyn TensorSource,
    name: &str,
    dims: &[u64],
) -> Result<&'static [u8], String> {
    let bytes = checked_tensor(source, name, dims, GGMLType::Q8_0)?;
    Ok(unsafe { std::mem::transmute::<&[u8], &'static [u8]>(bytes) })
}

pub fn static_tensor(
    source: &dyn TensorSource,
    name: &str,
    dims: &[u64],
    ggml_type: GGMLType,
) -> Result<&'static [u8], String> {
    let bytes = checked_tensor(source, name, dims, ggml_type)?;
    Ok(unsafe { std::mem::transmute::<&[u8], &'static [u8]>(bytes) })
}

pub fn checked_tensor<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    dims: &[u64],
    ggml_type: GGMLType,
) -> Result<&'a [u8], String> {
    let info: &TensorInfo = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims || info.ggml_type != ggml_type {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected {:?} {:?}",
            info.dims, info.ggml_type, dims, ggml_type
        ));
    }
    let expected = usize::try_from(
        info.checked_nbytes()
            .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?,
    )
    .map_err(|_| format!("Tensor byte size does not fit usize: {name}"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len() != expected {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected}",
            bytes.len()
        ));
    }
    Ok(bytes)
}

pub fn checked_product(name: &str, left: usize, right: usize) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{name} overflows usize"))
}

pub fn check_allocation(name: &str, len: usize, element_bytes: usize) -> Result<(), String> {
    let bytes = checked_product(name, len, element_bytes)?;
    if bytes > isize::MAX as usize {
        return Err(format!("{name} allocation is too large"));
    }
    Ok(())
}

pub fn usize_to_u64(value: usize, name: &str) -> Result<u64, String> {
    u64::try_from(value).map_err(|_| format!("{name} does not fit u64"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen3::trunk::config::{Qwen3Config, Qwen3Rope};
    use crate::models::qwen3::trunk::tests::qwen3vl_metadata_source;

    #[test]
    fn qwen3vl_requires_qk_norm_and_fixed_imrope_sections() {
        let config = Qwen3Config::from_source(&qwen3vl_metadata_source()).unwrap();
        assert!(config.has_qk_norm);
        assert_eq!(
            config.rope,
            Qwen3Rope::Interleaved {
                sections: [24, 20, 20, 0],
                n_dims: 128,
            }
        );
    }

    #[test]
    fn session_capacity_is_prompt_plus_generation_not_model_context() {
        assert_eq!(checked_session_capacity(23, 17, 65_536).unwrap(), 40);
        assert!(checked_session_capacity(65_500, 37, 65_536).is_err());
        assert!(checked_session_capacity(usize::MAX, 1, 65_536).is_err());
    }

    #[test]
    fn decoder_does_not_evaluate_the_last_generated_token() {
        assert_eq!(checked_decoder_steps(23, 17, 65_536).unwrap(), 39);
    }

    #[test]
    fn generated_positions_continue_from_prompt_text_positions() {
        let prompt = [[7, 8, 9, 10], [42, 100, 200, 300]];
        assert_eq!(
            checked_generated_position(&prompt, 0).unwrap(),
            [43, 43, 43, 0]
        );
        assert_eq!(
            checked_generated_position(&prompt, 1).unwrap(),
            [44, 44, 44, 0]
        );
        assert!(checked_generated_position(&[[usize::MAX; 4]], 0).is_err());
    }

    #[test]
    fn decoder_input_rejects_position_and_embedding_shape_mismatch() {
        assert!(validate_input_shapes(3, 1024, 2, None).is_err());
        assert!(validate_input_shapes(3, 1024, 3, Some(3 * 1024 - 1)).is_err());
    }

    #[test]
    fn greedy_ties_choose_the_lowest_token_id() {
        assert_eq!(greedy_token(&[1.0, 2.0, 2.0]).unwrap(), 1);
        assert!(greedy_token(&[1.0, f32::NAN]).is_err());
    }
}
