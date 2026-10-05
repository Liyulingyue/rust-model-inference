use super::{dit::trace, CONTEXT_DIM};
use crate::core::scratchpad::{KvFormat, KvLifecycle};
use crate::core::tokenizer::EncodeOptions;
use crate::models::qwen3::vision::{VisionEncoder, VisionScratchpad};
use crate::models::qwen3::{Qwen3Input, Qwen3Model, Qwen3Session};

const SYSTEM: &str = "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:";
const EDIT_SYSTEM: &str = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.";
const IMAGE: &str = "<|vision_start|><|image_pad|><|vision_end|>";

pub struct ReferenceFeatures {
    pub embeddings: Vec<f32>,
    pub deepstack: Vec<f32>,
}

pub fn encode_reference(
    encoder: &VisionEncoder<'_>,
    pixels: &[f32],
    height: usize,
    width: usize,
) -> Result<ReferenceFeatures, String> {
    if height > 512 || width > 512 || pixels.iter().any(|v| !v.is_finite()) {
        return Err("Invalid Mage reference pixels".into());
    }
    let mut scratch = VisionScratchpad::new(&encoder.config);
    encoder.encode_image(pixels, width, height, &mut scratch)?;
    Ok(ReferenceFeatures {
        embeddings: scratch.projected,
        deepstack: scratch.deepstack,
    })
}

pub fn prompt_text(prompt: &str, references: usize) -> String {
    let body = if references == 0 {
        prompt.to_string()
    } else {
        (1..=references)
            .map(|i| format!("Image {i}: {IMAGE}"))
            .collect::<String>()
            + prompt
    };
    format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n",if references==0 {SYSTEM} else {EDIT_SYSTEM})
}

pub fn encode_prompt(
    model: &Qwen3Model,
    prompt: &str,
    references: &[ReferenceFeatures],
) -> Result<Vec<f32>, String> {
    if model.config().architecture != "qwen3vl"
        || model.config().n_embd != CONTEXT_DIM
        || model.config().n_deepstack_layers != 3
        || references.len() > 3
    {
        return Err("Mage conditioning requires the released Qwen3-VL 4B text model and at most three references".into());
    }
    let raw = model.tokenizer().encode(
        &prompt_text(prompt, references.len()),
        EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    );
    let mut tokens = Vec::new();
    let mut slots = Vec::new();
    let mut reference = 0;
    for token in raw {
        if token == 151655 {
            let image = references
                .get(reference)
                .ok_or("Image placeholder without a reference")?;
            if image.embeddings.is_empty()
                || image.embeddings.len() % CONTEXT_DIM != 0
                || image.deepstack.len() != 3 * image.embeddings.len()
            {
                return Err("Invalid reference embeddings".into());
            }
            let rows = image.embeddings.len() / CONTEXT_DIM;
            slots.push((tokens.len(), rows));
            tokens.extend(std::iter::repeat_n(token, rows));
            reference += 1;
        } else {
            tokens.push(token);
        }
    }
    let drop = if references.is_empty() { 34 } else { 64 };
    if reference != references.len() || tokens.len() <= drop || tokens.len() - drop > 2048 {
        return Err(
            "Mage prompt/reference token count is invalid or exceeds 2048 conditioning tokens"
                .into(),
        );
    }
    trace(
        "text.tokens",
        1,
        tokens.len(),
        &tokens.iter().map(|&v| v as f32).collect::<Vec<_>>(),
    )?;
    let positions: Vec<_> = (0..tokens.len()).map(|p| [p; 4]).collect();
    let mut embeddings = model.embed_tokens(&tokens)?;
    let mut deepstack = vec![0.0; 3 * embeddings.len()];
    for ((start, rows), image) in slots.into_iter().zip(references) {
        let width = rows * CONTEXT_DIM;
        let start = start * CONTEXT_DIM;
        embeddings[start..start + width].copy_from_slice(&image.embeddings);
        for level in 0..3 {
            deepstack[level * embeddings.len() + start..level * embeddings.len() + start + width]
                .copy_from_slice(&image.deepstack[level * width..(level + 1) * width]);
        }
    }
    let mut session = Qwen3Session::new_with_kv_state(
        model,
        tokens.len(),
        KvFormat::F32,
        KvLifecycle::Ephemeral,
    )?;
    let output = session.forward_hidden_sequence(Qwen3Input {
        token_ids: &tokens,
        positions: &positions,
        embeddings: Some(&embeddings),
        deepstack_embeddings: (!references.is_empty()).then_some(deepstack.as_slice()),
    })?;
    let output = output[drop * CONTEXT_DIM..].to_vec();
    trace("text.context", 1, output.len(), &output)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn templates_keep_the_training_reference_order() {
        assert!(prompt_text("蓝猫", 0).contains("user\n蓝猫<|im_end|>"));
        assert!(prompt_text("edit", 2).contains(&format!("Image 1: {IMAGE}Image 2: {IMAGE}edit")));
    }
}
