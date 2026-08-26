use crate::core::tensor::TensorSource;
use crate::models::diffusion::pig;
use crate::models::qwen3::base::Qwen3Model;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use std::sync::Arc;
use std::time::Instant;

pub fn run_pig_image(
    source: std::sync::Arc<dyn TensorSource>,
    vae_source: Option<std::sync::Arc<dyn TensorSource>>,
    text_encoder_source: Option<std::sync::Arc<dyn TensorSource>>,
    prompt: &str,
    steps: usize,
    resolution: usize,
    n_threads: usize,
) -> Result<(), String> {
    let started = Instant::now();

    let pool = Arc::new(ComputePool::new(n_threads.max(1)));
    let model = pig::PigModel::from_source(source.clone(), pool)?;

    println!("Model: pig (Z-Image) | layers={} | loaded in {}ms",
        model.config().n_layer,
        started.elapsed().as_millis());

    let vae = if let Some(vs) = vae_source {
        match pig::PigVAE::from_source(vs.as_ref()) {
            Ok(v) => {
                println!("VAE loaded successfully");
                Some(v)
            }
            Err(e) => {
                println!("Failed to load VAE: {}", e);
                None
            }
        }
    } else {
        None
    };

    println!("Generating image for prompt: {}", prompt);

    let mut session = pig::PigSession::new(&model, resolution)?;
    if let Some(ref v) = vae {
        session.set_vae(v);
    }

    let text_context = if let Some(ref te_source) = text_encoder_source {
        let te_pool = Arc::new(ComputePool::new(n_threads.max(1)));
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| te_source.metadata(k).cloned())
            .map_err(|e| format!("Failed to load text encoder tokenizer: {}", e))?;
        let text_model = Qwen3Model::from_source(
            Arc::clone(te_source),
            Arc::new(tokenizer),
            te_pool,
        ).map_err(|e| format!("Failed to load text encoder model: {}", e))?;

        let full_prompt = format!("<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n", prompt);
        let token_ids = text_model.tokenizer().encode(&full_prompt, Default::default());
        let n_tokens = token_ids.len();
        let positions: Vec<[usize; 4]> = (0..n_tokens)
            .map(|i| [i, 0, 0, 0])
            .collect();

        println!("Encoding text: {} tokens", n_tokens);
        let text_embeddings = text_model.text_encode(
            &token_ids.iter().map(|&t| t as u32).collect::<Vec<u32>>(),
            &positions,
        ).map_err(|e| format!("Text encoding failed: {}", e))?;
        println!("Text encoding done: {} dimensions", text_embeddings.len());

        text_embeddings
    } else {
        println!("WARNING: No text encoder provided; using zero context");
        let cap_dim = 2560;
        let context_len = 256;
        vec![0.0f32; cap_dim * context_len]
    };

    match session.generate_image(&text_context, steps) {
        Ok(pixels) => {
            println!("Generated {} bytes image in {}ms",
                pixels.len(), started.elapsed().as_millis());

            let img_side = (pixels.len() / 4) as u32;
            let img = image::RgbaImage::from_raw(
                img_side, img_side, pixels
            ).ok_or("Failed to create image from pixels")?;
            img.save("output.png").map_err(|e| format!("Failed to save PNG: {}", e))?;
            println!("Image saved to output.png");
        }
        Err(e) => {
            return Err(format!("Image generation failed: {}", e));
        }
    }

    Ok(())
}
