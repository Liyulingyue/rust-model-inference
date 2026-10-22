//! FLUX 3 Image Klein distilled BF16 (BFL bfc8889a).
mod dit;

use crate::core::{
    tensor::TensorSource,
    thread_pool::ComputePool,
    tokenizer::{BPETokenizer, EncodeOptions},
};
use crate::models::diffusion::{
    dreamx::kernels::load_float_values,
    z_image::{dit::TorchMt19937, vae::FluxVae},
    DiffusionRgb,
};
use crate::models::qwen3::Qwen3Model;
use std::sync::Arc;

const HIDDEN: usize = 3072;
const TEXT_WIDTH: usize = 20480;
const OUTPUT_LAYERS: [usize; 8] = [4, 8, 12, 16, 20, 24, 28, 32];
const MAX_TEXT: usize = 4096;

pub struct Flux3Pipeline {
    dit: dit::Flux3Dit,
    text: Qwen3Model,
    tokenizer: Arc<BPETokenizer>,
    vae: FluxVae,
    mean: Vec<f32>,
    std: Vec<f32>,
}

pub(crate) struct TextInput {
    pub ids: Vec<u32>,
    pub positions: Vec<[usize; 4]>,
    pub mask: Vec<bool>,
}

pub(crate) fn text_input(tokenizer: &BPETokenizer, prompt: &str) -> TextInput {
    let rendered = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    let mut ids = tokenizer.encode(
        &rendered,
        EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    );
    ids.truncate(MAX_TEXT);
    let real = ids.len();
    ids.resize(real.div_ceil(80).min(MAX_TEXT.div_ceil(80)) * 80, 151643);
    ids.truncate(MAX_TEXT);
    let positions = (0..ids.len())
        .map(|i| [if i < real { i } else { 1 }; 4])
        .collect();
    let mask = (0..ids.len()).map(|i| i < real).collect();
    TextInput {
        ids,
        positions,
        mask,
    }
}

/// Pinned official Pipeline.generate default, rather than the Hub config's older alpha.
pub fn schedule(steps: usize) -> Result<Vec<f32>, String> {
    if steps == 0 || steps > 1000 {
        return Err("FLUX3 steps must be in 1..=1000".into());
    }
    Ok((0..=steps)
        .map(|i| {
            let t = 1.0 - i as f32 / steps as f32;
            16.0 * t / (1.0 + 15.0 * t)
        })
        .collect())
}

fn image_ids(side: usize, reference: Option<usize>) -> Vec<[f32; 4]> {
    (0..side * side)
        .map(|i| {
            [
                0.0,
                (i / side) as f32 * 0.5,
                (i % side) as f32 * 0.5,
                reference.map_or(0.0, |r| -(1.0 + r as f32)),
            ]
        })
        .collect()
}

fn bf16_round(values: &mut [f32]) {
    for value in values {
        *value = crate::ops::bf16_to_f32(crate::ops::f32_to_bf16(*value));
    }
}

fn encode_text(
    model: &Qwen3Model,
    tokenizer: &BPETokenizer,
    prompt: &str,
) -> Result<Vec<f32>, String> {
    let input = text_input(tokenizer, prompt);
    let raw_ids: Vec<_> = input.ids.iter().map(|&id| id as f32).collect();
    trace("flux3.token_ids", &[input.ids.len()], &raw_ids)?;
    let mut text = model.text_encode_hidden_layers(
        &input.ids,
        &input.positions,
        &input.mask,
        &OUTPUT_LAYERS,
    )?;
    bf16_round(&mut text);
    trace("flux3.text", &[input.ids.len(), TEXT_WIDTH], &text)?;
    Ok(text)
}

fn validate_vae(source: &dyn TensorSource) -> Result<(), String> {
    use crate::core::tensor::GGMLType;
    use crate::models::diffusion::common::require_tensor;
    let norm = |name: &str, width: u64| -> Result<(), String> {
        for suffix in ["weight", "bias"] {
            require_tensor(source, &format!("{name}.{suffix}"), &[width], GGMLType::F32)?;
        }
        Ok(())
    };
    let conv = |name: &str, input: u64, output: u64, kernel: u64| -> Result<(), String> {
        require_tensor(
            source,
            &format!("{name}.weight"),
            &[kernel, kernel, input, output],
            GGMLType::F16,
        )?;
        require_tensor(source, &format!("{name}.bias"), &[output], GGMLType::F32)
    };
    let block = |name: &str, input, output| -> Result<(), String> {
        norm(&format!("{name}.norm1"), input)?;
        norm(&format!("{name}.norm2"), output)?;
        conv(&format!("{name}.conv1"), input, output, 3)?;
        conv(&format!("{name}.conv2"), output, output, 3)?;
        if input != output {
            conv(&format!("{name}.nin_shortcut"), input, output, 1)?;
        }
        Ok(())
    };
    for (mode, input, first, last, output) in
        [("encoder", 3, 128, 512, 64), ("decoder", 32, 512, 128, 3)]
    {
        conv(&format!("{mode}.conv_in"), input, first, 3)?;
        for i in [1, 2] {
            block(&format!("{mode}.mid.block_{i}"), 512, 512)?;
        }
        norm(&format!("{mode}.mid.attn_1.norm"), 512)?;
        for name in ["q", "k", "v", "proj_out"] {
            conv(&format!("{mode}.mid.attn_1.{name}"), 512, 512, 1)?;
        }
        norm(&format!("{mode}.norm_out"), last)?;
        conv(&format!("{mode}.conv_out"), last, output, 3)?;
    }
    conv("encoder.quant_conv", 64, 64, 1)?;
    conv("post_quant_conv", 32, 32, 1)?;
    for (stage, input, output) in [(0, 128, 128), (1, 128, 256), (2, 256, 512), (3, 512, 512)] {
        for i in 0..2 {
            block(
                &format!("encoder.down.{stage}.block.{i}"),
                if i == 0 { input } else { output },
                output,
            )?;
        }
        if stage != 3 {
            conv(
                &format!("encoder.down.{stage}.downsample.conv"),
                output,
                output,
                3,
            )?;
        }
    }
    for (stage, input, output) in [(3, 512, 512), (2, 512, 512), (1, 512, 256), (0, 256, 128)] {
        for i in 0..3 {
            block(
                &format!("decoder.up.{stage}.block.{i}"),
                if i == 0 { input } else { output },
                output,
            )?;
        }
        if stage != 0 {
            conv(
                &format!("decoder.up.{stage}.upsample.conv"),
                output,
                output,
                3,
            )?;
        }
    }
    for name in ["bn.running_mean", "bn.running_var"] {
        require_tensor(source, name, &[128], GGMLType::F32)?;
    }
    Ok(())
}

impl Flux3Pipeline {
    pub fn load(
        dit: Arc<dyn TensorSource>,
        text: Arc<dyn TensorSource>,
        vae: Arc<dyn TensorSource>,
        threads: usize,
    ) -> Result<Self, String> {
        if threads == 0 {
            return Err("FLUX3 requires positive threads".into());
        }
        if text
            .metadata("general.architecture")
            .and_then(|v| v.to_string_val())
            != Some("qwen3vl")
            || vae
                .metadata("general.architecture")
                .and_then(|v| v.to_string_val())
                != Some("flux3_vae")
        {
            return Err("FLUX3 requires its Qwen3-VL-4B text encoder and FLUX3 VAE".into());
        }
        dit::validate_dit(dit.as_ref())?;
        validate_vae(vae.as_ref())?;
        let pool = Arc::new(ComputePool::new(threads));
        let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|key| {
            text.metadata(key).cloned()
        })?);
        for (piece, expected) in [
            ("<|endoftext|>", 151643),
            ("<|im_start|>", 151644),
            ("<|im_end|>", 151645),
        ] {
            if tokenizer.token_id(piece) != Some(expected) {
                return Err(format!("Invalid FLUX3 tokenizer: {piece}"));
            }
        }
        let text = Qwen3Model::from_source(text, tokenizer.clone(), pool.clone())?;
        let cfg = text.config();
        if cfg.n_embd != 2560
            || cfg.n_layer != 36
            || cfg.n_head != 32
            || cfg.n_head_kv != 8
            || cfg.n_embd_head_k != 128
            || cfg.n_ff != 9728
            || cfg.freq_base != 5000000.0
        {
            return Err("FLUX3 requires the released Qwen3-VL-4B text dimensions".into());
        }
        let mean = load_float_values(vae.as_ref(), "bn.running_mean", &[128])?;
        let variance = load_float_values(vae.as_ref(), "bn.running_var", &[128])?;
        crate::models::diffusion::common::require_finite(&mean, "FLUX3 VAE mean")?;
        crate::models::diffusion::common::require_finite(&variance, "FLUX3 VAE variance")?;
        if variance.iter().any(|v| *v < 0.0) {
            return Err("Invalid FLUX3 VAE variance".into());
        }
        let std = variance.iter().map(|v| (v + 1e-4).sqrt()).collect();
        Ok(Self {
            dit: dit::Flux3Dit::load(dit, pool.clone())?,
            text,
            tokenizer,
            vae: FluxVae::load_flux3(vae, pool)?,
            mean,
            std,
        })
    }

    pub fn generate(
        &self,
        prompt: &str,
        side: usize,
        steps: usize,
        seed: u64,
        references: &[image::RgbImage],
    ) -> Result<DiffusionRgb, String> {
        if !(64..=4096).contains(&side) || side % 16 != 0 || references.len() > 10 {
            return Err("FLUX3 requires a square 64..4096 image on the 16-pixel grid and at most 10 references".into());
        }
        let schedule = schedule(steps)?;
        let text = encode_text(&self.text, &self.tokenizer, prompt)?;
        let mut reference_tokens = Vec::new();
        let mut reference_ids = Vec::new();
        for (index, rgb) in references.iter().enumerate() {
            if rgb.width() != rgb.height()
                || !(64..=4096).contains(&rgb.width())
                || rgb.width() % 16 != 0
            {
                return Err(
                    "FLUX3 reference images must be square, 64..4096, on the 16-pixel grid".into(),
                );
            }
            let ref_side = rgb.width() as usize / 16;
            let mean = self
                .vae
                .encode_flux3_mean(rgb.as_raw(), rgb.width() as usize)?;
            let mut tokens = pack_mean(&mean, ref_side, &self.mean, &self.std)?;
            bf16_round(&mut tokens);
            trace(
                &format!("flux3.reference.{index}"),
                &[ref_side * ref_side, 128],
                &tokens,
            )?;
            reference_tokens.extend(tokens);
            reference_ids.extend(image_ids(ref_side, Some(index)));
        }
        let latent_side = side / 16;
        let mut latent = vec![0.0; latent_side * latent_side * 128];
        TorchMt19937::new(seed).fill_normal(&mut latent);
        bf16_round(&mut latent);
        trace("flux3.noise", &[latent_side * latent_side, 128], &latent)?;
        let ids = image_ids(latent_side, None);
        for (step, pair) in schedule.windows(2).enumerate() {
            let velocity = self.dit.predict(
                &latent,
                &ids,
                &text,
                &reference_tokens,
                &reference_ids,
                pair[0],
                pair[1],
            )?;
            for (x, v) in latent.iter_mut().zip(velocity) {
                *x += (pair[1] - pair[0]) * v;
            }
            trace(
                &format!("flux3.latent.{step}"),
                &[latent_side * latent_side, 128],
                &latent,
            )?;
            eprintln!("[flux3] denoise {}/{}", step + 1, steps);
        }
        let planar = unpack_latent(&latent, latent_side, &self.mean, &self.std)?;
        trace(
            "flux3.vae_input",
            &[32, latent_side * 2, latent_side * 2],
            &planar,
        )?;
        self.vae.decode_mapped_rgb(&planar, latent_side * 2)
    }
}

fn pack_mean(planar: &[f32], side: usize, mean: &[f32], std: &[f32]) -> Result<Vec<f32>, String> {
    if side == 0 || planar.len() != side * side * 128 || mean.len() != 128 || std.len() != 128 {
        return Err("Invalid FLUX3 latent shape".into());
    }
    let mut tokens = vec![0.0; planar.len()];
    for y in 0..side {
        for x in 0..side {
            for c in 0..128 {
                let offset = ((c / 4 * side * 2 + y * 2 + c / 2 % 2) * side * 2) + x * 2 + c % 2;
                tokens[(y * side + x) * 128 + c] = (planar[offset] - mean[c]) / std[c];
            }
        }
    }
    Ok(tokens)
}

fn unpack_latent(
    tokens: &[f32],
    side: usize,
    mean: &[f32],
    std: &[f32],
) -> Result<Vec<f32>, String> {
    if side == 0 || tokens.len() != side * side * 128 || mean.len() != 128 || std.len() != 128 {
        return Err("Invalid FLUX3 latent shape".into());
    }
    let mut planar = vec![0.0; tokens.len()];
    for y in 0..side {
        for x in 0..side {
            for c in 0..128 {
                let offset = ((c / 4 * side * 2 + y * 2 + c / 2 % 2) * side * 2) + x * 2 + c % 2;
                planar[offset] = tokens[(y * side + x) * 128 + c] * std[c] + mean[c];
            }
        }
    }
    Ok(planar)
}

fn trace(_name: &str, _shape: &[usize], _values: &[f32]) -> Result<(), String> {
    #[cfg(feature = "parity-trace")]
    if crate::parity_trace::enabled(_name) {
        crate::parity_trace::checkpoint(_name, None, _shape, _values).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn official_flow_map_schedule_and_reference_positions() {
        assert_eq!(schedule(2).unwrap(), vec![1.0, 16.0 / 17.0, 0.0]);
        assert!(schedule(0).is_err());
        assert_eq!(
            image_ids(2, Some(1)),
            vec![
                [0.0, 0.0, 0.0, -2.0],
                [0.0, 0.0, 0.5, -2.0],
                [0.0, 0.5, 0.0, -2.0],
                [0.0, 0.5, 0.5, -2.0]
            ]
        );
    }
    #[test]
    fn patchify_uses_channel_dy_dx_order() {
        let planar: Vec<_> = (0..512).map(|v| v as f32).collect();
        let tokens = pack_mean(&planar, 2, &[0.0; 128], &[1.0; 128]).unwrap();
        assert_eq!(&tokens[..8], &[0.0, 1.0, 4.0, 5.0, 16.0, 17.0, 20.0, 21.0]);
        assert_eq!(
            unpack_latent(&tokens, 2, &[0.0; 128], &[1.0; 128]).unwrap(),
            planar
        );
    }
    #[test]
    #[ignore = "requires the converted tokenizer from the real released package"]
    fn released_tokenizer_matches_transformers_fixtures() {
        let path = std::env::var("RMI_FLUX3_TOKENIZER").unwrap();
        let source = crate::core::loader::GGUFLoader::from_file(&path).unwrap();
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned()).unwrap();
        let fixtures: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tools/oracle/flux3/tokenizer-fixtures.json"
        ))
        .unwrap();
        for fixture in fixtures.as_array().unwrap() {
            let input = text_input(&tokenizer, fixture["prompt"].as_str().unwrap());
            let expected: Vec<u32> = fixture["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(&input.ids[..expected.len()], &expected);
            assert_eq!(
                input.ids.len(),
                fixture["padded_length"].as_u64().unwrap() as usize
            );
            assert!(input.ids[expected.len()..].iter().all(|&v| v == 151643));
            assert!(input.mask[..expected.len()].iter().all(|v| *v));
            assert!(input.mask[expected.len()..].iter().all(|v| !*v));
            assert!(input.positions[expected.len()..]
                .iter()
                .all(|&v| v == [1; 4]));
        }
        let input = text_input(&tokenizer, &"a ".repeat(5000));
        assert_eq!(input.ids.len(), 4096);
        assert!(input.mask.iter().all(|v| *v));
        assert_eq!(input.positions[4095], [4095; 4]);
    }
    #[test]
    #[ignore = "requires the real converted FLUX3 text encoder"]
    fn released_text_hidden_features() {
        let path = std::env::var("RMI_FLUX3_TEXT").unwrap();
        let source: Arc<dyn TensorSource> = Arc::from(
            crate::format::ggufrs::open_model_source(
                std::path::Path::new(&path),
                crate::format::ggufrs::ComponentRole::Llm,
            )
            .unwrap(),
        );
        let tokenizer =
            Arc::new(BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned()).unwrap());
        let model =
            Qwen3Model::from_source(source, tokenizer.clone(), Arc::new(ComputePool::new(8)))
                .unwrap();
        let features = encode_text(&model, &tokenizer, "A red cube on a white table.").unwrap();
        assert_eq!(features.len(), 80 * TEXT_WIDTH);
    }

    #[test]
    #[ignore = "requires the real converted FLUX3 VAE"]
    fn released_vae_encode_and_decode() {
        let path = std::env::var("RMI_FLUX3_VAE").unwrap();
        let source: Arc<dyn TensorSource> = Arc::from(
            crate::format::ggufrs::open_model_source(
                std::path::Path::new(&path),
                crate::format::ggufrs::ComponentRole::Mmproj,
            )
            .unwrap(),
        );
        validate_vae(source.as_ref()).unwrap();
        let mean = load_float_values(source.as_ref(), "bn.running_mean", &[128]).unwrap();
        let variance = load_float_values(source.as_ref(), "bn.running_var", &[128]).unwrap();
        let std: Vec<_> = variance.iter().map(|v| (v + 1e-4).sqrt()).collect();
        let vae = FluxVae::load_flux3(source, Arc::new(ComputePool::new(8))).unwrap();
        let side = 64;
        let pixels: Vec<u8> = (0..side * side * 3)
            .map(|i| ((i / 3 % side * 3 + i / 3 / side * 5 + i % 3 * 17) % 256) as u8)
            .collect();
        let moments = vae.encode_flux3_mean(&pixels, side).unwrap();
        let mut reference = pack_mean(&moments, side / 16, &mean, &std).unwrap();
        bf16_round(&mut reference);
        trace("flux3.reference.0", &[16, 128], &reference).unwrap();
        let mut noise = vec![0.0; 16 * 128];
        TorchMt19937::new(42).fill_normal(&mut noise);
        bf16_round(&mut noise);
        trace("flux3.noise", &[16, 128], &noise).unwrap();
        let planar = unpack_latent(&noise, 4, &mean, &std).unwrap();
        trace("flux3.vae_input", &[32, 8, 8], &planar).unwrap();
        let rgb = vae.decode_mapped_rgb(&planar, 8).unwrap();
        assert_eq!(
            (rgb.width, rgb.height, rgb.bytes.len()),
            (64, 64, 64 * 64 * 3)
        );
    }
}
