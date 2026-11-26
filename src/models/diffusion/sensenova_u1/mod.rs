//! SenseNova-U1.6-8B-MoT: a unified understanding/generation model whose
//! image path is pixel-space flow matching ("MoT" = mixture of transformers).
//!
//! Unlike every other pipeline under `diffusion/`, the checkpoint is a *single*
//! GGUF that carries the text backbone, the two vision embedding towers and the
//! pixel head together, and there is no VAE at all — the decoder head emits RGB
//! directly. The pipeline therefore takes one tensor source plus the tokenizer
//! files that ship beside the weights.
//!
//! Contract: `docs/develop/SENSENOVA_U1.md`.
//! Oracle: `tools/oracle/sensenova_u1/`.

pub(crate) mod config;
pub(crate) mod fm;
pub(crate) mod sampler;
pub(crate) mod tokenizer;
pub(crate) mod trunk;
pub(crate) mod vision;

use std::path::Path;
use std::sync::Arc;

use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;

pub use config::ARCH;
pub use sampler::Options;
pub(crate) use sampler::{denoise, to_rgb8};
pub(crate) use trunk::{PrefixCache, Trunk};
pub(crate) use vision::VisionEmbeddings;

/// Decoded RGB image, channel-interleaved (`R, G, B` per pixel).
pub type Rgb8 = Vec<u8>;

/// A loaded SenseNova-U1.6 T2I pipeline.
pub struct SensenovaU1Pipeline {
    source: Arc<dyn TensorSource>,
    tokenizer: tokenizer::Tokenizer,
    sampler: sampler::Sampler,
}

impl SensenovaU1Pipeline {
    /// Validate the checkpoint and load the components that are not already
    /// validated by [`config::validate`].
    pub fn load(
        source: Arc<dyn TensorSource>,
        tokenizer_dir: &Path,
        n_threads: usize,
    ) -> Result<Self, String> {
        config::validate(source.as_ref())?;
        let trunk = Trunk::new(source.as_ref())?;
        let sampler = sampler::Sampler {
            vision_gen: VisionEmbeddings::load(source.as_ref(), "fm_modules.vision_model_mot_gen")?,
            timestep: fm::TimestepEmbedder::load(source.as_ref(), "fm_modules.timestep_embedder")?,
            noise_scale_embedder: fm::TimestepEmbedder::load(
                source.as_ref(),
                "fm_modules.noise_scale_embedder",
            )?,
            decoder: fm::ConvDecoder::load(source.as_ref())?,
            trunk,
        };
        let tokenizer = tokenizer::Tokenizer::load(tokenizer_dir)?;
        // `n_threads` is a cap; the pool clamps it to the machine.
        let _ = ComputePool::new(n_threads.max(1));
        Ok(Self {
            source,
            tokenizer,
            sampler,
        })
    }

    /// Generate one image for `prompt`.
    ///
    /// Returns `width * height * 3` interleaved RGB bytes.
    pub fn generate_rgb(&self, prompt: &str, options: &Options) -> Result<Rgb8, String> {
        sampler::validate_geometry(options.width, options.height)?;
        let pool = ComputePool::new(num_threads());
        self.generate_with_pool(prompt, options, &pool)
    }

    fn generate_with_pool(
        &self,
        prompt: &str,
        options: &Options,
        pool: &ComputePool,
    ) -> Result<Rgb8, String> {
        let source = self.source.as_ref();

        // Two prefix passes: the conditional query and the CFG negative.
        let conditional = self.tokenizer.conditional_prompt(prompt);
        let tokens_cond = self.tokenizer.encode(&conditional)?;
        if tokens_cond.is_empty() {
            return Err("SenseNova-U1.6 conditional prompt tokenised to nothing".into());
        }
        let (cache_cond, prefix_hidden_cond) =
            self.sampler
                .trunk
                .encode_prefix(source, &tokens_cond, pool)?;
        if std::env::var_os("RMI_SENSENOVA_TRACE").is_some() {
            sampler::trace_dump("prefix_last_hidden_state_cond", &prefix_hidden_cond);
            // The oracle dumps `[kv_heads, seq_slice, head_dim]` with the
            // sequence sliced to its first 64 positions.
            let slice = 64.min(cache_cond.len);
            let kv_dim = config::HEAD_DIM;
            for (layer_index, layer) in cache_cond.layers.iter().enumerate() {
                if ![0usize, 1, 21, 41].contains(&layer_index) {
                    continue;
                }
                let mut k = Vec::with_capacity(config::KV_HEADS * slice * kv_dim);
                let mut v = Vec::with_capacity(config::KV_HEADS * slice * kv_dim);
                for head in 0..config::KV_HEADS {
                    let base = head * cache_cond.len * kv_dim;
                    k.extend_from_slice(&layer.k[base..base + slice * kv_dim]);
                    v.extend_from_slice(&layer.v[base..base + slice * kv_dim]);
                }
                sampler::trace_dump(&format!("prefix_kv_layer{layer_index:02}_k"), &k);
                sampler::trace_dump(&format!("prefix_kv_layer{layer_index:02}_v"), &v);
            }
        }

        let (cache_uncond, text_len_uncond) = if options.cfg_scale > 1.0 {
            let unconditional = self.tokenizer.unconditional_prompt();
            let tokens = self.tokenizer.encode(&unconditional)?;
            let (cache, _) = self.sampler.trunk.encode_prefix(source, &tokens, pool)?;
            let len = tokens.len();
            (Some(cache), len)
        } else {
            (None, 0)
        };

        let image = denoise(
            &self.sampler,
            source,
            &cache_cond,
            cache_uncond.as_ref(),
            tokens_cond.len(),
            text_len_uncond,
            options,
            pool,
        )?;
        if image.len() != 3 * options.width * options.height {
            return Err(format!(
                "SenseNova-U1.6 denoise produced {} values, expected {}",
                image.len(),
                3 * options.width * options.height
            ));
        }
        Ok(to_rgb8(&image, options.width, options.height))
    }

    /// The tokenizer, exposed for parity harnesses that need the exact ids.
    pub fn tokenizer(&self) -> &tokenizer::Tokenizer {
        &self.tokenizer
    }
}

fn num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1)
}
