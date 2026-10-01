mod ar;
mod config;
mod nar;
pub mod protocol;
mod vae;
pub use ar::{YuE2ArSession, YuE2Model};
pub use config::YuE2Config;
pub use config::YuE2VaeConfig;
pub use nar::{song_chunks, YuE2Chunk, YuE2NarSession};
pub use protocol::{SamplingConfig, YuE2Protocol, YuE2Request};
pub use vae::YuE2Vae;

use crate::core::tensor::{MetaValue, TensorSource};

pub(super) fn require_string(
    source: &dyn TensorSource,
    key: &str,
    expected: &str,
) -> Result<(), String> {
    match source.metadata(key) {
        Some(MetaValue::String(value)) if value == expected => Ok(()),
        Some(value) => Err(format!(
            "Invalid {key}: expected {expected:?}, got {value:?}"
        )),
        None => Err(format!("Missing {key}: expected {expected:?}")),
    }
}

pub(super) fn require_u64(
    source: &dyn TensorSource,
    key: &str,
    expected: u64,
) -> Result<(), String> {
    match source.metadata(key).and_then(MetaValue::to_u64) {
        Some(value) if value == expected => Ok(()),
        Some(value) => Err(format!("Invalid {key}: expected {expected}, got {value}")),
        None => Err(format!("Missing or invalid {key}: expected {expected}")),
    }
}

pub(super) fn require_f64(
    source: &dyn TensorSource,
    key: &str,
    expected: f64,
) -> Result<(), String> {
    let actual = match source.metadata(key) {
        Some(MetaValue::Float32(value)) => Some(f64::from(*value)),
        Some(MetaValue::Float64(value)) => Some(*value),
        Some(MetaValue::Uint32(value)) => Some(f64::from(*value)),
        Some(MetaValue::Uint64(value)) => Some(*value as f64),
        _ => None,
    };
    match actual {
        Some(value) if value == expected => Ok(()),
        Some(value) => Err(format!("Invalid {key}: expected {expected}, got {value}")),
        None => Err(format!("Missing or invalid {key}: expected {expected}")),
    }
}

use protocol::{CODEC_OFFSET, CODEC_SIZE};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YuE2GenerateOptions {
    pub abc: SamplingConfig,
    pub semantic: SamplingConfig,
    pub steps: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct YuE2Generation {
    pub abc_ids: Vec<u32>,
    pub semantic_ids: Vec<u32>,
    pub latents: Vec<f32>,
    pub latent_frames: usize,
    pub channel_major_audio: Vec<f32>,
    pub samples_per_channel: usize,
}

impl YuE2Model {
    pub fn generate(
        &self,
        vae: &YuE2Vae,
        request: &YuE2Request,
        options: &YuE2GenerateOptions,
    ) -> Result<YuE2Generation, String> {
        options.abc.validate()?;
        options.semantic.validate()?;
        if options.steps == 0 {
            return Err("YuE2 NAR steps must be greater than zero".into());
        }
        let protocol = YuE2Protocol {
            abc: options.abc,
            semantic: options.semantic,
        };
        // A full generation is four long CPU stages (two autoregressive
        // samplers, a NAR diffusion solve and the VAE decode), so report each
        // transition; without this the CLI looks hung for minutes at a time.
        let started = std::time::Instant::now();
        let mut stage = |name: &str| {
            eprintln!(
                "[yue2] +{:.1}s {name} (max_tokens abc={} semantic={}, steps={})",
                started.elapsed().as_secs_f64(),
                options.abc.max_tokens,
                options.semantic.max_tokens,
                options.steps,
            );
        };
        stage("building ABC prefix");
        let prefix = protocol.abc_prefix(self.tokenizer(), request)?;
        stage("ABC sampling");
        let abc_ids = self.generate_abc(&prefix, options.abc, request.seed)?;
        eprintln!(
            "[yue2] +{:.1}s ABC done: {} tokens",
            started.elapsed().as_secs_f64(),
            abc_ids.len()
        );
        stage("building semantic prefix");
        let prefix = protocol.semantic_prefix(self.tokenizer(), request, &abc_ids)?;
        stage("semantic sampling");
        let semantic_ids = self.generate_semantic(&prefix, options.semantic, request.seed)?;
        eprintln!(
            "[yue2] +{:.1}s semantic done: {} tokens",
            started.elapsed().as_secs_f64(),
            semantic_ids.len()
        );
        let codec_ids = semantic_ids
            .iter()
            .map(|&token| {
                token
                    .checked_sub(CODEC_OFFSET)
                    .filter(|&token| (token as usize) < CODEC_SIZE)
                    .ok_or_else(|| {
                        format!("YuE2 semantic token {token} is outside the codec range")
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let latent_frames = codec_ids.len();
        eprintln!(
            "[yue2] +{:.1}s NAR: {latent_frames} latent frames",
            started.elapsed().as_secs_f64()
        );
        let chunks = song_chunks(&prefix, &codec_ids, request.seed, self.config().context)?;
        let chunk_total = chunks.len();
        let latent_len = latent_frames
            .checked_mul(self.config().latent_channels)
            .ok_or("YuE2 latent length overflow")?;
        let mut latents = Vec::with_capacity(latent_len);
        for (index, chunk) in chunks.into_iter().enumerate() {
            eprintln!(
                "[yue2] +{:.1}s NAR chunk {}/{chunk_total}",
                started.elapsed().as_secs_f64(),
                index + 1,
            );
            latents.extend(YuE2NarSession::new(self, chunk)?.solve(options.steps)?);
        }
        stage("NAR done");
        if latents.len() != latent_len {
            return Err("YuE2 NAR produced the wrong latent shape".into());
        }
        if latent_frames == 0
            || latents.is_empty()
            || latents.len() % latent_frames != 0
            || latents.iter().any(|value| !value.is_finite())
        {
            return Err("YuE2 NAR produced invalid frame-major latents".into());
        }
        stage("VAE decode");
        let channel_major_audio = vae.decode_tiled(&latents, latent_frames, 1024, 16)?;
        stage("done");
        if channel_major_audio.is_empty()
            || channel_major_audio.len() % 2 != 0
            || channel_major_audio.iter().any(|value| !value.is_finite())
        {
            return Err("YuE2 VAE produced invalid channel-major stereo audio".into());
        }
        let samples_per_channel = channel_major_audio.len() / 2;
        Ok(YuE2Generation {
            abc_ids,
            semantic_ids,
            latents,
            latent_frames,
            channel_major_audio,
            samples_per_channel,
        })
    }
}

pub(crate) fn interleave_stereo(
    channel_major: &[f32],
    samples_per_channel: usize,
) -> Result<Vec<f32>, String> {
    if samples_per_channel == 0
        || samples_per_channel.checked_mul(2) != Some(channel_major.len())
        || channel_major.iter().any(|sample| !sample.is_finite())
    {
        return Err("YuE2 audio must be finite channel-major stereo".into());
    }
    let (left, right) = channel_major.split_at(samples_per_channel);
    let mut interleaved = Vec::with_capacity(channel_major.len());
    for (&left, &right) in left.iter().zip(right) {
        interleaved.extend([left, right]);
    }
    Ok(interleaved)
}

#[cfg(test)]
mod tests;
