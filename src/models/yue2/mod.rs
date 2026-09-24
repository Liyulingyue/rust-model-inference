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
        run_pipeline(
            || {
                let prefix = protocol.abc_prefix(self.tokenizer(), request)?;
                self.generate_abc(&prefix, options.abc, request.seed)
            },
            |abc_ids| {
                let prefix = protocol.semantic_prefix(self.tokenizer(), request, abc_ids)?;
                self.generate_semantic(&prefix, options.semantic, request.seed)
            },
            |abc_ids, semantic_ids| {
                let prefix = protocol.semantic_prefix(self.tokenizer(), request, abc_ids)?;
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
                let chunks = song_chunks(&prefix, &codec_ids, request.seed, self.config().context)?;
                let mut latents = Vec::with_capacity(
                    codec_ids
                        .len()
                        .checked_mul(self.config().latent_channels)
                        .ok_or("YuE2 latent length overflow")?,
                );
                for chunk in chunks {
                    latents.extend(YuE2NarSession::new(self, chunk)?.solve(options.steps)?);
                }
                if latents.len()
                    != codec_ids
                        .len()
                        .checked_mul(self.config().latent_channels)
                        .ok_or("YuE2 latent length overflow")?
                {
                    return Err("YuE2 NAR produced the wrong latent shape".into());
                }
                Ok((latents, codec_ids.len()))
            },
            |latents, frames| vae.decode_tiled(latents, frames, 1024, 16),
            |_| {},
        )
    }
}

fn run_pipeline<Abc, Semantic, Nar, Vae, Stage>(
    abc: Abc,
    semantic: Semantic,
    nar: Nar,
    vae: Vae,
    mut stage: Stage,
) -> Result<YuE2Generation, String>
where
    Abc: FnOnce() -> Result<Vec<u32>, String>,
    Semantic: FnOnce(&[u32]) -> Result<Vec<u32>, String>,
    Nar: FnOnce(&[u32], &[u32]) -> Result<(Vec<f32>, usize), String>,
    Vae: FnOnce(&[f32], usize) -> Result<Vec<f32>, String>,
    Stage: FnMut(&'static str),
{
    stage("abc");
    let abc_ids = abc()?;
    stage("semantic");
    let semantic_ids = semantic(&abc_ids)?;
    stage("nar");
    let (latents, latent_frames) = nar(&abc_ids, &semantic_ids)?;
    if latent_frames == 0
        || latents.is_empty()
        || latents.len() % latent_frames != 0
        || latents.iter().any(|value| !value.is_finite())
    {
        return Err("YuE2 NAR produced invalid frame-major latents".into());
    }
    stage("vae");
    let channel_major_audio = vae(&latents, latent_frames)?;
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
