use std::sync::{Arc, Mutex};

use image::RgbImage;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::creator::{
    add_residual, apply_rope, gated_residual, modulated_layer_norm, Branch, BranchDimensions,
    BranchKind, Rope,
};
use super::kernels::{
    attention_online, checked_len, layer_norm_rows, rms_norm_rows, AttentionSpec,
};
use super::lightvae::LightVae;
use super::text::TextConditioning;
use super::upsampler::LatentUpsampler;
use super::video_vae::{VideoLatent, Wan22Vae};
use super::{DreamXOptions, DreamXRefinerOptions, RefinerDecoderKind};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;

const DIT_PREFIX: &str = "dreamx.refiner.dit";
const CHANNELS: usize = 48;
const DIM: usize = 3072;
const HEADS: usize = 24;
const HEAD_DIM: usize = 128;
const TEXT_TOKENS: usize = 512;
const LAYERS: usize = 30;
const CHUNK_FRAMES: usize = 3;
const EPSILON: f32 = 1e-6;
const SIGMA_START: f32 = 0.6251;
const REFINER_TIMESTEPS: [f32; 4] = [1000.0, 750.0, 500.0, 250.0];

fn refiner_timesteps() -> [f32; 4] {
    REFINER_TIMESTEPS
}

struct TinyKvCache<T> {
    max_frames: usize,
    first_frame_index: usize,
    frames: Vec<T>,
}

impl<T> TinyKvCache<T> {
    fn new(max_frames: usize) -> Self {
        Self {
            max_frames,
            first_frame_index: 0,
            frames: Vec::new(),
        }
    }

    fn push(&mut self, mut frames: Vec<T>) {
        self.frames.append(&mut frames);
        let dropped = self.frames.len().saturating_sub(self.max_frames);
        self.frames.drain(..dropped);
        self.first_frame_index += dropped;
    }

    fn frames(&self) -> usize {
        self.frames.len()
    }

    fn first_frame_index(&self) -> usize {
        self.first_frame_index
    }

    fn clear(&mut self) {
        self.frames.clear();
        self.first_frame_index = 0;
    }
}

#[derive(Clone, Copy)]
struct WindowSpec {
    cached_frames: usize,
    block: [usize; 2],
    radius: [usize; 2],
}

impl WindowSpec {
    fn released() -> Self {
        Self::with_cached_frames(9)
    }

    fn with_cached_frames(cached_frames: usize) -> Self {
        Self {
            cached_frames,
            block: [4, 4],
            radius: [3, 3],
        }
    }
}

#[derive(Clone, Copy)]
struct QueryChunk {
    first_frame: usize,
    frames: usize,
}

impl QueryChunk {
    fn new(first_frame: usize, frames: usize) -> Self {
        Self {
            first_frame,
            frames,
        }
    }
}

fn window_key_indices(spec: WindowSpec, query: QueryChunk) -> Vec<usize> {
    let first = query.first_frame.saturating_sub(spec.cached_frames);
    let end = query.first_frame.saturating_add(query.frames);
    (first..end).collect()
}

fn axis_blocks(size: usize, block: usize, radius: usize) -> Vec<(usize, usize, usize, usize)> {
    if size == 0 || block == 0 {
        return Vec::new();
    }
    let blocks = size.div_ceil(block);
    (0..blocks)
        .map(|index| {
            let query_start = index * block;
            let query_end = ((index + 1) * block).min(size);
            let key_start = index.saturating_sub(radius) * block;
            let key_end =
                (index.saturating_add(radius).saturating_add(1).min(blocks) * block).min(size);
            (
                query_start,
                query_end - query_start,
                key_start,
                key_end - key_start,
            )
        })
        .collect()
}

struct KvFrame {
    key: Vec<f32>,
    value: Vec<f32>,
}

struct RefinerModel {
    source: Arc<dyn TensorSource>,
    pool: Arc<ComputePool>,
    branch: Branch,
    caches: Vec<TinyKvCache<KvFrame>>,
    window: WindowSpec,
}

impl RefinerModel {
    fn load(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
        cached_frames: usize,
    ) -> Result<Self, String> {
        if cached_frames == 0 {
            return Err("DreamX refiner KV length must be non-zero".into());
        }
        let branch = Branch::load(
            source.as_ref(),
            DIT_PREFIX,
            BranchKind::Video,
            BranchDimensions::VIDEO,
        )?;
        Ok(Self {
            source,
            pool,
            branch,
            caches: (0..LAYERS)
                .map(|_| TinyKvCache::new(cached_frames))
                .collect(),
            window: WindowSpec::with_cached_frames(cached_frames),
        })
    }

    fn clear_cache(&mut self) {
        for cache in &mut self.caches {
            cache.clear();
        }
    }

    fn forward_chunk(
        &mut self,
        latent: &VideoLatent,
        context: &[f32],
        timestep: f32,
        temporal_offset: usize,
        refresh_cache: bool,
    ) -> Result<VideoLatent, String> {
        let [channels, frames, height, width] = latent.shape();
        if channels != CHANNELS
            || frames == 0
            || frames > CHUNK_FRAMES
            || !height.is_multiple_of(2)
            || !width.is_multiple_of(2)
            || !timestep.is_finite()
            || !(0.0..=1000.0).contains(&timestep)
            || context.len() != TEXT_TOKENS * DIM
        {
            return Err("Invalid DreamX refiner chunk".into());
        }
        temporal_offset
            .checked_add(frames)
            .ok_or("DreamX refiner temporal offset overflow")?;

        let grid = [frames, height / 2, width / 2];
        let tokens = checked_len("DreamX refiner tokens", &grid)?;
        let mut hidden = self
            .branch
            .patch_video(&self.pool, latent.as_slice(), latent.shape())?;
        let (time_embedding, time_modulation) =
            self.branch
                .time_embeddings(self.source.as_ref(), &self.pool, &[timestep])?;

        let source = self.source.as_ref();
        let pool = self.pool.as_ref();
        let branch = &self.branch;
        for (block, cache) in branch.blocks.iter().zip(&mut self.caches) {
            let normalized = modulated_layer_norm(
                &hidden,
                tokens,
                DIM,
                &block.modulation,
                &time_modulation,
                1,
                0,
                1,
            )?;
            let attention = &block.self_attention;
            let mut query = attention.q.forward(source, pool, &normalized, tokens)?;
            let mut key = attention.k.forward(source, pool, &normalized, tokens)?;
            let value = attention.v.forward(source, pool, &normalized, tokens)?;
            query = rms_norm_rows(&query, tokens, &attention.norm_q, EPSILON)?;
            key = rms_norm_rows(&key, tokens, &attention.norm_k, EPSILON)?;
            let rope = Rope::VideoOffset(grid, temporal_offset);
            apply_rope(&mut query, tokens, HEADS, rope)?;
            apply_rope(&mut key, tokens, HEADS, rope)?;

            let attended = window_attention(
                pool,
                &query,
                &key,
                &value,
                grid,
                temporal_offset,
                cache,
                self.window,
            )?;
            let residual = attention.o.forward(source, pool, &attended, tokens)?;
            gated_residual(
                &mut hidden,
                &residual,
                tokens,
                DIM,
                &block.modulation,
                &time_modulation,
                1,
                2,
            )?;

            let normalized = layer_norm_rows(
                &hidden,
                tokens,
                Some(&block.norm3_weight),
                Some(&block.norm3_bias),
                EPSILON,
            )?;
            let residual = block.text_attention.forward(
                source,
                pool,
                &normalized,
                tokens,
                context,
                TEXT_TOKENS,
                Rope::None,
                Rope::None,
            )?;
            add_residual(&mut hidden, &residual)?;
            block.feed_forward(source, pool, &mut hidden, tokens, &time_modulation, 1)?;

            if refresh_cache {
                cache.push(split_kv_frames(&key, &value, frames, grid[1] * grid[2])?);
            }
        }

        let patches = branch
            .head
            .forward(source, pool, &hidden, tokens, &time_embedding, 1)?;
        let flow = branch.unpatch_video(&patches, latent.shape())?;
        let sigma = timestep as f64 / 1000.0;
        let values = latent
            .as_slice()
            .iter()
            .zip(flow.as_slice())
            .map(|(&sample, &velocity)| (sample as f64 - sigma * velocity as f64) as f32)
            .collect();
        VideoLatent::new(values, latent.shape())
    }
}

enum Decoder {
    Wan,
    Light(LightVae),
}

pub struct DreamXRefiner {
    model: RefinerModel,
    vae: Wan22Vae,
    upsampler: LatentUpsampler,
    decoder: Decoder,
    selected: DreamXRefinerOptions,
}

impl DreamXRefiner {
    pub fn load(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
        options: DreamXRefinerOptions,
    ) -> Result<Self, String> {
        if options.kv_len == 0 {
            return Err("DreamX refiner KV length must be non-zero".into());
        }
        let model = RefinerModel::load(source.clone(), pool.clone(), options.kv_len)?;
        let vae = Wan22Vae::load(source.clone(), pool.clone())?;
        let upsampler =
            LatentUpsampler::load(options.latent_upsample, source.clone(), pool.clone())?;
        let decoder = match options.decoder {
            RefinerDecoderKind::Wan => Decoder::Wan,
            RefinerDecoderKind::LightVae => Decoder::Light(LightVae::load(source, pool)?),
        };
        Ok(Self {
            model,
            vae,
            upsampler,
            decoder,
            selected: options,
        })
    }

    pub fn refine(
        &mut self,
        frames: &[RgbImage],
        conditioning: &TextConditioning,
        options: &DreamXOptions,
    ) -> Result<Vec<RgbImage>, String> {
        if options.refiner != self.selected || frames.is_empty() {
            return Err("DreamX refiner options do not match the loaded graph".into());
        }
        let low_resolution = self.vae.encode_frames(frames)?;
        let upsampled = self.upsampler.upsample(&low_resolution)?;
        let context = self.model.branch.embed_context(
            self.model.source.as_ref(),
            &self.model.pool,
            &conditioning.positive,
        )?;
        self.model.clear_cache();
        let refined = refine_latent(&mut self.model, &upsampled, &context, options.seed as u64);
        self.model.clear_cache();
        let refined = refined?;
        match &self.decoder {
            Decoder::Wan => self.vae.decode_frames(&refined),
            Decoder::Light(decoder) => decoder.decode_frames(&refined),
        }
    }
}

fn refine_latent(
    model: &mut RefinerModel,
    upsampled: &VideoLatent,
    context: &[f32],
    seed: u64,
) -> Result<VideoLatent, String> {
    let shape = upsampled.shape();
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = gaussian_values(&mut rng, upsampled.as_slice().len());
    let noisy = blend_with_noise(upsampled.as_slice(), &noise, SIGMA_START)?;
    let mut output = vec![0.0; upsampled.as_slice().len()];

    for start in (0..shape[1]).step_by(CHUNK_FRAMES) {
        let frame_count = CHUNK_FRAMES.min(shape[1] - start);
        let mut chunk = slice_frames(&noisy, shape, start, frame_count)?;
        for (step, timestep) in refiner_timesteps().into_iter().enumerate() {
            let latent = VideoLatent::new(chunk, [shape[0], frame_count, shape[2], shape[3]])?;
            let denoised = model.forward_chunk(&latent, context, timestep, start, false)?;
            if step + 1 == REFINER_TIMESTEPS.len() {
                chunk = denoised.into_values();
            } else {
                let noise = gaussian_values(&mut rng, denoised.as_slice().len());
                let next_sigma = REFINER_TIMESTEPS[step + 1] / 1000.0;
                chunk = blend_with_noise(denoised.as_slice(), &noise, next_sigma)?;
            }
        }
        let denoised = VideoLatent::new(chunk, [shape[0], frame_count, shape[2], shape[3]])?;
        write_frames(&mut output, shape, start, &denoised)?;
        model.forward_chunk(&denoised, context, 0.0, start, true)?;
    }
    VideoLatent::new(output, shape)
}

fn window_attention(
    pool: &ComputePool,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    grid: [usize; 3],
    temporal_offset: usize,
    cache: &TinyKvCache<KvFrame>,
    spec: WindowSpec,
) -> Result<Vec<f32>, String> {
    let [frames, height, width] = grid;
    let frame_tokens = checked_len("DreamX refiner frame tokens", &[height, width])?;
    let tokens = checked_len("DreamX refiner attention tokens", &grid)?;
    let expected = checked_len("DreamX refiner attention width", &[tokens, DIM])?;
    let visible = window_key_indices(spec, QueryChunk::new(temporal_offset, frames));
    let history = visible.len().saturating_sub(frames);
    if query.len() != expected
        || key.len() != expected
        || value.len() != expected
        || cache.frames() != history
        || cache.frames() > spec.cached_frames
        || cache.first_frame_index().checked_add(cache.frames()) != Some(temporal_offset)
        || cache.frames.iter().any(|frame| {
            frame.key.len() != frame_tokens * DIM || frame.value.len() != frame_tokens * DIM
        })
    {
        return Err("Invalid DreamX refiner window tensors".into());
    }

    let height_blocks = axis_blocks(height, spec.block[0], spec.radius[0]);
    let width_blocks = axis_blocks(width, spec.block[1], spec.radius[1]);
    let windows: Vec<_> = height_blocks
        .iter()
        .flat_map(|height| width_blocks.iter().map(move |width| (*height, *width)))
        .collect();
    let mut output = vec![0.0; query.len()];
    let output_address = output.as_mut_ptr() as usize;
    let error = Mutex::new(None);
    pool.compute(|thread, threads| {
        for window in (thread..windows.len()).step_by(threads) {
            if error.lock().unwrap().is_some() {
                return;
            }
            let (
                (query_y, query_height, key_y, key_height),
                (query_x, query_width, key_x, key_width),
            ) = windows[window];
            let query_tokens = frames * query_height * query_width;
            let key_tokens = (cache.frames() + frames) * key_height * key_width;
            let mut gathered_query = Vec::with_capacity(query_tokens * DIM);
            let mut gathered_key = Vec::with_capacity(key_tokens * DIM);
            let mut gathered_value = Vec::with_capacity(key_tokens * DIM);
            gather_current(
                query,
                grid,
                query_y,
                query_height,
                query_x,
                query_width,
                &mut gathered_query,
            );
            for frame in &cache.frames {
                gather_frame(
                    &frame.key,
                    height,
                    width,
                    key_y,
                    key_height,
                    key_x,
                    key_width,
                    &mut gathered_key,
                );
                gather_frame(
                    &frame.value,
                    height,
                    width,
                    key_y,
                    key_height,
                    key_x,
                    key_width,
                    &mut gathered_value,
                );
            }
            gather_current(
                key,
                grid,
                key_y,
                key_height,
                key_x,
                key_width,
                &mut gathered_key,
            );
            gather_current(
                value,
                grid,
                key_y,
                key_height,
                key_x,
                key_width,
                &mut gathered_value,
            );
            let attended = attention_online(
                &gathered_query,
                &gathered_key,
                &gathered_value,
                AttentionSpec {
                    query_tokens,
                    key_tokens,
                    query_heads: HEADS,
                    key_value_heads: HEADS,
                    head_dim: HEAD_DIM,
                    causal: false,
                    scale: 1.0 / (HEAD_DIM as f32).sqrt(),
                },
            );
            match attended {
                Ok(attended) => unsafe {
                    scatter_current(
                        output_address as *mut f32,
                        &attended,
                        grid,
                        query_y,
                        query_height,
                        query_x,
                        query_width,
                    )
                },
                Err(message) => *error.lock().unwrap() = Some(message),
            }
        }
    });
    if let Some(error) = error.into_inner().unwrap() {
        return Err(error);
    }
    Ok(output)
}

fn gather_current(
    values: &[f32],
    grid: [usize; 3],
    top: usize,
    height: usize,
    left: usize,
    width: usize,
    output: &mut Vec<f32>,
) {
    let [frames, grid_height, grid_width] = grid;
    let frame_width = grid_height * grid_width * DIM;
    for frame in 0..frames {
        gather_frame(
            &values[frame * frame_width..(frame + 1) * frame_width],
            grid_height,
            grid_width,
            top,
            height,
            left,
            width,
            output,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn gather_frame(
    values: &[f32],
    _grid_height: usize,
    grid_width: usize,
    top: usize,
    height: usize,
    left: usize,
    width: usize,
    output: &mut Vec<f32>,
) {
    for y in top..top + height {
        for x in left..left + width {
            let start = (y * grid_width + x) * DIM;
            output.extend_from_slice(&values[start..start + DIM]);
        }
    }
}

unsafe fn scatter_current(
    output: *mut f32,
    values: &[f32],
    grid: [usize; 3],
    top: usize,
    height: usize,
    left: usize,
    width: usize,
) {
    let [frames, grid_height, grid_width] = grid;
    let mut source = 0;
    for frame in 0..frames {
        for y in top..top + height {
            for x in left..left + width {
                let target = ((frame * grid_height + y) * grid_width + x) * DIM;
                std::ptr::copy_nonoverlapping(values.as_ptr().add(source), output.add(target), DIM);
                source += DIM;
            }
        }
    }
}

fn split_kv_frames(
    key: &[f32],
    value: &[f32],
    frames: usize,
    frame_tokens: usize,
) -> Result<Vec<KvFrame>, String> {
    let width = checked_len("DreamX refiner KV frame", &[frame_tokens, DIM])?;
    if frames == 0 || key.len() != frames * width || value.len() != key.len() {
        return Err("Invalid DreamX refiner KV cache update".into());
    }
    Ok((0..frames)
        .map(|frame| KvFrame {
            key: key[frame * width..(frame + 1) * width].to_vec(),
            value: value[frame * width..(frame + 1) * width].to_vec(),
        })
        .collect())
}

fn slice_frames(
    values: &[f32],
    shape: [usize; 4],
    start: usize,
    frames: usize,
) -> Result<Vec<f32>, String> {
    let [channels, total_frames, height, width] = shape;
    let plane = checked_len("DreamX refiner frame plane", &[height, width])?;
    if frames == 0
        || start
            .checked_add(frames)
            .is_none_or(|end| end > total_frames)
        || values.len() != checked_len("DreamX refiner latent", &shape)?
    {
        return Err("Invalid DreamX refiner frame slice".into());
    }
    let mut output = Vec::with_capacity(channels * frames * plane);
    for channel in 0..channels {
        let offset = channel * total_frames * plane + start * plane;
        output.extend_from_slice(&values[offset..offset + frames * plane]);
    }
    Ok(output)
}

fn write_frames(
    output: &mut [f32],
    shape: [usize; 4],
    start: usize,
    chunk: &VideoLatent,
) -> Result<(), String> {
    let [channels, total_frames, height, width] = shape;
    let chunk_shape = chunk.shape();
    let frames = chunk_shape[1];
    let plane = checked_len("DreamX refiner frame plane", &[height, width])?;
    if chunk_shape[0] != channels
        || chunk_shape[2..] != shape[2..]
        || start
            .checked_add(frames)
            .is_none_or(|end| end > total_frames)
        || output.len() != checked_len("DreamX refiner output", &shape)?
    {
        return Err("Invalid DreamX refiner frame write".into());
    }
    for channel in 0..channels {
        let target = channel * total_frames * plane + start * plane;
        let source = channel * frames * plane;
        output[target..target + frames * plane]
            .copy_from_slice(&chunk.as_slice()[source..source + frames * plane]);
    }
    Ok(())
}

fn blend_with_noise(clean: &[f32], noise: &[f32], sigma: f32) -> Result<Vec<f32>, String> {
    if clean.len() != noise.len() || !sigma.is_finite() || !(0.0..=1.0).contains(&sigma) {
        return Err("Invalid DreamX refiner noise blend".into());
    }
    Ok(clean
        .iter()
        .zip(noise)
        .map(|(&clean, &noise)| {
            ((1.0 - sigma) as f64 * clean as f64 + sigma as f64 * noise as f64) as f32
        })
        .collect())
}

fn gaussian_values<R: Rng + ?Sized>(rng: &mut R, len: usize) -> Vec<f32> {
    (0..len)
        .map(|_| {
            let first = rng.gen::<f32>().max(1e-9);
            let second = rng.gen::<f32>();
            (-2.0 * first.ln()).sqrt() * (2.0 * std::f32::consts::PI * second).cos()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames(count: usize) -> Vec<usize> {
        (0..count).collect()
    }

    #[test]
    fn refiner_uses_four_warped_steps() {
        assert_eq!(refiner_timesteps(), [1000.0, 750.0, 500.0, 250.0]);
    }

    #[test]
    fn kv_cache_keeps_only_requested_latent_frames() {
        let mut cache = TinyKvCache::new(9);
        cache.push(frames(12));
        assert_eq!(cache.frames(), 9);
        assert_eq!(cache.first_frame_index(), 3);
    }

    #[test]
    fn causal_window_never_reads_future_chunks() {
        let keys = window_key_indices(WindowSpec::released(), QueryChunk::new(6, 3));
        assert!(keys.iter().all(|&frame| frame <= 8));
    }

    #[test]
    fn released_spatial_window_uses_block_radius() {
        assert_eq!(
            axis_blocks(20, 4, 3),
            vec![
                (0, 4, 0, 16),
                (4, 4, 0, 20),
                (8, 4, 0, 20),
                (12, 4, 0, 20),
                (16, 4, 4, 16),
            ]
        );
    }
}
