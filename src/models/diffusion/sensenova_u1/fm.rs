//! SenseNova-U1.6 flow-matching modules: the timestep embedder and the
//! pixel-space `ConvDecoder` head.
//!
//! See `references/src/sensenova_u1/models/neo_unify/modeling_fm_modules.py`
//! (`TimestepEmbedder` at ~line 23, `ConvDecoder` at ~line 573) and
//! `docs/develop/SENSENOVA_U1.md` §5.1 / §5.2.
//!
//! There is no VAE: `fm_head` is the [`ConvDecoder`], which upsamples the LLM's
//! `4096`-wide image tokens straight into a `3 x H x W` pixel prediction with
//! three pixel-shuffle stages.

use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::{QuantizedTensor, Weight};
use crate::ops::nn::{conv2d, load_float_values};
use crate::ops::rope::rope_sin_cos;
use crate::ops::{gelu_erf_inplace, silu_inplace};

use super::config::{CONV_DECODER_HIDDEN, CONV_DECODER_MID, HIDDEN, TIMESTEP_FREQ_DIM, TOKEN_SIDE};
use super::vision::load_conv_weight;

/// `256 -> 4096 -> SiLU -> 4096` sinusoidal timestep embedder.
///
/// `noise_scale_embedder` is the same module with different weights, so it can
/// be loaded through the same `prefix`.
pub(crate) struct TimestepEmbedder {
    /// `mlp.0.weight`, `[4096, 256]` row-major (out, in).
    input: Weight<'static>,
    /// `mlp.0.bias`, `[4096]`.
    input_bias: Vec<f32>,
    /// `mlp.2.weight`, `[4096, 4096]` row-major (out, in).
    output: Weight<'static>,
    /// `mlp.2.bias`, `[4096]`.
    output_bias: Vec<f32>,
}

impl TimestepEmbedder {
    /// `prefix` is `"fm_modules.timestep_embedder"` or
    /// `"fm_modules.noise_scale_embedder"`.
    pub(crate) fn load(source: &dyn TensorSource, prefix: &str) -> Result<Self, String> {
        // `nn.Sequential(Linear, SiLU, Linear)`, so the projections are `.0`
        // and `.2` -- `.1` is the activation and has no tensors.
        let (input, input_bias) = load_linear(
            source,
            &format!("{prefix}.mlp.0"),
            TIMESTEP_FREQ_DIM,
            HIDDEN,
        )?;
        let (output, output_bias) =
            load_linear(source, &format!("{prefix}.mlp.2"), HIDDEN, HIDDEN)?;
        Ok(Self {
            input,
            input_bias,
            output,
            output_bias,
        })
    }

    /// `t` is the scalar flow time; returns `[HIDDEN]`.
    ///
    /// Upstream feeds `t.expand(batch * token_h * token_w)` through the MLP
    /// (`modeling_neo_chat.py:1608-1610`), i.e. every image token of a step sees
    /// the *same* embedding. In the T2I path all image tokens share one scalar
    /// `t`, so the per-token repetition is exactly a broadcast of this vector and
    /// the caller adds it to every token row.
    pub(crate) fn forward(&self, t: f32) -> Vec<f32> {
        let mut hidden = self.input.matmul(&timestep_embedding(t));
        add_bias(&mut hidden, &self.input_bias);
        silu_inplace(&mut hidden);
        let mut output = self.output.matmul(&hidden);
        add_bias(&mut output, &self.output_bias);
        output
    }

    /// Whether this checkpoint carried the embedder's weights.
    ///
    /// [`load`](Self::load) only returns `Ok` once both projections and both
    /// biases are present (and `config::validate` requires
    /// `fm_modules.noise_scale_embedder.mlp.2.weight`), so a constructed
    /// embedder is always present. The sampler asks the *weights* rather than
    /// `config.json`'s `add_noise_scale_embedding` flag, so the answer is derived
    /// from the loaded tensors instead of a config constant.
    pub(crate) fn is_present(&self) -> bool {
        !self.input_bias.is_empty() && !self.output_bias.is_empty()
    }
}

/// `TimestepEmbedder.timestep_embedding(t, 256)`: `cat([cos(t * f), sin(t * f)])`
/// with `f[i] = exp(-ln(10000) * i / 128)`.
fn timestep_embedding(t: f32) -> Vec<f32> {
    let half = TIMESTEP_FREQ_DIM / 2;
    let log_max_period = 10000.0f32.ln();
    let mut embedding = vec![0.0f32; TIMESTEP_FREQ_DIM];
    for i in 0..half {
        let freq = (-log_max_period * i as f32 / half as f32).exp();
        // `rope_sin_cos` is the repo's f32 sin/cos entry point (it honours the
        // scalar/oracle mode); it returns `(cos, sin)`.
        let (cos, sin) = rope_sin_cos(t * freq);
        embedding[i] = cos;
        embedding[half + i] = sin;
    }
    embedding
}

/// Pixel-space `ConvDecoder` (`fm_head`).
///
/// ```text
/// [4096, th, tw] -> PixelShuffle(2) -> conv1(1024->1024, k3, p1) -> GELU
///                -> PixelShuffle(2) -> conv2(256->192,  k3, p1)
///                -> PixelShuffle(8) -> [3, 32*th, 32*tw]
/// ```

thread_local! {
    /// Which CFG branch the next decoder stage dumps belong to.
    static TRACE_SUFFIX: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

/// Tag subsequent [`ConvDecoder`] stage dumps with the CFG branch name.
pub(crate) fn set_trace_suffix(suffix: &str) {
    TRACE_SUFFIX.with(|cell| *cell.borrow_mut() = suffix.to_string());
}

fn trace_suffix() -> String {
    TRACE_SUFFIX.with(|cell| cell.borrow().clone())
}

pub(crate) struct ConvDecoder {
    conv1: Conv2dLayer,
    conv2: Conv2dLayer,
}

impl ConvDecoder {
    pub(crate) fn load(source: &dyn TensorSource) -> Result<Self, String> {
        const PREFIX: &str = "fm_modules.fm_head";
        Ok(Self {
            conv1: Conv2dLayer::load(
                source,
                &format!("{PREFIX}.conv1"),
                CONV_DECODER_HIDDEN,
                CONV_DECODER_HIDDEN,
                3,
            )?,
            // `PixelShuffle(2)` already folded 4096 -> 1024 -> 256.
            conv2: Conv2dLayer::load(
                source,
                &format!("{PREFIX}.conv2"),
                CONV_DECODER_MID,
                CONV_DECODER_HIDDEN / 4,
                3,
            )?,
        })
    }

    /// `input` is `[HIDDEN, token_h, token_w]` (channel-major 2D map), the
    /// layout `h.view(1, token_h, token_w, 4096).permute(0, 3, 1, 2)` produces
    /// upstream. Returns `[3, token_h * 32, token_w * 32]`.
    pub(crate) fn forward(
        &self,
        input: &[f32],
        token_h: usize,
        token_w: usize,
        pool: &ComputePool,
    ) -> Result<Vec<f32>, String> {
        if token_h == 0 || token_w == 0 {
            return Err(format!("ConvDecoder: empty token grid {token_h}x{token_w}"));
        }
        let expected = HIDDEN * token_h * token_w;
        if input.len() != expected {
            return Err(format!(
                "ConvDecoder: input has {} values, expected {expected} \
                 ({HIDDEN} channels of {token_h}x{token_w})",
                input.len()
            ));
        }

        let (shuffled, shape) = pixel_shuffle(input, HIDDEN, token_h, token_w, 2)?;
        if super::sampler::tracing() {
            super::sampler::trace_dump(&format!("{}_{}", "dec_ps1", trace_suffix()), &shuffled);
        }
        let (mut hidden, shape) = self.conv1.forward(pool, &shuffled, shape)?;
        if super::sampler::tracing() {
            super::sampler::trace_dump(&format!("{}_{}", "dec_conv1", trace_suffix()), &hidden);
        }
        // `nn.GELU()` is the exact erf form; `gelu_inplace` is the tanh
        // approximation and would drift ~1e-3.
        gelu_erf_inplace(&mut hidden);
        if super::sampler::tracing() {
            super::sampler::trace_dump(&format!("{}_{}", "dec_gelu", trace_suffix()), &hidden);
        }

        let (shuffled, shape) = pixel_shuffle(&hidden, shape[0], shape[1], shape[2], 2)?;
        if super::sampler::tracing() {
            super::sampler::trace_dump(&format!("{}_{}", "dec_ps2", trace_suffix()), &shuffled);
        }
        let (hidden, shape) = self.conv2.forward(pool, &shuffled, shape)?;
        if super::sampler::tracing() {
            super::sampler::trace_dump(&format!("{}_{}", "dec_conv2", trace_suffix()), &hidden);
        }
        let (output, shape) = pixel_shuffle(&hidden, shape[0], shape[1], shape[2], 8)?;
        if super::sampler::tracing() {
            super::sampler::trace_dump(&format!("{}_{}", "dec_ps3", trace_suffix()), &output);
        }

        debug_assert_eq!(shape, [3, token_h * TOKEN_SIDE, token_w * TOKEN_SIDE]);
        Ok(output)
    }
}

/// One `Conv2d(k=3, padding=1, stride=1)` with a bias.
struct Conv2dLayer {
    /// `[out, in, 3, 3]` row-major, as `conv2d` indexes it.
    weight: Vec<f32>,
    /// `[out]`.
    bias: Vec<f32>,
    shape: [usize; 4],
}

impl Conv2dLayer {
    fn load(
        source: &dyn TensorSource,
        name: &str,
        out_channels: usize,
        in_channels: usize,
        kernel: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: load_conv_weight(
                source,
                &format!("{name}.weight"),
                out_channels,
                in_channels,
                kernel,
            )?,
            bias: load_float_values(source, &format!("{name}.bias"), &[out_channels as u64])?,
            shape: [out_channels, in_channels, kernel, kernel],
        })
    }

    fn forward(
        &self,
        pool: &ComputePool,
        input: &[f32],
        shape: [usize; 3],
    ) -> Result<(Vec<f32>, [usize; 3]), String> {
        conv2d(
            pool,
            input,
            shape,
            &self.weight,
            self.shape,
            Some(&self.bias),
            [1; 2],
            [1; 2],
            [1; 2],
            1,
        )
    }
}

/// `torch.nn.PixelShuffle(factor)`:
/// `out[c, h*r + i, w*r + j] = in[c*r*r + i*r + j, h, w]`.
///
/// The input channel of an output channel is `c*r*r + i*r + j` -- the block of
/// `r*r` channels that belongs to `c`, indexed row-major by the sub-position
/// `(i, j)` inside the `r x r` block that the shuffle expands into the spatial
/// grid. Getting this order backwards (e.g. `(i, j)` swapped, or a plain
/// channel-major split) silently permutes the image, so it is pinned by tests.
fn pixel_shuffle(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    factor: usize,
) -> Result<(Vec<f32>, [usize; 3]), String> {
    if factor == 0 || channels % (factor * factor) != 0 {
        return Err(format!(
            "PixelShuffle({factor}): {channels} channels is not a multiple of {}",
            factor * factor
        ));
    }
    if input.len() != channels * height * width {
        return Err(format!(
            "PixelShuffle({factor}): {} values, expected {}",
            input.len(),
            channels * height * width
        ));
    }
    let out_channels = channels / (factor * factor);
    let out_height = height * factor;
    let out_width = width * factor;
    let plane = height * width;
    let out_plane = out_height * out_width;
    let mut output = vec![0.0f32; out_channels * out_plane];
    for channel in 0..out_channels {
        for i in 0..factor {
            for j in 0..factor {
                let source_channel = channel * factor * factor + i * factor + j;
                let source = &input[source_channel * plane..(source_channel + 1) * plane];
                for row in 0..height {
                    let target_row = (row * factor + i) * out_width;
                    let source_row = row * width;
                    for column in 0..width {
                        output[channel * out_plane + target_row + column * factor + j] =
                            source[source_row + column];
                    }
                }
            }
        }
    }
    Ok((output, [out_channels, out_height, out_width]))
}

/// Load `nn.Linear` weights plus bias as a `[n_out, n_in]` kernel.
fn load_linear(
    source: &dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<(Weight<'static>, Vec<f32>), String> {
    // GGUF stores a `nn.Linear` weight of PyTorch shape `[out, in]` with header
    // dims `[in, out]` and the payload contiguous in `[out][in]` order, which is
    // what `Weight` (and `conv2d`) index: `w[o * n_in + i]`.
    let data = load_float_values(
        source,
        &format!("{name}.weight"),
        &[n_in as u64, n_out as u64],
    )?;
    let bias = load_float_values(source, &format!("{name}.bias"), &[n_out as u64])?;
    let weight = Weight::from_quantized(QuantizedTensor::F32 { data, n_in, n_out });
    Ok((weight, bias))
}

fn add_bias(values: &mut [f32], bias: &[f32]) {
    debug_assert_eq!(values.len(), bias.len());
    for (value, &bias) in values.iter_mut().zip(bias) {
        *value += bias;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{GGMLType, MetaValue, TensorInfo};
    use std::collections::HashMap;

    /// In-memory `TensorSource` holding F32 tensors (see
    /// `src/models/qwen35/trunk/tests.rs` for the same double).
    struct TestSource {
        infos: Vec<TensorInfo>,
        data: HashMap<String, Vec<u8>>,
    }

    impl TestSource {
        fn new() -> Self {
            Self {
                infos: Vec::new(),
                data: HashMap::new(),
            }
        }

        fn add(&mut self, name: &str, dims: &[u64], values: &[f32]) {
            self.infos.push(TensorInfo {
                name: name.to_string(),
                dims: dims.to_vec(),
                ggml_type: GGMLType::F32,
                offset: 0,
            });
            self.data.insert(
                name.to_string(),
                values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            );
        }
    }

    impl TensorSource for TestSource {
        fn metadata(&self, _: &str) -> Option<&MetaValue> {
            None
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.infos.iter().find(|info| info.name == name)
        }

        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            self.data.get(name).map(Vec::as_slice)
        }
    }

    /// Deterministic pseudo-random values in `[-scale/2, scale/2)`.
    fn pattern(len: usize, scale: f32, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
                (unit - 0.5) * scale
            })
            .collect()
    }

    /// Hand-computed `PixelShuffle(2)` on `[4, 1, 2]`.
    #[test]
    fn pixel_shuffle_two_matches_hand_computed_reference() {
        // channel 0: [1, 2]   channel 1: [3, 4]   channel 2: [5, 6]   channel 3: [7, 8]
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (output, shape) = pixel_shuffle(&input, 4, 1, 2, 2).expect("shuffle");
        assert_eq!(shape, [1, 2, 4]);
        // PyTorch: out[c, h, w] = in[c*r*r + (h%r)*r + (w%r), h/r, w/r], so with
        // r = 2 the four source channels fill a 2x2 block row-major:
        //   out[0][0] = [in0(w0), in1(w0), in0(w1), in1(w1)] = [1, 3, 2, 4]
        //   out[0][1] = [in2(w0), in3(w0), in2(w1), in3(w1)] = [5, 7, 6, 8]
        assert_eq!(output, vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]);
    }

    /// Hand-computed `PixelShuffle(8)` on `[192, 1, 1]` (the `fm_head` tail:
    /// `192 / 64 = 3` output channels at 8x).
    #[test]
    fn pixel_shuffle_eight_matches_hand_computed_reference() {
        let channels = 3 * 8 * 8;
        let input: Vec<f32> = (0..channels).map(|value| value as f32).collect();
        let (output, shape) = pixel_shuffle(&input, channels, 1, 1, 8).expect("shuffle");
        assert_eq!(shape, [3, 8, 8]);
        for channel in 0..3 {
            for i in 0..8 {
                for j in 0..8 {
                    let expected = (channel * 64 + i * 8 + j) as f32;
                    assert_eq!(
                        output[channel * 64 + i * 8 + j],
                        expected,
                        "channel {channel} ({i}, {j})"
                    );
                }
            }
        }
        // A mis-ordered `(i, j)` would put channel 1's first row here.
        assert_eq!(output[0], 0.0);
        assert_eq!(output[1], 1.0);
        assert_eq!(output[8], 8.0);
    }

    #[test]
    fn pixel_shuffle_rejects_indivisible_channels() {
        assert!(pixel_shuffle(&[0.0; 4], 2, 2, 1, 2).is_err());
        assert!(pixel_shuffle(&[0.0; 3], 4, 1, 1, 2).is_err());
    }

    /// Required check: the GGUF convolution layout feeds `conv2d` exactly like a
    /// hand-written `[out][in][kh][kw]` 3x3 convolution with padding 1.
    #[test]
    fn conv_weight_layout_matches_direct_convolution() {
        let (out_channels, in_channels, kernel) = (2usize, 3usize, 3usize);
        let (height, width) = (4usize, 5usize);
        // Written the way gguf-py writes it: contiguous PyTorch `[out][in][kh][kw]`
        // payload with the header dims reversed to `[kw, kh, in, out]`.
        let torch_order = pattern(out_channels * in_channels * kernel * kernel, 1.0, 7);
        let mut source = TestSource::new();
        source.add(
            "conv.weight",
            &[
                kernel as u64,
                kernel as u64,
                in_channels as u64,
                out_channels as u64,
            ],
            &torch_order,
        );
        let loaded = load_conv_weight(&source, "conv.weight", out_channels, in_channels, kernel)
            .expect("load");
        assert_eq!(loaded.len(), torch_order.len());

        let input = pattern(in_channels * height * width, 1.0, 9);
        let pool = ComputePool::new(1);
        let (actual, shape) = conv2d(
            &pool,
            &input,
            [in_channels, height, width],
            &loaded,
            [out_channels, in_channels, kernel, kernel],
            None,
            [1; 2],
            [1; 2],
            [1; 2],
            1,
        )
        .expect("conv2d");
        assert_eq!(shape, [out_channels, height, width]);

        for out_channel in 0..out_channels {
            for y in 0..height {
                for x in 0..width {
                    let mut expected = 0.0f32;
                    for in_channel in 0..in_channels {
                        for ky in 0..kernel {
                            for kx in 0..kernel {
                                let sy = y as isize + ky as isize - 1;
                                let sx = x as isize + kx as isize - 1;
                                if sy < 0 || sx < 0 || sy >= height as isize || sx >= width as isize
                                {
                                    continue;
                                }
                                let weight = torch_order[((out_channel * in_channels
                                    + in_channel)
                                    * kernel
                                    + ky)
                                    * kernel
                                    + kx];
                                expected += weight
                                    * input
                                        [(in_channel * height + sy as usize) * width + sx as usize];
                            }
                        }
                    }
                    let index = (out_channel * height + y) * width + x;
                    assert!(
                        (actual[index] - expected).abs() < 1e-5,
                        "({out_channel}, {y}, {x}): {} vs {expected}",
                        actual[index]
                    );
                }
            }
        }
    }

    #[test]
    fn timestep_embedding_matches_hand_computed_sinusoid() {
        let embedding = timestep_embedding(0.5);
        assert_eq!(embedding.len(), TIMESTEP_FREQ_DIM);
        for i in [0usize, 1, 2, 37, 127] {
            let freq = (-(10000.0f32.ln()) * i as f32 / 128.0).exp();
            let angle = 0.5 * freq;
            assert!(
                (embedding[i] - angle.cos()).abs() < 1e-6,
                "cos[{i}]: {} vs {}",
                embedding[i],
                angle.cos()
            );
            assert!(
                (embedding[128 + i] - angle.sin()).abs() < 1e-6,
                "sin[{i}]: {} vs {}",
                embedding[128 + i],
                angle.sin()
            );
        }
        // `freqs[0] == 1`, so the first cosine is `cos(t)`.
        assert!((embedding[0] - 0.5f32.cos()).abs() < 1e-6);
        assert!(embedding.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn timestep_embedder_forward_is_finite_and_sized() {
        let embedder = {
            let mut source = TestSource::new();
            source.add(
                "fm_modules.timestep_embedder.mlp.0.weight",
                &[TIMESTEP_FREQ_DIM as u64, HIDDEN as u64],
                &pattern(TIMESTEP_FREQ_DIM * HIDDEN, 0.05, 3),
            );
            source.add(
                "fm_modules.timestep_embedder.mlp.0.bias",
                &[HIDDEN as u64],
                &pattern(HIDDEN, 0.01, 4),
            );
            source.add(
                "fm_modules.timestep_embedder.mlp.2.weight",
                &[HIDDEN as u64, HIDDEN as u64],
                &pattern(HIDDEN * HIDDEN, 0.01, 5),
            );
            source.add(
                "fm_modules.timestep_embedder.mlp.2.bias",
                &[HIDDEN as u64],
                &pattern(HIDDEN, 0.01, 6),
            );
            TimestepEmbedder::load(&source, "fm_modules.timestep_embedder").expect("load")
        };

        let output = embedder.forward(0.5);
        assert_eq!(output.len(), HIDDEN);
        assert!(output.iter().all(|value| value.is_finite()));
        // The bias is the only non-random term; a zero-output MLP would mean
        // the projection never ran.
        assert!(output.iter().any(|value| *value != 0.0));

        // The same `t` must produce the same vector (pure function of `t`).
        assert_eq!(output, embedder.forward(0.5));
        assert_ne!(output, embedder.forward(0.25));
    }

    #[test]
    fn conv_decoder_forward_one_token_is_3x32x32() {
        let decoder = {
            let mut source = TestSource::new();
            source.add(
                "fm_modules.fm_head.conv1.weight",
                &[3, 3, CONV_DECODER_HIDDEN as u64, CONV_DECODER_HIDDEN as u64],
                &pattern(CONV_DECODER_HIDDEN * CONV_DECODER_HIDDEN * 9, 0.02, 21),
            );
            source.add(
                "fm_modules.fm_head.conv1.bias",
                &[CONV_DECODER_HIDDEN as u64],
                &pattern(CONV_DECODER_HIDDEN, 0.01, 22),
            );
            source.add(
                "fm_modules.fm_head.conv2.weight",
                &[
                    3,
                    3,
                    (CONV_DECODER_HIDDEN / 4) as u64,
                    CONV_DECODER_MID as u64,
                ],
                &pattern(CONV_DECODER_MID * (CONV_DECODER_HIDDEN / 4) * 9, 0.05, 23),
            );
            source.add(
                "fm_modules.fm_head.conv2.bias",
                &[CONV_DECODER_MID as u64],
                &pattern(CONV_DECODER_MID, 0.01, 24),
            );
            ConvDecoder::load(&source).expect("load")
        };

        let input = pattern(HIDDEN, 1.0, 25);
        let pool = ComputePool::new(1);
        let output = decoder.forward(&input, 1, 1, &pool).expect("forward");
        assert_eq!(output.len(), 3 * 32 * 32);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.iter().any(|value| *value != 0.0));

        // The pixel grid is square, so `token_h != token_w` must scale each axis
        // independently.
        let wide = pattern(HIDDEN * 2, 1.0, 26);
        let output = decoder.forward(&wide, 1, 2, &pool).expect("forward 1x2");
        assert_eq!(output.len(), 3 * 32 * 64);
    }

    #[test]
    fn conv_decoder_rejects_bad_shapes() {
        let pool = ComputePool::new(1);
        let decoder = ConvDecoder {
            conv1: Conv2dLayer {
                weight: Vec::new(),
                bias: Vec::new(),
                shape: [0, 0, 3, 3],
            },
            conv2: Conv2dLayer {
                weight: Vec::new(),
                bias: Vec::new(),
                shape: [0, 0, 3, 3],
            },
        };
        assert!(decoder.forward(&[], 0, 0, &pool).is_err());
        assert!(decoder.forward(&[0.0; 4], 2, 2, &pool).is_err());
    }
}
