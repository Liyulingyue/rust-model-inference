//! Qwen-Image-2.1's dedicated 64-channel, 16x spatial RGBA VAE.

use super::pipeline::trace;
use crate::core::tensor::{load_f32_tensor, TensorSource};
use crate::format::safetensors::SafetensorSource;
use crate::ops::{dot_f16_f16_bytes_ggml, f32_to_f16};
use rayon::prelude::*;
use std::path::Path;

struct Map {
    data: Vec<f32>,
    c: usize,
    h: usize,
    w: usize,
}

pub(crate) struct QwenImage21Vae {
    source: SafetensorSource,
    pool: rayon::ThreadPool,
    mean: [f32; 64],
    std: [f32; 64],
}

impl QwenImage21Vae {
    pub(crate) fn open(path: &Path, threads: usize) -> Result<Self, String> {
        let config_path = path
            .parent()
            .ok_or("VAE path needs a parent")?
            .join("config.json");
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&config_path)
                .map_err(|e| format!("Read {}: {e}", config_path.display()))?,
        )
        .map_err(|e| e.to_string())?;
        if config["_class_name"] != "AutoencoderKLQwenImage21"
            || config["z_dim"] != 64
            || config["in_channels"] != 4
            || config["base_dim"] != 96
            || config["decoder_base_dim"] != 144
            || config["dim_mult"] != serde_json::json!([1, 2, 4, 8, 8])
            || config["num_res_blocks"] != 2
            || config["scale_factor_spatial"] != 16
        {
            return Err("Expected Qwen-Image-2.1's dedicated RGBA VAE and config.json".into());
        }
        let stats = |key: &str| -> Result<[f32; 64], String> {
            let values: Vec<f32> =
                serde_json::from_value(config[key].clone()).map_err(|e| e.to_string())?;
            if values
                .iter()
                .any(|v| !v.is_finite() || (key == "latents_std" && *v <= 0.0))
            {
                return Err(format!("Invalid VAE {key}"));
            }
            values
                .try_into()
                .map_err(|_| format!("VAE {key} must contain 64 values"))
        };
        let model = Self {
            source: SafetensorSource::open(&[path])?,
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(threads.max(1))
                .build()
                .map_err(|e| e.to_string())?,
            mean: stats("latents_mean")?,
            std: stats("latents_std")?,
        };
        for (name, dims) in [
            ("post_quant_conv.weight", vec![1, 1, 64, 64]),
            ("decoder.conv_in.weight", vec![3, 3, 64, 1152]),
            ("decoder.conv_out.weight", vec![3, 3, 144, 4]),
        ] {
            if model
                .source
                .tensor_info(name)
                .is_none_or(|i| i.dims != dims)
            {
                return Err(format!("Invalid Qwen-Image-2.1 VAE tensor {name}"));
            }
        }
        Ok(model)
    }

    fn values(&self, name: &str) -> Result<Vec<f32>, String> {
        let info = self
            .source
            .tensor_info(name)
            .ok_or_else(|| format!("Missing VAE {name}"))?;
        load_f32_tensor(&self.source, name, &info.dims)
    }

    fn report(&self, name: &str, x: &Map) -> Result<(), String> {
        trace(&format!("qwen.vae.{name}"), &[x.w, x.h, 1, x.c], &x.data)
    }

    fn conv(
        &self,
        name: &str,
        x: &Map,
        stride: usize,
        padding: usize,
        pad_end: bool,
    ) -> Result<Map, String> {
        let key = format!("{name}.weight");
        let info = self
            .source
            .tensor_info(&key)
            .ok_or_else(|| format!("Missing VAE {key}"))?;
        let dims = &info.dims;
        if dims.len() != 4
            || dims[2] != x.c as u64
            || !matches!(dims[0], 1 | 3)
            || dims[1] != dims[0]
        {
            return Err(format!("Invalid VAE convolution {name}: {dims:?}"));
        }
        let k = dims[0] as usize;
        let c = dims[3] as usize;
        let h = (x.h + 2 * padding + usize::from(pad_end))
            .checked_sub(k)
            .ok_or("VAE kernel exceeds image")?
            / stride
            + 1;
        let w = (x.w + 2 * padding + usize::from(pad_end))
            .checked_sub(k)
            .ok_or("VAE kernel exceeds image")?
            / stride
            + 1;
        let patch_len = x.c.checked_mul(k * k).ok_or("VAE patch overflow")?;
        let spatial = h.checked_mul(w).ok_or("VAE size overflow")?;
        let mut patches = vec![
            0u16;
            spatial
                .checked_mul(patch_len)
                .ok_or("VAE im2col overflow")?
        ];
        let weights: Vec<u8> = self
            .values(&key)?
            .into_iter()
            .flat_map(|v| f32_to_f16(v).to_le_bytes())
            .collect();
        let bias = self.values(&format!("{name}.bias"))?;
        if bias.len() != c {
            return Err(format!("Invalid VAE bias {name}"));
        }
        self.pool.install(|| {
            patches
                .par_chunks_mut(patch_len)
                .enumerate()
                .for_each(|(pixel, patch)| {
                    for ic in 0..x.c {
                        for ky in 0..k {
                            for kx in 0..k {
                                let iy = (pixel / w * stride + ky) as isize - padding as isize;
                                let ix = (pixel % w * stride + kx) as isize - padding as isize;
                                if iy >= 0 && ix >= 0 && iy < x.h as isize && ix < x.w as isize {
                                    patch[(ic * k + ky) * k + kx] = f32_to_f16(
                                        x.data[(ic * x.h + iy as usize) * x.w + ix as usize]
                                            * (1.0 / 128.0),
                                    );
                                }
                            }
                        }
                    }
                })
        });
        let mut y = Map {
            data: vec![0.0; c.checked_mul(spatial).ok_or("VAE output overflow")?],
            c,
            h,
            w,
        };
        self.pool.install(|| {
            y.data
                .par_chunks_mut(spatial)
                .enumerate()
                .for_each(|(oc, out)| {
                    let row = &weights[oc * patch_len * 2..(oc + 1) * patch_len * 2];
                    for (pixel, value) in out.iter_mut().enumerate() {
                        *value = dot_f16_f16_bytes_ggml(
                            &patches[pixel * patch_len..(pixel + 1) * patch_len],
                            row,
                            patch_len,
                        ) * 128.0
                            + bias[oc];
                    }
                })
        });
        self.report(name, &y)?;
        Ok(y)
    }

    fn norm(&self, name: &str, x: &Map) -> Result<Map, String> {
        let gamma = self.values(&format!("{name}.gamma"))?;
        if gamma.len() != x.c {
            return Err(format!("Invalid VAE norm {name}"));
        }
        let spatial = x.h * x.w;
        let mut rows = vec![0.0; x.data.len()];
        self.pool.install(|| {
            rows.par_chunks_mut(x.c).enumerate().for_each(|(p, row)| {
                for (c, v) in row.iter_mut().enumerate() {
                    *v = x.data[c * spatial + p];
                }
                crate::ops::rms_norm_inplace(row, &gamma, 1e-12);
            })
        });
        let y = Map {
            data: (0..rows.len())
                .map(|i| rows[(i % spatial) * x.c + i / spatial])
                .collect(),
            c: x.c,
            h: x.h,
            w: x.w,
        };
        self.report(name, &y)?;
        Ok(y)
    }

    fn activate(&self, mut x: Map) -> Map {
        crate::ops::silu_approx_inplace(&mut x.data);
        x
    }

    fn residual(&self, name: &str, x: Map) -> Result<Map, String> {
        let mut y = self.activate(self.norm(&format!("{name}.norm1"), &x)?);
        y = self.conv(&format!("{name}.conv1"), &y, 1, 1, false)?;
        y = self.activate(self.norm(&format!("{name}.norm2"), &y)?);
        y = self.conv(&format!("{name}.conv2"), &y, 1, 1, false)?;
        let shortcut = if x.c != y.c {
            self.conv(&format!("{name}.conv_shortcut"), &x, 1, 0, false)?
        } else {
            x
        };
        for (v, skip) in y.data.iter_mut().zip(shortcut.data) {
            *v += skip;
        }
        self.report(name, &y)?;
        Ok(y)
    }

    fn attention(&self, name: &str, x: Map) -> Result<Map, String> {
        let h = self.norm(&format!("{name}.norm"), &x)?;
        let qkv = self.conv(&format!("{name}.to_qkv"), &h, 1, 0, false)?;
        let n = x.h * x.w;
        let mut q = vec![0.0; x.data.len()];
        let mut k = q.clone();
        let v = &qkv.data[2 * x.c * n..];
        for p in 0..n {
            for c in 0..x.c {
                q[p * x.c + c] = qkv.data[c * n + p];
                k[p * x.c + c] = qkv.data[(x.c + c) * n + p];
            }
        }
        let mut out = vec![0.0; x.data.len()];
        let scale = 1.0f32 / (x.c as f32).sqrt();
        self.pool.install(|| {
            out.par_chunks_mut(x.c).enumerate().for_each(|(p, row)| {
                let mut probs: Vec<_> = (0..n)
                    .map(|j| crate::ops::dot_f32(&q[p * x.c..], &k[j * x.c..], x.c) * scale)
                    .collect();
                crate::ops::softmax_approx_inplace(&mut probs);
                for c in 0..x.c {
                    row[c] = crate::ops::dot_f32(&v[c * n..], &probs, n);
                }
            })
        });
        let out = Map {
            data: (0..out.len()).map(|i| out[(i % n) * x.c + i / n]).collect(),
            c: x.c,
            h: x.h,
            w: x.w,
        };
        self.report("attention.values", &out)?;
        let mut y = self.conv(&format!("{name}.proj"), &out, 1, 0, false)?;
        for (v, skip) in y.data.iter_mut().zip(x.data) {
            *v += skip;
        }
        self.report(name, &y)?;
        Ok(y)
    }

    pub(crate) fn encode(
        &self,
        rgba: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, String> {
        let n = width
            .checked_mul(height)
            .filter(|&n| n > 0)
            .ok_or("VAE input size overflow")?;
        if width % 16 != 0
            || height % 16 != 0
            || n.checked_mul(4) != Some(rgba.len())
            || rgba.iter().any(|v| !v.is_finite())
        {
            return Err(
                "Qwen VAE input must be finite planar RGBA, with dimensions divisible by 16".into(),
            );
        }
        let mut x = Map {
            data: rgba.to_vec(),
            c: 4,
            h: height,
            w: width,
        };
        self.report("input", &x)?;
        x = self.conv("encoder.conv_in", &x, 1, 1, false)?;
        for stage in 0..5 {
            let prefix = format!("encoder.down_blocks.{stage}");
            let before = x;
            x = Map {
                data: before.data.clone(),
                c: before.c,
                h: before.h,
                w: before.w,
            };
            for block in 0..2 {
                x = self.residual(&format!("{prefix}.resnets.{block}"), x)?;
            }
            if stage < 4 {
                x = self.conv(&format!("{prefix}.downsampler.resample.1"), &x, 2, 0, true)?;
            }
            let skip = average_down(
                &before,
                x.c,
                if stage > 0 && stage < 4 { 2 } else { 1 },
                if stage < 4 { 2 } else { 1 },
            )?;
            for (v, s) in x.data.iter_mut().zip(skip.data) {
                *v += s;
            }
            self.report(&prefix, &x)?;
        }
        x = self.residual("encoder.mid_block.resnets.0", x)?;
        x = self.attention("encoder.mid_block.attentions.0", x)?;
        x = self.residual("encoder.mid_block.resnets.1", x)?;
        x = self.activate(self.norm("encoder.norm_out", &x)?);
        x = self.conv("encoder.conv_out", &x, 1, 1, false)?;
        x = self.conv("quant_conv", &x, 1, 0, false)?;
        if x.c != 128 {
            return Err("Invalid Qwen VAE encoder output".into());
        }
        x.c = 64;
        x.data.truncate(64 * x.h * x.w);
        self.report("encoded_mean", &x)?;
        let spatial = x.h * x.w;
        for (i, v) in x.data.iter_mut().enumerate() {
            *v = (*v - self.mean[i / spatial]) / self.std[i / spatial];
        }
        if x.data.iter().any(|v| !v.is_finite()) {
            return Err("Non-finite Qwen reference latent".into());
        }
        self.report("encoded", &x)?;
        Ok(x.data)
    }

    pub(crate) fn decode(
        &self,
        latent: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, String> {
        let n = width
            .checked_mul(height)
            .filter(|&n| n > 0)
            .ok_or("VAE size overflow")?;
        if n.checked_mul(64) != Some(latent.len()) || latent.iter().any(|v| !v.is_finite()) {
            return Err("Invalid Qwen VAE latent".into());
        }
        let x = Map {
            data: latent
                .iter()
                .enumerate()
                .map(|(i, &v)| v * self.std[i / n] + self.mean[i / n])
                .collect(),
            c: 64,
            h: height,
            w: width,
        };
        self.report("denormalized", &x)?;
        let mut x = self.conv("post_quant_conv", &x, 1, 0, false)?;
        x = self.conv("decoder.conv_in", &x, 1, 1, false)?;
        x = self.residual("decoder.mid_block.resnets.0", x)?;
        x = self.attention("decoder.mid_block.attentions.0", x)?;
        x = self.residual("decoder.mid_block.resnets.1", x)?;
        for stage in 0..5 {
            let prefix = format!("decoder.up_blocks.{stage}");
            let before = x;
            // Wan 2.2 residual upsampling also groups temporal shortcut channels
            // for this single-frame image VAE; time_conv itself is skipped.
            x = Map {
                data: before.data.clone(),
                c: before.c,
                h: before.h,
                w: before.w,
            };
            for block in 0..3 {
                x = self.residual(&format!("{prefix}.resnets.{block}"), x)?;
            }
            if stage < 4 {
                x = self.conv(
                    &format!("{prefix}.upsampler.resample.1"),
                    &nearest(&x),
                    1,
                    1,
                    false,
                )?;
                let shortcut = duplicate_up(&before, x.c, if stage < 3 { 2 } else { 1 })?;
                for (v, skip) in x.data.iter_mut().zip(shortcut.data) {
                    *v += skip;
                }
            }
            self.report(&prefix, &x)?;
        }
        x = self.activate(self.norm("decoder.norm_out", &x)?);
        x = self.conv("decoder.conv_out", &x, 1, 1, false)?;
        if x.data.iter().any(|v| !v.is_finite()) {
            return Err("Qwen VAE produced NaN or infinity".into());
        }
        self.report("output", &x)?;
        Ok(x.data)
    }
}

fn nearest(x: &Map) -> Map {
    let h = x.h * 2;
    let w = x.w * 2;
    Map {
        data: (0..x.c * h * w)
            .map(|i| x.data[(i / (h * w) * x.h + i / w % h / 2) * x.w + i % w / 2])
            .collect(),
        c: x.c,
        h,
        w,
    }
}

fn duplicate_up(x: &Map, output_channels: usize, temporal: usize) -> Result<Map, String> {
    let factor = temporal * 4;
    if output_channels * factor % x.c != 0 {
        return Err("Invalid Qwen VAE shortcut channel grouping".into());
    }
    let repeats = output_channels * factor / x.c;
    let h = x.h * 2;
    let w = x.w * 2;
    let n = x.h * x.w;
    let data = (0..output_channels * h * w)
        .map(|i| {
            let c = i / (h * w);
            let y = i / w % h;
            let z = i % w;
            let grouped = (c * temporal + temporal - 1) * 4 + y % 2 * 2 + z % 2;
            x.data[grouped / repeats * n + y / 2 * x.w + z / 2]
        })
        .collect();
    Ok(Map {
        data,
        c: output_channels,
        h,
        w,
    })
}

fn average_down(x: &Map, channels: usize, temporal: usize, spatial: usize) -> Result<Map, String> {
    let factor = temporal * spatial * spatial;
    if x.c * factor % channels != 0 || x.h % spatial != 0 || x.w % spatial != 0 {
        return Err("Invalid Qwen VAE downsample grouping".into());
    }
    let group = x.c * factor / channels;
    let (h, w) = (x.h / spatial, x.w / spatial);
    let mut data = vec![0.0; channels * h * w];
    for c in 0..channels {
        for y in 0..h {
            for z in 0..w {
                let mut sum = 0.0f64;
                for g in 0..group {
                    let index = c * group + g;
                    let t = (index / spatial / spatial) % temporal;
                    if t + 1 == temporal {
                        let src_c = index / factor;
                        let dy = index / spatial % spatial;
                        let dx = index % spatial;
                        sum += f64::from(
                            x.data[(src_c * x.h + y * spatial + dy) * x.w + z * spatial + dx],
                        );
                    }
                }
                data[(c * h + y) * w + z] = (sum as f32) / group as f32;
            }
        }
    }
    Ok(Map {
        data,
        c: channels,
        h,
        w,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires QI21_VAE and QI21_LATENT, raw trace through RMI_PARITY_TRACE"]
    fn real_vae_component() {
        let model =
            QwenImage21Vae::open(Path::new(&std::env::var("QI21_VAE").unwrap()), 8).unwrap();
        let input =
            crate::app::read_f32_file(Path::new(&std::env::var("QI21_LATENT").unwrap())).unwrap();
        let output = model.decode(&input, 4, 2).unwrap();
        assert_eq!(output.len(), 4 * 64 * 32);
        assert!(output.iter().all(|v| v.is_finite()));
    }
    #[test]
    fn single_frame_shortcut_keeps_temporal_channel_grouping() {
        let x = Map {
            data: vec![1.0, 2.0, 3.0, 4.0],
            c: 4,
            h: 1,
            w: 1,
        };
        assert_eq!(
            duplicate_up(&x, 2, 2).unwrap().data,
            [2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0]
        );
        assert_eq!(
            duplicate_up(&x, 2, 1).unwrap().data,
            [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
        );
    }
}
