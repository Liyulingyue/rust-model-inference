use crate::core::{
    tensor::{GGMLType, TensorSource},
    thread_pool::ComputePool,
};
use crate::models::diffusion::dreamx::kernels::{self as k, Linear};
use crate::ops::{gelu_erf_inplace, softmax_inplace};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct Feature {
    pub values: Vec<f32>,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl Feature {
    pub fn new(
        values: Vec<f32>,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<Self, String> {
        if channels == 0
            || height == 0
            || width == 0
            || values.len() != k::checked_len("WeVisOne feature", &[channels, height, width])?
            || values.iter().any(|v| !v.is_finite())
        {
            return Err("invalid WeVisOne feature values or shape".into());
        }
        Ok(Self {
            values,
            channels,
            height,
            width,
        })
    }

    pub fn from_premerge(
        tokens: &[f32],
        channels: usize,
        height: usize,
        width: usize,
        merge: usize,
    ) -> Result<Self, String> {
        if merge == 0
            || height % merge != 0
            || width % merge != 0
            || tokens.len() != k::checked_len("premerge tokens", &[channels, height, width])?
        {
            return Err("invalid WeVisOne premerge grid".into());
        }
        let mut values = vec![0.0; tokens.len()];
        for y in 0..height {
            for x in 0..width {
                let token = (((y / merge) * (width / merge) + x / merge) * merge + y % merge)
                    * merge
                    + x % merge;
                for c in 0..channels {
                    values[(c * height + y) * width + x] = tokens[token * channels + c];
                }
            }
        }
        Self::new(values, channels, height, width)
    }

    fn rows(&self) -> Vec<f32> {
        let n = self.height * self.width;
        let mut rows = vec![0.0; self.values.len()];
        for t in 0..n {
            for c in 0..self.channels {
                rows[t * self.channels + c] = self.values[c * n + t];
            }
        }
        rows
    }

    fn from_rows(rows: &[f32], c: usize, h: usize, w: usize) -> Result<Self, String> {
        if rows.len() != k::checked_len("feature rows", &[c, h, w])? {
            return Err("invalid feature rows".into());
        }
        let mut values = vec![0.0; rows.len()];
        for t in 0..h * w {
            for ch in 0..c {
                values[ch * h * w + t] = rows[t * c + ch];
            }
        }
        Self::new(values, c, h, w)
    }

    pub fn resize(&self, h: usize, w: usize) -> Result<Self, String> {
        if h == 0
            || w == 0
            || self.height == 0
            || self.width == 0
            || self.values.len()
                != k::checked_len("resize input", &[self.channels, self.height, self.width])?
        {
            return Err("invalid feature resize shape".into());
        }
        let mut out = vec![0.0; k::checked_len("resize output", &[self.channels, h, w])?];
        for c in 0..self.channels {
            for y in 0..h {
                for x in 0..w {
                    let sy = ((y as f32 + 0.5) * self.height as f32 / h as f32 - 0.5).max(0.0);
                    let sx = ((x as f32 + 0.5) * self.width as f32 / w as f32 - 0.5).max(0.0);
                    let y0 = (sy as usize).min(self.height - 1);
                    let x0 = (sx as usize).min(self.width - 1);
                    let y1 = (y0 + 1).min(self.height - 1);
                    let x1 = (x0 + 1).min(self.width - 1);
                    let fy = sy - y0 as f32;
                    let fx = sx - x0 as f32;
                    let a = self.values[(c * self.height + y0) * self.width + x0] * (1.0 - fx)
                        + self.values[(c * self.height + y0) * self.width + x1] * fx;
                    let b = self.values[(c * self.height + y1) * self.width + x0] * (1.0 - fx)
                        + self.values[(c * self.height + y1) * self.width + x1] * fx;
                    out[(c * h + y) * w + x] = a * (1.0 - fy) + b * fy;
                }
            }
        }
        Self::new(out, self.channels, h, w)
    }

    fn pool(&self, h: usize, w: usize) -> Result<Self, String> {
        let mut out = vec![0.0; k::checked_len("pool output", &[self.channels, h, w])?];
        for c in 0..self.channels {
            for y in 0..h {
                for x in 0..w {
                    let ya = y * self.height / h;
                    let yb = ((y + 1) * self.height).div_ceil(h);
                    let xa = x * self.width / w;
                    let xb = ((x + 1) * self.width).div_ceil(w);
                    let mut sum = 0.0;
                    for iy in ya..yb {
                        for ix in xa..xb {
                            sum += self.values[(c * self.height + iy) * self.width + ix];
                        }
                    }
                    out[(c * h + y) * w + x] = sum / ((yb - ya) * (xb - xa)) as f32;
                }
            }
        }
        Self::new(out, self.channels, h, w)
    }

    fn gelu(mut self) -> Self {
        gelu_erf_inplace(&mut self.values);
        self
    }
    pub fn sigmoid(mut self) -> Self {
        self.values
            .iter_mut()
            .for_each(|v| *v = 1.0 / (1.0 + (-*v).exp()));
        self
    }
    fn add(mut self, other: &Self) -> Result<Self, String> {
        if (self.channels, self.height, self.width) != (other.channels, other.height, other.width) {
            return Err("WeVisOne residual shape mismatch".into());
        }
        crate::ops::vec_add_into(&other.values, &mut self.values);
        Ok(self)
    }
}

pub struct Decoder<'a> {
    source: &'a dyn TensorSource,
    pool: &'a ComputePool,
    shapes: BTreeMap<String, Vec<usize>>,
}

impl<'a> Decoder<'a> {
    pub fn new(source: &'a dyn TensorSource, pool: &'a ComputePool) -> Result<Self, String> {
        if source
            .metadata("general.architecture")
            .and_then(|v| v.to_string_val())
            != Some("wevisone_vl")
        {
            return Err("expected a wevisone_vl pixel decoder GGUF".into());
        }
        let config: serde_json::Value = serde_json::from_str(
            source
                .metadata("wevisone_vl.config")
                .and_then(|v| v.to_string_val())
                .ok_or("missing WeVisOne decoder config")?,
        )
        .map_err(|e| e.to_string())?;
        let expected: serde_json::Value =
            serde_json::from_str(include_str!("decoder-config.json")).map_err(|e| e.to_string())?;
        if config != expected {
            return Err("unsupported WeVisOne decoder config".into());
        }
        let shapes: BTreeMap<String, Vec<usize>> =
            serde_json::from_str(include_str!("decoder-tensors.json"))
                .map_err(|e| e.to_string())?;
        let decoder = Self {
            source,
            pool,
            shapes,
        };
        for (name, dims) in &decoder.shapes {
            let info = source
                .tensor_info(name)
                .ok_or_else(|| format!("missing WeVisOne decoder tensor: {name}"))?;
            if info.ggml_type != GGMLType::F32
                || info.dims != dims.iter().rev().map(|&d| d as u64).collect::<Vec<_>>()
            {
                return Err(format!("invalid WeVisOne decoder shape/type: {name}"));
            }
            let data = decoder.tensor(name)?;
            if data.len() != k::checked_len(name, dims)? || data.iter().any(|v| !v.is_finite()) {
                return Err(format!("invalid WeVisOne decoder payload: {name}"));
            }
        }
        Ok(decoder)
    }

    fn tensor(&self, name: &str) -> Result<&[f32], String> {
        bytemuck::try_cast_slice(
            self.source
                .tensor_slice(name)
                .ok_or_else(|| format!("missing tensor: {name}"))?,
        )
        .map_err(|e| format!("invalid F32 tensor {name}: {e}"))
    }

    fn linear(&self, name: &str, input: &[f32], rows: usize) -> Result<Vec<f32>, String> {
        let shape = self
            .shapes
            .get(&format!("{name}.weight"))
            .ok_or_else(|| format!("missing linear: {name}"))?;
        Linear::from_source(
            self.source,
            &format!("{name}.weight"),
            Some(&format!("{name}.bias")),
            shape[1],
            shape[0],
        )?
        .forward(self.pool, input, rows)
    }

    fn norm(&self, name: &str, input: &[f32], rows: usize) -> Result<Vec<f32>, String> {
        k::layer_norm_rows(
            input,
            rows,
            Some(self.tensor(&format!("{name}.weight"))?),
            Some(self.tensor(&format!("{name}.bias"))?),
            1e-5,
        )
    }

    fn group_norm(&self, name: &str, x: &Feature) -> Result<Feature, String> {
        let mut groups = 32.min(x.channels);
        while x.channels % groups != 0 {
            groups -= 1;
        }
        Feature::new(
            k::group_norm_ncthw(
                &x.values,
                [x.channels, 1, x.height, x.width],
                groups,
                self.tensor(&format!("{name}.weight"))?,
                self.tensor(&format!("{name}.bias"))?,
                1e-5,
            )?,
            x.channels,
            x.height,
            x.width,
        )
    }

    fn conv(
        &self,
        name: &str,
        x: &Feature,
        stride: usize,
        groups: usize,
    ) -> Result<Feature, String> {
        let shape = &self.shapes[&format!("{name}.weight")];
        let dims: [usize; 4] = shape
            .as_slice()
            .try_into()
            .map_err(|_| "invalid conv dimensions")?;
        let bias = if self.shapes.contains_key(&format!("{name}.bias")) {
            Some(self.tensor(&format!("{name}.bias"))?)
        } else {
            None
        };
        let (values, [c, h, w]) = k::conv2d(
            self.pool,
            &x.values,
            [x.channels, x.height, x.width],
            self.tensor(&format!("{name}.weight"))?,
            dims,
            bias,
            [stride; 2],
            [dims[2] / 2, dims[3] / 2],
            [1; 2],
            groups,
        )?;
        Feature::new(values, c, h, w)
    }

    fn project(&self, name: &str, x: &Feature) -> Result<Feature, String> {
        self.group_norm(
            &format!("{name}.1"),
            &self.conv(&format!("{name}.0"), x, 1, 1)?,
        )
        .map(Feature::gelu)
    }

    fn transpose(&self, name: &str, x: &Feature) -> Result<Feature, String> {
        let s = &self.shapes[&format!("{name}.weight")];
        let (values, [c, _, h, w]) = k::conv_transpose3d(
            self.pool,
            &x.values,
            [x.channels, 1, x.height, x.width],
            self.tensor(&format!("{name}.weight"))?,
            [s[0], s[1], 1, s[2], s[3]],
            None,
            k::ConvTranspose3dSpec {
                stride: [1, 2, 2],
                padding: [0, 1, 1],
                output_padding: [0; 3],
                dilation: [1; 3],
                groups: 1,
            },
        )?;
        Feature::new(values, c, h, w)
    }

    fn shuffle(&self, name: &str, x: &Feature, residual: bool) -> Result<Feature, String> {
        let v = self.conv(&format!("{name}.conv"), x, 1, 1)?;
        let c = v.channels / 4;
        let (h, w) = (v.height * 2, v.width * 2);
        let mut values = vec![0.0; v.values.len()];
        for ch in 0..c {
            for y in 0..h {
                for xx in 0..w {
                    values[(ch * h + y) * w + xx] = v.values
                        [((ch * 4 + (y % 2) * 2 + xx % 2) * v.height + y / 2) * v.width + xx / 2];
                }
            }
        }
        let out = Feature::new(values, c, h, w)?;
        if residual {
            x.resize(h, w)?.add(&out)
        } else {
            Ok(out.gelu())
        }
    }

    fn ffn(&self, name: &str, x: &[f32], rows: usize) -> Result<Vec<f32>, String> {
        let mut v = self.linear(&format!("{name}.linear1"), x, rows)?;
        gelu_erf_inplace(&mut v);
        self.linear(&format!("{name}.linear2"), &v, rows)
    }

    fn encoded_levels(&self, features: &[Feature]) -> Result<Vec<Feature>, String> {
        let (h, w) = (features[0].height, features[0].width);
        let n = h * w;
        let mut levels = Vec::new();
        for (i, f) in features.iter().enumerate() {
            levels.push(self.group_norm(
                &format!("visual_encoder.input_proj.{i}.1"),
                &self.conv(&format!("visual_encoder.input_proj.{i}.0"), f, 1, 1)?,
            )?);
        }
        let level_embed = self.tensor("visual_encoder.level_embed")?;
        let mut positions = vec![vec![0.0; n * 192]; 4];
        for l in 0..4 {
            for y in 0..h {
                for x in 0..w {
                    for c in 0..192 {
                        let j = c % 96;
                        let phase = if c < 96 {
                            (y as f32 + 0.5) / h as f32
                        } else {
                            (x as f32 + 0.5) / w as f32
                        } * std::f32::consts::TAU
                            / 10000.0f32.powf((2 * (j / 2)) as f32 / 96.0);
                        positions[l][(y * w + x) * 192 + c] =
                            if j % 2 == 0 { phase.sin() } else { phase.cos() }
                                + level_embed[l * 192 + c];
                    }
                }
            }
        }
        for layer in 0..3 {
            let name = format!("visual_encoder.encoder.layers.{layer}");
            for l in 0..4 {
                let rows = levels[l].rows();
                let added = add(&rows, &positions[l])?;
                let spatial = Feature::from_rows(&added, 192, h, w)?;
                let mixed = self
                    .conv(
                        &format!("{name}.pointwise"),
                        &self
                            .conv(&format!("{name}.depthwise"), &spatial, 1, 192)?
                            .gelu(),
                        1,
                        1,
                    )?
                    .rows();
                let src = self.norm(&format!("{name}.norm1"), &add(&rows, &mixed)?, n)?;
                let out = self.norm(
                    &format!("{name}.norm2"),
                    &add(&src, &self.ffn(&name, &src, n)?)?,
                    n,
                )?;
                levels[l] = Feature::from_rows(&out, 192, h, w)?;
            }
        }
        Ok(levels)
    }

    fn queries(
        &self,
        hidden: &[f32],
        boxes: &[[f32; 4]],
        levels: &[Feature],
    ) -> Result<Vec<f32>, String> {
        let n = boxes.len();
        let mut target = self.norm(
            "query_builder.content_norm",
            &self.linear("query_builder.content_proj", hidden, n)?,
            n,
        )?;
        let mut encoded = Vec::with_capacity(n * 256);
        for b in boxes {
            for v in [
                b[0],
                b[1],
                0.2 * (b[2] - b[0]).max(1e-6).ln() + 0.5,
                0.2 * (b[3] - b[1]).max(1e-6).ln() + 0.5,
            ] {
                for cosine in [false, true] {
                    for frequency in 0..32 {
                        let phase = std::f32::consts::TAU * v * 2.0f32.powi(frequency);
                        encoded.push(if cosine { phase.cos() } else { phase.sin() });
                    }
                }
            }
        }
        let mut p = self.linear("query_builder.box_position.mlp.0", &encoded, n)?;
        gelu_erf_inplace(&mut p);
        let position = self.norm(
            "query_builder.position_norm",
            &self.linear("query_builder.box_position.mlp.2", &p, n)?,
            n,
        )?;
        let memory = levels.iter().flat_map(Feature::rows).collect::<Vec<_>>();
        let plane = levels[0].height * levels[0].width;
        for layer in 0..6 {
            let name = format!("transformer.layers.{layer}");
            let qk = add(&target, &position)?;
            let weight = self.tensor(&format!("{name}.self_attn.in_proj_weight"))?;
            let bias = self.tensor(&format!("{name}.self_attn.in_proj_bias"))?;
            let project = |input: &[f32], part: usize| {
                let mut output = vec![0.0; n * 192];
                for row in 0..n {
                    for o in 0..192 {
                        output[row * 192 + o] = crate::ops::dot_f32(
                            &input[row * 192..(row + 1) * 192],
                            &weight[(part * 192 + o) * 192..(part * 192 + o + 1) * 192],
                            192,
                        ) + bias[part * 192 + o];
                    }
                }
                output
            };
            let sa = k::attention_scalar(
                &project(&qk, 0),
                &project(&qk, 1),
                &project(&target, 2),
                k::AttentionSpec {
                    query_tokens: n,
                    key_tokens: n,
                    query_heads: 8,
                    key_value_heads: 8,
                    head_dim: 24,
                    causal: false,
                    scale: 1.0 / 24.0f32.sqrt(),
                },
            )?;
            target = self.norm(
                &format!("{name}.norm1"),
                &add(
                    &target,
                    &self.linear(&format!("{name}.self_attn.out_proj"), &sa, n)?,
                )?,
                n,
            )?;
            let query = add(&target, &position)?;
            let ca = format!("{name}.cross_attn");
            let values = self.linear(&format!("{ca}.value_proj"), &memory, plane * 4)?;
            let offsets = self.linear(&format!("{ca}.sampling_offsets"), &query, n)?;
            let mut weights = self.linear(&format!("{ca}.attention_weights"), &query, n)?;
            for row in weights.chunks_exact_mut(16) {
                softmax_inplace(row);
            }
            let mut output = vec![0.0; n * 192];
            let (h, w) = (levels[0].height, levels[0].width);
            for q in 0..n {
                for head in 0..8 {
                    for level in 0..4 {
                        for point in 0..4 {
                            let i = ((q * 8 + head) * 4 + level) * 4 + point;
                            let b = boxes[q];
                            let sx = ((b[0] + b[2]) * 0.5
                                + offsets[i * 2] / 4.0 * (b[2] - b[0]).max(1e-4) * 0.5)
                                * w as f32
                                - 0.5;
                            let sy = ((b[1] + b[3]) * 0.5
                                + offsets[i * 2 + 1] / 4.0 * (b[3] - b[1]).max(1e-4) * 0.5)
                                * h as f32
                                - 0.5;
                            let x = sx.floor() as isize;
                            let y = sy.floor() as isize;
                            let fx = sx - x as f32;
                            let fy = sy - y as f32;
                            for c in 0..24 {
                                let mut sample = 0.0;
                                for (iy, wy) in [(y, 1.0 - fy), (y + 1, fy)] {
                                    for (ix, wx) in [(x, 1.0 - fx), (x + 1, fx)] {
                                        if iy >= 0
                                            && ix >= 0
                                            && (iy as usize) < h
                                            && (ix as usize) < w
                                        {
                                            sample += values[(level * plane
                                                + iy as usize * w
                                                + ix as usize)
                                                * 192
                                                + head * 24
                                                + c]
                                                * wx
                                                * wy;
                                        }
                                    }
                                }
                                output[q * 192 + head * 24 + c] += sample * weights[i];
                            }
                        }
                    }
                }
            }
            target = self.norm(
                &format!("{name}.norm2"),
                &add(
                    &target,
                    &self.linear(&format!("{ca}.output_proj"), &output, n)?,
                )?,
                n,
            )?;
            target = self.norm(
                &format!("{name}.norm3"),
                &add(&target, &self.ffn(&name, &target, n)?)?,
                n,
            )?;
        }
        Ok(target)
    }

    fn pixels(&self, image: &Feature, levels: &[Feature]) -> Result<Feature, String> {
        let mut fused = self.conv("pixel_fusion.fpn.lateral.3", &levels[3], 1, 1)?;
        for (i, l) in (0..3).rev().enumerate() {
            fused = fused.add(&self.conv(
                &format!("pixel_fusion.fpn.lateral.{l}"),
                &levels[l],
                1,
                1,
            )?)?;
            fused = self.project(&format!("pixel_fusion.fpn.output.{i}"), &fused)?;
        }
        let mut fpn = self.shuffle(
            "pixel_fusion.upsample.1",
            &self.shuffle("pixel_fusion.upsample.0", &fused, false)?,
            false,
        )?;
        let cnn = self
            .group_norm(
                "pixel_fusion.cnn.net.1",
                &self.conv("pixel_fusion.cnn.net.0", image, 2, 1)?,
            )?
            .gelu();
        let cnn = self
            .group_norm(
                "pixel_fusion.cnn.net.4",
                &self.conv("pixel_fusion.cnn.net.3", &cnn, 1, 1)?,
            )?
            .gelu();
        let (h, w) = (cnn.height / 2, cnn.width / 2);
        if (fpn.height, fpn.width) != (h, w) {
            let (ph, pw) = (fpn.height.min(h), fpn.width.min(w));
            if (ph, pw) != (fpn.height, fpn.width) {
                fpn = fpn.pool(ph, pw)?;
            }
            if (ph, pw) != (h, w) {
                fpn = fpn.resize(h, w)?;
            }
        }
        let fused = self
            .group_norm(
                "pixel_fusion.smooth.0",
                &self.conv("pixel_fusion.fpn_proj", &fpn, 1, 1)?,
            )?
            .gelu();
        let fused = self.conv("pixel_fusion.smooth.2", &fused, 1, 1)?;
        self.shuffle("pixel_fusion.s2_upsample", &fused, true)?
            .add(&self.conv("pixel_fusion.cnn_proj", &cnn, 1, 1)?)
    }

    fn condition(
        &self,
        stage: &str,
        mut feature: Feature,
        query: &[f32],
    ) -> Result<Feature, String> {
        let affine = self.linear(&format!("highres_refine.query_{stage}"), query, 1)?;
        for c in 0..48 {
            for v in &mut feature.values
                [c * feature.height * feature.width..(c + 1) * feature.height * feature.width]
            {
                *v = *v * (1.0 + 0.1 * affine[c].tanh()) + 0.1 * affine[c + 48];
            }
        }
        for b in 0..2 {
            let name = format!("highres_refine.blocks_{stage}.{b}.net");
            let v = self.group_norm(&format!("{name}.0"), &feature)?.gelu();
            let v = self.conv(&format!("{name}.2"), &v, 1, 1)?;
            let v = self.group_norm(&format!("{name}.3"), &v)?.gelu();
            feature = feature.add(&self.conv(&format!("{name}.5"), &v, 1, 1)?)?;
        }
        Ok(feature)
    }

    /// One FP32 logit canvas per instance; callers sigmoid before resizing.
    pub fn forward(
        &self,
        image: &Feature,
        features: &[Feature],
        hidden: &[f32],
        boxes: &[[f32; 4]],
    ) -> Result<Vec<Feature>, String> {
        let valid_feature = |f: &Feature| {
            f.channels
                .checked_mul(f.height)
                .and_then(|n| n.checked_mul(f.width))
                == Some(f.values.len())
                && f.channels > 0
                && f.height > 0
                && f.width > 0
                && f.values.iter().all(|v| v.is_finite())
        };
        if !valid_feature(image)
            || features.iter().any(|f| !valid_feature(f))
            || features.len() != 4
            || image.channels != 3
            || image.height != image.width
            || image.height % 4 != 0
            || image.height > 1024
            || boxes.len() > 64
            || hidden.len() != boxes.len() * 2560
            || hidden.iter().any(|v| !v.is_finite())
            || boxes.iter().any(|b| {
                b.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
                    || b[2] <= b[0]
                    || b[3] <= b[1]
            })
            || features.iter().any(|f| {
                f.channels != 1024 || f.height != features[0].height || f.width != features[0].width
            })
        {
            return Err("invalid WeVisOne pixel-decoder input".into());
        }
        if boxes.is_empty() {
            return Ok(Vec::new());
        }
        let levels = self.encoded_levels(features)?;
        let queries = self.queries(hidden, boxes, &levels)?;
        let pixels = self.pixels(image, &levels)?;
        let mut params = self.linear("mask_predictor.kernel.0", &queries, boxes.len())?;
        gelu_erf_inplace(&mut params);
        let params = self.linear("mask_predictor.kernel.2", &params, boxes.len())?;
        let semantic = self.project("highres_refine.semantic", &pixels)?;
        let (h, w) = (pixels.height, pixels.width);
        let (h4, w4) = (h / 2, w / 2);
        let semantic4 = semantic.resize(h4, w4)?;
        let rgb4 = self.project("highres_refine.rgb_s4", &image.resize(h4, w4)?)?;
        let rgb2 = self.project("highres_refine.rgb_s2", &image.resize(h, w)?)?;
        let semantic1 = semantic.resize(image.height, image.width)?;
        let rgb1 = self.project("highres_refine.rgb_s1", image)?;
        let mut result = Vec::new();
        for q in 0..boxes.len() {
            let mut values = vec![params[q * 65 + 64]; h * w];
            for (c, &kernel) in params[q * 65..q * 65 + 64].iter().enumerate() {
                for (out, &v) in values
                    .iter_mut()
                    .zip(&pixels.values[c * h * w..(c + 1) * h * w])
                {
                    *out += kernel * v;
                }
            }
            let coarse = Feature::new(values, 1, h, w)?;
            let coarse4 = coarse.resize(h4, w4)?;
            let f4 = concat(&[
                &semantic4,
                &rgb4,
                &self.project("highres_refine.coarse_s4", &coarse4)?,
            ])?;
            let f4 = self.condition(
                "s4",
                self.conv("highres_refine.fuse_s4", &f4, 1, 1)?,
                &queries[q * 192..(q + 1) * 192],
            )?;
            let logits4 = coarse4.add(&self.conv("highres_refine.output_s4", &f4, 1, 1)?)?;
            let previous2 = logits4.resize(h, w)?;
            let f2 = concat(&[
                &self.transpose("highres_refine.up_s4_to_s2", &f4)?,
                &semantic,
                &rgb2,
                &self.project("highres_refine.coarse_s2", &coarse)?,
                &self.project("highres_refine.previous_s2", &previous2.sigmoid())?,
            ])?;
            let f2 = self.condition(
                "s2",
                self.conv("highres_refine.fuse_s2", &f2, 1, 1)?,
                &queries[q * 192..(q + 1) * 192],
            )?;
            let logits2 = coarse.add(&self.conv("highres_refine.output_s2", &f2, 1, 1)?)?;
            let previous1 = logits2.resize(image.height, image.width)?;
            let f1 = concat(&[
                &self.transpose("highres_refine.up_s2_to_s1", &f2)?,
                &semantic1,
                &rgb1,
                &self.project("highres_refine.previous_s1", &previous1.clone().sigmoid())?,
            ])?;
            let f1 = self.condition(
                "s1",
                self.conv("highres_refine.fuse_s1", &f1, 1, 1)?,
                &queries[q * 192..(q + 1) * 192],
            )?;
            result.push(previous1.add(&self.conv("highres_refine.output_s1", &f1, 1, 1)?)?);
        }
        Ok(result)
    }
}

fn add(a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
    if a.len() != b.len() {
        return Err("WeVisOne row shape mismatch".into());
    }
    Ok(a.iter().zip(b).map(|(a, b)| a + b).collect())
}
fn concat(xs: &[&Feature]) -> Result<Feature, String> {
    if xs.is_empty()
        || xs
            .iter()
            .any(|x| x.height != xs[0].height || x.width != xs[0].width)
    {
        return Err("WeVisOne concat shape mismatch".into());
    }
    Feature::new(
        xs.iter().flat_map(|x| x.values.iter().copied()).collect(),
        xs.iter().map(|x| x.channels).sum(),
        xs[0].height,
        xs[0].width,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn premerge_restores_spatial_order_and_rejects_bad_grids() {
        let rows = (0..24 * 3).map(|x| x as f32).collect::<Vec<_>>();
        let image = Feature::from_premerge(&rows, 3, 4, 6, 2).unwrap();
        assert_eq!(&image.values[..6], &[0.0, 3.0, 12.0, 15.0, 24.0, 27.0]);
        assert!(Feature::from_premerge(&rows, 3, 3, 8, 2).is_err());
        assert!(Feature::new(vec![f32::NAN], 1, 1, 1).is_err());
    }

    fn fixture_root() -> PathBuf {
        PathBuf::from(
            std::env::var_os("WEVISONE_FIXTURE")
                .expect("set WEVISONE_FIXTURE to the pinned Oracle output directory"),
        )
    }

    fn read(root: &std::path::Path, name: &str) -> Vec<f32> {
        std::fs::read(root.join(format!("{name}.f32")))
            .unwrap()
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect()
    }

    fn compare(name: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{name} shape");
        let mut worst = 0.0f32;
        for (&a, &e) in actual.iter().zip(expected) {
            assert!(a.is_finite());
            worst = worst.max((a - e).abs());
            assert!(
                (a - e).abs() <= 5e-4 + 3e-4 * e.abs(),
                "{name}: actual={a} expected={e}, max_abs={worst}"
            );
        }
        eprintln!("{name}: max_abs={worst}");
    }

    #[test]
    #[ignore = "requires actual decoder weights and the pinned official Python Oracle"]
    fn official_decoder_fixture() {
        let root = fixture_root();
        let source = crate::open_model_source(
            &root.join("decoder.gguf"),
            crate::format::ggufrs::ComponentRole::Llm,
        )
        .unwrap();
        let pool = ComputePool::new(2);
        let decoder = Decoder::new(source.as_ref(), &pool).unwrap();
        let image = Feature::new(read(&root, "image"), 3, 32, 32).unwrap();
        let features = (0..4)
            .map(|i| Feature::new(read(&root, &format!("feature{i}")), 1024, 4, 6).unwrap())
            .collect::<Vec<_>>();
        let hidden = read(&root, "hidden");
        let boxes = read(&root, "boxes")
            .chunks_exact(4)
            .map(|v| v.try_into().unwrap())
            .collect::<Vec<[f32; 4]>>();
        let levels = decoder.encoded_levels(&features).unwrap();
        for (i, level) in levels.iter().enumerate() {
            compare(
                &format!("encoded{i}"),
                &level.values,
                &read(&root, &format!("encoded{i}")),
            );
        }
        let queries = decoder.queries(&hidden, &boxes, &levels).unwrap();
        compare("queries", &queries, &read(&root, "queries"));
        compare(
            "pixels",
            &decoder.pixels(&image, &levels).unwrap().values,
            &read(&root, "pixels"),
        );
        let output = decoder.forward(&image, &features, &hidden, &boxes).unwrap();
        compare(
            "mask_logits",
            &output
                .into_iter()
                .flat_map(|v| v.values)
                .collect::<Vec<_>>(),
            &read(&root, "mask_logits"),
        );
        assert!(decoder
            .forward(&image, &features, &hidden, &[[0.9, 0.1, 0.1, 0.7]])
            .is_err());
    }
}
