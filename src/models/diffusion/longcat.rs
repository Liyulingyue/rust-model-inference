//! LongCat Image Edit / Edit Turbo Q8_0 Transformer, packed latent input.
//! This is explicitly a LongCat contract despite GGUF's generic `flux` metadata.
//! Full image editing additionally needs Qwen2.5-VL-7B, its mmproj and a Flux VAE.
//! Scalar arithmetic for the ggml parity contract: start the process with
//! `RMI_SCALAR=1`. Without it the packed NEON/AVX2 kernels are used instead —
//! same weight precision, same exact Q8_0 integer accumulation, different
//! summation order, so results are no longer bitwise identical to the Oracle.

use crate::core::tensor::{load_f32_tensor, GGMLType, MetaValue, TensorSource};
use crate::ops;
use rayon::prelude::*;

pub const HIDDEN: usize = 3072;
pub const TEXT_WIDTH: usize = 3584;
pub const IMAGE_WIDTH: usize = 64;
const HEAD_DIM: usize = 128;
const MLP: usize = 12288;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LongCatKind {
    Edit,
    EditTurbo,
}

fn tensor<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    dims: &[u64],
    dtype: GGMLType,
) -> Result<&'a [u8], String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("LongCat missing tensor: {name}"))?;
    if info.dims != dims || info.ggml_type != dtype {
        return Err(format!(
            "LongCat invalid {name}: {:?} {:?}; expected {dims:?} {dtype:?}",
            info.dims, info.ggml_type
        ));
    }
    let data = source
        .tensor_slice(name)
        .ok_or_else(|| format!("LongCat missing data: {name}"))?;
    if info.checked_nbytes().and_then(|n| usize::try_from(n).ok()) != Some(data.len()) {
        return Err(format!("LongCat invalid tensor byte length: {name}"));
    }
    Ok(data)
}

fn vector(source: &dyn TensorSource, name: &str, len: usize) -> Result<Vec<f32>, String> {
    tensor(source, name, &[len as u64], GGMLType::F32)?;
    let values = load_f32_tensor(source, name, &[len as u64])?;
    if values.iter().any(|x| !x.is_finite()) {
        return Err(format!("LongCat non-finite {name}"));
    }
    Ok(values)
}

struct Linear<'a> {
    weight: &'a [u8],
    bias: Vec<f32>,
    dtype: GGMLType,
    input: usize,
    output: usize,
}
impl<'a> Linear<'a> {
    fn load(
        source: &'a dyn TensorSource,
        name: &str,
        input: usize,
        output: usize,
        dtype: GGMLType,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: tensor(
                source,
                &format!("{name}.weight"),
                &[input as u64, output as u64],
                dtype,
            )?,
            bias: vector(source, &format!("{name}.bias"), output)?,
            dtype,
            input,
            output,
        })
    }
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len() % self.input, 0);
        // `RMI_SCALAR=1` keeps the ggml parity contract (BF16-rounded
        // activations, F64 accumulation, scalar Q8_0). Without it the packed
        // NEON/AVX2 kernels run instead: same weight precision, same exact
        // integer Q8_0 accumulation, different summation order.
        let scalar = ops::scalar_mode();
        let mut out = vec![0.0; x.len() / self.input * self.output];
        for (x, y) in x
            .chunks_exact(self.input)
            .zip(out.chunks_exact_mut(self.output))
        {
            match self.dtype {
                GGMLType::BF16 => {
                    if scalar {
                        let rounded: Vec<u8> = x
                            .iter()
                            .flat_map(|&v| ops::f32_to_bf16(v).to_le_bytes())
                            .collect();
                        y.par_chunks_mut(128).enumerate().for_each(|(part, rows)| {
                            for (offset, value) in rows.iter_mut().enumerate() {
                                let row = part * 128 + offset;
                                *value = ops::kernel::bf16::scalar::dot_bf16(
                                    &self.weight[row * self.input * 2..(row + 1) * self.input * 2],
                                    &rounded,
                                );
                            }
                        });
                    } else {
                        ops::kernel::bf16::matmul_bf16_vs_f32_range(
                            self.weight,
                            x,
                            y,
                            self.input,
                            0,
                            self.output,
                        );
                    }
                }
                GGMLType::Q8_0 => {
                    let mut q8 = vec![0; self.input];
                    let mut scales = vec![0.0; self.input / 32];
                    if scalar {
                        ops::quant::q8_0::quantize_q8_0_into_scalar_range(
                            x,
                            self.input,
                            &mut q8,
                            &mut scales,
                            0,
                            self.input / 32,
                        );
                    } else {
                        ops::quant::q8_0::quantize_q8_0_into(
                            x,
                            self.input,
                            &mut q8,
                            &mut scales,
                        );
                    }
                    y.par_chunks_mut(128).enumerate().for_each(|(part, rows)| {
                        let start = part * 128;
                        if scalar {
                            ops::kernel::q8_0::scalar::matmul_q8_0_quantized_scalar_range(
                                self.weight,
                                &q8,
                                &scales,
                                rows,
                                self.input,
                                start,
                                start + rows.len(),
                            );
                        } else {
                            ops::kernel::q8_0::dispatch::matmul_q8_0_quantized_range(
                                self.weight,
                                &q8,
                                &scales,
                                rows,
                                self.input,
                                start,
                                start + rows.len(),
                            );
                        }
                    });
                }
                _ => unreachable!("validated LongCat dtype"),
            }
            for (value, bias) in y.iter_mut().zip(&self.bias) {
                *value += bias;
            }
        }
        out
    }
}

struct QkNorm {
    query: Vec<f32>,
    key: Vec<f32>,
}
impl QkNorm {
    fn load(source: &dyn TensorSource, name: &str) -> Result<Self, String> {
        Ok(Self {
            query: vector(source, &format!("{name}.query_norm.weight"), HEAD_DIM)?,
            key: vector(source, &format!("{name}.key_norm.weight"), HEAD_DIM)?,
        })
    }
    fn apply(&self, q: &mut [f32], k: &mut [f32]) {
        for (q, k) in q
            .chunks_exact_mut(HEAD_DIM)
            .zip(k.chunks_exact_mut(HEAD_DIM))
        {
            ops::rms_norm_inplace(q, &self.query, 1e-6);
            ops::rms_norm_inplace(k, &self.key, 1e-6);
        }
    }
}

struct Stream<'a> {
    modulation: Linear<'a>,
    qkv: Linear<'a>,
    proj: Linear<'a>,
    up: Linear<'a>,
    down: Linear<'a>,
    norm: QkNorm,
}
impl<'a> Stream<'a> {
    fn load(source: &'a dyn TensorSource, name: &str, stream: &str) -> Result<Self, String> {
        let q8 = GGMLType::Q8_0;
        Ok(Self {
            modulation: Linear::load(
                source,
                &format!("{name}.{stream}_mod.lin"),
                HIDDEN,
                HIDDEN * 6,
                q8,
            )?,
            qkv: Linear::load(
                source,
                &format!("{name}.{stream}_attn.qkv"),
                HIDDEN,
                HIDDEN * 3,
                q8,
            )?,
            proj: Linear::load(
                source,
                &format!("{name}.{stream}_attn.proj"),
                HIDDEN,
                HIDDEN,
                q8,
            )?,
            up: Linear::load(source, &format!("{name}.{stream}_mlp.0"), HIDDEN, MLP, q8)?,
            down: Linear::load(source, &format!("{name}.{stream}_mlp.2"), MLP, HIDDEN, q8)?,
            norm: QkNorm::load(source, &format!("{name}.{stream}_attn.norm"))?,
        })
    }
    fn prepare(&self, x: &[f32], modulation: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        qkv(
            &self.qkv.forward(&modulate(x, modulation)),
            HIDDEN * 3,
            &self.norm,
        )
    }
    fn finish(&self, mut x: Vec<f32>, attn: &[f32], modulation: &[f32]) -> Vec<f32> {
        residual(
            &mut x,
            &self.proj.forward(attn),
            &modulation[HIDDEN * 2..HIDDEN * 3],
        );
        let mut mlp = self.up.forward(&modulate(&x, &modulation[HIDDEN * 3..]));
        for v in &mut mlp {
            *v = ops::gelu_ggml_f32(*v);
        }
        residual(&mut x, &self.down.forward(&mlp), &modulation[HIDDEN * 5..]);
        x
    }
}
struct Double<'a> {
    img: Stream<'a>,
    txt: Stream<'a>,
}
struct Single<'a> {
    modulation: Linear<'a>,
    linear1: Linear<'a>,
    linear2: Linear<'a>,
    norm: QkNorm,
}

/// mmap-borrowing Transformer for both supplied LongCat checkpoints.
/// `kind` explicitly selects Edit or Turbo; the files share one tensor contract.
pub struct LongCatTransformer<'a> {
    pub kind: LongCatKind,
    img_in: Linear<'a>,
    txt_in: Linear<'a>,
    time_in: Linear<'a>,
    time_out: Linear<'a>,
    double: Vec<Double<'a>>,
    single: Vec<Single<'a>>,
    final_mod: Linear<'a>,
    final_out: Linear<'a>,
}
impl<'a> LongCatTransformer<'a> {
    pub fn load(source: &'a dyn TensorSource, kind: LongCatKind) -> Result<Self, String> {
        if source.metadata("general.architecture") != Some(&MetaValue::String("flux".into())) {
            return Err("LongCat requires explicit flux GGUF tensor contract".into());
        }
        // Reject known sibling contracts; the remaining required shapes are checked below.
        for name in [
            "vector_in.in_layer.weight",
            "guidance_in.in_layer.weight",
            "double_blocks.10.img_mod.lin.weight",
            "single_blocks.20.linear1.weight",
        ] {
            if source.tensor_info(name).is_some() {
                return Err(format!("Not the LongCat Edit contract: {name}"));
            }
        }
        let bf16 = GGMLType::BF16;
        let q8 = GGMLType::Q8_0;
        // sd.cpp name conversion sorts raw names; norm_out overwrites this BF16 duplicate.
        // Validate both tensors but use norm_out, including its own bias, without half swapping.
        tensor(
            source,
            "final_layer.adaLN_modulation.1.weight",
            &[HIDDEN as u64, (HIDDEN * 2) as u64],
            bf16,
        )?;
        vector(source, "final_layer.adaLN_modulation.1.bias", HIDDEN * 2)?;
        let mut double = Vec::with_capacity(10);
        for i in 0..10 {
            let name = format!("double_blocks.{i}");
            double.push(Double {
                img: Stream::load(source, &name, "img")?,
                txt: Stream::load(source, &name, "txt")?,
            });
        }
        let mut single = Vec::with_capacity(20);
        for i in 0..20 {
            let name = format!("single_blocks.{i}");
            single.push(Single {
                modulation: Linear::load(
                    source,
                    &format!("{name}.modulation.lin"),
                    HIDDEN,
                    HIDDEN * 3,
                    q8,
                )?,
                linear1: Linear::load(
                    source,
                    &format!("{name}.linear1"),
                    HIDDEN,
                    HIDDEN * 3 + MLP,
                    q8,
                )?,
                linear2: Linear::load(
                    source,
                    &format!("{name}.linear2"),
                    HIDDEN + MLP,
                    HIDDEN,
                    q8,
                )?,
                norm: QkNorm::load(source, &format!("{name}.norm"))?,
            });
        }
        Ok(Self {
            kind,
            double,
            single,
            img_in: Linear::load(source, "img_in", IMAGE_WIDTH, HIDDEN, bf16)?,
            txt_in: Linear::load(source, "txt_in", TEXT_WIDTH, HIDDEN, bf16)?,
            time_in: Linear::load(source, "time_in.in_layer", 256, HIDDEN, bf16)?,
            time_out: Linear::load(source, "time_in.out_layer", HIDDEN, HIDDEN, bf16)?,
            final_mod: Linear::load(source, "norm_out.linear", HIDDEN, HIDDEN * 2, q8)?,
            final_out: Linear::load(source, "final_layer.linear", HIDDEN, IMAGE_WIDTH, bf16)?,
        })
    }

    /// Batch size 1. Row-major packed image tokens include target then references.
    /// Text positions precede image positions in `[modality, row, col]` order.
    /// Timestep uses the pipeline's [0,1] convention (multiplied by 1000 internally).
    /// Returns all image tokens; callers slice target tokens before unpacking.
    pub fn forward(
        &self,
        image: &[f32],
        text: &[f32],
        positions: &[[f32; 3]],
        timestep: f32,
    ) -> Result<Vec<f32>, String> {
        let (ni, nt) = validate_input(image, text, positions, timestep)?;
        let pe = rope_embedding(positions);
        let mut img = self.img_in.forward(image);
        let mut vec = self.time_in.forward(&timestep_embedding(timestep));
        for v in &mut vec {
            *v = ops::silu(*v);
        }
        vec = self.time_out.forward(&vec);
        let mut txt = self.txt_in.forward(text);
        trace("longcat.prelude.img", &img, ni, HIDDEN)?;
        trace("longcat.prelude.txt", &txt, nt, HIDDEN)?;
        trace("longcat.prelude.vec", &vec, 1, HIDDEN)?;
        let activated: Vec<f32> = vec.iter().copied().map(ops::silu).collect();
        for (i, block) in self.double.iter().enumerate() {
            let im = block.img.modulation.forward(&activated);
            let tm = block.txt.modulation.forward(&activated);
            let (iq, ik, iv) = block.img.prepare(&img, &im);
            let (mut q, mut k, mut v) = block.txt.prepare(&txt, &tm);
            q.extend(iq);
            k.extend(ik);
            v.extend(iv);
            let attn = attention(q, k, &v, &pe);
            img = block.img.finish(img, &attn[nt * HIDDEN..], &im);
            txt = block.txt.finish(txt, &attn[..nt * HIDDEN], &tm);
            trace(&format!("longcat.double.{i}.img"), &img, ni, HIDDEN)?;
            trace(&format!("longcat.double.{i}.txt"), &txt, nt, HIDDEN)?;
        }
        txt.extend(img);
        let mut x = txt;
        for (i, block) in self.single.iter().enumerate() {
            let modulation = block.modulation.forward(&activated);
            let fused = block.linear1.forward(&modulate(&x, &modulation));
            let width = HIDDEN * 3 + MLP;
            let (q, k, v) = qkv(&fused, width, &block.norm);
            let attn = attention(q, k, &v, &pe);
            let mut joined = Vec::with_capacity((ni + nt) * (HIDDEN + MLP));
            for (row, a) in fused.chunks_exact(width).zip(attn.chunks_exact(HIDDEN)) {
                joined.extend_from_slice(a);
                joined.extend(row[HIDDEN * 3..].iter().copied().map(ops::gelu_ggml_f32));
            }
            residual(
                &mut x,
                &block.linear2.forward(&joined),
                &modulation[HIDDEN * 2..],
            );
            trace(&format!("longcat.single.{i}"), &x, ni + nt, HIDDEN)?;
        }
        let modulation = self.final_mod.forward(&activated);
        let out = self
            .final_out
            .forward(&modulate(&x[nt * HIDDEN..], &modulation));
        if out.iter().any(|v| !v.is_finite()) {
            return Err("LongCat non-finite Transformer output".into());
        }
        trace("longcat.output", &out, ni, IMAGE_WIDTH)?;
        Ok(out)
    }
}

fn validate_input(
    image: &[f32],
    text: &[f32],
    positions: &[[f32; 3]],
    timestep: f32,
) -> Result<(usize, usize), String> {
    let ni = image.len() / IMAGE_WIDTH;
    let nt = text.len() / TEXT_WIDTH;
    if ni == 0
        || nt == 0
        || image.len() % IMAGE_WIDTH != 0
        || text.len() % TEXT_WIDTH != 0
        || ni.checked_add(nt) != Some(positions.len())
        || !timestep.is_finite()
        || !(0.0..=1.0).contains(&timestep)
        || image
            .iter()
            .chain(text)
            .chain(positions.iter().flatten())
            .any(|v| !v.is_finite())
    {
        return Err("Invalid LongCat packed input: nonempty image[Ni,64], text[Nt,3584], positions[Nt+Ni,3], finite timestep in [0,1] required".into());
    }
    Ok((ni, nt))
}

fn timestep_embedding(t: f32) -> Vec<f32> {
    let mut out = vec![0.0; 256];
    for j in 0..128 {
        let freq = (-10000.0f32.ln() * j as f32 / 128.0).exp();
        let arg = (t * 1000.0) * freq;
        out[j] = arg.cos();
        out[j + 128] = arg.sin();
    }
    out
}

fn modulate(x: &[f32], modulation: &[f32]) -> Vec<f32> {
    let mut out = x.to_vec();
    for row in out.chunks_exact_mut(HIDDEN) {
        // ggml norm: sum rounds to F32 before mean; variance divides in F64.
        let mean = ops::sum_f32(row) as f32 / HIDDEN as f32;
        let variance = (ops::sum_sq_centered_f32(row, mean) / HIDDEN as f64) as f32;
        let scale = 1.0 / (variance + 1e-6).sqrt();
        for (j, value) in row.iter_mut().enumerate() {
            let normalized = (*value - mean) * scale;
            *value = (normalized + normalized * modulation[HIDDEN + j]) + modulation[j];
        }
    }
    out
}
fn residual(x: &mut [f32], output: &[f32], gate: &[f32]) {
    for (i, (x, y)) in x.iter_mut().zip(output).enumerate() {
        *x += *y * gate[i % HIDDEN];
    }
}
fn qkv(x: &[f32], width: usize, norm: &QkNorm) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = Vec::with_capacity(x.len() / width * HIDDEN);
    let mut k = Vec::with_capacity(q.capacity());
    let mut v = Vec::with_capacity(q.capacity());
    for row in x.chunks_exact(width) {
        q.extend_from_slice(&row[..HIDDEN]);
        k.extend_from_slice(&row[HIDDEN..HIDDEN * 2]);
        v.extend_from_slice(&row[HIDDEN * 2..HIDDEN * 3]);
    }
    norm.apply(&mut q, &mut k);
    (q, k, v)
}
fn rope_embedding(positions: &[[f32; 3]]) -> Vec<[f32; 2]> {
    let mut out = Vec::with_capacity(positions.len() * HEAD_DIM / 2);
    for pos in positions {
        for (axis, dim) in [16, 56, 56].into_iter().enumerate() {
            let end = (dim as f32 - 2.0) / dim as f32;
            let step = end / (dim / 2 - 1) as f32;
            for j in 0..dim / 2 {
                let freq = 1.0 / 10000.0f32.powf(j as f32 * step);
                let angle = pos[axis] * freq;
                out.push([angle.cos(), angle.sin()]);
            }
        }
    }
    out
}
fn rotate(x: &mut [f32], pe: &[[f32; 2]]) {
    for (token, row) in x.chunks_exact_mut(HIDDEN).enumerate() {
        for head in row.chunks_exact_mut(HEAD_DIM) {
            for (j, pair) in head.chunks_exact_mut(2).enumerate() {
                let [c, s] = pe[token * HEAD_DIM / 2 + j];
                let (a, b) = (pair[0], pair[1]);
                pair[0] = a * c + b * -s;
                pair[1] = a * s + b * c;
            }
        }
    }
}
fn attention(mut q: Vec<f32>, mut k: Vec<f32>, v: &[f32], pe: &[[f32; 2]]) -> Vec<f32> {
    rotate(&mut q, pe);
    rotate(&mut k, pe);
    let n = q.len() / HIDDEN;
    let mut out = vec![0.0; q.len()];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut values = vec![0.0; n];
    let mut scores = vec![0.0; n];
    for head in 0..HIDDEN / HEAD_DIM {
        let h = head * HEAD_DIM;
        for token in 0..n {
            let query = &q[token * HIDDEN + h..token * HIDDEN + h + HEAD_DIM];
            for (j, score) in scores.iter_mut().enumerate() {
                *score = ops::dot_f32(
                    query,
                    &k[j * HIDDEN + h..j * HIDDEN + h + HEAD_DIM],
                    HEAD_DIM,
                ) * scale;
            }
            ops::softmax_inplace(&mut scores);
            for d in 0..HEAD_DIM {
                for j in 0..n {
                    values[j] = v[j * HIDDEN + h + d];
                }
                out[token * HIDDEN + h + d] = ops::dot_f32(&values, &scores, n);
            }
        }
    }
    out
}
fn trace(name: &str, values: &[f32], rows: usize, width: usize) -> Result<(), String> {
    #[cfg(feature = "parity-trace")]
    if crate::parity_trace::enabled(name) {
        crate::parity_trace::checkpoint(name, None, &[rows, width], values)
            .map_err(|e| e.to_string())?;
    }
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, values, rows, width);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::TensorInfo;
    struct Source {
        architecture: MetaValue,
        info: Option<TensorInfo>,
        bytes: Vec<u8>,
    }
    impl TensorSource for Source {
        fn metadata(&self, _: &str) -> Option<&MetaValue> {
            Some(&self.architecture)
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.info.as_ref().filter(|i| i.name == name)
        }
        fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
            Some(&self.bytes)
        }
    }
    #[test]
    fn rejects_wrong_architecture_shapes_and_inputs() {
        let mut source = Source {
            architecture: MetaValue::String("z-image".into()),
            info: None,
            bytes: vec![],
        };
        assert!(LongCatTransformer::load(&source, LongCatKind::Edit)
            .err()
            .unwrap()
            .contains("flux"));
        source.architecture = MetaValue::String("flux".into());
        assert!(LongCatTransformer::load(&source, LongCatKind::EditTurbo)
            .err()
            .unwrap()
            .contains("missing tensor"));
        source.info = Some(TensorInfo {
            name: "w".into(),
            dims: vec![32, 1],
            ggml_type: GGMLType::Q8_0,
            offset: 0,
        });
        assert!(tensor(&source, "w", &[64, 1], GGMLType::Q8_0).is_err());
        assert!(tensor(&source, "w", &[32, 1], GGMLType::BF16).is_err());
        assert!(tensor(&source, "w", &[32, 1], GGMLType::Q8_0).is_err());
        let image = vec![0.0; 64];
        let text = vec![0.0; 3584];
        let pos = [[0.0; 3], [1.0; 3]];
        assert_eq!(validate_input(&image, &text, &pos, 0.5).unwrap(), (1, 1));
        assert!(validate_input(&image, &text, &pos[..1], 0.5).is_err());
        assert!(validate_input(&image[..63], &text, &pos, 0.5).is_err());
        assert!(validate_input(&image, &text, &pos, f32::NAN).is_err());
        assert!(validate_input(&image, &text, &pos, 1.1).is_err());
        assert!(validate_input(&[], &text, &pos, 0.5).is_err());
    }
}
