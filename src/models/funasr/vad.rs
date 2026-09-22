//! FSMN-VAD: Voice Activity Detection for Fun-ASR-Nano.
//!
//! Pipeline: WAV (16kHz) → 80-mel fbank → LFR(m=5,n=1) → CMVN →
//! FSMN encoder (4 layers) → softmax → state machine → speech segments.
//!
//! Reference: FunASR llama.cpp `funasr-common/funasr_vad.h`.

use crate::core::loader::load_static_weight;
use crate::core::tensor::{GGMLType, MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::Weight;
use crate::ops::{softmax_inplace, vec_mad_per_channel_f32};
use rayon::prelude::*;
use std::sync::Arc;

const FS: usize = 16_000;
const WINLEN: usize = 400;
const SHIFT: usize = 160;
const NFFT: usize = 512;
const NMEL: usize = 80;
const PREEMPH: f32 = 0.97;
const LOWF: f32 = 20.0;
const HIGHF: f32 = 8000.0;
const FLT_EPS: f32 = 1.1920929e-7;

const LFR_M: usize = 5;
const LFR_N: usize = 1;
const FRAME_MS: usize = 10;
const WIN_FRAMES: usize = 20;
const SIL_TO_SPEECH: usize = 15;
const SPEECH_TO_SIL: usize = 15;
const LOOKAHEAD_END: usize = 10;
const CHUNK_FRAMES: usize = 6000;

#[inline]
fn mel_hz(f: f32) -> f32 {
    1127.0 * (1.0 + f / 700.0).ln()
}

struct VadConfig {
    input_dim: usize,
    proj_dim: usize,
    fsmn_layers: usize,
    lorder: usize,
    output_dim: usize,
    lfr_m: usize,
    lfr_n: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            input_dim: 400,
            proj_dim: 128,
            fsmn_layers: 4,
            lorder: 20,
            output_dim: 248,
            lfr_m: 5,
            lfr_n: 1,
        }
    }
}

struct VadLinear {
    weight: Weight<'static>,
    bias: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

struct FsmnVadLayer {
    linear: VadLinear,
    fsmn_kernel: Vec<f32>,
    affine: VadLinear,
}

pub struct FsmnVad {
    config: VadConfig,
    cmvn_shift: Vec<f32>,
    cmvn_scale: Vec<f32>,
    in_linear1: VadLinear,
    in_linear2: VadLinear,
    fsmn_layers: Vec<FsmnVadLayer>,
    out_linear1: VadLinear,
    out_linear2: VadLinear,
}

struct SharedMut<T>(*mut T);
unsafe impl<T> Send for SharedMut<T> {}
unsafe impl<T> Sync for SharedMut<T> {}
impl<T> SharedMut<T> {
    #[inline]
    unsafe fn slice(&self, start: usize, len: usize) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.0.add(start), len)
    }
}

pub fn is_fsmn_vad(source: &dyn TensorSource) -> bool {
    source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        .is_some_and(|arch| arch == "fsmn-vad")
}

pub struct VadSegment {
    pub start_ms: usize,
    pub end_ms: usize,
}

impl FsmnVad {
    pub fn new(source: Arc<dyn TensorSource>) -> Result<Self, String> {
        let arch = source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            .unwrap_or_default();
        if arch != "fsmn-vad" {
            return Err(format!("Expected fsmn-vad architecture, got {arch:?}"));
        }
        let def = VadConfig::default();
        let rd = |key: &str, default: usize| {
            source
                .metadata(key)
                .and_then(MetaValue::to_u64)
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        let config = VadConfig {
            input_dim: rd("vad.input_dim", def.input_dim),
            proj_dim: rd("vad.proj_dim", def.proj_dim),
            fsmn_layers: rd("vad.fsmn_layers", def.fsmn_layers),
            lorder: rd("vad.lorder", def.lorder),
            output_dim: rd("vad.output_dim", def.output_dim),
            lfr_m: rd("vad.lfr_m", def.lfr_m),
            lfr_n: rd("vad.lfr_n", def.lfr_n),
        };

        let cmvn_shift = load_f32_vec(source.as_ref(), "cmvn.shift")?;
        let cmvn_scale = load_f32_vec(source.as_ref(), "cmvn.scale")?;
        let in_linear1 = load_vad_linear(source.as_ref(), "encoder.in_linear1.linear.")?;
        let in_linear2 = load_vad_linear(source.as_ref(), "encoder.in_linear2.linear.")?;

        let mut fsmn_layers = Vec::with_capacity(config.fsmn_layers);
        for i in 0..config.fsmn_layers {
            let p = format!("encoder.fsmn.{i}.");
            fsmn_layers.push(FsmnVadLayer {
                linear: load_vad_linear(source.as_ref(), &format!("{p}linear.linear."))?,
                fsmn_kernel: load_f32_vec(source.as_ref(), &format!("{p}fsmn_block.conv_left.weight"))?,
                affine: load_vad_linear(source.as_ref(), &format!("{p}affine.linear."))?,
            });
        }

        let out_linear1 = load_vad_linear(source.as_ref(), "encoder.out_linear1.linear.")?;
        let out_linear2 = load_vad_linear(source.as_ref(), "encoder.out_linear2.linear.")?;

        Ok(Self {
            config,
            cmvn_shift,
            cmvn_scale,
            in_linear1,
            in_linear2,
            fsmn_layers,
            out_linear1,
            out_linear2,
        })
    }

    /// Run VAD on 16kHz mono samples → speech segments [start_ms, end_ms].
    pub fn segments(&self, wav: &[f32], max_seg_ms: usize) -> Vec<VadSegment> {
        let fbank = compute_fbank_80(wav);
        if fbank.is_empty() {
            return Vec::new();
        }
        let t_frames = fbank.len() / NMEL;

        let (feats, t_lfr) = lfr_stack(&fbank, t_frames, self.config.lfr_m, self.config.lfr_n);
        if t_lfr == 0 {
            return Vec::new();
        }

        let idim = self.config.input_dim;
        let mut feats = feats;
        for t in 0..t_lfr {
            for d in 0..idim {
                feats[t * idim + d] = (feats[t * idim + d] + self.cmvn_shift[d]) * self.cmvn_scale[d];
            }
        }

        let scores = self.forward(&feats, t_lfr);
        let od = self.config.output_dim;

        self.state_machine(&scores, t_lfr, od, max_seg_ms)
    }

    fn forward(&self, feats: &[f32], t: usize) -> Vec<f32> {
        let idim = self.config.input_dim;
        let pd = self.config.proj_dim;
        let lorder = self.config.lorder;
        let od = self.config.output_dim;

        let mut h = vad_linear_fwd(&self.in_linear1, feats, t);
        h = vad_linear_fwd(&self.in_linear2, &h, t);
        relu_inplace(&mut h);

        for layer in &self.fsmn_layers {
            let z = vad_linear_fwd(&layer.linear, &h, t);
            let fsmn = fsmn_conv_shift(&z, t, pd, lorder, &layer.fsmn_kernel);
            let mut acc = z;
            for i in 0..acc.len() {
                acc[i] += fsmn[i];
            }
            let a = vad_linear_fwd(&layer.affine, &acc, t);
            h = a;
            relu_inplace(&mut h);
        }

        h = vad_linear_fwd(&self.out_linear1, &h, t);
        h = vad_linear_fwd(&self.out_linear2, &h, t);

        for i in 0..t {
            softmax_inplace(&mut h[i * od..(i + 1) * od]);
        }
        h
    }

    fn state_machine(&self, scores: &[f32], t: usize, od: usize, max_seg_ms: usize) -> Vec<VadSegment> {
        let max_seg = (if max_seg_ms > 0 { max_seg_ms } else { 60000 }) / FRAME_MS;
        let start_lookback = WIN_FRAMES + 20;

        let mut acc: usize = 0;
        let mut insp: usize = 0;
        let mut max_end_sil: usize = 0;
        let mut end_lookback: usize = 0;
        let mut recompute = |acc: usize, max_end_sil: &mut usize, end_lookback: &mut usize| {
            let s = if acc <= 10000 { 2000 }
                else if acc <= 20000 { 1000 }
                else if acc <= 30000 { 800 }
                else if acc <= 40000 { 600 }
                else if acc <= 50000 { 400 }
                else if acc <= 60000 { 200 }
                else { 100 };
            let ms = if s > 150 { s - 150 } else { 0 };
            *max_end_sil = ms / FRAME_MS;
            *end_lookback = if *max_end_sil > LOOKAHEAD_END + 1 { *max_end_sil - LOOKAHEAD_END - 1 } else { 0 };
        };
        recompute(acc, &mut max_end_sil, &mut end_lookback);

        let mut wbuf = vec![0i32; WIN_FRAMES];
        let mut wpos = 0usize;
        let mut wsum = 0i32;
        let mut pre = 0i32;
        let mut st = 0i32;
        let mut cstart: i32 = -1;
        let mut csil = 0usize;
        let mut prev_end = 0usize;
        let mut segs: Vec<(usize, usize)> = Vec::new();

        for t_idx in 0..t {
            if t_idx > 0 && t_idx % CHUNK_FRAMES == 0 {
                if st == 1 || insp == 1 { acc += 60000; insp = 1; }
                recompute(acc, &mut max_end_sil, &mut end_lookback);
            }
            let sil = scores[t_idx * od];
            let fs = if (1.0 - sil) >= sil + 0.5 { 1 } else { 0 };
            wsum -= wbuf[wpos];
            wsum += fs;
            wbuf[wpos] = fs;
            wpos = (wpos + 1) % WIN_FRAMES;

            let ch = if pre == 0 && wsum >= SIL_TO_SPEECH as i32 { 3 }
                else if pre == 1 && wsum <= SPEECH_TO_SIL as i32 { 1 }
                else if pre == 0 { 0 } else { 2 };

            match ch {
                3 => {
                    csil = 0;
                    if st == 0 {
                        cstart = t_idx as i32 - start_lookback as i32;
                        if cstart < prev_end as i32 { cstart = prev_end as i32; }
                        if cstart < 0 { cstart = 0; }
                        st = 1;
                    } else if st == 1 && t_idx as i32 - cstart + 1 > max_seg as i32 {
                        let s = cstart as usize;
                        let e = t_idx.min(t);
                        if e > s.max(prev_end) { segs.push((s.max(prev_end), e)); prev_end = e; }
                        wbuf.fill(0); wpos = 0; wsum = 0; pre = 0; csil = 0; st = 0; cstart = -1; acc = 0; insp = 0;
                        recompute(acc, &mut max_end_sil, &mut end_lookback);
                    }
                }
                1 | 2 => {
                    csil = 0;
                    if st == 1 && t_idx as i32 - cstart + 1 > max_seg as i32 {
                        let s = cstart as usize;
                        let e = t_idx.min(t);
                        if e > s.max(prev_end) { segs.push((s.max(prev_end), e)); prev_end = e; }
                        wbuf.fill(0); wpos = 0; wsum = 0; pre = 0; csil = 0; st = 0; cstart = -1; acc = 0; insp = 0;
                        recompute(acc, &mut max_end_sil, &mut end_lookback);
                    }
                }
                _ => {
                    csil += 1;
                    if st == 1 {
                        if csil >= max_end_sil {
                            let end = if t_idx > end_lookback { t_idx - end_lookback } else { 0 };
                            let s = cstart as usize;
                            let e = end.min(t);
                            if e > s.max(prev_end) { segs.push((s.max(prev_end), e)); prev_end = e; }
                            wbuf.fill(0); wpos = 0; wsum = 0; pre = 0; csil = 0; st = 0; cstart = -1; acc = 0; insp = 0;
                            recompute(acc, &mut max_end_sil, &mut end_lookback);
                        } else if t_idx as i32 - cstart + 1 > max_seg as i32 {
                            let s = cstart as usize;
                            let e = t_idx.min(t);
                            if e > s.max(prev_end) { segs.push((s.max(prev_end), e)); prev_end = e; }
                            wbuf.fill(0); wpos = 0; wsum = 0; pre = 0; csil = 0; st = 0; cstart = -1; acc = 0; insp = 0;
                            recompute(acc, &mut max_end_sil, &mut end_lookback);
                        }
                    }
                }
            }
        }
        if st == 1 {
            let s = cstart as usize;
            let e = t;
            if e > s.max(prev_end) { segs.push((s.max(prev_end), e)); }
        }

        segs
            .into_iter()
            .map(|(s, e)| VadSegment {
                start_ms: s * FRAME_MS,
                end_ms: e * FRAME_MS,
            })
            .collect()
    }
}

// ======================= fbank (reuse 80-mel, no LFR) =======================

fn compute_fbank_80(wav: &[f32]) -> Vec<f32> {
    if wav.len() < WINLEN {
        return Vec::new();
    }
    let mut samples: Vec<f32> = wav.to_vec();
    for v in &mut samples {
        *v *= 32768.0;
    }
    let n = samples.len();
    let t_frames = (n - WINLEN) / SHIFT + 1;
    let nbin = NFFT / 2 + 1;
    let bw = FS as f32 / NFFT as f32;
    let ml = mel_hz(LOWF);
    let mh = mel_hz(HIGHF);
    let dm = (mh - ml) / (NMEL + 1) as f32;
    let mut fb = vec![vec![0.0f32; nbin]; NMEL];
    for m in 0..NMEL {
        let l = ml + m as f32 * dm;
        let c = ml + (m + 1) as f32 * dm;
        let r = ml + (m + 2) as f32 * dm;
        for k in 0..nbin {
            let mf = mel_hz(bw * k as f32);
            if mf > l && mf < r {
                fb[m][k] = if mf <= c { (mf - l) / (c - l) } else { (r - mf) / (r - c) };
            }
        }
    }
    let mut feat = vec![0.0f32; t_frames * NMEL];
    let mut re = vec![0.0f32; NFFT];
    let mut im = vec![0.0f32; NFFT];
    let mut frame = vec![0.0f32; WINLEN];
    let mut win = vec![0.0f32; WINLEN];
    for i in 0..WINLEN {
        win[i] = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (WINLEN - 1) as f32).cos();
    }
    for t in 0..t_frames {
        let s = &samples[t * SHIFT..];
        let mut mn = 0.0f64;
        for i in 0..WINLEN { mn += s[i] as f64; }
        mn /= WINLEN as f64;
        for i in 0..WINLEN { frame[i] = s[i] - mn as f32; }
        for i in (1..WINLEN).rev() { frame[i] -= PREEMPH * frame[i - 1]; }
        frame[0] -= PREEMPH * frame[0];
        for i in 0..NFFT {
            re[i] = if i < WINLEN { frame[i] * win[i] } else { 0.0 };
            im[i] = 0.0;
        }
        fft(&mut re, &mut im, NFFT);
        for m in 0..NMEL {
            let mut e = 0.0f32;
            for k in 0..nbin {
                if fb[m][k] > 0.0 { e += fb[m][k] * (re[k] * re[k] + im[k] * im[k]); }
            }
            feat[t * NMEL + m] = if e > FLT_EPS { e } else { FLT_EPS }.ln();
        }
    }
    feat
}

fn fft(re: &mut [f32], im: &mut [f32], n: usize) {
    let mut j = 0;
    for i in 1..n {
        let mut b = n >> 1;
        while j & b != 0 { j ^= b; b >>= 1; }
        j ^= b;
        if i < j { re.swap(i, j); im.swap(i, j); }
    }
    let mut len = 2;
    while len <= n {
        let a = -2.0 * std::f32::consts::PI / len as f32;
        let (wr, wi) = (a.cos(), a.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0f32, 0.0f32);
            for k in 0..len / 2 {
                let ur = re[i + k];
                let ui = im[i + k];
                let vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
                let vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                let n2 = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = n2;
            }
            i += len;
        }
        len <<= 1;
    }
}

fn lfr_stack(feat: &[f32], t: usize, m: usize, n: usize) -> (Vec<f32>, usize) {
    if t < 1 { return (Vec::new(), 0); }
    let pad = (m - 1) / 2;
    let tl = (t + n - 1) / n;
    let mut padded: Vec<Vec<f32>> = Vec::with_capacity(t + pad + m);
    for _ in 0..pad { padded.push(feat[0..NMEL].to_vec()); }
    for t_idx in 0..t { padded.push(feat[t_idx * NMEL..(t_idx + 1) * NMEL].to_vec()); }
    while padded.len() < (tl - 1) * n + m { padded.push(feat[(t - 1) * NMEL..t * NMEL].to_vec()); }
    let d = m * NMEL;
    let mut out = vec![0.0f32; tl * d];
    for i in 0..tl {
        for j in 0..m {
            let src = &padded[i * n + j];
            out[i * d + j * NMEL..i * d + (j + 1) * NMEL].copy_from_slice(&src[..NMEL]);
        }
    }
    (out, tl)
}

// ======================= forward primitives =======================

fn vad_linear_fwd(lin: &VadLinear, input: &[f32], t: usize) -> Vec<f32> {
    let in_dim = lin.in_dim;
    let out_dim = lin.out_dim;
    let mut out = vec![0.0f32; t * out_dim];
    out.par_chunks_mut(out_dim)
        .zip(input.par_chunks(in_dim))
        .for_each(|(o, row)| {
            lin.weight.kernel.forward(row, o, in_dim, out_dim);
            if !lin.bias.is_empty() {
                for i in 0..out_dim { o[i] += lin.bias[i]; }
            }
        });
    out
}

fn fsmn_conv_shift(z: &[f32], t: usize, dim: usize, lorder: usize, kernel: &[f32]) -> Vec<f32> {
    let pad = lorder - 1;
    let mut padded = vec![0.0f32; (t + pad) * dim];
    for i in 0..t {
        padded[(i + pad) * dim..(i + pad + 1) * dim].copy_from_slice(&z[i * dim..(i + 1) * dim]);
    }
    let mut out = vec![0.0f32; t * dim];
    out.par_chunks_mut(dim)
        .enumerate()
        .for_each(|(t_idx, o)| {
            for j in 0..lorder {
                let pad_idx = t_idx + j;
                let k_row = &kernel[j * dim..(j + 1) * dim];
                let pad_row = &padded[pad_idx * dim..(pad_idx + 1) * dim];
                vec_mad_per_channel_f32(o, pad_row, k_row);
            }
        });
    out
}

#[inline]
fn relu_inplace(x: &mut [f32]) {
    for v in x { if *v < 0.0 { *v = 0.0; } }
}

// ======================= weight loading =======================

fn load_vad_linear(source: &dyn TensorSource, prefix: &str) -> Result<VadLinear, String> {
    let weight_name = format!("{prefix}weight");
    let bias_name = format!("{prefix}bias");
    let info = source
        .tensor_info(&weight_name)
        .ok_or_else(|| format!("tensor {weight_name} not found"))?;
    let (in_dim, out_dim) = if info.dims.len() == 2 {
        (info.dims[0] as usize, info.dims[1] as usize)
    } else {
        return Err(format!("unexpected dims for {weight_name}: {:?}", info.dims));
    };
    let weight = load_static_weight(source, &weight_name, in_dim, out_dim);
    let bias = load_f32_vec_optional(source, &bias_name);
    Ok(VadLinear { weight, bias, in_dim, out_dim })
}

fn load_f32_vec_optional(source: &dyn TensorSource, name: &str) -> Vec<f32> {
    let Some(info) = source.tensor_info(name) else { return Vec::new() };
    let Some(bytes) = source.tensor_slice(name) else { return Vec::new() };
    let count = info.dims.iter().product::<u64>() as usize;
    let mut out = vec![0.0f32; count];
    match info.ggml_type {
        GGMLType::F32 => {
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
        }
        GGMLType::F16 => {
            for (i, chunk) in bytes.chunks_exact(2).enumerate() {
                out[i] = crate::ops::f16_to_f32(u16::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
        _ => return Vec::new(),
    }
    out
}

fn load_f32_vec(source: &dyn TensorSource, name: &str) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("tensor {name} not found"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("tensor slice {name} not found"))?;
    let count = info.dims.iter().product::<u64>() as usize;
    let mut out = vec![0.0f32; count];
    match info.ggml_type {
        GGMLType::F32 => {
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
        }
        GGMLType::F16 => {
            for (i, chunk) in bytes.chunks_exact(2).enumerate() {
                out[i] = crate::ops::f16_to_f32(u16::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
        _ => return Err(format!("unsupported type {:?} for {name}", info.ggml_type)),
    }
    Ok(out)
}
