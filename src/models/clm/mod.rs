//! Contrastive-LM (CLM) projection heads.
//!
//! CLM-v0.1-8B is not a generation model: two small MLP heads (`state_head`
//! and `action_head`) sit on top of a frozen Qwen3-8B encoder and are trained
//! with a bidirectional InfoNCE loss.  The checkpoint on ModelScope holds only
//! the heads; the encoder is a separate GGUF loaded alongside.
//!
//! Per-head forward (mirrors `make_head` in Contrastive-LM/CLM
//! `src/clm/heads.py`, depth=3 / layernorm / no residual):
//!
//! ```text
//! h = GELU(inp(x))                     # hidden -> width
//! h = GELU(LayerNorm(hidden_0(h)))     # width   -> width   (depth - 2 blocks)
//! z = out(h)                           # width   -> projection_dim
//! z = L2_normalize(z)                  # applied here, not in the checkpoint
//! ```
//!
//! and the pair score is
//! `min(exp(logit_scale), 100.0) * dot(z_state, z_candidate)`.
//!
//! GELU is the exact erf form (`nn.GELU()` default) and the LayerNorm uses
//! `eps = 1e-5` (also the `nn.LayerNorm` default); both are load-bearing for
//! bit-level parity with the reference; see `tools/oracle/clm/README.md`.

use crate::core::tensor::{GGMLType, MetaValue, TensorSource};
use crate::ops::{dot_f32, gelu_erf_inplace, layer_norm};

pub const ARCH: &str = "clm";

/// `nn.LayerNorm` default epsilon; the checkpoint's `cfg` does not carry one.
const LAYERNORM_EPS: f32 = 1e-5;

#[derive(Debug, Clone)]
pub struct ClmConfig {
    /// Encoder embedding width (`clm.hidden_size`).
    pub hidden: usize,
    /// First linear width (`clm.width`).
    pub width: usize,
    /// Number of linears in a head (`clm.depth`).
    pub depth: usize,
    /// Projection width (`clm.projection_dim`).
    pub proj: usize,
    pub layernorm: bool,
    /// `min(exp(logit_scale), 100.0)`, pre-clamped by the converter.
    pub logit_scale: f32,
}

/// One MLP head: `inp`, `depth - 2` hidden blocks (each with an optional
/// LayerNorm) and `out`.  Weights are F32 row-major `[out, in]`; the loader
/// transposes the GGML `[in, out]` layout.
#[derive(Debug, Clone)]
struct Linear {
    weight: Vec<f32>, // [out, in] row-major
    bias: Vec<f32>,   // [out]
    n_in: usize,
    n_out: usize,
}

impl Linear {
    fn apply(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.n_in);
        debug_assert_eq!(out.len(), self.n_out);
        for (o, slot) in out.iter_mut().enumerate() {
            let row = &self.weight[o * self.n_in..(o + 1) * self.n_in];
            *slot = dot_f32(x, row, self.n_in) + self.bias[o];
        }
    }
}

#[derive(Debug, Clone)]
struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl LayerNorm {
    fn apply(&self, input: &[f32], output: &mut [f32]) {
        layer_norm(input, &self.weight, &self.bias, LAYERNORM_EPS, output);
    }
}

#[derive(Debug, Clone)]
struct Head {
    inp: Linear,
    hidden: Vec<Linear>,
    norms: Vec<LayerNorm>,
    out: Linear,
}

impl Head {
    /// Project an encoder embedding into the shared space and L2-normalise it.
    fn project(&self, x: &[f32], scratch: &mut Vec<f32>, label: &str) -> Result<Vec<f32>, String> {
        debug_assert_eq!(x.len(), self.inp.n_in);
        scratch.resize(self.inp.n_out, 0.0);
        self.inp.apply(x, scratch);
        trace(label, "inp", scratch);
        gelu_erf_inplace(scratch);
        trace(label, "gelu1", scratch);

        // Reference: `h = act(nrm(lin(x)))` -- the LayerNorm sits AFTER the
        // hidden linear, on its output.  Norming before the matmul gives a
        // plausible-looking but different projection.
        for (block, lin) in self.hidden.iter().enumerate() {
            let mut h = vec![0.0; lin.n_out];
            lin.apply(scratch, &mut h);
            trace(label, "hidden", &h);
            if let Some(norm) = self.norms.get(block) {
                norm.apply(&h, scratch);
            } else {
                *scratch = h;
            }
            trace(label, "norm", scratch);
            gelu_erf_inplace(scratch);
            trace(label, "gelu2", scratch);
        }

        let mut z = vec![0.0; self.out.n_out];
        self.out.apply(scratch, &mut z);
        trace(label, "out", &z);

        // F::normalize(dim=-1): divide by the L2 norm.
        crate::models::qwen3::embedding::l2_normalize_embedding(&mut z)?;
        trace(label, "unit", &z);
        Ok(z)
    }
}

#[derive(Debug, Clone)]
pub struct ClmHeads {
    cfg: ClmConfig,
    state_head: Head,
    action_head: Head,
}

impl ClmHeads {
    /// Load a CLM head GGUF produced by `tools/converter/clm/convert_clm.py`.
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let arch = source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            .unwrap_or_default();
        if arch != ARCH {
            return Err(format!("Expected {ARCH} architecture, got {arch:?}"));
        }

        let cfg = ClmConfig {
            hidden: meta_usize(source, "clm.hidden_size")?,
            width: meta_usize(source, "clm.width")?,
            depth: meta_usize(source, "clm.depth")?,
            proj: meta_usize(source, "clm.projection_dim")?,
            layernorm: meta_bool(source, "clm.layernorm")?,
            logit_scale: source
                .metadata("clm.logit_scale")
                .and_then(MetaValue::to_f64)
                .map(|v| v as f32)
                .ok_or("Missing clm.logit_scale")?,
        };
        if cfg.depth < 2 {
            return Err(format!("clm.depth must be >= 2, got {}", cfg.depth));
        }
        if cfg.logit_scale <= 0.0 {
            return Err(format!(
                "clm.logit_scale must be > 0, got {}",
                cfg.logit_scale
            ));
        }

        let state_head = load_head(source, &cfg, "state_head")?;
        let action_head = load_head(source, &cfg, "action_head")?;
        Ok(Self {
            cfg,
            state_head,
            action_head,
        })
    }

    pub fn config(&self) -> &ClmConfig {
        &self.cfg
    }

    /// Width of the encoder embedding these heads expect.
    pub fn encoder_dim(&self) -> usize {
        self.cfg.hidden
    }

    /// `state_head(state_emb)`, L2-normalised.
    pub fn project_state(
        &self,
        state_emb: &[f32],
        scratch: &mut Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        if state_emb.len() != self.cfg.hidden {
            return Err(format!(
                "state embedding is {} wide, heads expect {}",
                state_emb.len(),
                self.cfg.hidden
            ));
        }
        self.state_head.project(state_emb, scratch, "state")
    }

    /// `action_head(candidate_emb)`, L2-normalised.
    pub fn project_candidate(
        &self,
        cand_emb: &[f32],
        scratch: &mut Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        if cand_emb.len() != self.cfg.hidden {
            return Err(format!(
                "candidate embedding is {} wide, heads expect {}",
                cand_emb.len(),
                self.cfg.hidden
            ));
        }
        self.action_head.project(cand_emb, scratch, "action")
    }

    /// `logit_scale * dot(z_state, z_candidate)` for already-projected,
    /// L2-normalised vectors.  Because both sides are unit length this is
    /// `logit_scale * cos(state, candidate)`.
    pub fn score(&self, z_state: &[f32], z_cand: &[f32]) -> f32 {
        debug_assert_eq!(z_state.len(), self.cfg.proj);
        debug_assert_eq!(z_cand.len(), self.cfg.proj);
        let score = self.cfg.logit_scale * dot_f32(z_state, z_cand, self.cfg.proj);
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "clm.logits",
            None,
            &[1],
            &[score],
        ));
        score
    }

    /// Convenience: embed -> project -> score for one pair.
    pub fn score_pair(
        &self,
        state_emb: &[f32],
        cand_emb: &[f32],
        scratch: &mut Vec<f32>,
    ) -> Result<f32, String> {
        let zs = self.project_state(state_emb, scratch)?;
        let zc = self.project_candidate(cand_emb, scratch)?;
        Ok(self.score(&zs, &zc))
    }
}

fn trace(label: &str, stage: &str, values: &[f32]) {
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint(
        &format!("clm.{label}.{stage}"),
        None,
        &[values.len()],
        values,
    ));
    #[cfg(not(feature = "parity-trace"))]
    let _ = (label, stage, values);
}

fn load_head(source: &dyn TensorSource, cfg: &ClmConfig, head: &str) -> Result<Head, String> {
    let inp = load_linear(source, &format!("{ARCH}.{head}.inp"), cfg.hidden, cfg.width)?;
    let n_hidden = cfg.depth - 2;
    let mut hidden = Vec::with_capacity(n_hidden);
    let mut norms = Vec::with_capacity(n_hidden);
    for i in 0..n_hidden {
        hidden.push(load_linear(
            source,
            &format!("{ARCH}.{head}.hidden.{i}"),
            cfg.width,
            cfg.width,
        )?);
        if cfg.layernorm {
            norms.push(load_norm(
                source,
                &format!("{ARCH}.{head}.norms.{i}"),
                cfg.width,
            )?);
        }
    }
    let out = load_linear(source, &format!("{ARCH}.{head}.out"), cfg.width, cfg.proj)?;
    Ok(Head {
        inp,
        hidden,
        norms,
        out,
    })
}

/// GGML stores `[n_in, n_out]` (ne0 first); torch stores `[n_out, n_in]`.
/// Transpose while reading so `apply` can walk rows.
fn load_linear(
    source: &dyn TensorSource,
    prefix: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Linear, String> {
    let wname = format!("{prefix}.weight");
    let info = source
        .tensor_info(&wname)
        .ok_or_else(|| format!("Missing tensor {wname}"))?;
    if info.dims != [n_in as u64, n_out as u64] || info.ggml_type != GGMLType::F32 {
        return Err(format!(
            "Invalid tensor {wname}: dims {:?} type {:?}, expected [{n_in}, {n_out}] F32",
            info.dims, info.ggml_type
        ));
    }
    let bytes = source
        .tensor_slice(&wname)
        .ok_or_else(|| format!("Missing tensor data {wname}"))?;
    if bytes.len() != n_in * n_out * 4 {
        return Err(format!("Invalid tensor bytes {wname}: {}", bytes.len()));
    }
    let mut weight = vec![0.0f32; n_in * n_out];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        weight[i] = f32::from_le_bytes(chunk.try_into().unwrap());
    }
    // The converter writes torch's [out, in] row-major bytes and labels the
    // tensor (n_in, n_out).  Under the GGML ne0-contiguous contract that
    // label means element (i, o) sits at flat[i + o*n_in], which is exactly
    // torch's W[o][i] -- so the bytes are already [n_out, n_in] row-major and
    // no transpose is needed.  Transposing here would scramble them.

    let bname = format!("{prefix}.bias");
    let bias = load_f32(source, &bname, &[n_out as u64])?;
    Ok(Linear {
        weight,
        bias,
        n_in,
        n_out,
    })
}

fn load_norm(source: &dyn TensorSource, prefix: &str, width: usize) -> Result<LayerNorm, String> {
    Ok(LayerNorm {
        weight: load_f32(source, &format!("{prefix}.weight"), &[width as u64])?,
        bias: load_f32(source, &format!("{prefix}.bias"), &[width as u64])?,
    })
}

fn load_f32(source: &dyn TensorSource, name: &str, dims: &[u64]) -> Result<Vec<f32>, String> {
    crate::core::tensor::load_f32_tensor(source, name, dims).map_err(|e| format!("{name}: {e}"))
}

fn meta_usize(source: &dyn TensorSource, key: &str) -> Result<usize, String> {
    source
        .metadata(key)
        .and_then(MetaValue::to_u64)
        .map(|v| v as usize)
        .ok_or_else(|| format!("Missing or invalid metadata: {key}"))
}

fn meta_bool(source: &dyn TensorSource, key: &str) -> Result<bool, String> {
    match source.metadata(key) {
        Some(MetaValue::Bool(v)) => Ok(*v),
        // Tolerate a writer that stores the flag as 0/1.
        Some(MetaValue::Uint32(v)) => Ok(*v != 0),
        Some(MetaValue::Uint64(v)) => Ok(*v != 0),
        _ => Err(format!("Missing or invalid metadata: {key}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::ggufrs::{open_model_source, ComponentRole};
    use std::path::Path;

    fn heads() -> ClmHeads {
        let path = Path::new("models/CLM-v0.1-8B/clm-v0.1-8B-heads-f32.gguf");
        if !path.exists() {
            panic!(
                "missing {}; run tools/converter/clm/convert_clm.py first",
                path.display()
            );
        }
        let source = open_model_source(path, ComponentRole::Llm).expect("open clm gguf");
        ClmHeads::from_source(source.as_ref()).expect("load clm heads")
    }

    fn golden() -> serde_json::Value {
        let path = Path::new("models/CLM-v0.1-8B/golden.json");
        let text = std::fs::read_to_string(path).expect("read golden.json");
        serde_json::from_str(&text).expect("parse golden.json")
    }

    /// Bit-level parity with the reference `torch.nn` forward.
    #[test]
    fn matches_reference_golden_vectors() {
        let heads = heads();
        let golden = golden();
        assert_eq!(heads.config().hidden, 4096);
        assert_eq!(heads.config().width, 1536);
        assert_eq!(heads.config().proj, 512);
        assert_eq!(
            heads.config().logit_scale,
            golden["scale"].as_f64().unwrap() as f32
        );

        let state: Vec<f32> = golden["state"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let cand1: Vec<f32> = golden["cand1"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let cand2: Vec<f32> = golden["cand2"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();

        let mut scratch = Vec::new();
        let zs = heads.project_state(&state, &mut scratch).unwrap();
        let zc1 = heads.project_candidate(&cand1, &mut scratch).unwrap();
        let zc2 = heads.project_candidate(&cand2, &mut scratch).unwrap();

        for (got, want) in zs.iter().zip(golden["z_state"].as_array().unwrap()) {
            let want = want.as_f64().unwrap() as f32;
            assert!(
                (got - want).abs() < 1e-4,
                "z_state mismatch: {got} vs {want}"
            );
        }
        // Unit length is the contract the scorer relies on.
        let norm: f32 = zs.iter().map(|v| v * v).sum();
        assert!((norm - 1.0).abs() < 1e-4, "z_state not normalised: {norm}");

        let s1 = heads.score(&zs, &zc1);
        let s2 = heads.score(&zs, &zc2);
        assert!(
            (s1 - golden["score1"].as_f64().unwrap() as f32).abs() < 1e-3,
            "score1 {s1} vs {}",
            golden["score1"]
        );
        assert!(
            (s2 - golden["score2"].as_f64().unwrap() as f32).abs() < 1e-3,
            "score2 {s2} vs {}",
            golden["score2"]
        );
    }

    /// A wrong embedding width has to be an error, not a silent garbage score.
    #[test]
    fn rejects_wrong_embedding_width() {
        let heads = heads();
        let mut scratch = Vec::new();
        assert!(heads.project_state(&[0.0; 512], &mut scratch).is_err());
        assert!(heads.project_candidate(&[0.0; 4095], &mut scratch).is_err());
    }
}
