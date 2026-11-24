use super::{Block, D, FF, HEAD_DIM, PIXELS, REGISTERS, TEXT_LEN, VOCAB};
use crate::core::tensor::TensorSource;
use crate::ops::nn::{load_float_values, Linear};

/// Projection slot in the executor's ordered list of immutable GGUF weights.
#[derive(Clone, Copy)]
pub(super) struct Projection(pub(super) usize);

pub struct Weights<'a> {
    pub(super) linears: Vec<Linear<'a>>,
    pub(super) color_mlp: [Projection; 2],
    pub(super) pool_mlp: [Projection; 2],
    pub(super) ratio_mlp: [Projection; 2],
    pub(super) text_proj: Projection,
    pub(super) self_cond: Projection,
    pub(super) modulation: Projection,
    pub(super) down: Projection,
    pub(super) up: Projection,
    pub(super) final_mod: Projection,
    pub(super) head: Projection,
    pub(super) stages: [Vec<Block>; 3],
    pub(super) positions: Vec<f32>,
    pub(super) embedding: Vec<f32>,
    pub(super) lab: Vec<f32>,
    rgb: Vec<[u8; 3]>,
    pub(super) registers: Vec<f32>,
    pub(super) null_text: Vec<f32>,
    pub(super) null_cond: Vec<f32>,
    pub(super) text_norm: Vec<f32>,
    pub(super) pool_norm: Vec<f32>,
    pub(super) down_norm: Vec<f32>,
}

struct Loader<'a> {
    source: &'a dyn TensorSource,
    linears: Vec<Linear<'a>>,
}

impl Loader<'_> {
    fn projection(
        &mut self,
        name: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Projection, String> {
        let weight = format!("{name}.weight");
        let bias = bias.then(|| format!("{name}.bias"));
        let linear = Linear::from_source(self.source, &weight, bias.as_deref(), input, output)?;
        let slot = self.linears.len();
        self.linears.push(linear);
        Ok(Projection(slot))
    }

    fn values(&self, name: &str, dims: &[usize]) -> Result<Vec<f32>, String> {
        let dims = dims.iter().map(|&d| d as u64).collect::<Vec<_>>();
        let values = load_float_values(self.source, name, &dims)?;
        if values.iter().any(|x| !x.is_finite()) {
            return Err(format!("Non-finite Tessera tensor {name}"));
        }
        Ok(values)
    }

    fn blocks(&mut self, stage: &str, depth: usize) -> Result<Vec<Block>, String> {
        (0..depth)
            .map(|i| {
                let base = format!("blocks_{stage}.{i}");
                let kernel = self.values(&format!("{base}.mlp.dw.weight"), &[3, 3, 1, FF])?;
                // Pixel activations are channels-last. Place adjacent channel weights
                // together so every tap is a contiguous multiply-add.
                let kernel = (0..9 * FF).map(|i| kernel[(i % FF) * 9 + i / FF]).collect();
                Ok(Block {
                    qkv: self.projection(&format!("{base}.attn.qkv"), D, 3 * D, false)?,
                    proj: self.projection(&format!("{base}.attn.proj"), D, D, false)?,
                    w12: self.projection(&format!("{base}.mlp.w12"), D, 2 * FF, false)?,
                    w3: self.projection(&format!("{base}.mlp.w3"), FF, D, false)?,
                    q_norm: self.values(&format!("{base}.attn.q_norm.weight"), &[HEAD_DIM])?,
                    k_norm: self.values(&format!("{base}.attn.k_norm.weight"), &[HEAD_DIM])?,
                    kernel,
                    bias: self.values(&format!("{base}.mlp.dw.bias"), &[FF])?,
                    mod_bias: self.values(&format!("{base}.mod_bias"), &[D, 6])?,
                })
            })
            .collect()
    }
}

impl<'a> Weights<'a> {
    pub fn load(source: &'a dyn TensorSource) -> Result<Self, String> {
        let mut load = Loader {
            source,
            linears: Vec::new(),
        };
        let raw_rgb = load.values("palette_rgb", &[3, VOCAB])?;
        if raw_rgb
            .iter()
            .any(|x| !(0.0..=255.0).contains(x) || x.fract() != 0.0)
        {
            return Err("Invalid Tessera RGB palette".into());
        }
        let rgb = raw_rgb
            .chunks_exact(3)
            .map(|p| [p[0] as u8, p[1] as u8, p[2] as u8])
            .collect();
        Ok(Self {
            color_mlp: [
                load.projection("color_mlp.0", 40, D, true)?,
                load.projection("color_mlp.2", D, D, true)?,
            ],
            pool_mlp: [
                load.projection("pool_mlp.1", D, D, true)?,
                load.projection("pool_mlp.3", D, D, true)?,
            ],
            ratio_mlp: [
                load.projection("ratio_mlp.0", 16, D, true)?,
                load.projection("ratio_mlp.2", D, D, true)?,
            ],
            text_proj: load.projection("text_proj", D, D, true)?,
            self_cond: load.projection("sc_proj", VOCAB, D, false)?,
            modulation: load.projection("t_block.1", D, 6 * D, true)?,
            down: load.projection("down.1", 4 * D, D, true)?,
            up: load.projection("up", D, 4 * D, true)?,
            final_mod: load.projection("final_mod.1", D, 2 * D, true)?,
            head: load.projection("head", D, VOCAB, true)?,
            stages: [
                load.blocks("in", 2)?,
                load.blocks("core", 8)?,
                load.blocks("out", 2)?,
            ],
            positions: load.values("pos_emb", &[D, PIXELS, 1])?,
            embedding: load.values("tok_emb.weight", &[D, VOCAB + 1])?,
            lab: load.values("palette_lab", &[3, VOCAB])?,
            rgb,
            registers: load.values("registers", &[D, REGISTERS, 1])?,
            null_text: load.values("null_text", &[D, TEXT_LEN, 1])?,
            null_cond: load.values("null_cond", &[D, 1])?,
            text_norm: load.values("text_norm.weight", &[D])?,
            pool_norm: load.values("pool_mlp.0.weight", &[D])?,
            down_norm: load.values("down.0.weight", &[4 * D])?,
            linears: load.linears,
        })
    }

    pub fn palette_rgb(&self) -> &[[u8; 3]] {
        &self.rgb
    }
}
