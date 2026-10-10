use crate::core::tensor::TensorSource;

/// Vision hyper-parameters for the `nemotron_v2_vl` projector, which is what
/// ZDTaichu5.0-9B ships.
///
/// These keys come straight from the GGUF metadata of
/// `mmproj-ZDTaichu5.0-9B-BF16.gguf`. The upstream reference reads them in
/// `clip.cpp` (`PROJECTOR_TYPE_NEMOTRON_V2_VL`) and builds the graph in
/// `tools/mtmd/models/nemotron-v2-vl.cpp`.
#[derive(Debug, Clone)]
pub struct ZdtVisionConfig {
    pub projection_dim: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_layer: usize,
    pub n_head: usize,
    /// `clip.vision.projector.scale_factor`; drives the patch merge step.
    pub scale_factor: usize,
    /// Width of the projector's hidden layer, read from
    /// `mm.model.mlp.1.weight` because the metadata does not carry it.
    pub projector_hidden: usize,
    pub eps: f32,
    pub use_gelu: bool,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl ZdtVisionConfig {
    pub fn from_source<S: TensorSource + ?Sized>(source: &S) -> Result<Self, String> {
        let get_u32 = |key: &str| -> Result<usize, String> {
            source
                .metadata(key)
                .and_then(|v| v.to_u64())
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| format!("Missing clip metadata: {key}"))
        };
        let get_f32 = |key: &str| -> Result<f32, String> {
            source
                .metadata(key)
                .and_then(|v| v.to_f64())
                .map(|v| v as f32)
                .ok_or_else(|| format!("Missing clip metadata: {key}"))
        };
        let get_bool = |key: &str| -> bool {
            source
                .metadata(key)
                .and_then(|v| match v {
                    crate::core::tensor::MetaValue::Bool(b) => Some(*b),
                    _ => None,
                })
                .unwrap_or(false)
        };

        let projection_dim = get_u32("clip.vision.projection_dim")?;
        let image_size = get_u32("clip.vision.image_size")?;
        let patch_size = get_u32("clip.vision.patch_size")?;
        let n_embd = get_u32("clip.vision.embedding_length")?;
        let n_ff = get_u32("clip.vision.feed_forward_length")?;
        let n_layer = get_u32("clip.vision.block_count")?;
        let n_head = get_u32("clip.vision.attention.head_count")?;
        let scale_factor = source
            .metadata("clip.vision.projector.scale_factor")
            .and_then(|v| v.to_u64())
            .and_then(|v| usize::try_from(v).ok())
            .ok_or("Missing clip metadata: clip.vision.projector.scale_factor")?;

        if n_embd % n_head != 0 {
            return Err(format!(
                "vision embedding_length {n_embd} is not divisible by attention.head_count {n_head}"
            ));
        }
        if scale_factor < 2 {
            return Err(format!(
                "projector.scale_factor must be >= 2, got {scale_factor}"
            ));
        }

        let image_mean = read_triplet(source, "clip.vision.image_mean")?;
        let image_std = read_triplet(source, "clip.vision.image_std")?;

        Ok(Self {
            projection_dim,
            image_size,
            patch_size,
            n_embd,
            n_ff,
            n_layer,
            n_head,
            scale_factor,
            projector_hidden: 0,
            eps: get_f32("clip.vision.attention.layer_norm_epsilon")?,
            use_gelu: get_bool("clip.use_gelu"),
            image_mean,
            image_std,
        })
    }

    pub fn d_head(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// Patches along one axis for the fixed-size preprocessor.
    pub fn grid(&self) -> usize {
        self.image_size / self.patch_size
    }
}

fn read_triplet<S: TensorSource + ?Sized>(source: &S, key: &str) -> Result<[f32; 3], String> {
    let raw = source
        .metadata(key)
        .and_then(|v| match v {
            crate::core::tensor::MetaValue::Float32(a) => Some(vec![*a]),
            crate::core::tensor::MetaValue::Float64(a) => Some(vec![*a as f32]),
            crate::core::tensor::MetaValue::Array(_, values) => Some(
                values
                    .iter()
                    .map(|v| v.to_f64().unwrap_or(0.0) as f32)
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .ok_or_else(|| format!("Missing clip metadata: {key}"))?;
    if raw.len() != 3 {
        return Err(format!("{key} must hold 3 values, got {}", raw.len()));
    }
    Ok([raw[0] as f32, raw[1] as f32, raw[2] as f32])
}
