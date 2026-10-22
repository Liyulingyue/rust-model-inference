//! BFL bfc8889: independent image/reference/text blocks followed by joint blocks.
use super::{trace, HIDDEN, TEXT_WIDTH};
use crate::core::{
    tensor::{GGMLType, TensorSource},
    thread_pool::ComputePool,
};
use crate::models::diffusion::common::{require_finite, require_tensor};
use crate::models::diffusion::dreamx::kernels::{
    attention_scalar, layer_norm_rows, load_float_values, AttentionSpec, Linear,
};
use std::sync::Arc;

const HEADS: usize = 24;
const HEAD_DIM: usize = 128;
const FFN: usize = 9216;

pub(crate) fn validate_dit(source: &dyn TensorSource) -> Result<(), String> {
    if source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        != Some("flux3")
        || !matches!(
            source.metadata("flux3.flow_map"),
            Some(crate::core::tensor::MetaValue::Bool(true))
        )
    {
        return Err("FLUX3 requires an explicitly registered distilled flow-map checkpoint".into());
    }
    let matrix =
        |name: &str, input, output| require_tensor(source, name, &[input, output], GGMLType::BF16);
    matrix("emb_in.image.weight", 128, 3072)?;
    matrix("emb_in.image_cond.weight", 128, 3072)?;
    matrix("txt_in.weight", 20480, 3072)?;
    matrix("final_layer.image.linear.weight", 3072, 128)?;
    matrix("final_layer.image.adaLN_modulation.1.weight", 3072, 6144)?;
    for time in ["time_in", "ref_time_in"] {
        matrix(&format!("{time}.in_layer.weight"), 256, 3072)?;
        matrix(&format!("{time}.out_layer.weight"), 3072, 3072)?;
    }
    for stage in ["early_stream_modulations", "single_stream_modulations"] {
        for mode in ["image", "image_cond", "txt"] {
            matrix(&format!("{stage}.{mode}.lin.weight"), 3072, 9216)?;
        }
    }
    for (stack, count) in [
        ("content_mode_blocks.image", 5),
        ("content_mode_blocks.image_cond", 5),
        ("txt_mode_blocks", 5),
        ("single_blocks", 28),
    ] {
        for layer in 0..count {
            for name in ["q_proj", "k_proj", "v_proj", "attn_out"] {
                matrix(&format!("{stack}.{layer}.{name}.weight"), 3072, 3072)?;
            }
            matrix(&format!("{stack}.{layer}.mlp_in.weight"), 3072, 18432)?;
            matrix(&format!("{stack}.{layer}.mlp_out.weight"), 9216, 3072)?;
            for name in ["query_norm", "key_norm"] {
                require_tensor(
                    source,
                    &format!("{stack}.{layer}.norm.{name}.scale"),
                    &[128],
                    GGMLType::BF16,
                )?;
            }
        }
    }
    Ok(())
}

pub(crate) struct Flux3Dit {
    source: Arc<dyn TensorSource>,
    pool: Arc<ComputePool>,
}

impl Flux3Dit {
    pub(crate) fn load(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        validate_dit(source.as_ref())?;
        Ok(Self { source, pool })
    }

    fn linear(
        &self,
        name: &str,
        input: &[f32],
        n_in: usize,
        n_out: usize,
    ) -> Result<Vec<f32>, String> {
        Linear::from_source(self.source.as_ref(), name, None, n_in, n_out)?.forward(
            &self.pool,
            input,
            input.len() / n_in,
        )
    }

    fn time(&self, prefix: &str, timestep: f32) -> Result<Vec<f32>, String> {
        let mut embedding = vec![0.0; 256];
        for i in 0..128 {
            let frequency = (-(10000.0f32).ln() * i as f32 / 128.0).exp();
            let angle = (1000.0 * timestep) * frequency;
            embedding[i] = angle.cos();
            embedding[i + 128] = angle.sin();
        }
        let mut hidden = self.linear(
            &format!("{prefix}.in_layer.weight"),
            &embedding,
            256,
            HIDDEN,
        )?;
        crate::ops::silu_inplace(&mut hidden);
        self.linear(
            &format!("{prefix}.out_layer.weight"),
            &hidden,
            HIDDEN,
            HIDDEN,
        )
    }

    fn modulation(
        &self,
        stage: &str,
        mode: &str,
        time: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>, String> {
        let mut input = time.to_vec();
        crate::ops::silu_inplace(&mut input);
        Ok(self
            .linear(
                &format!("{stage}.{mode}.lin.weight"),
                &input,
                HIDDEN,
                3 * HIDDEN,
            )?
            .repeat(rows))
    }

    fn block(
        &self,
        prefix: &str,
        input: &[f32],
        ids: &[[f32; 4]],
        modulation: &[f32],
    ) -> Result<Vec<f32>, String> {
        let rows = ids.len();
        let normalized = modulate(input, modulation, 3)?;
        let mut q = self.linear(
            &format!("{prefix}.q_proj.weight"),
            &normalized,
            HIDDEN,
            HIDDEN,
        )?;
        let mut k = self.linear(
            &format!("{prefix}.k_proj.weight"),
            &normalized,
            HIDDEN,
            HIDDEN,
        )?;
        let v = self.linear(
            &format!("{prefix}.v_proj.weight"),
            &normalized,
            HIDDEN,
            HIDDEN,
        )?;
        for (values, name) in [(&mut q, "query_norm"), (&mut k, "key_norm")] {
            let scale = load_float_values(
                self.source.as_ref(),
                &format!("{prefix}.norm.{name}.scale"),
                &[128],
            )?;
            for (token, id) in values.chunks_exact_mut(HIDDEN).zip(ids) {
                for head in token.chunks_exact_mut(HEAD_DIM) {
                    crate::ops::rms_norm_inplace(head, &scale, 1e-6);
                    apply_rope(head, *id);
                }
            }
        }
        let attn = attention_scalar(
            &q,
            &k,
            &v,
            AttentionSpec {
                query_tokens: rows,
                key_tokens: rows,
                query_heads: HEADS,
                key_value_heads: HEADS,
                head_dim: HEAD_DIM,
                causal: false,
                scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            },
        )?;
        let attn = self.linear(&format!("{prefix}.attn_out.weight"), &attn, HIDDEN, HIDDEN)?;
        let ff = self.linear(
            &format!("{prefix}.mlp_in.weight"),
            &normalized,
            HIDDEN,
            2 * FFN,
        )?;
        let mut activated = vec![0.0; rows * FFN];
        for (out, row) in activated
            .chunks_exact_mut(FFN)
            .zip(ff.chunks_exact(2 * FFN))
        {
            for i in 0..FFN {
                out[i] = crate::ops::silu(row[i]) * row[FFN + i];
            }
        }
        let ff = self.linear(&format!("{prefix}.mlp_out.weight"), &activated, FFN, HIDDEN)?;
        let mut output = input.to_vec();
        for row in 0..rows {
            for i in 0..HIDDEN {
                let j = row * HIDDEN + i;
                output[j] += modulation[row * 3 * HIDDEN + 2 * HIDDEN + i] * (attn[j] + ff[j]);
            }
        }
        require_finite(&output, prefix)?;
        trace(prefix, &[rows, HIDDEN], &output)?;
        Ok(output)
    }

    pub(crate) fn predict(
        &self,
        image: &[f32],
        image_ids: &[[f32; 4]],
        text: &[f32],
        references: &[f32],
        reference_ids: &[[f32; 4]],
        t: f32,
        target: f32,
    ) -> Result<Vec<f32>, String> {
        if image.is_empty()
            || image.len() != image_ids.len() * 128
            || text.is_empty()
            || text.len() % TEXT_WIDTH != 0
            || references.len() != reference_ids.len() * 128
            || !t.is_finite()
            || !target.is_finite()
        {
            return Err("Invalid FLUX3 prediction inputs".into());
        }
        require_finite(image, "FLUX3 image")?;
        require_finite(text, "FLUX3 text")?;
        require_finite(references, "FLUX3 references")?;
        let ref_time = self.time("ref_time_in", target)?;
        let add_ref = |mut time: Vec<f32>| {
            for (v, r) in time.iter_mut().zip(&ref_time) {
                *v += r;
            }
            time
        };
        let image_time = add_ref(self.time("time_in", t)?);
        let context_time = add_ref(self.time("time_in", 0.0)?);
        let text_ids: Vec<_> = (0..text.len() / TEXT_WIDTH)
            .map(|i| [0.0, 0.0, 0.0, i as f32])
            .collect();
        let mut streams = Vec::new();
        for (mode, prefix, projection, values, ids, width, time) in [
            (
                "image",
                "content_mode_blocks.image",
                "emb_in.image.weight",
                image,
                image_ids,
                128,
                &image_time,
            ),
            (
                "image_cond",
                "content_mode_blocks.image_cond",
                "emb_in.image_cond.weight",
                references,
                reference_ids,
                128,
                &context_time,
            ),
            (
                "txt",
                "txt_mode_blocks",
                "txt_in.weight",
                text,
                text_ids.as_slice(),
                TEXT_WIDTH,
                &context_time,
            ),
        ] {
            if ids.is_empty() {
                continue;
            } // Empty reference phantom tokens never participate in attention.
            let hidden = self.linear(projection, values, width, HIDDEN)?;
            trace(
                &format!("flux3.{mode}.input"),
                &[ids.len(), HIDDEN],
                &hidden,
            )?;
            let early = self.modulation("early_stream_modulations", mode, time, ids.len())?;
            let single = self.modulation("single_stream_modulations", mode, time, ids.len())?;
            streams.push((prefix, hidden, ids.to_vec(), early, single));
        }
        for layer in 0..5 {
            for (prefix, hidden, ids, early, _) in &mut streams {
                *hidden = self.block(&format!("{prefix}.{layer}"), hidden, ids, early)?;
            }
        }
        let mut hidden = Vec::new();
        let mut ids = Vec::new();
        let mut single = Vec::new();
        for (_, values, positions, _, modulation) in streams {
            hidden.extend(values);
            ids.extend(positions);
            single.extend(modulation);
        }
        for layer in 0..28 {
            hidden = self.block(&format!("single_blocks.{layer}"), &hidden, &ids, &single)?;
        }
        let mut time = image_time;
        crate::ops::silu_inplace(&mut time);
        let modulation = self
            .linear(
                "final_layer.image.adaLN_modulation.1.weight",
                &time,
                HIDDEN,
                2 * HIDDEN,
            )?
            .repeat(image_ids.len());
        let image = modulate(&hidden[..image_ids.len() * HIDDEN], &modulation, 2)?;
        let velocity = self.linear("final_layer.image.linear.weight", &image, HIDDEN, 128)?;
        require_finite(&velocity, "FLUX3 velocity")?;
        trace("flux3.velocity", &[image_ids.len(), 128], &velocity)?;
        Ok(velocity)
    }
}

fn modulate(input: &[f32], modulation: &[f32], parts: usize) -> Result<Vec<f32>, String> {
    let rows = input.len() / HIDDEN;
    let mut output = layer_norm_rows(input, rows, Some(&vec![1.0; HIDDEN]), None, 1e-6)?;
    for row in 0..rows {
        for i in 0..HIDDEN {
            output[row * HIDDEN + i] = (1.0 + modulation[row * parts * HIDDEN + HIDDEN + i])
                * output[row * HIDDEN + i]
                + modulation[row * parts * HIDDEN + i];
        }
    }
    Ok(output)
}

fn apply_rope(head: &mut [f32], id: [f32; 4]) {
    for (axis, position) in id.into_iter().enumerate() {
        for pair in 0..16 {
            let angle = position as f64 / 10000.0f64.powf((2 * pair) as f64 / 32.0);
            let (sin, cos) = angle.sin_cos();
            let (sin, cos) = (sin as f32, cos as f32);
            let j = axis * 32 + pair * 2;
            let (a, b) = (head[j], head[j + 1]);
            head[j] = a * cos + b * -sin;
            head[j + 1] = a * sin + b * cos;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{MetaValue, TensorInfo};

    struct Contract {
        arch: MetaValue,
        flow: MetaValue,
        first: TensorInfo,
        bytes: Vec<u8>,
    }
    impl TensorSource for Contract {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            match key {
                "general.architecture" => Some(&self.arch),
                "flux3.flow_map" => Some(&self.flow),
                _ => None,
            }
        }
        fn tensor_info(&self, key: &str) -> Option<&TensorInfo> {
            (key == self.first.name).then_some(&self.first)
        }
        fn tensor_slice(&self, key: &str) -> Option<&[u8]> {
            (key == self.first.name).then_some(self.bytes.as_slice())
        }
    }
    #[test]
    fn explicit_architecture_flow_map_shape_dtype_and_bytes_are_required() {
        let mut source = Contract {
            arch: MetaValue::String("flux".into()),
            flow: MetaValue::Bool(true),
            first: TensorInfo {
                name: "emb_in.image.weight".into(),
                dims: vec![128, 3072],
                ggml_type: GGMLType::BF16,
                offset: 0,
            },
            bytes: vec![0; 128 * 3072 * 2],
        };
        assert!(validate_dit(&source)
            .unwrap_err()
            .contains("explicitly registered"));
        source.arch = MetaValue::String("flux3".into());
        source.flow = MetaValue::Bool(false);
        assert!(validate_dit(&source).unwrap_err().contains("flow-map"));
        source.flow = MetaValue::Bool(true);
        source.first.dims[0] = 129;
        assert!(validate_dit(&source).unwrap_err().contains("dimensions"));
        source.first.dims[0] = 128;
        source.first.ggml_type = GGMLType::F16;
        assert!(validate_dit(&source).unwrap_err().contains("type"));
        source.first.ggml_type = GGMLType::BF16;
        source.bytes.pop();
        assert!(validate_dit(&source).unwrap_err().contains("byte length"));
    }
}
