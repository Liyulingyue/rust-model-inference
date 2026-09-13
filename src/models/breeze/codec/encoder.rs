use super::*;

struct Residual {
    conv1: Conv,
    conv2: Conv,
    down: Conv,
}

pub(super) struct Encoder {
    input: Conv,
    blocks: Vec<Residual>,
    output: Conv,
    transformer: Vec<TransformerLayer>,
    downsample: Conv,
    semantic_projection: Conv,
    acoustic_projection: Conv,
    semantic: Codebook,
    acoustic: Vec<Codebook>,
}

impl Encoder {
    pub(super) fn load(source: &dyn TensorSource) -> Result<Self, String> {
        let input = Conv::load(
            source,
            "encoder.encoder.layers.0.conv",
            1,
            64,
            7,
            1,
            1,
            1,
            true,
            false,
        )?;
        let mut blocks = Vec::new();
        for (index, ratio) in [4, 5, 6, 8].into_iter().enumerate() {
            let channels = 64 << index;
            let layer = 1 + index * 3;
            let prefix = format!("encoder.encoder.layers.{layer}.block");
            blocks.push(Residual {
                conv1: Conv::load(
                    source,
                    &format!("{prefix}.1.conv"),
                    channels,
                    channels / 2,
                    3,
                    1,
                    1,
                    1,
                    true,
                    false,
                )?,
                conv2: Conv::load(
                    source,
                    &format!("{prefix}.3.conv"),
                    channels / 2,
                    channels,
                    1,
                    1,
                    1,
                    1,
                    true,
                    false,
                )?,
                down: Conv::load(
                    source,
                    &format!("encoder.encoder.layers.{}.conv", layer + 2),
                    channels,
                    channels * 2,
                    ratio * 2,
                    ratio,
                    1,
                    1,
                    true,
                    false,
                )?,
            });
        }
        let transformer = (0..8)
            .map(|i| {
                TransformerLayer::load(
                    source,
                    &format!("encoder.encoder_transformer.layers.{i}"),
                    false,
                )
            })
            .collect::<Result<_, _>>()?;
        let semantic_prefix = "encoder.quantizer.semantic_residual_vector_quantizer";
        let acoustic_prefix = "encoder.quantizer.acoustic_residual_vector_quantizer";
        let acoustic = (0..15)
            .map(|i| {
                Codebook::load(
                    source,
                    &format!("{acoustic_prefix}.layers.{i}.codebook"),
                    false,
                )
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            input,
            blocks,
            transformer,
            output: Conv::load(
                source,
                "encoder.encoder.layers.14.conv",
                1024,
                DIM,
                3,
                1,
                1,
                1,
                true,
                false,
            )?,
            // Mimi explicitly overrides the SEANet padding mode here.
            downsample: Conv::load(
                source,
                "encoder.downsample.conv",
                DIM,
                DIM,
                4,
                2,
                1,
                1,
                false,
                true,
            )?,
            semantic_projection: Conv::load(
                source,
                &format!("{semantic_prefix}.input_proj"),
                DIM,
                CODE_DIM,
                1,
                1,
                1,
                1,
                false,
                false,
            )?,
            acoustic_projection: Conv::load(
                source,
                &format!("{acoustic_prefix}.input_proj"),
                DIM,
                CODE_DIM,
                1,
                1,
                1,
                1,
                false,
                false,
            )?,
            semantic: Codebook::load(
                source,
                &format!("{semantic_prefix}.layers.0.codebook"),
                false,
            )?,
            acoustic,
        })
    }

    pub(super) fn forward(&self, audio: &[f32]) -> Result<Vec<[u32; 16]>, String> {
        let mut x = self.input.forward(audio)?;
        trace("breeze.codec.encoder.conv", Some(0), &x, 64);
        for (index, block) in self.blocks.iter().enumerate() {
            let mut branch = x.clone();
            elu(&mut branch);
            branch = block.conv1.forward(&branch)?;
            elu(&mut branch);
            branch = block.conv2.forward(&branch)?;
            for (v, residual) in x.iter_mut().zip(branch) {
                *v += residual;
            }
            elu(&mut x);
            x = block.down.forward(&x)?;
            trace(
                "breeze.codec.encoder.conv",
                Some(index + 1),
                &x,
                128 << index,
            );
        }
        elu(&mut x);
        x = self.output.forward(&x)?;
        trace("breeze.codec.encoder.seanet", None, &x, DIM);
        for (index, layer) in self.transformer.iter().enumerate() {
            x = layer.forward(x, index);
            trace("breeze.codec.encoder.transformer", Some(index), &x, DIM);
        }
        x = self.downsample.forward(&x)?;
        trace("breeze.codec.encoder.downsample", None, &x, DIM);
        if x.iter().any(|v| !v.is_finite()) {
            return Err("Breeze codec encoder produced non-finite embeddings".into());
        }
        let mut frames = vec![[0; 16]; x.len() / DIM];
        let mut semantic = self.semantic_projection.forward(&x)?;
        trace(
            "breeze.codec.encoder.semantic_input",
            None,
            &semantic,
            CODE_DIM,
        );
        for (frame, id) in frames.iter_mut().zip(self.semantic.encode(&mut semantic)) {
            frame[0] = id;
        }
        // Both RVQs project the same continuous embedding. Semantic codes are
        // not subtracted from the input of the acoustic quantizer.
        let mut acoustic = self.acoustic_projection.forward(&x)?;
        trace(
            "breeze.codec.encoder.acoustic_input",
            None,
            &acoustic,
            CODE_DIM,
        );
        for (index, codebook) in self.acoustic.iter().enumerate() {
            for (frame, id) in frames.iter_mut().zip(codebook.encode(&mut acoustic)) {
                frame[index + 1] = id;
            }
        }
        Ok(frames)
    }
}
