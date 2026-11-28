//! The release checkpoint's Hugging Face YaRN frequencies and interleaved axes.
use serde_json::Value;

#[derive(Clone)]
pub(crate) struct YarnRope {
    pub(crate) inv_freq: Vec<f32>,
    pub(crate) attention_factor: f32,
}

impl YarnRope {
    pub(crate) fn from_config(text: &Value, dim: usize, base: f32) -> Result<Option<Self>, String> {
        let rope = &text["rope_parameters"];
        if rope["rope_type"] == "default" {
            return Ok(None);
        }
        let factor = rope["factor"].as_f64().unwrap_or(f64::NAN);
        let original = rope["original_max_position_embeddings"]
            .as_u64()
            .unwrap_or(0);
        if rope["rope_type"] != "yarn"
            || !factor.is_finite()
            || factor < 1.0
            || original == 0
            || !base.is_finite()
            || base <= 1.0
            || dim == 0
            || dim % 2 != 0
            || rope.as_object().is_none_or(|fields| {
                fields.keys().any(|key| {
                    !matches!(
                        key.as_str(),
                        "rope_type"
                            | "rope_theta"
                            | "partial_rotary_factor"
                            | "mrope_interleaved"
                            | "mrope_section"
                            | "factor"
                            | "original_max_position_embeddings"
                    )
                })
            })
        {
            return Err("Unsupported SenseNova YaRN parameters".into());
        }
        let correction = |rotations: f64| {
            dim as f64 * (original as f64 / (rotations * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * (base as f64).ln())
        };
        let low = correction(32.0).floor().max(0.0) as f32;
        let high = correction(1.0).ceil().min(dim as f64 - 1.0) as f32;
        let inv_freq = (0..dim / 2)
            .map(|i| {
                let frequency = base.powf((2 * i) as f32 / dim as f32);
                let extrapolation =
                    1.0 - ((i as f32 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
                (1.0 / (factor as f32 * frequency)) * (1.0 - extrapolation)
                    + (1.0 / frequency) * extrapolation
            })
            .collect();
        Ok(Some(Self {
            inv_freq,
            attention_factor: (1.0 + 0.1 * factor.ln()) as f32,
        }))
    }

    pub(crate) fn fill_coefficients(
        &self,
        out: &mut [f32],
        positions: [usize; 4],
        sections: [i32; 4],
    ) {
        let half = self.inv_freq.len();
        assert_eq!(out.len(), 2 * half);
        for (i, &frequency) in self.inv_freq.iter().enumerate() {
            let axis = if i % 3 == 1 && i < 3 * sections[1] as usize {
                1
            } else if i % 3 == 2 && i < 3 * sections[2] as usize {
                2
            } else {
                0
            };
            let (cos, sin) = crate::ops::rope::rope_sin_cos(positions[axis] as f32 * frequency);
            out[i] = cos * self.attention_factor;
            out[i + half] = sin * self.attention_factor;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_yarn_matches_pinned_transformers_frequencies_and_magnitude() {
        let config: Value = serde_json::from_str(include_str!(
            "../../../tools/oracle/sensenova_loop/release-config.json"
        ))
        .unwrap();
        let expected: Value = serde_json::from_str(include_str!(
            "../../../tools/oracle/sensenova_loop/release-yarn.json"
        ))
        .unwrap();
        let rope = YarnRope::from_config(&config["text_config"], 64, 10_000_000.0)
            .unwrap()
            .unwrap();
        for (&actual, expected) in rope
            .inv_freq
            .iter()
            .zip(expected["inv_freq"].as_array().unwrap())
        {
            let expected = expected.as_f64().unwrap() as f32;
            assert!((actual - expected).abs() <= 2e-7 * expected.abs());
        }
        assert!(
            (rope.attention_factor as f64 - expected["attention_factor"].as_f64().unwrap()).abs()
                < 1e-7
        );
        let mut coefficients = vec![0.0; 64];
        rope.fill_coefficients(&mut coefficients, [0; 4], [11, 11, 10, 0]);
        assert!(coefficients[..32]
            .iter()
            .all(|&v| v == rope.attention_factor));
        assert!(coefficients[32..].iter().all(|&v| v == 0.0));
        rope.fill_coefficients(&mut coefficients, [7, 11, 13, 0], [11, 11, 10, 0]);
        for (i, position) in [(0, 7.0), (1, 11.0), (2, 13.0), (31, 11.0)] {
            let expected = (position * rope.inv_freq[i]).cos() * rope.attention_factor;
            assert!((coefficients[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn preview_rope_is_unchanged_and_invalid_yarn_is_rejected() {
        let preview: Value = serde_json::from_str(include_str!(
            "../../../tools/oracle/sensenova_loop/config.json"
        ))
        .unwrap();
        assert!(
            YarnRope::from_config(&preview["text_config"], 64, 10_000_000.0)
                .unwrap()
                .is_none()
        );
        let release: Value = serde_json::from_str(include_str!(
            "../../../tools/oracle/sensenova_loop/release-config.json"
        ))
        .unwrap();
        for (key, value) in [
            ("factor", serde_json::json!(0)),
            ("original_max_position_embeddings", serde_json::json!(true)),
            ("attention_factor", serde_json::json!(2.0)),
        ] {
            let mut text = release["text_config"].clone();
            text["rope_parameters"][key] = value;
            assert!(YarnRope::from_config(&text, 64, 10_000_000.0).is_err());
        }
    }
}
