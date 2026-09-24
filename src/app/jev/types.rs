pub struct JevQuestionInput {
    pub text: String,
    pub options: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JevMode {
    Choice,
    Binary,
    Score,
    MultiSelect,
    BlockChoice,
}

#[derive(Clone, Debug)]
pub struct JevResult {
    pub mode: JevMode,
    pub question: String,
    pub labels: Vec<char>,
    pub descriptions: Vec<String>,
    pub values: Vec<f32>,
    pub probabilities: Vec<f32>,
    pub choice_label: Option<char>,
    pub positive_label: Option<char>,
    pub probability_positive: Option<f32>,
    pub score: Option<f32>,
    pub confidence: f32,
    pub entropy: f32,
    pub margin: f32,
    pub prefill_ms: u128,
}

// ---------------------------------------------------------------------------
// Grouped JEV: MultiSelect + BlockChoice
//
// Fully isolated from Choice/Binary/Score above. Uses per-group softmax
// (groups normalized independently) instead of global softmax. This avoids
// cross-group probability contamination: a high-scoring candidate in group 1
// does not suppress probabilities in group 2.
//
// - MultiSelect: each pair of options forms a binary group. Per-group softmax
//   gives independent yes/no probability per item.
// - BlockChoice: user explicitly defines blocks; each block is a group with
//   its own independent softmax.
//
// The forward pass is identical (single prefill, read last-token logits).
// Only the payload construction and post-processing differ.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct JevGroupedOption {
    pub description: String,
}

#[derive(Clone, Debug)]
pub struct JevGroupInput {
    pub label: String,
    pub options: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct JevGroupedQuestionInput {
    pub text: String,
    pub groups: Vec<JevGroupInput>,
}

pub(crate) struct PreparedGroup {
    pub(crate) label: String,
    pub(crate) descriptions: Vec<String>,
    pub(crate) values: Vec<f32>,
}

pub(crate) struct PreparedGroupedQuestion {
    pub(crate) mode: JevMode,
    pub(crate) text: String,
    pub(crate) groups: Vec<PreparedGroup>,
}

#[derive(Clone, Debug)]
pub struct JevGroupResult {
    pub label: String,
    pub labels: Vec<char>,
    pub descriptions: Vec<String>,
    pub values: Vec<f32>,
    pub probabilities: Vec<f32>,
    pub choice_label: char,
    pub score: Option<f32>,
    pub confidence: f32,
    pub entropy: f32,
    pub margin: f32,
}

#[derive(Clone, Debug)]
pub struct JevGroupedResult {
    pub mode: JevMode,
    pub question: String,
    pub groups: Vec<JevGroupResult>,
    pub prefill_ms: u128,
}

impl serde::Serialize for JevGroupedResult {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("JevGroupedResult", 4)?;
        st.serialize_field(
            "mode",
            match self.mode {
                JevMode::MultiSelect => "multi_select",
                JevMode::BlockChoice => "block_choice",
                _ => "unknown",
            },
        )?;
        st.serialize_field("question", &self.question)?;
        let groups: Vec<serde_json::Value> = self
            .groups
            .iter()
            .map(|g| {
                let probs: serde_json::Map<String, serde_json::Value> = g
                    .labels
                    .iter()
                    .zip(g.probabilities.iter())
                    .map(|(l, p)| (l.to_string(), serde_json::json!(p)))
                    .collect();
                let mut obj = serde_json::json!({
                    "label": g.label,
                    "choice": g.choice_label.to_string(),
                    "probabilities": probs,
                    "confidence": g.confidence,
                    "entropy": g.entropy,
                    "margin": g.margin,
                });
                if let Some(score) = g.score {
                    obj.as_object_mut()
                        .unwrap()
                        .insert("score".to_string(), serde_json::json!(score));
                }
                obj
            })
            .collect();
        st.serialize_field("groups", &groups)?;
        st.serialize_field("prefill_ms", &self.prefill_ms)?;
        st.end()
    }
}

impl serde::Serialize for JevResult {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("JevResult", 10)?;
        st.serialize_field(
            "mode",
            match self.mode {
                JevMode::Choice => "choice",
                JevMode::Binary => "binary",
                JevMode::Score => "score",
                JevMode::MultiSelect => "multi_select",
                JevMode::BlockChoice => "block_choice",
            },
        )?;
        st.serialize_field("question", &self.question)?;
        let labels_str: Vec<String> = self.labels.iter().map(|c| c.to_string()).collect();
        st.serialize_field("labels", &labels_str)?;
        st.serialize_field("descriptions", &self.descriptions)?;
        if self.mode == JevMode::Score {
            st.serialize_field("values", &self.values)?;
        }
        let mut probs = serde_json::Map::new();
        for (i, p) in self.probabilities.iter().enumerate() {
            probs.insert(self.labels[i].to_string(), serde_json::json!(p));
        }
        st.serialize_field("probabilities", &probs)?;
        if self.mode == JevMode::Choice {
            st.serialize_field("choice", &self.choice_label.map(|c| c.to_string()))?;
        }
        if self.mode == JevMode::Binary {
            st.serialize_field("choice", &self.choice_label.map(|c| c.to_string()))?;
            st.serialize_field("positive", &self.positive_label.map(|c| c.to_string()))?;
            st.serialize_field("probability", &self.probability_positive)?;
        }
        if self.mode == JevMode::Score {
            st.serialize_field("score", &self.score)?;
        }
        st.serialize_field("confidence", &self.confidence)?;
        st.serialize_field("entropy", &self.entropy)?;
        st.serialize_field("margin", &self.margin)?;
        st.serialize_field("prefill_ms", &self.prefill_ms)?;
        st.end()
    }
}
