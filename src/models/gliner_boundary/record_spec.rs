//! `record_metadata` normalization and `RecordSpec` compilation.
//!
//! Mirrors `gliner2.processing.records`
//! (`target/gliner2-oracle/gliner2/processing/records.py`).
//!
//! Records are not a separate schema family: `RECORD_TASK_TYPES` is
//! `("json_structures",)`, so a `json_structures` group *with* a `mode` in its
//! `record_metadata` compiles to a record spec, and one *without* keeps the
//! legacy structure path. "Unannotated" is therefore a meaningful state rather
//! than a default, and it is the one case here that is a silent no-op.
//!
//! Everything in this module is pure — a metadata mapping plus a query layout in,
//! specs out — so it is fully unit-testable without a model.

use std::collections::BTreeMap;

/// How many mentions a field may bind within one record instance.
///
/// `FieldCardinality`. `is_scalar` and `allows_absent` are both consulted by the
/// decoder: scalars go through the assignment solver, lists do not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldCardinality {
    /// 0 or 1. Scalar, may be ABSENT.
    OptionalOne,
    /// Exactly 1. Scalar, never ABSENT.
    RequiredOne,
    /// List, may be empty.
    ZeroOrMore,
    /// List, at least one.
    OneOrMore,
}

impl FieldCardinality {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OptionalOne => "optional_one",
            Self::RequiredOne => "required_one",
            Self::ZeroOrMore => "zero_or_more",
            Self::OneOrMore => "one_or_more",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "optional_one" => Some(Self::OptionalOne),
            "required_one" => Some(Self::RequiredOne),
            "zero_or_more" => Some(Self::ZeroOrMore),
            "one_or_more" => Some(Self::OneOrMore),
            _ => None,
        }
    }

    /// `optional_one` and `required_one` are scalars; the rest are lists.
    pub fn is_scalar(self) -> bool {
        matches!(self, Self::OptionalOne | Self::RequiredOne)
    }

    /// `optional_one` and `zero_or_more` may bind nothing. This is what lets the
    /// decoder's solver add an ABSENT column at all.
    pub fn allows_absent(self) -> bool {
        matches!(self, Self::OptionalOne | Self::ZeroOrMore)
    }
}

/// `_default_cardinality` (`records.py:57`).
///
/// The anchor is always `required_one`: an instance without its anchor is not an
/// instance. Otherwise a `str` dtype means an optional scalar and anything else
/// falls through to a list.
pub fn default_cardinality(dtype: Option<&str>, is_anchor: bool) -> FieldCardinality {
    if is_anchor {
        return FieldCardinality::RequiredOne;
    }
    if dtype == Some("str") {
        return FieldCardinality::OptionalOne;
    }
    FieldCardinality::ZeroOrMore
}

/// One field of a record group, bound to its boundary query id.
#[derive(Clone, Debug, PartialEq)]
pub struct RecordFieldSpec {
    /// The extractive query id — the same id space relations use, so a mixed
    /// schema has to agree with the prompt's group order.
    pub query_id: usize,
    pub name: String,
    pub role_index: usize,
    pub cardinality: FieldCardinality,
    pub is_anchor: bool,
    /// A mention bound here cannot bind elsewhere. Exclusive fields are the ones
    /// that go through the assignment solver.
    pub exclusive: bool,
}

/// A record/event schema group compiled against a concrete query layout.
#[derive(Clone, Debug, PartialEq)]
pub struct RecordSpec {
    pub task_index: usize,
    pub task_name: String,
    pub task_type: String,
    /// `natural` | `latent` | `anchorless`.
    pub mode: String,
    pub fields: Vec<RecordFieldSpec>,
    /// Only ever `Some` for `natural`.
    pub anchor_query_id: Option<usize>,
    pub occurrence_policy: String,
}

pub const VALID_MODES: [&str; 3] = ["natural", "latent", "anchorless"];
pub const VALID_OCCURRENCE_POLICIES: [&str; 4] =
    ["all", "first", "error_on_ambiguous", "latent_all"];

/// The validated form of one group's `record_metadata` entry.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedRecordConfig {
    pub mode: String,
    pub anchor: Option<String>,
    pub occurrence_policy: String,
    /// Keyed by field name, cardinality resolved and `exclusive` defaulted.
    pub fields: BTreeMap<String, (FieldCardinality, bool)>,
}

/// `normalize_record_metadata` (`records.py:142`).
///
/// A group with no `mode` is **skipped**, not defaulted — the reference is
/// explicit that "this function never invents a mode for groups the user did not
/// annotate". That is the one silent path in this module, which is why the
/// fixture has a case for it.
pub fn normalize_record_metadata(
    raw: &serde_json::Value,
    field_dtypes: &BTreeMap<String, BTreeMap<String, String>>,
) -> Result<BTreeMap<String, NormalizedRecordConfig>, String> {
    let Some(object) = raw.as_object() else {
        if raw.is_null() {
            return Ok(BTreeMap::new());
        }
        return Err("record_metadata must be a mapping".into());
    };
    let mut out = BTreeMap::new();
    for (name, config) in object {
        let config = config
            .as_object()
            .ok_or_else(|| format!("record_metadata['{name}'] must be a mapping"))?;
        let Some(mode) = config.get("mode").and_then(|v| v.as_str()) else {
            // Unannotated -> legacy. Not an error, and not a default.
            continue;
        };
        if !VALID_MODES.contains(&mode) {
            return Err(format!(
                "record_metadata['{name}'].mode must be one of {VALID_MODES:?}, got {mode:?}"
            ));
        }
        let anchor = config.get("anchor").and_then(|v| v.as_str());
        if mode == "natural" && anchor.is_none() {
            return Err(format!(
                "record_metadata['{name}'] mode='natural' requires 'anchor'"
            ));
        }
        if mode != "natural" && anchor.is_some() {
            return Err(format!(
                "record_metadata['{name}'] mode={mode:?} must not set 'anchor'"
            ));
        }
        let policy = config
            .get("occurrence_policy")
            .and_then(|v| v.as_str())
            .unwrap_or("latent_all");
        if !VALID_OCCURRENCE_POLICIES.contains(&policy) {
            return Err(format!(
                "record_metadata['{name}'].occurrence_policy must be one of \
                 {VALID_OCCURRENCE_POLICIES:?}, got {policy:?}"
            ));
        }
        let dtypes = field_dtypes.get(name);
        let mut fields = BTreeMap::new();
        if let Some(declared) = config.get("fields").and_then(|v| v.as_object()) {
            for (field_name, field_config) in declared {
                let empty = serde_json::Map::new();
                let field_config = field_config.as_object().unwrap_or(&empty);
                let is_anchor = anchor == Some(field_name.as_str());
                let cardinality = match field_config.get("cardinality").and_then(|v| v.as_str()) {
                    Some(text) => FieldCardinality::parse(text).ok_or_else(|| {
                        format!(
                            "record_metadata['{name}'].fields['{field_name}'].cardinality invalid: {text:?}"
                        )
                    })?,
                    None => default_cardinality(
                        dtypes
                            .and_then(|map| map.get(field_name))
                            .map(String::as_str),
                        is_anchor,
                    ),
                };
                fields.insert(
                    field_name.clone(),
                    (
                        cardinality,
                        field_config
                            .get("exclusive")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false),
                    ),
                );
            }
        }
        out.insert(
            name.clone(),
            NormalizedRecordConfig {
                mode: mode.to_string(),
                anchor: anchor.map(str::to_string),
                occurrence_policy: policy.to_string(),
                fields,
            },
        );
    }
    Ok(out)
}

/// One query in a concrete layout — the extractive side of `QuerySpec`.
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutQuery {
    pub query_id: usize,
    pub task_index: usize,
    pub task_type: String,
    pub task_name: String,
    pub role_index: usize,
    pub role_name: String,
}

/// `compile_record_specs` (`records.py:219`).
///
/// Only `json_structures` groups with a mode compile. Fields are sorted by
/// `role_index` so the result is right even when a caller reorders the schema,
/// and each field's `query_id` comes from the layout — the same id space the
/// relations head reads as head/tail slots.
pub fn compile_record_specs(
    queries: &[LayoutQuery],
    record_metadata: &serde_json::Value,
    field_dtypes: &BTreeMap<String, BTreeMap<String, String>>,
) -> Result<BTreeMap<usize, RecordSpec>, String> {
    let normalized = normalize_record_metadata(record_metadata, field_dtypes)?;

    // Group the layout's extractive queries by task, keeping their order.
    let mut by_task: BTreeMap<usize, Vec<&LayoutQuery>> = BTreeMap::new();
    for query in queries {
        by_task.entry(query.task_index).or_default().push(query);
    }

    let mut specs = BTreeMap::new();
    for (task_index, mut group) in by_task {
        let Some(first) = group.first() else {
            continue;
        };
        let task_name = first.task_name.clone();
        let task_type = first.task_type.clone();
        if task_type != "json_structures" {
            continue;
        }
        let Some(config) = normalized.get(&task_name) else {
            continue;
        };
        group.sort_by_key(|query| query.role_index);

        let mut fields = Vec::with_capacity(group.len());
        let mut anchor_query_id = None;
        for query in &group {
            let is_anchor =
                config.mode == "natural" && config.anchor.as_deref() == Some(&query.role_name);
            // An explicit `fields` entry wins; otherwise fall back to the dtype
            // default, which is what `compile_record_specs` does for a field the
            // metadata did not mention.
            let (cardinality, exclusive) = match config.fields.get(&query.role_name) {
                Some(entry) => *entry,
                None => (
                    default_cardinality(
                        field_dtypes
                            .get(&task_name)
                            .and_then(|map| map.get(&query.role_name))
                            .map(String::as_str),
                        is_anchor,
                    ),
                    false,
                ),
            };
            if is_anchor {
                anchor_query_id = Some(query.query_id);
            }
            fields.push(RecordFieldSpec {
                query_id: query.query_id,
                name: query.role_name.clone(),
                role_index: query.role_index,
                cardinality,
                is_anchor,
                exclusive,
            });
        }
        if config.mode == "natural" && anchor_query_id.is_none() {
            return Err(format!(
                "record {task_name:?} declares anchor {:?} but no matching field query was \
                 found in the layout",
                config.anchor.clone().unwrap_or_default()
            ));
        }
        if config.mode != "natural" && anchor_query_id.is_some() {
            return Err(format!(
                "record {task_name:?} mode={:?} must not declare a fixed anchor query",
                config.mode
            ));
        }
        specs.insert(
            task_index,
            RecordSpec {
                task_index,
                task_name,
                task_type,
                mode: config.mode.clone(),
                fields,
                anchor_query_id,
                occurrence_policy: config.occurrence_policy.clone(),
            },
        );
    }
    Ok(specs)
}
