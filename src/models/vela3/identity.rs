//! The recorded snapshot, rather than tensor shapes, identifies compatible weights.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::io::Read;
use std::path::Path;

fn snapshots() -> Result<Value, String> {
    serde_json::from_str(include_str!("../../../tools/oracle/vela3/snapshots.json"))
        .map_err(|e| format!("invalid built-in Vela3 snapshots: {e}"))
}

fn verify_file(path: &Path, size: Option<u64>, sha256: &str) -> Result<(), String> {
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("Open {}: {e}", path.display()))?;
    if size.is_some_and(|n| file.metadata().map(|m| m.len() != n).unwrap_or(true)) {
        return Err(format!("Vela3 snapshot size mismatch: {}", path.display()));
    }
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|e| format!("Read {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    if format!("{:x}", hash.finalize()) != sha256 {
        return Err(format!(
            "Vela3 snapshot SHA256 mismatch: {}",
            path.display()
        ));
    }
    Ok(())
}

pub(super) fn verify_d3(config: &Value, root: &Path) -> Result<(), String> {
    let pins = snapshots()?;
    let base = &config["base"];
    let repo = base["repo_id"]
        .as_str()
        .ok_or("missing Vela3 base.repo_id")?;
    let revision = base["revision"]
        .as_str()
        .ok_or("missing Vela3 base.revision")?;
    let pinned_base = pins
        .as_object()
        .ok_or("invalid Vela3 snapshots")?
        .values()
        .map(|v| &v["contract"]["base"])
        .find(|b| b["repo_id"] == repo && b["revision"] == revision)
        .ok_or_else(|| {
            format!("unsupported Vela3 base {repo}@{revision}; use the pinned base.revision")
        })?;
    let name = pinned_base["repo_id"]
        .as_str()
        .and_then(|r| r.strip_prefix("vllm-sr/"))
        .ok_or("invalid pinned d3 repo_id")?;
    let snapshot = &pins[name];
    let pinned = &snapshot["pinned_reference"];
    let mut weights = BTreeSet::new();
    for entry in snapshot["files"]
        .as_array()
        .ok_or("missing pinned d3 files")?
    {
        let file = entry["Path"].as_str().ok_or("invalid pinned d3 filename")?;
        if file.ends_with(".safetensors") && pinned.is_object() {
            continue;
        }
        let override_hash = pinned["verified_contract_sha256"][file].as_str();
        verify_file(
            &root.join(file),
            if override_hash.is_some() {
                None
            } else {
                entry["Size"].as_u64()
            },
            override_hash
                .or_else(|| entry["Sha256"].as_str())
                .ok_or("missing pinned d3 hash")?,
        )?;
        if file.starts_with("model") && file.ends_with(".safetensors") {
            weights.insert(file.to_owned());
        }
    }
    for entry in pinned["files"].as_array().into_iter().flatten() {
        let file = entry["path"].as_str().ok_or("invalid pinned d3 filename")?;
        verify_file(
            &root.join(file),
            entry["size"].as_u64(),
            entry["sha256"].as_str().ok_or("missing pinned d3 hash")?,
        )?;
        if file.starts_with("model") {
            weights.insert(file.to_owned());
        }
    }
    // open_weights prefers a single file; refuse aliases and indexes selecting other shards.
    let actual = if root.join("model.safetensors").is_file() {
        BTreeSet::from(["model.safetensors".to_owned()])
    } else {
        let index: Value = serde_json::from_slice(
            &std::fs::read(root.join("model.safetensors.index.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        index["weight_map"]
            .as_object()
            .ok_or("missing d3 weight_map")?
            .values()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| "invalid d3 shard name".to_owned())
            })
            .collect::<Result<BTreeSet<_>, _>>()?
    };
    if weights.is_empty() || actual != weights {
        return Err("d3 index selects weights outside the pinned base.revision".into());
    }
    Ok(())
}

pub(super) fn vision_backbone<'a>(config: &'a Value, name: &str) -> Result<&'a str, String> {
    let (kind, expected) = match name {
        "decision" => ("decision-h2", "backbone_h2.safetensors"),
        "box" => ("decision-v2", "backbone_v2.safetensors"),
        "span" | "ocr" => ("raw", "model.safetensors"),
        _ => return Err(format!("unknown UltraVBERT task {name}")),
    };
    let cfg = &config["tasks"][name];
    let file = match cfg.get("backbone_file") {
        Some(file) => file.as_str(),
        None => {
            // The published raw tasks explicitly refer to backbone.model, with no per-task file.
            (cfg["backbone"] == "raw")
                .then(|| config["backbone"]["model"].as_str())
                .flatten()
        }
    }
    .ok_or_else(|| format!("missing UltraVBERT {name} backbone_file"))?;
    if cfg["backbone"] != kind || file != expected {
        return Err(format!(
            "UltraVBERT {name} requires its pinned {kind} backbone {expected}"
        ));
    }
    Ok(file)
}

pub(super) fn verify_vision(root: &Path, config: &Value, name: &str) -> Result<(), String> {
    let backbone = vision_backbone(config, name)?;
    let expected_adapter = format!("{name}_adapter.safetensors");
    if config["tasks"][name]["file"] != expected_adapter {
        return Err(format!("invalid UltraVBERT {name} adapter file"));
    }
    let pins = snapshots()?;
    let files = pins["vela3-fast-vision"]["files"]
        .as_array()
        .ok_or("missing vision snapshots")?;
    for file in [backbone, &expected_adapter] {
        let entry = files
            .iter()
            .find(|e| e["Path"] == file)
            .ok_or("missing pinned vision file")?;
        verify_file(
            &root.join(file),
            entry["Size"].as_u64(),
            entry["Sha256"]
                .as_str()
                .ok_or("missing pinned vision hash")?,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn same_size_substitution_is_rejected_by_content() {
        let path = std::env::temp_dir().join(format!("vela3-hash-{}", std::process::id()));
        std::fs::write(&path, b"good").unwrap();
        let hash = format!("{:x}", Sha256::digest(b"good"));
        verify_file(&path, Some(4), &hash).unwrap();
        std::fs::write(&path, b"evil").unwrap();
        assert!(verify_file(&path, Some(4), &hash)
            .unwrap_err()
            .contains("SHA256 mismatch"));
        assert!(verify_file(&path, Some(5), &hash)
            .unwrap_err()
            .contains("size mismatch"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn base_revision_is_required_and_must_be_recorded() {
        let root = Path::new("/missing");
        assert!(
            verify_d3(&json!({"base":{"repo_id":"vllm-sr/d3-mini"}}), root)
                .unwrap_err()
                .contains("base.revision")
        );
        assert!(verify_d3(
            &json!({"base":{"repo_id":"vllm-sr/d3-mini","revision":"main"}}),
            root
        )
        .unwrap_err()
        .contains("pinned base.revision"));
    }

    #[test]
    fn vision_pairing_requires_an_explicit_matching_backbone() {
        let mut c = json!({"backbone":{"model":"model.safetensors"},"tasks":{
            "span":{"backbone":"raw"}, "decision":{"backbone":"decision-h2","backbone_file":"backbone_h2.safetensors"}}});
        assert_eq!(vision_backbone(&c, "span").unwrap(), "model.safetensors");
        assert_eq!(
            vision_backbone(&c, "decision").unwrap(),
            "backbone_h2.safetensors"
        );
        c["tasks"]["decision"]
            .as_object_mut()
            .unwrap()
            .remove("backbone_file");
        assert!(vision_backbone(&c, "decision")
            .unwrap_err()
            .contains("missing"));
        c["tasks"]["decision"]["backbone_file"] = json!("backbone_v2.safetensors");
        assert!(vision_backbone(&c, "decision").is_err());
        c["tasks"]["span"]["backbone_file"] = json!(42);
        assert!(vision_backbone(&c, "span").is_err());
        c["tasks"]["span"]
            .as_object_mut()
            .unwrap()
            .remove("backbone_file");
        c.as_object_mut().unwrap().remove("backbone");
        assert!(vision_backbone(&c, "span").is_err());
    }
}
