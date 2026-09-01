use serde_json::json;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn checkpoint_schema_records_decode_step() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let path = std::env::temp_dir().join(format!(
        "rmi-parity-trace-step-{}-{}.jsonl",
        std::process::id(),
        line!()
    ));
    let _ = std::fs::remove_file(&path);
    std::env::set_var("RMI_PARITY_TRACE", &path);

    let binary = rust_model_inference::parity_trace::checkpoint_at(
        "attn_norm",
        Some(3),
        Some(7),
        &[2],
        &[1.0, 3.0],
    )
    .unwrap()
    .unwrap();
    let record: serde_json::Value =
        serde_json::from_str(std::fs::read_to_string(&path).unwrap().trim()).unwrap();

    assert_eq!(record["name"], "attn_norm");
    assert_eq!(record["layer"], 3);
    assert_eq!(record["step"], 7);
    assert_eq!(record["shape"], json!([2]));
    assert_eq!(record["occurrence"], 0);
    assert_eq!(std::fs::read(&binary).unwrap().len(), 8);

    std::fs::remove_file(binary).unwrap();
    std::fs::remove_file(path).unwrap();
    std::env::remove_var("RMI_PARITY_TRACE");
}
