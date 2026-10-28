use rust_model_inference::app::{
    run_jev_decision_data, run_jev_grouped_decision_data, JevGroupInput, JevGroupedQuestionInput,
    JevMode, JevQuestionInput,
};
use rust_model_inference::core::tensor::{MetaValue, TensorInfo, TensorSource};
use std::sync::Arc;

struct HySource(MetaValue);

impl TensorSource for HySource {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        (key == "general.architecture").then_some(&self.0)
    }
    fn tensor_info(&self, _: &str) -> Option<&TensorInfo> {
        None
    }
    fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
        None
    }
}

fn source() -> Arc<dyn TensorSource> {
    Arc::new(HySource(MetaValue::String("hy_v3".into())))
}

#[test]
fn hy_text_jev_reaches_its_decoder_without_audio() {
    let error = run_jev_decision_data(
        source(),
        "The sky is blue.",
        &[JevQuestionInput {
            text: "Is that true?".into(),
            options: vec!["yes".into(), "no".into()],
        }],
        None,
        1,
        8,
        Default::default(),
    )
    .unwrap_err();
    assert!(error.contains("tokenizer"), "wrong Hy dispatch: {error}");
    assert!(!error.contains("not yet supported"), "{error}");
}

#[test]
fn hy_grouped_jev_reaches_its_decoder_without_audio() {
    let error = run_jev_grouped_decision_data(
        source(),
        "The sky is blue.",
        &[JevGroupedQuestionInput {
            text: "Classify the statement.".into(),
            groups: vec![JevGroupInput {
                label: "truth".into(),
                options: vec!["yes".into(), "no".into()],
            }],
        }],
        JevMode::BlockChoice,
        1,
        8,
        Default::default(),
    )
    .unwrap_err();
    assert!(error.contains("tokenizer"), "wrong Hy dispatch: {error}");
    assert!(!error.contains("not yet supported"), "{error}");
}

#[cfg(test)]
mod real_weights {
    use base64::Engine as _;
    use std::path::PathBuf;
    use std::process::{Child, Command, Stdio};
    use std::sync::{Arc, Barrier, Mutex};
    use std::time::{Duration, Instant};

    static MODEL_RUN: Mutex<()> = Mutex::new(());

    fn model() -> PathBuf {
        std::env::var_os("RMI_HY_SIMULMT_MODEL")
            .map(PathBuf::from)
            .expect("set RMI_HY_SIMULMT_MODEL to a Hy-SimulMT GGUF")
    }

    struct Server(Child);

    impl Drop for Server {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    fn post(
        client: &reqwest::blocking::Client,
        url: &str,
        body: serde_json::Value,
    ) -> (u16, serde_json::Value) {
        let response = client
            .post(url)
            .header("content-type", "application/json")
            .body(body.to_string())
            .send()
            .unwrap();
        let status = response.status().as_u16();
        let text = response.text().unwrap();
        let value = serde_json::from_str(&text).unwrap_or_else(|error| panic!("{error}: {text}"));
        (status, value)
    }

    fn assert_distribution(value: &serde_json::Value) {
        let probabilities = value["probabilities"].as_object().unwrap();
        assert_eq!(probabilities.len(), 2, "{value}");
        let sum: f64 = probabilities.values().map(|p| p.as_f64().unwrap()).sum();
        assert!((sum - 1.0).abs() < 1e-5, "{value}");
        assert!(probabilities.values().all(|p| {
            let p = p.as_f64().unwrap();
            p.is_finite() && (0.0..=1.0).contains(&p)
        }));
    }

    #[test]
    #[ignore = "requires RMI_HY_SIMULMT_MODEL and RMI_HY_SIMULMT_WAV"]
    fn hy_http_serves_both_uploads_and_text_jev() {
        let _guard = MODEL_RUN.lock().unwrap_or_else(|error| error.into_inner());
        let wav_path = std::env::var_os("RMI_HY_SIMULMT_WAV")
            .map(PathBuf::from)
            .expect("set RMI_HY_SIMULMT_WAV to a short 16 kHz mono PCM16 WAV");
        let wav = std::fs::read(&wav_path).unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let mut server = Server(
            Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
                .arg("--serve")
                .arg("--model")
                .arg(model())
                .arg("--audio")
                .arg(&wav_path)
                .args(["--host", "127.0.0.1", "--port", &port.to_string()])
                .args(["--compute", "cpu", "--threads", "4", "--max-tokens", "1"])
                .stdout(Stdio::null())
                .spawn()
                .unwrap(),
        );
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(180))
            .build()
            .unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            assert!(
                server.0.try_wait().unwrap().is_none(),
                "server exited at startup"
            );
            if client.get(format!("{base}/health")).send().is_ok() {
                break;
            }
            assert!(Instant::now() < deadline, "server startup timed out");
            std::thread::sleep(Duration::from_millis(100));
        }
        let encoded = base64::engine::general_purpose::STANDARD.encode(&wav);
        let barrier = Arc::new(Barrier::new(2));
        let url = format!("{base}/v1/audio/transcriptions_json");
        let request = serde_json::json!({"input":encoded,"language":"en"});
        let mut responses = std::thread::scope(|scope| {
            let handles = [0, 1].map(|_| {
                let client = client.clone();
                let barrier = barrier.clone();
                let url = url.clone();
                let request = request.clone();
                scope.spawn(move || {
                    barrier.wait();
                    post(&client, &url, request)
                })
            });
            handles.map(|handle| handle.join().unwrap())
        });
        responses.sort_by_key(|(status, _)| *status);
        assert_eq!(
            [responses[0].0, responses[1].0],
            [200, 429],
            "{responses:?}"
        );
        let [(_, json), (_, busy)] = responses;
        assert!(
            busy["error"]
                .as_str()
                .unwrap()
                .contains("processing another audio request"),
            "{busy}"
        );
        assert!(json["text"].is_string(), "{json}");

        let boundary = "hy-simulmt-upload";
        let upload_body = |language: &str| {
            let mut body = format!("--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n{language}\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n").into_bytes();
            body.extend(&wav);
            body.extend(format!("\r\n--{boundary}--\r\n").bytes());
            body
        };
        let response = client
            .post(format!("{base}/v1/audio/transcriptions"))
            .header(
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(upload_body("en"))
            .send()
            .unwrap();
        assert_eq!(response.status().as_u16(), 200);
        let upload: serde_json::Value = serde_json::from_str(&response.text().unwrap()).unwrap();
        assert_eq!(
            upload, json,
            "both routes must pass the same target language"
        );

        let (status, invalid) = post(
            &client,
            &format!("{base}/v1/audio/transcriptions_json"),
            serde_json::json!({"input":encoded,"language":" "}),
        );
        assert_eq!(status, 422, "{invalid}");
        assert!(invalid["error"]
            .as_str()
            .unwrap()
            .contains("target language"));

        let response = client
            .post(format!("{base}/v1/audio/transcriptions"))
            .header(
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(upload_body(" "))
            .send()
            .unwrap();
        assert_eq!(response.status().as_u16(), 422);
        assert!(response.text().unwrap().contains("target language"));

        for (field, value) in [("response_format", "text"), ("temperature", "0.5")] {
            let mut invalid_body = format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"\r\n\r\n{value}\r\n").into_bytes();
            invalid_body.extend(upload_body("en"));
            let response = client
                .post(format!("{base}/v1/audio/transcriptions"))
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(invalid_body)
                .send()
                .unwrap();
            assert_eq!(response.status().as_u16(), 400, "{field}={value}");
        }
        for invalid in [
            serde_json::json!({"input":encoded,"response_format":"text"}),
            serde_json::json!({"input":encoded,"temperature":0.5}),
        ] {
            let (status, error) = post(
                &client,
                &format!("{base}/v1/audio/transcriptions_json"),
                invalid,
            );
            assert_eq!(status, 400, "{error}");
        }

        let (status, single) = post(
            &client,
            &format!("{base}/v1/jev/score"),
            serde_json::json!({"context":"The sky is blue.","questions":[{"text":"Is that true?","options":["yes","no"]}]}),
        );
        assert_eq!(status, 200, "{single}");
        assert_distribution(&single["results"][0]);
        let (status, grouped) = post(
            &client,
            &format!("{base}/v1/jev/grouped"),
            serde_json::json!({"context":"The sky is blue.","mode":"block_choice","questions":[{"text":"Classify.","groups":[{"label":"truth","options":["yes","no"]}]}]}),
        );
        assert_eq!(status, 200, "{grouped}");
        assert_distribution(&grouped["results"][0]["groups"][0]);
    }

    #[test]
    #[ignore = "requires RMI_HY_SIMULMT_MODEL"]
    fn hy_cli_jev_uses_no_audio_and_normalizes_single_and_grouped_labels() {
        let _guard = MODEL_RUN.lock().unwrap_or_else(|error| error.into_inner());
        for grouped in [false, true] {
            let mut command = Command::new(env!("CARGO_BIN_EXE_rust-model-inference"));
            command
                .arg("--model")
                .arg(model())
                .args(["--compute", "cpu", "--threads", "4", "--jev"])
                .args([
                    "--jev-context",
                    "The sky is blue.",
                    "--jev-question",
                    "Is that true?",
                ])
                .args(["--jev-output", "json"]);
            if grouped {
                command.args(["--jev-block", "truth"]);
            }
            let output = command
                .args(["--jev-option", "yes", "--jev-option", "no"])
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            if grouped {
                assert_distribution(&result["groups"][0]);
            } else {
                assert_distribution(&result);
            }
        }
    }

    #[test]
    #[ignore = "requires RMI_HY_SIMULMT_MODEL and RMI_HY_SIMULMT_WAV (at least 1600ms)"]
    fn hy_final_audio_eot_does_not_require_an_extra_kv_slot() {
        use rust_model_inference::core::scratchpad::KvFormat;
        use rust_model_inference::core::tensor::TensorSource;
        use rust_model_inference::core::tokenizer::BPETokenizer;
        use rust_model_inference::format::ggufrs::{open_model_source, ComponentRole};
        use rust_model_inference::models::hy_simulmt::HySimulMt;
        use rust_model_inference::ops::audio::frontend::decode_pcm16_wav;
        use rust_model_inference::ComputePool;

        let _guard = MODEL_RUN.lock().unwrap_or_else(|error| error.into_inner());
        let source: Arc<dyn TensorSource> =
            Arc::from(open_model_source(&model(), ComponentRole::Llm).unwrap());
        let tokenizer =
            BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).unwrap();
        let eot = tokenizer.token_id("<｜hy_EOT｜>").unwrap();
        let runtime = HySimulMt::from_source(source, Arc::new(ComputePool::new(4))).unwrap();
        let wav =
            std::fs::read(std::env::var_os("RMI_HY_SIMULMT_WAV").expect("set RMI_HY_SIMULMT_WAV"))
                .unwrap();
        let mut samples = decode_pcm16_wav(&wav).unwrap();
        samples.truncate(25_600);
        assert_eq!(
            samples.len(),
            25_600,
            "the fixture must contain at least 1600ms"
        );
        let full = runtime
            .translate(&samples, "en", None, 512, 8192, 64, KvFormat::F16)
            .unwrap();
        assert_eq!(
            full.token_ids.last(),
            Some(&eot),
            "expected the first audio turn to end in EOT"
        );
        let capacity = full.prompt_tokens + full.token_ids.len() - 1;
        let bounded = runtime
            .translate(&samples, "en", None, 512, capacity, 64, KvFormat::F16)
            .unwrap();
        assert_eq!(bounded.token_ids, full.token_ids);
    }
}
