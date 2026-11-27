use rust_model_inference::models::laya::request::*;
use serde_json::Value;
use tokenizers::{models::wordlevel::WordLevel, Tokenizer};

fn tokenizer() -> Tokenizer {
    let model = WordLevel::builder()
        .vocab(
            [
                ("<unk>".into(), 0),
                ("<bos>".into(), 2),
                ("<eos>".into(), 1),
                ("<mask>".into(), 4),
            ]
            .into_iter()
            .collect(),
        )
        .unk_token("<unk>".into())
        .build()
        .unwrap();
    Tokenizer::new(model)
}

#[test]
fn builds_markers_and_preserves_question_and_candidate_order() {
    let request: Request = serde_json::from_str(
        r#"{"state":"中文", "questions":{
        "z":{"type":"choice","instructions":"部门","criteria":{"z":"退款","a":"其他"}},
        "a":{"type":"noul","instructions":"要求退款"}}}"#,
    )
    .unwrap();
    let result = prepare(&tokenizer(), &request, 1024, 256).unwrap();
    assert_eq!(
        result.iter().map(|q| q.id.as_str()).collect::<Vec<_>>(),
        ["z", "a"]
    );
    assert_eq!(result[0].labels, [Value::from("z"), Value::from("a")]);
    assert_eq!(result[0].ids[0], 2);
    assert_eq!(result[0].ids.last(), Some(&1));
    assert_eq!(result[0].markers, [3, 5]);
    assert_eq!(result[1].qtype, 2);
}

#[test]
fn rejects_bad_questions_and_sequence_limits() {
    for questions in [
        r#"{"q":{"type":"choice","instructions":"Q","criteria":{}}}"#,
        r#"{"q":{"type":"noul","instructions":"Q","criteria":{"typo":"x"}}}"#,
        r#"{"q":{"type":"score","instructions":"Q","criteria":[null]}}"#,
        r#"{"q":{"type":"choice","instructions":"Q","criteria":["x"],"labels":{}}}"#,
    ] {
        let request =
            serde_json::from_str::<Request>(&format!(r#"{{"state":"S","questions":{questions}}}"#))
                .unwrap();
        assert!(prepare(&tokenizer(), &request, 1024, 256).is_err());
    }
    let request: Request =
        serde_json::from_str(r#"{"state": "S", "questions":{}, "max_len":0}"#).unwrap();
    assert!(prepare(&tokenizer(), &request, 1024, 256).is_err());
}

#[test]
fn structured_fields_use_python_json_spacing() {
    let tok = Tokenizer::new(
        WordLevel::builder()
            .vocab(
                [
                    ("<unk>".into(), 0),
                    ("<bos>".into(), 2),
                    ("<eos>".into(), 1),
                    ("<mask>".into(), 4),
                    (
                        r#"choice question: {"问题": "退款", "flags": [1, false]}"#.into(),
                        5,
                    ),
                    (r#" only: {"desc": "唯一", "active": false}"#.into(), 6),
                    (r#"{"text": "中文", "count": 1}"#.into(), 7),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("<unk>".into())
            .build()
            .unwrap(),
    );
    let r = serde_json::from_str(
        r#"{"state":{"text":"中文","count":1},"questions":{
        "q":{"type":"choice","instructions":{"问题":"退款","flags":[1,false]},
        "criteria":{"only":{"desc":"唯一","active":false}}}}}"#,
    )
    .unwrap();
    let q = prepare(&tok, &r, 1024, 256).unwrap();
    assert_eq!(q[0].ids, [2, 5, 1, 4, 6, 1, 7, 1]);
}
