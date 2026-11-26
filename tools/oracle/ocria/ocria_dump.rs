//! Instrumentation for the OCRia-nano-1 oracle: run the pinned LiteParse engine
//! on one page image and print the line list as JSON, in image pixel
//! coordinates.
//!
//! This file is **not** built by this crate. It is kept as the exact source of
//! the recorded page oracles: copy it into a temporary checkout of LiteParse
//! `44ceb6ab160c64cc8b43aab4ed85251719495629` at
//! `crates/liteparse/examples/ocria_dump.rs` and run
//!
//! ```sh
//! cargo run --release --no-default-features --features ocria-nano \
//!   --example ocria_dump -- <model-dir> <page.png> [long-side] [threshold]
//! ```
//!
//! It calls `OcriaEngine::run` directly, which is why the boxes it prints are
//! image pixels and not the PDF points `lit parse` reports.

use liteparse::ocr::ocria::{OcriaEngine, OcriaOptions};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: ocria_dump <model-dir> <page.png> [long_side] [thresh]");
        std::process::exit(2);
    }
    let dir = PathBuf::from(&args[1]);
    let runtime = std::env::var("ORT_DYLIB_PATH").ok().map(PathBuf::from);
    let mut opts = OcriaOptions::default();
    if let Some(v) = args.get(3) {
        opts.det_long_side = v.parse().expect("long side");
    }
    if let Some(v) = args.get(4) {
        opts.det_thresh = Some(v.parse().expect("threshold"));
    }
    let engine = OcriaEngine::load(&dir, runtime.as_deref(), opts).expect("load engine");
    let image = image::open(&args[2]).expect("open page").to_rgb8();
    let (w, h) = (image.width(), image.height());
    let lines = engine.run(image.as_raw(), w, h).expect("run");
    let mut out = String::from("{\n  \"width\": ");
    out.push_str(&w.to_string());
    out.push_str(",\n  \"height\": ");
    out.push_str(&h.to_string());
    out.push_str(",\n  \"lines\": [");
    for (i, line) in lines.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let polygon = match line.polygon {
            Some(p) => format!(
                "[[{}, {}], [{}, {}], [{}, {}], [{}, {}]]",
                p[0][0], p[0][1], p[1][0], p[1][1], p[2][0], p[2][1], p[3][0], p[3][1]
            ),
            None => "null".into(),
        };
        out.push_str(&format!(
            "\n    {{\"text\": {}, \"bbox\": [{}, {}, {}, {}], \"confidence\": {}, \"faint\": {}, \"polygon\": {}}}",
            serde_json::to_string(&line.text).unwrap(),
            line.bbox[0],
            line.bbox[1],
            line.bbox[2],
            line.bbox[3],
            line.confidence,
            line.faint,
            polygon
        ));
    }
    out.push_str("\n  ]\n}\n");
    print!("{out}");
}
