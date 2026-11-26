//! Ablation: can LayoutNet run from OCR lines alone, without the projection
//! signals (`proj = None`)?
//!
//! The reference refuses this path outright (`run_page_framed` returns an empty
//! page when an OCR page has no `ProjSignals`), so the question is whether the
//! model still produces sane tables when the eight projection features and the
//! XY-cut leaf ids are absent. This example builds units from the OCR engine's
//! own line output and runs the table model with `sig = None`.
//!
//! Not built by this crate. It goes into a temporary checkout of LiteParse
//! `44ceb6ab160c64cc8b43aab4ed85251719495629` at
//! `crates/liteparse/examples/tablenet_ablate.rs`, together with a
//! `PageInput::from_ocr_units` constructor added to
//! `crates/liteparse/src/table_model/mod.rs` (the struct's `raw` field is
//! private), and then:
//!
//! cargo run --release --no-default-features --features ocria-nano \
//!   --example tablenet_ablate -- <model-dir> <page.png>

use liteparse::ocr::ocria::{OcriaEngine, OcriaOptions};
use liteparse::table_model;
use liteparse::table_model::PageInput;
use liteparse::table_model::decode::DecodeOptions;
use liteparse::table_model::units::Unit;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(&args[1]);
    let page = &args[2];
    let runtime = std::env::var("ORT_DYLIB_PATH").ok().map(PathBuf::from);
    let engine = OcriaEngine::load(&dir, runtime.as_deref(), OcriaOptions::default()).expect("engine");
    let image = image::open(page).expect("page").to_rgb8();
    let (w, h) = (image.width(), image.height());
    let lines = engine.run(image.as_raw(), w, h).expect("ocr");

    let units: Vec<Unit> = lines
        .iter()
        .filter(|l| !l.text.trim().is_empty())
        .map(|l| {
            let [x0, y0, x1, y1] = l.bbox;
            let bx = [x0 as f64, y0 as f64, x1 as f64, y1 as f64];
            Unit {
                bx,
                pbox: bx,
                text: l.text.trim().to_string(),
                h: (bx[3] - bx[1]) / 1.4,
                size: 0.0,
                bold: None,
                q: 0,
                minor: false,
                words: Vec::new(),
            }
        })
        .collect();
    println!("ocr lines: {} (units {})", lines.len(), units.len());
    if units.len() < 2 {
        return;
    }
    let net = table_model::load(&dir.join("tablenet.bin")).expect("tablenet.bin");
    let input = PageInput::from_ocr_units(units, (w as f64, h as f64));
    for (label, sig) in [("proj=None", None)] {
        let model = table_model::run_page(net, Some(net), &input, sig, &DecodeOptions::default());
        println!(
            "{label}: tables {}, classes {}, blocks {}",
            model.tables.len(),
            model.classes.len(),
            model.blocks.len()
        );
        for (i, t) in model.tables.iter().enumerate() {
            println!(
                "  table {i}: {}x{} grid, {} cells, head rows {:?}",
                t.grid.n_rows,
                t.grid.n_cols,
                t.grid.cells.len(),
                t.grid.head_rows
            );
            for c in t.grid.cells.iter().take(6) {
                println!("    [{},{}..{},{}] {:?}", c.r0, c.c0, c.r1, c.c1, c.text);
            }
        }
    }
}
