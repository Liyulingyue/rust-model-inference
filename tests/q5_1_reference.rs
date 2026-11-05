//! Q5_1 dequantization cross-check against ggml's reference C code.
//!
//! The oracle is a small standalone C program
//! (`tools/oracle/q5_1/q5_1_oracle.c`) that transcribes ggml's
//! `dequantize_row_q5_1` verbatim and dumps the F32 result. The test
//! feeds the same Q5_1 blocks to the Rust implementation and compares
//! the two outputs element by element.
//!
//! The oracle binary is built on demand with `cc`; the test is skipped
//! (not failed) when no C compiler is available, so the suite still
//! runs on a machine without a toolchain.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Deterministic PRNG (xorshift64*) so the blocks are reproducible
/// across runs and machines.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// A random IEEE-754 binary16 value, including subnormals and the
    /// occasional NaN-free infinity, so the F16 conversion path is
    /// exercised beyond the ordinary range.
    fn next_f16_bits(&mut self) -> u16 {
        let bits = self.next_u32() as u16;
        // Keep NaN out of the corpus: ggml's F16 -> F32 conversion and
        // the Rust one must agree bit for bit, and NaN payloads are not
        // worth the ambiguity here.
        if (bits & 0x7C00) == 0x7C00 && (bits & 0x03FF) != 0 {
            return bits & 0xFC00;
        }
        bits
    }
}

/// Build a corpus of Q5_1 blocks.
///
/// The `d` / `m` fields are random F16 values and the quantized levels
/// sweep the whole 0..31 range, so every combination of nibble bits and
/// 5-th bits appears.
fn build_blocks(count: usize) -> Vec<u8> {
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    let mut out = Vec::with_capacity(count * 24);
    for index in 0..count {
        let d = rng.next_f16_bits();
        let m = rng.next_f16_bits();
        let mut qs = [0u8; 16];
        let mut qh = 0u32;
        for j in 0..16 {
            // Sweep the 5-bit levels so the corpus covers 0..31 for
            // both nibbles, plus pseudo-random levels on top.
            let lo = ((index as u8).wrapping_mul(7).wrapping_add(j as u8) % 32) | (rng.next_u32() as u8 & 0xE0);
            let hi = ((index as u8).wrapping_mul(11).wrapping_add(j as u8) % 32) | (rng.next_u32() as u8 & 0xE0);
            qs[j] = (lo & 0x0F) | ((hi & 0x0F) << 4);
            qh |= ((lo >> 4) as u32 & 1) << j;
            qh |= ((hi >> 4) as u32 & 1) << (j + 16);
        }
        out.extend_from_slice(&d.to_le_bytes());
        out.extend_from_slice(&m.to_le_bytes());
        out.extend_from_slice(&qh.to_le_bytes());
        out.extend_from_slice(&qs);
    }
    out
}

fn oracle_source() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/oracle/q5_1/q5_1_oracle.c")
}

fn build_oracle(work_dir: &Path) -> Option<PathBuf> {
    let source = oracle_source();
    if !source.exists() {
        return None;
    }
    let binary = work_dir.join("q5_1_oracle");
    for compiler in ["cc", "gcc", "clang"] {
        let status = Command::new(compiler)
            .args(["-O2", "-ffp-contract=off", "-o"])
            .arg(&binary)
            .arg(&source)
            .status();
        match status {
            Ok(status) if status.success() => return Some(binary),
            Ok(_) => continue,
            Err(_) => continue,
        }
    }
    None
}

#[test]
fn q5_1_matches_the_ggml_c_reference() {
    let work_dir = std::env::temp_dir().join("rmi_q5_1_oracle");
    std::fs::create_dir_all(&work_dir).expect("create work dir");

    let Some(oracle) = build_oracle(&work_dir) else {
        eprintln!("SKIP: no C compiler available to build the Q5_1 oracle");
        return;
    };

    const BLOCKS: usize = 4096;
    let blocks = build_blocks(BLOCKS);
    let in_path = work_dir.join("blocks.bin");
    let out_path = work_dir.join("oracle.f32");
    std::fs::write(&in_path, &blocks).expect("write blocks");

    let status = Command::new(&oracle)
        .arg(&in_path)
        .arg(&out_path)
        .arg((BLOCKS * 32).to_string())
        .status()
        .expect("run oracle");
    assert!(status.success(), "oracle exited with {status}");

    let reference = std::fs::read(&out_path).expect("read oracle output");
    let reference: Vec<f32> = reference
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(reference.len(), BLOCKS * 32, "oracle produced the wrong length");

    let mut ours = vec![0.0f32; BLOCKS * 32];
    rust_model_inference::ops::quant::dequantize_row_q5_1(&blocks, &mut ours);

    let mut mismatches = 0usize;
    for (index, (lhs, rhs)) in ours.iter().zip(reference.iter()).enumerate() {
        // Both sides compute `level * d + m` in F32 with the same
        // operands, so the results must be bit-identical. Compare the
        // bit patterns rather than `==` so a NaN would also be caught.
        if lhs.to_bits() != rhs.to_bits() {
            if mismatches < 8 {
                eprintln!(
                    "mismatch at {index}: rust={lhs:?} ({:#010x}) oracle={rhs:?} ({:#010x})",
                    lhs.to_bits(),
                    rhs.to_bits()
                );
            }
            mismatches += 1;
        }
    }
    assert_eq!(
        mismatches, 0,
        "{mismatches} of {} Q5_1 elements differ from the ggml reference",
        ours.len()
    );
}
