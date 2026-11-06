//! SiLU cross-check against ggml's ARM kernel.
//!
//! The oracle is a small standalone C program
//! (`tools/oracle/silu/silu_oracle.c`) that reproduces ggml's ARM
//! `ggml_vec_silu_f32` verbatim and dumps the F32 result. The test feeds
//! the same values to the repository's `silu_ggml` and compares the two
//! outputs bit for bit.
//!
//! Why this test exists: ggml's vectorized SiLU does not use libm's
//! `expf` but the polynomial approximation in `ggml_v_expf`, and the two
//! differ by up to one ULP. In the Nemotron-3-Nano 30B SSM branch that
//! one ULP is enough to flip an 8-bit Q8_K activation level in the Q4_K
//! projection that follows, which turns a 1e-7 relative difference into a
//! 4e-4 one -- so the polynomial has to be right, not just close.
//!
//! The oracle binary is built on demand with `cc`; the test is skipped
//! (not failed) when no C compiler is available, so the suite still runs
//! on a machine without a toolchain. It is skipped on non-ARM hosts,
//! because the reference is the ARM NEON kernel.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Deterministic PRNG (xorshift64*) so the corpus is reproducible across
/// runs and machines.
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
}

fn oracle_source() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/oracle/silu/silu_oracle.c")
}

fn build_oracle(work_dir: &Path) -> Option<PathBuf> {
    let source = oracle_source();
    if !source.exists() {
        return None;
    }
    let binary = work_dir.join("silu_oracle");
    for compiler in ["cc", "gcc", "clang"] {
        let mut command = Command::new(compiler);
        command.args(["-O2", "-ffp-contract=off"]);
        if cfg!(target_arch = "aarch64") {
            command.arg("-march=armv8.2-a+dotprod");
        }
        let status = command.arg("-o").arg(&binary).arg(&source).status();
        match status {
            Ok(status) if status.success() => return Some(binary),
            // A compiler that fails to build the oracle is a *failure*, not
            // a skip: silently skipping would report "ok" for a test that
            // compared nothing.
            Ok(status) => panic!("{compiler} failed to build the SiLU oracle: {status}"),
            Err(_) => continue,
        }
    }
    None
}

#[test]
fn silu_matches_the_ggml_arm_reference() {
    if !cfg!(target_arch = "aarch64") {
        eprintln!("SKIP: the SiLU reference is the ARM NEON kernel");
        return;
    }
    let work_dir = std::env::temp_dir().join("rmi_silu_oracle");
    std::fs::create_dir_all(&work_dir).expect("create work dir");

    let Some(oracle) = build_oracle(&work_dir) else {
        eprintln!("SKIP: no C compiler available to build the SiLU oracle");
        return;
    };

    // A corpus that covers the interesting range: small magnitudes (where
    // the polynomial and libm disagree most often), the sigmoid knee, and
    // the large magnitudes the SSM branch produces.
    let mut rng = Rng(0x5EED_0000_0000_5117);
    let mut input: Vec<f32> = Vec::new();
    for i in 0..4096 {
        let t = (i % 512) as f32 / 512.0;
        let spread = match i % 4 {
            0 => 1.0,
            1 => 10.0,
            2 => 100.0,
            _ => 0.01,
        };
        let v = ((rng.next_u32() as i32 % 20001 - 10000) as f32 / 10000.0) * spread;
        input.push(v + t * 0.5);
    }

    let in_path = work_dir.join("in.f32");
    let out_path = work_dir.join("out.f32");
    let mut bytes = Vec::with_capacity(input.len() * 4);
    for value in &input {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(&in_path, &bytes).expect("write input");

    let status = Command::new(&oracle)
        .arg(&in_path)
        .arg(&out_path)
        .status()
        .expect("run oracle");
    assert!(status.success(), "oracle exited with {status}");

    let reference = std::fs::read(&out_path).expect("read oracle output");
    let reference: Vec<f32> = reference
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(
        reference.len(),
        input.len(),
        "oracle produced the wrong length"
    );

    let mut ours = input.clone();
    rust_model_inference::ops::activation::silu_ggml_inplace(&mut ours);

    let mut mismatches = 0usize;
    for (index, (lhs, rhs)) in ours.iter().zip(reference.iter()).enumerate() {
        if lhs.to_bits() != rhs.to_bits() {
            if mismatches < 8 {
                eprintln!(
                    "mismatch at {index}: rust={lhs:?} ({:#010x}) oracle={rhs:?} ({:#010x}) input={:?}",
                    lhs.to_bits(),
                    rhs.to_bits(),
                    input[index]
                );
            }
            mismatches += 1;
        }
    }
    assert_eq!(
        mismatches,
        0,
        "{mismatches} of {} SiLU values differ from ggml's ARM kernel",
        ours.len()
    );
}
