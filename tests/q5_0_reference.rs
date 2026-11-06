//! Q5_0 x Q8_0 dot-product cross-check against ggml's ARM NEON kernel.
//!
//! The oracle is a small standalone C program
//! (`tools/oracle/q5_0/q5_0_oracle.c`) that transcribes ggml's
//! `ggml_vec_dot_q5_0_q8_0` for the ARM NEON target and dumps the F32
//! dot products. The test builds the same Q5_0 weights and Q8_0
//! activations, runs the repository's kernel over them, and compares the
//! two outputs bit for bit.
//!
//! Why this test exists: the 30B's `ssm_in.weight` is Q5_0 while the 4B's
//! is Q8_0, and the 4B's Mamba-2 branch matches the upstream oracle to
//! 1 ULP while the 30B's diverges by ~1e-4. That points at the quantized
//! weight path rather than the Mamba-2 arithmetic, and this test is where
//! that gets pinned down.
//!
//! The oracle binary is built on demand with `cc`; the test is skipped
//! (not failed) when no C compiler is available, so the suite still runs
//! on a machine without a toolchain. It is also skipped on non-ARM hosts,
//! because the reference is the ARM NEON kernel.

use std::path::{Path, PathBuf};
use std::process::Command;

use rust_model_inference::ops::kernel::Kernel;

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

/// Build a corpus of Q5_0 rows.
///
/// Each row is `n_in` elements stored as 22-byte blocks: an F16 `d`, a
/// 32-bit `qh` holding the 5-th bit of every element, and 16 bytes of
/// packed nibbles. The quantized levels sweep 0..31 so every combination
/// of nibble bits and 5-th bits appears.
fn build_weights(n_rows: usize, n_in: usize) -> Vec<u8> {
    assert_eq!(n_in % 32, 0);
    let blocks_per_row = n_in / 32;
    let mut rng = Rng(0x5EED_0000_0000_0005);
    let mut out = Vec::with_capacity(n_rows * n_in / 32 * 22);
    for row in 0..n_rows {
        for block in 0..blocks_per_row {
            // Keep the scale in a sane range so the dot products stay
            // finite and the F32 comparisons are meaningful.
            let d = rust_model_inference::ops::f32_to_f16(
                0.01f32 * (1.0 + (rng.next_u32() % 1000) as f32 / 100.0),
            );
            let mut qs = [0u8; 16];
            let mut qh = 0u32;
            for j in 0..16 {
                let seed = (row * 31 + block * 7 + j) as u32;
                let lo = (seed.wrapping_mul(7) % 32) | (rng.next_u32() & 0xE0);
                let hi = (seed.wrapping_mul(11) % 32) | (rng.next_u32() & 0xE0);
                qs[j] = ((lo & 0x0F) | ((hi & 0x0F) << 4)) as u8;
                qh |= ((lo >> 4) as u32 & 1) << j;
                qh |= ((hi >> 4) as u32 & 1) << (j + 16);
            }
            out.extend_from_slice(&d.to_le_bytes());
            out.extend_from_slice(&qh.to_le_bytes());
            out.extend_from_slice(&qs);
        }
    }
    out
}

/// Build the Q8_0 activation blocks in ggml's `block_q8_0` layout: the F16
/// scale first, then the 32 int8 levels, one block after another.
///
/// The values are the levels ggml's `quantize_row_q8_0_ref` would produce,
/// i.e. `round(x * id)` with `id = 127 / amax`.
///
/// Returns the packed blocks (what the oracle reads), the per-block scales
/// (what the Rust kernel takes separately, since it is handed the scales
/// as F32 rather than re-reading them from the block) and the bare levels
/// (what the Rust kernel takes as the activation vector).
fn build_activations(n_in: usize) -> (Vec<u8>, Vec<f32>, Vec<u8>) {
    assert_eq!(n_in % 32, 0);
    let blocks = n_in / 32;
    let mut rng = Rng(0x5EED_0000_0000_00A8);
    let mut raw = Vec::with_capacity(blocks * 34);
    let mut scales = Vec::with_capacity(blocks);
    let mut levels = Vec::with_capacity(n_in);
    for _block in 0..blocks {
        let mut values = [0i8; 32];
        let mut amax = 0.0f32;
        for slot in values.iter_mut() {
            let raw_value = (rng.next_u32() as i32 % 512 - 256) as f32 / 256.0;
            *slot = raw_value.clamp(-127.0, 127.0) as i8;
            amax = amax.max((*slot as f32).abs());
        }
        let scale = amax / 127.0;
        let id = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let d = rust_model_inference::ops::f32_to_f16(scale);
        // ggml stores the scale as F16; the kernel reads it back through
        // `GGML_CPU_FP16_TO_FP32`, so the F16 rounding is part of the
        // contract and must be applied on both sides.
        let d_f32 = rust_model_inference::ops::f16_to_f32(d);
        raw.extend_from_slice(&d.to_le_bytes());
        for slot in values.iter() {
            let q = ((*slot as f32) * id).round();
            let q = q.clamp(-128.0, 127.0) as i8;
            raw.push(q as u8);
            levels.push(q as u8);
        }
        scales.push(d_f32);
    }
    (raw, scales, levels)
}

/// The IEEE-754 half-to-single conversion ggml's `GGML_FP16_TO_FP32`
/// implements, including the subnormal handling.
fn oracle_source() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/oracle/q5_0/q5_0_oracle.c")
}

fn build_oracle(work_dir: &Path) -> Option<PathBuf> {
    let source = oracle_source();
    if !source.exists() {
        return None;
    }
    let binary = work_dir.join("q5_0_oracle");
    for compiler in ["cc", "gcc", "clang"] {
        let mut command = Command::new(compiler);
        command.args(["-O2", "-ffp-contract=off"]);
        if cfg!(target_arch = "aarch64") {
            // The reference is ggml's ARM kernel, whose per-lane integer
            // dot grouping decides how the F32 accumulation rounds. The
            // repository's kernel uses the native `vdotq_s32`, so both
            // sides have to be compiled with the `dotprod` extension for
            // the comparison to mean anything.
            command.arg("-march=armv8.2-a+dotprod");
        }
        let status = command
            .arg("-o")
            .arg(&binary)
            .arg(&source)
            .status();
        match status {
            Ok(status) if status.success() => return Some(binary),
            // A compiler that fails to build the oracle is a *failure*, not
            // a skip: silently skipping would report "ok" for a test that
            // compared nothing, which is exactly the kind of green that
            // hides a bug.
            Ok(status) => panic!("{compiler} failed to build the Q5_0 oracle: {status}"),
            Err(_) => continue,
        }
    }
    None
}

#[test]
fn q5_0_dot_matches_the_ggml_arm_reference() {
    if !cfg!(target_arch = "aarch64") {
        eprintln!("SKIP: the Q5_0 reference is the ARM NEON kernel");
        return;
    }
    let work_dir = std::env::temp_dir().join("rmi_q5_0_oracle");
    std::fs::create_dir_all(&work_dir).expect("create work dir");

    let Some(oracle) = build_oracle(&work_dir) else {
        eprintln!("SKIP: no C compiler available to build the Q5_0 oracle");
        return;
    };

    const N_OUT: usize = 8;
    const N_IN: usize = 256;
    let weights = build_weights(N_OUT, N_IN);
    let (activations, scales, levels) = build_activations(N_IN);

    let weights_path = work_dir.join("weights.bin");
    let acts_path = work_dir.join("activations.bin");
    let out_path = work_dir.join("oracle.f32");
    std::fs::write(&weights_path, &weights).expect("write weights");
    std::fs::write(&acts_path, &activations).expect("write activations");

    let status = Command::new(&oracle)
        .arg(&weights_path)
        .arg(&acts_path)
        .arg(&out_path)
        .arg(N_OUT.to_string())
        .arg(N_IN.to_string())
        .status()
        .expect("run oracle");
    assert!(status.success(), "oracle exited with {status}");

    let reference = std::fs::read(&out_path).expect("read oracle output");
    let reference: Vec<f32> = reference
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(reference.len(), N_OUT, "oracle produced the wrong length");

    // The repository's kernel takes the weights as raw bytes, the
    // activations as the raw Q8_0 levels (contiguous, without the
    // interleaved F16 scales), and the per-block activation scales as F32
    // (the F16-rounded values the kernel multiplies by).
    //
    // The comparison goes through the kernel's public entry point so that
    // it exercises the dispatch the model actually uses, not a private
    // helper.
    let mut ours = vec![0.0f32; N_OUT];
    let levels_f32: Vec<f32> = levels.iter().map(|&q| q as i8 as f32).collect();
    let kernel = rust_model_inference::ops::kernel::q5_0::Q5_0Kernel::new(&weights, N_IN, N_OUT);
    kernel.forward_prepared(
        &levels_f32,
        &levels,
        &scales,
        None,
        &mut ours,
        N_IN,
        N_OUT,
        0,
        1,
    );

    let mut mismatches = 0usize;
    for (index, (lhs, rhs)) in ours.iter().zip(reference.iter()).enumerate() {
        if lhs.to_bits() != rhs.to_bits() {
            if mismatches < 8 {
                eprintln!(
                    "mismatch at row {index}: rust={lhs:?} ({:#010x}) oracle={rhs:?} ({:#010x})",
                    lhs.to_bits(),
                    rhs.to_bits()
                );
            }
            mismatches += 1;
        }
    }
    assert_eq!(
        mismatches, 0,
        "{mismatches} of {N_OUT} Q5_0 dot products differ from ggml's ARM kernel"
    );
}
