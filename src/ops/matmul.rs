// Phase 2.7-final: Q8_0 production matmul functions moved to
// `kernel::q8_0::parallel`. Re-export them under their historical names
// so existing callers (`bin/server.rs`, `bin/micro_bench.rs`,
// `app/embedding.rs`, `app/text.rs`) keep working without rewrites.
pub use super::kernel::q8_0::parallel::{
    matmul_q8_0, matmul_q8_0_parallel, matmul_q8_0_quantized,
    matmul_q8_0_quantized_dynamic, matmul_q8_0_quantized_parallel,
    matmul_q8_0_quantized_parallel_rows,
};
pub use super::kernel::q8_0::legacy::{
    matmul_q8_0_via_q8, matmul_q8_0_via_q8_parallel,
};
pub use super::kernel::q8_0::batch::{matmul_q8_0_batch, MatmulTask};

// Phase 2.7-final cleanup: `ProcessedWeight` enum has been retired.
// Weight handling now flows entirely through the `Kernel` trait in
// `ops::kernel::*`. Per-quant weight structs (Q4_0 / Q4_1 / Q6_K /
// Q4_K / Q5_K) live below as plain data holders used by the Kernel
// impls.

#[cfg(test)]
#[path = "matmul_tests.rs"]
mod neon_tests;
