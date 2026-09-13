# Task 6A report

Implemented host row-aware dispatch/recording, row-stride push constants, shader row decoding, regenerated SPIR-V/manifest, and accepted `--all-formats --rows N` in `vk_ops_check`.

Validation:

- `cargo test --lib --features vulkan batched_matmul`: passed (2 tests).
- `bash scripts/vulkan-shaders.sh update`: passed.
- `bash scripts/vulkan-shaders.sh check`: passed.
- `cargo test --example vk_ops_check --features vulkan --no-run`: passed.
- Real `cargo run --features vulkan --example vk_ops_check -- --all-formats --rows 3`: reached Apple M3 Max Vulkan device, then failed quantize tie-even check (`gpu=-21`, `cpu=-20` at index 0). This is unresolved and blocks claiming device parity.

Known limitation: the example accepts `--rows`, but its existing operator fixture remains single-row; the full device batch comparison is deferred with the runtime-owner phase.
