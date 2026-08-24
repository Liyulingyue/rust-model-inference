# Task 7 report: native Z-Image route and atomic PNG publication

## Status

`DONE_WITH_CONCERNS`.

The native CLI route, strict three-component pipeline, atomic PNG writer, and
removal of the provisional `pig.rs` implementation are complete and pass the
focused tests, library check, and release build. The required real 512x512
smoke did not finish within the bounded run: it was interrupted after 622.67
seconds while still doing CPU work, with no PNG or sibling temporary file.
This report therefore does not claim end-to-end image generation, practical
runtime, or numerical/oracle parity.

## User-visible control flow

1. CLI parsing and `validate_cli_options` run before model loading.
2. `z_image_cli_options` recognizes the presence of the text encoder or VAE,
   requires `--model`, `--text-encoder`, `--vae`, a non-empty `--prompt`, and
   `--out`, and resolves steps, resolution, and seed.
3. Z-Image rejects TTS, ASR/reference audio, multimodal input, embedding,
   logit/bench/profile modes, GPU, thinking, F32 KV, language, token/temperature
   generation controls, and raw embedding output before any component is
   opened.
4. `main` opens exactly the diffusion, text, and VAE sources, passes the
   resolved thread count to `run_z_image_cli`, and returns before generic model
   dispatch.
5. `ZImagePipeline::load` validates the exact DiT, Qwen3 text, and Flux VAE
   tensor signatures before constructing any component.
6. Generation executes Qwen3 layer-35 conditioning, validates complete
   2560-wide context rows, runs DiT denoising, releases the context, validates
   the `[16, resolution/8, resolution/8]` latent, decodes with Flux VAE, and
   checks the final RGB dimensions.
7. Only after a complete RGB result exists does the CLI encode and atomically
   publish the requested PNG.

Opening a Pig/Z-Image diffusion file through the generic route now exits with a
missing-components error. It no longer uses zero context, optional VAE, fallback
pixels, or an implicit `output.png`.

## Atomic PNG semantics

`write_png_atomically`:

- computes `width * height * 3` with checked arithmetic and rejects an exact
  byte-length mismatch before touching the destination;
- encodes the complete RGB PNG into memory before opening a temporary file;
- creates a hidden sibling through `OpenOptions::create_new(true)` with a
  bounded PID/counter suffix search;
- uses `write_all` and `sync_all`, closes the file, then renames it over the
  destination;
- on write, sync, or rename failure, removes only the temporary path created by
  this invocation;
- never deletes or truncates the pre-existing output on validation/encoding
  failure.

Focused tests cover invalid RGB preservation, a decodable successful PNG,
invalid/missing output paths, multiplication overflow, cleanup after rename
failure, and the absence of leftover sibling files.

## TDD history

### Planned RED

The router and publication tests were added before their production APIs.

```text
cargo test --lib app::image::tests -- --nocapture
exit 101
```

Compilation failed because `run_z_image_cli`, `write_png_atomically`, and the
pipeline helpers did not exist. This was the expected missing-interface RED.

### CLI rejection RED

The model-required and conflicting-mode cases were then tightened before their
validation logic. The conflict case initially returned `Ok(())`, and the
complete-components-without-model case also did not fail. After adding the
pre-load conflict and model checks, the CLI suite passed both contracts.

### GREEN

After wiring the native pipeline and writer:

```text
cargo test --lib app::image::tests -- --nocapture
exit 0; 5 passed, 0 failed

cargo test --lib app::cli::tests -- --nocapture
exit 0; 17 passed, 0 failed
```

## Fresh verification

All commands below were run after the final behavior and after restoring
unrelated formatter churn.

```text
cargo test --lib app::image::tests -- --nocapture
exit 0; 5 passed, 0 failed, 340 filtered out

cargo test --lib app::cli::tests -- --nocapture
exit 0; 17 passed, 0 failed, 328 filtered out

cargo test --lib models::diffusion::z_image:: -- --nocapture
exit 0; 51 passed, 0 failed, 5 ignored, 289 filtered out

cargo check --lib
exit 0; 89 warnings

cargo build --release --bin rust-model-inference
exit 0; release profile finished in 24.31s; 89 warnings

rustfmt --edition 2021 --check --config skip_children=true \
  src/main.rs src/app/image.rs src/models/diffusion/z_image/mod.rs
exit 0

git diff --check
exit 0
```

The ignored Z-Image tests require supplied GGUF environment variables and are
reported as ignored rather than counted as passing. The repository's broad
warning noise remains visible; it was not suppressed or treated as a clean
lint result. Pre-existing formatting drift in `src/app/cli.rs` and the export
lists was kept intact to avoid unrelated whole-file churn; new code is locally
formatted and the executable/model files pass the scoped formatter check.

Release-level routing checks used the freshly built binary:

```text
target/release/rust-model-inference \
  --model models/z-image-gguf/z-image-turbo-q8_0.gguf
exit 1
Inference error: Z-Image model requires --text-encoder, --vae, --prompt, and --out

target/release/rust-model-inference \
  --model /tmp/rmi-task7-missing-dit.gguf \
  --text-encoder /tmp/rmi-task7-missing-text.gguf \
  --vae /tmp/rmi-task7-missing-vae.gguf \
  --prompt fox --out /tmp/rmi-task7-should-not-exist.png --gpu
exit 2
Z-Image cannot be used with --gpu
```

The second command returned the CLI conflict before attempting to open the
missing model paths, and the requested output remained absent.

## Real-model functional smoke

The exact requested command was run with `/usr/bin/time -p` after confirming
that `/tmp/rmi-z-image-42.png` did not exist:

```text
cargo run --release --bin rust-model-inference -- \
  --model models/z-image-gguf/z-image-turbo-q8_0.gguf \
  --text-encoder models/z-image-gguf/qwen3_4b_f32-q8_0.gguf \
  --vae models/z-image-gguf/pig_flux_vae_fp32-f16.gguf \
  --prompt "A red fox sleeping beneath a pine tree" \
  --steps 8 --resolution 512 --seed 42 --threads 1 \
  --out /tmp/rmi-z-image-42.png
```

Observed behavior:

- all three components loaded in 66 ms;
- the process then remained CPU-bound at approximately 99-100% of one core;
- a late sample showed RSS 10,879,552 KiB, and earlier `vmmap` evidence was
  consistent with the pre-sized DiT scratch allocation rather than VAE or PNG
  publication;
- the command was interrupted at the agreed bounded threshold;
- `/usr/bin/time -p` reported `real 622.67`, `user 605.51`, `sys 3.21`;
- process exit was 1 due to SIGINT;
- the target PNG and `.rmi-z-image-42.png.tmp-*` sibling were both absent after
  termination.

Because the command did not exit successfully, `file /tmp/rmi-z-image-42.png`
was not run and the brief's expected 512x512 PNG result is not satisfied. This
is an explicit runtime/functional-smoke concern for the next gate, not parity
evidence.

## Scope and deletion

- Replaced `run_pig_image` with `run_z_image_cli`.
- Added the pipeline orchestration and atomic publisher.
- Added focused router/publication/pipeline/CLI tests.
- Removed both provisional Pig dispatch branches and all `Pig*` exports.
- Deleted `src/models/diffusion/pig.rs` and its module registration.
- A repository scan found no remaining code references to `run_pig_image`,
  `diffusion::pig`, `PigConfig`, `PigModel`, `PigVAE`, or `PigSession` outside
  documentation/report paths.

The intended commit contains only the requested Rust paths, the Pig deletion,
and this report. Existing `.omo/` and 2026-08-21 untracked documents remain
untouched and unstaged.

## Remaining concerns

- The correctness-first scalar CPU implementation did not complete the 8-step
  512x512 smoke within 622.67 seconds on one thread. Practical runtime and a
  successful full decode/publication remain unverified.
- No final latent checkpoint, RGB digest, prompt-sensitivity comparison, or
  pinned oracle tolerance was measured. Those remain Task 8 gates.
- The atomic writer itself is covered with small real PNG encode/decode tests,
  but the real-model run never reached it.
