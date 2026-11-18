# FastFrames-L validation — 2026-11-19

**Experimental / partial acceptance. Final-code BF16/F32 CPU CLI and HTTP pass at 16×16 / 5 frames.**
The remaining format/backend/transport gates are incomplete in [matrix.json](matrix.json).
Observed summaries and exact transport commands are retained in
[observed-results.json](observed-results.json). Small reports are retrieved while
tests run; model weights and full tensor sidecars remain on remote storage.
JEV and text-token streaming do not apply to this video-generation contract.

## Real-model evidence

The same BF16 GGUF was consumed by Rust CPU and independently reconstructed
pinned official forwards, using F32 activations/KV and the identical noise file.
Both final-code **CLI and HTTP** pass all **411** full checkpoints: maximum relative
RMSE **2.794884733482099e-5**, minimum cosine **0.9999999997176138**.
Limits, fixed before execution, are relative
RMSE ≤ 0.005 and cosine ≥ 0.9999. This covers umT5, every DiT block through the
multi-frame rollout, repaired K/V and decoder pixel tokens. Token IDs match.
The decoded MP4 comparison also passes: RGB24 / 255 RMSE **0**, cosine
**0.9999999999999999**, 3,840 decoded bytes per video. Both implementations' RGB
was encoded with the same libx264 settings before decoding. This diagnostic case
does not establish 480p quality, longer cache reuse or other storage formats.

- BF16 GGUF: 41,892,093,248 bytes,
  SHA256 `d0619fed092f8d33a15a5e3cb69827a131cfe3d32d253f768fc128242837cea9`.
- Native CPU CLI exited successfully. The actual HTTP request returned 200,
  `Content-Type: video/mp4`, `Cache-Control: no-store`; both videos decoded correctly.
- Final strict Vulkan CLI also completed and passed all 411 checkpoints: maximum
  relative RMSE **2.5552589342221195e-5**, minimum cosine **0.9999999998047929**;
  decoded MP4 pixel RMSE **0**. Its 2,202 Vulkan operator records and 5,628 device
  submissions prove execution. Actual Vulkan HTTP is still pending.
- On an earlier host, strict Vulkan executed at least 4,737 device submissions. A 238-checkpoint prefix
  compared with CPU without failure (worst relative RMSE 3.988701595524663e-5).
  The last observed trace had 274 records. That incomplete result did not establish
  device acceptance; the final-code CLI run above now supersedes this limitation.
- Final F32 CPU CLI and HTTP both pass all 411 official checkpoints after the mmap
  repair (maximum relative RMSE 2.794884733482099e-5, minimum cosine
  0.9999999997176138), with decoded MP4 pixel RMSE 0. F32 SHA256 is
  `59a25d64eee0d6981a9f57db479291f31329d745f644d32c77d09d69938c2735`
  (83,742,203,200 bytes). F16 conversion completed; SHA256 is
  `ab3ab0ca8e1294255a249d68fbfae2d1c59c00d34e8274aa1349dd829383fd39`.
  (41,892,093,248 bytes). Remaining formats/backends retain separate matrix gates.

The earlier BF16 result (relative RMSE 4.240734428259513e-5) preceded final oracle
hardening. The 2026-11-19 run supersedes that limitation: source head
`b44d4b3019fc900ad17b76e61deee98e8b08bf71` passed final provenance, exact shared
noise, CLI/HTTP checkpoints and decoded-video checks. All eight pinned source and
tokenizer assets were downloaded and SHA256-verified directly on the new host.
Reports are retained under `target/fastframes/port31667/`; earlier engineering logs
remain under `target/fastframes/port36085/`. No disconnect is diagnosed as OOM.

## Assets and host

Official commit, ModelScope revisions, source hashes and schemas are pinned in
`contract.json` and `reference-files.json`. Download/conversion verify source hashes.
All source weights, GGUFs and model tensor sidecars stayed on the authorized Linux
host. No weights were downloaded to the Mac. ModelScope downloaded directly on the
host, so a Mac disk cache or streaming relay was unnecessary.

| Source | Bytes |
| --- | ---: |
| FastFrames-L DiT | 28,577,096,680 |
| KV refiner | 1,780,342,784 |
| Pixel decoder | 286,952,688 |
| Wan umT5 | 11,361,920,418 |

Host: 128 logical CPUs, ~503 GiB physical RAM, **55 GiB cgroup limit**
(59,055,800,320 bytes), AMD gfx1100 / RADV NAVI31 with ~48 GiB VRAM.
`/workspace` is 98 GB and shared; completed large source files and GGUFs were moved
to `/root/rust-model-inference-fastframes/models`, retaining source-path symlinks.
Only this task's disposable baseline build directory was removed after ENOSPC.
No other projects, system Python, shell rc, GPU reset or remote service restart changed.

## Engineering checks

| Check | Observed result |
| --- | --- |
| Native model tests, macOS arm64 | 13 passed / 1 ignored after the tokenizer and F32 borrowing fixes. |
| Final native model tests, fresh Linux host | 13 passed / 1 ignored on commit `f61e7833f6c86a834e3ae3c68d0b9eefcaee173b`, built with Vulkan/parity trace; the pinned real-tokenizer test passed separately. |
| Transport validation | Two tests passed for unsupported flags, geometry, unknown JSON fields and CLI/server boundaries. |
| Real tokenizer, Linux | 11 official fixture cases including Unicode, mojibake, HTML, ANSI and truncation; RED on the incorrect vocabulary assumption, then GREEN. Tokenizer has 256,300 entries, embedding has 256,384 rows. |
| Python converter/comparator/streaming, Linux | Final fresh-host run: 11 passed, including exact shared noise, parallel/serial encoded-byte equality, streamed reload and F16/Q4_1 activation semantics. Earlier host: 10 passed before the noise gate. |
| Comparator noise gate, macOS | 3 passed; differing input noise is rejected even if inside floating tolerances. |
| macOS full regression | Base: 1611 passed / 13 failed / 117 ignored. Final modified: 1626 passed / identical 13 failures / 118 ignored. Includes final F32 borrowing and ignored-tokenizer checks. |
| Linux full regression | Base: 1581 passed / 14 failed / 118 ignored. Modified: 1595 passed / identical 14 failures / 118 ignored. Failure names compared exactly. |
| Latest Linux Vulkan/parity binaries | All nine bins built successfully on the prior replacement host with the final F32 mmap-loader repair; inference/server rebuilt on the current host. The optional fat-LTO release build was deferred to free memory for acceptance. |
| GitHub CI | Run `44340649724` on source head `b44d4b301` passed Rustfmt, x86 dispatch regressions (including FastFrames model/transport/comparator checks), and Vulkan compile/shaders. |
| Cargo formatting and binary inventory | Formatting passed; original nine binary targets retained. No external Rust acceleration dependency added. |

Historical regression failures were compared by name; they are not newly passing
checks. Local logs live under ignored `target/fastframes/`, including baseline,
regression, tokenizer-independent checks and metadata inventories.

## Review and repairs

A separate review and subsequent self-review covered pinned official schedule,
cache writes, grouped refiner orientation/norm, causal decoder layout and both transports.
The shared Vulkan Conv3D rejects groups != 1, so the decoder depthwise convolutions
are explicitly CPU work; only projections/attention have device paths. Shared
attention may retry/repair on CPU, whereas explicit projection failures in strict
Vulkan return errors; this is a hybrid graph, not a fully device-resident graph.

- Fixed prompt cleaning and actual tokenizer vocabulary/hash validation.
- Bounded GPU weight uploads to eight DiT blocks; full F32 DiT cannot all reside in
  48 GiB. Device execution occurred, but final peak memory remains unmeasured.
- Kept output publication atomic/no-clobber, private temporary directories and the
  HTTP generation permit alive through cancellation until computation finishes.
- Removed Auto initialization-as-execution reporting; actual operators must appear
  in compute traces. CPU-only stages are documented.
- Streamed official modules one block at a time for the 55 GiB host. Added F16
  prepared activation rounding and Q4_1's independently rounded Q8_1 sum term.
- Enforced exact input noise for comparison, retained numeric tolerances for lossy
  compute, and added wrapper/tokenizer/dependency provenance. Final provenance
  additions passed the final BF16 CPU CLI/HTTP full-model rerun.
- Found that the generic F32 loader copies all ~84 GB of weights. The model-local
  loader now uses the existing borrowed F32 API; a pointer-identity/invalid-storage
  test failed before repair, then passed with the other 12 model tests. Remote disconnect cause
  is unknown and no OOM diagnosis is claimed.

## Remaining acceptance

Finish the running independent format oracles and CPU/Auto/strict Vulkan matrix,
with actual CLI and HTTP separately. Compare shared noise, discrete tokens,
checkpoint counts and numeric results; decode MP4s. Exercise at least nine frames
to consume repaired KV in a subsequent latent frame. Run published-BF16 drift,
480p multi-frame quality, peak memory and clean same-host repeated CPU/Vulkan/reference
benchmarks. Existing trace-inclusive reference timings are not speedup evidence.
Keep Draft/Experimental until these gates pass.
