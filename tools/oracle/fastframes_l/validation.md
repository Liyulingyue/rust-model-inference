# FastFrames-L validation — 2026-11-19

**Experimental / partial acceptance. BF16 and F32 CPU/Auto/Vulkan pass both transports at 16×16 / 5 frames. F16 passes checkpoints but fails decoded-video gates on all three backends. Quantized gates have failed. The 480p case passes checkpoints but fails decoded-video cosine and published-BF16 drift gates.**
All 54 format/backend/transport combinations completed: 12 passed and 42 failed,
as retained in [matrix.json](matrix.json).
Observed summaries and exact transport commands are retained in
[observed-results.json](observed-results.json). Small reports are retrieved while
tests run; model weights and full tensor sidecars remain on remote storage.
JEV and text-token streaming do not apply to this video-generation contract.

## Replacement-host oracle repair

Testing resumed on a replacement Linux instance after the previous instance was lost.
All eight pinned ModelScope source/tokenizer files were downloaded directly to
remote storage and SHA256-verified; restored BF16, F16 and Q8_0 GGUFs reproduce
the earlier hashes. No weights or full tensor sidecars were copied to the Mac.
The new environment uses Rust 1.100.0, PyTorch 2.9.1 ROCm and FFmpeg 9.0.2.
The earlier FFmpeg version was not recorded, so codec-dependent pixel changes
across hosts cannot establish a production fix. This host again shares its
16-core quota, 55 GiB cgroup and 48 GiB GPU with other work.

Self-review found an oracle activation-quantization defect: PyTorch's
`127. / maximum` rounds a reciprocal before multiplying. Against one direct F32
division, 4 of 3,200,000 integer codes differed on both CPU and GPU. A captured
32-value fixture quantized lane 7 to -75 instead of -74; all six Q-format
subtests and Q4_1's sum correction failed before repair. The wrapper now uses
explicit tensor/tensor division, including the scale division. All **12**
converter/comparator/streaming tests pass. Production Rust arithmetic, padding,
encoding parameters and acceptance thresholds are unchanged.

The corrected Q8_0 text-only diagnostic still fails final-output gates:
relative RMSE **0.02767814** for CPU/10 tokens, **0.02770654** for GPU/512-padding
and **0.03131954** for GPU/10 tokens. Padding alone therefore does not explain
the failure. Temporary trace-only native instrumentation was restored after
capture, and the preserved production binary is used for real transports.
Replaying 28 individual official linear projections at layers 0, 1, 12 and 23
with the same captured native inputs gives maximum relative RMSE
**4.959657988753818e-7**. The first norm differs by only **3.6967634e-8** relative
RMSE, yet one of 40,960 activation codes changes, producing prepared-input
relative RMSE **0.0004839344**. This demonstrates quantization sensitivity at
the first split; it neither proves the whole graph correct nor overrides its
failed gates. The earlier 54-case matrix and per-format metrics remain historical
evidence from the pre-repair wrapper; corrected-wrapper reruns and exact
provenance are recorded separately in [replacement-host-results.json](replacement-host-results.json).

Corrected-wrapper CPU CLI and HTTP reruns have completed for all three restored
formats. BF16 passes all 411 checkpoints and decoded-video RMSE 0. F16 again
passes all checkpoints but fails decoded RMSE **0.014410152859932579** and cosine
**0.9995483949878057**. Q8_0 still first fails at `fastframes.text` (relative RMSE
**0.027697485017157433**), reaches full-model relative RMSE
**0.07609434268844385**, and fails decoded RMSE **0.02451650235264669**. Both
transports reproduce the same metrics. These supersede the original wrapper's
Q8_0 CPU values for this rerun; the original failed measurements remain retained.

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
- Final strict Vulkan CLI and HTTP completed and passed all 411 checkpoints each: maximum
  relative RMSE **2.5552589342221195e-5**, minimum cosine **0.9999999998047929**;
  decoded MP4 pixel RMSE **0**. Its 2,202 Vulkan operator records and 5,628 device
  submissions per transport prove execution. These counts cover instrumented
  projections, not every operation in the hybrid graph.
- Real Auto CLI and HTTP pass all 411 checkpoints each: maximum relative RMSE
  **2.5236295358634716e-5**, minimum cosine **0.9999999997347359**, decoded-video
  RMSE **0**. Each has 143 Vulkan and 2,059 CPU projection trace records.
- With process-local Vulkan ICD paths pointing to a missing file, Auto CLI and
  HTTP pass all 411 checkpoints and decoded-video RMSE 0, with zero device and
  2,203 CPU projection records. Strict Vulkan exits 2 with a driver-initialization
  error and creates no output. This fallback check is separate from device acceptance.
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
  (41,892,093,248 bytes). All nine formats retain separate matrix results.
- Final F32 strict Vulkan CLI and HTTP also pass all 411 checkpoints each: maximum relative
  RMSE **4.0885956235754525e-5**, minimum cosine **0.9999999991869644**, decoded-video
  RMSE **0**. It records 2,202 Vulkan projections and 5,628 submissions, uploads
  366,407,058,496 bytes, and reaches a device-allocator peak of 29,011,415,008 bytes.
- Final F32 Auto CLI and HTTP pass all 411 checkpoints each: maximum relative
  RMSE **4.775628860976526e-5**, minimum cosine **0.9999999988828286**, decoded-video
  RMSE **0**. Each has 143 Vulkan and 2,059 CPU projection records, 395 device
  submissions, and a device-allocator peak of 28,901,314,496 bytes.

### F16 decoded-video failure

Both CPU transports generate valid MP4s and pass all 411 checkpoints (maximum
relative RMSE **0.001728212584007973**, minimum cosine **0.9999988745859908**).
Both fail the unchanged decoded-video limits: RMSE **0.014410152859932579** > 0.01
and cosine **0.9995483949878057** < 0.9999. The matrix records **失败**.

Reconstructed native RGB before encoding differs from official RGB by at most
**1/255** (138 of 3,840 components, RMSE **0.0007434191611815244**, cosine
**0.9999988157999307**). Independently re-encoding those native pixels produces
exactly the decoded production pixels. This identifies lossy H.264 amplification
of small F16 differences; it does not establish an output-layout or encoder bug.
No thresholds or production arithmetic were changed to hide the failure.

### Quantized implementation failures

Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K fail the unchanged checkpoint and decoded-video gates in both
transports on CPU, Auto and strict Vulkan. Q8_0 CPU first fails at `fastframes.text`
(relative RMSE **0.024168239126007365**, cosine **0.9997082747515944**); worst full
checkpoint relative RMSE is **0.06915133836694705**. The first 24 text-block
checkpoints individually pass, with final pre-normalization relative RMSE
**0.00246098840286423**. Independent final-RMSNorm reconstruction agrees with
each implementation within relative RMSE **6.5e-8**, identifying amplification
of prior differences rather than a final-normalization implementation error.
The remaining source of quantized-path drift is unresolved; no threshold or
production accumulator was changed to make these cases pass. Exact per-format
and per-backend values remain in `transport-results.json`.

A separate pinned-official text-block replay uses captured native inputs at layers
1, 6, 12, 17, 18, 22 and 23. All 21 isolated block comparisons pass: maximum
relative RMSE is **3.2239118316123155e-5** for Q8_0, **5.1645542778042046e-5**
for Q4_K and **9.054559811931801e-5** for Q6_K. Replaying captured reference
inputs on short-sequence CPU instead of the acceptance oracle's padded GPU
sequence differs by at most **1.4278859959984067e-4**. This narrows the observed
problem toward accumulated drift; it does not prove the complete quantized path
correct or replace any failed full-model gate. The source hash, per-layer metrics
and replay boundary are retained in `supplemental-results.json`.

Q4_K CPU first exceeds tolerance at `fastframes.text.18` (relative RMSE
**0.005935670667030438**); its worst full-model relative RMSE is
**0.6519903473918246**, and decoded-video RMSE is **0.03717332830546875**.
Its strict Vulkan and Auto transports also fail; this is not accepted as
ordinary storage-quality loss. The independently reconstructed Q4_K recipe's
raw-RGB drift versus the BF16 controlled baseline is separately **0.061805**.

Q5_K CPU first fails at final text output (relative RMSE **0.046764759439796216**),
with worst full-model relative RMSE **0.11004985784146036** and decoded-video
RMSE **0.031042289968687298**. Q6_K CPU first fails at text block 18 (relative
RMSE **0.005431247993474857**), with worst full-model relative RMSE
**0.12235495926964259** and decoded-video RMSE **0.029655849925988575**.
Both formats also fail on Auto and strict Vulkan through CLI and HTTP.

F16 Auto and strict Vulkan also pass all 411 checkpoints in both transports,
but fail decoded-video gates. Their maximum checkpoint relative RMSE is
**0.001529806115542585** and **0.0013954515093433373** respectively; decoded-video
RMSE is **0.011841220411469965** and **0.013809123119434285** respectively.

### Spatial and repaired-cache diagnostic

BF16 CPU at **32×32 / 9 frames** passes **578** checkpoints, maximum relative RMSE
**7.116039816112045e-5**, minimum cosine **0.9999999975714114**. Decoded-video RMSE
is **0.007592180372839715**, cosine **0.9999269925418053**, passing the same limits.
This exercises nonzero spatial positions and consumes repaired KV in a later
latent frame. The same-input published-BF16 recipe drift also passes its separate RMSE limit:
**0.011251397167858146** ≤ 0.08. The completed 832×480 / 5-frame strict Vulkan case is recorded separately below.
Commands and supplementary measurements are in [supplemental-results.json](supplemental-results.json).

### Completed 480p diagnostic — failed quality gates

BF16 strict Vulkan at **832×480 / 5 frames** generated a valid MP4 and passed
all **411** full-model checkpoints: maximum relative RMSE
**0.003756432053285011**, minimum cosine **0.9999929447946526**. Decoded-video
RMSE **0.008177586129195849** passes 0.01, but cosine
**0.9998758482355368** fails 0.9999. The case therefore remains **失败**.

Before encoding, native RGB versus the controlled official RGB has RMSE
**0.000387182768588293**, cosine **0.9999997259876185**; 50,845 of 5,990,400
components differ, by at most **6/255**. Independently unpatchifying the official
pixel tokens reproduces its RGB exactly. Independently rebuilding and encoding
the native pixel tokens reproduces the production MP4's decoded pixels exactly.
This supports H.264 amplification of small input differences rather than a
layout/encoder wiring error; the original failed compressed-video gate is retained.

The controlled and published-BF16 recipes also exceed their separate quality
drift limit: raw-RGB RMSE **0.10040362877346713** > **0.08**, cosine
**0.9815560650382746**. Native raw RGB versus the same published recipe is
**0.10039486713436548**. This recipe gap is present in the official forwards
independently of the native encoder; published-quality acceptance remains failed.
The published run used benchmark mode (RGB only, no checkpoint claim).

Native generation exited 0 after **10,845.66 seconds** using traced release-fast;
the controlled traced official process took **535.40 seconds** and the published
benchmark-mode process **115.54 seconds**. Different profiles, tracing/loading
boundaries and concurrent host work prevent a clean speedup or isolated latency
claim. The native trace records **2,211** Vulkan projection events, **5,640**
submissions and a device-allocator peak of **17,215,993,808 bytes**. These are
projection/device-allocation measurements, separate from full host/process peaks.

The ahead-of-native references used the exact native noise and pinned assets.
Only a verified duplicate reference process was stopped after native exit 0;
no native computation or unrelated project was signalled. Completed reference
manifests, hashes, reuse boundary and both failed quality gates are retained in
`supplemental-results.json`; full payloads remain remote.

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
| Transport validation | Two unit tests passed. Real HTTP checks also return 400 for unknown model, empty prompt, invalid width/frames and 422 for unknown fields. Real CLI rejects invalid width, steps and KV override without creating output files. |
| Real tokenizer, Linux | 11 official fixture cases including Unicode, mojibake, HTML, ANSI and truncation; RED on the incorrect vocabulary assumption, then GREEN. Tokenizer has 256,300 entries, embedding has 256,384 rows. |
| Python converter/comparator/streaming, Linux | Final fresh-host run: 11 passed, including exact shared noise, parallel/serial encoded-byte equality, streamed reload and F16/Q4_1 activation semantics. Earlier host: 10 passed before the noise gate. |
| Comparator noise gate, macOS | 3 passed; differing input noise is rejected even if inside floating tolerances. |
| macOS full regression | Base: 1611 passed / 13 failed / 117 ignored. Final modified: 1626 passed / identical 13 failures / 118 ignored. Includes final F32 borrowing and ignored-tokenizer checks. |
| Linux full regression | Base: 1581 passed / 14 failed / 118 ignored. Modified: 1595 passed / identical 14 failures / 118 ignored. Failure names compared exactly. |
| Latest Linux Vulkan/parity binaries | All nine bins built successfully on the prior replacement host with the final F32 mmap-loader repair; inference/server rebuilt on the current host. Fat-LTO release compilation subsequently completed; binary hashes are in `supplemental-results.json`. |
| GitHub CI | Run `44390529400` on head `37a996adc` passed Rustfmt, x86 dispatch regressions (including FastFrames model/transport/comparator checks), and Vulkan compile/shaders. |
| Cargo formatting and binary inventory | Formatting passed; original nine binary targets retained. No external Rust acceleration dependency added. |

Historical regression failures were compared by name; they are not newly passing
checks. Local logs live under ignored `target/fastframes/`, including baseline,
regression, tokenizer-independent checks and metadata inventories.

### Real HTTP busy and disconnect lifecycle

A private-port BF16 CPU server (`--threads 4`) passed a real client-disconnect
check. A concurrent request returned **429**; immediately after closing the first
client with a TCP reset, the next request still returned **429**. After that
computation finished, a new request returned **200** and decoded to the expected
**3,840 RGB bytes**. The trace contains two noise/pixel-token generations, with
the first pixel-token event before the second token-ID event. This confirms the
permit remained held through cancelled computation; the two-generation check
took **138.29 seconds** under shared load. Commands, response timings and trace
ordering are retained in `supplemental-results.json#http-concurrency-results.json`.

### Release timing samples

The earlier fat-LTO deferral is superseded by successful release compilation.
At 16×16 / 5 frames, three native CPU processes with tracing disabled took
**63.40 / 73.74 / 78.48 seconds** (median **73.74 seconds**, including load and
encoding). Their process peak RSS ranged from **27.95 to 31.34 GB**; this is
process RSS, separate from the shared cgroup and device totals. The completed
CPU, Auto and strict Vulkan each completed three measured processes. Auto took
**99.86 / 106.49 / 110.53 seconds**; strict Vulkan took
**544.67 / 553.27 / 824.41 seconds**. Official repeats have also completed.
The completed native timing videos match controlled BF16 decoded pixels exactly.
These samples share the host
with acceptance and conversion work, so they are diagnostic timings, without
a clean performance or speedup claim. Per-process commands and memory boundaries
are retained in [performance.json](performance.json).
Strict Vulkan's traced BF16 case uploads **183.20 GB** and submits **5,628**
device commands, versus Auto's **13.86 GB** and **395** submissions. Bounded
eight-block caches trade memory for repeated uploads, and strict mode offloads
small projections that Auto leaves on CPU. These measured costs explain part
of the gap; this is not a controlled attribution benchmark. At 480p, the shared
full CPU attention check adds a separate substantial cost.

| Case | Median whole-process seconds | Median reference generation-region seconds |
| --- | ---: | ---: |
| Native CPU | 73.74 | Not separately measured |
| Native Auto | 106.49 | Not separately measured |
| Native strict Vulkan | 553.27 | Not separately measured |
| Official controlled, same BF16 GGUF | 269.16 | 164.54 |
| Official published BF16 recipe | 123.76 | 55.77 |

Each case has one warmup and three measured processes. Native requested 24
threads, reference requested 8; the shared cgroup has a 16-core quota. Reference
whole-process times include source hashing and initialization, while its
generation region includes streamed weights and RGB writes. Native load and
generation are not separately timed. Assigned-device VRAM samples include other
processes. These limits prevent a clean speedup claim. All four controlled timing
outputs reproduce the acceptance raw RGB; published-BF16 raw-RGB drift is
**0.004773227941892371** in all four runs, passing its separate **0.08** gate.
This is diagnostic quality evidence, without benchmark-mode checkpoint claims.

## Review and repairs

A separate review and subsequent self-review covered pinned official schedule,
cache writes, grouped refiner orientation/norm, causal decoder layout and both transports.
The shared Vulkan Conv3D rejects groups != 1, so the decoder depthwise convolutions
are explicitly CPU work; only projections/attention have device paths. Shared
attention computes a full CPU reference even for a healthy GPU result (the
existing TODO-044 correctness check), and may retry/repair on CPU. This retains
substantial CPU attention work at 480p. Explicit projection failures in strict
Vulkan return errors; this is a hybrid graph, not a fully device-resident graph.

Shared follow-up S1 belongs to the `src/ops/nn.rs`, `src/ops/kernel/vulkan.rs`
and `src/vulkan/ops.rs` maintainers: add grouped/depthwise device Conv3D, which
the existing kernel rejects. Shared follow-up S2 belongs to the attention runtime
maintainers in `src/ops/nn.rs` and `src/ops/kernel/vulkan.rs`: repair TODO-044 before
removing duplicate CPU computation. A model-local adapter cannot supply those
missing shared capabilities; this PR retains the CPU paths and correctness check.

- Fixed prompt cleaning and actual tokenizer vocabulary/hash validation.
- Bounded GPU weight uploads to eight DiT blocks; full F32 DiT cannot all reside in
  48 GiB. Device allocator peaks are retained per transport; full-size host/process peaks remain unmeasured.
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

All scheduled format oracles and native CPU/Auto/strict Vulkan CLI/HTTP cases
completed: the tiny matrix has 12 passed and 42 failed transports. The nine-frame
spatial/KV and real HTTP busy/disconnect checks pass. The completed 480p case
passes all checkpoints but fails compressed-video cosine and published-BF16 drift.
Retain and resolve the F16 decoded-video, six quantized numeric/video, and 480p
quality failures before readiness; isolated block/raw-pixel checks do not replace
them. Continue investigation at the first prepared-activation split for quantized
paths, and validate any proposed codec/precision-recipe repair against the fixed
output gates rather than changing thresholds or production accumulation to match bits.

Shared follow-ups S1/S2 above retain the missing grouped device convolution and
TODO-044 attention-runtime work with their module owners. Full host/process peak
memory and clean isolated benchmarks remain unverified. The 832×480/81-frame
default has about 53.67 GB of F32 KV alone and is not exercised under this shared
55 GiB cgroup. Keep Draft/Experimental until the failed gates are resolved.
