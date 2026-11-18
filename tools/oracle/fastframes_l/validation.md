# FastFrames-L validation — 2026-11-18

**Experimental / partial acceptance. BF16 CPU alignment passed at 16×16 / 5 frames.**
The remaining format/backend/transport gates are incomplete in [matrix.json](matrix.json).
Observed summaries are retained in [observed-results.json](observed-results.json);
remote raw sidecars could not be retrieved after a second SSH interruption.
JEV and text-token streaming do not apply to this video-generation contract.

## Real-model evidence

The same BF16 GGUF was consumed by Rust CPU and independently reconstructed
pinned official forwards, using F32 activations/KV and the identical noise file.
All **411** full checkpoints passed: maximum relative RMSE **4.240734428259513e-5**,
minimum cosine **0.9999999991360493**. Limits, fixed before execution, are relative
RMSE ≤ 0.005 and cosine ≥ 0.9999. This covers umT5, every DiT block through the
multi-frame rollout, repaired K/V and decoder pixel tokens. Token IDs match.
It does not establish final RGB/MP4 quality or other storage formats.

- BF16 GGUF: 41,892,093,248 bytes,
  SHA256 `d0619fed092f8d33a15a5e3cb69827a131cfe3d32d253f768fc128242837cea9`.
- Native CPU CLI exited successfully and emitted the requested MP4.
- A separate seed-only BF16 CPU CLI run succeeded; the actual HTTP request returned
  200 and MP4. HTTP numerical comparison and decoded-video checks remain unverified.
- Strict Vulkan executed at least 4,737 device submissions. A 238-checkpoint prefix
  compared with CPU without failure (worst relative RMSE 3.988701595524663e-5).
  The last observed trace had 274 records. No completed device rollout, refiner,
  decoder or MP4 result was observed; this is **not a Vulkan acceptance pass**.
- F32 and F16 conversion completed; F16 SHA256 is
  `ab3ab0ca8e1294255a249d68fbfae2d1c59c00d34e8274aa1349dd829383fd39`.
  The F32 manifest was observed, but its hash was not retrieved. Q8 conversion was
  running when the replacement endpoint disconnected. Remaining formats are unverified.

The earlier “no real-model evidence” statement was superseded after the user
provided a replacement endpoint. That endpoint later refused connections too;
final task results and remote raw artifacts remain unavailable. The completed
BF16 CPU result remains evidence for its measured revision, not an unrun final rerun.

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
| Transport validation | Two tests passed for unsupported flags, geometry, unknown JSON fields and CLI/server boundaries. |
| Real tokenizer, Linux | 11 official fixture cases including Unicode, mojibake, HTML, ANSI and truncation; RED on the incorrect vocabulary assumption, then GREEN. Tokenizer has 256,300 entries, embedding has 256,384 rows. |
| Python converter/comparator/streaming, Linux | 10 passed, including parallel/serial encoded-byte equality, streamed reload and F16/Q4_1 activation semantics. |
| Comparator noise gate, macOS | 3 passed; differing input noise is rejected even if inside floating tolerances. |
| macOS full regression | Base: 1611 passed / 13 failed / 117 ignored. Final modified: 1626 passed / identical 13 failures / 118 ignored. Includes final F32 borrowing and ignored-tokenizer checks. |
| Linux full regression | Base: 1581 passed / 14 failed / 118 ignored. Modified: 1595 passed / identical 14 failures / 118 ignored. Failure names compared exactly. |
| Latest Linux Vulkan/parity binaries | Built successfully before final F32-only mmap-loader repair. |
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
  additions have syntax/unit checks but await a full remote rerun.
- Found that the generic F32 loader copies all ~84 GB of weights. The model-local
  loader now uses the existing borrowed F32 API; a pointer-identity/invalid-storage
  test failed before repair, then passed with the other 12 model tests. Remote disconnect cause
  is unknown and no OOM diagnosis is claimed.

## Remaining acceptance

Recover the remote assets/logs; rerun after the final loader/oracle repairs.
Complete each independently decoded format oracle and CPU/Auto/strict Vulkan,
with actual CLI and HTTP separately. Compare shared noise, discrete tokens,
checkpoint counts and numeric results; decode MP4s. Run published-BF16 drift,
480p multi-frame quality, peak memory and clean same-host repeated CPU/Vulkan/reference
benchmarks. Existing trace-inclusive reference timings are not speedup evidence.
Keep Draft/Experimental until these gates pass.
