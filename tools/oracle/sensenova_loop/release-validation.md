# SenseNova Loop 9B release validation

Date: 2026-11-19 (Asia/Shanghai). Base: `44527e5ad`.
Status: **Experimental / partial adaptation**.
The 108-row [coverage matrix](release-matrix.csv) currently has 30 passed,
78 failed and 0 pending combinations. Implementation and
verification status are separate. [Machine-readable results](release-results.json)
retain numeric metrics, interface comparisons, exported hashes and failure names.
The counts retain the historical pre-correction snapshot; the corrected shared
GPU path has not completed full-model acceptance.

**GPU continuation blocked:** SSH to `36.150.116.206:33162` works, but real Qwen GPU inference and tiny F32/Q8 Vulkan operator probes fail with `device lost`. The retained interface snapshot is incomplete. See `continuation` and `validation_blockers` for raw errors and baseline probes. The 108-row aggregate does not mean the interface matrix is complete.

## Contract and assets

- ModelScope `SenseNova/SenseNova-Loop-9B`, revision `3fa2faf34a8ed0dba63d320e2312e2b900a7e8be`;
  [all 25 pinned source files](release-source-manifest.json), 19,339,043,365 bytes.
- Config SHA256 `614dbd44b3a7fba8cd3adf5118b91ee40d51765b874de9dfadf67ea4e16ba68e`.
- Official modeling source SHA256
  `26e11d3522aa730c0ba031340ce42632bc2834da8cc0b6804006bb207d979a5b`.
- BF16 source projections, F32 mHC, 32 physical layers and 56 visits:
  `0..4 + (4..28) * 2 + 28..32`, with independent KV/conv/SSM state per visit.
- YaRN factor 2, original context 262144, configured maximum 524288, theta 1e7,
  head width 256, rotary prefix 64, interleaved sections `[11,11,10,0]`.
- The unused `(4096,)` F32 training `loop_gate.logits` is validated and omitted;
  the pinned official forward does not read it.
- Norms, recurrent parameters, mHC routers and the vision projector stay F32.
  Only the 202 text projections use the selected storage format.
  Preview and release decoder/projector config identities must match.

Numeric/regression/performance host: `j-oxuoqo5tstkquvgqhepkaa5cm6`, SSH port 31667, remote repository
`/workspace/rust-model-inference-sensenova-loop-9b`. x86_64, 16-CPU quota,
55 GiB cgroup RAM, gfx1100 / RADV NAVI31 with approximately 48 GiB VRAM.
Rust 1.94.1, PyTorch 2.9.1 ROCm 7.2 and transformers 5.12.0.
Models are under `/root/rmi-sensenova-loop-9b-models`, symlinked as `models`.
All weights were downloaded directly from ModelScope on this host and checked
against the manifest. No source checkpoint or full-model GGUF was stored on Mac.
Dependencies remain in repository `.venv` / `target`; no system package,
shell configuration, GPU reset or unrelated task was changed.

Earlier ports 35167 and 36085 became unavailable, followed by port 31667 during
the final interface checks. The retained snapshot had 14/27 complete groups.
The continuation host is `j-5fyejyon53kz4ytgmxuf5ne2m5`, SSH port 33162. It completed 0 additional interface groups. All 25 pinned source files and 7 re-exported text files plus projector were checked against the original sizes/SHA256. The final CLI/server build passed. Per-group host provenance is recorded in JSON and CSV; numeric, regression and timing evidence below remains attributed to port 31667. No throughput comparison combines the two hosts.

All timing tables below come from port 31667.

| Text storage | Bytes | SHA256 |
| --- | ---: | --- |
| F32 | 35,838,774,432 | `f90fd3400b3e04a4ffba3b0577dcb0df39bbb9709450663080e6df8ed98ed6cd` |
| F16 | 17,945,873,600 | `c48ffd5deed96ef5a63b41efa6f320b72d7a115c904875cc3a729cfc19731762` |
| BF16 | 17,945,873,600 | `c02e743afe65f9d09ccb46506333c38b1ea8fd9b78802aff786ce6df0299129a` |
| Q8_0 | 9,558,576,320 | `f358fc13a395fbb30553ce1f5a7e29aa2dda442744d949f36f282e8ba98f422f` |
| Q4_0 | 5,085,351,104 | `c86aa59de1fdd0ad8c20eb2bb24ad9fe0a43701608e57296199ba0b25d72c721` |
| Q4_1 | 5,644,504,256 | `603f629d3ca15a874e8226537f4ce40b3031e9421d09495f82549c1059b987b3` |
| Q4_K | 5,085,351,104 | `ece489d129d4150349599b08f98cbbfd31346a028f7ea0063188458d35823ba1` |
| Q5_K | 6,203,657,408 | `e5e7c374a271dcd89c7096d0652d74ff45f27a49a9a81cac697f5627f1c31613` |
| Q6_K | 7,391,857,856 | `144f8be287eec370c2afb314d201787af7166d0eb2c0a29528d4e05ba675997d` |

The projector's size/hash and export timing/RSS are in the result JSON.
The temporary six-worker K-export driver only splits independent complete
blocks; a fixed random probe of all three K formats was byte-identical to the
serial encoder. Production conversion still streams 16 MiB source chunks.

## Recorded numeric gates and results (before the mapped-range correction)

Tokenizer, shape, checkpoint count, special tokens and greedy IDs are exact.
Arrays must be finite. Floating-point arithmetic is compared with these gates,
declared before the low-precision full-model comparisons:

```text
F32: max_abs <= 1e-3 + 1e-4 * reference_max_abs
     RMSE    <= 1e-5 + 1e-4 * reference_RMS
lossy: max_abs <= 0.1 + 0.05 * reference_max_abs
       RMSE    <= 0.02 + 0.02 * reference_RMS
       cosine  >= 0.995
```

No bitwise PyTorch/SIMD/GPU claim is made. Each low-precision reference loads the
same GGUF through independent gguf-py decoding, then runs the official F32 eager
forward; this measures implementation error separately from quantization loss
relative to the original F32 weights. Embedding lookup can remain on CPU without
changing its values. `--host-inputs` places vision, embeddings and output
projection on CPU with an ROCm decoder, fitting current shared VRAM. Earlier
F32 references predate CPU-head placement and the placement-recording hook.
New references write actual device placement to `execution.json`, retained in
the result JSON. The original F32 text reference used all-ROCm. The exact
recorded placement of newer references takes precedence over command flags.

| Storage | Fixture | Requested | Raw checks / failed | Same-prefix checks / failed | Worst raw relative RMSE | Greedy |
| --- | --- | --- | ---: | ---: | ---: | --- |
| F32 | text | cpu | 570 / 0 | 570 / 0 | 1.23433e-05 | exact |
| F32 | text | auto | 570 / 0 | 570 / 0 | 1.1838e-05 | exact |
| F32 | text | vulkan | 570 / 0 | 570 / 0 | 1.1838e-05 | exact |
| F32 | image | cpu | 172 / 46 | 172 / 46 | 0.000369287 | exact |
| F32 | image | auto | 172 / 46 | 172 / 46 | 0.000368799 | exact |
| F32 | image | vulkan | 172 / 46 | 172 / 46 | 0.000368799 | exact |
| F16 | text | cpu | 570 / 0 | 570 / 0 | 0.00217805 | exact |
| F16 | text | auto | 570 / 0 | 570 / 0 | 0.00185394 | exact |
| F16 | text | vulkan | 570 / 0 | 570 / 0 | 0.00185394 | exact |
| F16 | image | cpu | 172 / 0 | 172 / 0 | 0.00255879 | exact |
| F16 | image | auto | 172 / 0 | 172 / 0 | 0.00252029 | exact |
| F16 | image | vulkan | 172 / 0 | 172 / 0 | 0.00252029 | exact |
| BF16 | text | cpu | 570 / 0 | 570 / 0 | 1.08455e-05 | exact |
| BF16 | text | auto | 570 / 0 | 570 / 0 | 1.16453e-05 | exact |
| BF16 | text | vulkan | 570 / 0 | 570 / 0 | 1.16453e-05 | exact |
| BF16 | image | cpu | 172 / 0 | 172 / 0 | 0.000101511 | exact |
| BF16 | image | auto | 172 / 0 | 172 / 0 | 9.99252e-05 | exact |
| BF16 | image | vulkan | 172 / 0 | 172 / 0 | 9.99252e-05 | exact |
| Q8_0 | text | cpu | 570 / 121 | 570 / 121 | 0.0822453 | exact |
| Q8_0 | text | auto | 570 / 102 | 570 / 102 | 0.0860444 | exact |
| Q8_0 | text | vulkan | 570 / 102 | 570 / 102 | 0.0860444 | exact |
| Q8_0 | image | cpu | 172 / 39 | 172 / 39 | 0.106571 | exact |
| Q8_0 | image | auto | 172 / 43 | 172 / 43 | 0.0918748 | exact |
| Q8_0 | image | vulkan | 172 / 43 | 172 / 43 | 0.0918748 | exact |
| Q4_0 | text | cpu | 570 / 116 | 570 / 116 | 0.099507 | different |
| Q4_0 | text | auto | 570 / 111 | 570 / 111 | 0.098862 | exact |
| Q4_0 | text | vulkan | 570 / 111 | 570 / 111 | 0.098862 | exact |
| Q4_0 | image | cpu | 172 / 49 | 172 / 49 | 0.089328 | exact |
| Q4_0 | image | auto | 172 / 132 | 58 / 18 | 1.28148 | different |
| Q4_0 | image | vulkan | 172 / 132 | 58 / 18 | 1.28148 | different |
| Q4_1 | text | cpu | 570 / 120 | 570 / 120 | 0.0987963 | exact |
| Q4_1 | text | auto | 570 / 112 | 570 / 112 | 0.109673 | exact |
| Q4_1 | text | vulkan | 570 / 112 | 570 / 112 | 0.109673 | exact |
| Q4_1 | image | cpu | 172 / 56 | 172 / 56 | 0.1261 | exact |
| Q4_1 | image | auto | 172 / 50 | 172 / 50 | 0.117876 | exact |
| Q4_1 | image | vulkan | 172 / 50 | 172 / 50 | 0.117876 | exact |
| Q4_K | text | cpu | 570 / 405 | 570 / 405 | 0.18309 | exact |
| Q4_K | text | auto | 570 / 406 | 570 / 406 | 0.178606 | exact |
| Q4_K | text | vulkan | 570 / 406 | 570 / 406 | 0.178606 | exact |
| Q4_K | image | cpu | 172 / 154 | 58 / 40 | 0.885296 | different |
| Q4_K | image | auto | 172 / 156 | 58 / 42 | 0.881074 | different |
| Q4_K | image | vulkan | 172 / 156 | 58 / 42 | 0.881074 | different |
| Q5_K | text | cpu | 570 / 406 | 570 / 406 | 0.192269 | exact |
| Q5_K | text | auto | 570 / 406 | 570 / 406 | 0.198285 | exact |
| Q5_K | text | vulkan | 570 / 406 | 570 / 406 | 0.198285 | exact |
| Q5_K | image | cpu | 172 / 132 | 172 / 132 | 0.176897 | exact |
| Q5_K | image | auto | 172 / 131 | 172 / 131 | 0.173894 | exact |
| Q5_K | image | vulkan | 172 / 131 | 172 / 131 | 0.173894 | exact |
| Q6_K | text | cpu | 570 / 398 | 570 / 398 | 0.182046 | exact |
| Q6_K | text | auto | 570 / 396 | 570 / 396 | 0.195868 | exact |
| Q6_K | text | vulkan | 570 / 396 | 570 / 396 | 0.195868 | exact |
| Q6_K | image | cpu | 172 / 127 | 172 / 127 | 0.177998 | exact |
| Q6_K | image | auto | 172 / 131 | 172 / 131 | 0.184914 | exact |
| Q6_K | image | vulkan | 172 / 131 | 172 / 131 | 0.184914 | exact |

After greedy divergence, subsequent inputs differ. Raw comparisons and their
original failures are retained, with `same_input_prefix` identifying checks
through the first divergence. The same-prefix columns bound implementation
comparison to equal inputs; later raw errors include the changed token prefix.
Greedy divergence still fails acceptance. No gate or prior failed result was
removed; the comparator's regression test covers this distinction.

F32 text produces `[9419,0,2500,628,353,7543,488,3242]` (`Hello! How can I assist you today`),
Chinese `[109430]` and Unicode `[34]` in all three native backends and the official reference.
Auto and strict Vulkan have observed `resident_hybrid_decoder backend=Vulkan`
and nonzero submission counts. Low-precision formats retain their encoded
storage on the CPU/device paths; computation, activation preparation and F32 KV
are described separately in the matrix. Vision is CPU-side for every backend.

### Storage quality versus original F32

These are reference-to-reference observations, separate from native same-GGUF
implementation gates. Prefill logits below always have the same input prefix.
Later logits are compared only through the first greedy divergence; subsequent
inputs differ. Generated IDs/text and all comparable steps are retained in the
result JSON. The F32 quality baseline now uses the same CPU vision/embedding/head
and ROCm decoder placement as each precision reference, with the baseline path
recorded alongside its metrics. No broad task-quality or compression-quality
pass is implied.

| Storage/fixture | Case | Prefill max abs | Relative RMSE | Cosine | First greedy difference step |
| --- | ---: | ---: | ---: | ---: | --- |
| q4_0-image | 0 | 5.69387 | 0.393946 | 0.9191544 | 0 |
| q5_k-image | 0 | 3.50039 | 0.236829 | 0.9717048 | None |
| q4_0 | 0 | 6.78244 | 0.576585 | 0.8321515 | 3 |
| q4_0 | 1 | 5.2738 | 0.42048 | 0.90883 | None |
| q4_0 | 2 | 6.06293 | 0.446079 | 0.8957483 | 0 |
| f16 | 0 | 0.000139236 | 1.22267e-05 | 1 | None |
| f16 | 1 | 8.39233e-05 | 6.93078e-06 | 1 | None |
| f16 | 2 | 8.67248e-05 | 7.00278e-06 | 1 | None |
| q6_k-image | 0 | 1.71416 | 0.134711 | 0.9909979 | None |
| q8_0 | 0 | 1.26603 | 0.0930795 | 0.995663 | None |
| q8_0 | 1 | 0.648081 | 0.0554285 | 0.9984628 | None |
| q8_0 | 2 | 0.969611 | 0.0637552 | 0.9979999 | None |
| q4_k | 0 | 6.1979 | 0.539022 | 0.8668642 | 3 |
| q4_k | 1 | 4.45351 | 0.479486 | 0.9093961 | None |
| q4_k | 2 | 4.98178 | 0.359219 | 0.940106 | None |
| bf16 | 0 | 0.00012958 | 1.08498e-05 | 1 | None |
| bf16 | 1 | 8.30889e-05 | 7.34619e-06 | 1 | None |
| bf16 | 2 | 0.00011158 | 7.4136e-06 | 1 | None |
| f16-image | 0 | 0.000115275 | 1.00235e-05 | 1 | None |
| q6_k | 0 | 2.6004 | 0.230822 | 0.973072 | None |
| q6_k | 1 | 1.62197 | 0.149301 | 0.9897637 | None |
| q6_k | 2 | 2.51968 | 0.191553 | 0.9835657 | None |
| bf16-image | 0 | 0.000109673 | 5.85333e-06 | 1 | None |
| q4_k-image | 0 | 4.50549 | 0.312199 | 0.9505371 | None |
| q8_0-image | 0 | 0.64336 | 0.0476247 | 0.9989632 | None |
| q4_1-image | 0 | 5.41467 | 0.353224 | 0.9364004 | None |
| q5_k | 0 | 4.57247 | 0.360825 | 0.9369516 | 3 |
| q5_k | 1 | 2.89605 | 0.246857 | 0.9709489 | None |
| q5_k | 2 | 3.0763 | 0.234923 | 0.97206 | None |
| q4_1 | 0 | 6.51692 | 0.571521 | 0.8304392 | 3 |
| q4_1 | 1 | 4.7297 | 0.404388 | 0.9187831 | None |
| q4_1 | 2 | 4.89885 | 0.377515 | 0.9321547 | None |

### F32 image discrepancy

The 256x256 blue image produces the same three greedy IDs `[1919,2099,369]`.
After the GDN correction below, the CPU run fails 46
of 172 original F32 gates. First failure:
`loop.visit.10`, max_abs 0.02618694305419922, RMSE
4.968730290245209e-05. Worst relative RMSE:
0.00036928722776546083 at `loop.visit.55`.
Auto/Vulkan results are listed separately above. Final-position logits pass,
but the internal failures remain failures. No threshold was widened.

Review found a concrete GDN normalization contract defect: the official formula
is `x * rsqrt(sum(x*x) + 1e-6)`, while the reused legacy path clamps the norm.
SenseNova now opts into the official formula on CPU and Vulkan; existing Qwen
and THX02 keep their original default path. Small-vector/zero/history probes
exercise both formulas, and the full-model comparisons above use the fix.

Before that fix, substituting the exact native visual projection into the
official decoder still failed 12 gates, starting at visit 44. With the fix,
the same diagnostic passes 172 checks with
0 failures using the official chunk GDN. Replacing
chunk prefill with the official recurrent GDN also passes
172 checks with
0 failures. The independent end-to-end image
comparison still fails: the visual projection itself passes its F32 gate, but
its error propagates through the repeated decoder. These substituted-input
experiments isolate that propagation; they are not independent acceptance runs.
The previous report's claim that no concrete decoder defect had been found is
superseded by the GDN fix. Shared vision numeric calibration remains open;
the adapter stays Experimental rather than claiming full alignment.

A separate CPU-only vision probe loads the official tower directly from the
pinned source, using four CPU threads and eager attention. It compares patch
projection, position addition, all 27 transformer blocks and the final
projection with the saved native trace. All 30 checks pass the original F32
gates; the final projection has max_abs 0.000217438 and relative RMSE 1.04835e-5.
This is a placement/roundoff diagnostic, not an end-to-end model acceptance run.
Its per-layer metrics are retained as `vision_diagnostic` in the results.

The full F32 oracle was also rerun independently with recorded CPU vision,
embeddings and output head plus the ROCm decoder. All three text backends still
pass 570 checks. The image comparison improves to 3
failures of 172 for each native backend, with exact
greedy IDs and passing final logits. On CPU the first remaining failure is
`loop.visit.50`; the worst relative RMSE is
0.00010245368487052321. The original
46-failure reference results above remain preserved. Placement changes and
substituted-input diagnostics do not clear the remaining image failures, and
no threshold was widened.

### Native activation precision

BF16 initially inherited the shared family's scalar ggml activation-rounding
contract. Review replaced it with the existing BF16 SIMD/F32-input kernel only
for SenseNova, through a per-projection loader callback. The original
`load_trunk` wrapper still selects the same loader for Qwen3.5, Edge0 and THX02.
A small projection test proves that SenseNova preserves F32 input values while
the original family loader retains BF16 rounding. The previous BF16 full-model
results are preserved under `historical_before_bf16_simd`.

The first Q8_0 comparison decoded the same GGUF weights but omitted native
activation quantization. It retained exact greedy IDs but failed 97 CPU and
91 Auto/Vulkan gates. Those results are preserved under
`historical_before_activation_qdq`. Block-format references now explicitly use
`--activation-quantization`: existing independent NumPy Q8_0/Q8_K input QDQ,
plus Q4_1's separately F16-rounded Q8_1 block sum, precedes the official Linear
operations. No native matmul or decoder is used by this reference. Execution
metadata records every hooked projection; embeddings and F32 mHC stay outside
the activation hook. The numeric gates are unchanged. Quality relative to the
original F32 checkpoint includes both weight and activation loss in these runs.

Replacing chunk prefill with the official recurrent GDN does not clear Q8_0:
text fails 120/570 checks on CPU and 113/570 on Auto/Vulkan; image fails 36/172
and 32/172 respectively. Those independent-input diagnostics retain exact
greedy IDs, and do not replace acceptance references.

An additional independent official CPU reference uses the same Q8_0 GGUF,
activation QDQ and the same three input prefills as the recorded ROCm reference.
Comparing the two official placements themselves fails
19 of 171
original lossy gates. The first failure is
`loop.visit.48`; the worst relative RMSE
is 0.09590495912509847 at final
logits, with the same first greedy token in all three cases. This establishes
reference cross-device variation beyond the fixed whole-model gate; it does
not prove every native discrepancy has the same cause. The metrics are in
`official_device_diagnostic`. No threshold was widened and no native failure
was converted to a pass.

Using that official CPU reference as the target also leaves 19/171 prefill
failures for native CPU and 20/171 for Auto/Vulkan. The CPU comparison first
fails at visit 49, with worst relative RMSE 0.085675; device comparisons first
fail at visit 48, with worst relative RMSE 0.097029. First greedy IDs remain
exact in all three cases. These are prefill diagnostics only, recorded in
`native_cpu_prefill_diagnostic`. Moving the reference onto CPU does not clear
the native failures, so cross-device reference variation alone cannot explain
the entire alignment gap.

The activation-aware Q8_0 reference still fails 121/570 text checks on CPU and
102/570 on Auto/Vulkan. Image comparisons fail 39/172 on CPU and 43/172 on the
device backends; greedy IDs remain exact. The other block formats also retain
full-model failures; the table records greedy divergences and common-input
counts. Adding QDQ did not clear acceptance, and
these failures remain in the numerical table and coverage matrix.

A separate fixed-input diagnostic samples 21 real projections per block format
from physical layers 0, 15, 31 and the output head, with 64 evenly selected
output rows and four seeded F32 input rows. Independent gguf-py decoding,
NumPy input QDQ and PyTorch CPU Linear are compared with the shared native CPU
kernel and strict Vulkan projection execution. All 252 comparisons pass the
diagnostic relative-RMSE gate of 5e-5; the worst observed value is 1.22e-6.
The device probe asserts successful dispatch and records 253 submissions.
This bounds the sampled projection error; it does not explain or clear the
full-model failures. Shapes, input generation, per-format metrics and remote
evidence paths are retained in `projection_diagnostic` in the results.

The separate CPU embedding diagnostic samples 27 real input/generated token IDs
in each of the six block formats, comparing native lookup with independent
gguf-py row decoding. All six sets of 110,592 F32 values pass max_abs <= 1e-7;
the measured maximum absolute error is zero. This checks the sampled decoding
contract separately from accumulated whole-model error. IDs, counts and metrics
are retained in `embedding_diagnostic`.

## Interfaces and review fixes

All 54/54 native numeric runs
are recorded; 14/27
format/backend interface groups are complete. The continuation host is `j-5fyejyon53kz4ytgmxuf5ne2m5`, SSH port 33162. It completed 0 additional interface groups. All 25 pinned source files and 7 re-exported text files plus projector were checked against the original sizes/SHA256. The final CLI/server build passed. Per-group host provenance is recorded in JSON and CSV; numeric, regression and timing evidence below remains attributed to port 31667. No throughput comparison combines the two hosts.

The interface count is checked
separately because a failed numeric gate can mark a matrix row failed before
its interface requests run. The custom-template group is additional coverage.

Every completed interface combination runs real CLI text/image generation,
single/grouped text JEV, single/grouped image JEV, HTTP chat in stream/nonstream
modes, and `/v1/jev/score`, `/v1/jev/grouped`, `/v1/jev/image`, `/v1/jev/image_grouped`.
It checks CLI/HTTP generation equality, SSE DONE, and probability max_abs <= 1e-4.
Commands, request bodies, raw responses and backend traces remain under
`target/release-validation/interfaces-FORMAT-BACKEND/` on the host.

Review fixes:

1. Local YaRN coefficients reuse existing SIMD rotations and Vulkan rope
   inputs; other models retain the default coefficient path.
2. Match decoder/projector identities in CLI generation, HTTP startup and
   the shared multimodal logits helper.
3. Release the Vulkan submission lock before staged trace readback; remove
   the duplicate plain-Qwen checkpoint that reacquired the lock.
4. Add CLI grouped-image JEV using existing payload/scoring helpers.
5. Pass configured Jinja templates to all four HTTP JEV modes. Qwen3.5 default
   single/grouped scores are exact versus baseline; default/Jinja/custom
   CLI/HTTP scores agree exactly after the fix.
6. Stream quantizer output and atomically publish without overwriting a
   concurrent destination. Failure cleanup removes temporary output.
7. Correct Q4_K's first scale estimate/search range and Q6_K's valid signed
   subscale -128. Independent decoding covers positive/negative constants;
   existing GGUF decoders/files are unchanged, regenerated exports change.
8. Materialize CPU-offloaded reference parameters before copying GGUF weights;
   removing an offload hook can restore CUDA parameters, so explicitly move the
   detached module to CPU. Reject meta tensors rather than accepting an
   ineffective copy. Use non-overlapping device-map subtrees: a CUDA ancestor
   otherwise temporarily moves CPU descendants onto the GPU during hook init.
   Keep the F32 output head on CPU as well, and record actual placement/peak
   allocation. The pinned meta-model confirms all parameters have one owner,
   with 27,690,847,744 bytes of GPU parameters; tiny hook tests verify CPU
   modules never receive a CUDA-migration request.
9. Correct SenseNova GDN epsilon placement on CPU/Vulkan, preserving the legacy
   norm by default for other models.
10. Honor HTTP generation Jinja/custom templates for SenseNova image requests,
    preserving text/image order, image bytes and one marker per image. Reuse
    the existing marker expansion and count its net tokens during preflight.
11. Keep SenseNova BF16 on the existing SIMD/F32-input kernel, preserving the
    shared family's legacy activation contract through the default loader.
12. Identify equal-input checkpoints after greedy divergence while retaining
    all original raw comparisons; later error includes different token prefixes.
13. Move Vulkan Q8_0/Q8_K and attention startup self-check diagnostics to stderr.
    The real quantized GPU CLI checks exposed invalid `--jev-output json`
    despite exit code zero. Broken outputs and commands are retained in
    `historical_jev_stdout_failures`. The continuation driver is prepared to
    retry six CLI operations and eight HTTP requests per remaining group, but
    its GPU preflight fails before those groups can run. Completed groups parse
    JEV stdout directly as JSON and compare it with HTTP responses.
    This changes the diagnostic channel without changing arithmetic. Retry
    completion is `False`; an unavailable
    result remains unverified. The final four-log change compiled on port 33162, and CPU CLI JSON parses without filtering. The GPU probe aborts with `device lost` before completing the JSON regression; this remains unverified. See `continuation` and `validation_blockers`.
14. Align shared mapped-memory flush/invalidate ranges to `nonCoherentAtomSize`,
    bounded by the allocation, and map the full allocation so aligned tail ranges
    remain mapped. Validation of a tiny F32 operator exposed six offset/size
    errors even with HOST_COHERENT memory; the patch removes all six errors in
    the same active-layer probe. The new boundary/overflow test passes on Mac;
    the final Vulkan module suite has 9 passed / 20 hardware tests ignored and
    is now included in CI.
    Callers are shared static/device upload helpers and `Qwen3Ops` host writes
    and reads; this cannot be corrected only in the SenseNova model.
    The operator still reports `device lost` after the range fix. The unmodified
    baseline Qwen CLI also fails (exit 1, 4.929 s, 39.2 GB free VRAM). F32 and Q8
    operator probes, including `RUST_GPU_DP4A=0`, fail too. That override controls
    operator variants, not the legacy warmup, so it does not exclude every DP4A
    instruction. CPU Qwen JSON and a bounded ROCm mul/add
    probe pass. This establishes a Vulkan blocker in the supplied environment,
    not its physical/driver root cause. No GPU reset or system change was made.
    Switching RADV from ACO to its existing LLVM backend also fails; requesting
    GPU-assisted validation does not reveal an additional reported error.
    Original and patched logs, source hashes and probe commands are in
    `continuation`; the 54 numeric runs and old-model GPU regression below
    predate this cache-maintenance fix. They are retained as historical evidence,
    and do not prove the post-fix full-model GPU path.
15. Investigate Vulkan outside this repository with an independent integer-write
    shader, validated with `spirv-val --target-env vulkan1.0`, and buffer fills.
    Universal family 0 leaves the sentinel unchanged despite successful fence
    and idle results. Dedicated compute family 1 passes once, but four repeat
    probes fail. A temporary dedicated-queue runtime also fails the F32 operator
    check; that selection change was reverted. Neither host-coherent GTT nor
    mapped VRAM fixes the independent write or increment shader.
    An isolated Ubuntu Mesa 24.0.5 driver reports `GPU hung detected in this
    process (VK_ERROR_DEVICE_LOST)` on both queue families; system Mesa 25.2.8
    instead sometimes signals completion without a write. Private drivers and
    strace stay in `target`; DRM submit/sync ioctls return success, with no
    failed ioctl indicating a container permission rejection. A fresh ROCm
    4096-element mul/add/readback still passes afterward (max_abs 0). These
    controls isolate a reproducible Vulkan/DRM execution failure below the model,
    but do not establish a hardware, firmware, kernel or sharing root cause.
    Kernel logging is restricted (`dmesg`: Operation not permitted). No GPU reset,
    other-job change or persistent driver configuration change was performed.
    Commands, raw errors, package hashes and repeats are in `continuation`.

Q8_0 real-model fallback/boundary probes pass: process-local invalid Vulkan ICD
settings produce Auto CPU execution and strict Vulkan failure, without changing
the system or another job. CLI and server reject mismatched projector identity.
HTTP nonstream/stream reject multiple images, historical images and an expanded
prompt beyond context with 400 before SSE starts; a subsequent valid request is
still 200. Their commands, statuses and backend evidence are in the result JSON.

The first F32 strict HTTP image-JEV run returned 500 after RADV reported
`context lost / hard recovery`. Before recovery probes, only 33.1 GB of physical
VRAM was free, below the measured 36.7 GB native F32 peak. A fresh-process
unmodified-baseline Qwen GPU run and the current tiny operator suite both passed;
this does not prove the full-model failure's cause. The failed server responses
and commands remain in `runtime_events`, rather than being hidden by retries.
F32 native retries now wait for at least 40 GB free; smaller formats run first.
No GPU reset or unrelated process was changed. A separate initial F16 reference
OOM during ancestor hook initialization is preserved and addressed by the
non-overlapping map above.

Cross-model edits are confined to shared coefficient application, optional
trunk coefficient/norm inputs, tracing lock lifetime and template/JEV wiring.
Before the mapped-range correction, Qwen3.5-0.8B Q8_0 real CPU/GPU regression
on port 31667 is bitwise unchanged: CPU compares
17,959,424 F32 words and 22 IDs; GPU compares 5,214,720 words and eight IDs.
The GPU trace selects complete logits to avoid the baseline's tracing deadlock.
The shared quantizer fixes affect new exports; broader old-model K-quant quality
was not benchmarked.

## Engineering regression and performance

Python converter/quantizer/comparator/independent-reference checks: **89 passed**
on the host; Mac: **88 passed / 1 CUDA-dependent test skipped**.
Rustfmt and compile checks pass. Final remote focused checks: **18 passed,
1 ignored**; isolated tracing-lock regression: **1 passed**. Full library:
**1715 passed / 25 failed / 260 ignored**, versus baseline **1708 / 25 / 259**;
failure names are identical. All nine Cargo binary targets are unchanged.
Mac's fresh suite has one additional mHC/YaRN shader-initialization failure on
MoltenVK, alongside the existing initialization failures; it does not contradict
the passed physical AMD tiny graph test. No full-suite green claim is made.

Same-host text process wall, including loading and tracing:

| Storage | Implementation | Backend | Seconds | Peak RSS KiB |
| --- | --- | --- | ---: | ---: |
| F32 | official PyTorch | ROCm | 71.401 | 19,448,616 |
| F32 | native | cpu | 73.017 | 31,372,416 |
| F32 | native | auto -> Vulkan | 179.743 | 30,820,660 |
| F32 | native | vulkan | 170.249 | 30,156,988 |
| F16 | official PyTorch | ROCm | 101.572 | 29,231,404 |
| F16 | native | cpu | 33.115 | 15,964,224 |
| F16 | native | auto -> Vulkan | 113.802 | 16,036,144 |
| F16 | native | vulkan | 114.707 | 16,036,144 |
| BF16 | official PyTorch | ROCm | 73.961 | 29,259,096 |
| BF16 | native | cpu | 34.643 | 15,964,692 |
| BF16 | native | auto -> Vulkan | 85.211 | 16,036,636 |
| BF16 | native | vulkan | 85.031 | 16,036,012 |
| Q8_0 | official PyTorch | ROCm | 86.366 | 28,879,568 |
| Q8_0 | native | cpu | 28.178 | 8,712,712 |
| Q8_0 | native | auto -> Vulkan | 73.473 | 8,775,808 |
| Q8_0 | native | vulkan | 72.750 | 8,778,464 |
| Q4_0 | official PyTorch | ROCm | 78.876 | 29,062,308 |
| Q4_0 | native | cpu | 28.798 | 4,832,668 |
| Q4_0 | native | auto -> Vulkan | 70.051 | 4,908,388 |
| Q4_0 | native | vulkan | 71.542 | 4,906,800 |
| Q4_1 | official PyTorch | ROCm | 118.156 | 29,192,248 |
| Q4_1 | native | cpu | 28.534 | 5,316,672 |
| Q4_1 | native | auto -> Vulkan | 72.381 | 5,389,344 |
| Q4_1 | native | vulkan | 73.382 | 5,392,404 |
| Q4_K | official PyTorch | ROCm | 95.158 | 29,183,992 |
| Q4_K | native | cpu | 28.797 | 4,831,756 |
| Q4_K | native | auto -> Vulkan | 77.175 | 4,903,220 |
| Q4_K | native | vulkan | 80.910 | 4,904,108 |
| Q5_K | official PyTorch | ROCm | 97.364 | 29,113,940 |
| Q5_K | native | cpu | 28.889 | 5,805,580 |
| Q5_K | native | auto -> Vulkan | 80.996 | 5,870,424 |
| Q5_K | native | vulkan | 80.736 | 5,872,212 |
| Q6_K | official PyTorch | ROCm | 89.887 | 29,077,188 |
| Q6_K | native | cpu | 27.953 | 6,827,532 |
| Q6_K | native | auto -> Vulkan | 91.202 | 6,891,824 |
| Q6_K | native | vulkan | 89.953 | 6,893,596 |
| F32 | official PyTorch (recorded placement) | ROCm decoder; CPU input/head | 28.442 | 29,193,116 |

The native F32 text runner creates three sessions, cumulatively uploading
95,278,067,264 bytes and reading back 7,221,903,712 bytes for tracing; the
official runner loads once. Peak device allocation is 36,171,088,608 bytes.
Other jobs share CPU/memory. These are observed process times, not controlled
release-build throughput or capacity estimates. In particular, the earlier
518-second CPU observation overlapped conversion/other jobs and is superseded
by this run; its difference must not be attributed to the GDN fix.
See [performance.json](performance.json) for text/image process times, RSS,
conversion observations and device counters.
Q4_K/Q5_K/Q6_K native CPU, Auto and Vulkan numeric runs execute concurrently;
their process times include that contention and cannot rank backend latency.

Baseline failure names (also retained in the result JSON):

```text
models::breeze::tests::main_matrices_require_original_dtype_exact_shape_and_length
models::breeze::tests::main_source_preserves_original_bf16_and_legacy_codec_f32
models::breeze::tests::main_source_rejects_non_bf16_heads_eoi_and_norms
models::clm::tests::matches_reference_golden_vectors
models::clm::tests::rejects_wrong_embedding_width
models::diffusion::z_image::dit::tests::attention_matches_the_ggml_neon_softmax_and_value_reduction
models::diffusion::z_image::dit::tests::torch_mt19937_recomputes_the_final_sixteen_values
models::dots::vocoder::tests::q8_lstm_matches_independent_gate_order_bias_and_recurrent_state
models::gemma4::trunk::tests::failed_later_gemma4_chunk_preserves_successful_prefix
models::gemma4::trunk::tests::gemma4_trace_rows_match_batch_one_raw_bits
models::gemma4::trunk::tests::per_layer_bf16_projection_uses_the_repository_bf16_kernel
models::qwen35::trunk::tests::later_gpu_failure_recomputes_from_committed_cpu_shadow_and_stays_session_local
models::qwen_drive::tests::planner_scalar_fixture_matches_every_f32_word
models::yue2::tests::nar_noise_matches_torch_2_10_cpu_bits_across_tail_boundaries
models::yue2::vae::tests::convolution_fuses_bias_for_torch_matrix_tiles
models::yue2::vae::tests::convolution_matches_torch_strided_tail_k_blocks
models::yue2::vae::tests::convolution_selects_torch_bias_tile_width
ops::dot::tests::dot_bf16_f32_is_bitwise_stable_against_ascending_reference
ops::kernel::q8_0::tests::q8_kernel_thread_partition
ops::math::exp::avx2_tests::avx2_exp_stays_finite_above_the_clamp
ops::matmul::neon_tests::rope_neox_inplace_matches_pinned_ggml_recurrence_and_fused_rotation
ops::rope::tests::vision_rope_matches_ggml_libc_trig_bits
vulkan::nemotron_h::tests::ssm_conv_matches_cpu_bits
vulkan::nemotron_h::tests::ssm_gate_matches_cpu_bits
vulkan::nemotron_h::tests::ssm_scan_matches_cpu_bits
```

The tested context is 512 with F32 KV and bounded short fixtures. Vulkan's
existing attention session limit is 4096; larger capacities require Auto CPU
fallback or are rejected by strict Vulkan. One final-message image is supported;
historical images, multiple images, audio/video and MTP are not implemented for
this adapter. Arbitrary resizing, long-context and broad task quality remain
unvalidated. Reproduction commands are in [README.md](README.md).
