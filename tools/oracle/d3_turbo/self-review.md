# Self-review

Scope: standalone `d3_turbo`, existing decision CLI/HTTP/JEV orchestration,
conversion and pinned-reference validation. This is an experimental layer, not
approval for a fully aligned model or a working GPU implementation.

## Fixed during review

- JEV applied THX-02's four-image contract before choosing the model. Validation
  now belongs to the selected engine; a five-image planning test covers this.
- Named HTTP image assets could reach the CLI file decoder. Images/assets now
  share URL validation; local assets and local/remote video strings are rejected
  before decoding. Inline videos have byte/body limits and restricted demuxing.
  Their temporary files are created with Unix mode 0600 and removed by RAII;
  the real decoder fixture checks both permissions and cleanup.
- FFmpeg 9 removed `-vsync`. The decoder uses `-fps_mode passthrough`, verified
  against the actual moving-box fixture and current FFmpeg binary.
- Video prompts omitted the official processor's outer vision token pair. A
  fixed-string check and the complete 198-check video comparison cover the fix.
- Readout metadata alone did not establish actual F32 storage. Both the loader
  and converter now check the readout tensor; the converter has a negative test.
- Spatial clips lacked the official 60-second bound. Box/track now validate it.
- Score questions with front images bypassed the reference's media refusal.
  Prompt validation now rejects score with front images, videos or named assets;
  a focused test also checks that ordinary text scoring remains accepted.
- BF16 vision auxiliary vectors were read through an F32 interface, and temporal
  pixels lost precision. Explicit standalone F32 opt-in widens auxiliaries and
  keeps paired patch activations in F32; the legacy default is unchanged.
- Independent references omitted F16 activation rounding, Q4_1's separately
  rounded Q8_1 sum and a lazy sibling-module import. These were repaired in the
  reference wrapper without editing the pinned original model files. Lossy
  embeddings use numerical gates, not exact equality.
- A shared Cargo target had reused a baseline test artifact. The fingerprint
  and test binary were rebuilt; the final run demonstrably contains D3 tests.
  Reports use the outer test result and compare failure names, not just totals.
- Native trace manifests append by design. Entry probes now reset the manifest
  before each rerun; the superseded aborted video-box prefix was removed before
  comparison while retaining the final run's sidecar paths and checkpoint order.

## Layer and compatibility review

The common decision facade explicitly selects THX-02 or D3. App modules load
requests, normalize transport inputs and serialize outputs; model semantics
stay under `models/`. No model-specific executable or dependency was added.

The existing Vela D3 encoder already implements this backbone. Its standalone
constructor accepts a generic tensor source and explicit F32 execution, while
the legacy constructor retains adapter/BF16 defaults. Quantized preparation in
the shared tensor helper uses existing kernels; it does not add scalar or
external accelerated kernels. Shared Qwen3.5 vision changes are opt-in. A
strict-device projection failure is now surfaced rather than reported as a
successful CPU operation. Real legacy Vela inference remains unverified because
its matching model assets are unavailable on the test host.

## Remaining findings

- Whole-graph quantized and some visual intermediate gates still fail, despite
  passing isolated projections and several final logit checks. Their first
  failures and unchanged thresholds must remain visible in validation reports.
- The shared Vulkan row-projection test independently loses the device. The
  adapter cannot claim GPU acceptance from Auto fallback or strict errors.
  Layer 3 belongs to shared Vulkan/kernel maintainers, then the D3 matrix owner.
- Remaining format/media/JEV crosses, large contexts, multiwindow scale and
  actual legacy model regression belong to layer 2. Untested combinations are
  retained as unrun in the coverage matrix, not inferred from another format.
- Prefix state is recomputed per question. This known performance limit is
  recorded beside the computation with `shortcut:`.

Evidence and exact commands are in the accompanying validation reports.
