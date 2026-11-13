# Standalone d3-turbo

The reference is the user-selected ModelScope `vllm-sr/d3-turbo` at
`0c9281c53740765406dcec7f20140179838a6fa5`. Its weights differ from the older
Hugging Face revision used by the Vela3 adapter integration. Do not interchange
their hashes or validation results.

Implementation plan and scope:

- Reuse the existing D3 Qwen3.5 prefix-bidirectional encoder. Add an explicit
  standalone loader; retain the Vela3 adapter loader and its default behavior.
- Keep native prompt, codebook, request/answer, image/video and spatial decision
  semantics in `models/d3_turbo`; connect the existing CLI, HTTP and JEV interfaces.
- Export F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K with existing
  quantizers. Preserve F32 readout, norms and sensitive auxiliary tensors.
- Validate each precision with CPU, Auto and strict Vulkan. Device execution is
  mixed: eligible matrix projections on Vulkan; recurrent scans, normalization
  and request postprocessing on CPU. Full device residency depends on the shared
  D3 device graph and is outside the current shared implementation.
- Pin tokenizer IDs and prompt boundaries exactly. Compare lossy arithmetic by
  calibrated numerical metrics, separately from quantization quality drift.
- Run real CLI/HTTP/JEV requests, record same-host performance, then self-review
  request validation, entry routing, formats, device evidence and regressions.

All model assets, conversions, Python environments, Rust builds and large traces
stay on the authorized SSH host under `/workspace/rust-model-inference-d3-turbo`.
Downloads stream directly from ModelScope to that host; the Mac receives only
source changes and small validation reports.

The final coverage matrix and measurements are recorded in `validation.md`.

## Reproduce on the model host

Use a project-local Python environment with the versions listed in
`validation.md`. The original `d3_runtime.py`, `d3_compose.py` and checkpoint
remain read-only Oracle inputs. `trace.py --official` runs the runtime's default
BF16 mode; the diagnostic path streams its official graph through F32.

```bash
python tools/oracle/d3_turbo/trace.py --model models/d3-turbo \
  --request tools/oracle/d3_turbo/text.json --out target/d3-oracle.jsonl --threads 8
RMI_PARITY_TRACE="$PWD/target/d3-native.jsonl" \
  target/release-fast/rust-model-inference --model models/d3-turbo \
  --system-one-request tools/oracle/d3_turbo/text.json --compute cpu --threads 8
python tools/oracle/d3_turbo/compare.py target/d3-oracle.jsonl target/d3-native.jsonl
```

For a same-GGUF reference add `--gguf MODEL.gguf` and
`--activation-quantization` to `trace.py`; use `compare.py --lossy` for F16 and
block-quantized CPU projections. F16 activation rounding is automatically
included when the GGUF contains F16 matrices. The independent Q4_1 reference
also accounts for its separately rounded Q8_1 sum. Never substitute an original
F32 checkpoint comparison for a same-weight quantized comparison.

`make_media.py` creates small deterministic image/video requests on that host.
Video checks require FFmpeg/ffprobe and the official runtime's OpenCV decoder.
`protocol_fixture.py` regenerates prompt/layout and spatial distribution goldens
from the pinned official Python files.

`projection.py` extracts one fixed native input and independently dequantizes
each GGUF projection. Run the ignored Rust test with
`RMI_D3_PROJECTION_FIXTURE` pointing to its generated `fixtures.json` to separate
weight decoding/activation rounding from recurrent whole-graph drift.

```bash
python tools/oracle/d3_turbo/entries.py --request target/d3-turbo/one.json \
  --out target/d3-turbo/entries --threads 8 --trace
python -m unittest tools.converter.d3_turbo.test_convert \
  tools.oracle.d3_turbo.test_compare
```

`entries.py` runs real CLI and HTTP calls sequentially; `--jev` accepts a
single-question JEV request. Its reports distinguish execution, CPU fallback,
strict device failure and CLI/HTTP equality. Numerical acceptance is separate.
Empty/incomplete traces are rejected. Full coverage and remaining layers are in
[validation.md](validation.md), with [user-facing usage](../../../docs/develop/D3_TURBO.md).
