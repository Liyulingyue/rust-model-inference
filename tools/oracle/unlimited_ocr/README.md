# Unlimited-OCR-2

Native inference for the pinned [PaddlePaddle/Unlimited-OCR-2](https://modelscope.cn/models/PaddlePaddle/Unlimited-OCR-2) checkpoint. Loads the original two BF16 safetensors directly or a self-contained `unlimited-ocr` GGUF. Rust inference has no Python dependency. `config.json` must declare `model_type = unlimited-ocr` and the checked architecture contract.

The Rust implementation reuses the repository's mmap loader, Linear/norm kernels, Rust SIMD operators and Pillow-compatible bicubic resize. Compute, activations and KV are F32; normalized pixels are rounded to BF16 once, matching the official processor. The reference runs the original, unmodified official modules in PyTorch F32 on the same BF16 weights and input pixels. This does not establish equivalence to native BF16 activation arithmetic.

## Run

Download revision `7eadb7cd7afe660602040524eab37c8a9b91fe53` through the ModelScope CLI, using a repository-local environment:

```sh
python3 -m venv models/.venv
models/.venv/bin/pip install modelscope
models/.venv/bin/modelscope download --model PaddlePaddle/Unlimited-OCR-2 \
  --revision 7eadb7cd7afe660602040524eab37c8a9b91fe53 \
  --local_dir models/Unlimited-OCR-2
cargo build --profile release-fast
cargo run --profile release-fast -- \
  --model models/Unlimited-OCR-2 --image first.png --reference second.png \
  --max-tokens 2048 --max-context 8192 --threads 8 --out result.txt
```

Use `--reference` repeatedly for subsequent pages. `--prompt` is instruction text; the trained `[BOS] + (<PAGE> + 273 image tokens) * pages + instruction` layout is inserted automatically. Default instruction: `Multi page parsing.` Output preserves the model's special/page markers and drops EOS, as the official inference entry does. `--out` refuses an existing destination unless `--overwrite` is set.

This entry uses the official **base** mode (`crop_mode=False`): every page has one 1024×1024 letterboxed global view. SAM ViT-B feeds CLIP-L, then a linear projector; the decoder has 12 layers, 64 experts/top-6, shared experts, Page2D RoPE and inference DASH routing (head-CLS summaries, top-16 entmax prior/top-12 remote blocks plus an exact 128-token local window). It supports greedy generation, default page-count guard and a 20,000-token per-page guard. Floating point reductions/activations are verified numerically, without a bitwise requirement.

Limits: 1–128 eight-bit page images, bounded F32 KV, context at most 131072 tokens (8192 by default). Dynamic tiles/gundam, PDF rasterization, sampling and constant-memory unlimited streaming are not implemented. `--compute auto` / `--gpu` uses shared Vulkan projection offload with CPU fallback; `--compute vulkan` is rejected because attention, routing and other operations still run on CPU. Large page sets and long contexts have not been benchmarked; memory grows with the retained context. A token budget can truncate a page before EOS.

## Weight formats and compute

```sh
.venv/bin/python -m tools.converter.unlimited_ocr.convert \
  models/Unlimited-OCR-2 models/Unlimited-OCR-2-Q8_0.gguf --type q8_0
cargo build --profile release-fast --features vulkan
# --compute cpu works in both builds; Auto falls back when Vulkan is unavailable.
target/release-fast/rust-model-inference --model models/Unlimited-OCR-2-Q8_0.gguf \
  --image first.png --compute auto --max-tokens 128
```

The converter accepts `bf16`, `f16`, `f32`, `q8_0`, `q4_k` and `q6_k`. It embeds the original config/tokenizer and preserves tensor names and I64 buffers. Quantized exports only quantize aligned 2-D projection/embedding matrices; convolutions, positions, vectors and unaligned matrices keep their source type. BF16 export preserves the original weights exactly; F16/F32 conversion and all quantization modes have small mixed-source converter fixtures. Real BF16 and Q8_0 exports have been run through CLI; the other four whole-model formats have not been validated. F32 projections currently copy their mapped values when constructing a request view, increasing request startup time and memory.

BF16 matrices use the shared AVX2/FMA, NEON or scalar dot dispatcher. Quantized projections use the existing weight kernels and prepare activations internally. `RMI_SCALAR=1` selects CPU scalar reference arithmetic. BF16 Auto projections use the existing CPU verification/retry path; small projections and unsupported device operations fall back to CPU. `RMI_COMPUTE_TRACE=1` prints actual device submissions/uploads for projection offload. This is hybrid execution, and no speedup is claimed from concurrent correctness probes.

## HTTP

The server uses the common Chat, Anthropic Messages, Responses, SSE, stop-sequence and cancellation infrastructure. OCR accepts one user turn containing ordered page images and optional instruction text/system instructions; conversation history, tools, thinking, JEV and sampling options are rejected. Temperature defaults to zero for this backend. Scanned special/tool/thinking markers are preserved as text. JEV endpoints remain reserved for language-model backends and return an unsupported-backend error for Unlimited-OCR.

```sh
target/release-fast/rust-model-inference --serve --model models/Unlimited-OCR-2 \
  --host 127.0.0.1 --port 8080 --threads 8 --compute auto
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"first.png"}}]}],"max_tokens":128}'
```

Probe a running server with the pinned page fixture, including Anthropic token counting, Chat/SSE, Messages, Responses, validation and concurrent-request rejection:

```sh
.venv/bin/python tools/oracle/unlimited_ocr/check_http.py \
  --url http://127.0.0.1:8080 --output artifacts/unlimited_ocr/http-probes.json
```

The checker uses `max_tokens=1` for OCR transport checks; complete generated documents are covered by the separate oracle runs.


## Pinned inputs and oracle

The runner verifies SHA256 of the official source, tokenizer and weights before importing or loading them. It never edits the reference implementation. Vision features are injected as `inputs_embeds` to avoid the official multimodal wrapper's unconditional CUDA mask transfer; all SAM/CLIP/projector/decoder arithmetic runs in the original modules. Inference DASH is explicitly enabled, with ring/sliding KV disabled as in official `_uocr_generate`.

| File | Bytes | SHA256 |
|---|---:|---|
| model-00001-of-00002.safetensors | 4998430312 | `4c40b740f209fd7682040d4ebf19409208acde5c401ccd2dde426d0f41542f18` |
| model-00002-of-00002.safetensors | 1689875264 | `9490b82fa47f89e5110eb5577f8ab1b6694c12121905dbc086ddb10207a4ba0c` |
| fixtures/page-1.png | — | `24bbe8347d2a33c071a3ab576275ffe4deea93fa30713aaa8eac3e96c9bed9b4` |
| fixtures/page-2.png | — | `f543d588f1ed13c2dee05af94244d8cabcf817c0a340a5de9f0fb6243e5936ec` |

### 生成页面 fixture

两页图片**不入库**，由脚本合成：

```sh
.venv/bin/python tools/oracle/unlimited_ocr/make_pages.py
```

字体是 Liberation Sans（Debian `fonts-liberation`），脚本按 SHA-256 固定它；Pillow 的
版本由 `requirements.txt` 固定。两页都是白底黑字：800×1000，标题 48 px，正文 34 px，
行距 110 px，左边距 60 px。脚本会打印每页的 SHA-256，与上表对照即可知道本机的渲染
结果是否与记录一致。

`check_http.py` 在缺图时会自己调用生成器，所以 HTTP 检查不需要先手动生成。

早先提交的页面是用 macOS 的系统字体渲染的，无法在 Linux 上逐字节复现；换成
Liberation Sans 后两页的**文字完全相同**，模型预测的页面坐标有 1–3 px 的位移，
因此 `verification.json` 里记录的 token 序列随之更新。

Create a Python 3.12 environment **inside the repository** for the oracle:

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r tools/oracle/unlimited_ocr/requirements.txt
.venv/bin/python tools/oracle/unlimited_ocr/verify.py \
  --model-root models/Unlimited-OCR-2 --primitives \
  --output artifacts/unlimited_ocr/primitives.json
.venv/bin/python -m unittest discover -s tools/oracle/unlimited_ocr -p 'test_*.py'
cargo test --profile release-fast --lib unlimited_ocr
RMI_UNLIMITED_OCR_MODEL=models/Unlimited-OCR-2 \
  cargo test --profile release-fast --lib real_tokenizer_preserves_markers_and_unicode -- --ignored
cargo test --profile release-fast --test unlimited_ocr_cli
```

Real single-page alignment (use a fresh oracle directory and trace filename each time):

```sh
cargo build --profile release-fast --features parity-trace
RMI_PARITY_TRACE="$PWD/artifacts/unlimited_ocr/single-rust.jsonl" \
  target/release-fast/rust-model-inference --model models/Unlimited-OCR-2 \
  --image tools/oracle/unlimited_ocr/fixtures/page-1.png --threads 8 --max-tokens 128 \
  --out artifacts/unlimited_ocr/single.txt
.venv/bin/python tools/oracle/unlimited_ocr/verify.py \
  --model-root models/Unlimited-OCR-2 --threads 8 --max-tokens 128 \
  --images tools/oracle/unlimited_ocr/fixtures/page-1.png \
  --output artifacts/unlimited_ocr/single-oracle
.venv/bin/python tools/oracle/unlimited_ocr/verify.py \
  --compare artifacts/unlimited_ocr/single-oracle/trace.jsonl artifacts/unlimited_ocr/single-rust.jsonl \
  --output artifacts/unlimited_ocr/single-comparison.json
```

The default whole-graph comparison above exits with a numeric failure on the measured fixtures; see the retained MoE cutoff exceptions below. For the multi-page case add `--reference tools/oracle/unlimited_ocr/fixtures/page-2.png` to Rust and a second path to the oracle's `--images`, and set both token budgets to 256; use separate output paths. With `CARGO_TARGET_DIR` set, adjust the binary path accordingly.

The comparator checks every checkpoint's name, order, shape and occurrence count. Token IDs, BF16-normalized pixels and generated IDs must match exactly. For floating point checkpoints, the fixed bounds are maximum absolute error divided by oracle peak ≤ 0.002 and cosine ≥ 0.99999. NaNs, missing/extra checkpoints, wrong payload lengths and the first failing numeric checkpoint stop verification. Optional route diagnostics are checked separately by the pinned primitive fixture.

Compare the complete vision component from the same CLI trace, then test the decoder with the official visual embeddings as its input:

```sh
.venv/bin/python tools/oracle/unlimited_ocr/verify.py \
  --compare artifacts/unlimited_ocr/single-oracle/trace.jsonl artifacts/unlimited_ocr/single-rust.jsonl \
  --scope vision --output artifacts/unlimited_ocr/single-vision-comparison.json
RMI_UNLIMITED_OCR_MODEL=models/Unlimited-OCR-2 \
  RMI_UNLIMITED_OCR_ORACLE=artifacts/unlimited_ocr/single-oracle \
  RMI_PARITY_TRACE="$PWD/artifacts/unlimited_ocr/single-decoder-rust.jsonl" \
  cargo test --profile release-fast --features parity-trace --lib \
    real_decoder_from_official_embeddings -- --ignored --test-threads 1
.venv/bin/python tools/oracle/unlimited_ocr/verify.py \
  --compare artifacts/unlimited_ocr/single-oracle/trace.jsonl artifacts/unlimited_ocr/single-decoder-rust.jsonl \
  --scope decoder --output artifacts/unlimited_ocr/single-decoder-comparison.json
```

Repeat with `multi-oracle` and fresh multi-page trace/output names. Scope selection keeps the same numeric bounds and requires every checkpoint of that component; it does not skip failing tensors within the component. The decoder test also asserts the entire expected generated token sequence. Trace files append, so each rerun needs a fresh filename.

## Measured alignment

[verification.json](verification.json) records the 2026-10-28 Apple ARM CPU runs against the pinned, unmodified official modules. Both cases completed through EOS: all 110 single-page and 212 double-page output token IDs, decoded text, prompt IDs and normalized pixels match exactly. All 111/213 end-to-end logit checkpoints pass the fixed bounds; their maximum peak-relative errors are 0.00013437 and 0.00012958.

| Identical-input component | Checkpoints | Maximum peak-relative error | Minimum cosine | Result |
|---|---:|---:|---:|---|
| Single-page vision | 43 | 0.00002550 | 0.999999999935 | Pass |
| Single-page decoder | 2999 | 0.00003711 | 0.999999999919 | Pass |
| Double-page vision | 84 | 0.00002684 | 0.999999999797 | Pass |
| Double-page decoder | 5753 | 0.00004023 | 0.999999999872 | Pass |

The **whole graph does not pass every pointwise peak bound**. Each case has five prefill attention exceptions, at decoder layers 2, 3, 5, 7 and 8. The worst peak-relative error is 0.012268; whole-graph minimum cosine remains at least 0.99999935. These failures are retained in `verification.json`, and the default comparator reports them without loosening the bounds.

The first divergence comes from a discrete MoE top-6 cutoff. At decoder layer 1, image row 173, the official sixth/seventh expert scores (58/48) differ by only 0.00000149. Feeding the Rust visual embeddings to the unmodified official decoder reverses those two experts, reproducing the route change. With identical official embeddings, the Rust decoder passes all checkpoints above. This isolates upstream floating point rounding crossing a routing boundary; exact greedy output on these fixtures does not guarantee the same route or output on every document.

The integration checks cover OCR through Chat, Anthropic Messages, Responses and SSE, including token counting, option validation and concurrent-request rejection. Auto on Apple M3 Max uses the shared Vulkan projection path and falls back to CPU when the driver is unavailable. The checker uses a one-token OCR budget; it is a transport smoke test rather than a throughput benchmark.

The real BF16 GGUF export reproduces all 69 first-prefill checkpoints bit-for-bit against the original safetensors trace. Real Q8_0 runs produce the first OCR token `<PAGE>`. This is a smoke check; a full quantized-weight graph oracle has not been run.

After the integration fixes, the focused OCR model units, common API tests and CLI regressions pass, along with the converter, oracle self-checks, HTTP checker self-checks, shared kernels and scalar BF16 checks. The original full library run had 1292 passes, 86 ignored and the same 20 failures reproduced on the base commit (1280 passes, 84 ignored). That full-library result predates these integration fixes; the full suite has not been rerun. Recorded runtimes include concurrent validation jobs and are not clean performance benchmarks.

### Linux 上的一次独立复验

[verification-linux.json](verification-linux.json) 记录 2026-10-30 在 Linux x86_64（Intel 集显）上
的一次复验，用的是**重新生成的页面**（见上面「生成页面 fixture」）。12 个官方源码 SHA 与
2 个权重 SHA 全部匹配，官方模块 unmodified。

| 用例 | 检查点 | 结果 | 最大 peak-relative | 最小 cosine |
|---|---:|---|---:|---:|
| 单页 | 3040 | **通过** | 0.00009388 | 0.999999999 |
| 双页 | 5835 | **未通过** | 0.01024003 | 0.999999569 |

双页在 `uocr.attention` decoder layer 2 occurrence 2 处越界（门槛 0.002）—— **与作者
在 macOS 上保留的例外同一签名**（attention，decoder layer 2/3/5/7/8），量级也相近
（作者 0.012268，本机 0.010240）。也就是说，那个 MoE 路由边界上的不稳定**不是作者
机器的环境问题**，换机器、换渲染字体后依然出现。

官方模块在本机重算的 primitive 与仓库里 pin 的 `tests/fixtures/unlimited_ocr_oracle.json`
在 float32 的 1–2 ULP 上有差异（rotated rope 与 route bias）；那是环境差异，不是 Rust
实现差异 —— 该 fixture 存的是官方代码自己的输出，Rust 测试的容差覆盖得住。

六种 GGUF 格式（bf16 / f16 / f32 / q8_0 / q4_k / q6_k）全部转换成功并能加载运行；
`f16` 与 `bf16` 的输出**逐字节相同**，`q8_0` 与 `q4_k` 的文字相同而预测框位移 1 px。
`f32` 导出在 22 GB 的机器上占 22 GB 常驻。

Auto 路径在本机集显上**比 CPU 慢 2.03 倍**（128 token：58.12 s 对 118.18 s），输出
与 CPU 逐字节相同。离 CPU 的门槛是 16384 个输出元素，所以单行 decode 的投影留在 host，
只有 lm_head 上了设备；`RUST_GPU_MIN_OUTPUTS=0` 时提交次数从 2517 升到 7628，输出不变。

The committed primitive fixture covers Unicode/whitespace/special-token IDs, page coordinates, resampled Page2D RoPE, entmax route indices/biases and the full normalized pixel hash, including Pillow's ties-to-even pad placement.
