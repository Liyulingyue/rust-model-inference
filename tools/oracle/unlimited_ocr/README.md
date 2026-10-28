# Unlimited-OCR-2

Native CPU inference for the pinned [PaddlePaddle/Unlimited-OCR-2](https://modelscope.cn/models/PaddlePaddle/Unlimited-OCR-2) checkpoint. Loads the original two BF16 safetensors directly; no GGUF conversion or Python inference dependency. `config.json` must declare `model_type = unlimited-ocr` and the checked architecture contract.

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

Limits: CPU CLI only, 1–128 eight-bit page images, bounded F32 KV, context at most 131072 tokens (8192 by default). Dynamic tiles/gundam, PDF rasterization, GPU, server API, sampling and constant-memory unlimited streaming are not implemented. Large page sets and long contexts have not been benchmarked; memory grows with the retained context. A token budget can truncate a page before EOS.

## Pinned inputs and oracle

The runner verifies SHA256 of the official source, tokenizer and weights before importing or loading them. It never edits the reference implementation. Vision features are injected as `inputs_embeds` to avoid the official multimodal wrapper's unconditional CUDA mask transfer; all SAM/CLIP/projector/decoder arithmetic runs in the original modules. Inference DASH is explicitly enabled, with ring/sliding KV disabled as in official `_uocr_generate`.

| File | Bytes | SHA256 |
|---|---:|---|
| model-00001-of-00002.safetensors | 4998430312 | `4c40b740f209fd7682040d4ebf19409208acde5c401ccd2dde426d0f41542f18` |
| model-00002-of-00002.safetensors | 1689875264 | `9490b82fa47f89e5110eb5577f8ab1b6694c12121905dbc086ddb10207a4ba0c` |
| fixtures/page-1.png | — | `82b33a6d81c692ae96a7ee5514c4c07a4361b6feadbf58085ca4765f90cce487` |
| fixtures/page-2.png | — | `917808b1c31211e4fd0fc4fb36ead391a390d3c6b3c76cb96c0fe8b23f46f53b` |

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

Focused model tests (11), safetensors tests (2), CLI tests (2), oracle self-checks (3), real tokenizer and both real decoder fixture tests pass. The default full library run has 1292 passes, 86 ignored and the same 20 failures reproduced on the base commit (1280 passes, 84 ignored); no new failing tests were introduced. Recorded runtimes include concurrent validation jobs and are not clean performance benchmarks.

The committed primitive fixture covers Unicode/whitespace/special-token IDs, page coordinates, resampled Page2D RoPE, entmax route indices/biases and the full normalized pixel hash, including Pillow's ties-to-even pad placement.
