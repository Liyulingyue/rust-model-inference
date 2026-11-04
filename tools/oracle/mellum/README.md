# Mellum 2.1 bit-parity oracle

Reference implementation for `general.architecture = "mellum"`
(`src/models/mellum/`). Used by `tests/mellum_reference.rs`.

## Pin

| | |
| --- | --- |
| upstream | [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) |
| commit | `8e2d31e0eb658d0382d3106a61369dff81f0c080` (2026-10-09) |
| why this commit | it is the newest upstream commit whose `src/llama-graph.cpp` still matches the read-only `references/llama.cpp` checkout byte for byte (blob `1514b8ae…`), so the local oracle tree and this pin are the same code |
| `mellum` support added | `4fb16eccce5e451b40014355f97374d692480a4d` ("model: add Mellum architecture", #23966) |

Blob hashes re-verified by `build_mellum_oracle.sh`:

| file | blob |
| --- | --- |
| `src/models/mellum.cpp` | `3ab79747af276a93631ebe2d052edb63277e8971` |
| `src/llama-graph.cpp` | `1514b8aebe8c3b121c46c968060be3105e0e4d0b` |
| `src/llama-model.cpp` | `b3a0cf5d536f72f52a9bec6928ed50c8be89dbaa` |
| `src/llama-vocab.cpp` | `dbadaf100b64827003e9c5c289f4bfd4f9aee173` |

## Model under test

| | |
| --- | --- |
| repo | `JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF` (ModelScope) |
| file | `Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf` |
| size | 12 925 417 760 bytes |
| SHA-256 | `69e0ac02a16586c2c9e991f1e3b8a1eb403882e7ab9e69ea01eb23dbbbc6841f` |

Only Q8_0 is verified. `MXFP4_MOE`, `Q4_K_M` and `Q6_K` are **not** covered:
`MXFP4_MOE` needs a dequant path the repo does not implement, and the k-quants
would additionally have to reproduce ggml's f16-accumulating dot.

## Build

```sh
# materialise the pinned tree (codeload works even when git clone does not)
curl -L -o llama.tar.gz \
  https://codeload.github.com/ggml-org/llama.cpp/tar.gz/8e2d31e0eb658d0382d3106a61369dff81f0c080
mkdir -p llama-mellum && tar xzf llama.tar.gz -C llama-mellum --strip-components=1

tools/oracle/mellum/build_mellum_oracle.sh llama-mellum
```

The script refuses to build unless the four blob hashes above match, then
replaces `examples/eval-callback/eval-callback.cpp` with
`mellum-eval-callback.patch`. **No model source is patched** — `mellum.cpp`
already labels every checkpoint we compare through
`llm_graph_context::cb(...)`.

Scalar settings (required for bit-exact parity): `-U__ARM_NEON
-ffp-contract=off`, `GGML_NATIVE/BLAS/ACCELERATE/OPENMP/LLAMAFILE/CUDA/VULKAN=OFF`,
plus the runtime ones set by the harness itself — 1 thread, `n_batch =
n_ubatch = 1`, F32 KV cache, flash attention off, `warmup = false`.

## Run

```sh
MODEL=models/Mellum2.1-12B-A2.5B-Thinking-GGUF/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf

# 1. oracle: replay the RAW prompt (no chat template), one token per decode
RMI_PARITY_TRACE=/tmp/mellum/oracle \
  llama-mellum/build/bin/llama-eval-callback -m "$MODEL" -p "What is 2+2?" -n 4

# 2. rust: same raw prompt through the same trunk
RUSTFLAGS='-C no-vectorize-loops -C no-vectorize-slp' \
  cargo build --profile release-fast --features parity-trace --test mellum_reference

RMI_SCALAR=1 RMI_PARITY_TRACE=/tmp/mellum/rust RMI_MELLUM_PROMPT="What is 2+2?" \
RMI_MELLUM_N_PREDICT=4 \
  cargo test --profile release-fast --features parity-trace \
    --test mellum_reference mellum_q8_matches -- --ignored --nocapture

# 3. compare
python3 tools/oracle/mellum/check_scalar.py /tmp/mellum/oracle /tmp/mellum/rust
```

`check_scalar.py` joins both manifests on `(name, layer, step)` and diffs the
raw F32 words — no tolerance, no cosine similarity. It trims `kq` /
`kq_soft_max` to `step + 1` entries per head because llama.cpp dumps them at
the padded KV width.

## Verified

`RMI_MELLUM_PROMPT="What is 2+2?" RMI_MELLUM_N_PREDICT=4` (7 prompt + 4
generated = 11 steps):

* tokenizer: `prompt_ids = [2821, 359, 255, 52, 45, 52, 65]` — identical
* `7425 / 7425` checkpoints bit-identical, `0` mismatches
* `generated_ids = [233, 233, 629, 2780]` — identical

The rendered ChatML prompt the CLI actually feeds (this also pins the chat
template), `RMI_MELLUM_N_PREDICT=12`:

* `prompt_ids = [27, 1397, 233, 3969, 255, 99, 2717, 669, 410, 38903, 255,
  99, 995, 48, 28, 233, 27, 8091, 233, 23, 233, 233, 24, 233, 233]` — identical
* `generated_ids = [2116, 895, 333, 6246, 434, 255, 99, 2717, 669, 410, 38903,
  255]` — identical
* `24975 / 24975` checkpoints bit-identical

Long context, crossing the 1024-token sliding-window boundary
(`"hello world 123 " * 215`, 1290 prompt + 2 generated = 1292 steps):

* `generated_ids = [26299, 1685]` — identical
* every checkpoint bit-identical at steps 1022–1026 on an SWA layer (0) and a
  non-SWA layer (3), i.e. the window edge is exact

The full 1292-step sweep is ~1.1M checkpoint files; the boundary sample plus
the identical continuation is the recorded evidence, not a full 1292-step diff.

Note on greedy quality: at `--temp 0` this checkpoint repeats itself on longer
generations ("Straightforward. We can provide..."). That is the model, not the
trunk — the same 12 token ids come out of the pinned oracle above.

## Checkpoint names

Emitted by the harness (the graph labels them, the harness only selects):

```
embd
attn_norm  Qcur_normed  Kcur_normed  Vcur  Qcur  Kcur
kq  kq_soft_max  kqv  kqv_out
ffn_inp  ffn_norm
ffn_moe_logits  ffn_moe_probs  ffn_moe_topk  ffn_moe_weights  ffn_moe_weights_norm
ffn_moe_up  ffn_moe_gate  ffn_moe_swiglu  ffn_moe_down  ffn_moe_weighted  ffn_moe_out
ffn_out  l_out
result_norm  result_output
```

## Notes that mattered for parity

These were each a real divergence found by this harness, not speculation:

1. **YaRN magnitude.** ggml's `rope_yarn` multiplies `cos`/`sin` by
   `1 + 0.1 * log(1/freq_scale)` *inside* the rope kernel whenever
   `ext_factor != 0`. `llama_context`'s own `attn_factor` cancels to exactly
   `1.0` here, so the checkpoint only shows the inner factor
   (`0.1 * ln(16) + 1 = 1.2772589`) — visible as `Qcur != Qcur_normed` at
   `pos = 0` on a non-SWA layer.
2. **SWA layers use plain RoPE.** `mellum.cpp` passes `freq_scale = 1`,
   `ext_factor = 0`, `attn_factor = 1` for `hparams.is_swa(il)`, i.e. the
   unscaled table, while the other 21 layers use YaRN.
3. **`ggml_argsort_top_k` does not swap.** It is `ggml_argsort(DESC)` plus a
   view; the `std::swap(dst[0], dst[1])` lives in the *different*
   `ggml_top_k` entry point, which Mellum does not use. The captured
   `ffn_moe_topk` is therefore in plain descending order.
4. **`kqv` reduces in f64.** `ggml_mul_mat(v, kq)` is one `ggml_vec_dot_f32`
   per output lane — f32 products accumulated in `ggml_float`. Accumulating in
   f32 survives a handful of keys and then diverges (first seen at step 6).
5. **The GGUF's YaRN keys are write-only.**
   `mellum.rope.scaling.yarn_attn_factor` / `yarn_beta_fast` / `yarn_beta_slow`
   are emitted by `llama_model_saver` but read by no loader, so the effective
   `attn_factor` is 1.0 and `beta_fast/slow` are the llama.cpp defaults
   32 / 1. Using the GGUF values changes every YaRN angle.

## Vulkan device path

The trunk also runs on the Vulkan device (`--compute auto`). It is **correct**
but **not faster**, and the reason is structural rather than a missing
optimisation: the Mellum trunk walks the model one token at a time, so every
projection is a single row.

Setup (the box needs the NVIDIA ICD spelled out):

```sh
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
MODEL=models/Mellum2.1-12B-A2.5B-Thinking-GGUF/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf

# device path (the output head is the only shape that lands on the GPU)
target/release-fast/rust-model-inference --model "$MODEL" \
  --prompt "What is 2+2?" --max-tokens 8 --compute auto --threads 8
```

### The device path is bit-identical to the CPU

`RMI_PARITY_TRACE` on both backends over the same prompt, then join the two
traces on `(name, layer, step)` and compare the raw F32 words:

```
cpu checkpoints 15466  gpu checkpoints 15466  common 15466
no divergence in the traced checkpoints
```

All 15466 checkpoints agree exactly — `--compute auto` and `--compute cpu`
produce the same sums, mins, maxes and heads. JEV agrees too: the single
question run reports `A: 0.2219 / B: 0.7781` on both backends.

### But the device path is slower than the CPU path

Measured on the GB10 box (12B Q8_0, `--threads 8`, prompt "What is 2+2?",
8 output tokens, wall clock of the whole run):

| backend | wall | note |
| --- | --- | --- |
| `--compute cpu` | 1006–1119 ms | NEON kernels, 8 pool workers |
| `--compute auto` (before the fix) | 1391–1506 ms | ~40% slower than the CPU |
| `--compute auto` (after the fix) | 1061–1137 ms | projections stay on the CPU |

The per-shape numbers behind that gap (device vs CPU for one projection):

| projection | CPU | Vulkan, native layout | Vulkan, transposed layout |
| --- | --- | --- | --- |
| output head (2304 → 98304) | 2.6 ms/token | 4.0 ms/token | **1.5 ms/token** |

and the device-side breakdown of the 4.0 ms: `record` 0.011 ms,
`submit + wait_for_fences` 3.688 ms, `read` 0.016 ms — i.e. the device is
busy for 3.7 ms, it is the *kernel* that is slow, not the submission.

Three causes, two of which are fixed:

1. **The Q8_0 device kernel read the weight badly** — fixed. The kernel
   gave every lane its own output row, so a warp read 32 sectors one row
   apart and every 4-byte load straddled the 34-byte block boundary: 60
   GB/s against the 115 GB/s of the NEON path. The transposed layout
   (`src/vulkan/q8_layout.rs`, `shaders/glsl/q8_matmul_transposed.comp`)
   groups 64 output rows so a warp reads one contiguous run, and the same
   kernel then reaches **163 GB/s** — 2.7x, and above the 130 GB/s of
   PyTorch CUDA on this box. The result is bit-for-bit the same
   (0 mismatches over the output head).
2. **A one-off ~320 ms setup** — mostly fixed. 279 ms of it was the Vulkan
   context (the driver bringing the device up), 12 ms the pipeline
   compilation and 17 ms the weight upload. The device is now initialised
   on a background thread while the model loads
   (`ComputePolicy::configure_cli`), which took the 21 token prompt + 16
   output token run from 1009 ms to 685–750 ms.
3. **What is left is structural**: the single-row projection is only 240 MB
   of the 2.35 GB a decode step reads, and the CPU decode is not bandwidth
   bound anyway (53 GB/s against the 115 GB/s it reaches on the output head
   in isolation) — it is the 785 pool dispatches per token that hold it
   back. So the device and the CPU come out even (128 tokens: CPU
   5672–5742 ms, device 5134–5951 ms).

`RUST_GPU_SUBMIT_TRACE=1` confirms the shape of the problem for the whole
run: `submissions=30`, `3.710 ms/submission` of which `0.017 ms` is
`queue_submit` and `3.688 ms` is the fence wait. Forcing the threshold to
`RUST_GPU_MIN_OUTPUTS=0` (so every projection goes to the device) makes it
worse, not better: 5785 ms versus 2053 ms — every projection pays its own
upload, because the device path caches one weight at a time.

### What the trunk does about it

`src/models/mellum/trunk/forward.rs::project` runs the projections inside
`ComputePool::disable_gpu_matmul_for_scope()`, so the single-row projections
stay on the CPU and `--compute auto` matches `--compute cpu` instead of
losing 40%. The scope is per call, so nothing else in the trunk is affected.

The scope can go once the rest of the trunk moves: the output head on the
device is now level with the CPU (1.5 ms against 2.6 ms per token), so
leaving it on the CPU costs about 1 ms per token. What is missing is weight
residency (the device path re-uploads every weight) and one submission per
layer instead of one per projection; see TODO-049.

### The batched prefill

The fix for the *shape* is to stop evaluating one token at a time during
the prefill. `MellumRunner::forward_tokens` (`src/models/mellum/trunk/batch.rs`)
evaluates the prompt as one batch, so every projection carries all the rows
at once.

The arithmetic does not change: a projection is a per-row product and the
Q8_0 kernels accumulate each row independently, so a row's result does not
depend on how many rows share the call. The device path was checked
directly — the 2304 -> 98304 output head at 2, 4 and 8 rows gives **0
mismatches** against the single row device path and against the CPU path.
The attention, the MoE routing and the expert fold run per row, in the same
order as the single token walk, so the whole prompt is bit-identical:

```sh
# tests/mellum_batched_prefill.rs — the two paths, compared bit for bit
cargo test --profile release-fast --features vulkan \
  --test mellum_batched_prefill -- --nocapture
# compared 42 prompt tokens, 98304 logits
```

What the batch buys is traffic. One token streams 2.1 GB of weights (22.5 MB
of attention weights and 24 experts x 3 x 2.19 MB per layer, over 28
layers), and the prompt streamed that once per token. Batched, every weight
is read once for the whole prompt:

| measurement | single token | batched |
| --- | --- | --- |
| JEV single question (model load + prefill) | 2585 ms | 1014-1102 ms |
| end to end, 21 token prompt + 8 output tokens | 1006-1119 ms | 594-643 ms |
| end to end, 183 token prompt + 8 output tokens | 7558-8211 ms | 2911-3078 ms |

8 threads, three runs each, `--compute cpu`; `--compute auto` is the same
because the device no longer takes a Mellum projection. The end to end
numbers are the clean ones — the CLI's timer starts after the model is
loaded, so they are prefill plus decode and nothing else. The JEV number
includes the ~250 ms model load, because that path builds the model inside
the timer.

The device does **not** win the batched prefill on this box, and the reason
is the upload, not the shape. The device reads a weight once for the whole
batch, but it has to upload it first, and the host -> device path runs at
~10 GB/s while a DRAM read runs at ~100 GB/s. A prefill touches every
weight exactly once, so the upload is pure overhead: with the device taking
`wq` and `wo` of all 28 layers (`RMI_MELLUM_DEVICE_PREFILL=1`), the JEV
prefill went from 1073 ms to 1460 ms. The batched prefill therefore stays
on the CPU by default, and the knob is there for the serving case, where
the weights are uploaded once and reused by every request.

The one shape where the device does win is the 240 MB output head, because
that weight does not fit in cache and the CPU has to re-read it per row:

| rows | device | CPU |
| --- | --- | --- |
| 1 | 3.43 ms | 2.36 ms |
| 8 | 3.53 ms | 17.55 ms |
| 32 | 5.32 ms | 75.60 ms |

At 8 rows the device is 5x faster and at 32 rows 14x faster — but the
prefill only needs the logits of the last position, so that shape is not
available to the batched path.

## PyTorch reference

`tools/bench_mellum_pytorch.py` runs the official safetensors (bf16) through
`transformers`, so the Rust Q8_0 path has an apples-to-apples reference.

**CUDA, on an idle GPU (2026-11-04)**: PyTorch 2.15.0+cu132, 23.13 GB peak,
15 token prompt, 16 generated tokens:

| | decode |
| --- | --- |
| PyTorch CUDA bf16 | 29.4 tok/s (33.9 ms/token) |

**CPU, bf16 vs Q8_0 (2026-11-03)**: the same script on the CPU.

| path | weights | decode |
| --- | --- | --- |
| PyTorch CPU bf16 | 24.3 GB | 1.2 tok/s |
| Rust CPU Q8_0 | 12.9 GB | 1.2 tok/s |

Same decode speed at half the memory.

The CUDA run needs about 48 GB of peak (the bf16 weights *plus* the CPU
staging copy `from_pretrained` keeps while sharding), so it does not fit
while another tenant holds most of the unified memory — the 2026-11-03
attempt failed for that reason.

### What the CUDA number says about our device path

One decode step reads the attention weights of all 28 layers, the top-8
experts of all 28 layers and the output head: 2.21 B parameters, i.e.
4.42 GB in bf16 and 2.35 GB in Q8_0.

| path | bytes/token | decode | effective bandwidth |
| --- | --- | --- | --- |
| PyTorch CUDA bf16 | 4.42 GB | 29.4 tok/s | **130 GB/s** |
| Rust CPU Q8_0 | 2.35 GB | 22.7 tok/s | 53 GB/s |
| Rust Vulkan Q8_0 (output head only) | 2.35 GB | — | 66 GB/s |

So the device **can** stream 130 GB/s on this box. The Q8_0 device kernel
used to reach only 66 GB/s of that; with the transposed weight layout it
reaches **163 GB/s**, above the CUDA number. Two consequences:

* our CPU path is not bandwidth bound (53 GB/s against the 115 GB/s it
  reaches on the output head in isolation); the gap is the 785 pool
  dispatches per token, not the memory system;
* a Q8_0 decode at the kernel's 163 GB/s would be 14 ms/token, i.e. **70
  tok/s against the CPU's 22.7 tok/s** — the device wins Mellum's decode by
  3x if the weights stay resident and the projections are submitted per
  layer instead of per projection. See TODO-049.

## Entry points

| | |
| --- | --- |
| CLI text | `rust-model-inference --model <gguf> --prompt ...` |
| CLI JEV | `--jev --jev-question ... --jev-option ... --jev-context ...` |
| HTTP JEV | `POST /v1/jev/score`, `POST /v1/jev/grouped` |

The JEV probabilities agree bit for bit between the CLI and the HTTP
endpoint. A recorded CLI run:

```
choice: B
probabilities:
  A: 0.7335
  B: 0.2665
```

## Limits

* Scalar only. The Rust side needs `RMI_SCALAR=1`; the repo's NEON kernels
  accumulate differently and are not expected to match bit for bit.
* The oracle harness is single-token: it replays the prompt one token at a
  time, and the trunk has to do the same for the traces to line up. The
  trunk *also* has a batched prefill now
  ([`MellumRunner::forward_tokens`](../../../src/models/mellum/trunk/batch.rs)),
  and it is bit-for-bit the single token walk
  (`tests/mellum_batched_prefill.rs`), but it does not emit the per-slot
  expert checkpoints, so there is no *oracle* parity evidence for it.
* F32 KV cache only (`--kv-cache f32`); F16 KV is exercised only by the
  CLI smoke path.
* JEV is wired (CLI and HTTP, probabilities bit-identical) but there is no
  JEV parity evidence against the oracle. Embedding and rerank entry points
  do not exist for `mellum`.
* The CUDA reference ran on an idle GPU (29.4 tok/s, 130 GB/s); it needs
  ~48 GB of peak memory, so it does not fit while another tenant holds most
  of the unified memory.
