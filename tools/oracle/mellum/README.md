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

## Limits

* Scalar only. The Rust side needs `RMI_SCALAR=1`; the repo's NEON kernels
  accumulate differently and are not expected to match bit for bit.
* Single-token replay only (the harness and the trunk both step one token at a
  time). There is no batched-prefill parity evidence.
* F32 KV cache only (`--kv-cache f32`); F16 KV is exercised only by the
  CLI smoke path.
* No JEV / embedding / rerank parity — those entry points are not wired for
  `mellum`.
