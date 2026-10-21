# LFM2.5-24B-A2B (`lfm2moe`) bit-parity oracle

Reference implementation for `general.architecture = "lfm2moe"`
(`src/models/lfm2moe/`), used by `tests/lfm2moe_reference.rs`.

## Pin

| | |
| --- | --- |
| upstream | [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) |
| commit | `749f688fcaa4c472ec034b08cb8a907c45cfaa02` |
| why this commit | it is the newest commit the repo's shared oracle tooling already pins, and `src/models/lfm2moe.cpp` at that commit already has the `case 40: type = LLM_TYPE_24B_A2B;` arm this model needs |
| `lfm2moe` graph | `llama_model_lfm2moe::graph` *is* `llama_model_lfm2::graph<iswa>` (`models.h:2248`), so every checkpoint comes from `src/models/lfm2.cpp` |

## Model under test

| | |
| --- | --- |
| repo | `LiquidAI/LFM2.5-24B-A2B-GGUF` (ModelScope) |
| file | `LFM2.5-24B-A2B-QAD-Q4_0.gguf` |
| size | 13 578 190 880 bytes |
| SHA-256 | `d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5` |
| arch | `lfm2moe` |

Contract read back from the GGUF (asserted by
`lfm25_24b_a2b_config_matches_the_pinned_contract`):

| key | value |
| --- | --- |
| `block_count` | 40 |
| `embedding_length` | 2048 |
| `attention.head_count` | 32 (head_dim 64) |
| `attention.head_count_kv` | `[0,0,8,0] * 10` — attention on layers 2, 6, …, 38 |
| `feed_forward_length` | 11776 (leading dense blocks only) |
| `leading_dense_block_count` | 2 |
| `expert_count` / `expert_used_count` | 64 / 4 |
| `expert_feed_forward_length` | 1536 |
| `expert_gating_func` | 2 (sigmoid) |
| `shortconv.l_cache` | 3 |
| `rope.freq_base` | 10000000 |
| `context_length` | 131072 |
| `vocab_size` | 128000 |
| tokenizer | `gpt2` BPE, `pre = lfm2`, bos 124894, eos 124900 |
| weights | Q4_0 everywhere, `token_embd` Q6_K, norms/router F32 |

## Build

```sh
# materialise the pinned tree (git archive works where a clone may not)
git -C /path/to/llama.cpp archive 749f688fcaa4c472ec034b08cb8a907c45cfaa02 \
    -o llama.tar && mkdir llama-lfm2moe && tar xf llama.tar -C llama-lfm2moe

tools/oracle/lfm2moe/build_lfm2moe_oracle.sh llama-lfm2moe
```

The script applies `lfm2moe-eval-callback.patch` and builds
`llama-eval-callback`. **No model maths is patched** — the patch only

1. rewrites `examples/eval-callback/eval-callback.cpp` into a checkpoint
   dumper, and
2. adds canonical checkpoint names to `src/models/lfm2.cpp`
   (`attn_norm`, `q_norm`, `k_norm`, `q_rope`, `k_rope`, `attn_proj`,
   `conv_in_proj`, `conv_conv`, `conv_out_proj`, `ffn_norm`,
   `post_attn_residual`) plus one line in `src/llama-graph.cpp` that labels
   `ffn_down` when the projection exists (upstream only labels it under
   `if (down_b)`, which LFM2 never sets).

### Scalar settings that actually matter

`/fp:precise` keeps MSVC from reassociating; it does **not** emit FMA by
itself. The load-bearing flag is **`GGML_AVX2=OFF` / `GGML_FMA=OFF` /
`GGML_F16C=OFF`**: on this llama.cpp version `GGML_NATIVE=OFF` only drops
`-march=native` and leaves the explicit x86 variants at `ON`, so
`arch/x86/quants.c:701` compiles the `#if defined(__AVX2__)` branch of
`ggml_vec_dot_q4_0_q8_0`, which accumulates with `_mm256_fmadd_ps`. That path
is ~2 ULP away from the reference dot and makes ~83% of every checkpoint
differ. With the variants off, `GGML_CPU_GENERIC` selects
`ggml_vec_dot_q4_0_q8_0_generic` and `quantize_row_q8_0_ref`.

`GGML_LLAMAFILE=OFF` keeps tinyBLAS out (it accumulates differently again).

Runtime settings the harness itself pins: 1 thread, `n_batch = n_ubatch = 1`,
F32 KV cache, flash attention off, `warmup = false`.

## Run

```sh
MODEL=models/LFM2.5-24B-A2B-GGUF/LFM2.5-24B-A2B-QAD-Q4_0.gguf

# 1. oracle: replay the RAW prompt (no chat template), one token per decode
RMI_PARITY_TRACE=/tmp/lfm2moe/oracle \
  llama-eval-callback -m "$MODEL" -p "What is 2+2?" -n 4

# 2. rust: same raw prompt through the same trunk
RUSTFLAGS='-C no-vectorize-loops -C no-vectorize-slp' \
  cargo build --profile release-fast --features parity-trace --test lfm2moe_reference

RMI_SCALAR=1 RMI_PARITY_TRACE=/tmp/lfm2moe/rust RMI_LFM2MOE_PROMPT="What is 2+2?" \
RMI_LFM2MOE_N_PREDICT=4 \
  cargo test --profile release-fast --features parity-trace \
    --test lfm2moe_reference lfm2moe_q4_0_matches -- --ignored --nocapture

# 3. compare
python3 tools/oracle/lfm2moe/check_scalar.py /tmp/lfm2moe/oracle /tmp/lfm2moe/rust
```

`check_scalar.py` joins both manifests on `(name, layer, step)` and diffs the
raw F32 words — no tolerance, no cosine similarity.

## Checkpoint names

```
embd
attn_norm  q_proj  k_proj  v_proj  q_norm  k_norm  q_rope  k_rope
attn_scores  attn_probs  attn_values  attn_proj
conv_in_proj  conv_conv  conv_out_proj
post_attn_residual  ffn_norm
ffn_up  ffn_gate  ffn_swiglu  ffn_down                      (dense layers 0-1)
ffn_moe_logits  ffn_moe_probs  ffn_moe_probs_biased  ffn_moe_topk
ffn_moe_weights  ffn_moe_weights_norm  ffn_moe_up  ffn_moe_gate
ffn_moe_swiglu  ffn_moe_down  ffn_moe_weighted  ffn_moe_out
l_out
result_norm  result_output
```

`attn_scores` / `attn_probs` are the oracle's `kq` / `kq_soft_max` trimmed to
the live `step + 1` entries per head and (for the scores) multiplied by
`1/sqrt(head_dim)`, which is the same value ggml folds into `soft_max_ext`.

## Verified

`RMI_LFM2MOE_PROMPT="What is 2+2?" RMI_LFM2MOE_N_PREDICT=4`
(8 prompt + 4 generated = 12 steps, 9926 checkpoint records per side):

* tokenizer: `prompt_ids = [124894, 2992, 355, 229, 26, 19, 26, 39]` — identical
* `generated_ids = [440, 5141, 355, 229]` — identical
* every compared checkpoint bit-identical

## Limits

* Scalar only. The Rust side needs `RMI_SCALAR=1`; the repo's AVX2/NEON
  kernels accumulate differently and are not expected to match bit for bit.
* Single-token replay only; there is no batched-prefill parity evidence.
* F32 KV cache only (the harness forces it and `Lfm2MoeSession::forward_logits_chunked`
  still asks for F16, so the parity run goes through `forward_token`).
* Only `QAD-Q4_0` is covered. `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `F16` and
  `BF16` builds of the same model are not.
* JEV / embedding entry points are not wired for `lfm2moe`.
