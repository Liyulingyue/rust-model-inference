# Nemotron-3-Nano 30B-A3B (`nemotron_h_moe`) oracle

Upstream llama.cpp already implements `nemotron_h_moe`, so the oracle is a
plain upstream build plus a callback harness. No fork is needed.

## Files

| File | Purpose |
|---|---|
| `oracle.cpp` | `llama-eval-callback` harness; dumps the named tensors listed below |
| `build_oracle.sh` | extracts the pinned llama.cpp tarball and builds the harness |
| `gguf_header.py` | reads the KV metadata + tensor table from a partial GGUF download |
| `check_moe.py` | joins the oracle trace against the Rust `RMI_PARITY_TRACE` |

## Checkpoints

| Name | Shape | Meaning |
|---|---|---|
| `embd` | `[n_embd]` per token | token rows out of `ggml_get_rows` |
| `l_out` | `[n_embd]` per token | layer output after the residual add |
| `ffn_moe_logits` | `[n_expert]` | router output |
| `ffn_moe_probs` | `[n_expert]` | `sigmoid(logits)` |
| `ffn_moe_topk` | `[n_expert_used]` (I32) | selected expert ids, in selection order |
| `ffn_moe_weights` | `[n_expert_used]` | routing weights of the selected experts |
| `ffn_moe_out` | `[n_embd]` | routed term, before the shared expert |
| `ffn_shexp` | `[n_embd]` | shared expert output |
| `ffn_out` | `[n_embd]` | `moe_out + ffn_shexp` |
| `result_norm` | `[n_embd]` | final RMS norm |
| `result_output` | `[vocab]` | logits |

## Build

```sh
sh tools/oracle/nemotron3_nano/build_oracle.sh /tmp/opencode/llama_n3.tar.gz
```

## Capture

```sh
RMI_PARITY_TRACE=/tmp/opencode/n3_30b_trace/oracle.jsonl \
RMI_NEMOTRON_TRACE_FROM=0 \
RMI_NEMOTRON_THREADS=1 \
RMI_NEMOTRON_KV=f32 \
target/nemotron3-nano-oracle/build/bin/llama-eval-callback \
    models/Nemotron-3-Nano-30B-A3B-GGUF/Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf \
    /tmp/opencode/n3_30b_trace/prompt.txt 4
```

`RMI_NEMOTRON_TRACE_FROM` skips the first N positions, which is how a
divergence is bisected: capture from the first layer that differs instead of
dumping all 52.

## Compare

```sh
python3 tools/oracle/nemotron3_nano/check_moe.py \
    /tmp/opencode/n3_30b_trace/oracle.jsonl \
    /tmp/opencode/n3_30b_trace/rust.jsonl
```

## Shape notes

Two shapes are easy to get wrong, and both produced a wrong trace before they
were understood:

* **`ffn_moe_weights` is `[1, n_expert_used, n_tokens]`.** `ggml_get_rows`
  produces that, so `ne[1]` is the top-k slot axis and `ne[2]` is the token
  axis. Reading `ne[1]` as the row count silently writes one value per token
  instead of six.
* **The per-token checkpoints are `[n, n_tokens]`, not `[n_tokens, n]`.**
  `ne[0]` is the feature axis and `ne[1]` the token axis, so a row is
  `ne[0]` values at offset `row * nb[1]`. Writing `ne[1]` values instead
  produces a 4-byte file for a 2688-wide tensor, which the compare script
  reports as a length mismatch rather than as a shape bug.

## Rust side

`tests/nemotron3_nano_moe.rs` drives the trunk with explicit token ids rather
than going through the CLI, because the CLI wraps the prompt in ChatML while
the oracle feeds the raw prompt. That difference is real but it is about
prompt formatting, not about the trunk's arithmetic.
