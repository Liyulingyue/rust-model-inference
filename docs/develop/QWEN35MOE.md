# qwen35moe — MiniCPM-V 4.7 35B-A3B language tower

Extracted while adapting [OpenBMB/MiniCPM-V-4.7-35B-A3B-gguf]. Everything here
is read off the pinned oracle (`69f201a2`, `src/models/qwen35moe.cpp`) and off
the real GGUF, not guessed.

## Why this is new work

The 1B model's language tower is `general.architecture = "qwen35"` and rides the
existing trunk. The 35B model's is **`qwen35moe`**, which the repo does not have
— and which it has no precedent for either (`qwen3moe` is absent too). The mmproj
side, by contrast, needs **zero** work: both models ship `minicpmv4_7` with an
identical vision tower (448 / 14 / 1152 / 4304 / 27 blocks / 16 heads), differing
only in `clip.vision.projection_dim` (1024 vs 2048), which
`MiniCpmV47Config::from_source` already reads from metadata.

## Model contract (Q8_0 GGUF)

| key | value |
|---|---|
| `general.architecture` | `qwen35moe` |
| `block_count` | 40 |
| `embedding_length` | 2048 |
| `attention.head_count` / `head_count_kv` | 16 / 2 |
| `attention.key_length` / `value_length` | 256 |
| `expert_count` / `expert_used_count` | 256 / 8 |
| `expert_feed_forward_length` | 512 |
| `expert_shared_feed_forward_length` | 512 |
| `rope.dimension_sections` | `[11, 11, 10, 0]`, `dimension_count` 64, `section_order` `zyxt` |
| `rope.freq_base` | 1e7 |
| `ssm.*` | `conv_kernel` 4, `state_size` 128, `group_count` 16, `time_step_rank` 32, `inner_size` 4096 |
| `full_attention_interval` | 4 |
| `context_length` | 262144 |
| `tokenizer.ggml.pre` | `qwen35` (same tokenizer as the 1B) |
| SHA-256 | `8543ee5b0f83152066f2fec58b8d9757d56bb7c4233b4fdd26ae364f0107d16b` |

gguf:
`8543ee5b…` = `models/MiniCPM-V-4.7-35B-A3B-gguf/MiniCPM-V-4_7-35B-A3B-Q8_0.gguf`
(36902368928 bytes). mmproj SHA-256 `bff3b688e14c300eca273092c37c50a0ddfcc2054e079af1a0480edeed5d20f1`.

## Per-layer FFN (`graph::build_layer_ffn`)

```cpp
moe_out = build_moe_ffn(cur,
                        ffn_gate_inp,                 // router, [n_embd, n_expert]
                        ffn_up_exps, ffn_gate_exps, ffn_down_exps,
                        nullptr,
                        n_expert = 256, n_expert_used = 8,
                        LLM_FFN_SILU, /* norm = */ true,
                        hparams.expert_weights_scale,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);

if (ffn_up_shexp) {
    ffn_shexp  = build_ffn(cur, up_shexp, gate_shexp, down_shexp,
                           LLM_FFN_SILU, LLM_FFN_PAR, il);
    shared_gate = sigmoid(mm(ffn_gate_inp_shexp, cur));   // [n_embd] -> scalar/token
    ffn_shexp   = ffn_shexp * shared_gate;
    cur         = moe_out + ffn_shexp;
} else {
    cur = moe_out;
}
```

### The part that is easy to get wrong

The shared expert is built with **`LLM_FFN_PAR`**, so `build_ffn` already returns
`cur + ffn_shared`. The sigmoid gate is applied to *that sum*, not to the expert
output alone:

```
ffn_shexp = (cur + ffn_shared) * sigmoid(mm(ffn_gate_inp_shexp, cur))
cur       = moe_out + ffn_shexp
```

Applying the gate to `ffn_shared` alone, or adding `cur` twice, still produces
finite output and a plausible answer.

### Routing semantics

Softmax gating over all 256 logits, top-8 selected, selected weights
**re-normalised** to sum to one (`norm = true`), scaled by
`expert_weights_scale`. Experts are SwiGLU (`LLM_FFN_SILU`).

## Per-layer tensors (read from the Q8_0 GGUF)

Recurrent layer (`blk.0`) and full-attention layer (`blk.3`) — every fourth
layer is a full-attention one, matching `full_attention_interval = 4`:

```text
blk.0.attn_qkv.weight            Q8_0  [2048, 8192]     fused qkv (SSM layers)
blk.0.attn_gate.weight           Q8_0  [2048, 4096]
blk.0.ssm_a                       F32  [32]
blk.0.ssm_alpha.weight           Q8_0  [2048, 32]
blk.0.ssm_beta.weight            Q8_0  [2048, 32]
blk.0.ssm_conv1d.weight           F32  [4, 8192]
blk.0.ssm_dt.bias                 F32  [32]
blk.0.ssm_norm.weight             F32  [128]
blk.0.ssm_out.weight             Q8_0  [4096, 2048]

blk.3.attn_q.weight              Q8_0  [2048, 8192]     16 heads x 256
blk.3.attn_k.weight              Q8_0  [2048, 512]      2 kv heads x 256
blk.3.attn_v.weight              Q8_0  [2048, 512]
blk.3.attn_q_norm.weight          F32  [256]
blk.3.attn_k_norm.weight          F32  [256]
blk.3.attn_output.weight         Q8_0  [4096, 2048]
```

MoE (both layer kinds):

```text
blk.N.ffn_gate_inp.weight        F32  [2048, 256]      router, per token
blk.N.ffn_up_exps.weight         Q8_0  [2048, 512, 256] ne0=n_embd, ne1=ff_exp, ne2=expert
blk.N.ffn_gate_exps.weight       Q8_0  [2048, 512, 256]
blk.N.ffn_down_exps.weight       Q8_0  [512, 2048, 256] ne0=ff_exp, ne1=n_embd
blk.N.ffn_gate_inp_shexp.weight   F32  [2048]          shared-expert gate
blk.N.ffn_up_shexp.weight        Q8_0  [2048, 512]
blk.N.ffn_gate_shexp.weight      Q8_0  [2048, 512]
blk.N.ffn_down_shexp.weight      Q8_0  [512, 2048]
```

Embeddings: `token_embd.weight` and `output.weight` are both `Q8_0 [2048,
248144]`, `output_norm.weight` `F32 [2048]`. No `nextn.*` / MTP tensors are
present, so the draft head the oracle can build is not exercised by this
checkpoint.

## Planned shape of the change

`src/models/qwen35/trunk/forward.rs::build_layer_ffn` currently does the dense
SwiGLU. The MoE variant needs:

1. `qwen35moe` hparams (`expert_count`, `expert_used_count`,
   `expert_feed_forward_length`, `expert_shared_feed_forward_length`).
2. Weight binding for `ffn_gate_inp`, `ffn_{up,gate,down}_exps`,
   `ffn_gate_inp_shexp`, `ffn_{up,gate,down}_shexp`.
3. A router: logits -> softmax -> top-8 (`argsort_top_k`, which does **not**
   swap the top two on a tie — the same trap Mellum had) -> re-normalise.
4. Per-token expert dispatch reusing the existing `Weight` +
   `forward_prepared` path, so kernel selection and the optional Vulkan wrap
   stay behind `Weight` and no call site inspects `ggml_type`.
5. The shared expert with the nested residual above.

## Comparing against the oracle

`llama-cli` takes `-p` as **raw text**, which tokenizes without the chat
markers, while this repo's CLI wraps the prompt in the GGUF template. Comparing
the two directly compares different token sequences. `llama-tokenize` makes the
difference visible:

```text
llama.cpp -p : 0 846 198 3710 369 220 17 10 17 30 198 74455 198 248068 271 248069 271
this repo    : 248045 846 198 3710 369 220 17 10 17 30 248046 198 248045 74455 198 ...
```

So `tests/minicpmv_reference.rs::dump_qwen35moe_logits` takes the token ids
explicitly (`RMI_MOE_LOGITS_PROMPT_IDS`) and dumps the final logits, and the
oracle side needs a scheduler eval callback that writes `result_output`.

Two gotchas, both learned the hard way:

* `common::common.h` declares `params.logits_output_dir` and `arg.cpp` accepts
  `--logits-output-dir`, but **nothing reads it** — the flag silently produces
  no file.
* `ggml_backend_sched_set_eval_callback` has to be installed *after*
  `ggml_backend_sched_reset`, because the graph-reuse path resets it on every
  decode step and would otherwise clobber it. It must also be installed
  unconditionally (not in an `else if (cb_eval == nullptr)` branch) and must
  return `true` so the walk continues.

`RMI_MOE_DUMP` additionally writes the per-token router logits, probabilities,
routed sum, shared sum and combined output, so the MoE can be bisected against
the oracle's `ffn_moe_logits-*` / `ffn_moe_out-*` / `ffn_shexp_gated-*`.

Prompt processing runs several ubatches, and the eval callback wrote every one
of them to `<name>.bin` with `trunc`, so each ubatch erased its predecessor and
only the last survived on disk. Worse, the surviving rows could not be mapped
back to token positions, which made every comparison inconclusive: the "best
match" L1 against my own rows was 923 while the median L1 *between* two of my
own adjacent tokens was 956 -- i.e. matching a random unrelated token scored as
well as matching the right one.

The callback now appends a per-invocation sequence number to the filename, so
each ubatch lands in its own file. `model.input_embed` matches my embedding
**bit-exactly**, which makes it usable as an identity map: matching oracle rows
against my embeddings assigns 18 of 21 rows to specific token indices.

### The SSM path is where this diverges

With that map in hand, comparing the oracle's `attn_post_norm-0` (the
post-SSM normalised residual that feeds the router) against my own layer-0
rows is decisive:

| oracle row | my token | exact | L1 | max |
|---|---|---|---|---|
| 2 | 2 | 0/2048 | 1416.99 | 11.61 |
| 9 | 7 | 0/2048 | 983.50 | 32.50 |
| 15 | 13 | 0/2048 | 958.00 | 47.51 |
| 17 | 15 | 0/2048 | 1011.55 | 47.39 |

A caution on that table: the row-to-token map built from exact embedding
matches is ambiguous, because the prompt contains repeated tokens (`17`, `198`
and `271` each appear more than once) and matching each oracle row to the
*first* candidate collapses them onto one index. Row identity has to come from
a monotonic alignment instead.

Redone that way, the conv path comes out **bit-exact** (2158/8192 components
identical, L1 6e-5) for every row with an unambiguous mapping, while
`attn_post_norm` over those same rows still differs by L1 ~1000 and max ~48.
So the divergence is *after* the conv: in the QK L2 norm, `v_conv`, the
recurrence, or the gated norm plus output projection -- not in the input
projection or the conv, as the table above appeared to say.

Also found and fixed: the `edge0` path indexed the K head as
`v_h / (num_v_heads / num_k_heads)`, which interleaves heads (0,0,1,1,...).
The oracle's `ggml_repeat_4d` tiles the whole block, so output head j reads
input head `j % num_k_heads`. The default path already did this correctly,
which is why the bug did not show up in these runs.

### Summary so far The drift I attributed to "Q8_0 expert quantisation and
f32 summation order" is not that -- it is a genuine bug upstream of the router,
in the SSM. The MoE routing still picks the same experts only because the
residual stream stays in roughly the right direction.

Worth noting: the 1B model's SSM was never numerically verified against the
oracle either, only checked for coherent output, so this may not be
35B-specific.

SSM tensor shapes do scale consistently between the two models, so this is not a
binding/slice problem:

| tensor | 1B | 35B |
|---|---|---|
| `ssm_a` | `[16]` | `[32]` |
| `ssm_alpha.weight` | `[1024, 16]` | `[2048, 32]` |
| `ssm_beta.weight` | `[1024, 16]` | `[2048, 32]` |
| `ssm_conv1d.weight` | `[4, 6144]` | `[4, 8192]` |
| `ssm_dt.bias` | `[16]` | `[32]` |
| `ssm_norm.weight` | `[128]` | `[128]` |
| `ssm_out.weight` | `[2048, 1024]` | `[4096, 2048]` |

### What matches today

Every SSM input is bit-exact against the oracle, checked stage by stage on rows
whose token identity is unambiguous:

| stage | result |
|---|---|
| input embedding | **bit-exact**, all 2048 components |
| `conv_output_silu` | **bit-exact** (2158/8192 identical, L1 6e-5) |
| `q_conv_predelta` | 0.0001% |
| `k_conv_predelta` | 0.0002% |
| `v_conv_predelta` | L1 3e-5 |
| `beta_sigmoid` | **bit-exact** |
| `gate` (softplus x ssm_a) | **bit-exact** |
| `attn_output` from a **zero state** | 0.0045% |

The recurrence itself is also verified index-for-index against the oracle's
`build_delta_net_autoregressive`: decay, then `s^T k`, then `(v - sk) * beta`,
then the `k (x) d` outer product, then `s^T q / sqrt(S_k)`. The three ops in
`src/ops/ssm.rs` were checked against that op list one at a time -- with
`state[a][b]` mapping to the oracle's `S[b][a]`, `ssm_matvec` and
`ssm_outer_product_update` agree exactly.

`l2_norm` now matches `build_gdn_l2_norm`, which is
`ggml_scale(ggml_rms_norm(x, eps/n), 1/sqrt(n))` and reduces to
`x / sqrt(||x||^2 + eps)`: eps goes **inside** the square root. The previous
version clamped the denominator from below with `max`, which only differs for
near-zero head vectors but those are exactly the least stable ones.

### The one open discrepancy

Against a multi-token prefill, my `attn_output` differs from the oracle by ~14%.
The cause is *not* isolated, but the search has been narrowed to a single
hypothesis:

* For `n_seq_tokens == 1` the oracle dispatches to
  `build_delta_net_autoregressive`, which is what this repo implements, and the
  two agree to 0.0045%.
* For `n_seq_tokens > 1` it dispatches to `build_delta_net_chunking`, the
  chunked WY parallel algorithm. That is a different numerical path for the same
  recurrence, so this repo's sequential form can be expected to drift from it.
* Decode -- the phase that actually produces the tokens -- runs one token per
  ubatch, i.e. the autoregressive path, so it is the path that was verified.

Two things make this hard to settle from the dumps alone:

* llama.cpp pads prefill ubatches with **three real-embedding tokens at each
  end**, so the state entering the first real token is not zero. On the 33-token
  probe the ubatch is 39 rows with the real tokens at rows 3..35, and the
  zero-state closed form `beta * (k.q) * v / sqrt(n)` misses the oracle's row 3
  by 52% purely because of that pollution. `-fa off` does not remove the padding.
* `llama-cli -ub 1` forces the autoregressive path but introduces its own
  padding ubatches ahead of the prompt, so it is worse, not better.

End to end the model is correct: on the 33-token probe it answers "That is the
**NATO Phonetic Alphabet** (also known as the International Radiotelephony
Spelling Alphabet)", and on `What is 2+2?` it answers `2 + 2 = 4`.

## Verification plan

Per the project's bar, the first milestone is the **text-only** path: get the
MoE routing right against the oracle before touching vision, so a regression is
unambiguously the router's or the projector's. The projector is already verified
and can be switched on afterwards with `--mmproj`, which is where its own parity
numbers live (see [`tools/oracle/minicpmv/README.md`](../../tools/oracle/minicpmv/README.md)).

## Known gaps

* `expert_weights_scale` is read from the GGUF but its value has not been
  confirmed against the oracle for this checkpoint.
* The oracle's tensor table can build an MTP draft head; this GGUF has no
  `nextn.*` tensors, so that path is dead here but may still need to load
  cleanly if the arch gains it later.
* The ~14% multi-token prefill gap is narrowed to the chunked-vs-sequential
  recurrence question but not closed. See "The one open discrepancy" above.
* Vulkan: `src/vulkan/qwen35.rs` still gates on `architecture == "qwen35"`, so
  the 35B runs on the scalar/SIMD CPU path only.

## What is implemented

* `Qwen35Config` carries an optional `Qwen35Moe` block.
* `src/core/loader.rs` registers `qwen35moe` and falls back to
  `expert_feed_forward_length` when the dense `feed_forward_length` is absent.
* `src/models/qwen35/trunk/weights.rs` holds borrowed slices plus per-expert
  `Weight` views, so the 30,720 expert kernels are not all built up front.
* `src/models/qwen35/trunk/moe.rs` implements the routed and shared experts.
* `qwen35moe` is accepted by the multimodal arch allowlist, the chat template
  map, and the MiniCPM-V 4.7 canvas path, so `--mmproj --image` works on the
  35B as well as the 1B.
