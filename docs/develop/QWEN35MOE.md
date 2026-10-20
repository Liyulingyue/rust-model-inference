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

### The MoE was never dumpable on this side either

`RMI_MOE_DUMP` used to write the per-token router logits, probabilities, routed
sum, shared sum and combined output, so the MoE could in principle be bisected.
It could not actually be used. The dumper opened six files with
`OpenOptions::append`, had no layer or token index, no length header and no
occurrence counter, so with 40 layers times N tokens every record interleaved
into the same six files and none could be attributed to a `(layer, token)` pair.
A dump nobody can attribute cannot falsify anything.

That is now gone. `src/models/qwen35/trunk/trace.rs` provides the same keyed
`(name, layer, step)` checkpoint the LFM2-MoE trunk uses, and `moe.rs` emits ten
of them -- `ffn_moe_input`, `ffn_moe_logits`, `ffn_moe_probs`, `ffn_moe_topk`,
`ffn_moe_weights_norm`, `ffn_moe_out`, `ffn_shexp_gate`, `ffn_moe_scale`,
`ffn_shexp_gated`, `ffn_moe_combined` -- under `RMI_PARITY_TRACE` with the
`parity-trace` feature. `forward_qwen35moe_token` now takes the layer and step
indices; before, it had no positional arguments at all.

`ffn_moe_topk` is the checkpoint that can falsify a claim. "The same experts
were selected" has to mean the same ids in the same order -- not a
plausible-looking output, and not first-exact-match token mapping.

### What the oracle side still lacks

`tools/oracle/qwen35/qwen35-llama-trace.patch` has **no MoE node in its
whitelist**: `selected_data` covers `conv_output_raw`, `*_conv_predelta`,
`state_predelta`, `attn_norm`, `Qcur*`/`Kcur*`, `l_out` and `result_*` only.
The `ffn_moe_*` dumps reachable in this repo belong to
`tools/oracle/lfm2moe/` and `tools/oracle/mellum/`, which are different models.
The same patch also rejects the architecture outright
(`architecture != "qwen35"` throws), so `qwen35moe` cannot load it at all.

So the MoE gap is larger than "the oracle output is hard to line up": **the
oracle has never produced those numbers for this architecture.** See
[`tools/oracle/qwen35/qwen35moe-ffn-dump.patch.README.md`](../../tools/oracle/qwen35/qwen35moe-ffn-dump.patch.README.md)
for the node list and the alignment rule. That patch is deliberately not
written yet -- authoring one needs a llama.cpp checkout to apply and build it,
and an unverifiable diff is the same class of artifact this section is about.

### Alignment is a single token, not a ubatch

llama.cpp pads a multi-token prefill ubatch with three rows at each end, so a
3-token prompt arrives as `[pad,pad,pad,t1,t2,t3,pad,pad,pad]`. Comparing "our
token i" against "oracle row i" is a misalignment, which is how this branch once
reported ~14% that had to be withdrawn. Running **one** token on each side
removes the padding row entirely and makes `step == 0` the same real token, so
no injection switch through the forward path is needed.

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
| `attn_output` from a zero state | 0.0045% |
| naive vs chunked recurrence, same inputs and state | 0.0063% |

The recurrence itself is also verified index-for-index against the oracle's
`build_delta_net_autoregressive`: decay, then `s^T k`, then `(v - sk) * beta`,
then the `k (x) d` outer product, then `s^T q / sqrt(S_k)`. The three ops in
`src/ops/ssm.rs` were checked against that op list one at a time -- with
`state[a][b]` mapping to the oracle's `S[b][a]`, `ssm_matvec` and
`ssm_outer_product_update` agree exactly.

`l2_norm` now matches `ggml_compute_forward_l2_norm_f32`
(`ggml/src/ggml-cpu/ops.cpp:4333`):

```c
ggml_float sum = 0.0;
for (...) sum += (ggml_float)(xi * xi);   // f64 accumulator, f32 products
const float scale = 1.0f/fmaxf(sqrtf(sum), eps);
```

Two details matter, and an earlier revision of this file got both wrong:

* **eps floors the denominator, it is not added under the root.** The kernel
  computes `1 / max(sqrt(sum), eps)`, not `1 / sqrt(sum + eps)`. The Rust side
  used `1 / sqrt(sum + eps)`; after the change to `1 / (sum as f32).sqrt().max(eps)`
  layer 1's whole SSM chain is bit-exact.
* **`sqrtf` narrows the f64 accumulator to f32 before rooting.** The sum
  accumulates in `ggml_float` (double), but `sqrtf` takes a `float`, so the
  value is rounded to single precision first. `(sum as f32).sqrt()` reproduces
  that; `sum.sqrt()` on the f64 would not.

The old pin, `qwen35_l2_norm_matches_pinned_llama_cpp_bits`, did not catch any of
this: it agreed 128/128 with the old Rust formula and 0/128 with the ggml one,
because it had been generated from the Rust code it was supposed to check. It is
regenerated from the kernel above, and
`qwen35_l2_norm_clamps_the_denominator_at_eps` now pins the floor directly.

With that fixed, every SSM stage at layer 1 is bit-exact end to end
(`conv_output_raw`, `q_conv`, `q_conv_predelta`, `k_conv_predelta`,
`state_predelta`, `final_output`), and the per-layer `l_out` drift collapses:

| layer | kind | `l_out` rms |
|---|---|---|
| 0 | SSM | 1.90e-09 |
| 1 | SSM | 2.96e-09 |
| 2 | SSM | 6.64e-09 |
| 3 | dense (first, `full_attention_interval = 4`) | 3.86e-05 |

### Layer 3: the remaining split is inside the attention core

Layers 0-2 are GDN/SSM and layers 3, 7, 11, ... are dense attention
(`full_attention_interval = 4`), so layer 3 is the first layer on the attention
path. Tracing it end to end on a single token gives:

| node | max abs | rms | state |
|---|---|---|---|
| `attn_norm` | 3.8147e-06 | 9.9098e-08 | differs (inherited from layer 2) |
| `Qcur` | 0 | 0 | **bit-exact** |
| `Kcur` | 0 | 0 | **bit-exact** |
| `Vcur` | 0 | 0 | **bit-exact** |
| `Qcur_normed` | 0 | 0 | **bit-exact** |
| `Kcur_normed` | 0 | 0 | **bit-exact** |
| `attn_gated` | 7.2300e-05 | 1.9874e-06 | differs |
| `attn_output` | 5.0902e-05 | 1.1363e-05 | differs |
| `attn_residual` | 5.0664e-05 | 1.1363e-05 | differs |
| `attn_post_norm` | 7.8106e-04 | 1.8525e-04 | differs |

Two hypotheses were checked and eliminated rather than assumed:

* **The score scale.** llama.cpp uses
  `kq_scale = hparams.f_attention_scale == 0 ? 1/sqrtf(n_embd_head) : hparams.f_attention_scale`
  while this repo hardcodes `1.0 / (n_embd_head as f32).sqrt()`. Reading the
  GGUF header shows **`qwen35moe.attention.scale` is absent**, so llama.cpp also
  takes the `1/sqrtf(n_embd_head)` branch and the two agree. (The hardcoding is
  still a portability gap relative to the llama convention at
  `src/models/llama/trunk/forward.rs:1282`, which honours
  `<arch>.attention.scale`; it is just not this model's bug.)
* **The value path.** `Vcur` is bit-exact, and with a single token `n_attend`
  is 1, so the softmax is exactly `[1, 0, 0, ...]` regardless of the score.
  The padded reduction in this repo (summed over `ceil(n_attend/256)*256`
  terms) cannot move the result either: adding exact `0.0` to an f64
  accumulator is exact.

With the value path clean, `attn_gated = Vcur * sigmoid(gate)` forces the
difference into the gate. The oracle cannot confirm it directly: `cb()` only
renames a tensor, and ggml fuses `sigmoid` into the following `mul`, so neither
the raw gate nor its sigmoid survives as a scheduled node. Both were emitted
and both were absent from the manifest, which is how the fusion was confirmed.
The gate lives in the other half of the fused QKV projection; `Qcur`, `Kcur`
and `Vcur` from that same projection are bit-exact, which makes a
per-column rounding difference unlikely but does not rule it out.

So layer 3 is where the split sits, and the first two things to try are the
fused QKV projection and the attention core's own output rather than more nodes
after the gate.

#### Where it actually comes from: llama.cpp's KV cache is F16

`attn_pregate` is the attention core's output before the gate, and at one
cached token the softmax is exactly `[1, 0, 0, ...]`, so `attn_pregate` must
equal `Vcur`. Checking that rather than assuming it:

| comparison | max | rms | bit-exact cells |
|---|---|---|---|
| oracle `attn_pregate` vs its `Vcur` | 9.2864e-04 | 9.3395e-05 | 0 / 4096 |
| **rust** `attn_pregate` vs its `Vcur` | **0** | **0** | **4096 / 4096** |
| oracle `attn_pregate` vs `f32(f16(Vcur))` | **0** | **0** | **4096 / 4096** |

This repo's attention output is `Vcur` exactly, which is the only answer the
arithmetic allows. llama.cpp's is `f32(f16(Vcur))` to the last bit, because its
KV cache defaults to half precision (`src/llama-context.cpp:3631-3632`,
`type_k` and `type_v` both `GGML_TYPE_F16`): the value is rounded once on the
way into the cache and widened back on the way out.

That is the whole of the layer-3 difference, and it is not a defect here.
Everything that depends only on the weights and the residual stream -- the fused
QG projection `Qcur_full` (8192 values), `gate_reshaped`, `Qcur`, `Kcur`,
`Vcur`, `Qcur_normed`, `Kcur_normed` -- is bit-exact, and the deviation begins
the moment the value passes through llama.cpp's F16 cache, then rides the out
projection into `attn_output` and the residual.

The relative size matches F16 and nothing else: `|difference|/|Vcur|` is
1.90e-04, 1.71e-04, 1.80e-04 and 1.66e-04 across four quarters of the
element-magnitude range -- a pure relative error of about 1.8e-04, against F16's
2^-11 = 4.88e-04 unit roundoff, and additive in nothing.

Two things follow. Matching llama.cpp bit-for-bit here means storing the KV
cache in F16 as well, which is a policy choice rather than a bug fix, and on
that reading this repo is the more accurate of the two.

#### A wrong turn worth recording

This one was reached by arithmetic and it was wrong. The gate cannot be observed
on the attention path -- llama.cpp takes it as a view of the fused QG projection
and fuses sigmoid into the following mul, so neither the gate nor its sigmoid is
a separate result. Recovering it as `gate = logit(attn_gated / Vcur)` assumed
`attn_out == Vcur`, and on the oracle side that assumption is false by exactly
the F16 round trip above. The reconstruction reported a gate difference of
2.13e-04 rms, ~1955 ULP, uniform across all 16 heads, and concluded the gate was
computed differently.

It was not. The gate is bit-exact; the arithmetic was measuring the F16 KV cache
and blaming the gate. Three explanations were killed before the right one:
that `cb()` cannot rename a tensor twice (it can, and a later `cb()` does
silently replace an earlier name -- real, but it only ever hid
`gate_reshaped`); that the eval callback skips nodes before the RoPE (it does
not; `RMI_CB_TRACE=1` shows `Qcur_full-3` and `gate_reshaped-3` each offered
exactly once); and that ggml's sigmoid is an approximation (it is not,
`op_sigmoid` is `1.f / (1.f + expf(-x))`, the same expression this repo uses).

The real cause was the least exotic one. Capture, selection and writing are
three separate lists: `Qcur_full`, `gate_reshaped` and `attn_pregate` were
captured correctly, passed the selection filter, and were then never written,
because the writer enumerates its own node list and they were not on it. So
"missing from the manifest" meant nothing until all three were checked, and
`RMI_CB_TRACE=1` -- which logs what the scheduler actually offers -- settles it
in one run.

#### The L2 fix did not move the end-to-end numbers

Seven single-token prompts, oracle tokenises, Rust replays the same id, compare
once, after the `l2_norm` fix:

| prompt | ids | top-2 margin | worst logit disagreement | verdict |
|---|---|---|---|---|
| `Hello` | 9419 | 1.2156 | 0.1683 | same token |
| `The` | 760 | 0.0350 | 0.1237 | same token |
| `dog` | 17907 | 0.3988 | 0.4640 | same token |
| `cat` | 4466 | 1.3751 | 0.2131 | same token |
| `run` | 5917 | 0.3308 | 0.1906 | same token |
| `water` | 12615 | 0.3000 | 0.9868 | same token |
| `1` | 16 | 1.3271 | 0.1907 | same token |

7/7 agree on the argmax. What matters more is that **3 of 7 have a margin below
the disagreement** (`The`, `dog`, `water`) -- and that is exactly the same 3 of
7 as before the `l2_norm` fix. The fix is a real correctness win at the layer
level: layers 0-2 are now bit-exact instead of merely close. It changed nothing
end to end, and there is now a measured reason rather than a shrug: every layer
runs its attention value path through llama.cpp's F16 KV cache, across 40 layers
and both KV heads, and that alone puts the final logits about 0.2 apart.

So the remaining `margin < disagreement` cases are **not** evidence of a defect
in this repo. They are the expected cost of keeping the KV cache in F32 while
the oracle keeps it in F16. Reading the table as "the model is broken here"
would be wrong, and so would reading it as "it works" because all seven tokens
match.

Note on shapes: the attention runs with `head_dim = 256`, so `Qcur` and
`attn_gated` are 4096 wide and `wo` projects 4096 -> 2048. That is not
`embedding_length / head_count`, and reading it as 128 makes the two sides look
misaligned when they are not.

### There is no chunked-vs-sequential gap

An earlier revision of this file claimed ~14% on multi-token prefill and
blamed `build_delta_net_chunking`. That was wrong, and so was the measurement.

llama.cpp pads prefill ubatches: a 3-token prompt produces a 9-row ubatch laid
out `[pad, pad, pad, tok1, tok2, tok3, pad, pad, pad]`, with the real tokens at
rows 3..5. Comparing "my token *i*" against "oracle row *i*" therefore compares
two different state trajectories, and no amount of care about which algorithm
runs fixes it.

The way out is the oracle's own `state_predelta` checkpoint, which is the state
entering the ubatch. Feeding the oracle's exact inputs into a plain sequential
recurrence from that state reproduces its chunked output:

| comparison | result |
|---|---|
| naive recurrence from `state_predelta` vs oracle `attn_output` | **avg 0.0063%, max 0.0786%** |
| Python naive recurrence vs this repo's `attn_out` | avg 0.0052%, max 0.0403% |

So the chunked WY formulation and the sequential recurrence agree here, and
this repo implements the sequential one. There is no algorithmic discrepancy to
close.

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
* The JEV surfaces accept it too — `src/app/jev/single.rs`
  (`IMAGE_JEV_ARCHES` + dispatch), `src/app/jev/grouped.rs` (both the
  `build_jev_token_ids_for_arch` match and the grouped dispatch) and
  `src/app/server/api.rs` (`run_multimodal_text_only`, which
  `/v1/jev/image` and `/v1/jev/image_grouped` share). It was registered on the
  generation path only, so the 35B could chat with an image but not be scored
  on one; since the MoE tower is a `Qwen35Model` with `moe` set, the existing
  `qwen35` scorers are reused as-is. See
  [`TODO.md`](TODO.md#入口面覆盖2026-10-19-审计).
* Vulkan is still gated to `qwen35` (`src/vulkan/qwen35.rs:17`), so the 35B runs
  on the scalar/SIMD CPU path only.
