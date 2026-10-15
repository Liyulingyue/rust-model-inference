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

## Verification plan

Per the project's bar, the first milestone is the **text-only** path: get the
MoE routing right against the oracle before touching vision, so a regression is
unambiguously the router's or the projector's. The projector is already verified
and can be switched on afterwards with `--mmproj`, which is where its own parity
numbers live (see [`tools/oracle/minicpmv/README.md`](../../tools/oracle/minicpmv/README.md)).

## Known gaps at the time of writing

* Nothing in the language tower is implemented yet — this file records the
  contract, not a result.
* `expert_weights_scale` is read from the GGUF but its value has not been
  confirmed against the oracle for this checkpoint.
* The oracle's tensor table can build an MTP draft head; this GGUF has no
  `nextn.*` tensors, so that path is dead here but may still need to load
  cleanly if the arch gains it later.
