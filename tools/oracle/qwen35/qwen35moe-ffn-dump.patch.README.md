Adds the `qwen35moe` routed + shared expert checkpoints to the qwen35 oracle.

This is a separate patch from `qwen35-llama-trace.patch` on purpose: that patch
is pinned to `general.architecture == "qwen35"` and reproduces the 1B parity
result recorded in `docs/develop/QWEN35MOE.md`. Editing it would invalidate that
reproduction. Apply both only when targeting the 35B-A3B tower.

## Why this patch exists

`qwen35-llama-trace.patch` has no MoE node in its whitelist. Its
`selected_data` covers `conv_output_raw`, the `*_conv_predelta` family,
`state_predelta`, `attn_norm`, `Qcur*`/`Kcur*`, `l_out` and `result_*` only --
which is why the SSM chain could be pinned but the MoE could not. The
`ffn_moe_*` dumps people remember seeing belong to
`tools/oracle/lfm2moe/` and `tools/oracle/mellum/`, which are different models
with different node names.

So the MoE parity gap is not "the oracle output is hard to line up". The oracle
never produced those numbers for this architecture at all.

## Alignment: one token, not a ubatch

llama.cpp pads a multi-token prefill ubatch with three rows at each end, so a
3-token prompt arrives as `[pad,pad,pad,t1,t2,t3,pad,pad,pad]`. Comparing "our
token i" against "oracle row i" is then a misalignment, which is how this
branch once reported a ~14% discrepancy that was later withdrawn.

This patch therefore insists on a single-token decode. With one token there is
no padding row to misalign against, and `step == 0` is the same real token on
both sides. It refuses to run on a multi-token prompt rather than emit numbers
that cannot be lined up.

The token to use is whatever `common_tokenize` produces for your prompt; print
it with `-v` and feed the same id to the Rust side. Both sides must process
that one id.

## Nodes dumped

| node | meaning |
|---|---|
| `ffn_moe_input` | the block input `cur`, the argument to both expert paths |
| `ffn_moe_logits` | router output, one per expert |
| `ffn_moe_probs` | softmax over all experts |
| `ffn_moe_topk` | the selected expert ids, in descending-probability order (I32, written as f32) |
| `ffn_moe_weights_norm` | selected probabilities, renormalised over the chosen set and scaled by `expert_weights_scale` |
| `ffn_moe_out` | routed term alone, before the shared expert |
| `ffn_shexp_gate` | `ffn_gate_inp_shexp . cur`, pre-sigmoid |
| `ffn_moe_scale` | the sigmoid of that gate |
| `ffn_shexp_gated` | the shared expert's SwiGLU output |
| `ffn_moe_combined` | `moe_out + shexp * sigmoid(gate)`, i.e. `out` |

`ffn_moe_topk` is the one that can falsify a claim. "The same experts were
selected" has to mean the same ids in the same order; an earlier attempt on this
branch used first-exact-match token mapping over a prompt with repeated tokens
and collapsed every token onto one index, so it proved nothing.

The Rust side emits all ten under the same names via
`src/models/qwen35/trunk/trace.rs`.

## Apply

```sh
tools/oracle/qwen35/build_qwen35_oracle.sh          # as before
git -C $ORACLE_SRC apply tools/oracle/qwen35/qwen35moe-ffn-dump.patch
# rebuild, then:
RMI_QWEN35_DUMP=$DUMP ./build/bin/llama-qwen35-cli -m MiniCPM-V-4_7-35B-A3B-Q8_0.gguf -p '<one token>' -n 1 --temp 0 -v
```

Set `RMI_QWEN35_MOE_LAYER` to choose which layer's experts are dumped;
default 0. Dumping every layer's 256-wide router for 40 layers is large and the
router is the least interesting part -- the interesting comparison is
`ffn_moe_out` and `ffn_moe_combined` at one or two layers.
