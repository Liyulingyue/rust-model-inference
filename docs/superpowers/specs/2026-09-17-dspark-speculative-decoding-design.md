# DSpark speculative decoding design

## Goal

Add exact greedy DSpark speculative decoding for these target pairs:

- Qwen3-4B with `deepseek-ai/dspark_qwen3_4b_block7`
- LFM2.5-1.2B-Instruct with the matching LiquidAI DSpark GGUF sidecar

The target model remains authoritative. Enabling DSpark must produce the same
greedy token sequence as target-only decoding while reducing target decode
passes when draft tokens are accepted.

## Reference contract

Use `ggml-org/llama.cpp` merge commit
`84075273c82f7681d43436b692073cbd4ab15fe9` as the read-only Oracle. The
sidecar must declare `general.architecture = dflash`, a positive
`dflash.block_size`, non-empty target layer IDs, and the DSpark Markov and
confidence tensors. Target and sidecar vocabulary, hidden width, tokenizer,
embedding, and output projection shapes must match.

The DSpark sidecar contains:

- an encoder projection from concatenated target layer inputs;
- a small non-causal Qwen-style decoder with its own KV cache;
- `markov_w1`, `markov_w2`, and `conf_proj` heads;
- no token embedding or LM head, which are shared from the target model.

Unknown or incompatible metadata and tensor shapes fail before generation.
There is no architecture fallback.

## User interface

Keep target-only behavior unchanged unless `--draft-model PATH` is present.
The first version adds:

- `--draft-model PATH`
- `--spec-draft-n-max N`, defaulting to the sidecar block size and clamped to it
- `--spec-draft-conf-min P`, default `0`, validated in `0..=1`

DSpark is greedy-only in this version. Combining `--draft-model` with a
non-zero temperature returns a clear error before model loading.

## Runtime design

One shared DSpark module owns sidecar loading, feature fusion, draft KV state,
block drafting, confidence truncation, target verification, and counters. It
does not add a general speculative-decoding framework.

Qwen3 and LFM2.5 expose only the operations the shared loop needs:

1. decode target tokens while capturing the input hidden state at the
   sidecar's requested target layers;
2. return logits for every verified token position;
3. checkpoint and restore mutable decode state at a token boundary.

For each cycle, the engine feeds newly committed target features into the
sidecar encoder, drafts up to `N` tokens in one non-causal block, then verifies
the block with the target. It commits the matching prefix and the target's
first differing token. Target KV rows after the committed length are ignored
and overwritten on the next pass.

Qwen3 rollback changes the committed sequence length. LFM2.5 rollback also
restores short-convolution state because it is mutated outside the attention
KV cache. No full model or weight copy is made.

The CLI reports drafted, accepted, and target-evaluation counts so a run can
prove DSpark was used rather than silently falling back.

## Validation

Tests are written before implementation and cover:

- rejecting a non-`dflash` sidecar and target/sidecar shape mismatches;
- clamping block size and confidence-based truncation;
- accepting a full matching block and rejecting at the first mismatch;
- restoring Qwen3 and LFM2.5 state after a rejected tail;
- CLI routing only when `--draft-model` is supplied.

Real-model acceptance uses fixed artifacts and records SHA256 values:

1. compare target-only and DSpark greedy token IDs for Qwen3-4B;
2. compare target-only and DSpark greedy token IDs for LFM2.5-1.2B-Instruct;
3. compare DSpark draft tokens, confidence values, accepted lengths, and final
   target tokens with the fixed llama.cpp Oracle;
4. run formatting, focused tests, release build, and both real CLI commands.

GPU drafting and non-greedy speculative sampling are outside this version.
They can be added after the CPU greedy path is bitwise aligned and benchmarked.
