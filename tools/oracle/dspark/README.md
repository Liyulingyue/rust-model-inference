# DSpark speculative decoding oracle record

Reference: pinned llama.cpp `b96806d` — the same commit the Qwen3.5 /
Qwen3.8 CPU oracle uses. Upstream entry point (from that build's
`llama-server --help`):

```sh
llama-server --model <target.gguf> \
  --spec-draft-model <sidecar.gguf> \
  --spec-type draft-dspark \
  --spec-draft-n-max 8
```

`--spec-draft-n-max` defaults to 3 upstream. The `llama-completion` CLI
form does not expose the speculative flags — only `llama-server` does.

## Artifacts

| Role | File (repo-relative) | Bytes | SHA-256 |
|---|---|---|---|
| target | `models/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf` | 2,012,012,800 | `1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1` |
| sidecar | `models/Qwen3.5-2B-GGUF/Qwen3.5-2B-DSpark-Q8_0.gguf` | 491,066,752 | `82217c1d23dcdbeead2e68c79b448527cddd0d985f8a306d608c1b6bc39321bf` |

## What is verified

**PR #252** (merge commit `44527e5`), on Qwen3.5-2B Q8_0 + its DSpark
sidecar Q8_0:

- Draft tokens match `llama-server --spec-type draft-dspark` (b96806d)
  **token for token** on the same prompt.
- The speculative output equals the plain greedy decode **token for
  token**; the recorded CLI run committed 5 rounds with 14/40 drafts
  accepted.

**The HTTP entry follow-up** (same weights, this record):

- CLI `--dspark-model` and the runtime built by `--serve
  --dspark-model` produce **byte-identical** text for the same prompt,
  and both equal the plain greedy decode — the sidecar changes how many
  forward passes the tokens take, never which tokens come out. Pinned by
  the sentinel `cli_and_http_agree_on_dspark_speculative_decode`
  (`tests/cli_http_agreement.rs`, env `RMI_AGREEMENT_DSPARK`).
- Live smoke (`"Say hi to me"`, 24-token budget): both entries answer
  `Hello! 👋 How can I help you today?`; the CLI prints
  `[dspark] 11 rounds, 13/88 drafts accepted (14.8%)`. `temperature > 0`
  over HTTP returns `--dspark-model needs greedy decoding (temperature
  0)` instead of silently degrading.

## Why there is no fresh llama-server replay here

The two servers do not share a prompt builder: llama-server's default
chat template leaves thinking on and renders a 14-token prompt where this
repo's builder renders 16, so a fresh end-to-end replay would compare
two different prompts. PR #252's token-for-token parity was established
by feeding the same prompt to both stacks, and that conclusion stands;
re-deriving it through both HTTP fronts would first need a shared raw
prompt (this repo's server has no raw `/completion` endpoint), which is
out of scope for the entry-point wiring.

## Reproduce

Run from the repository root; `--profile release-fast` matches the repo's
test convention.

```sh
# CLI and HTTP agreement (greedy + image cases ride the same sentinel file)
RMI_AGREEMENT_MODEL=models/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf \
RMI_AGREEMENT_DSPARK=models/Qwen3.5-2B-GGUF/Qwen3.5-2B-DSpark-Q8_0.gguf \
cargo test --profile release-fast --test cli_http_agreement

# Sidecar load + end-to-end greedy parity (PR #252's oracle tests)
RMI_DFLASH_SIDECAR=models/Qwen3.5-2B-GGUF/Qwen3.5-2B-DSpark-Q8_0.gguf \
RMI_DFLASH_TARGET=models/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf \
RMI_QWEN35_MODEL=models/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf \
cargo test --profile release-fast --lib -- --ignored dflash:: dspark

# CLI smoke
./target/release-fast/rust-model-inference \
  --model models/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf \
  --dspark-model models/Qwen3.5-2B-GGUF/Qwen3.5-2B-DSpark-Q8_0.gguf \
  --prompt "Say hi to me" --n-gen 24 --temp 0

# HTTP smoke
./target/release-fast/rust-model-inference \
  --model models/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf \
  --dspark-model models/Qwen3.5-2B-GGUF/Qwen3.5-2B-DSpark-Q8_0.gguf \
  --serve --host 127.0.0.1 --port 8080
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi to me"}],"max_tokens":24,"temperature":0}'
```

Acceptance-rate caveat: the rates above are single-prompt smoke numbers
on a chatty prompt, not a benchmark; the sidecar's value shows up on
longer continuations.
