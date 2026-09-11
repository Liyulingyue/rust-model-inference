# Nemotron-3 Nano 4B parity fixtures

Reference outputs produced by **pinned `llama.cpp` build**:

```
build: b10120-96013c511 (2026-09-11)
```

Source: [`references/llama.cpp/`](../../../references/llama.cpp/) (local
checkout of ggml-org/llama.cpp at commit `96013c511`).
Binary: `references/llama.cpp/build-release/bin/llama-cli`.
Pin is recorded in [`docs/REFERENCE_IMPLEMENTATIONS.md`](../../../REFERENCE_IMPLEMENTATIONS.md).

Each fixture contains the raw interactive-mode output for a specific
prompt with `--temp 0.0 -n 32`. Invocation:

```sh
LD_LIBRARY_PATH=references/llama.cpp/build-release/bin \
  references/llama.cpp/build-release/bin/llama-cli \
  -m models/NVIDIA-Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron-3-Nano-4B-Q4_0.gguf \
  --single-turn -p "Hello" -n 32 --no-display-prompt --temp 0.0 --log-disable
```

## Usage in parity tests

The expected test path:
1. Run our Rust `nemotron_h` trunk on the same prompt with temp 0.0.
2. Strip the `[Start thinking]…[End thinking]` prefix (we don't
   reproduce the thinking block; the parity check focuses on the final
   response after `[End thinking]`).
3. Compare the post-thinking tokens / text against the file in
   this directory.

As of 2026-09-11 the Rust implementation produces coherent-but-wrong
English for short prompts (e.g. `believing Doudur asymptomatic
DouglasBe ...` for `Hello`) — the Mamba2 scan structure matches
llama.cpp but residual magnitudes drift over the 42 layers. The
parity test in [`tests/nemotron_h_parity.rs`](../../../tests/nemotron_h_parity.rs)
is `#[ignore]` until bit-exact match is recovered.

| Fixture | Prompt | Expected response (post-thinking) |
|---|---|---|
| `Hello.txt` | `Hello` | `Hello! How can I assist you today?` |
| `Translate_to_French:_hello_world.txt` | `Translate to French: hello world` | `Bonjour monde` |
| `What_is_2+2?.txt` | `What is 2+2?` | `2 + 2 = 4` |
| `The_quick_brown_fox.txt` | `The quick brown fox` | (thinking block, response cut at 32 tokens) |
| `Once_upon_a_time.txt` | `Once upon a time` | (thinking block, response cut at 32 tokens) |

The two cut-off fixtures are useful for verifying that the thinking
prefix is reproduced; the truncated response portion is `0` and
should not be compared.
