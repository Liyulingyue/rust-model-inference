# Nemotron-3 Nano 4B parity fixtures

Reference outputs produced by `llama.cpp` (pinned commit unknown — see
[`docs/REFERENCE_IMPLEMENTATIONS.md`](../../../REFERENCE_IMPLEMENTATIONS.md)
for the oracle pinning workflow). Each fixture contains the raw
interactive-mode output for a specific prompt with `--temp 0.0 -n 32`.

## Usage in parity tests

The expected test path:
1. Run our Rust `nemotron_h` trunk on the same prompt with temp 0.0.
2. Strip the `[Start thinking]…[End thinking]` prefix (we don't
   reproduce the thinking block; the parity check focuses on the final
   response after `[End thinking]`).
3. Compare the post-thinking tokens / text against the file in
   this directory.

The current Rust implementation produces degenerate output (see
`--prompt "Hello"` → `_On_On_On_On_On_On_On_On`). The fixtures
below represent the **target** behaviour after the Mamba2 selective
scan is implemented and the chat template is matched.

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
