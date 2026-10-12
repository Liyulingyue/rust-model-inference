# Whistle conv stem — TODO

The Whistle audio front-end is a 6-layer Conv2d + Linear stack
applied to a 16 kHz mono audio buffer before it reaches the encoder:

```
mel-frontend → 128-bin log-mel  →  stem  →  [B, T', 512] encoder input
```

Where the stem weights shipped in `whistle.safetensors` are:

| Tensor | Shape | Likely role |
|--------|-------|-------------|
| `stem/dw_1` | `[3, 3, 1, 128]` | Conv2d(1 → 128, k=3×3) |
| `stem/dw_2` | `[3, 3, 1, 128]` | Conv2d(1 → 128, k=3×3) |
| `stem/w`    | `[3, 3, 1, 128]` | Conv2d(1 → 128, k=3×3) |
| `stem/pw_1` | `[128, 128]`     | Conv2d(128 → 128, k=1×1) |
| `stem/pw_2` | `[128, 128]`     | Conv2d(128 → 128, k=1×1) |
| `stem/out`  | `[1280, 512]`    | Linear(1280 → 512) |

The `1280` input dim of the final linear is the puzzle — none of the
obvious interpretations of the three depthwise blocks (kernel 3×3,
in_ch=1, out_ch=128) collapse cleanly to that number. The closest
plausible layout is "three dw layers each downsampling 2× in time,
8× total = 10× downsample ⇒ 100 mel frames → 10 frames at 12.5 Hz,
matching `encoder_frame_ms=80`," but this is *speculative*.

## Why it's not implemented

* The cactus-compute public repo (`cactus-compute/needle`) ships the
  LLM stack only — the audio encoder + conv stem live in the closed
  C++ runtime `cactus` and are not exposed via the Python wrapper
  (which is ctypes-only and has no `forward_hook` for intermediate
  tensors).
* The `whistle.cact` artefact in the repo is a custom binary
  container with Cactus-Quants 2–4-bit weights — reverse-engineering
  the tensor order is documented in
  `cactuscompute.com/blog/cact-format` but the audio encoder stem is
  not part of that writeup.
* The Python Oracle + Rust port agree at 1 ULP per layer for the LLM
  half (encoder + stack + decoder logits). The conv stem is the only
  remaining piece with no public reference.

## Workaround

Until the stem is reverse-engineered, callers that need real ASR can
take a hybrid path:

1. Use the standalone dev tool at `tools/needle_ffi/main.rs` to run
   the closed C++ runtime against a WAV and capture the transcript
   (or, once `--dump-encoder-hidden` is added, the encoder hidden
   states as `.npy`). This tool is intentionally outside the main
   crate — the main framework stays pure Rust.
2. Save the encoder hidden states to a small `.npy` and pass them via
   a future `--inject-mel-hidden` CLI option (currently not
   implemented).
3. The Rust port then drives the encoder + stack + decoder logits
   from that precomputed tensor.

For end-to-end ASR today, the dev tool at `tools/needle_ffi/main.rs`
is the only path that produces real transcripts. Our Rust port is a
partial implementation: full coverage of the LLM half, all layers
1 ULP accurate against the Python reference, but the audio front-end
remains the open piece.