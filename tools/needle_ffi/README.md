# `needle-ffi` — standalone dev tool for the closed Cactus runtime

A single-file Rust program (no Cargo, no extra deps) that
`dlopen`s the closed `libneedle3.so` and prints a JSON transcript for
a given `.cact` + WAV pair.

This is **not** part of the main `rust-model-inference` crate. The
crate is pure Rust; this tool exists only to give us oracle data while
the conv stem is being reverse-engineered (see
`docs/develop/whistle-stem-todo.md`).

## Build

```bash
rustc tools/needle_ffi/main.rs -o /tmp/needle-ffi --edition 2021
```

No `Cargo.toml`, no third-party crates — only `std` and the libc
provided by the kernel for `dlopen` / `dlsym`.

## Run

```bash
/tmp/needle-ffi <whistle.cact> <audio.wav> [libneedle3.so]
```

`libneedle3.so` defaults to `<cact_dir>/libneedle3.so`; override with
the third arg or `WHISTLE_NEEDLE_LIB_PATH`. Language overrides via
`WHISTLE_LANG=en` (default: auto-detect).

Stdout: a single JSON object with `text`, `language`, `ttft_ms`,
`decode_tps`, `words` — pipe through `jq .text` for the transcript.

## Example

```bash
$ /tmp/needle-ffi models/whistle-probe/whistle.cact /tmp/whistle-probe/synthetic_speech.wav
needle-ffi: loaded 24000 samples (1.50 s) from /tmp/whistle-probe/synthetic_speech.wav
{"text":"I don't know what that looks like.","language":"en","ttft_ms":46.1,"decode_tps":188.0,"words":[...]}
```

## Exit codes

- `0` — success
- `2` — bad args
- `3` — load error (cact / wav / dlopen)
- `4` — transcribe error

## Why is this not a `pub mod needle_ffi` in the main crate?

Because the main crate's value proposition is "pure-Rust GGUF
inference". Any FFI dependency would (a) violate that premise and
(b) require non-default `dlopen` build configs that the release
binary doesn't want to drag along. This tool is opt-in dev
infrastructure: build it once with `rustc` and keep the binary in
`/tmp/`.
