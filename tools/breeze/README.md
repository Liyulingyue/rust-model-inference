# Breeze TTS 2

Convert the complete original checkpoint, preserving BF16/F32 bytes and tensor names:

```sh
MODEL_DIR=/path/to/Breeze-TTS-2
uv run --with numpy python tools/breeze/convert_breeze.py "$MODEL_DIR"
uv run --with numpy python tools/breeze/test_convert_breeze.py
cargo build --release --bin rust-model-inference
BIN=target/release/rust-model-inference
MODEL="$MODEL_DIR/breeze-tts-2-BF16.gguf"
CODEC="$MODEL_DIR/breeze-tts-2-mmproj-F32.gguf"
```

Plain text, instruction, and reference voice modes (choose new output paths):

```sh
"$BIN" --tts --model "$MODEL" --mmproj "$CODEC" --prompt '你好。' --out plain.wav --seed 42
"$BIN" --tts --model "$MODEL" --mmproj "$CODEC" --prompt '你好。' --instruction '温柔地说。' --cfg-scale 3 --out instruction.wav --seed 42
"$BIN" --tts --model "$MODEL" --mmproj "$CODEC" --prompt '再见。' --ref-audio plain.wav --ref-text '你好。' --out clone.wav --seed 42
```

Reference input is PCM16 WAV, mixed to mono and resampled to 24 kHz. Sampling defaults are `--temperature 0.9 --top-k 50 --top-p 1 --seed 42`; `--temperature 0` selects greedy decoding. Repetition penalty is 1.0. A fixed seed reproduces the Rust sampler. CFG defaults to 3 for a nonempty instruction and 1 otherwise.

Compare exact F32 checkpoint bits between native revisions when debugging a calculation change:

```sh
TRACE_DIR=$(mktemp -d)
cargo build --release --features parity-trace --bin rust-model-inference
RMI_PARITY_TRACE="$TRACE_DIR/candidate.jsonl" "$BIN" --tts --model "$MODEL" --mmproj "$CODEC" --prompt '你好。' --temperature 0 --max-tokens 2 --threads 4 --out "$TRACE_DIR/candidate.wav"
python3 tools/breeze/compare_breeze_trace.py /path/to/reference.jsonl "$TRACE_DIR/candidate.jsonl"
```

The comparator fails at the first differing F32 bit pattern. Trace a failure through tensor layout, masks, normalization, and matrix dimensions. Use 24 kHz mono reference WAVs when checking the codec.
