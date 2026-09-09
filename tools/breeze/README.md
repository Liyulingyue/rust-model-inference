# Breeze TTS 2

Convert the complete original checkpoint, preserving BF16/F32 bytes and tensor names:

```sh
MODEL_DIR=/path/to/Breeze-TTS-2
uv run --with numpy python tools/breeze/convert_breeze.py "$MODEL_DIR"
uv run --with numpy python tools/breeze/test_convert_breeze.py
cargo build --release --bin rust-model-inference
BIN=target/release/rust-model-inference
MODEL="$MODEL_DIR/breeze-tts-2-BF16.gguf"
CODEC="$MODEL_DIR/breeze-tts-2-codec-F32.gguf"
```

Plain text, instruction, and reference voice modes (choose new output paths):

```sh
"$BIN" --tts --model "$MODEL" --mmproj "$CODEC" --prompt '你好。' --out plain.wav --seed 42
"$BIN" --tts --model "$MODEL" --mmproj "$CODEC" --prompt '你好。' --instruction '温柔地说。' --cfg-scale 3 --out instruction.wav --seed 42
"$BIN" --tts --model "$MODEL" --mmproj "$CODEC" --prompt '再见。' --ref-audio plain.wav --ref-text '你好。' --out clone.wav --seed 42
```

Reference input is PCM16 WAV, mixed to mono and resampled to 24 kHz. Sampling defaults are `--temperature 0.9 --top-k 50 --top-p 1 --seed 42`; `--temperature 0` selects greedy decoding. Repetition penalty is 1.0. A fixed seed reproduces the Rust sampler; it does not imply matching PyTorch random draws. CFG defaults to 3 for a nonempty instruction and 1 otherwise.

Compare full F32 checkpoint bits against the clean official checkout at `e2c5ac2f54fe15daa94237a7dbf31e446660a4c9`:

```sh
ORACLE=/path/to/clean/Breeze-TTS-checkout
# Oracle Python environment: torch==2.9.1 torchaudio==2.9.1 transformers==4.57.3 qwen-tts==0.1.1
TRACE_DIR=$(mktemp -d)
python tools/breeze/run_breeze_oracle.py --checkout "$ORACLE" --model-dir "$MODEL_DIR" --text '你好。' --frames 2 --threads 4 --trace "$TRACE_DIR/oracle.jsonl" --out "$TRACE_DIR/oracle.wav"
cargo build --release --features parity-trace --bin rust-model-inference
RMI_PARITY_TRACE="$TRACE_DIR/native.jsonl" "$BIN" --tts --model "$MODEL" --mmproj "$CODEC" --prompt '你好。' --temperature 0 --max-tokens 2 --threads 4 --out "$TRACE_DIR/native.wav"
python3 tools/breeze/compare_breeze_trace.py "$TRACE_DIR/oracle.jsonl" "$TRACE_DIR/native.jsonl"
```

The comparator fails at the first mismatched checkpoint; generation alone is not evidence of numerical parity. Use 24 kHz mono reference WAVs when checking codec parity.
