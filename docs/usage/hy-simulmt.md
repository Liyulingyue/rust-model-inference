# Hy-SimulMT

Use the official single-file GGUF from
[Tencent's Hy-SimulMT collection](https://modelscope.cn/collections/Tencent-Hunyuan/Hy-SimulMT).
Both translation-only and Interleave models include the audio encoder; no
separate mmproj is needed.

```sh
cargo build --release --bin rust-model-inference
target/release/rust-model-inference \
  --model models/Hy-SimulMT-30B-A3B-GGUF/Hy-SimulMT-30B-A3B-Q4_K_M.gguf \
  --audio speech.wav --language zh --temp 0 --max-tokens 512 \
  --threads 4 --compute cpu --kv-format f32 --max-context 8192
```

Input must be mono **16 kHz PCM16 WAV**. `--language` is the **target** language
(default Chinese); `zh`/`Chinese` and `en`/`English` are accepted aliases.
`--prompt` overrides the translation instruction, so its requested direction
should agree with the target language. Audio is processed in the model's 1.6 s
first turn and 0.8 s subsequent turns. The token limit applies to the whole
recording; a full KV context returns an error rather than silently dropping
history.

For Interleave, select `Hy-SimulMT-30B-A3B-Interleave-Q4_K_M.gguf`. Its metadata
selects the source-branch prompt automatically. Generated
`<｜hy_place▁holder▁no▁791｜>` separates source text from translated text.
The API returns token IDs and each turn's tokens in addition to rendered text.
CLI text is printed when the recording completes.

CPU only. Q4_K_M with F32 KV validation and numerical evidence are recorded in
[the Oracle guide](../../tools/oracle/hy_simulmt/README.md). GPU, TTS, HTTP ASR,
FP8 and other quantization formats are outside this integration's tested scope.
