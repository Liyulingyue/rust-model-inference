# Microsoft Aion Instruct 2.3B

The converter reads Microsoft's original BF16 Safetensors, config, tokenizer and chat template from one directory. It writes a native `aion_instruct` GGUF with a tied token embedding and vocabulary projection. The embedded tokenizer preserves the original Unicode pre-tokenizer and the whitespace rules on added tokens.

The checkpoint has 28 decoder layers, hidden width 2048, FFN width 8192, 24 query heads and eight KV heads of width 128. Full attention uses YaRN with a base of 32000, an original context of 2048 and a scaling factor of 16. The remaining layers use unscaled RoPE and a 2048-token window. The maximum context is 32768 tokens.

## Convert and run

Run from the repository root with NumPy available in the repository's Python environment. Set `model_dir` to the downloaded ModelScope snapshot.

```sh
python -m tools.converter.aion.convert "$model_dir" "$model_dir/Aion-2.3B-BF16.gguf" --precision BF16
cargo build --locked --profile release-fast --features vulkan --bin rust-model-inference --bin rust-model-server
target/release-fast/rust-model-inference --model "$model_dir/Aion-2.3B-BF16.gguf" \
  --prompt 'Reply exactly with OK.' --temp 0 --max-tokens 3 --compute cpu
target/release-fast/rust-model-server --model "$model_dir/Aion-2.3B-BF16.gguf" \
  --host 127.0.0.1 --port 8080 --compute auto
```

The converter also accepts F32, F16, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K. RMSNorm weights are stored in F32. Runtime activation and accumulation precision follow the shared kernels for the chosen format. `--kv-cache f16|f32` controls only KV storage. Aion chat uses its checkpoint template by default and accepts a `--chat-template-file` override. HTTP JEV scoring uses the checkpoint template and stores KV in F32.

Chat is available through `POST /v1/chat/completions`. Tool arguments are returned as JSON, and tools run on the client side. Text decision scores are available through `/v1/jev/score` and `/v1/jev/grouped`, and the equivalent CLI uses `--jev`.

## Reproduce alignment

The reference is Microsoft's model source at the file revision in [validation.md](validation.md). The reference math runs in F32 with eager attention. For quantized and F16 GGUFs, current gguf-py independently unpacks the same stored values into the official PyTorch module. This preserves the model contract while exposing the error from the runtime's activation rounding separately.

```sh
python -m tools.oracle.aion.oracle "$model_dir" target/aion-oracle/bf16
AION_GGUF="$model_dir/Aion-2.3B-BF16.gguf" AION_ORACLE=target/aion-oracle/bf16 \
  cargo test --locked --profile release-fast --features vulkan --test aion_oracle -- --ignored --nocapture

PYTHONPATH="$llama_cpp/gguf-py" python -m tools.oracle.aion.oracle \
  "$model_dir" target/aion-oracle/q8_0 --gguf "$model_dir/Aion-2.3B-Q8_0.gguf" \
  --follow target/aion-oracle/bf16/cases.json
AION_GGUF="$model_dir/Aion-2.3B-Q8_0.gguf" AION_ORACLE=target/aion-oracle/q8_0 AION_COMPUTE=vulkan \
  cargo test --locked --profile release-fast --features vulkan --test aion_oracle -- --ignored --nocapture

python -m tools.oracle.aion.verify_endpoints \
  "$model_dir/Aion-2.3B-BF16.gguf" target/aion-oracle/endpoints --compute cpu --jev
```

`--long-context` generates the 2094-token sliding-window case. `AION_KV=f16` selects half-precision KV in the Rust probe. `AION_COMPUTE=auto|vulkan` selects the device policy and `RMI_COMPUTE_TRACE=1` exposes projection dispatches and device submissions. On macOS, set `VK_DRIVER_FILES` to the installed MoltenVK ICD JSON.

## Replay packed projections

The whole-model rounded-activation comparison can show drift from different FP32 reductions feeding later integer rounding. Replay the captured row at every projection to check the packed kernels with an identical input. This includes Q/K/V, attention output, both FFN input projections, the FFN output and the tied vocabulary projection.

```sh
cargo build --locked --profile release-fast --features vulkan,parity-trace --bin rust-model-inference
RMI_PARITY_TRACE="$PWD/target/aion-oracle/projections.jsonl" target/release-fast/rust-model-inference \
  --model "$model_dir/Aion-2.3B-Q8_0.gguf" --prompt 'Reply exactly with OK.' \
  --compute cpu --temp 0 --max-tokens 2 --kv-cache f32 --threads 8
PYTHONPATH="$llama_cpp/gguf-py" python -m tools.oracle.aion.replay_projections \
  "$model_dir/Aion-2.3B-Q8_0.gguf" target/aion-oracle/projections.jsonl
```

Use a new trace file for each run. Set `--compute vulkan` to trace the same projections on a supported device. Add `--round-activations` to the oracle to reproduce the separate end-to-end quantitative gate for packed kernels.

[Verification results and remaining coverage](validation.md)
