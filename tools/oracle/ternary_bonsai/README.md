# Ternary Bonsai 2.5 27B

The publisher's GGUF revision is `7d514b3e5efec2964456e23ddf012cc209b71ca8`. The model card calls the base Qwen3.8-27B. The decoder declares the existing `qwen35` architecture, with 64 trunk layers, width 5120, FFN width 17408, 24 query heads, four KV heads of width 256, 48 recurrent layers and a 248320-token vocabulary. A draft checkpoint has a 65th, dense NextN block with eight Q4_0 matrices. [Published artifacts](https://modelscope.cn/models/prism-ml/Ternary-Bonsai-2.5-27B-GGUF)

All six decoders store 402 folded matrices. The forward transform covers 401 named projections, including the output head. Embedding lookup uses the inverse transform. The three sign vectors span 5120, 6144 and 17408 lanes. Each normalized Sylvester transform covers 1024 lanes. Before `ssm_out`, the gather also transposes the recurrent value heads from 16 groups of three. The matching `clip/qwen3vl_merger` projectors have 27 vision blocks, patch size 16, merge size 2 and output width 5120.

The file prefix in the table is `Ternary-Bonsai-2.5-27B-`. The runtime borrows the original packed matrices from the GGUF. On first use, each CPU projection caches a lossless Q4_0 representation of its codes and FP16 scales. The 401 CPU projection caches total 13.41 GiB, in addition to the mapped GGUF, KV cache and session state. Embedding lookup decodes a single row. Vulkan converts and uploads one matrix at a time, then releases the host staging copy. The largest such conversion is the 682.03 MiB output head. CPU projections apply the signed transform before the shared SIMD Q4_0 dot. CPU and Vulkan use one signed Q8_K activation scale per 256 values for PQ2_0. Each Q4_0 projection block reads a copy of that scale. PTQ1_0 and Q2_0 use one Q8_0 scale per 32 values. The Vulkan path applies the transform on device. Ordinary dense Qwen weights have no transform. The GGUFs are distributed ready to run.

| File suffix | Bytes | SHA-256 |
|---|---:|---|
| PTQ1_0.gguf | 5993834848 | `929560ec9d37cf3cadd4b00cc87b51c570082a628dd830b4692dc0b2ed18dbbe` |
| PTQ1_0-mtp.gguf | 6232818944 | `e0fc5f3d47b9c926d52677d303b19df356a2376614b3e1cb9973865fc04ca431` |
| PQ2_0.gguf | 7253354848 | `103a11c94324029d8bae3f8152f30bba414730064cfb77438f8589db72e1aa85` |
| PQ2_0-mtp.gguf | 7492338944 | `ededc9fc0a8a6a8dd30914d107b2d2ce7674d789f1d05a1a481f240609aa6c23` |
| Q2_0.gguf | 7673194848 | `84496171ca79d4d33fb45612a2afd8303ec494baf916281023b7dc9cb3b3b9fa` |
| Q2_0-mtp.gguf | 7912178944 | `b0f83de8481c3200c529aef5fb495dbfedf2a4a3f017f7a3ea0a69c67be1d306` |
| mmproj-BF16.gguf | 931145856 | `e287342d92332fa3577ed1d42e921dac9370c08da58ba9337fa450f6cc76cfd7` |
| mmproj-Q8_0.gguf | 629246976 | `6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903` |

## Reproduce the comparison

On the inference host, set `MODEL` to one decoder and `PUBLISHER` to a checkout of PrismML's llama.cpp at `41c86f7d6d0ab27c5d0529958d7ee7aa3091309f`. Stock versions differ in their ternary decoder and Hadamard support. [Reference source](https://github.com/PrismML-Eng/llama.cpp/tree/41c86f7d6d0ab27c5d0529958d7ee7aa3091309f)

```sh
cmake -S "$PUBLISHER" -B "$PUBLISHER/build" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON -DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_WEBUI=OFF
cmake --build "$PUBLISHER/build" --target llama mtmd -j8
probe=tools/oracle/ternary_bonsai
lib="$PUBLISHER/build/bin"
c++ -O2 -std=c++17 -I"$PUBLISHER/include" -I"$PUBLISHER/ggml/include" \
  "$probe/reference.cpp" -L"$lib" -Wl,-rpath,"$lib" -lllama -lggml -lggml-base -o /tmp/bonsai-ref
# In a disposable checkout. These two examples are validation drivers.
cp "$probe/probe.rs" examples/bonsai_text_probe.rs
cp "$probe/image_probe.rs" examples/bonsai_image_probe.rs
cargo build --locked --profile release-fast --features vulkan \
  --example bonsai_text_probe --example bonsai_image_probe
prompt="The capital of France is"
/tmp/bonsai-ref "$MODEL" "$prompt" oracle > oracle.txt
target/release-fast/examples/bonsai_text_probe "$MODEL" vulkan "$prompt" oracle.txt rust.f32
python3 "$probe/compare.py" oracle.f32 rust.f32
```

For image logits use the fixed red-image driver in `tools/oracle/holo4/27b/image.cpp` against the same publisher checkout. Substitute the publisher SHA for its printed `oracle_pin` label. Link the same libraries plus `-lmtmd` and include `$PUBLISHER/tools/mtmd`. Pass `MODEL MMPROJ oracle-image.f32`. Give the Rust image probe `MODEL MMPROJ cpu|auto|vulkan RED.png rust-image.f32` with a 64×64 solid RGB red PNG and compare both dumps with the same script.

The text probe also supports `MODEL tokens PROMPT`. Compare the printed IDs to `/tmp/bonsai-ref --tokens "$MODEL" "$PROMPT"`. [Acceptance and observed runs](validation.md) · [Runtime examples](../../../docs/develop/TERNARY_BONSAI_25.md)

### 8,192-token retrieval and forced continuation

The prompt stores a recovery phrase before the archive entries and asks for it at the end. Generate it with the pinned tokenizer. Its SHA-256 is `3d5da3a016dd8b7120d87da1a253e8fd238f3beb2b8fbe91a53dafa537e8406f`. The drivers compare the final prefill logits and the next 15 positions using the publisher's continuation. Reuse one reference per decoder for all three compute policies.

```sh
python3 "$probe/long_prompt.py" /tmp/bonsai-ref "$MODEL" 8192 > archive.txt
prompt=$(cat archive.txt)
/tmp/bonsai-ref "$MODEL" "$prompt" archive-ref 8256 128 16 > archive-ref.txt
for device in cpu auto vulkan; do
  target/release-fast/examples/bonsai_text_probe "$MODEL" "$device" "$prompt" \
    archive-ref.txt "archive-$device.f32" 8256 128 > "archive-$device.log"
  python3 "$probe/compare.py" archive-ref.f32 "archive-$device.f32" --steps 16
done
```

Each `step` in the Rust log records its greedy token ID, elapsed milliseconds and device submission count. The comparison reports the two highest logits from each runtime whenever the greedy tokens differ.
