# Qwen3.8-27B NariQ GGUF

The release uses the existing dense `qwen35` text model and the `qwen3vl_merger` vision projector. Its 64 layers interleave three recurrent layers with one attention layer. The hidden width is 5120, the FFN width is 17408 and the vocabulary has 248320 entries. Text generation and JEV use the existing Qwen CLI and HTTP routes. Image requests also need the released BF16 projector.

| File suffix after `Qwen3.8-27B-NariQ-` | Bytes | SHA-256 |
| --- | ---: | --- |
| `2.0bpw.gguf` | 6753954304 | `3a9e4955329dba8ba0be5509a9552220e5ccdae2f66bcc72c21c4580cd49bbb2` |
| `2.5bpw.gguf` | 8443119712 | `2445372e6de20620b940fbfab53aa718af9ba741f202da30cd2f02b5df78d05c` |
| `3.0bpw.gguf` | 10123626592 | `1eaad934aa05579b8eba47ed9b26450392f9875fc2354efe8cfa8b29be8f4419` |
| `mmproj-BF16.gguf` | 931145760 | `8832425f593068d3278e08eda4c1591072c7a2d8dc90f21e4f86394e4b2eb002` |

ModelScope revision `e703e809f2e673744424652a80f79e045ba309cc`. The three bpw labels describe mixed-precision exports. TRL1–TRL6 have fork-specific type IDs 44–49 under `trellis.version = UINT32 1`. Each 16×16 tile stores 32 × bits bytes. Each matrix has F16 block scales, F32 input/output scales and a descriptor. Optional gather tensors and vocabulary bands determine logical row order. The release uses identity and normalized block Hadamards up to 4096. Unsupported transform kinds produce a load error. The companion projector uses BF16 and F32 tensors. [Published files](https://modelscope.cn/models/nari-labs/Qwen3.8-27B-NariQ-GGUF/files) · [Pinned format and native checkpoint converter](https://github.com/nari-labs/nari-llama.cpp/blob/2644f7de461c27675eef373faa1776f20b393004/nariq/FORMAT.md)

## Capture and compare

Run the following on the machine that stores the weights. Download the named files from the pinned revision and verify their SHA-256. All paths below are on that machine. The reference fork's existing `test-trellis-model` writes metadata and complete F32 logit rows. Its CPU path reads the same coded weights as Rust.

```sh
git clone https://github.com/nari-labs/nari-llama.cpp oracle
git -C oracle checkout --detach 2644f7de461c27675eef373faa1776f20b393004
cmake -S oracle -B oracle/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build oracle/build -j 8 --target test-trellis-model mtmd
mkdir -p runs
prefix=models/Qwen3.8-27B-NariQ-GGUF/Qwen3.8-27B-NariQ-
for bpw in 2.0 2.5 3.0; do
  export NARIQ_MODEL="${prefix}${bpw}bpw.gguf" NARIQ_ORACLE="runs/oracle-$bpw"
  OMP_NUM_THREADS=8 oracle/build/bin/test-trellis-model --model "$NARIQ_MODEL" \
    --output "$NARIQ_ORACLE" --prompt 'The capital of France is' \
    --predict 3 --threads 8 --ubatch 8 --gpu-layers 0
  for mode in cpu auto vulkan; do
    NARIQ_COMPUTE="$mode" cargo test --locked --profile release-fast --features vulkan \
      --test nariq_reference nariq_prefill_and_decode_match_oracle -- --ignored --nocapture
  done
done
```

The test compares exact prompt and greedy token IDs, then compares the last batched prefill row and two teacher-forced decode rows over the whole vocabulary. It checks finiteness and applies the upstream maximum KL threshold of 0.001. It reports relative L2 and maximum absolute error for diagnosis. Strict Vulkan also requires a positive device submission count. [Reference comparison](https://github.com/nari-labs/nari-llama.cpp/blob/2644f7de461c27675eef373faa1776f20b393004/nariq/compare_logits.py)

The existing fixed-image probe supplies a 64×64 red image, the non-thinking chat template and F32 reference KV. Compile it against this fork and record its revision in the capture.

```sh
export LD_LIBRARY_PATH="$PWD/oracle/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
sed 's/b96806d96061049a5b574269b049bf6241d63d46/2644f7de461c27675eef373faa1776f20b393004/' \
  tools/oracle/holo4/27b/image.cpp > runs/nariq-image.cpp
c++ -std=c++17 -O2 -Ioracle/include -Ioracle/ggml/include -Ioracle/tools/mtmd \
  runs/nariq-image.cpp -Loracle/build/bin -lmtmd -lllama -lggml -lggml-base -o runs/nariq-image
export NARIQ_MMPROJ="${prefix}mmproj-BF16.gguf"
for bpw in 2.0 2.5 3.0; do
  export NARIQ_MODEL="${prefix}${bpw}bpw.gguf" NARIQ_IMAGE_ORACLE="runs/vision-$bpw"
  runs/nariq-image "$NARIQ_MODEL" "$NARIQ_MMPROJ" "$NARIQ_IMAGE_ORACLE.f32" > "$NARIQ_IMAGE_ORACLE.json"
  for mode in cpu auto vulkan; do
    NARIQ_COMPUTE="$mode" cargo test --locked --profile release-fast --features vulkan \
      --test nariq_reference nariq_image_logits_match_oracle -- --ignored --nocapture
  done
done
```

## Entry points

Set `NARIQ_MODEL` to the desired bpw file and `NARIQ_MMPROJ` to the BF16 projector. The image examples use a local PNG named `red.png`.

```sh
cargo build --locked --profile release-fast --features vulkan --bins
target/release-fast/rust-model-inference --model "$NARIQ_MODEL" --compute cpu \
  --prompt Hi --temp 0 --max-tokens 2 --max-context 256
target/release-fast/rust-model-inference --model "$NARIQ_MODEL" --mmproj "$NARIQ_MMPROJ" \
  --compute auto --image red.png --prompt 'Color?' --temp 0 --max-tokens 2 --max-context 256
target/release-fast/rust-model-server --model "$NARIQ_MODEL" --mmproj "$NARIQ_MMPROJ" \
  --compute vulkan --host 127.0.0.1 --port 8080 --max-context 256
```

Use the model ID returned by `GET /v1/models` in `POST /v1/chat/completions`. Chat accepts text and OpenAI-style `image_url` content, including PNG data URLs. Set `"stream": true` to receive SSE chunks through `[DONE]`.

The following CLI and HTTP requests score the same two labels. For image scoring add `--mmproj "$NARIQ_MMPROJ" --image red.png` to the CLI and send the image data URL in `image_url` to `/v1/jev/image`.

```sh
target/release-fast/rust-model-inference --model "$NARIQ_MODEL" --compute cpu \
  --threads 8 --prefill-batch-size 8 --max-context 256 --jev \
  --jev-context Red --jev-question 'Color?' --jev-option Red --jev-option Blue \
  --jev-positive A --jev-output json
curl -fsS http://127.0.0.1:8080/v1/jev/score -H 'Content-Type: application/json' \
  -d '{"context":"Red","questions":[{"text":"Color?","options":["Red","Blue"]}],"positive":"A"}'
```

The validation record reports full-vocabulary error metrics, observed backends and public entry results for each released file. [Measured coverage](validation.md)
