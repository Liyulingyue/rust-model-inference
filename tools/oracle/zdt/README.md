# ZDTaichu5.0-9B oracle

The numerical record for ZDTaichu5.0-9B: a `qwen35` text tower with a
`nemotron_v2_vl` projector (a RADIOv4-H vision tower plus a ReLU-squared MLP
head).

Everything here was measured on 2026-11-05 on an aarch64 Linux host (CPU,
NEON, 8 inference threads), with the engine code of this branch.

## The artifacts

| File | SHA-256 | Notes |
| --- | --- | --- |
| `models/ZDTaichu5.0-9B-GGUF/ZDTaichu5.0-9B-Q4_K_M.gguf` | `83bda0ff1a752979b99c895bec84f2d6cfff0c3ab4c32bdc2da731774a4ac2f0` | text tower |
| `models/ZDTaichu5.0-9B-GGUF/mmproj-ZDTaichu5.0-9B-BF16.gguf` | `96f58047108b8f251b2b244f52e0b44be853f1231dffccc90787c898ee6b9684` | projector |
| `models/ZDTaichu5.0-9B-GGUF/mmproj-ZDTaichu5.0-9B-oracle-F32.gguf` | (derived) | the BF16 values widened exactly to F32 |

The two upstream files come from `ngquocvinh/ZDTaichu5.0-9B-GGUF` and match
the repository's `SHA256SUMS.txt`. The text tower also ships as Q5_K_M, Q6_K
and Q8_0; all four were verified against `SHA256SUMS.txt`.

The original weights, for the PyTorch reference, come from
`TaichuAI/ZDTaichu5.0-9B` on ModelScope. Only `model-5-of-5.safetensors` is
needed: it holds the whole vision tower (`vision_model.radio_model.*`) and the
projector head (`mlp1.*`).

## The reference build

The reference is ggml-org/llama.cpp commit
`11fe02151f79c41d0d4af7da708755d73b9c0da6`, pinned. Use a **temporary
checkout**:

```sh
git clone https://github.com/ggml-org/llama.cpp target/zdt/llama.cpp
git -C target/zdt/llama.cpp checkout 11fe02151f79c41d0d4af7da708755d73b9c0da6
./tools/oracle/zdt/build.sh target/zdt/llama.cpp target/zdt/oracle-build
```

`build.sh` applies `llama-trace.patch`, which adds the `zdt.projected` dump
hook to llama.cpp's `nemotron_v2_vl` graph and the `result_output` hook to the
text path. Set `RMI_ORACLE_SCALAR=1` for the scalar build.

## The vision projection

`llama-mtmd-debug -p encode --image blue -n 512` supplies already normalized
RGB floats `[0,0,1]` directly; it does not preprocess a blue PNG. The Rust
test feeds the same pixels.

```sh
export RMI_ZDT_MMPROJ=models/ZDTaichu5.0-9B-GGUF/mmproj-ZDTaichu5.0-9B-BF16.gguf
export RMI_ZDT_TEXT=models/ZDTaichu5.0-9B-GGUF/ZDTaichu5.0-9B-Q4_K_M.gguf

# The F32 reference: the same weight values, widened exactly.
PYTHONPATH=target/zdt/llama.cpp/gguf-py ./.venv/bin/python tools/oracle/zdt/widen_projector.py \
  "$RMI_ZDT_MMPROJ" models/ZDTaichu5.0-9B-GGUF/mmproj-ZDTaichu5.0-9B-oracle-F32.gguf

mkdir -p target/zdt/vision-ref-f32 target/zdt/vision-ref-bf16
RMI_ORACLE_DUMP=target/zdt/vision-ref-f32 RMI_ORACLE_FILTER='^zdt.projected$' \
  target/zdt/oracle-build/bin/llama-mtmd-debug -m "$RMI_ZDT_TEXT" \
  --mmproj models/ZDTaichu5.0-9B-GGUF/mmproj-ZDTaichu5.0-9B-oracle-F32.gguf \
  -p encode -n 512 --image blue -c 512 --no-warmup --no-repack -ngl 0 -t 8 -fa off > target/zdt/vision-ref-f32.log 2>&1
RMI_ORACLE_DUMP=target/zdt/vision-ref-bf16 RMI_ORACLE_FILTER='^zdt.projected$' \
  target/zdt/oracle-build/bin/llama-mtmd-debug -m "$RMI_ZDT_TEXT" \
  --mmproj "$RMI_ZDT_MMPROJ" \
  -p encode -n 512 --image blue -c 512 --no-warmup --no-repack -ngl 0 -t 8 -fa off > target/zdt/vision-ref-bf16.log 2>&1

export RMI_ZDT_REFERENCE=target/zdt/vision-ref-f32/0000.f32
export RMI_ZDT_REFERENCE_BF16=target/zdt/vision-ref-bf16/0000.f32
cargo test --profile release-fast --test zdt_oracle -- --ignored --nocapture
```

Measured:

| Reference | relative L2 | Bound |
| --- | --- | --- |
| llama.cpp, the BF16 values widened exactly to F32 | **0.00035952** | 0.001 |
| llama.cpp, the original BF16 weights | **0.00470340** | 0.01 |

## The upstream PyTorch reference

`llama.cpp` is itself a port of the upstream code, so it cannot catch a
shared misreading of the checkpoint. The upstream checkpoint's own PyTorch
implementation is the second reference. It lives in the checkpoint:
`cradio_model.py` (the C-RADIO tower), `cradio_config.py` and the `mlp1`
projector head.

```sh
# The tower with the input conditioner disabled, so it sees the same pixels
# as the Rust encoder, and with the GGUF's pre-downsampled position table.
./.venv/bin/python tools/oracle/zdt/pytorch_reference.py \
  --hf-dir models/ZDTaichu5.0-9B \
  --out target/zdt/pytorch-gguf-positions.f32 \
  --mode conditioner-identity --precision f32 \
  --positions models/ZDTaichu5.0-9B-GGUF/mmproj-ZDTaichu5.0-9B-BF16.gguf

export RMI_ZDT_PYTORCH_REFERENCE=target/zdt/pytorch-gguf-positions.f32
cargo test --profile release-fast --test zdt_oracle -- --ignored --nocapture
```

Measured: **0.00147076** relative L2 (bound 0.005).

The script needs `timm` (the upstream tower is a `timm` VisionTransformer)
and `torch`; both belong in the repository's `.venv`.

### What the upstream does differently

Two upstream details do not match the GGUF, and both are visible in the
numbers:

1. **The position embeddings.** The upstream keeps a 128x128 position table
   and interpolates it to the 32x32 grid at inference with
   `align_corners=False`. The GGUF was converted with `align_corners=True`
   (verified: bilinear with `align_corners=True` reproduces the GGUF's
   `v.position_embd.weight` exactly, relative L2 0.000000). The two tables
   differ by relative L2 **0.629774**, and that propagates to the projected
   features: the Rust encoder against the upstream's own interpolation
   measures **0.2015** relative L2.
2. **The normalization.** The upstream image processor normalizes with the
   ImageNet statistics from `preprocessor_config.json` and the tower's
   `InputConditioner` then normalizes again with the CLIP statistics stored
   in the checkpoint. llama.cpp and the Rust engine normalize once, with the
   GGUF's `clip.vision.image_mean` / `clip.vision.image_std`. The faithful
   upstream path therefore differs from llama.cpp by relative L2 **4.54**.

Neither difference is a Rust bug: the Rust encoder implements the GGUF
contract, which is also llama.cpp's contract. Reproducing the upstream's own
numbers needs a converter that re-derives the mmproj from the original
weights; that is tracked in `docs/develop/TODO.md`.

## The device path

The tower's projection matrices are loaded through the repo's `Weight`
abstraction, which the Vulkan feature wraps in a device kernel, so the matmuls
go to the GPU when one is available. The weights keep the GGUF's BF16 and
borrow the mapped bytes, so the encoder holds no F32 copy of them.

```sh
RMI_ZDT_GPU=1 cargo test --profile release-fast --features vulkan --test zdt_oracle -- --ignored --nocapture
RMI_COMPUTE_TRACE=1 target/release-fast/rust-model-inference --model "$RMI_ZDT_TEXT" \
  --mmproj "$RMI_ZDT_MMPROJ" --image path/to/blue-64.png --prompt 'hi' \
  --max-tokens 2 --compute vulkan
```

The trace prints `compute scope=linear backend=Vulkan rows=1034` for the
tower's matmuls, which is the projection running on the device.

Measured on the oracle's single 512x512 image:

| Path | Projection time | F32 reference | BF16 reference | upstream PyTorch |
| --- | --- | --- | --- | --- |
| CPU | 36.93 s | 0.00035960 | 0.00470345 | 0.00147076 |
| device | 27.62 s | 0.00035911 | 0.00470339 | 0.00147061 |

The CPU and device paths agree to about 5e-7 relative on every reference, so
they pass the same bounds. The first device call also uploads the 1.6 GB of
BF16 weights.

The device path is still only 25% faster rather than the order of magnitude
the tower's arithmetic would allow. The trace accounts for the whole 27.6 s:

| | measured | share |
| --- | --- | --- |
| runtime creation | 213 ms x 131 weights = 27.9 s | ~100% |
| fence latency | 2.5 ms x 229 submissions = 0.57 s | 2% |
| host/device traffic | 2.7 GB = ~30 ms | 0.1% |

`GpuLinear` caches its `BatchedLinearRuntime` per weight pointer, and every
weight pays for a fresh pipeline, so the tower's 131 weights rebuild the same
handful of shapes 131 times. The tower's weights have about six distinct
shapes, so keying the runtime on the shape instead of the weight would cut
the creation count by more than an order of magnitude.

The activation traffic is not the problem, so keeping the activations
resident is not the next step; the runtime cache is. That is tracked in
`docs/develop/TODO.md`.

## The text tower

The text tower is the shared `qwen35` path, so the text checks are the
standard ones.

```sh
export RMI_LLAMA_TOKENIZE=target/zdt/oracle-build/bin/llama-tokenize
cargo test --profile release-fast --test zdt_tokenizer_oracle -- --ignored --nocapture
```

The tokenizer IDs match llama.cpp exactly for the empty string, ASCII, mixed
CJK/emoji/whitespace, a chat turn and a vision placeholder run.

For the logits, the reference is `llama-eval-callback`:

```sh
prompt=$'<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
mkdir -p target/zdt/text-reference
RMI_ORACLE_DUMP=target/zdt/text-reference RMI_ORACLE_FILTER='^result_output$' \
  target/zdt/oracle-build/bin/llama-eval-callback -m "$RMI_ZDT_TEXT" \
  -p "$prompt" -n 8 -c 512 -ngl 0 -t 8 -fa off -ctk f32 -ctv f32 > target/zdt/text-reference.log 2>&1
cargo build --profile release-fast --features parity-trace --bin rust-model-inference
RMI_PARITY_TRACE=target/zdt/text-rust.jsonl RMI_PARITY_FILTER='qwen35.prompt_ids,prompt_ids,result_output' \
  target/release-fast/rust-model-inference --model "$RMI_ZDT_TEXT" \
  --prompt 'What is 2+2?' --max-tokens 8 --max-context 512 --threads 8 --temperature 0 > target/zdt/text-rust.log 2>&1
./.venv/bin/python tools/oracle/zdt/compare_text.py target/zdt/text-rust.jsonl target/zdt/text-reference target/zdt/text-reference.log --relative-l2 0.02
```

Measured: the greedy IDs `[19, 248046]` match llama.cpp token for token, and
the two steps measure relative L2 **0.00380972** and **0.00104810**
(bound 0.02).

`--temperature 0` is required: the CLI samples by default, so a sampled
token would make the Rust history differ from the reference history.

## The entry points

Measured with the Q4_K_M text tower and the BF16 projector:

| Entry | Result |
| --- | --- |
| CLI, text | "What is 2+2?" -> `4` |
| CLI, image | the blue fixture -> `blue`, 256 vision tokens |
| CLI, image | the red and green fixtures -> `red` and `green` |
| HTTP, text | `/v1/chat/completions` -> `4` |
| HTTP, image | `/v1/chat/completions` -> `blue`, and `red` / `green` for the other two fixtures |
| JEV, text | `red` 0.9825 / `blue` 0.0175 |
| JEV, image | `blue` 0.9999 / `red` 0.0001 |
| Vulkan, text | the 40-step greedy output is identical to the CPU's |
| Vulkan, image | `backend=partial mode=mmproj`, the vision tower runs on the CPU |

The three-colour check matters: a model that never saw the image can still
answer "blue" for a blue question.

## The quantization tiers

The text tower ships as Q4_K_M, Q5_K_M, Q6_K and Q8_0. The greedy output is
the same for all four (`4` for "What is 2+2?"), and the choice is the same,
but the confidence drifts:

| Tier | JEV `red` / `blue` | logit gap |
| --- | --- | --- |
| Q4_K_M | 0.9825 / 0.0175 | 4.03 |
| Q5_K_M | 0.9447 / 0.0553 | 2.85 |
| Q6_K | 0.9410 / 0.0590 | 2.77 |
| Q8_0 | 0.9159 / 0.0841 | 2.39 |

The vision projector is only published as BF16. The F32 file used above is
the BF16 values widened exactly, not a separate export.
