# Holo4-27B GGUF validation

The official [Hcompany/Holo4-27B-GGUF](https://modelscope.cn/models/Hcompany/Holo4-27B-GGUF)
uses the existing `qwen35` decoder and `clip` / `qwen3vl_merger` image encoder.
It does not declare a new GGUF architecture. Use `--jinja` to render its embedded
Hcompany chat template. Weights are published under CC-BY-NC-4.0.

## Fixed artifacts

ModelScope revision: `f9ab82b4f7484f3ffb156f649cdf8b1b35abb1ac`.
The official Hugging Face mirror revision is
`045834b6864fb5cee8e9c72e47ef4c19140ce932`; its LFS hashes match these files.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `q4_k_m.gguf` | 16875182592 | `8071d3ed5f59c88282e6466add0b344b51d49c8c630ae8f93d673ac0911ab962` |
| `mmproj.f16.gguf` | 927606816 | `b4038c228fcf84b0528e327c314204d165f6fa97ebb74d766424f49d2eba0d9c` |
| `chat_template.jinja` | 8952 | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |

Decoder contract: 64 implemented layers, no MTP, 48 recurrent / 16 dense
attention layers, hidden 5120, FFN 17408, vocabulary 248320, context 262144,
24 query / 4 KV heads of width 256. SSM conv/state/groups/rank/inner are
4/128/16/48/6144. MRoPE sections are `[11,11,10,0]`, dimension 64,
frequency base 10000000. The 851 tensors mix Q4_K, Q6_K and F32;
embedding and output matrices are Q6_K `[5120,248320]`.

Image contract: 334 tensors, 27 blocks, hidden 1152, FFN 4304, 16 heads,
patch 16, merge 2, nominal image size 768, projection 5120, mean/std
`[0.5,0.5,0.5]`. Linear weights are F16, norms and biases F32; no DeepStack
weights are present. The CLI uses the existing Qwen3.5 vision encoder, which keeps the original
F16 weights. The ARM NEON layer-normalization reduction previously loaded four
channels but advanced by eight, skipping half of every row. This adaptation
fixes the shared reduction and covers vector boundaries and scalar tails.

Tokenizer: GPT-2 BPE, `pre=qwen35`, BOS 248044 (not added), EOS 248046.
The embedded template matches the sidecar. The contract test checks empty
input, Chinese, combining Unicode, emoji, whitespace, digits, special tokens
and the exact non-thinking chat prompt against `llama-tokenize`.

## Reproduce

Use a read-only llama.cpp checkout at
`b96806d96061049a5b574269b049bf6241d63d46`. The builders instrument only a
disposable clone, disable Accelerate, BLAS, Metal and GPU execution, and reuse
the existing K-quant and softmax Oracle patches. The scalar binary is saved,
then only the scalar K-quant patch is reversed to build the native K-quant
reference with identical flags. No external acceleration library is used.

```sh
export RMI_LLAMA_CPP=/absolute/path/to/pinned/llama.cpp
export RMI_HOLO4_MODEL="/absolute/path/to/models/Holo4-27B-GGUF/q4_k_m.gguf"
export RMI_HOLO4_MMPROJ="/absolute/path/to/models/Holo4-27B-GGUF/mmproj.f16.gguf"
sh tools/oracle/holo4/build_oracles.sh \
  "$RMI_LLAMA_CPP" "$PWD/target/holo4/oracle"
export RMI_HOLO4_ORACLE="$PWD/target/holo4/oracle/build/bin/llama-eval-callback"
export RMI_HOLO4_SCALAR_ORACLE="$PWD/target/holo4/oracle/build/bin/llama-eval-callback-scalar"
export RMI_HOLO4_VISION_ORACLE="$PWD/target/holo4/oracle/build/bin/rmi-holo4-vision"
export RMI_HOLO4_ARTIFACTS="$PWD/target/holo4/text-parity"
cargo test --offline --profile release-fast --features parity-trace \
  --test qwen35_reference holo4_ -- --ignored --nocapture --test-threads=1
```

Choose fresh Oracle and text-artifact directories for each run; the builder
creates the clone and trace files append records. Vision tracing runs in-process and changes trace environment
variables, so keep these ignored tests serial.

Text comparison uses prompt `你好`, four greedy steps, CPU / one thread,
F32 KV and batch size 64. Token IDs, MRoPE positions, recurrent-layer flags,
checkpoint names/order/shape/occurrences are exact checks. Numerical traces
sample recurrent layer 0, dense layers 3 and 63, and the final norm/logits;
other layers are exercised but not individually traced. Embedding lookup
and the lossless initial recurrent-state checkpoint are compared as raw F32
words. Quantized matmuls, activations, norms and accumulated state use the
Holo4-specific relative-L2 envelope, leaving existing models' contracts unchanged.
For each checkpoint, measure the maximum error over the 16 forwards against the
native K-quant Oracle. Compare it with the independent maximum native/scalar
Oracle spread at that checkpoint: the bound is `max(1e-5, 1.5 * spread)`, capped
at `0.35`. The Oracle spread itself must also be at most `0.35`; nonfinite values
are rejected. These are smoke-test bounds, not a guarantee of general numerical
accuracy. All four greedy IDs must agree across all three implementations.

Vision comparison sends the same deterministic normalized 64×64 RGB fixture
directly to both encoders, bypassing image resize/normalization. All 29
checkpoints (embedding, 27 blocks, projection) must have identical names,
order and shapes. F16 activation rounding and SIMD
reductions use finite numerical comparisons with relative-L2 below 0.05; they
do not claim bitwise parity. The fixture produces four projected tokens of width 5120.

CLI examples:

```sh
cargo build --offline --profile release-fast --features parity-trace \
  --bin rust-model-inference
target/release-fast/rust-model-inference --model "$RMI_HOLO4_MODEL" \
  --jinja --prompt '你好' --temp 0 --threads 1 --kv-cache f32 --max-tokens 4
target/release-fast/rust-model-inference --model "$RMI_HOLO4_MODEL" \
  --mmproj "$RMI_HOLO4_MMPROJ" --image /absolute/path/to/image.png \
  --jinja --prompt 'Describe this image.' --temp 0 --threads 1 \
  --kv-cache f32 --max-tokens 4
```

## Coverage limits

On Apple ARM64, all three text traces contain 325 matching records. The prompt
IDs are `[248045,846,198,109266,248046,198,248045,74455,198,248068,271,248069,271]`.
All three produce greedy IDs `[109266,6115,98691,95815]` (`你好！有什么我`).
The maximum relative-L2 errors over the 16 forwards are:

| Checkpoint | Rust vs native Oracle | Scalar vs native Oracle |
| --- | ---: | ---: |
| recurrent layer 0 output | 0.000000240 | 0.000700695 |
| dense layer 3 output | 0.00312316 | 0.00486700 |
| dense layer 63 output | 0.260022 | 0.293617 |
| final norm | 0.260543 | 0.294095 |
| final logits | 0.201388 | 0.208964 |

Q8 activation quantization thresholds amplify small rounding differences over
64 layers: changing only the Oracle K-quant kernel already produces a 20.9%
logit difference on this sample. Rust is within that observed precision spread,
but the late-layer differences remain substantial. This is explicitly numerical
acceptance, not bitwise or 5% text parity. Initial zero state and embedding
words match exactly; later recurrent state is accumulated arithmetic and uses
the numerical bound. Running Rust with `RMI_SCALAR=1` did not remove the late
quantization spread, so no production arithmetic was rewritten to chase bits.

The real CLI also processes a solid red PNG through resize, image encoding,
placeholder insertion and decoding, returning `Red` with greedy sampling.

On Apple ARM64, the normalized-image fixture has matching checkpoint structure
and maximum relative-L2 `0.006871748` across all 29 checkpoints. The final
projection has relative-L2 `0.006871748`, maximum absolute difference `0.6018639`.
The largest intermediate absolute difference is `30.69519` in block 26, whose
relative-L2 is `0.006175620`. Before the NEON stride fix, block 0 already had
relative-L2 `0.3760240` and the final projection `0.7447838`.

A separate Linux x86-64 host (AMD EPYC 9334) passes the same 21 vision unit
tests and three reference-helper tests with a repository-local Rust 1.99.1
toolchain and offline vendored dependencies. Its SHA-verified F16 mmproj produces
4×5120 finite values for the same normalized-image fixture. Comparing those
values with the fixed CPU Oracle gives relative-L2 `0.007208926`, maximum
absolute difference `0.6387634`. These checks exercise the original F16 encoder;
the remote full-decoder CLI is still pending model transfer. No system packages
or global toolchain settings were changed.

The full library run reports 1237 passed, 80 ignored and 22 existing failures.
Compared with the unchanged baseline (1235 passed / 80 ignored / 23 failed),
the existing layer-normalization failure is fixed, the lane/tail regression
test passes, and no new failures are introduced. The remaining failures cover
Breeze, CLM, Z-Image, FunASR, Gemma4, Qwen3.5 invalid tensor dimensions, argmax,
softmax and RoPE; they are outside this adaptation.

Validation is scoped to these CPU Q4_K_M/F16 artifacts, the fixed text prompt
and small image fixture. Other quantizations, GPU/Vulkan, long context, video,
multi-image server execution and thinking-mode generation are unverified. The repository's Jinja renderer
does not supply tool definitions; this change does not implement the upstream
computer-use agent harness or validate task success on computer-use benchmarks.
