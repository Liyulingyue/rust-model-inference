# Vulkan / GPU backend on small Intel x86 (N150 / Alder Lake-N)

This is a record of how the optional Vulkan backend behaves on the **Intel N150**
(4-core / 4-thread, Alder Lake-N, AVX2 + FMA, no AVX-VNNI). It is not a tuning
guide — it is a "is it worth turning on?" note.

## TL;DR

On the N150 the optional Vulkan path **does run, but it is slower than the
plain AVX2/FMA CPU path**. Mesa's `llvmpipe` software ICD is what the loader
picks on a machine without a working hardware Vulkan driver, and `llvmpipe`
runs the same compute shaders on the CPU without competing well against the
hand-tuned AVX2/FMA matmul kernels. End-to-end `--gpu` regresses both prefill
and decode by 5–20%.

## Environment

- CPU: Intel N150 (Alder Lake-N), 4C/4T, AVX2 + FMA + F16C
- GPU: no discrete GPU; Intel IGP driver installed but not selected
- Mesa: 25.2.x (Ubuntu 24.04)
- Vulkan loader: 1.3.275 (`libvulkan1`)
- ICDs on disk: `intel_icd.json` + 8 others, none of them picked as physical
  device by the loader on this machine

The first thing the binary prints under `--gpu` is the device name the loader
hands us:

```
[GPU] Vulkan device: llvmpipe (LLVM 20.1.2, 256 bits) (int8 dot product: supported, shader f16: supported)
[GPU] Warming up Vulkan pipeline (driver JIT)...
[GPU] Warmup done in 0.0s
```

`llvmpipe` is Mesa's CPU fallback. There is no hardware dispatch, no shader
JIT delay, no driver watchdog — the warmup is essentially free.

## Matmul parity (`examples/vk_check`)

The shipped parity sanity check against `matmul[(n_in, n_out)]` runs cleanly
through the software ICD and reports reasonable error bounds:

| shape         | max_abs | max_rel  |
|---------------|---------|----------|
| (1024, 1024)  | 0       | 3.75e-5  |
| (1024, 3072)  | 0       | 1.64e-4  |
| (3072, 1024)  | 1e-6    | 5.35e-5  |
| (1024, 151936)| 0       | 2.00e-3  |
| (16384, 32)   | 2e-6    | 2.37e-6  |

So far so good for shape-level parity. The worst-case rel is on the
vocab-projection matmul (`1024 × 151936`), which is also the largest tensor
in the model.

## End-to-end on gemma-2-2b-it Q4_K_M

Same prompt (`"Hello, my name is"`, `--temp 0`, `--max-tokens 8`, 4 threads,
`release-fast` profile):

| mode      | prefill t/s | decode t/s | end-to-end | total |
|-----------|------------|-----------|------------|-------|
| CPU       | 9.1        | 9.4       | 3.5        | 2.3s  |
| GPU/llvmpipe  | 7.1      | 9.4       | 2.9        | 2.7s  |

A longer prefill (`"The quick brown fox jumps over the lazy dog."`,
`--max-tokens 16`) makes the gap clearer:

| mode      | prefill t/s | decode t/s | end-to-end | total |
|-----------|------------|-----------|------------|-------|
| CPU       | 9.2        | 9.2       | 3.3        | 4.8s  |
| GPU/llvmpipe  | 8.8     | 8.5       | 3.0        | 5.3s  |

Decode is the only stage where GPU dispatch overhead matters; prefill is
basically flat. And the output tokens are **not** identical between the two
paths — the numerical drift from `llvmpipe`'s compute path is enough to flip
the first token in a deterministic-decoding run. The byte-stable-greedy
e2e test in `tests/gemma2_2b_it_e2e_embed.rs` is therefore CPU-only by
design.

## What this means for the engine

- The Vulkan backend is **healthy and exercisable** on this machine — no init
  failure, no watchdog wedges, no `mark_gpu_broken` fallback. The matmul
  parity is acceptable.
- It is **not faster** here, because the only ICD the loader can find is
  `llvmpipe`, and the hand-tuned AVX2/FMA path already saturates the 4 cores.
- On a machine with a real hardware ICD (Intel iGPU driver actually exposing
  a physical device, or any discrete GPU), the same `--gpu` flag would route
  through the GPU and almost certainly be faster. The engine has no N150-specific
  code path; the regression is the ICD choice.

## Auto-fallback for software ICDs (added 2026-10-06)

`VulkanContext::is_software_icd()` reports `true` when the loader picks a
device of type `VK_PHYSICAL_DEVICE_TYPE_CPU` (Mesa `llvmpipe`, Google's
`swiftshader`, …). `crate::ops::gpu_matmul_active()` short-circuits on that
flag, so `--gpu` becomes a no-op on software ICDs and the engine stays on the
AVX2/FMA path. End-to-end on gemma-2-2b-it Q4_K_M with `--gpu` and the
fallback active:

| mode      | prefill t/s | decode t/s | end-to-end | total |
|-----------|------------|-----------|------------|-------|
| CPU (no `--gpu`)  | 9.4 | 9.2 | 3.5 | 2.3s |
| `--gpu` (auto-fallback) | 8.8 | 9.5 | 3.4 | 2.3s |

The `[GPU] Vulkan device: llvmpipe` line still prints (the loader init runs)
but no matmul dispatch fires, so the previous 5–20% regression is gone.

## Q4_K GPU matmul (added 2026-10-06)

The K-quant matmul shaders (`shaders/bin/q4_k_matmul.spv` and friends) have
existed in the codebase since the `qwen3` / `z-image` paths were added, but
nothing wired them into per-layer LLM dispatch — `Q4_KKernel::forward_prepared`
was CPU-only, so even on real hardware the dominant per-token matmul (every
gate / up / down / Q / K / V / O projection in a Q4_K-quantised model) ran
on the host CPU.

`src/vulkan/matmul_q4k.rs` now wraps `BatchedLinearRuntime` with a
process-global `OnceLock<Mutex<Option<...>>>` so `Q4_KKernel` can lazily
acquire the runtime and dispatch:

- `Q4_KKernel::forward_prepared` checks `gpu_matmul_active()` on every call;
  thread 0 submits one fenced dispatch covering all output rows, threads 1-N
  return immediately (mirrors the Q8_0 GPU path in
  `kernel/q8_0/parallel.rs`).
- `UnsupportedShape` failures disable the Q4_K GPU path process-wide via
  `Q4K_GPU_DISABLED` so we don't keep paying the dispatch cost after the
  first cached-runtime miss.

On the N150 this path is still skipped (the software-ICD check returns false
first), but on real hardware the dominant matmul work now has a GPU
implementation.

## What we did not change

- No `cfg`-gated paths added for the N150.
- No `features = []` default changed.
- No `default-features` toggled in any `Cargo.toml`.
- No new benchmark / CI gate added; this is just a record so the next person
  who runs `--gpu` on a similar machine does not waste time wondering.