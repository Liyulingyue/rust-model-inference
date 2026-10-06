# YuE2 AR on the GPU: the k-quant matmul shaders use 1 of 64 lanes

`--gpu --yue2` runs the AR half on the device (`src/vulkan/yue2.rs`). It is
correct end to end, and it is much slower than the CPU. This file records the
profile, the cause, and what has to change.

## The measurement

Same binary, same shapes, same seed; only the weight format differs.

| weights | per token | 28 layers | per layer | achieved |
| --- | --- | --- | --- | --- |
| BF16 (`yue2.gguf`) | 223 ms | 222.5 ms | 7.9 ms | 13.8 GOP/s |
| Q4_K_M (`yue2-ar_q4_k_m.gguf`) | 936 ms | 935.9 ms | 33.4 ms | 3.3 GOP/s |

Q4_K_M reads a quarter of the bytes and takes **4.2x longer**.

## Where the time is

`YUE2_GPU_PROFILE=1` splits the chunk into record / layers / head / submit /
read. For the Q4_K_M model:

```
record 0.5ms   layers 0.5ms   head 0.0ms   submit 936ms   read 0.0ms
```

Recording is free, readback is free, and `queue_submit` is 0.02 ms
(`RUST_GPU_SUBMIT_TRACE=1`). All of it is `wait_for_fences`, i.e. the device
is genuinely busy. `RUST_GPU_DISPATCH_TRACE=1` attributes it to two
pipelines: 168 quantize + 112 Q6_K + 56 Q4_K + 1 BF16 (`lm_head`) per token.

Two of those measurements had to be taken to get here:

- **Not the lm_head.** 184704 x 2048 is 20% of the FLOPs and looks like the
  obvious suspect. Skipping it changes 888 ms to 904 ms, i.e. nothing.
- **Not per-layer bookkeeping.** Truncating the layer loop gives 0.1 / 32.2 /
  66.4 / 129.4 / 258.4 / 904 ms for 0 / 1 / 2 / 4 / 8 / 28 layers: a perfectly
  linear 32 ms per layer, so nothing is amortizing badly and no single layer
  is special.

## The cause

Every matmul shader guards on `gl_LocalInvocationID.x != 0u`:

```glsl
layout(local_size_x = 64) in;
...
if (slot >= pc.group_count || row >= rows || gl_LocalInvocationID.x != 0u) return;
```

So each 64-thread workgroup does its work on lane 0 alone and the other 63
threads exit immediately. That is fine for the BF16 and F16 kernels, whose
inner loop is a straight `n_in` walk. It is expensive for the k-quants, whose
inner loop is 256 iterations of per-element unpacking:

```glsl
for (uint block = 0u; block < pc.blocks_per_row; ++block) {
    ...
    for (uint local = 0u; local < 256u; ++local) {
        uint packed = weight_byte(slot, offset + 16u + (group >> 1u) * 32u + (local & 31u));
        int value = int((group & 1u) == 0u ? packed & 15u : packed >> 4u);
        int activation = q8(block * 256u + local);
        partial[(local & 31u) >> 2u] += int(scale(slot, offset, group)) * activation * value;
        minimum_partial[group >> 1u] += int(minimum(slot, offset, group)) * activation;
    }
}
```

That whole 256-element block loop, plus the 6-bit unpack and the min-term
accumulator, runs on one thread of a 64-thread workgroup. The k-quants are the
only shaders with this much serial work per invocation, which is why they lose
4x while the dense shaders lose nothing.

The same guard is in `q4_0`, `q4_1`, `q5_k`, `q6_k`, `bf16`, `f16` and
`f32` matmul; the k-quants and `q4_0`/`q4_1` are the ones where it will
matter.

## What the device can actually do

`cargo run --release --features vulkan --example gpu_ceiling` on this GB10:

```
  calls    total   us/call    GOP/s
      1   0.41 ms    412.4      215
    256  94.93 ms    370.8      239
```

215-239 GOP/s, against a CPU reference of 201 GOP/s for the same kernel. So
the hardware, the driver and the dispatch path are all healthy: 3.3 GOP/s for
a k-quant matmul is the shader, not the device. The same table shows the fixed
submit round trip is about 370 us, which matters for a per-token decode but
does not explain this -- the whole chunk is one submission.

## 修 k-quant shader 的 1/64 lane：5.05× 但正确性未确认（已回退）

诊断（`docs/usage/yue2-gpu.md` 上文）说 Q4_K/Q6_K shader 慢 4.2× 的原因是
`gl_LocalInvocationID.x != 0u` 让 63/64 个 lane 立即返回，整条 256 元素解包循环
压在 lane 0。`q8_matmul_tiled_dp4a`（Z-Image 用，跑到 4088–5320 GOP/s）是**每 lane
一个输出、归约不跨 lane** 的范式，所以照抄：让每个 workgroup 处理 64 个输出行、
一行一个 lane，host 侧 k-quant 的 dispatch 从 `x=n_out` 改成 `x=ceil(n_out/64)`。

每个 lane 算自己那行的完全相同的求和顺序，**数学上逐位一致**。

**实测：AR 单 token 930 ms → 184 ms，5.05×。**

但**输出验证没通过**（PSNR 16 dB，GPU 与 CPU 音频差异明显），而且用 BF16 GPU
（不经过 k-quant shader）去比 CPU 也是 PSNR 16 dB、且两次 mean|Δ| 都恰好 ~3763，
**说明这个差异可能与 k-quant shader 无关，而是 AR 权重张量在 GPU/CPU 两条路径
本身就有别的分歧**（BF16 GPU 本该最接近 CPU）。

已回退，不留未验证的数值改动。留下的结论：

- k-quant shader 确实有 1/64 lane 的问题，且**按 tiled-dp4a 范式改成每 lane 一行
  能拿到 5×** —— 这个方向是对的，dispatch 网格推导也自洽
- 但 YuE2 AR 的 GPU 路径**在 k-quant 之外还有一个独立的正确性问题**，需要先定位它
  才能放心用 shader 修复。线索：BF16 GPU 与 CPU 也是 16 dB，且 mean|Δ| 在
  Q4KM-GPU / Q4KM-CPU / BF16-GPU 三者之间都接近 3763，提示差异可能来自 VAE/NAR
  （仍在 CPU）或 attention kernel 的 CPU/GPU 实现不同，而非 AR 权重本身
- 下一步应是：把 VAE/NAR 关掉、只跑 AR 单步，逐段对拍 BF16-GPU vs BF16-CPU 的
  hidden state，定位第一个分歧点，再回来收 shader 修复

## What would fix it

Split the 256-element inner loop across the workgroup and reduce once, instead
of returning 63 lanes. The accumulators already have the right shape for it:
`partial[8]` indexed by `(local & 31u) >> 2u` and `minimum_partial[4]` by
`group >> 1u` are both partial sums over disjoint subsets of the 256 values,
so a 64-thread version reduces `partial[8]` and `minimum_partial[4]` across
lanes and then does the same final `fma` chain on lane 0. The scale/minimum
terms are per-`group` constants, so they can be loaded once per group rather
than re-fetched 256 times.

Expectation: the k-quant matmul should approach the dense shader's
per-invocation throughput, which is the 4.2x between the two rows of the
table above. Even at parity with BF16 the device is 8x slower than the CPU's
21 tok/s for this stage, so the shaders are necessary but not sufficient --
single-token decode on a 20-core CPU is a hard target to beat.

## Status

`--gpu --yue2` is functional and falls back to the CPU on any error, but it is
about 20x slower than not using the device, so it is not advertised. Measured
stage cost for the 8-step / 960-frame config, for reference:

| stage | time |
| --- | --- |
| ABC | 29.8 s |
| semantic | 139.2 s |
| NAR prefill | 24.1 s |
| NAR 4 steps | 181.2 s |
| VAE decode | 159.9 s |

The AR half is 169 s of a 537 s render, which is the ceiling on what this
session can win.
