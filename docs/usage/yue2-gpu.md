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

## 继续收窄：分歧起于「第一个读历史的 attention 输出」

按 row 逐点 dump 后（`YUE2_DUMP_XIN`，临时插桩已回退）：

| pos | CPU embedding | GPU embedding |
|---|---|---|
| 36 | 0.028076172, -0.05908203 | **同左，逐位一致** |
| 37 | -0.021118164, 0.007019043 | -0.023925781, 0.003967285 ← **不同** |
| 38 | -0.0021209717, 0.013000488 | -0.023925781, 0.003967285 ← **GPU 重复了 37 的值** |

pos=38 的 GPU 值等于 pos=37 的 GPU 值，不是算错而是**下游发散**：pos=37 的输入
取决于 pos=36 的**28 层输出**，而 pos=36 的 embedding 本身一致。也就是说分歧**不在
embedding、不在 rms_norm 的输入**，而在 **pos=36 这一行经过 attention 之后的残差流**。

为什么之前测「prefill 的 attention 输出逐位一致」？因为那个 dump 读的是
`layout.attn` 的 region 起点（第 0 行 = prefix 的 token），不是当前 chunk 的第 36 行。
所以它一直「一致」，是**读错了行**。同一条记录里 pos=38 重复 pos=37 的值，也是同一个
坑：region 起点 vs 当前 row。

逐层 trace（`YUE2_TRACE_LAYER`，每层 submit+回读）证实 n=35（prefill 中间）全部层
逐位一致 —— 但因为每层都同步，GPU trace 只能跑到 n=35，到不了出问题的 n=37，所以
「逐层第一个偏离点」还没拿到。

⚠️ **这是今晚在这上面踩的第三个读法坑**，值得单独记：arena region 的 `read_f32(region)`
从 region 起点读，region 大小是 `max_rows * width`，**不是**整条序列。所以
1) prefill(36行) 时读 region 起点得到 token 0，不是 token 35
2) decode(1行) 时读 region 起点得到本 chunk 那一行（这个对）
3) 要看第 k 行必须按 `(k * width)` 偏移自己切片

## 分歧的精确范围（逐段 dump 实测）

在真实 GGUF 上用临时插桩逐段对比 CPU/GPU（插桩已回退）：

| 段 | n=36 (prefill, rows=36) | n=37 (decode, rows=1) |
|---|---|---|
| embedding (x 入口) | 一致 | 一致 |
| normed = rms_norm(x) | **一致** | **不同** |
| q = qk_norm_rope(normed) | **一致** | **不同** |
| KV cache (k,v) | **一致** | 一致 |
| attention 输出 | **一致** | **不同** |
| logits | **一致** | **不同** |

即：**分歧出在 rms_norm 之后、qkv 之前的 normed，而且只在 rows=1 的 decode
chunk 里出现**。因为 normed = rms_norm(x)，KV 和 embedding 又都一致，所以 x
（28 层残差累积）在第一个 decode chunk 内就已经错了 —— 而 prefill（rows=36）
整条 x 流是对的。

也就是说：rows>1 时残差流正确，rows=1 时错。最可能的原因是某个 recorder 在
rows=1 时依赖了跨 chunk 残留的 region 内容（例如 `layout.x` 只写了 rows 行，
而某个 in-place 的 residual add / 投影读到了上一 chunk 的残留，或 attention/scores
region 的有效范围在 rows=1 时算错）。

⚠️ 注意：我尝试逐层 dump `x` 时，读 `layout.x` region 起始位置拿到的是 prefix
残留（第 0 行）而非当前 chunk 的 row，导致一度误读成"GPU base_position 跑偏"。
逐层 dump 需要按 row 偏移读，不是按 region 起点。这是排查时的一个坑。

下一步应当：在 forward_chunk 内按 row 偏移 dump 每层后的 x（rows=1 时取
`layout.x + base_position*hidden`），定位第几层的残差开始偏离。

## 分歧定位：多 token prefill 逐位一致，单 token decode 全错

用 `YUE2_DUMP_LOGITS`（临时插桩，已回退）在真实模型上对比 CPU/GPU 的 AR logits：

```
prefill#0 (prefix, 多 token)   SAME   逐位一致
prefill#1 (第 1 个单 token)     DIFF
prefill#2+                      DIFF
```

**#0 完整一致，#1 起全错。** #0 与 #1 的唯一区别是 rows=N vs rows=1，以及 #1 依赖
#0 写进 KV cache 的内容。由于 #0 逐位一致，权重上传、host 写、量化、28 层前向本身
都是对的 —— 错的只发生在**单 token decode** 这条路径上（KV carryover 或 rows=1 的
某个 recorder 分支）。

已排除：

- **host-write 缺失**：forward_chunk 里 `write_f32` 后没有 host→compute 屏障。
  补上 `host_write_barrier` 后仍然 #1 分歧（逐位一样），所以不是它。
- **AR 权重本身**：#0 逐位一致说明上传/量化/前向都对。
- **token 流不同**：同 seed 同 logits(#0) 下采样出同样的 token，#1 的输入相同。

结论：这是 AR **decode（rows=1）** 路径的独立 bug，与 k-quant shader 的 1/64 lane
问题**无关**（那条是纯性能）。修好 decode 正确性之后，k-quant 的 5× 才敢合。

下一步应直接对拍 rows=1 的单步：用一个测试在 GPU 上只跑一次 rows=1 的
record_attention_rows / record_kv_write_rows / record_qk_norm_rope_rows，与 CPU 的
单步 forward 逐层比对，定位第一个分歧的算子。现有 tiny_for_test 走不了 GPU
（没有 tensor source），所以需要一个真实 GGUF 的对拍 harness。

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
