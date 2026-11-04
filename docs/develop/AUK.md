# AuK（Base / Flash）状态与实现记录

> 本页只汇总状态、限制和实现沿革。逐检查点的数值证据、性能表和复现命令在
> [oracle 记录](../../tools/oracle/auk/README.md)；模型清单见 [MODEL_LIST.md](../MODEL_LIST.md)。
>
> 更新于 2026-11-04。

## 入口

只有 CLI，未接入 HTTP / JEV。

```bash
rust-model-inference \
  --model <auk-diT-f16.gguf> --vae <auk-vae-f32.gguf> \
  --text-encoder <qwen2.5-omni-3b-q8_0.gguf> \
  --prompt "The next train leaves in ten minutes." --out out.wav
```

- `--auk-sway <coef>`（默认 `-1.0`）：官方 Base 的 sway 采样网格。
- `--auk-flash`：锁定 Flash 配方（4 步、cfg 0、显式时间网格）。
- 这两个参数是 AuK 专属，用在非 AuK 模型上会明确报错退出（exit 2）；`--cfg-scale`
  为负数或 NaN 同样报错退出（exit 1）。

## 组件与权重

scaffold + dispatch + 6/6 contract tests + 文本编码器 + Flux2Edit blocks +
BigVGANFlow VAE decoder（18 个 resblock）+ Euler flow + CFG + WAV writer。

| 组件 | 权重 | 格式 |
| --- | --- | --- |
| DiT | `auk-base-f16.gguf` / `auk-flash-f16.gguf` | F16 |
| VAE | BigVGANFlow checkpoint | F32 |
| 文本编码器 | `qwen2.5-omni-3b-q8_0.gguf`（4.25 GB） | Q8_0 / BF16 |

AuK-Flash 是 4 步蒸馏版（guidance 0、不跑 CFG unconditional 分支），与 Base
共用同一份架构和 scaffold，零代码改动。

运行路径：CPU 与 GPU（`--compute auto`）都能产出有效 24 kHz 单声道 WAV；文本编码器
在 GPU 上回退 CPU（`qwen2vl` 架构还没有 Vulkan 支持），DiT/VAE 投影走 Vulkan。

## 数值对齐

2026-11-03 以官方 `Tencent-Hunyuan/AuK` 的 PyTorch 源码为参考（不是 audio.cpp
移植），逐中间张量比较后修掉 16 处语义差异。

DiT（10 处）

1. **RoPE 配对方式**：官方是相邻配对 `(2i, 2i+1)`（GPT-J 风格），Rust 原来用
   NeoX 半拆分 —— 这是「生成噪声而非语音」的主因。
2. **timestep embedding**：分块布局 `[sin(0..127), cos(0..127)]`，`scale=1000` 在内部施加。
3. **Euler 步进方向**：`t` 是 0 → 1，Rust 原来是 1 → 0。
4. **AdaLN 输入**：`emb = linear(silu(emb))`，不是 raw time embedding。
5. **最终 norm**：`AdaLayerNorm_Final` 含 LayerNorm。
6. **文本流用独立的 `c_q_norm` / `c_k_norm`**，不是 x 流的 `q_norm` / `k_norm`。
7. **RMSNorm eps**：`torch.finfo(float32).eps` = 1.1920929e-07。
8. **CFG 公式**：`v_cond + cfg·(v_cond - v_uncond)`。
9. **CFG 阈值**：官方只在 `cfg < 1e-5` 时跳过 CFG 分支。
10. **uncond 分支在 model 空间清零文本**：先投影 + 归一化，再清零。

VAE（6 处）

11. **缺 anti-aliased `Activation1d`**：每个 SnakeBeta 外面包了 12 抽头 Kaiser FIR
    的上采样/下采样。
12. **resblock 内部是串行残差累加**。
13. **每级 3 个 resblock 是并行分支**，求和后再 `/3`。
14. **causal padding**：`conv_post` 只在左侧补 6，`conv_pre` 对称补 3。
15. **ConvTranspose1d polyphase 差一位**。
16. **weight-norm 归一化轴**：Conv1d 按输出通道、ConvTranspose1d 按输入通道。

结果：

| 检查 | 结果 |
| --- | --- |
| VAE 6 个 stage + 前后处理 | cos = 1.00000000（maxdiff ~1e-5，F32 噪声） |
| DiT text embed | cos = 0.99999995 |
| 每步 velocity | cos ≥ 0.9999993（maxdiff ≈ 3e-3，F16 权重噪声） |
| 4 秒波形 · 4 步 · cfg 2.0 | cos = 0.99976（CPU）/ 0.99977（Vulkan） |
| 4 秒波形 · 32 步 · cfg 2.0 | cos = 0.99999523 |

输出从噪声变成真实语音（zcr 0.82 → 0.046，频谱集中在 0–4 kHz）。逐检查点表格和
复现命令见 [oracle 记录](../../tools/oracle/auk/README.md#对齐结果)。

## Vulkan 上的 VAE 解码器

控制流仍在主机侧（阶段循环、resblock 循环、残差累加都留在 Rust 里），只有算子换成
设备实现（`src/models/diffusion/auk/vae_gpu.rs`）：稠密 Conv1d、ConvTranspose1d 多相、
抗混叠 FIR 重采样、SnakeBeta、残差求和与缩放。激活缓冲留在显存，resblock 不回传 host，
只有权重每次上传。

4 秒解码：CPU 4.6 s → Vulkan 0.40 s。比官方 PyTorch CUDA 慢 2.2 倍、比官方 PyTorch
CPU（oneDNN 关）快 26 倍 —— 官方 CPU 路径必须关掉 oneDNN，否则超过 1.5 秒的音频会
崩在上游的 `illegal immediate parameter` 上。

## audio.cpp 任务覆盖（16/16）

`AukAudio` 的 `pitch_shift` / `speed_change` / `volume_change` DSP 编辑、
`AukOptions.instruct` 字段和完整 CFMEdit attention 路径都有 Rust 实现：audio.cpp
任务表里的 12 个 reference-audio 任务（zero-shot TTS、speech editing、timbre、
emotion、de-accent、whisper、enhance、separate、target speaker 等）和 4 个
text-only / DSP 任务（instruct TTS、pitch、speed、volume）。

CFMEdit 参考音频 conditioning 走
`AukPipeline::generate_audio_with_audio(prompt, audio_embeddings, audio_tokens, options)`：
接受预计算的 audio embeddings（`audio_tokens * TEXT_IN`），与 text 拼成
image/audio/text joint sequence 一起走 DiT。

## 已知限制

- **未接入 HTTP / JEV**：只有 CLI 入口。
- **Qwen2.5-Omni audio tower 未移植**（`thinker.audio_tower.*`）：CFMEdit 端到端需要
  外部提供 audio embeddings（例如用 Python 跑 Qwen audio tower 导出）。
- **`pitch_shift` 同时改音高和时长**：走线性重采样，+2 半音 = 缩短到 1/1.122。
  「音高变、时长不变」需要 phase vocoder，是 follow-up。
- **与 audio.cpp 的 oracle diff 仍待校准**：现在的数值基准是官方 PyTorch。
- **官方 PyTorch 的 oneDNN 长度上限**：CPU 路径对超过 ~1.5 秒的音频会抛
  `illegal immediate parameter`，CPU 对照要走官方 CUDA 或本仓库 Vulkan 路径。

## 实现沿革

- 主路径从 F16→Q8_0 预量化切换为按 GGMLType dispatch
  （`linear_into_dispatched` → `linear_into_scaled_impl`）：F16 走 Vulkan F16 GPU
  matmul（`BatchedLinearRuntime::matmul_rows` 单行 + format=F16），Q8_0 走原 Q8 GPU
  matmul，BF16/Q*_K 走 `QTensorOwned` CPU。去掉 21 s 载入、1.5 GB 常驻内存和量化噪声，
  数值直接对齐 F16 GGUF。
- 修掉 `linear_into_dispatched` + `linear_into_dispatch` 两处被 Q8 cache 100% 命中
  掩盖的无限递归。
- DiT 与 VAE 的其余优化（批量投影、并行 attention、线程池 parking、RoPE 表缓存、
  VAE 卷积与重采样并行化）见 [oracle 记录的性能表](../../tools/oracle/auk/README.md#性能)。

## 复现

见 [oracle 记录](../../tools/oracle/auk/README.md#复现)。
