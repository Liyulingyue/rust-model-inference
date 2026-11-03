# AuK (Base / Flash) 数值对齐记录

参考实现：**官方 `Tencent-Hunyuan/AuK` PyTorch 源码**
（`src/auk/model/flux2_edit.py`、`modules.py`、`cfm_edit.py`、`vae/bigvgan_flow_vae.py`），
不是第三方 audio.cpp 移植。数值对齐目标是逐个中间张量比较 Rust 与 PyTorch。

## 权重

| 组件 | 文件 | 说明 |
| --- | --- | --- |
| DiT（Base） | `models/AuK-Base-and-Flash-GGUF/auk-base-f16.gguf` | 420 张量，F16 |
| DiT（Flash） | `models/AuK-Base-and-Flash-GGUF/auk-flash-f16.gguf` | 与 Base 同结构，蒸馏权重 |
| VAE | `models/AuK-Base-and-Flash-GGUF/auk-vae-f32.gguf` | F32，BigVGANFlow |
| 文本编码器 | `models/AuK-Base-and-Flash-GGUF/qwen2.5-omni-3b-q8_0.gguf` | Qwen2.5-Omni-3B |
| 官方 safetensors | `models/AuK-Base-and-Flash-GGUF/auk_base.safetensors`、`vae.safetensors` | PyTorch oracle 用 |

GGUF 原始张量字节 == safetensors C-order 字节；GGUF `dims` 是反转的（GGML `ne`）顺序。

## 依赖

Python 依赖只装在仓库内 `.venv`（需要代理时自行加 `--proxy`）：

```bash
.venv/bin/pip install x_transformers torchdiffeq
```

官方源码放到仓库的 `target/auk-reference/AuK-main`（`src/` 布局），所有中间产物也落在
`target/auk-reference/` 下，不污染仓库其他目录。

## 插桩

Rust 侧用环境变量导出中间张量（`f32` little-endian）：

| 变量 | 导出内容 |
| --- | --- |
| `RMI_AUK_VAE_DUMP=<dir>` | VAE：`denorm` / `conv_pre` / `stage{i}_up` / `stage{i}_rb{j}` / `stage{i}` / `act_post` / `conv_post` / `out` |
| `RMI_AUK_DIT_DUMP=<dir>` | DiT：`text_conditioning.f32`、`latent_in_{step}.f32`、`audio_embed_{step}.f32`、`text_embed_{step}.f32`、`velocity_cond_{step}.f32`、`velocity_{step}.f32` |
| `RMI_AUK_DEBUG=1` | 打印每步 `t` / velocity rms / latent rms |

VAE 有 ignored 单测 `models::diffusion::auk::vae::tests::decodes_fixed_latent_for_reference_diff`，
用固定 latent 跑一遍并 dump（固定 latent 由 `auk_vae_ref.py` 确定性生成）：

```bash
RMI_AUK_VAE=models/AuK-Base-and-Flash-GGUF/auk-vae-f32.gguf \
RMI_AUK_VAE_LATENT=target/auk-reference/dumps/auk_latent.f32 \
RMI_AUK_VAE_DUMP=target/auk-reference/rust-vae \
cargo test --profile release-fast --lib auk::vae -- --ignored --nocapture
```

## 上游 PyTorch 的长度限制（oneDNN bug）

官方 VAE 的 alias-free 重采样调用

```python
F.conv1d(x, self.filter.expand(C, -1, -1), stride=2, groups=C)
```

本机的 PyTorch 2.15.0+cu132（aarch64）在输出超过约 100 万个元素时，
oneDNN 的分组卷积路径会让 OpenMP 分块越界，直接抛出
`RuntimeError: illegal immediate parameter (range error)`，
**任何长于 ~1.5 秒的音频都无法解码**。

这是 PyTorch/oneDNN 自身的问题，与 AuK 无关：

- 把权重 `.contiguous()` 后同样失败（`probe_torch_conv_bug.py`）；
- 关掉 oneDNN 后任意长度都正常（`probe_vae_length.py`）；
- 本仓库的 Rust 端口直接实现了同样的卷积数学，**没有这个长度上限**。

因此所有 oracle 脚本都设置了 `torch.backends.mkldnn.enabled = False`。
不这么做就只能对齐 1 秒以内的音频。

## 对齐结果

### VAE — 与官方 PyTorch 逐阶段一致

| 检查点 | maxdiff | cos |
| --- | --- | --- |
| `denorm` | 0 | 1.00000000 |
| `conv_pre` | 5.2e-06 | 1.00000000 |
| `stage0_up`（ConvTranspose1d） | 1.7e-05 | 1.00000000 |
| `stage0_rb0`（单个 resblock） | 1.8e-05 | 0.99999988 |
| `stage0` … `stage5` | ≤1.6e-04 | ≥0.99999988 |
| `act_post` | 2.6e-04 | 1.00000000 |
| `conv_post` | 4.7e-05 | 1.00000000 |
| `out`（24000 采样） | 4.7e-05 | 0.99999988 |

### DiT — 与官方 PyTorch 一致

text embed cos = 0.99999995；每个 timestep 的 velocity cos = 0.9999993 … 0.9999999
（maxdiff ≈ 3e-3，即 F16 权重量化噪声）。CPU 与 Vulkan 两条路径同精度：
Vulkan 的 velocity cos = 0.9999945 … 0.9999999。

### 端到端（4 秒 · 4 步 · cfg 2.0、官方 sway 网格）

| 路径 | 官方 rms / peak / 削波 | Rust rms / peak / 削波 | 波形 cos |
| --- | --- | --- | --- |
| CPU | 0.1477 / 0.9730 / 0% | 0.1478 / 0.9732 / 0% | 0.99976098 |
| Vulkan | 0.1477 / 0.9730 / 0% | 0.1479 / 0.9736 / 0% | 0.99977174 |

32 步时两侧都触及 VAE 末尾 `clamp(x, -1, 1)` 的上限 —— cfg 2.0 的强引导
`v_cond + 2·(v_cond - v_uncond)` 等效标准 CFG scale 3.0，会把 latent 推出
可表示范围；Rust 精确复现了官方这一行为。

两条路径的差异来自 F16 GEMM 的累加顺序（GPU 与 CPU 的 NEON 归约不同），
量级 1e-6 相对误差，在 cfg 2.0 的 32 步里被流匹配放大。

## 修复的官方语义差异

对齐过程修掉的都是「Rust 与官方 PyTorch 语义不一致」，记录在此便于回归。

### DiT

1. **RoPE 配对方式**：官方用 `x_transformers.RotaryEmbedding` +
   `apply_rotary_pos_emb`，freqs 是 `stack((freqs, freqs), dim=-1)` 后展平，
   即 **相邻配对 `(2i, 2i+1)`**（GPT-J 风格）。Rust 原来用 NeoX 半拆分
   `(i, i+32)`，旋转完全不同 → attention 输出是噪声。**这是"生成噪声而非语音"的主因。**
2. **timestep embedding**：官方 `SinusPositionEmbedding.forward(x, scale=1000)`
   产出 `[sin(0..127), cos(0..127)]`（分块，不是交错），`scale=1000` 在内部施加。
   Rust 原来写成交错的 `cos` 在前，且调用处预乘 1000 → 双重缩放。
3. **Euler 步进方向**：官方 `odeint(fn, y0, linspace(0, 1, steps+1))`，
   `y0 = randn`（纯噪声），流定义为 `φ = (1-t)x0 + t·x1` → `t` 是 **0 → 1**，
   步长 `dt = t_next - t`。Rust 原来用 `sigma = 1 - step/steps`（1 → 0）。
4. **AdaLN 输入**：官方 `emb = linear(silu(emb))`。Rust 原来喂的是 raw time embedding。
5. **最终 norm**：`AdaLayerNorm_Final` 用 `LayerNorm(dim, elementwise_affine=False, eps=1e-6)`，
   Rust 原来漏掉了 LayerNorm。
6. **文本流用独立的 `c_q_norm` / `c_k_norm`**：`JointAttnProcessor` 对文本流
   用 `attn.c_q_norm` / `attn.c_k_norm`，不是 `attn.q_norm` / `attn.k_norm`。
   Rust 原来两条流共用 x 流的 norm（GGUF 里确实存了 `c_q_norm` / `c_k_norm`）。
7. **RMSNorm eps**：`torch.nn.RMSNorm(dim_head)` 在 `eps=None` 时用
   `torch.finfo(float32).eps` = 1.1920929e-07，不是 1e-6。
8. **CFG 公式**：官方 `v_cond + (v_cond - v_uncond) * cfg_strength`
   （`v_cond` 的系数是 `1 + cfg`）。Rust 原来用 `uncond + cfg * (cond - uncond)`。
9. **CFG 阈值**：官方只在 `cfg_strength < 1e-5` 时跳过 CFG 分支。
   Rust 原来用 `guidance_scale > 1.0`，所以 `--cfg-scale 0.5` / `1.0` 会静默跳过 CFG。
10. **uncond 分支必须在 model 空间清零文本**：官方 `project_text(drop_text=True)`
    是 **先投影 + 归一化、再清零**。Rust 原来清零的是投影前的 2048 维 LLM hidden states，
    于是 `txt_proj(0) == txt_proj.bias` 残留，uncond 分支其实仍带文本信息。

### VAE

11. **缺 anti-aliased `Activation1d`**：官方 `AMPBlock1` 里每个 SnakeBeta 都包了
    `Activation1d`（`UpSample1d ×2` + 12 抽头 Kaiser FIR → SnakeBeta →
    `DownSample1d ×2` 低通）。checkpoint 里存了对应滤波器
    （`<prefix>.upsample.filter` / `<prefix>.downsample.lowpass.filter`），
    Rust 原来直接用裸 SnakeBeta → 高频噪声。
12. **resblock 内部是串行残差累加**：官方
    `for c1, c2, a1, a2 in zip(...): xt = c2(a2(c1(a1(x)))); x = xt + x`。
13. **每级 3 个 resblock 是并行分支**：官方 `xs = Σ resblocks[i*3+j](x)`，然后 `/ num_kernels`。
    Rust 原来串行作用在 `pre` 上。
14. **causal padding**：官方 `Conv1d(causal=True)` 只在左侧补
    `dilation * (kernel - 1)` 个 0。`conv_post` 是 `causal=True`（左补 6），
    `conv_pre` 是 `causal=False`（对称补 3）。
15. **ConvTranspose1d polyphase 差一位**：正确形式是
    `out[q*rate + p] = w[p] * in[q] + w[p+rate] * in[q-1]`，
    Rust 原来把第二项放到了 `q-1` 块。
16. **weight-norm 归一化轴**：PyTorch `weight_norm(dim=0)` 对 Conv1d
    （`[out, in, k]`）按输出通道归一化，对 ConvTranspose1d（`[in, out, k]`）
    按输入通道归一化。Rust 原来对两者都用 Conv1d 的约定。

## 采样配方

| 配方 | steps | cfg | 时间网格 |
| --- | --- | --- | --- |
| Base（官方默认） | 32 | 2.0 | sway `-1.0`：`t += c·(cos(π/2·t) - 1 + t)` |
| Flash（官方固定） | 4 | 0.0 | 显式 `[0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0]`（sway 关闭） |

Flash 的显式网格与 `sway=-1.0` 的 4 步网格在 f32 精度内一致
（差 < 1e-6，有单测 `flash_time_grid_matches_the_official_pinned_grid` 守住）。

CLI：`--auk-sway <coef>`（默认 `-1.0`）、`--auk-flash`（锁定 Flash 配方）。

## 性能

GB10（20 核）+ NVIDIA GB10，24 kHz，prompt「The next train leaves in ten
minutes.」，seed 42，`--threads 20`：

| 模型 | 路径 | 优化前 | 优化后 | 加速 |
| --- | --- | --- | --- | --- |
| Base（32 步 · 4 秒） | Vulkan | 685.9 s | **82.2 s** | 8.3x |
| Base（32 步 · 4 秒） | CPU | 416.1 s | **114.9 s** | 3.6x |
| Flash（4 步 · 4 秒） | Vulkan | 73.3 s | **11.0 s** | 6.7x |
| Flash（4 步 · 4 秒） | CPU | 53.7 s | **11.5 s** | 4.7x |

Base 32 步的分解（Vulkan）：denoise 77.1 s + VAE 4.4 s。VAE 解码从
32.1 s 降到 4.4 s（见 `auk/vae` 提交）。

主要改动（每一项都保持逐位一致，见上面的对齐结果）：

| 改动 | 效果 |
| --- | --- |
| 投影按整条 token 流批量下发（原来每个 token 一次提交） | 每次前向的设备提交 25411 → 206 |
| attention 按 (head, query) 拆成独立单元并行，K/V 转成 head-major 布局 | attention 从内存带宽瓶颈变成算力瓶颈 |
| 线程池空闲 worker 挂起而不是空转 | 之前 19 个空转 worker 会把 attention 线程挤掉（慢 4 倍） |
| RoPE 表每个前向只算一次 | 每前向 920 万次 libm `sin`/`cos` → 1.3 万次 |
| `conv_pos_embed` 四路累加器 + NEON | 211 ms → 55 ms 每前向 |
| VAE 卷积按时间轴累加 + ConvTranspose1d 多相分解 | 4 秒解码 32.1 s → 8.1 s |
| VAE 的抗混叠重采样并行化 + 卷积复用输入行 | 4 秒解码 8.1 s → 4.4 s |

`RMI_AUK_PROFILE=1` 打印 DiT 的分阶段耗时，
`RMI_AUK_VAE_PROFILE=1` 打印 VAE 的分阶段耗时。

## 复现

脚本都在本目录（`tools/oracle/auk/`），路径通过环境变量可改：

| 变量 | 默认 | 含义 |
| --- | --- | --- |
| `AUK_SRC` | `target/auk-reference/AuK-main/src` | 官方 `Tencent-Hunyuan/AuK` 检出（`src/` 布局） |
| `AUK_REF_DIR` | `target/auk-reference/dumps` | PyTorch 参考 dump 目录 |
| `AUK_RUST_DIR` | `target/auk-reference/rust-dit` | Rust `RMI_AUK_DIT_DUMP` 输出目录 |
| `AUK_VAE_RUST_DIR` | `target/auk-reference/rust-vae` | Rust `RMI_AUK_VAE_DUMP` 输出目录 |

### 1) DiT 逐检查点

```bash
# Rust dump（cfg 0 时只有条件分支，text_embed 才是有信息的）
rm -rf target/auk-reference/rust-dit && mkdir -p target/auk-reference/rust-dit
RMI_AUK_DIT_DUMP=target/auk-reference/rust-dit \
./target/release-fast/rust-model-inference \
  --model models/AuK-Base-and-Flash-GGUF/auk-base-f16.gguf \
  --vae models/AuK-Base-and-Flash-GGUF/auk-vae-f32.gguf \
  --text-encoder models/AuK-Base-and-Flash-GGUF/qwen2.5-omni-3b-q8_0.gguf \
  --prompt "Hello" --out out/auk_test.wav \
  --steps 4 --duration 1 --cfg-scale 0.0 --threads 20 --compute cpu

# PyTorch 参考 + 比较（顺序不能反：参考脚本要读 Rust 的 latent 输入）
.venv/bin/python3 tools/oracle/auk/auk_dit_ref.py
.venv/bin/python3 tools/oracle/auk/cmp_dit.py
```

`auk_dit_ref.py` 读 `text_conditioning.f32` 与 `latent_in_<step>.f32`，
按官方 sway 网格复算每个 timestep 的 velocity 并写 `dit_velocity_<step>.npy`；
`cmp_dit.py` 再逐检查点比较。

### 2) 端到端波形

```bash
.venv/bin/python3 tools/oracle/auk/auk_e2e_ref.py --steps 4 --cfg 0.0
```

用同一份 `text_conditioning.f32` + `latent_in_0.f32` 跑官方流匹配采样 +
官方 VAE 解码，产出 `e2e_latent_<step>.npy` 与 `e2e_audio.npy`，
再与 Rust 的 `out/auk_test.wav` 比较。

实测（同一 prompt + seed，24 kHz）：

| 配方 | 官方 rms / peak / 削波 | Rust rms / peak / 削波 | 波形 cos |
| --- | --- | --- | --- |
| 4 秒 · 4 步 · cfg 2.0 | 0.1477 / 0.9730 / 0% | 0.1478 / 0.9732 / 0% | 0.99976 |
| 4 秒 · 32 步 · cfg 2.0 | 0.3136 / 1.0000 / 0.43% | 0.3138 / 1.0000 / 0.44% | 0.99999523 |

32 步时两侧都触及 VAE 末尾 `clamp(x, -1, 1)` 的上限 —— cfg 2.0 的强引导
`v_cond + 2·(v_cond - v_uncond)` 等效标准 CFG scale 3.0，会把 latent 推出
可表示范围；Rust 精确复现了官方这一行为。

### 3) VAE 逐阶段

固定 latent 由 `auk_vae_ref.py` 生成（`target/auk-reference/dumps/auk_latent.npy` /
`auk_latent.f32`）：

```bash
.venv/bin/python3 tools/oracle/auk/auk_vae_ref.py   # 生成 ref_*.npy 与固定 latent

RMI_AUK_VAE=models/AuK-Base-and-Flash-GGUF/auk-vae-f32.gguf \
RMI_AUK_VAE_LATENT=target/auk-reference/dumps/auk_latent.f32 \
RMI_AUK_VAE_DUMP=target/auk-reference/rust-vae \
cargo test --profile release-fast --lib auk::vae -- --ignored --nocapture

.venv/bin/python3 tools/oracle/auk/cmp_vae.py        # 比较
```

`auk_vae_stage.py` 用于把 VAE 拆到单个 stage / 单个 resblock 逐层定位分歧。

### 4) 上游 oneDNN 长度限制探针

```bash
.venv/bin/python3 tools/oracle/auk/probe_vae_length.py      # 官方 VAE 各长度解码
.venv/bin/python3 tools/oracle/auk/probe_torch_conv_bug.py  # 最小复现
```
