# YuE2 用法

本仓库对 `m-a-p/YuE2-3B` + `m-a-p/YuE2-VAE` 的端到端命令行示例。

> 通用前置：`cargo build --profile release-fast --bin rust-model-inference`。
> 推理路径在 `src/models/yue2/`，AR（自回归）与 NAR（非自回归扩散）双流共用一套
> 28 层权重。

## 1. 权重来源

| 用途 | ModelScope | 说明 |
| --- | --- | --- |
| 主模型 | `m-a-p/YuE2-3B` | `YuE2ForCausalLM`，28 层，hidden 2048，vocab 184704 |
| VAE | `m-a-p/YuE2-VAE` | `YuE2VAE`，48 kHz 立体声，`downsampling_ratio=1920` |

**不要用 `m-a-p/YuE-s1-7B-anneal-en-cot`。** 那是 7B 的 Llama 架构检查点，与本仓库
`YuE2ForCausalLM` 的 AR+NAR 双流实现不兼容，metadata 校验会直接失败。

## 2. 转换

VAE 固定为 F32，只有主模型可以量化：

```bash
./.venv/bin/python -m tools.converter.yue2.convert_yue2 \
  --model-dir models/YuE2-3B \
  --vae-dir  models/YuE2-VAE \
  --main-out models/YuE2-gguf/yue2-q8_0.gguf \
  --vae-out  models/YuE2-gguf/yue2_vae.gguf \
  --quant q8_0
```

`--quant` 取值：

| 值 | 结果 | 备注 |
| --- | --- | --- |
| `bf16` | 7.26 GB，628 个 BF16 张量 | 默认，零拷贝直传，不做任何算术 |
| `f32` | 12.91 GB | 便于排查量化误差 |
| `q8_0` | 4.61 GB，394 Q8_0 + 234 BF16，约 25 s | **CPU 推理推荐**，ABC 采样 11 → 31.5 tok/s |
| `q4_0` | 3.20 GB，约 50 s | |
| `q4_k_m` | ~3.2 GB | 逻辑正确但导出极慢，见下 |
| `q6_k` | 3.93 GB | 同上 |

只有 338 个 transformer 投影 + 2 个 `time_embedder` 矩阵参与量化。1-D norm、
两个 184704 宽的词表矩阵（`embed_tokens` / `lm_head`）以及 position / bridge 查表
保持 BF16 —— loader 通过 `load_f32_tensor` 读取它们，且这些张量对量化误差最敏感。

> **K-quant 目前不可用。** `quantize_q4_k` 量化单个 6144×2048 矩阵需要约 97 s，
> 完整导出会耗时数小时。类型与解码链路都已打通并可往返校验，但 block search
> 需要批量化之后才能实际使用。

## 3. 推理

**风格描述走 `--prompt`，不是 `--style`**（没有 `--style` 这个参数）。写错会被
静默忽略，然后报一个完全不相干的冲突错误：

```
$ ... --style "City Pop" --lyrics "..."
YuE2 --yue2 cannot be used with --audio
```

真正的缺失参数是 `--prompt`。

```bash
./target/release-fast/rust-model-inference \
  --yue2 \
  --model models/YuE2-gguf/yue2-q8_0.gguf \
  --vae   models/YuE2-gguf/yue2_vae.gguf \
  --prompt "City Pop, upbeat, danceable, groovy bass" \
  --lyrics  "[Verse]
路灯眨着眼睛 偷看谁的身影

[Chorus]
今晚不眠 快乐无限" \
  --out out.wav \
  --seed 12300 --steps 32 --max-tokens 600
```

### 3.1 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--prompt` | 必填 | 风格 / 编曲描述 |
| `--lyrics` | 必填 | 带 `[Verse]` / `[Chorus]` 等 section 标记的歌词 |
| `--out` | 必填 | 输出路径，**必须以 `.wav` 结尾**，否则报错 |
| `--steps` | 32 | NAR 扩散步数。1～2 仅用于测速，音质明显劣化 |
| `--max-tokens` | 见下 | 限制 **semantic** 阶段上限；ABC 阶段仍用 metadata 里的 4096 |
| `--seed` | 831001 | |
| `--temperature` / `--top-k` / `--top-p` | 见 metadata | 同时覆盖 semantic 阶段 |

`--max-tokens` 存在一个坑：它只改 semantic 的 `max_tokens`，ABC 阶段的
`yue2.abc.max_tokens`（4096）不受影响。快速冒烟测试时 ABC 仍会跑满上千 token。

### 3.2 四个阶段与耗时

CPU（aarch64 20 核，Q8_0）实测：

| 阶段 | `--steps 1` | `--steps 32` |
| --- | --- | --- |
| ABC 采样 | 75 s | 75 s |
| semantic 采样 | 103 s | 103 s |
| prefix_kv prefill | 13 s | 13 s |
| NAR 扩散 | 6 s | 537 s |
| VAE 解码 | 52 s | 52 s |
| **合计** | **249 s** | **780 s** |

NAR 是绝对瓶颈：每步对全部 latent 位置做两次全模型前向。VAE 解码与 NAR 步数
无关，是固定开销。

`--threads` 不要超过 8。`ComputePool` 是纯自旋屏障线程池，线程数超过 8 后同一
cache line 的乒乓开销会压倒收益：实测 `--threads 20` 比默认慢 **16 倍**
（1.45 tok/s vs 23.96 tok/s）。默认值 `DEFAULT_THREAD_CAP = 8` 对本机是最优的。

## 4. 样例音频

本机在 `models/YuE2-gguf/samples/` 下留了真权重跑出的成品音频与说明
（`README.md` 内含各文件的权重精度 / 步数 / 耗时对照与复现命令）。

注意 `/models/` 在 `.gitignore` 中，这批样例只存在于本地，不随仓库分发。
