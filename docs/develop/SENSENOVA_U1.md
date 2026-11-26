# SenseNova-U1.6-8B-MoT — 适配契约

本文件固定 `SenseNova/SenseNova-U1.6-8B-MoT-GGUF` 的数值与结构契约，供 Rust 实现与
Oracle 对拍使用。改动此文件前先改 Oracle 并重新生成 `tools/oracle/sensenova_u1/reference/`。

## 1. 来源与固定输入

| 项 | 值 |
| --- | --- |
| ModelScope repo | `SenseNova/SenseNova-U1.6-8B-MoT-GGUF` |
| 基座 | `sensenova/SenseNova-U1.6-8B-MoT` |
| `source_config_sha256` | `b5c28b643a4cfc771d1303acd12c94126b7a725a6c9796b3d5ecca07e9a582ef` |
| 量化文件 | `Q4_0` 12.28 GB / `Q6_K` 16.62 GB / `Q8_0` 20.70 GB |
| GGUF sha256 (Q8_0) | `097ca40a942211c89fe0e945add3d159839e904a16feb5f9d0d14ba8e6644806` |
| GGUF sha256 (Q6_K) | `e538414f42201294471a5fb8ec58e54af57e260cac297daa529829be294d1b9b` |
| GGUF sha256 (Q4_0) | `135ccdde87ebf92f76a6f7a17d604f91df6ba15718f1dd847e0dbb766e8ad28b` |
| `general.architecture` | `sensenova_u1.6` |
| Oracle | `https://github.com/OpenSenseNova/SenseNova-U1` @ `d2d9bd1` |

GGUF 头部只有 4 个 metadata key（`general.architecture` / `general.file_type` /
`general.name` / `general.quantization_version`），**架构参数全部来自随仓 `config.json`**。
实现必须校验 `source_config_sha256`，避免拿错配置跑。

## 2. 模型结构

统一「理解 + 生成」模型（MoT = Mixture of Transformers）。一个 GGUF 内含全部组件，
**没有 VAE**：图像直接在像素空间用 flow matching 生成。

```
NEOChatModel
├── language_model  (Qwen3 8B, 42 层, 双分支)
├── vision_model                (理解分支：仅 embeddings)
├── fm_modules
│   ├── vision_model_mot_gen    (生成分支：仅 embeddings，与理解分支同构不同权重)
│   ├── timestep_embedder       (256 → 4096 → SiLU → 4096)
│   ├── noise_scale_embedder    (同上)
│   └── fm_head = ConvDecoder   (4096 → 1024 → 192 → 3，pixel shuffle)
```

### 2.1 关键超参（来自 config.json）

| 参数 | 值 |
| --- | --- |
| `llm.hidden_size` | 4096 |
| `llm.num_hidden_layers` | 42 |
| `llm.num_attention_heads` | 32 |
| `llm.num_key_value_heads` | 8 |
| `llm.head_dim` | 128 |
| `llm.intermediate_size` | 12288 |
| `llm.vocab_size` | 151936 |
| `llm.rms_norm_eps` | 1e-6 |
| `llm.rope_theta` (t 轴) | 5000000.0 |
| `llm.rope_theta_hw` (h/w 轴) | 10000.0 |
| `patch_size` | 16 |
| `downsample_ratio` | 0.5 → `merge_size` = 2 |
| `vision.hidden_size` | 1024 |
| `vision.llm_hidden_size` | 4096 |
| `vision.rope_theta_vision` | 10000.0 |
| `fm_head_dim` / `fm_head_layers` | 1536 / 2（**未使用**：`use_pixel_head=true`） |
| `use_pixel_head` | true → `fm_head` 是 `ConvDecoder` |
| `noise_scale` / `noise_scale_max_value` | 1.0 / 16.0 |
| `noise_scale_mode` | `resolution` |
| `noise_scale_base_image_seq_len` | 64 |
| `base_shift` / `max_shift` | 0.5 / 1.15 |
| `base_image_seq_len` / `max_image_seq_len` | 64 / 4096 |
| `time_schedule` | `standard` |
| `t_eps` | 0.05（`t2i_generate` 默认参数覆盖为 0.02，CLI 不传则用 0.02） |
| `template` | `neo1_0` |
| `min_pixels` / `max_pixels` | 65536 / 16777216 |

### 2.2 张量清单（1116 个，Q8_0）

| 组 | 数量 | 说明 |
| --- | --- | --- |
| `language_model.model.layers.{0..41}.*` | 42 × 26 = 1092 | 双分支 decoder |
| `language_model.lm_head.weight` | 1 | [4096, 151936] Q8_0 |
| `language_model.model.embed_tokens.weight` | 1 | [4096, 151936] **F32** |
| `language_model.model.norm.weight` | 1 | [4096] |
| `language_model.model.norm_mot_gen.weight` | 1 | [4096] |
| `vision_model.embeddings.*` | 4 | patch_embedding / dense_embedding |
| `fm_modules.vision_model_mot_gen.embeddings.*` | 4 | 同上 |
| `fm_modules.timestep_embedder.mlp.{0,2}.*` | 4 | |
| `fm_modules.noise_scale_embedder.mlp.{0,2}.*` | 4 | |
| `fm_modules.fm_head.conv{1,2}.*` | 4 | |

每层 26 个张量：

```
input_layernorm.weight                       [4096]  F32
input_layernorm_mot_gen.weight               [4096]  F32
post_attention_layernorm.weight              [4096]  F32
post_attention_layernorm_mot_gen.weight      [4096]  F32
mlp.{gate,up,down}_proj.weight                        Q8_0
mlp_mot_gen.{gate,up,down}_proj.weight                Q8_0
self_attn.q_proj.weight         [4096, 4096]  Q8_0
self_attn.k_proj.weight         [4096, 1024]  Q8_0
self_attn.v_proj.weight         [4096, 1024]  Q8_0
self_attn.o_proj.weight         [4096, 4096]  Q8_0
self_attn.q_proj_mot_gen.weight [4096, 4096]  Q8_0
self_attn.k_proj_mot_gen.weight [4096, 1024]  Q8_0
self_attn.v_proj_mot_gen.weight [4096, 1024]  Q8_0
self_attn.o_proj_mot_gen.weight [4096, 4096]  Q8_0
self_attn.{q,k}_norm{,_mot_gen}.weight        [64]  F32
self_attn.{q,k}_norm_hw{,_mot_gen}.weight     [64]  F32
```

无 attention bias，无 `q_proj.bias`。

## 3. 双分支 Attention + M-RoPE（对齐核心）

`head_dim = 128`，每层有 **两套** q/k/v/o 与两套 norm/MLP：

- **und 分支**（理解 / 文本前缀）：`q_proj/k_proj/v_proj/o_proj`、`q_norm/k_norm/q_norm_hw/k_norm_hw`、`input_layernorm`、`post_attention_layernorm`、`mlp`
- **gen 分支**（图像 token）：`*_mot_gen` 全套

单层前向（und 路径）：

```
h = input_layernorm(x)                      # RMSNorm(4096, eps=1e-6)
q = q_proj(h)  -> view [S, 32, 128]
  q_t, q_hw = q.chunk(2, -1)                # 64 + 64
  q_t  = q_norm(q_t)                        # RMSNorm(64)  — 注意是 64 维，不是 128
  q_hw = q_norm_hw(q_hw)                    # RMSNorm(64)
  q_h, q_w = q_hw.chunk(2, -1)              # 32 + 32
k/v 同构（k 有 k_norm/k_norm_hw；v 无 norm）
q_t,k_t = rope(q_t,k_t, pos=t, head_dim=64, theta=5e6)
q_h,k_h = rope(q_h,k_h, pos=h, head_dim=32, theta=1e4)
q_w,k_w = rope(q_w,k_w, pos=w, head_dim=32, theta=1e4)
q = cat([q_t, q_h, q_w], -1)                # 128
k = cat([k_t, k_h, k_w], -1)
attn = sdpa(q, k, v)                        # 32 q heads / 8 kv heads, scale=1/sqrt(128)
x = x + o_proj(attn)
h = post_attention_layernorm(x)
x = x + mlp(h)                              # SwiGLU: down(silu(gate) * up)
```

### 3.1 RoPE 精确契约

上游用 `_rope_init_fn_keep_freq_range`：先按 `head_dim*2` 算 inv_freq 再取 `[::2]`。
推导结果等价于**标准 NeoX RoPE 作用在子段上**：

- t 段：`inv_freq[i] = theta^(-i/32)`, i∈[0,32) — 等价 `rope_neox(head_dim=64, theta=5e6)`
- h/w 段：`inv_freq[i] = theta^(-i/16)`, i∈[0,16) — 等价 `rope_neox(head_dim=32, theta=1e4)`

旋转是 **half-split（rotate_half）**：`(x[i], x[i+half])` 成对，`emb = cat(freqs, freqs)`。
实现可直接用 `crate::ops::rope` 的标准入口，对三段分别以各自的 `pos/head_dim/theta` 施加；
需要一条 `rope_neox_segments` 支持「每头不等宽三段 + 每段独立位置」。

### 3.2 生成阶段（denoise）的注意力形状

denoise 每步：当前图像 token 走 **gen 分支**（`*_mot_gen` 投影 + `norm_mot_gen`），
但 KV cache 的 **前缀部分** 是 und 分支写入的 key/value。即：

- `Q` 来自 `q_proj_mot_gen`，`K_cur/V_cur` 来自 `k_proj_mot_gen`/`v_proj_mot_gen`
- attention 对 `[prefix_K (und), K_cur]` 做，**当前图像块内部 bidirectional**（`causal=False`）
- `o_proj_mot_gen` 输出

前缀 forward 的 mask：`create_block_causal_mask(t_index)`，文本段 `t = 0..L-1` 退化为普通因果。

### 3.3 M-RoPE 位置（indexes）

`indexes` 是 `[3, L]` 的 `(t, h, w)`：

- 文本前缀：`t = 0..L-1`, `h = w = 0`
- 图像 token：`t = text_len`（常数），`h = idx // token_w`, `w = idx % token_w`

## 4. 视觉 embeddings（无 Transformer block）

`NEOVisionEmbeddings` 只有三层：

```
pixels [N, 3, 16, 16]
  → conv2d(3→1024, k=16, s=16)      # patch_embedding
  → GELU
  → 2D RoPE：把 1024 维切成两半，前 512 用 x 坐标、后 512 用 y 坐标
     每半是 **interleaved（交错对）** 旋转，theta = 10000：
       x1 = x[..., 0::2]; x2 = x[..., 1::2]
       out[..., 0::2] = x1*cos - x2*sin;  out[..., 1::2] = x1*sin + x2*cos
     ⚠️ 注意与 LLM 主干不同：主干是 half-split（`rotate_half`），视觉塔是
     interleaved（`modeling_neo_vit.py::apply_rotary_emb_1d`，等价 ggml 的
     "norm" 配对）。频率两者相同（`inv_freq[i] = 10000^(-i/256)`），只有配对
     方式不同，从 position 1 起就会分叉。
     注意：先 to(f32) 做 rope，再 cast 回权重 dtype
  → reshape 到 [1, h, w, 1024]，permute 成 [1, 1024, h, w]
  → conv2d(1024→4096, k=2, s=2)     # dense_embedding
  → [1, 4096, h/2, w/2] → permute → [h*w/4, 4096]
```

2D RoPE 的坐标来自 `build_abs_positions_from_grid_hw`：`abs_x = i % W`, `abs_y = i // W`，
其中 `W` 是该图的 patch 宽度（`grid_w = image_w / 16`）。

## 5. 采样（T2I，50 步默认）

```
merge_size = 2, patch_size = 16
token_h = H / 32, token_w = W / 32        # 图像 token 网格
grid_h  = H / 16, grid_w  = W / 16        # patch 网格

# 1. 前缀（两分支：cond / uncond）
cond   = template(system=SYSTEM_MESSAGE_FOR_GEN, user=prompt) + "<think>\n\n</think>\n\n<img>"
uncond = template(system="", user="")                        + "<img>"
  渲染规则（neo1_0 / MPT，注意 `get_prompt()` 在 system_message 为空时整段省略 system）：
    cond   = "<|im_start|>system\n" + SYSTEM_MESSAGE_FOR_GEN + "<|im_end|>\n"
           + "<|im_start|>user\n" + prompt + "<|im_end|>\n"
           + "<|im_start|>assistant\n"
    uncond = "<|im_start|>user\n<|im_end|>\n"      # 空 user 仍输出 role + sep
           + "<|im_start|>assistant\n"
  两者之后分别追加 `<think>\n\n</think>\n\n<img>` / `<img>`。
prefix KV = LLM(cond/uncond, indexes=text, block-causal mask)     # und 分支

# 2. 初始噪声
noise_scale = sqrt((grid_h*grid_w)/(merge_size^2) / 64) * 1.0
noise_scale = min(noise_scale, 16.0)
x = noise_scale * randn(1, 3, H, W, seed)     # torch.Generator(device).manual_seed(seed)

# 3. 时间表
t = linspace(0, 1, num_steps+1)
sigma = 1 - t
sigma = shift*sigma / (1 + (shift-1)*sigma)   # shift = timestep_shift (默认 3.0)
t = 1 - sigma

# 4. 每步
for i in 0..num_steps:
    z = patchify(x, 32)                       # [1, L, 3072], channel-last
    img_in = patchify(x, 16, channel_first=True)  # [1, grid_h*grid_w, 768]
    emb = vision_mot_gen(img_in, grid_hw)         # [1, L, 4096]
    t_emb = timestep_embedder(t_i)                # [1, L, 4096]
    n_emb = noise_scale_embedder(noise_scale / 16.0)
    emb = emb + t_emb + n_emb
    v_c = predict_v(emb, cond_kv,  t_i, z)        # gen 分支
    v_u = predict_v(emb, uncond_kv, t_i, z)
    v = v_u + cfg_scale * (v_c - v_u)             # cfg_scale 默认 4.0
    z = z + (t_{i+1} - t_i) * v
    x = unpatchify(z, 32, H, W)
```

`predict_v`：

```
hidden = LLM_gen(emb, cond_kv, indexes=image, causal=false)   # gen 分支, 不写 cache
h = hidden[:, -L:]                                            # [1, L, 4096]
img_2d = h.view(1, token_h, token_w, 4096).permute(0,3,1,2)   # [1, 4096, th, tw]
out = ConvDecoder(img_2d)                                     # [1, 3, H, W]
x_pred = out.permute(0,2,3,1).view(1, L, 3072)
v = (x_pred - z) / max(1 - t_i, t_eps)
```

### 5.1 ConvDecoder

```
ps1 = PixelShuffle(2)      [1,4096,th,tw] → [1,1024,2th,2tw]
conv1 = Conv2d(1024→1024, k=3, pad=1)          weight [3,3,1024,1024]
act1 = GELU
ps2 = PixelShuffle(2)      → [1,256,4th,4tw]
conv2 = Conv2d(256→192, k=3, pad=1)            weight [3,3,256,192]
ps3 = PixelShuffle(8)      → [1,3,H,W]
```

注意 GGUF 里 conv 权重的**头形状**是 PyTorch 的反序 `[kh, kw, in, out]`，但
**payload 字节仍是 PyTorch 连续的 `[out][in][kh][kw]`**（`gguf_loader.py` 把
`GGUFReader` 的 `tensor.data`（shape `dims[::-1]`）直接喂给 `nn.Conv2d`）。
所以读入时**只重排声明的形状，不要重排数据**——物理置换元素会把卷积核打乱。
`crate::ops::nn::conv2d` 本来就按 `[out][in][kh][kw]` 索引，直接传原始 buffer 即可。

GELU 用 `nn.GELU()` 默认 `approximate='none'` = 精确 erf，对应
`crate::ops::activation::gelu_erf_inplace`，**不要**用 tanh 近似版本。

### 5.2 TimestepEmbedder

```
freqs = exp(-ln(10000) * arange(0,128,f32) / 128)          # 256/2 = 128
args = t[:, None] * freqs[None]
emb = cat([cos(args), sin(args)], -1)                       # 256
out = mlp[2](silu(mlp[0](emb)))                             # 256 → 4096 → 4096
```

## 6. Tokenizer

Qwen2 BPE：`vocab.json` + `merges.txt`，`tokenizer_config.json` 给出 pre-tokenizer 规则。
特殊 token：`<|im_start|>`、`<|im_end|>`、`<img>`、`<IMG_CONTEXT>`、`<|endoftext|>`。
`pad_token_id = 151643`，`eos_token_id = 151645`，`img_start_token_id = 151670`。

**必须与 Oracle 逐 token 一致**，由 `tools/oracle/sensenova_u1/reference/prompt_tokens.json` 固定。

## 7. 覆盖矩阵（交付时逐行填证）

见 `tools/oracle/sensenova_u1/validation.md`。维度：Q4_0 / Q6_K / Q8_0 × CPU / Auto / Vulkan ×
CLI / HTTP，外加性能对标。
