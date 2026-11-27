# MiniMind GGUF 转换

本目录包含两个转换器：

- `convert_minimind3.py`：官方 MiniMind-3 dense/MoE HF safetensors → `qwen3` /
  `qwen3moe` GGUF。
- `convert_minimind2moe.py`：MiniMind2-MoE `.pth` → `minimind2moe` GGUF。

## MiniMind-3 dense / MoE → qwen3 / qwen3moe

MiniMind-3 的 Hugging Face checkpoint 直接使用 `Qwen3ForCausalLM` /
`Qwen3MoeForCausalLM` 结构，因此分别映射到本仓库已有的 `qwen3` 和 `qwen3moe`
trunk。两个 checkpoint 都有 Q/K RMSNorm，且输入 embedding 与输出投影 tied。

### 输入

- dense：`gongjy/minimind-3` 的 `config.json`、`model.safetensors`、
  `tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`。
- MoE：`gongjy/minimind-3-moe` 同样六个文件。
- 已验证源文件：

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| dense `model.safetensors` | 127,834,168 | `3adf69402b5d22e693151cabadc12528f923c4ba6bf343738aaf13f0892162e8` |
| MoE `model.safetensors` | 396,852,784 | `4f7c497acc45d2bad911c3bdf04b3a8d269b010d6d5455dc7905aac1941ac044` |

共同维度：hidden 768、8 层、8 heads、4 KV heads、head_dim 96、vocab 6400、
context 32768、RoPE theta 1e6、RMS eps 1e-6、SiLU、无 attention bias、F16 权重。
MoE 额外为 4 routed experts、top-1 routing、expert FFN 2432，无 shared expert。
dense checkpoint 有 90 个张量；MoE 有 98 个。

### 转换

在仓库根目录运行：

```bash
.venv/bin/python -m tools.converter.minimind.convert_minimind3 \
  models/minimind-3-pytorch \
  --outfile models/MiniMind-3-F16.gguf \
  --quant f16

.venv/bin/python -m tools.converter.minimind.convert_minimind3 \
  models/minimind-3-moe-pytorch \
  --outfile models/MiniMind-3-MoE-F16.gguf \
  --quant f16
```

`--quant` 支持 `f16`（默认）、`f32`、`bf16`、`q8_0`、`q4_0`。embedding、
attention、dense FFN 和 expert 矩阵使用所选格式；norms、Q/K norms、router
始终写 F32。K-quant 被拒绝：2432 宽的矩阵行不能被 256 值 K-quant block 整除。

其他参数：`--name`、`--context-length`、`--rope-freq-base`、`--norm-eps`、
`--chat-template-file`。转换器不会覆盖已存在的输出文件。

### 张量映射与校验

- `model.embed_tokens.weight` → `token_embd.weight`。
- `input_layernorm` → `blk.N.attn_norm.weight`；`post_attention_layernorm` →
  `blk.N.ffn_norm.weight`；`q_norm` / `k_norm` →
  `blk.N.attn_q_norm.weight` / `blk.N.attn_k_norm.weight`（F32）。
- dense `mlp.{gate,up,down}_proj` → `blk.N.ffn_{gate,up,down}.weight`。
- MoE `mlp.gate.weight` → `blk.N.ffn_gate_inp.weight`（F32）；4 个
  `mlp.experts.E.*` 按 expert 顺序堆叠 → `blk.N.ffn_{gate,up,down}_exps.weight`。
- `model.norm.weight` → `output_norm.weight`；tied embedding 省略
  `output.weight`。若存在 `lm_head.weight`，必须与 embedding 完全相同。
- 转换器拒绝缺失/额外张量、错误 config、错误 attention/FFN/expert 形状、
  未 tie 的 `lm_head`、shared-expert 配置、非 `full_attention` 层和量化 block
  不对齐的矩阵。
- Tokenizer 写为 GPT-2 BPE、`tokenizer.ggml.pre=qwen2`、6400 tokens；
  BOS=1、EOS=2、UNK=0、PAD=0、`add_bos_token=false`、`add_eos_token=false`。
- 官方 tokenizer 把 ids 0-20 / 27-35 标为 control，21-26 标为 user-defined；
  转换器按同样的 `<|...|>` 规则生成 token type。
- metadata 与官方 F16 GGUF 对齐（`general.name` 为 `Minimind 3` /
  `Minimind 3 Moe`，dense 为 `qwen3`、MoE 为 `qwen3moe`）。官方 MoE GGUF
  缺少 `general.basename` / `general.version`，本转换器会写入；这是唯一
  metadata 键差异，不影响加载或推理。

### 验证

```bash
python3 -m unittest tools.converter.minimind.test_convert_minimind3 -v
```

当前 synthetic fixture 覆盖 dense/MoE 布局、Q/K norm、tokenizer metadata、
F16 权重往返、tied `lm_head`、缺失/额外张量拒绝、量化模式和 K-quant 拒绝。

真实权重验证：

- dense F16：metadata 键值全部相等；90 个 tensor 的 type/dims/payload 与官方
  `minimind-64M-3-F16.gguf` 完全一致。
- MoE F16：98 个 tensor 的 type/dims/payload 与官方
  `Minimind-3-Moe-4x64M-F16.gguf` 完全一致；metadata 只多
  `general.basename` / `general.version`。
- PyTorch oracle（`transformers` 5.14.1，F32，`do_sample=False`）固定 prompt
  `你好，请用一句话介绍你自己。`，27 prompt tokens，greedy 32 tokens。
- F32、F16、BF16、Q8_0 的 dense/MoE CLI 输出与该 oracle 的 32-token greedy
  文本相同；三档浮点与 Q8_0 均能加载。Q4_0 也能加载并生成连贯中文，但 64M
  模型上文本与 F16 不同，不宣称逐 token 一致。
- `--jinja` 使用 GGUF 内嵌模板渲染 F16 dense/MoE：27 prompt tokens，32-token
  greedy 输出与内置 ChatML、官方 F16 GGUF 和 PyTorch oracle 一致。
- CLI 五种格式全部运行；F16 dense/MoE 的 HTTP `/v1/chat/completions` 返回
  `prompt_tokens=27`、`completion_tokens=32`，文本与 CLI 相同。F32/BF16/Q8_0/
  Q4_0 的 HTTP 也逐档返回 27/32 tokens。
- JEV 单次在五种格式上均可运行；F16 dense/MoE 的单次与 MultiSelect 分组输出
  与官方 GGUF 一致，单次 JSON 的浮点概率完全相同。单次示例：dense `B` /
  `0.64565796`，MoE `B` / `0.48873088`。分组示例：dense `A/C/E`，MoE
  `A/C/F`。其他量化精度的单次概率会随量化漂移，MoE Q4_0 会把选择从 `B`
  改为 `C`。分组在五档格式上均可运行，选择稳定为 dense `A/C/E`、MoE
  `A/C/F`；只有 F16 与官方 GGUF 的显示概率逐项一致。这些结果只说明转换后
  路径一致，不代表
  64M 模型对 JEV 题目的语义判断可靠。
- Vulkan 设备路径未验证。

已验证导出物：

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| `MiniMind-3-F32.gguf` | 255,889,120 | `66f6b1201c0ea6bf745e821342f5c9a95efe75d9dac03b813ce890c100662bb8` |
| `MiniMind-3-F16.gguf` | 128,093,920 | `941d6b7993fc5cfdb46a8113526eca3f3469c80705099fb3b1d336d589c2064b` |
| `MiniMind-3-BF16.gguf` | 128,093,920 | `7343c52d689075fc99ad5d46bcdebe78a2554a80b3cc49632a4bbba4c98da614` |
| `MiniMind-3-Q8_0.gguf` | 68,189,920 | `35ccc1af7baca251dd0010a372d62fb467b8d2ae1e94086d5057b69fd845dd1c` |
| `MiniMind-3-Q4_0.gguf` | 36,241,120 | `237c504de672db685d7da2abf7d73d635018835933f7d9a8e56c0efcc8531985` |
| `MiniMind-3-MoE-F32.gguf` | 793,907,904 | `acb7f6bb977ba8073e19e8941a4b10d4fb848c8bfcbbc42b30153f3e542dce2f` |
| `MiniMind-3-MoE-F16.gguf` | 397,152,960 | `102c41d5871d0218059a0245d62348e55319b21d864ea31487f3f0b19e6e0ec2` |
| `MiniMind-3-MoE-BF16.gguf` | 397,152,960 | `391a8a738c24a8c7260e74042af1570ceae71ea1cb5f0c637fa17bad28ddbe83` |
| `MiniMind-3-MoE-Q8_0.gguf` | 211,174,080 | `fa475f631b377b5a9c39e417cfb54d4784a8f2eda0376abb077c1238dc7c32a0` |
| `MiniMind-3-MoE-Q4_0.gguf` | 111,985,344 | `367e091d21f511d839bc0be0f1a5a2d79fc3e594cf5747e6de2a2e93947edde3` |

## MiniMind2-MoE → minimind2moe

把 MiniMind2-MoE 的原始 PyTorch state dict（`full_sft_640_moe.pth`）转成本仓库的
`minimind2moe` GGUF。参考实现固定为 MiniMind 仓库
`561979c:model/model_minimind.py`；该版本的 trunk 是 Qwen3 风格 GQA/RMSNorm/SiLU，
但**没有 Q/K norm**，MoE 层在 4 个 routed experts 之外还有一个始终执行的 shared expert。

### 输入

- checkpoint：`gongjy/MiniMind2-PyTorch` 的 `full_sft_640_moe.pth`
- tokenizer 目录：至少包含 `tokenizer.json` 和 `tokenizer_config.json`，例如
  `gongjy/MiniMind2` 仓库。不要使用该仓库的 `config.json` 推导模型维度：它描述的是
  dense 768/16 层配置，不是 MiniMind2-MoE。
- 参考实现维度：hidden 640、8 层、8 heads、2 KV heads、head_dim 80、vocab 6400、
  4 routed experts top-2、shared FFN 1728、context 32768、RoPE theta 1e6、RMS eps 1e-5。

### 转换

在仓库根目录运行：

```bash
.venv/bin/python -m tools.converter.minimind.convert_minimind2moe \
  /path/to/full_sft_640_moe.pth \
  --tokenizer-dir /path/to/MiniMind2 \
  --outfile models/MiniMind2-MoE-F16.gguf \
  --quant f16
```

`--quant` 支持 `f16`（默认）、`f32`、`bf16`、`q8_0`、`q4_0`。norms 和 router 始终写
F32；embedding（同时作为 tied 输出投影）/attention/experts/shared experts 使用所选格式。K-quant 被拒绝：
矩阵行宽为 640 或 1728，不能被 256 值 K-quant block 整除。

其他常用参数：`--name`、`--context-length`、`--rope-freq-base`、`--norm-eps`、
`--num-attention-heads`、`--num-key-value-heads`、`--chat-template-file`。

### 张量映射与校验

- 4 个 routed experts 按 expert 顺序堆叠：
  `blk.N.ffn_{gate,up,down}_exps.weight`；router 为
  `blk.N.ffn_gate_inp.weight`（F32）。
- shared expert 写为 `blk.N.ffn_{gate,up,down}_shexp.weight`，metadata
  `minimind2moe.expert_shared_feed_forward_length=1728` 让它始终执行。
- `model.norm.weight` → `output_norm.weight`；`lm_head.weight` 若存在必须与
  `model.embed_tokens.weight` 完全相同，然后省略（tied embedding）。
- 转换器拒绝缺失/额外张量、非连续层号、错误 attention/FFN/expert 形状、未 tie 的
  `lm_head`，以及不满足量化 block 对齐的矩阵。
- Tokenizer 固定为 GPT-2 BPE、`tokenizer.ggml.pre=qwen2`、6400 tokens / 6141 merges；
  BOS=1、EOS=2、`add_bos_token=false`、`add_eos_token=false`。

### Chat template

默认写入 `561979c:model/tokenizer_config.json` 的 ChatML 模板。它会为没有 system
消息的对话补 `You are a helpful assistant`。当前 tokenizer 仓库里的
`chat_template.jinja` 已增加 tools 支持，与该训练版本不同；需要新模板时显式传
`--chat-template-file`。

运行时默认的 `--prompt`/HTTP 入口使用内置 ChatML（不带 system 轮）；`--jinja` 才会
渲染 GGUF 带 system 轮的模板，两者 token 数不同，这是预期行为。

### 验证

```bash
python3 -m unittest tools.converter.minimind.test_convert_minimind2moe -v
```

当前合成 fixture 测试覆盖 7 项：布局/张量映射、tokenizer metadata、f16 权重往返、
tied `lm_head`、缺失/额外张量拒绝、量化模式和 K-quant 拒绝。真实权重已导出 F16、
F32、BF16、Q8_0、Q4_0；F16 的 32-token greedy 输出与固定 `561979c` PyTorch oracle
逐 token 相同，CLI/HTTP/JEV 均通过，F32/BF16 的前 8 token 与 F16 相同。Q8_0/Q4_0
能加载并生成连贯文本；Q8_0 的 activation 也量化成 Q8_0，不宣称与 F16 逐 token 相同。
Vulkan 设备路径未验证。

已验证导出物的 SHA-256：

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| `MiniMind2-MoE-F32.gguf` | 580,351,200 | `4655957753f6177b89e5fb6dad5c29a92cefad25fd9616c9713446a78271f049` |
| `MiniMind2-MoE-BF16.gguf` | 290,354,400 | `52fbd8fa0adf8cd506b05e3c2f2a7673ea62a9c4031c2d6c31cc10c3d81bc44b` |
| `MiniMind2-MoE-F16.gguf` | 290,354,400 | `bd36058659aab8673c862068c0599a1f10e7d692e6ab4610746f4a592b83cf06` |
| `MiniMind2-MoE-Q8_0.gguf` | 154,418,400 | `d9a0d15f4e894f76291d6bcdbef5f942ee1dc4ab7b0ae263d41416fea9f1017b` |
| `MiniMind2-MoE-Q4_0.gguf` | 81,919,200 | `6faf5b4b222830c2771900d946df9c0141c0ba3b3c8f1efc86c9f5c870cda608` |