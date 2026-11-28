# Hy-MT3 Oracle（llama.cpp 对拍）

腾讯 **Hy-MT3** 三个 GGUF（1.8B / 7B / 30B-A3B）与本仓库的数值/离散契约验证。
参考实现固定为 **llama.cpp `9c003a97ab34a69cc49243f9b02014038205ffae`**（上游 master，
`src/models/hy-v3.cpp` 已实现 `hy_v3`，`src/models/hunyuan-dense.cpp` 实现
`hunyuan-dense`）。

## 为什么需要自己写 harness

llama.cpp 在 pin 的 commit 上**没有**通用 logits dump：

- `llama-cli` 只有 `--logit-bias` / `--verbose-prompt`；
- 唯一的 dump 是 perplexity 的 `--save-all-logits`，格式为
  `"_logits_" | i32 n_ctx | i32 n_vocab | i32 n_chunk | tokens[] | 每行 u16 量化 log-prob`，
  **有损**、只覆盖位置 `[n_ctx/2, n_ctx-2]`、且要求输入至少 `2*n_ctx` 个 token，
  对 23 token 的短 prompt 不可用。

所以本目录的 `oracle.cpp` 直接链接 `libllama`，输出精确 float32：

| 文件 | 内容 |
| --- | --- |
| `prompt_tokens.json` | `{n_prompt_tokens, add_bos, prompt_tokens[]}` |
| `logits_prompt.f32` | 最后一个 prompt 位置的 `n_vocab` 个 float32 |
| `logits_all.bin` | `LLOG` + u32 version + u32 n_vocab + u32 n_gen + `n_gen*n_vocab` float32 |
| `generated_tokens.json` | `{n_generated, generated_tokens[]}`（greedy argmax） |
| `text.txt` | 生成 token 拼回的文本 |
| `topk.txt` | 每步 top-k（id / logit / piece） |
| `summary.json` | 模型路径、n_vocab、上下文/线程/批次设置、llama.cpp commit |

Rust 侧的 `examples/hy_mt3_parity.rs` 写出**同样布局**的
`prompt_tokens.json` / `logits_prompt.f32` / `logits_all.bin` /
`generated_tokens.json` / `text.txt`，因此 `compare.py` 不关心哪边是参考。

## 构建与运行

```bash
# 1) llama.cpp（codeload tarball；本机 github.com 的 git 协议被代理挡住）
curl -sL -o llama.tar.gz \
  https://codeload.github.com/ggml-org/llama.cpp/tar.gz/9c003a97ab34a69cc49243f9b02014038205ffae
mkdir -p x && tar xzf llama.tar.gz -C x && mv x/llama.cpp-* llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build llama.cpp/build -j64 --target llama-cli

# 2) oracle
g++ -O2 -std=c++17 -I llama.cpp/include -I llama.cpp/ggml/include oracle.cpp \
    -L llama.cpp/build/bin -lllama -lggml -lggml-base -o oracle

# 3) 参考侧
LD_LIBRARY_PATH=llama.cpp/build/bin ./oracle \
  -m models/Hy-MT3-30B-A3B-GGUF/Hy-MT3-30B-A3B-Q4_K_M.gguf \
  --prompt-file prompt.txt -n 32 -t 16 -c 2048 --add-bos 0 --outdir out/oracle

# 4) 本仓库侧
cargo build --release --example hy_mt3_parity
./target/release/examples/hy_mt3_parity \
  models/Hy-MT3-30B-A3B-GGUF/Hy-MT3-30B-A3B-Q4_K_M.gguf \
  prompt.txt out/rust v3 32 16 2048

# 5) 比较
python3 compare.py out/oracle out/rust
```

`v3` 走 `HyDecoder` + `HySession`（30B-A3B），`dense` 走 qwen3 trunk（1.8B / 7B）。

固定设置（两侧必须一致）：greedy argmax、`n_gen=32`、`n_ctx=2048`、
`n_batch=n_ubatch=256`、KV **F32**、flash-attn 关闭、`n_gpu_layers=0`、
`add_bos=0`、`parse_special=true`。

## 三个 prompt 变体

| 变体 | 字节数 | 用途 |
| --- | --- | --- |
| `prompt.txt`（raw） | 143 | 纯文本 prompt，无模板歧义，**数值对齐的主证据** |
| `prompt-chat.txt` | 178 | `<｜hy_User｜>{prompt}<｜hy_Assistant｜>`，无 BOS |
| `prompt-chat-bos.txt` | 210 | 上一行再加 `<｜hy_begin▁of▁sentence｜>`，**与 GGUF 模板实际产出一致** |

模板实际会发 BOS（`{% else %}<｜hy_begin▁of▁sentence｜>{% endif %}`），所以
`chat_bos` 才是任务路径；`chat` 是不带 BOS 的诊断变体，模型对它处于分布外，
logits 分布更平坦，尾部排序对噪声敏感。

## Tokenizer 契约（不需要权重）

`tokenizer_parity.py` 直接拿官方 `tokenizer.json`（HF `tokenizers`）当权威，
和本仓库的分词器逐 ID 对比，覆盖数字/中日韩/组合字符/emoji/缩写/标点/空白等
14 组用例。`head_to_tokenizer_gguf.py` 可以把 GGUF 头部的 metadata 单独抽成
一个 0 张量的 tokenizer-only GGUF（几 MB），因此这项检查不需要下载几十 GB 权重。

```bash
pip install tokenizers
python3 head_to_tokenizer_gguf.py Hy-MT3-30B-A3B-Q8_0.head 30B.gguf
python3 tokenizer_parity.py . /path/to/case-dir     # 需要 <tag>.json + <tag>.gguf
```

结果见 [`validation.md`](validation.md)。
