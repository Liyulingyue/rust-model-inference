# MiniCPM5-1B llama.cpp Debug Notes

## 编译

1. 在 `src/llama-vocab.cpp` 的 `llama_tokenize()` 添加调试打印
2. `cd references/llama.cpp/build && cmake --build . --config Release --target llama-cli`

## 运行命令

```bash
.\llama-cli.exe -m MiniCPM5-1B-Q8_0.gguf -p "The capital of France is" -n 1 --temp 0 --log-disable
```

## llama.cpp 实际 tokenize 行为

### 输入
```text
text = "<s>user\nThe capital of France is\nassistant\n<think>\n"
text_len = 85 (实际从输入 + chat template 渲染后)
add_special = 1, parse_special = 1
```

### 实际生成的 token IDs (16 tokens)
```
0 130072 8448 220 608 4894 304 6918 357 130073 220 130072 130071 220 8 220
```

### Token 分解

| ID | 含义 | 来源 |
|----|------|------|
| 0 | `<s>` BOS | add_special=true |
| 130072 | `` | literal `<\|im_start\|>` in chat template |
| 8448 | `user` | BPE |
| 220 | `\n` | BPE |
| 608 | `The` | BPE |
| 4894 | ` capital` | BPE |
| 304 | ` of` | BPE |
| 6918 | ` France` | BPE |
| 357 | ` is` | BPE |
| 130073 | `` | literal `<\|im_end\|>` in chat template |
| 220 | `\n` | BPE |
| 130072 | `` | literal `<\|im_start\|>` in chat template |
| 130071 | `assistant` | BPE |
| 220 | `\n` | BPE |
| 8 | `<think>` | BPE (jinja enable_thinking=true) |
| 220 | `\n` | BPE |

## Chat Template 渲染

MiniCPM5 的 Jinja chat template 渲染格式（无 system, 无 tools, enable_thinking=true）:

```
<s><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n
```

**关键点**:
- 分隔符用 `\n` (id=220)，**不是** `<|im_sep|>` (id=4)
- 不使用 phi_4 chat template（因为 Jinja 模板优先于 auto-detect）
- 末尾 `<think>\n` 触发思考模式

## Token IDs 速查

| Token | ID |
|-------|-----|
| `<s>` | 0 |
| `</s>` | 1 |
| `<\|im_sep\|>` | 4 |
| `<think>` | 8 |
| `</think>` | 9 |
| `assistant` | 130071 |
| `<\|im_start\|>` | 130072 |
| `<\|im_end\|>` | 130073 |
| `<\|thought_begin\|>` | 130075 |
| `<\|thought_end\|>` | 130076 |

## Sample (`--temp 0` greedy)

第一 token: `0` (BOS, only as context)

后续真实生成 token (with prompt):
- ...

## 我们的实现对比

### 之前错误的实现 (旧 phi_4 风格)
```
[0, 130072, 8448, 4, 608, 4894, 304, 6918, 357, 130073, 130072, 130071, 4]
```
- ❌ 用 `<|im_sep|>` (4) 而非 `\n` (220)
- ❌ 缺少 `assistant\n<think>\n` 触发思考模式
- ❌ 缺少末尾 `\n`

### 目标实现 (对齐 llama.cpp)
```
0 130072 8448 220 608 4894 304 6918 357 130073 220 130072 130071 220 8 220
```
- ✅ 用 `\n` (220) 作为分隔符
- ✅ 添加 `<think>\n` 触发思考
- ✅ 完全对齐 llama.cpp 输出