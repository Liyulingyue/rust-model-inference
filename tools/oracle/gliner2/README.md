# GLiNER2.5-Decide 精度对齐

## 固定参考

- 上游 GLiNER2（提供 `SchemaTransformer`，也就是输入构造的权威实现）：
  `https://github.com/fastino-ai/GLiNER2`，checkout 放在 `target/gliner2-oracle`
  或用 `GLINER2_SRC` 指过去。
- 编码器配置：`microsoft/deberta-v3-large` 的 `config.json`。GLiNER2.5-Decide
  的 repo 里没有这一份（checkpoint 只带了一份 reshape 过的 DeBERTa 权重），
  脚本从 ModelScope 取一次放到 `target/gliner2-deberta-config`。
- `transformers` **4.48.1**（必须锁版本：`deberta_v2` 的
  `disentangled_attention_bias` 在小版本之间动过）。
- 权重：`fastino/GLiNER2.5-Decide` 的 `model.safetensors`，
  SHA256 `40a5a23ff860dc3dff426cecd1048cacdd29c648c96db209dad818e9686dc997`
  （转换器也把它写进 `gliner2.source_sha256`）。
- 已验证：Python 3.14、torch 2.14.0+cpu、sentencepiece 0.2.2、aarch64。

## 两个 fixture

| 文件 | 生成脚本 | 断言什么 |
|---|---|---|
| `tests/fixtures/gliner2-decide/classify-golden.json` | `dump_golden.py` | 6 个 case 的 `input_ids`、`[P]/[L]` subword 下标、每 label logit |
| `tests/fixtures/gliner2-decide/spm-pieces.json` | `dump_spm.py` | 69 条字符串的 SentencePiece pieces / ids |

两个脚本都不往仓库里写权重，只读 `models/GLiNER2.5-Decide/`。

## 复现

```sh
# 依赖（都在仓库 venv 里）
models/.venv/bin/pip install "transformers==4.48.1" sentencepiece safetensors

# 上游 checkout
git clone --depth 1 https://github.com/fastino-ai/GLiNER2 target/gliner2-oracle

# 编码器配置（GLiNER2.5-Decide repo 里没有）
models/.venv/bin/python - <<'EOF'
import json, urllib.request
url = "https://www.modelscope.cn/models/microsoft/deberta-v3-large/resolve/master/config.json"
with urllib.request.urlopen(url) as response:
    config = json.load(response)
with open("target/gliner2-deberta-config/config.json", "w") as handle:
    json.dump(config, handle)
EOF

# 重新生成 fixture（结果必须与仓库里的完全一致）
models/.venv/bin/python tools/oracle/gliner2/dump_spm.py
models/.venv/bin/python tools/oracle/gliner2/dump_golden.py
git diff --stat tests/fixtures/gliner2-decide/
```

## 跑对齐

```sh
# 转换器（先有 model.safetensors / config.json / spm.model / tokenizer_config.json）
models/.venv/bin/python tools/converter/gliner/convert_gliner.py \
    models/GLiNER2.5-Decide models/GLiNER2.5-Decide/gliner2-decide-f32.gguf

cargo test --profile release-fast --test gliner2_spm_parity
cargo test --profile release-fast --test gliner2_classify_parity
cargo test --profile release-fast --test gliner2_cli
```

## 对齐结果

`gliner2_classify_parity` 覆盖 6 个 case（单标签 4 标签 / 带指令的 yes-no /
多标签 aspects / 4 个头一次前向 / 带 label 描述 / 11 级 ordinal），三层都逐位
对齐：

| 层 | 断言 | 结果 |
|---|---|---|
| `input_ids` | 整条 subword 序列完全相等 | 6/6 相等 |
| `[P]/[L]` 下标 | `batch.schema_special_indices` 完全相等 | 6/6 相等 |
| logit | `abs(got - want) < 1e-4` | 最大偏差 **7.2e-6** |

7.2e-6 是 F32 累加顺序差（参考走 BLAS，Rust 侧逐元素走 SIMD 行 kernel）。
结构性错误（scale 取错、位置分桶错、marker 错位、位置投影跨层共用）会把 logit
整体挪掉至少 1，所以阈值离噪声有四个数量级的余量。

`gliner2_spm_parity` 的 69 条覆盖 NFKC 折叠（`①`→`1`、`Ⅷ`→`VIII`、`™`→`TM`、
全角数字）、byte fallback（emoji、Lisu、组合记号）、空白折叠和空串。

## 数值契约

- 点积/求和按元素顺序累加 F32（与仓库其它 encoder 一致）。
- LayerNorm 两遍总体方差，`eps = 1e-7`（`layer_norm_eps`），不是 `1e-5`。
- GELU 用 erf 形式（`ACT2FN["gelu"]`，`approximate="none"`）。
- 三项 attention（content·content / content·position / position·content）
  共用 `1/sqrt(64 * 3)`。
