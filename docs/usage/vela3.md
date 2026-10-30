# Vela3 System One

六个官方 safetensors 包使用独立的 `vela3` CLI 和 `models::vela3::system_one` Rust API。计算在 CPU 上执行，复用仓库的线程池、矩阵投影、激活与 Qwen3.5 视觉编码器；不调用 Python 或外部加速库。无需 GGUF 转换。

| 包 | 显式格式 / 架构 | 任务 |
|---|---|---|
| `vllm-sr/vela3-mini` | `vela3/3` / `qwen3_5` | choice、noul、score、span；图像输入为实验性 |
| `vllm-sr/vela3-lite` | 同上 | 同上 |
| `vllm-sr/vela3-flash` | 同上 | 同上 |
| `vllm-sr/vela3-turbo` | 同上 | 同上；普通决策 prefix-bidir，路由决策按配置使用 noncausal |
| `vllm-sr/vela3-fast` | `vela3-encoder/1` / `ultrabert` | choice、noul、span、训练过的 router choice/set |
| `vllm-sr/vela3-fast-vision` | `vela3v-encoder/1` / `ultravbert` | 图像 choice/noul、幻觉 span、OCR 检查、box |

## 权重目录

保留官方包的配置、tokenizer、chat template 和所用任务的 safetensors。下载位置使用 `models/<repo>/`，文件哈希与固定参考版本见 [snapshots.json](../../tools/oracle/vela3/snapshots.json)。

d3 四个变体还需要其 `vela3_config.json` 固定的独立底座，包含完整模型分片、`readout.safetensors`、`decision_config.json` 与 tokenizer。必须匹配 **base.revision**：当前魔搭 d3-mini、d3-turbo 镜像权重与 Vela 固定版本不同，不能把镜像最新版当成同一个参考 checkpoint。

Vela 的全部配置适配器在 d3 加载时校验。flash 的默认自动标签定义还使用 `labeldef_lm_head.safetensors`；缺失时按官方策略对未缓存标签使用名字。mini 使用 tied embedding 写定义，lite/turbo 默认使用名字；也可以在问题中给 `definitions` 对象或 `null`。不自动下载可选的外部 definition writer。

fast-vision 的三个底座分别为 `model.safetensors`、`backbone_h2.safetensors`、`backbone_v2.safetensors`，不能相互替代；还需对应的 span、decision、box、OCR 适配器。

## 运行

```bash
cargo build --profile release-fast --bin vela3
cargo run --profile release-fast --bin vela3 -- \
  --model models/vela3-lite --base-model models/d3-lite \
  --request request.json --threads 4
```

`--request -` 或省略 `--request` 从 stdin 读取。stdout 只输出响应 JSON；初始化失败退出非零，单个问题的参数错误放在该问题的 `error` 中。

```json
{
  "state": {"user": "Alice works at Acme.", "answer": "Alice works at Acme."},
  "questions": {
    "kind": {"type": "choice", "instructions": "Who is mentioned?", "criteria": {"person": "a human", "company": "a business"}},
    "entities": {"type": "span", "criteria": {"person": "a person's name", "company": "a company name"}, "definitions": null}
  }
}
```

fast/fast-vision 省略 `--base-model`。图像通过顶层 `images` 数组传本地文件、HTTP(S) URL 或 base64 image data URL；fast-vision 最多一张图。Span 的 start/end 是 Unicode 字符偏移，遵循对应官方解码器的 URL、CJK、阈值和边界规则。d3 的 span pass 不计入官方 `usage.input_tokens`，这里保留该口径。turbo 的 usage 按官方入口合并共享 state，并计入路由适配器的额外 pass；当前实现仍按问题重算 state，未验证长文本性能。

非空 `videos`、`assets` 和启用的 `long_context` 会在加载权重前明确报错；当前入口不支持这些模式。

```rust
let response = rust_model_inference::models::vela3::system_one(
    std::path::Path::new("models/vela3-lite"),
    Some(std::path::Path::new("models/d3-lite")),
    &request,
    4,
)?;
```

## 精度与边界

Token IDs、提示词、任务路由、窗口与输出偏移做精确检查；SIMD 归约、BF16 舍入、softmax、GELU 等浮点路径使用记录的误差门槛。默认不承诺逐位一致。[Oracle 命令与验证记录](../../tools/oracle/vela3/README.md) 区分真实整模、小型官方计算图和独立任务头。

这次接入为 CPU safetensors 路径，未接入 GGUF、主生成 CLI、HTTP server 或 GPU。mini/flash 的真实公开请求和四步 greedy IDs 通过对照，但非因果内部 BF16 activation 门槛未通过，仍为实验性。d3 图像复用 F32 视觉编码器，和官方默认 BF16 塔有可测差异，也属于实验性；长上下文性能、完整任务基准及所有尺寸的整模数值覆盖仍应按验证记录判断。
