# Vela3 System One

六个官方 safetensors 包通过现有 `rust-model-inference --system-one-request` CLI 和 `models::vela3::system_one` Rust API。计算在 CPU 上执行，复用仓库的线程池、矩阵投影、激活与 Qwen3.5 视觉编码器；不调用 Python 或外部加速库。无需 GGUF 转换。

这三个包是**三个模型族**（架构、格式串与实现都不同），不是同一模型的变体：
`Vela3-D3`（`vela3/3` + `qwen3_5` decoder，mini/lite/flash/turbo 是同架构的不同尺寸）、
`Vela3-Fast`（`vela3-encoder/1` + `ultrabert` 编码器，自包含）、
`Vela3-Fast-Vision`（`vela3v-encoder/1` + `ultravbert` 视觉编码器，文本侧复用 Fast 的路径）。

| 包 | 显式格式 / 架构 | 任务 |
|---|---|---|
| `vllm-sr/vela3-mini` | `vela3/3` / `qwen3_5` | choice、noul、score、span；图像输入为实验性 |
| `vllm-sr/vela3-lite` | 同上 | 同上 |
| `vllm-sr/vela3-flash` | 同上 | 同上 |
| `vllm-sr/vela3-turbo` | 同上 | 同上；普通决策 prefix-bidir，路由决策按配置使用 noncausal |
| `vllm-sr/vela3-fast` | `vela3-encoder/1` / `ultrabert` | choice、noul、span、训练过的 router choice/set |
| `vllm-sr/vela3-fast-vision` | `vela3v-encoder/1` / `ultravbert` | 图像 choice/noul、幻觉 span、OCR、box；文本 router choice/set、span、NER、evidence |

## 权重目录

保留官方包的配置、tokenizer、chat template 和所用任务的 safetensors。下载位置使用 `models/<repo>/`，文件哈希与固定参考版本见 [snapshots.json](../../tools/oracle/vela3/snapshots.json)。

d3 四个变体还需要其 `vela3_config.json` 固定的独立底座，包含完整模型分片、`readout.safetensors`、`decision_config.json` 与 tokenizer。必须匹配 **base.revision**：当前魔搭 d3-mini、d3-turbo 镜像权重与 Vela 固定版本不同，不能把镜像最新版当成同一个参考 checkpoint。加载时按内置 snapshots.json 流式校验配置、tokenizer、完整权重分片和 readout 的 SHA256，并确认分片索引只选择已校验文件；缺少或未知的 base.revision、同形状的错误底座都会报错。校验需要顺序读取权重，会增加启动时间。

Vela 的全部配置适配器在 d3 加载时校验。flash 的默认自动标签定义还使用 `labeldef_lm_head.safetensors`；缺失时按官方策略对未缓存标签使用名字。mini 使用 tied embedding 写定义，lite/turbo 默认使用名字；也可以在问题中给 `definitions` 对象或 `null`。不自动下载可选的外部 definition writer。

fast-vision 的三个底座分别为 `model.safetensors`、`backbone_h2.safetensors`、`backbone_v2.safetensors`，不能相互替代；还需对应的 span、decision、box、OCR 适配器。视觉任务的底座与 adapter 都校验固定尺寸和 SHA256；decision/box 必须提供匹配的 backbone_file，span/OCR 可使用官方明确声明的 raw + backbone.model，未声明时不会回退到默认权重。LoRA rank/alpha 必须完整且与任务匹配。

fast-vision 的文本任务使用包内 `text/` 的独立配置、tokenizer、`model.safetensors` 及 span/router/NER/domain/feedback 适配器，不能用普通 fast 的权重替代。无图时，训练过的 router choice/set 与 span 使用文本适配器，普通 choice/noul 仍使用视觉侧的 decision 网络。有图时，choice/noul、box、图像幻觉 span 和 OCR 使用视觉任务；其他 span 标签集和 set 仍走文本侧。

## 运行

```bash
cargo build --profile release-fast --bin rust-model-inference
cargo run --profile release-fast --bin rust-model-inference -- \
  --model models/d3-lite --adapter models/vela3-lite \
  --system-one-request request.json --threads 4
```

`--system-one-request -` 从 stdin 读取。stdout 只输出响应 JSON；初始化失败退出非零，单个问题的参数错误放在该问题的 `error: invalid_question` 中，不会覆盖同批成功题；usage 仍统计成功题。模型配置和权重错误会使请求失败。

```json
{
  "state": {"user": "Alice works at Acme.", "answer": "Alice works at Acme."},
  "questions": {
    "kind": {"type": "choice", "instructions": "Who is mentioned?", "criteria": {"person": "a human", "company": "a business"}},
    "entities": {"type": "span", "criteria": {"person": "a person's name", "company": "a company name"}, "definitions": null}
  }
}
```

省略 `--threads` 或设为 0 时沿用主程序的自动线程数。只接受 `--model`、`--adapter`、`--system-one-request`、`--threads` 和 `--compute cpu`，其他任务输入放在 JSON 中。`--model` 是主权重（d3 底座快照），`--adapter` 是 Vela3 适配器包（LoRA、heads 与 `vela3_config.json`）；fast/fast-vision 自包含，只给 `--model`（即整个包）。图像通过顶层 `images` 数组传本地文件、HTTP(S) URL 或 base64 image data URL；fast-vision 的 choice/noul 最多三张，图像 span/OCR/box 按官方实现使用第一张。Span 的 start/end 是 Unicode 字符偏移，遵循对应官方解码器的 URL、CJK、阈值和边界规则。d3 的 span pass 不计入官方 `usage.input_tokens`，这里保留该口径。turbo 的 usage 按官方入口合并共享 state，并计入路由适配器的额外 pass；当前实现仍按问题重算 state，未验证长文本性能。

fast 的 span 路由只读取包内配置，Rust 入口不采用 Python 的可选 `VELA3_ENTITY_ROUTE` 环境覆盖；阈值和上下文长度缺失会报错。视觉 tokenizer 的普通词表边界来自 tokenizer，起始 ID、上下文来自模型配置；实际使用的图像前缀特殊 token 必须是完整 added token。

fast-vision 按官方处理器输出固定 512×512 切片，connector 每片为 8×8（64 tokens），加载时检查处理器和模型几何配置一致；任意宽高比原图都会先进入这套处理，因此无需向 SigLIP 传非方形输入。box/point 是拼接热图上的 [0,1] 归一化坐标，不是原图像素。图像预处理按 image/edge 缓存，最终 image embedding 只在同一请求、同一任务内复用，避免混用不同 connector。

fast 仍保留官方的单个 shrink role 和窗口规则：只先缩减第一个非保护段，随后截取保护段并按需窗口化；剩余输入仍超长则报错，不另行改变官方提示词。

turbo 的普通决策按官方 `validated_length` 限制为 32768 tokens，超限问题返回 `max_length_exceeded` / `question_tokens`，不截断输入。非空 `videos`、`assets` 和启用的 `long_context` 会在加载权重前明确报错；当前入口不支持这些模式。

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

这次接入为主 CLI 的 CPU safetensors 请求模式，未接入 GGUF、文本生成、HTTP server 或 GPU。`--compute auto/vulkan` 和 `--gpu` 会明确拒绝。mini/flash/turbo 的真实公开请求通过对照，mini/flash 另有四步 greedy IDs 对照；三者的非因果内部 BF16 activation 门槛未通过，仍为实验性。turbo 不含 LM head，没有真实 causal greedy 验证。d3 图像复用 F32 视觉编码器，和官方默认 BF16 塔有可测差异，也属于实验性；长上下文性能、完整任务基准及所有尺寸的整模数值覆盖仍应按验证记录判断。

本轮 review 的逐项结论、保留的官方行为与验证范围见 [修复记录](../../tools/oracle/vela3/review-2026-11-05.md)。
