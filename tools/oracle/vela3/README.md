# Vela3 Oracle

参考为六个 ModelScope 包附带的 Python 实现及其固定 d3 版本。源码 SHA256、权重大小与哈希见 [snapshots.json](snapshots.json)，实际结果见 [verification.json](verification.json)。模型、临时 trace 和大数组不提交到 Git。

Native 只调用已有 Rust 算子。Python 依赖仅运行只读 Oracle，安装在仓库 `.venv`：Torch 2.9.1、Transformers 5.19.0、torchvision 0.24.1、numpy 2.5.3、Pillow 13.0.0、safetensors、accelerate、SciPy。`accelerate` 满足官方 CPU `device_map` 加载要求，SciPy 供官方 box 连通区域解码使用。

fast-vision 原包面向更新的 Transformers；Oracle 注册包内原始 `text/modeling_ultrabert.py`，仅兼容配置声明和类注册，保持计算实现。预处理匹配 torchvision 0.24.1 的 uint8 bicubic 回退，未验证更新版本的 Lanczos 路径。所有模型文件均须先按清单下载到本地；下列 Oracle 不联网换模型。

## 主程序入口

Vela3 使用现有 `rust-model-inference --system-one-request FILE|-`；`--model` 是主权重（d3 底座快照），d3 另传 `--adapter DIR` 指向 Vela3 适配器包。先前新增的模型专用 binary 改变了 Cargo 自动发现的编译产物，已移除；计算与公开 Rust API 保持在 `models::vela3`。下列复现命令已改用主程序，verification.json 中早期的专用 binary 命令仅为历史测量记录。当前入口与产物检查见 `entry_fix_2026_11_02`。

2026-11-02 入口修复验证：Cargo 的 9 个 binary targets 与 main 基线一致；主程序与相关 CLI 共 38 项测试通过，包含 6 项 System One 回归。fast（stdin）、lite（独立底座）和 fast-vision 文本入口共 29 问与已有固定 Oracle fixture 对照通过，完整 JSON 也与原 native 记录一致。全库仍为 1360 通过、20 失败、94 忽略，20 项失败名称与修复前完全相同。其他模型和图像逐层记录未重跑。

## 混合 Vulkan 投影（2026-11-06）

此前入口的 CPU-only 限制已解除。`--compute auto|vulkan` 和 `--gpu` 经共享 `ComputePolicy` 进入 F32/F16/BF16 投影分发；默认 CLI 和原 `system_one` Rust API 保持 CPU。显式 Rust API 为 `system_one_with_policy`。自动标签定义生成固定使用 CPU，避免改写 span 提示词；attention、递归状态与解码仍含 CPU 步骤，未实现完整设备常驻图，也没有新增 binary、依赖或共享 kernel 改动。

```bash
cargo +1.98.1 build --locked --profile release-fast --features vulkan --bin rust-model-inference
RMI_COMPUTE_TRACE=1 target/release-fast/rust-model-inference \
  --model models/vela3-fast --system-one-request tools/oracle/vela3/requests/fast.json \
  --threads 4 --compute vulkan > models/vela3-fast-vulkan.json
.venv/bin/python tools/oracle/vela3/compare_requests.py \
  models/vela3-fast-response.json models/vela3-fast-vulkan.json \
  --atol 5e-4 --rtol 5e-4 --out models/vela3-fast-vulkan-comparison.json
cargo +1.98.1 test --locked --profile release-fast --features vulkan --lib \
  gpu_projection_and_lora_match_cpu_for_all_float_formats -- --ignored --nocapture
```

Oracle 响应文件按下方 Fast 命令生成。将 `vulkan` 换成 `cpu`、`auto` 可对照策略；`--gpu` 等同 `auto`。macOS 需先配置可用 Vulkan loader / MoltenVK ICD，仅对测试子进程设置路径；设备缺失时 Auto 回退，严格 Vulkan 在加载权重前报错。trace 的 `scope=linear backend=Vulkan` 与实际提交增量证明投影上设备，软件 ICD 不计为物理 GPU 证据。

具体主机、逐请求数值/计数、回归基线和未验证范围见 [GPU 实测记录](gpu-2026-11-06.md) 与 `verification.json` 的 `gpu_2026_11_06`。本次不放宽已有概率、BF16 或内部激活门槛。原 #193 合并后，GPU 增量另基于最新 main 创建后续 PR；集成后的基线与 GPU 冒烟检查单列在同一报告，未冒充 57 问全量复跑。

## 便携检查

以下检查不需要权重或 Python 推理依赖。验证机器使用 Rust 1.98.1、macOS arm64；普通测试跳过需要本地 fixture 的检查。

```bash
cargo +1.98.1 test --profile release-fast --lib models::vela3::
cargo +1.98.1 test --profile release-fast --lib format::safetensors::tests::
.venv/bin/python -m unittest discover -s tools/oracle/vela3 -p 'test_*.py'
cargo +1.98.1 build --profile release-fast --bin rust-model-inference
cargo +1.98.1 test --locked --profile release-fast --test system_one_cli
```

早期适配验证（非本轮入口修复）：41 个 Vela3 测试通过、5 个需要本地权重的检查被忽略；比较器与 Oracle 生成契约共 6 个检查通过。完整库命令 `cargo +1.98.1 test --profile release-fast --lib` 为 1322 通过、20 失败、89 忽略；未修改的 `92d2eca0` 为 1280 通过、相同的 20 失败、84 忽略，失败名称保存在验证 JSON。Clippy 的 10 个错误均在本次未修改的模块，不能声称全仓检查通过。

## Fast 全路由

小型官方计算图包含九种适配器、15 个问题，以及独立的长窗口聚合检查。

```bash
.venv/bin/python tools/oracle/vela3/fast_oracle.py \
  --model models/vela3-fast --synthetic models/vela3-fast-tiny \
  --out models/vela3-fast-tiny-oracle.json
RMI_VELA3_FAST_ROOT=models/vela3-fast-tiny \
RMI_VELA3_FAST_FIXTURE=models/vela3-fast-tiny-oracle.json \
  cargo +1.98.1 test --profile release-fast --lib native_fast_matches_official \
  -- --ignored --nocapture

.venv/bin/python tools/oracle/vela3/fast_oracle.py \
  --model models/vela3-fast --synthetic models/vela3-fast-window \
  --span-window 64 --out models/vela3-fast-window-oracle.json
RMI_VELA3_FAST_ROOT=models/vela3-fast-window \
RMI_VELA3_FAST_FIXTURE=models/vela3-fast-window-oracle.json \
  cargo +1.98.1 test --profile release-fast --lib native_fast_matches_official \
  -- --ignored --nocapture

.venv/bin/python tools/oracle/vela3/fast_oracle.py \
  --model models/vela3-fast --request tools/oracle/vela3/requests/fast.json \
  --system-one --require-all-adapters --out models/vela3-fast-oracle.json
RMI_VELA3_FAST_ROOT=models/vela3-fast \
RMI_VELA3_FAST_FIXTURE=models/vela3-fast-oracle.json \
  cargo +1.98.1 test --profile release-fast --lib native_fast_matches_official \
  -- --ignored --nocapture
.venv/bin/python -c 'import json; print(json.dumps(json.load(open("models/vela3-fast-oracle.json"))["system_one"]))' \
  > models/vela3-fast-response.json
cargo +1.98.1 run --profile release-fast --bin rust-model-inference -- \
  --model models/vela3-fast --system-one-request tools/oracle/vela3/requests/fast.json --threads 4 \
  > models/vela3-fast-native.json
.venv/bin/python tools/oracle/vela3/compare_requests.py \
  models/vela3-fast-response.json models/vela3-fast-native.json \
  --atol 5e-4 --rtol 5e-4 --out models/vela3-fast-comparison.json
```

`--require-all-adapters` 拒绝缺失的可选适配器，避免把官方回退到共享底座误记成任务适配器已验证。

`requests/fast.json` 的真实权重检查包含全部九适配器和 18 个问题。`requests/fast-criteria.json` 单独覆盖普通 choice/router 的数组、重复选项，以及重复 span 标签报错；`requests/fast-thresholds.json` 覆盖 Unicode 数字、下划线、布尔值、巨大阈值和无效字符串。将上面 Oracle 的 `--request` 与 CLI 的 `--system-one-request` 参数一并替换即可复现；两组边界请求已在确定性小型官方图上通过。

`requests/fast-floats.json` 覆盖科学计数法 state、JSON instructions/description 和 router JSON state。此前 fast 使用 serde 的指数格式，`1e-7` 与 Python 的 `1e-07` 会改变 token IDs 和概率；现复用 d3 的 Python JSON 渲染。用此请求替换上述 Oracle 的 `--request` 和 CLI 的 `--system-one-request` 即可运行同一完整检查，不能将提示词差异计入浮点算子容差。

## Fast-vision 四种真实任务

每条任务自动生成同一张确定性测试图，逐层记录五个图像切片、SigLIP、pixel shuffle、connector、UltraBERT、选取 hidden、任务 logits，并比较完整公开 System One JSON。

```bash
for task in decision span ocr box; do
  .venv/bin/python tools/oracle/vela3/vision_oracle.py \
    --root models/vela3-fast-vision --task "$task" \
    --out "models/vela3-vision-$task"
  RMI_VELA3_VISION_FIXTURE="models/vela3-vision-$task" \
    cargo +1.98.1 test --profile release-fast --lib \
    models::vela3::vision::tests::official_fixture -- --exact --ignored --nocapture
done
```

省略 `--task` 可生成小型官方图；`--raw` 使用真实 raw 底座与确定性合成任务头。两者不能当成四个真实任务适配器的替代证据。

## Fast-vision 公开包装器与文本任务

此前四任务检查使用 `vela3v_encoder.py`，只能证明视觉子接口，不能证明 `vela3_encoder.py` 的完整公开包装器。公开包装器无图时把训练过的 router choice/set 和 span 分到包内 `text/`，普通 choice/noul 仍用视觉 decision；有图时，其他标签集的 span 和 set 仍走文本。原实现缺少此分流，现使用各自的独立权重并复用 fast 文本计算。

先按 snapshots.json 下载六个 `text/*.safetensors` 及其配置/tokenizer/源码。以下 Oracle 校验固定源码和文本文件哈希，保持公开包装器、文本 runtime 和模型计算不变，复用上述 Transformers 配置兼容层。第二条命令的图像来自上面的真实 decision fixture。

```bash
.venv/bin/python tools/oracle/vela3/vision_request_oracle.py \
  --root models/vela3-fast-vision --request tools/oracle/vela3/requests/fast-vision-text.json \
  --out models/vela3-vision-public-text.json
.venv/bin/python tools/oracle/vela3/vision_request_oracle.py \
  --root models/vela3-fast-vision --request tools/oracle/vela3/requests/fast-vision-mixed.json \
  --image models/vela3-vision-decision/image.png --out models/vela3-vision-public-mixed.json
for variant in text mixed; do
  RMI_VELA3_PUBLIC_FIXTURE="models/vela3-vision-public-$variant.json" \
    cargo +1.98.1 test --locked --profile release-fast --lib \
    models::vela3::vision::tests::public_system_one_matches_official_wrapper -- --exact --ignored
  target/release-fast/rust-model-inference --model models/vela3-fast-vision \
    --system-one-request "models/vela3-vision-public-$variant.request.json" --threads 4 \
    > "models/vela3-vision-public-$variant.native.json"
  .venv/bin/python tools/oracle/vela3/compare_requests.py \
    "models/vela3-vision-public-$variant.json" "models/vela3-vision-public-$variant.native.json" \
    --atol 5e-4 --rtol 5e-4 --out "models/vela3-vision-public-$variant.comparison.json"
done
```

12 问无图请求覆盖普通 decision/noul、共享 router、set、domain/feedback、NER、描述标签、evidence、halu、PII 和缺图 box；9 问混合请求覆盖有图 router choice 走视觉网络、set/开放标签走文本，以及图像 halu/OCR/box。零阈值 span 用于验证非空输出和 Unicode 边界，不代表抽取质量。2026-11-01 的实测与回归补记在 verification.json 的 `review_fixes_2026_11_01`，不替换下方历史测量。

## d3 请求、任务头和完整计算图

```bash
.venv/bin/python tools/oracle/vela3/head_oracle.py \
  --adapters models/vela3-lite models/vela3-mini models/vela3-flash models/vela3-turbo \
  --out models/vela3-head-fixture.json
RMI_VELA3_HEAD_FIXTURE=models/vela3-head-fixture.json \
  cargo +1.98.1 test --profile release-fast --lib real_heads_match -- --ignored --nocapture

.venv/bin/python tools/oracle/vela3/request_oracle.py \
  --model models/d3-lite --adapter models/vela3-lite \
  --request tools/oracle/vela3/requests/lite.json --out models/vela3-request-oracle.json
RMI_VELA3_REQUEST_FIXTURE=models/vela3-request-oracle.json \
  cargo +1.98.1 test --profile release-fast --lib official_request_token_ids -- --ignored --nocapture
cargo +1.98.1 run --profile release-fast --bin rust-model-inference -- \
  --model models/d3-lite --adapter models/vela3-lite \
  --system-one-request tools/oracle/vela3/requests/lite.json --threads 4 > models/vela3-request-native.json
.venv/bin/python tools/oracle/vela3/compare_requests.py \
  models/vela3-request-oracle.json models/vela3-request-native.json \
  --atol 0.01 --rtol 0.01 --out models/vela3-request-comparison.json
```

把请求换成 `requests/unicode.json` 可复现中文问题；mini 使用 `requests/mini.json`、`models/vela3-mini` / `models/d3-mini`，Oracle 和 CLI 均用 `--threads 4`。mini 的 F32 和四步 causal 复现沿用下方 suite 命令，替换对应模型目录即可。`compare_requests.py` 精确比较字段、选项、整数位置与 usage，仅容忍浮点误差；错误诊断文字可不同，错误码仍须一致。

逐层 trace 需要 `parity-trace`。下面自动取得实际 lib-test 可执行文件路径，避免依赖 Cargo 生成的文件名。

```bash
cargo +1.98.1 test --profile release-fast --lib --features parity-trace \
  --no-run --message-format=json > models/vela3-test-build.jsonl
vela3_test_bin=$(.venv/bin/python -c 'import json; print(next(r["executable"] for line in open("models/vela3-test-build.jsonl") if (r := json.loads(line)).get("executable") and r["target"]["kind"] == ["lib"]))')

.venv/bin/python tools/oracle/vela3/d3_oracle.py suite \
  --native "$vela3_test_bin" --model models/d3-tiny --adapter models/vela3-lite \
  --synthetic --dtype float32 --causal --generation-steps 4 \
  --atol 2e-6 --rtol 1e-6 --out models/vela3-d3-tiny-causal

.venv/bin/python tools/oracle/vela3/d3_oracle.py suite \
  --native "$vela3_test_bin" --model models/d3-tiny-prefix --adapter models/vela3-turbo \
  --synthetic --dtype float32 --state-tokens 2 --adapters \
  --atol 2e-6 --rtol 1e-6 --out models/vela3-d3-tiny-prefix

.venv/bin/python tools/oracle/vela3/d3_oracle.py suite \
  --native "$vela3_test_bin" --model models/d3-lite --adapter models/vela3-lite \
  --dtype bfloat16 --causal --generation-steps 4 \
  --out models/vela3-d3-lite-causal

.venv/bin/python tools/oracle/vela3/d3_oracle.py suite \
  --native "$vela3_test_bin" --model models/d3-lite --adapter models/vela3-lite \
  --dtype float32 --adapters span decision \
  --out models/vela3-d3-lite-f32

.venv/bin/python tools/oracle/vela3/d3_oracle.py suite \
  --native "$vela3_test_bin" --model models/d3-flash --adapter models/vela3-flash \
  --text 'Alice works at Acme.' --dtype float32 --out models/vela3-d3-flash-f32
.venv/bin/python tools/oracle/vela3/d3_oracle.py suite \
  --native "$vela3_test_bin" --model models/d3-flash --adapter models/vela3-flash \
  --text 'Alice works at Acme.' --dtype bfloat16 --causal --generation-steps 4 \
  --out models/vela3-d3-flash-causal

.venv/bin/python tools/oracle/vela3/stream_f32.py --self-test
.venv/bin/python tools/oracle/vela3/d3_oracle.py suite \
  --native "$vela3_test_bin" --model models/d3-turbo/pinned --adapter models/vela3-turbo \
  --text 'Alice works at Acme.' --dtype float32 --stream-f32-weights \
  --state-tokens 2 --adapters span decision \
  --out models/vela3-d3-turbo-f32

.venv/bin/python tools/oracle/vela3/d3_oracle.py suite \
  --native "$vela3_test_bin" --model models/d3-turbo/pinned --adapter models/vela3-turbo \
  --text 'Alice works at Acme.' --dtype bfloat16 --state-tokens 2 \
  --adapters span decision --out models/vela3-d3-turbo-bf16
```

turbo 的完整八问请求使用同一固定底座和请求文件，Oracle 与 Native 均为八线程：

```bash
.venv/bin/python tools/oracle/vela3/request_oracle.py \
  --model models/d3-turbo/pinned --adapter models/vela3-turbo \
  --request tools/oracle/vela3/requests/turbo.json --threads 8 \
  --out models/vela3-turbo-request-oracle.json
cargo +1.98.1 run --profile release-fast --bin rust-model-inference -- \
  --model models/d3-turbo/pinned --adapter models/vela3-turbo \
  --system-one-request tools/oracle/vela3/requests/turbo.json --threads 8 \
  > models/vela3-turbo-request-native.json
.venv/bin/python tools/oracle/vela3/compare_requests.py \
  models/vela3-turbo-request-oracle.json models/vela3-turbo-request-native.json \
  --atol 0.01 --rtol 0.01 --out models/vela3-turbo-request-comparison.json
```

turbo 底座未包含 LM head，且 embedding 未绑定输出权重，官方定义 writer 为 `names`；真实 turbo 不执行 causal greedy 验证。Oracle 在加载前拒绝无 tied/grafted head 的 causal 请求，避免把输入 embedding 当作输出头。

`--stream-f32-weights` 用于内存不足以同时保存整模 F32 权重时的短文本诊断。先检查每个源张量都是 BF16，再逐层临时转换成精确的 F32 权重值，计算后恢复原对象；不支持图像或 KV cache。小型官方图的三种 mask 与 LoRA 开关共 48 个 checkpoint 均与整模 F32 结果精确一致，真实 lite 的 75 个 checkpoint 也与原整模 F32 Oracle 的原始位一致。这些检查验证 Oracle 的存储方式，不代表 Rust 与 Python 逐位对齐。

Checkpoint Oracle 检查名称、顺序、shape、次数、Token IDs、mRoPE 和全部数值，并报告第一个不满足门槛的位置。`float32` 诊断保持真实 BF16 权重值、只关闭激活舍入；不另存一份整模 F32 权重。BF16 文本使用保存于报告中的逐层误差包络，logits 另外要求相同 argmax。flash 的 causal 命令会因内部 BF16 门槛失败而非零退出，报告仍会记录四步 logits 和精确 token IDs。`suite --images image.png --vision-detail` 可重现图像路径差异，该检查目前失败。flash 公开请求使用上面的 lite 请求命令，将两个模型目录和请求分别改为 `models/vela3-flash`、`models/d3-flash`、`requests/flash.json`，Oracle 与 CLI 均使用 `--threads 1`。

## 实际验证范围

- lite：14 个英文问题和 2 个中文问题的选项、非空 span 边界、usage 精确一致，浮点最大绝对误差分别为 `0.004634` / `0.004669`。四步 greedy IDs 与官方均为 `[2838, 682, 264, 220]`。
- d3 小型完整图：F32 最大误差 `4.77e-7`，BF16 最大误差 `0.00390625`；四个 d3 包共 15 个真实 span/NER/relation/image-halu 任务头最大误差 `6.68e-6`。
- mini：固定 `d3-mini@566da0ea` 的八问完整公开请求通过，覆盖六种文本 LoRA、自动定义和三个非空 Unicode span，usage 精确为 `571/0`，19 个浮点值最大误差 `0.007682`。base/span/decision 的 F32 诊断共 297 个 checkpoint 全部通过，最大绝对误差 `0.000286`；BF16 短图 286/297 通过、11 个内部激活最大绝对误差超限（最高 `1.0625`），最终 logits 全通过。四步 causal 图 396/396 通过，greedy IDs 均为 `[2838, 369, 264, 220]`。零阈值 span 只验证解码与偏移，不代表抽取质量。
- turbo prefix-bidir 小图：固定 runtime 源码 SHA 与镜像一致。额外使用真实 turbo 的 `6:1` GQA、`3:1` delta-head 分组验证 base/LoRA，F32 最大误差 `3.58e-7`，BF16 最大误差 `0.00390625`。
- turbo：固定 `d3-turbo@da17b5d0` 的全部 11 分片、readout 和 8 个任务文件均完成 SHA256 校验。八问完整公开请求覆盖五种文本 LoRA 与四个非空 Unicode span，选项、边界、路由和 usage `510/0` 精确一致，18 个浮点值最大误差 `0.005646`。base/span/decision 的 F32 诊断共 585 个 checkpoint 全部通过 `atol=5e-4, rtol=1e-4`，最终 readout 最大误差 `1.1e-05`。BF16 短图 440/585 通过、145 项超限（最大绝对误差 `16`）；其中 span LoRA 的诊断 decision readout argmax 为官方 `0`、Native `1`，其余两个 readout argmax 一致。该 readout 不是公开 span 请求使用的 SpanHeadV2；公开 span 响应按上面的完整请求另行比较。门槛未放宽。
- lite 真实短序列 F32 激活诊断：所有 checkpoint 满足 `atol=5e-4, rtol=1e-4`。BF16 四步 logits 最大误差 `0.0703125`，各步 argmax 精确一致。
- fast：真实九适配器、18 问完整响应及公开 CLI 均通过，usage 精确为 `1651/0`。七组决策 hidden 最大误差 `8.77e-4`、logits `3.40e-6`；CLI 的 52 个浮点值最大误差 `1.09e-6`，选项、span、路由、usage 精确一致。额外六问输入边界回归已验证 RED→GREEN。
- fast-vision：四条真实任务各 84 个模型 checkpoint、2 项预处理比较和完整响应通过，task logits 最大误差 `8.35e-6`。Box/point 坐标精确一致，confidence 差约 `5.96e-7`。OCR 使用零阈值确保比较非空 span。真实 raw 底座完整 hidden max `0.247871`、NRMSE `0.001068`，采用明确的缩放门槛；小图仍使用严格容差。
- flash：真实三问公开请求的 choice、noul、两个非空 NER span 和 usage 一致，浮点最大误差 `0.003435`；grafted head 的四步 greedy IDs 均为 `[2838, 369, 264, 4314]`。base/span/decision 的 F32 激活诊断共 297 个 checkpoint 均通过 `atol=5e-4, rtol=1e-4`，最大绝对误差分别 `0.000488/0.000854/0.000977`。BF16 短 trace 三条路由的 logits 满足既定容差且 argmax 一致，但内部 activation 最大误差 `1.5/2.5/4.0` 超过固定门槛，仍记为失败；差异集中在稀疏的大幅值 channel，尚未隔离具体算子。
- d3 图像：75 个展开 tokens 和三轴位置精确一致。F32 视觉塔相对官方 F32 投影 max `0.005106`、RMSE `0.000204`；对官方默认 BF16 塔则扩大为最终 hidden max `2.03125`、RMSE `0.10997`、logits max `0.08461`。样本 argmax 一致，但 **BF16 逐层门槛未通过**，未放宽门槛将其记为通过。

mini/turbo 的当前 ModelScope 底座与 Vela 固定版本不一致，不能用当前镜像冒充固定 checkpoint。两者均按授权使用官方固定 HF 权重，通过官方 HF 接口或 hf-mirror 定位固定版本的官方 CDN 地址，实际数据从官方 CDN 传输并校验完整 SHA；执行的 d3 源码 SHA 与固定 manifest 匹配，未使用目录中的新版 wrapper。mini/flash/turbo 的非因果 BF16 内部激活门槛以及 d3 图像门槛仍未通过；真实请求与 F32 诊断的通过结论不能替代这些失败记录。该早期记录未执行 GPU、GGUF、HTTP server、视频、长上下文性能、跨平台与完整公开基准；2026-11-06 补充的 GPU 范围见 [GPU 实测记录](gpu-2026-11-06.md)。


## 2026-11-05 review 修复复核

[逐项记录](review-2026-11-05.md) 区分已修复缺陷、官方契约和未验证边界。运行时直接使用 snapshots.json 的固定哈希校验 d3 底座与视觉任务配对；没有跳过校验的生产开关。归一化切到仓库算子后仍沿用浮点容差，不要求有损路径逐位相同。

新增宽高比预处理 Oracle 不加载权重，继续执行固定版本官方处理器，覆盖 6 种宽高比 × 2 个 edge：

```bash
.venv/bin/python tools/oracle/vela3/layout_oracle.py \
  --root models/vela3-fast-vision --out .codex/vela3/layouts
RMI_VELA3_LAYOUT_FIXTURE=.codex/vela3/layouts \
  cargo +1.98.1 test --locked --profile release-fast --lib \
  models::vela3::vision::tests::extreme_aspect_preprocessing_matches_official_processor \
  -- --exact --ignored --nocapture
```

输入 IDs、grid、路由、usage 精确检查；预处理像素 `atol=1e-6`；任务 logits 和请求概率沿用原记录容差。fast hidden 改用峰值与 RMSE 双门槛：`max_abs <= 1e-3 + 1e-5*ref_peak` 且 `RMSE <= 1e-5 + 1e-4*ref_RMS`。这替代原先单独的 `max_abs < 1e-3`，原因和实测差异写在逐项记录中；BF16 门槛未变。单一主机的 SIMD/标量及 1/4 线程检查不构成新硬件平台或 32K 吞吐证明。
