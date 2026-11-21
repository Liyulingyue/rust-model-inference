# TODO — RustModelInference Roadmap

This document merges the legacy `docs/TODO.md` (deep-dive format with
TODO-001…TODO-018) and the roadmap-style `docs/develop/TODO.md`
(checklist of upcoming work). The bottom half carries the detailed
investigation notes; the top half carries the at-a-glance priority list.

CPU SIMD capability assumed throughout: see [SIMD extension inventory](#simd-extension-inventory)
at the bottom of this file (AVX2 + FMA + F16C + AVX-VNNI on Intel Core Ultra 5 125H;
NEON on aarch64; no AVX-512).

---

## High Priority

- [ ] **YuE2 的 AR/NAR 设备开销与融合（批提交 / 避免 host 往返）** —
      2026-11-12 实测：VAE 已上设备（77.0 s → 25.6 s，3.0x，输出差 ≤1 个 PCM16 档位），
      整链 277.9 s 对 CPU 315.6 s（-12%）。但 AR/NAR 仍慢：ABC 74.8 s、semantic 113.6 s、
      NAR 63.9 s。原因：AR 逐 token 调用一次 lm_head（rows=1 的 GEMV，设备 ~70 ms vs CPU
      ~8 ms，已用 rows>=4 门禁挡掉），NAR 每步两次全模型前向（每次 28 层投影，rows=256
      时设备与 CPU 持平）。下一步按review 提示做**融合 / 批提交 / 避免 host↔device 往返**：
      把每层的多个投影合成一次提交（共享 linear executor 已有 `run_group`），以及把
      NAR 的 prefix+velocity 两次前向的合成一次提交；设备侧 GEMV 的效率也需要单独看
      （每 workgroup 64 输出通道 × 32 帧的 tiling 对 rows=1 是否合适）。
- [ ] **HTTP 带图对话的端到端验证（需要可配对的文本塔 + mmproj）** —
      2026-11-20：HTTP 多模态图片输入的实现已齐，但**真模型带图对话**还没在本机跑通——本地只有
      MiniCPM-V 4.7 的 mmproj（1B 文本塔缺失）和 Qwen3.5-2B 的文本塔（mmproj 缺失），
      没有可配对的图文组合。跑通方式：准备 `RMI_AGREEMENT_MODEL`（Qwen3.5 / Qwen3-VL /
      Qwen2-VL 的文本塔）+ `RMI_AGREEMENT_MMPROJ`（配套投影器），然后
      `cargo test --profile release-fast --test cli_http_agreement`，看
      `cli_and_http_agree_on_image_input` 这一档（CLI `--mmproj --image` 与 HTTP 带图
      请求逐 token 对齐）。拿到模型后补一条实测结论，并确认 HTTP 的
      `enable_thinking` / 工具调用在带图时也保持一致。
- [x] **Q2_K / Q3_K SIMD 加速** — 2026-11-21 完成（分支 `feat/q2k-q3k-avx2-simd`）。按 pinned llama.cpp
      `ggml_vec_dot_q2_K_q8_K` / `ggml_vec_dot_q3_K_q8_K` AVX2 结构重写 `avx2_k.rs` 两个 kernel：
      全 SIMD 2-bit 解包（`srli_epi16` + mask）+ `maddubs`（u8×i8→i16 对和）+ scale 经
      `get_scale_shuffle_q3k` 同款 shuffle 内嵌 `madd`，每 super-block 一个 i32 累加器。
      旧实现的瓶颈是 Q2_K 标量解包循环与两者每 16 元素一次的窄水平归约。
      Kernel 级（512 行 × 4096 列 microbench，`examples/bench_kquant_dot.rs`）：Q2_K **7.5×**、
      Q3_K **2.75×** vs scalar。端到端 Qwen3-0.6B（5 批中位）：Q2_K prefill 59→111 t/s、gen 42→65 t/s；
      Q3_K_M prefill 74→103 t/s、gen 50→59 t/s。parity：随机 64 super-block scalar↔AVX2 相对差
      ≤ 2.1e-6（f32 舍入级），新增 `q2k/q3k_avx2_matches_scalar_random_superblocks` 测试。
- [x] **IQ2_XS / IQ3_S / IQ2_S scalar forward_prequantized stub 修复 + 生产路径 SIMD**（选项 1、2 均已做）—
      选项 1（2026-11-21）：`src/ops/kernel/iq4_xs.rs` 的 `iq_kernel_impl!` macro 与 IQ4_XS /
      IQ2_XS 手写 impl 的 `forward_prequantized` 已从 stub（写 0）改为完整 dequant-to-f32 +
      dot 路径（与 IQ4_NL / Q4_K 同款）。生产仍走 `uses_q8_k()` → `forward_prepared`（Q8K
      路径），此路径只为接口完整性。8 个类型各一个 2-row 测试，与 `dequantize_row_*`
      参考逐位一致。
      选项 2（2026-11-21，原判断"暂无法做精度验证"已过时）：生产的 Q8K 热路径
      `vec_dot_iq*_q8k` 里 IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ1_M / IQ1_S
      七个类型当时全是 scalar——pinned llama.cpp `b96806d9` 的 x86 arch 里**已有**这七个
      的 AVX2 kernel，全部按其结构移植到 `src/ops/quant/avx2_k.rs` 并接通 dispatch。
      验证：① llama.cpp 逐位 oracle（`oracle_vecdot_harness` 链接 pinned 构建的
      libggml-cpu；注意必须先 `ggml_cpu_init()` 填 FP16 查找表）对随机 64 super-block
      输入，7 个类型 AVX2 输出与 llama.cpp **逐位一致**（`RMI_LLAMA_ORACLE=<harness>
      cargo test --lib iq_avx2_matches_llama_cpp_oracle_bitwise -- --ignored`）；
      ② scalar↔AVX2 parity（随机 super-block + 7 个本地 IQ 模型整行，rel < 1e-3）；
      ③ kernel 级基准（release，attn_q 1024 值/次）：iq2_xxs 2.60x、iq2_xs 7.21x、
      iq2_s 2.73x、iq3_xxs 6.01x、iq3_s 3.36x、iq1_m 3.42x、iq1_s 3.96x；
      ④ E2E：IQ2_S / IQ3_S greedy 生成正确（"The capital of France is Paris"）。
      **顺带修了 scalar `vec_dot_iq2_xs_q8k_scalar` 只处理半行的 bug**（16/32 个 u16，
      对照 llama.cpp generic 修正；此前无模型可验，从未被发现）。
      **模型资产**：`models/Qwen3-0.6B-GGUF/` 补齐 IQ2_XS / IQ2_S / IQ3_S——ModelScope
      （unsloth / bartowski 镜像）均无这三个文件，用 pinned llama-quantize 自 BF16 +
      unsloth imatrix 本地生成；SHA256：IQ2_XS `0587d6121f5401f8018f43309235590b0444e4
      af73a8bceca96722ddfdf1f109`、IQ2_S `ce788484aff33ee76e846bc1a52d015d30f05e27637
      d3fd30249a3ee4eb35925`、IQ3_S `fac96f9f3fe11987acd69c83294461f0176a259b1d78506
      ffa60bc290d9ea012`。已知边界：≤2.3 bpw 的 UD-IQ2_XXS / UD-IQ1_S / UD-IQ1_M /
      本地 IQ2_XS 在 0.6B 模型上 greedy 输出退化为乱码——**llama.cpp 同权重同 prompt
      同样乱码**（llama-cli b96806d9 实测：IQ2_XS "ention =="、UD-IQ2_XXS "-- exam.
      informat--"，而 IQ2_S / IQ3_S 正常），且 kernel 层已逐位对拍一致 → 是量化质量
      本身（2.0-2.3 bpw @ 0.6B 参数量），不是 kernel 问题；≥2.5 bpw 的 IQ2_S / IQ3_S
      正常（"The capital of France is Paris"）。
- [ ] **AVX-VNNI int8 dot 加速**（前提已修正：真正的目标是 **AVX-VNNI-INT8 的
      `_mm256_dpbssd_epi32`**，待做）— 详见 [TODO-AVX-VNNI](#todo-avx-vnni-int8-dot-加速)。
      原条目的两处前提错误已查证并实测（2026-11-21，Meteor Lake dev box）：
      ① `_mm256_dpbssd_epi32` 不是 AVX-VNNI，是 **AVX-VNNI-INT8**（CPUID `avxvnniint8`），
      本机只有 plain AVX-VNNI，按原计划实现直接 SIGILL（gdb 定位到 `vpdpbssd`）；
      ② plain AVX-VNNI 只有 `vpdpbusd`（unsigned×signed），去不掉 sign fixup，实测比
      AVX2 kernel 慢 2.2-2.4× → **`vpdpbusd` 这一层不做**（参考实现保留在
      `src/ops/kernel/q8_0/avxvnni.rs` 作测量记录 + bit-exact 测试基线，dispatch 未接线）。
      **待做**：`avxvnniint8` 路径（`vpdpbssd`，signed×signed 直接点积，去 2 条 sign fixup、
      horizontal add 缩 1 条、无 -128 边角）。目标机型：Arrow Lake / Lunar Lake（消费端、
      无 AVX-512，它是那里唯一可用的单指令 int8 dot）、Granite Rapids / Zen 5（那里更优的
      是 zmm 版 `_mm512_dpbssd_epi32`）。前置：需要一台 `avxvnniint8` 机器实跑 parity
      测试（复用 `avxvnni.rs` 的 3 测试模板），本机无法验证，不能盲接 dispatch。
- [ ] **IQ2/IQ3/IQ1 × Q8_K 的 NEON kernel（aarch64）** — 2026-11-21 的 AVX2 移植只覆盖
      x86；aarch64 上这七个类型仍走 scalar（`neon_k.rs` 目前只有 iq4_nl 的 NEON）。
      pinned llama.cpp `b96806d9` 的 `ggml-cpu/arch/arm/quants.c` 有全部七个的 NEON
      kernel（iq2_xxs / iq2_xs / iq2_s / iq3_xxs / iq3_s / iq1_s / iq1_m，gate 在
      `__ARM_FEATURE_DOTPROD`），按 AVX2 同款流程移植：结构对齐 + oracle 对拍。
      **前置**：本机是 x86，NEON kernel 必须在 aarch64 机器上开发验证（含
      `iq_avx2_matches_llama_cpp_oracle_bitwise` 的 ARM 版 oracle harness）；仓库当前
      无 aarch64 CI 作业（见下方"aarch64 CI"条目），接入前先补交叉编译或 CI 门禁，
      不盲接 dispatch。
- [ ] **IQ2/IQ3/IQ1 的 GPU（Vulkan）路径** — 仓库 Vulkan 侧目前只有 Q8_0 / Q4_K /
      （BF16 等）matmul，IQ 类型完全走 CPU（`src/vulkan/` 无任何 IQ shader）。
      pinned llama.cpp 有现成参考：`ggml/src/ggml-vulkan/vulkan-shaders/` 下七个类型的
      `mul_mat_vec_iq*.comp` + `dequant_iq*.comp`（iq1_m / iq1_s / iq2_s / iq2_xs /
      iq2_xxs / iq3_s / iq3_xxs）。移植时注意 llama.cpp 的 shader 是
      mul-mat-vec（per-token 解码侧）形态，与仓库现有 `matmul_q4k.rs` 的 prefill 形态
      不同，需决定先补 decode 侧还是两侧都要；数值验收照仓库惯例对 CPU 路径比
      （Vulkan IQ shader 一旦上线，`PreparedRows::matmul_group` 的 `uses_q8_k()` 权重
      会开始走 GPU，需与本次逐位对拍过的 AVX2 路径做同机对照）。验证设备：本机若无
      Vulkan 设备则按 `amd-cloud-gpu-test` skill（用户点名才用）。
- [x] **Q4_0 AVX2 kernel 重写：逐 lane FMA 累加 + 4 行分块**（2.07×，结构对齐 llama.cpp AVX2）— 见 [TODO-001](#todo-001-q4_0-avx2-kernel-不使用-fma性能受限)。
- [ ] **token 维寄存器分块：消除 per-token 权重重读** — 详见 [TODO-019](#todo-019-token-维寄存器分块消除-per-token-权重重读)。
      文本编码只跑到约 11% 的 NEON 峰值，瓶颈是每个 token 都从 DRAM 重读整个权重矩阵。
      预期 1.5-2× 加速。
- [ ] **AVX2 的 `exp_approx` clamp 与 shader 的 ±80、与上游的无 clamp 结构三者不一致** —
      x86 CPU 路径用 clamp（`src/ops/math/exp.rs` 的 `FAST_EXP_CLAMP`，现为 ±88.0），
      ARM NEON 路径用的是上游 llama.cpp 的结构（`|n| > 126` 走慢路径，**没有 clamp**），
      Vulkan shader 用 ±80（补偿算法需要 `1/den` 保持正规数，见
      [TODO-020](#todo-020-vulkan-silu-的-f32-次正规数缺陷---已修复但饱和边界不可回退)）。
      三者的饱和边界因此不同：`x = 88.2` 时 NEON 走慢路径给出接近真值的 2.4e38，
      x86 给 1.65e38（31% 偏差），GPU 给 exp(80) 的饱和值。
      现状：x86 至少不再返回 `+inf`（旧 clamp `88.376_26` 会），并且
      `FAST_EXP_CLAMP * log2(e) <= 127` 这个不变式由
      `fast_exp_clamp_keeps_the_exponent_inside_the_f32_exponent_field` 守住。
      选项：① 把 AVX2 改成镜像上游的 `|n| > 126 → 慢路径`（最彻底，x86 与 ARM 对齐，
      但需要 x86 机器实测）；② 把 CPU clamp 收到 ±80 与 shader 对齐（牺牲 [80, 88.4]
      的精度，换 CPU↔GPU 一致）；③ 维持现状，把三者的边界写进文档。
       **必须在 x86 上实测才能定，不能在 aarch64 上单方面决定。**
- [ ] **两份 F16 shader 改动缺 aarch64 / 真实权重验证（Timestral F16 路径 + NEON 回退）** —
      合并 `e0d01a7` 的解算对 `shaders/glsl/f16_matmul_tiled.comp` 加了
      `transposed = (group_count != CONV_GROUP) && (q8_stride == 2)` 分派（同时保住
      Timestral 的转置权重与 Z-Image conv / 行主序 F16 的行主序权重），并把
      `f16_matmul.comp` 的 NEON「连续 64 元素块」加载**回退**成 stride-16。
      两条路径本机都**没能验证**：
      - **Timestral F16**：`src/vulkan/timestral.rs:226` 走 `record_transposed_float_matmul_rows`
        （F16 置 `q8_stride=2`）→ 命中 transposed 分支。公式与 main `f4aa9d1`(#198) 修好
        Timestral 时用的完全一致，属**代码推断**；本机没有 Timestral 权重，未实测。
      - **`f16_matmul.comp` 的 NEON 回退**：模式 2 只在 aarch64 NEON 走
        （`attention_dot_mode()` 中 x86 走模式 8），本机是 x86 且 **CI 无 aarch64 作业** →
        该分支**从未被编译或测试**，只能确认「回到 `afed6d4`(#192) 之前的行为」。
      已实证的部分：conv 侧 main 偏差 **9.01** → 本分支 **~1e-6**（`RUST_GPU_MIN_OUTPUTS=0`
      下 `vulkan_vae_convolution_matches_cpu_across_tiles` 通过）；行主序 F16 tiled 由
      `batched_linear_device_f16_tiled_preserves_storage_layouts` 覆盖。
      选项：① 加 aarch64 CI 作业（至少 `cargo check --features vulkan` + shader 编译）；
      ② 在 arm 机器上跑 Timestral F16（209M）的 GPU 路径并对 llama.cpp oracle；
      ③ 把 #198 的 conv 回归与本分支的 `transposed` 分派反馈给 main。
- [ ] **parity-trace 路径在 GPU 上自死锁（`RMI_PARITY_TRACE` 一开就永久挂起）** — 2026-11-09 定位；
      **main 与本分支都有**（共享代码 `src/vulkan/qwen35.rs`），不是某条分支引入的。
      **现象**：编译了 `parity-trace` feature 且设置了 `RMI_PARITY_TRACE`（**空字符串也算** ——
      判定用的是 `var_os().is_some()`）时，Vulkan 推理在第一个 chunk 的 trace 点永久挂起：
      所有线程停在 futex、**GPU 利用率 0%**、显存已正常上传 → 纯宿主端锁死，**不是 GPU 挂死**。
      **根因**：`TokenCommands` 持有 `context.mutex` 的 guard；`submit_and_wait()` 消费 `self`
      因而释放锁，而 `flush()` **保留** guard（只提交并重开命令缓冲）。`forward_chunk_inner` 里
      `commands = commands.flush()?` 之后紧接 `trace_rows()` / `trace_region()` →
      `Qwen3Ops::read_f32_into()` → `TokenCommands::begin()` 再抢同一把
      **`std::sync::Mutex`（不可重入）** → 同线程自死锁。
      **证据**：gdb 栈 `Mutex::lock_contended` ← `TokenCommands::begin` ← `read_f32_into` ←
      `trace_rows` ← `forward_chunk`；`/proc/<pid>/task/*/wchan` 全为 `futex_wait_queue`；
      main 上是同一模式（`src/vulkan/qwen35.rs` 的 `flush()` + `read_f32()` 相邻）。
      **对照**：不带 `parity-trace` feature 的构建同权重同卡**完整跑通**（SenseNova Loop 9B
      Q4_0 在 L20 上输出 greedy tokens），独立 `vk_ops_check` 全过 → GPU 路径本身可用，
      此前「GPU 路径挂死」的结论是这条死锁造成的假象。
      **修法选项**：① trace 前先释放 `commands`（或改 `flush()` 语义让它交还 guard）；
      ② 读回改走不持锁的路径；③ `parity_trace::enabled()` 改用 `var()` + 非空判断，
      避免空串误开。
      **影响面**：所有 `qwen35` / sensenova GPU 会话的 parity 比对 —— 即**当前无法用 parity trace
      定位 GPU 与 CPU 的数值差异**（见下条）。
- [ ] **sensenova GPU 与 CPU 的 greedy 输出不一致** — 2026-11-09，SenseNova Loop 9B Q4_0
      （同一批 oracle fixture）：case0 GPU `[9419,0,2500,628,353,1438,488,3242]` vs CPU
      `[9419,0,2500,513,488,3604,3242,30]`；case1 一致 `[109430]`；case2 GPU `40` vs CPU `34`。
      GPU 侧激活走 Q8_0 int8，误差经 mHC 路由放大后 argmax 翻转。**需要 parity trace 逐层
      定位，但当前被上一条死锁阻塞**；先修死锁，再复跑逐层比对。
- [ ] **F32 权重的字节序防线要从运行时 cfg 变成编译期分支** — `src/ops/kernel/f32/mod.rs` 的
      `from_le_bytes` 用 `cfg!(target_endian = "little")` 决定是否把 GGUF 字节重解释成 `&[f32]`；
      而 `bytemuck::try_cast_slice` **只校验长度和对齐、不看字节序**，所以那个 `cfg!` 是**唯一**的
      字节序防线 —— 被重构误删后，BE 平台会**静默**产出字节序颠倒的浮点数（不崩溃、只是数值全错，
      且只有 BE 机器能暴露）。建议拆成 `#[cfg(target_endian = "little")]`（借用）与
      `#[cfg(target_endian = "big")]`（恒拷贝）两份实现，让守卫无法被误删。
      顺带：目前对齐失败是**静默**落回 `Cow::Owned`（整份 F32 权重进 RAM）。GGUF 规范
      `general.alignment` 默认 32、我们的写器也保证了，但第三方 GGUF 不保证 → 建议加载期校验张量
      偏移的 4 字节对齐，不满足就报错（至少告警一次），别偷偷吃掉 35 GB。
      **数值一致性现状已 OK**：BE 走 owned 时用 `f32::from_le_bytes`，与 LE 位级相同；
      `borrowed_f32_preserves_owned_kernel_bits_without_copying` 已断言 borrowed 与 owned 的输出
      逐位相同，且已覆盖未对齐回退分支。
- [ ] **F32 内核改成"永远借字节"，与 F16/BF16 同构，删掉 `Cow`** — `F32Kernel<'a>` 是全仓唯一
      带 `Cow` 的权重类型（`Cow<'a, [f32]>`，因为要同时支持"借 GGUF 字节"和"拥有拷贝"），而
      `F16Weight<'a>`/`BF16Weight<'a>`/`F16Kernel<'a>`/`BF16Kernel<'a>` 全都是 `weight: &'a [u8]`
      —— **永远只借**。目标：改成 `pub struct F32Kernel<'a> { pub weight: &'a [u8] }`，转换挪进
      kernel（LE 用 unaligned load reinterpret，AVX2/NEON 原生支持 → 对齐约束一并消失；
      BE 上 byteswap）。
      收益：① **内存画像跨平台完全一致**（永远零拷贝，35 GB 级 F32 权重不再有平台差异）；
      ② `Weight`/`QuantizedTensor` 抽象下 F32 与其它格式同构；③ 没有平台分支泄漏到调用方；
      ④ `Cow` 可整个删掉。代价：AVX2/NEON 的 F32 matmul 签名要从 `&[f32]` 改成 `&[u8]`
      （`avx2_q8::matmul_f32_vs_q8_avx2` 等），BE 热循环多一次 byteswap（BE 平台罕见）。
      **建议先做上一条作为止血，B 单独成 PR**（要动 SIMD kernel 签名，值得独立评审）。
      另注：main 目前**还没有** `Weight::from_f32_bytes`（F32 仍是 `QuantizedTensor::F32 { data:
      Vec<f32> }` 的整份拷贝语义），35 GB 级 F32 模型在 main 上会 OOM → 建议把这条零拷贝线
      （无论最终是 `Cow` 版还是 bytes-only 版）单独拆一个 PR 回 main。
      另注：**跨平台一致性 ≠ 跨架构一致性** —— AVX2 / NEON / scalar 的浮点累加顺序差异是另一个
      话题（见上面「两份 F16 shader 改动缺 aarch64 验证」那条）。
- [ ] **`_ggml` 后缀函数的收口** — 仓库原则是「实现并采用 `dot_f32` 这样的函数，
      尽量不采用 `dot_f32_ggml` 这样的函数，仅当无法绕开时才使用」（浮点加法不满足
      结合律，ggml 的累加结构既不是精度最优也不是速度最优）。当前仓库里已有一批
      `_ggml` 函数，是历次对拍留下的，需要逐条评估能否迁回默认入口。
      **注意**：`q5_0::neon` / `q5_1::neon` 的 NEON kernel **不是**对齐副本，而是
      这两个格式在 aarch64 上的唯一生产实现（被重写成 ggml 结构），所以其他模型也在用；
      收口前必须确认其他模型的既有 oracle 是否仍成立。
      清单、存在理由与收口方向见
      [GGML_PARITY_BOUNDARY.md](GGML_PARITY_BOUNDARY.md)。
- [ ] **复核 10 个同类 oracle 脚本的既有「逐位对齐」结论是否真的跑在标量路径上** —
      `GGML_NATIVE=OFF` 只去掉 `-march=native`，**不改 ggml 自身的默认值**：该 pin 下
      `GGML_CPU_ALL_VARIANTS` 仍为 ON（为每个 kernel 额外编译一份 AVX2 / AVX512 / FMA
      副本并运行时分发），`GGML_AVX2` / `GGML_FMA` / `GGML_F16C` 也仍为 ON —— 得到的
      是一个「看起来能跑」的 SIMD oracle：同一 commit、同一模型、同一 prompt 下 99.3%
      的 checkpoint 逐位不同，token 级结论存活，近平局处翻转。共享脚本
      `tools/oracle/shared/build_llama_oracle.sh` 已补齐全部 ISA 开关并实测定量（见
      [REFERENCE_IMPLEMENTATIONS.md](REFERENCE_IMPLEMENTATIONS.md) 的「本地 checkout
      约定」），但同类脚本仍有 10 个只带 `GGML_NATIVE=OFF`：`mellum` / `qwen35` /
      `qwen_drive` / `gemma4`（两个）/ `ernie_image` / `nanbeige` / `longcat` /
      `jina_bert_v2` / `qwen_image_2_1`。部分结论在 ARM 上取得（CMAKE_C_FLAGS 已带
      `-U__ARM_NEON`，x86 的陷阱不适用），其余应逐个 arch 确认其结论是否依赖浮点位级
      一致性，再决定是否重建 oracle。`tests/inference_parity.rs` 的两个 `#[ignore]`
      位比较测试（`q4_0` / `q4_k_m` `…_matches_pinned_scalar_oracle_bit_for_bit`）若用
      补开关之前构建的 `RMI_LLAMA_ORACLE` 二进制，需重跑。
- [ ] **标量 Q8_K 激活量化器与 ggml 契约不一致（已修，但调用面比想象的大）** — ✅ 已修（2026-10-15）：
      `quantize_row_q8_k_scalar_into` 用的是对称 `d = amax/127` + `f32::round()`，
      而 ggml 的 `quantize_row_q8_K_ref` 是 `iscale = -127/max`（`max` 是**带符号**的最大幅值元素）、
      `d = 1/iscale`、取整用 `nearest_int()`（12582912.0 技巧，**ties-to-even**），钳位是
      `min(127, v)`（只有上界）。负值主导的 block 因此符号相反、`d` 反号，Q6_K 的每个点积都差
      ~2 ULP。仓库里其实**已经**有一份正确实现 `quantize_row_q8_k_ggml_into`（ERNIE 在用），
      标量路径只是没跟上 —— 说明同一契约存在两份实现且会漂移。
      **待办：收敛成单一实现**，让 `quantize_row_q8_k_into` 的标量分支直接走 ggml 版本，
      删掉重复代码；否则下次改动仍可能只改一处。
      守卫：`ops::quant::q8k_padding_tests::scalar_quantizer_matches_the_signed_ggml_contract`。
- [ ] **MoE 权重归一化的累加精度** — ✅ 已修（2026-10-15）：归一化用的 sum 必须像
      `ggml_sum_rows` 一样在 **f64** 中累加，然后转 f32 再做 f32 除法。用 f32 累加会让专家权重
      差 2 ULP，在近平局处直接换专家。`src/models/lfm2moe/trunk/forward.rs` 已修；
      **待查** `mellum`、`qwen3vlmoe` 等其它 MoE trunk 是否也用 f32 累加。

- [ ] **AVX2 kernel 与 ggml 参考实现不逐位一致（已定位到入口，未定位到具体 kernel）** —
      测量：15 个稠密 LFM2 GGUF，Rust 不设 `RMI_SCALAR`（走 AVX2）vs 标量 oracle，**12/15 greedy 一致**；
      `RMI_SCALAR=1` 下是 **15/15**。失配三个的 top-2 间距为 **6.6e-3 ~ 9.6e-3 相对**（比 f32 epsilon 高三四个量级），
      属真实算术差异而非近平局；其中 `LFM2.5-2.6B-Q4_K_M` **第 0 个生成 token 就分歧**。
      **入口已确认**：逐层 trace 显示 `LFM2.5-2.6B-Q4_K_M` step 0 下，`RMI_SCALAR=1` 时
      **396 个 checkpoint 全部零失配**，AVX2 时第一个失配就是 `conv_in_proj`（layer 0，Q4_K × Q8_K 矩阵乘，
      max|delta| 2.5e-07），之后沿层级放大。`attn_norm` 仍然匹配，说明 RMS norm 未受影响。
      **性质**：2.5e-07 是求和顺序不同的 ULP 级差异，不是算法不同。
      **注意**：仓库自带的 `ops::quant::avx2_parity` 单元测试（q4k/q6k AVX2 vs scalar）**全部通过**，
      但那些只用合成的小 block，覆盖不了真实多行 / 真实权重的累加顺序。
      **未办**：尚未定位到具体 kernel。下一步是用真实 GGUF 权重构一个
      针对 `conv_in_proj` 矩阵乘的 scalar/AVX2 对照测试（而不是合成 block），该测试应能直接拍住求和顺序。
      **机制已看懂，但未定位到具体 kernel**：`vec_dot_q4k_q8k_avx2` 每个 block 用
      `_mm256_fmadd_ps(vd, sumi, acc)` 累加（**FMA，单次舍入**），最后走 `hsum_ps` 树形归约；
      标量路径是逐 block `sumf += d * sumi`（**两次舍入**）。两者算的是同一个和，
      但添加顺序不同 —— 与 2.5e-07 的偏差尺度一致。
      **这是性能与精度的取舍，不是 bug**：`LFM2.5-2.6B-Q4_K_M` 同一 prompt 下
      AVX2 输出 `"2"`（25.1 t/s）、scalar 输出 `"2."`（2.0 t/s）——两者都合理，
      但**AVX2 快 12.5 倍**。要让 AVX2 与标量 oracle 逐位一致，就要拉掉 FMA 和树形归约，
      代价是吞吐掉一个数量级。**不建议改**；应该把「逐位对齐」明确限定在
      `RMI_SCALAR=1`，并在文档里写清“默认 AVX2 路径不在逐位契约内”。
      **影响范围**：当前所有逐位结论都只在标量下成立。真实用户在有 AVX2 的机器上跑的是
      这条路径，**与 oracle 存在 ULP 级差异**，并会在近平局处翻转为 token 级分歧。

### JEV 决策评分 follow-ups

`--jev` 已在 `src/app/jev/single.rs::run_jev_decision` 中按 `general.architecture`
自动路由到 10 个 per-arch scorer（Qwen3 / Qwen3.5 / Llama 家族 / Gemma4 /
LFM2 / LFM2.5 / Spark / Nemotron-H / Hunyuan / LFM2-MoE），每个 scorer 实现
`JevScorer` trait。Grouped 模式（MultiSelect + BlockChoice）在
`src/app/jev/grouped.rs::run_jev_grouped_decision` 通过 `JevGroupedScorer` trait。
通用协议与 per-arch chat template 见 [`docs/usage/qwen3.md` §10](../usage/qwen3.md) 与
[`docs/develop/jev.md`](jev.md) §14（源码索引）+ §15（trait 重构）。

- [ ] **JEV Multi-question KV 共享** — 当前多 question 模式（`--jev-question × N`）
      每个 question 都会新建 session 重新 prefill 一遍 context + question 文本，
      耗时 N × prefill_time。KV 共享方案：让 session 支持 `reset_kv(seq_len)`
      把 KV cache 截到 context 之后的长度，对每个 question 只 prefill
      `{"question": ..., "candidates": ...}` 部分复用同一份 context KV。
      预期 N 个 question 总耗时从 `N × t` 降到 `t + (N-1) × t'`，其中 `t'`
      是 question-only 部分。
      适用 trunk：qwen3 / qwen35 / gemma4 / spark（有 Session API）；
      llama / lfm2 / lfm25 / nemotron_h / hunyuan 需要先把 monolith
      拆出 session API。
- [ ] **JEV Shared prefix batching** — 多个 `--jev-question` 不仅共享 prefix，
      还可以把 K 个不同 question 的最后 token batch 成一次前向（拼接
      成 `[prefix; q1_suffix; q2_suffix; ...; qK_suffix]`，attention 阶段
      mask 让 K 个问题互不干扰）。当 K 个 question 文本相似度高时（典型
      客服路由、FAQ 分类场景），可以把 prefill 提速近 K×。
      需要先解决：不同 question 的 last-token 位置索引；padding 到统一
      长度；attention mask 的构造。当前架构（每 question 一个独立 session）
      完全不支持此模式，需要结构化改造。
- [ ] **JEV instruct-tuned GGUF 文档** — 现状：本地 `models/qwen3-0.6b-gguf/`
      只有 base model Qwen3-0.6B（IQ4_NL / Q4_0 / Q8_0 等量化），不是
      Instruct 变体。OpenJEV 锁定 Qwen3-4B-Instruct-2507。要让 JEV 输出
      真正准确的 label，需要：
      1. 找/下载 `Qwen3-0.6B-Instruct` 的 GGUF（huggingface 上有）
      2. 同 §1（用 base model + IQ4_NL 跑通作为 baseline）
      3. 加 `tests/qwen3_instruct_jev_reference.rs` 比对 base vs instruct
- [ ] **JEV 与生成模式共享 KV（暂留）** — 当 `--prompt` 后面接 `--jev` 时，
      可以复用生成模式 prefill 出来的 KV cache，避免重复编码同一段 context。
      当前架构下两者用不同的 session path，需要在 dispatch 入口统一
      KV lifecycle（`KvLifecycle::Shared`）。
- [ ] **JEV Cardinality 扩展（26 → 255）** — 当前 label 固定 A-Z（最多 26 候选）。
      TypeSafe Jev 官方支持最高 255 cardinality。扩展路径：
      1. 候选 ≤ 26 时继续用 A-Z label（现有逻辑不变）
      2. 候选 > 26 时自动切换到数字 label（1, 2, 3, ...），验证每个数字 token
         在 tokenizer 中是单 token（0-9 在所有 BPE tokenizer 中都是单 token；
         10-99 大部分也是；100-255 需要逐个验证）
      3. 更高基数时参考 TypeSafe 的 2-stage 方案：先 score 所有候选（并行），
         再 top-K choose（第二次 forward pass）
      注意：当前 `verify_label_tokens_single` 只校验 A-Z；扩展后需要校验
      实际使用的 label set（数字或字母）。
- [ ] **35B 视觉塔未与 oracle 做数值对照** — 1B 的 `pos_embed` 逐位比过，35B 完全没有。
      projector 代码零改动（两个尺寸共用同一 `minicpmv4_7` 实现，仅 `projection_dim`
      1024 → 2048），但 2048 维输出、`attn_post_norm` 全层 checkpoint 都没验证过。
- [ ] **`qwen35moe` 的 Vulkan 整模数值验收未完成** — 共享路径已接入 GPU expert 投影与
      host 路由/归约。Holo4 在 M3 Max / RADV NAVI31 的 MoE 分层和 batch/reset 检查通过，
      整模 logits 的 0.02 CPU/GPU 门槛仍失败；此 MiniCPM 权重尚未验证 GPU。
      见 [Holo4 验证](../../tools/oracle/holo4/README.md)。
- [ ] **decode 每个投影一次 pool dispatch，784 次里 26 次的开销大于活** — Mellum 12B Q8_0
      在 aarch64 上 44 ms/token 里约 21 ms（48%）是 dispatch 固定开销（每次 ~26 µs：唤醒 7 个
      worker + 同步 + 尾部效应），而每个投影每 worker 的活只有 12–98 µs。线程曲线印证：
      1→8 线程只有 3.17× 加速，16/20 线程直接倒退。把一层里能并行的投影合并进一次
      dispatch（每层 4 次而不是 28 次）→ 44 → 26.5 ms/token（1.66×）。不需要新的 pool
      原语，现有 `compute` 就够（closure 里顺序做多个投影，各自按 `ith/nth` 切分）。
      风险是位序和 DRAM 局部性；收益的绝对值与机器无关，相对值随 CPU 带宽上升。
      详见 [TODO-052](#todo-052-decode-每个投影一次-pool-dispatch784-次里-26-次的开销大于活)。
- [~] **Nemotron-3-Nano 30B-A3B（`nemotron_h_moe`）已可用，剩按类型家族抽样复验** —
      B1–B7 全部完成：Q5_1 dequant kernel、arch/config、MoE 层、共享专家、oracle 对拍
      （全 52 层 × 6 步逐位一致，最终 logits 逐位一致，greedy 输出一致）、CLI / HTTP /
      JEV 三个入口（2026-11-09）与 Vulkan 设备路径（2026-11-10，52 层全部在设备上；
      Q8_0 档设备实测逐位，2026-11-11）。契约从 GGUF 头部读出（不必下完 24 GB），详见
      [TODO-053](#todo-053-nemotron-3-nano-30b-a3bnemotron_h_moe-已支持剩按类型家族抽样复验)
      与 [NEMOTRON3_NANO_30B_MOE.md](NEMOTRON3_NANO_30B_MOE.md)。
      剩余（低优先）：按类型家族抽样 3 档真实权重复验（`Q6_K` / `Q4_0` / `IQ4_XS`）；
      10 个 i-quant / Q2_K / Q3_K 档没有设备 kernel，设备 session 建不起来。
- [ ] **Dory 的默认提示在 CLI / HTTP / 模型自带模板之间三分** —
      CLI 15 token（无 think 标记）、HTTP 19 token（空 think 块）、模型自带模板 22 token
      （多一个空 system 轮），三者贪心输出不同，CLI↔HTTP 分叉且 `cli_http_agreement`
      未覆盖 dory。
      另外 `--compute vulkan --jev` 明明整条前向都在设备上（4.2× 于 CPU、概率一致）却打出
      「no complete device path for this mode」。详见
      [TODO-054](#todo-054-dory-的默认提示在-cli--http--模型自带模板之间三分jev-的backendpartial提示对-dory-是假的)。
- [ ] **Audio8 fused layer（单 submit/层）未完成** — 当前 GPU 模式每层 8 次 submit
       （2 norm + QKV + attention + O + gate_up + SiLU + down），每次 ~0.25ms 固定开销，
       36 层 × 116 步 ≈ 32K 次提交，submit 开销主导了 GPU 模式 13.5s 的总耗时
       （CPU 模式 10.5s 反而更快）。fused layer 的目标是把一整层的 8 个 op 录进同一个
       command buffer，单次 submit。**未实施原因**：
       1. 会破坏 `LinearExecutor` 的 CPU/GPU 自动分发——matmul 必须从 `LinearExecutor`
          抽到 fused path，`ComputePolicy::Auto` 的 per-op 粒度选择变成 per-layer 二元选择；
       2. `BatchedLinearRuntime` 和 `Audio8GpuCompute` 各有独立 `Qwen3Ops` arena，跨 arena
          数据必须经 CPU readback，不共享 arena 无法真正 fused；
       3. 需要大规模重构才能让数据在 GPU 内部流动（残差常驻设备、共享 arena 所有权）。
       **暂缓策略**：先把算子加速做好（F16/BF16 GPU matmul 带宽优化、k-quant shader
       u32 字解包、GPU attention/norm/SiLU 集成已完成），多探索几个模型验证通用性，
       再回头看 fused 问题——届时可能有更清晰的 session 所有权模型可以复用。
       关联文件：`src/models/audio8/gpu.rs`、`src/models/audio8/text.rs`、
       `src/compute/linear.rs`、`src/vulkan/ops.rs`。


## Medium Priority

- [ ] **PhysBrain 文本 logits oracle 的界限不可跨主机迁移** — `tools/oracle/physbrain` 的
      文本对照用固定提示 `What is 2+2?`、8 步 greedy、relative L2 ≤ 0.05 的界限。
      step 0 是**近似平局**（token 17 与 785 相差 0.075 logits），而记录的 0.0488
      已经是界限的 97.7%，所以换一台主机或换一个参考构建就会翻转：实测
      aarch64 + scalar llama.cpp 参考得到 relative L2 0.0677、首个 greedy token 就不同。
      llama.cpp 侧能**精确复现**记录的 ids（`[17,488,220,17,284,220,19,271]`），
      所以参考实现没有问题，是 fixture 的包络太紧。
      现状：main 与本分支**同样不过**，属预存问题，不是回归。
      选项：① 换一个 top-1 差距更大的提示并重录 oracle；② 放宽界限（例如 0.1）并
      如实记录实际值；③ 除 argmax 外同时记录 top-k 命中率，让近似平局不再翻转判定。
- [ ] **Intern-Decision 文本 logits oracle 的界限同样不可跨主机迁移** —
      `tools/oracle/intern_decision` 的文本对照用固定提示 `What is 2+2?`、8 步 greedy、
      relative L2 ≤ 0.03 的界限。记录主机（Apple Silicon）每步 0.0143–0.0295，
      在 aarch64 + NEON 上复测为 0.0156–0.0308：**step 6 恰好越过 0.03**（0.0308），
      其余七步都在界内，8 步 greedy ids 与 llama.cpp 参考**逐 token 相同**。
      与 PhysBrain 那条不同，这里不是近似平局翻转，而是包络本身太贴边。
      选项：① 把界限放宽到 0.035 并如实记录两台主机的实测值；② 换 top-1 差距更大的
      提示重录；③ 记录 top-k 命中率作为辅助判据。
- [ ] **ZDTaichu5.0 视觉塔上 GPU（先打通能力，再做计算优化）** — **第一步已完成（2026-11-05）**：
      塔的矩阵权重改走 `Weight`（保持 GGUF 的 BF16、零拷贝借映射字节，不再反量化成 F32：
      匿名内存 3.3GB→无副本），matmul 先试 `try_vulkan_rows` 再回落共享 kernel → 视觉塔
      （RADIO v4-H）的投影现在**能在设备上跑**，CPU↔设备相对差 ~5e-7。
      验收判据（同一 llama.cpp 参考下 GPU relative L2 ≤ 0.001，CPU↔GPU 相对差 ≤ 1e-3）已达成：
      设备 0.00035911 / 0.00470339 / 0.00147061，CPU 侧数值不变（0.00035960 / 0.00470345 /
      0.00147076，与改动前差 ~1e-7）。
      改动前的现状（留档）：**文本塔**能上 Vulkan，**视觉塔恒在 CPU** ——
      `src/models/zdt/vision/` 把权重读成裸 `Vec<f32>`（`load_f32_tensor` 全部反量化）再用
      `crate::ops::dot_f32` 逐元素算，结构上没有设备路径；CLI 报 `backend=partial mode=mmproj`，
      HTTP 直接拒绝（`--compute vulkan` + `--mmproj` → "full Vulkan runtime supports text
      input only"）。
      实测（aarch64，Q4_K_M 文本 + BF16 mmproj）：**视觉投影**（单独计时）
      CPU 36.93s → 设备 27.62s；文本 40 步 decode CPU 13.5 t/s、Vulkan 6.6 t/s。
      ⚠ 旧记录里"图像 1 token 的 prefill CPU 7.94s / Vulkan 23.03s（视觉塔独占大头）"是
      **误归因**：CLI 报的 `prefill=` 只含文本 prefill，**不含视觉编码**，视觉塔的耗时当时
      没被计入（所以数字看着比实际小一个量级）。
      **第二步（再优化）仍在**：dispatch 开销、批量、权重常驻与上传、文本塔 GPU 效率；
      本机 GPU 对 9B 文本塔慢 2 倍，GPU 的真正收益在批量与长上下文。
      顺带：改用 `Weight` 后 `qwen3::vision::matmul_f32_weight` 已退回私有。
      **第二步（优化）仍在，且第一步不是"激活常驻"**：trace 把 27.6s 全部解释掉了 ——
      `GpuLinear` 的 runtime 缓存按**权重指针**分桶（`src/ops/kernel/vulkan.rs`），每个权重
      都要重建一次管线，实测 **213ms/次 × 131 个权重 = 27.9s**（≈100%）；fence 只有
      2.5ms × 229 = 0.57s，host↔device 传输 2.7GB ≈ 30ms，都不是瓶颈。
      塔的权重只有约 6 种形状 → 把缓存键改成 `(format, n_in, n_out, tile_rows)` 共享，
      创建次数从 131 降到个位数，预期 27.6s → ~2s。详见 **TODO-049**。之后才轮到 fence 批处理
      （`BatchedLinearRuntime::with_output_slots` + `run_group_gpu` 已支持一层的多个 matmul
      进一个 command buffer）与激活常驻。另外本机 GPU 对 9B 文本塔反而慢 2 倍
      （dispatch 开销占比高），批量/长上下文才是 GPU 的强项。
- [ ] **讨论：MemoryArena 与 BlockAllocator 组合**
- [ ] **讨论：GPU 后端架构设计** — Vulkan / wgpu / CUDA 等多后端抽象
- [ ] **讨论：SIMD 扩展路线** — 当前 AVX2+FMA、NEON。后续可考虑 AVX-512 (高端 CPU)、ARM SVE、AVX-VNNI (int8 dot)
- [ ] **GLiNER 特性层补齐** — 12 个模型已 byte-exact，但用户可见**特性**未齐。
      详细清单、划界标准（模型家族特性 vs 框架层）、以及本轮抓到的三个真 bug
      见 [`GLINER_ADAPT_PLAN.md`](GLINER_ADAPT_PLAN.md) 的「特性层待办」一节。
      当前半成品：F-1 `choices` 解码侧（prefix 已进 prompt 流但无人消费，**优先收尾**）、
      F-2 `entity_attributes`（两套机制、零 ground truth）、
      F-3 长文本 `*_long` chunk+merge、F-4 relation 4 阶段 dedup。
- [ ] **讨论：两套线程调度统一** — ComputePool vs rayon。暂不统一（LLM 热路径不应轻易改动）
- [ ] **LFM3 的 GPU 路径只搬了 matmul，剩余算子占 CPU 解码时间 18%** — 详见
      [TODO-045](#todo-045-lfm3-的-gpu-路径只搬了-matmul剩余算子占-cpu-时间-18集显上-auto-比-cpu-慢-84925)。
      现状：GPU 只做线性投影，attention / shortconv / KV 留在 host。测算表明投影占 CPU
      解码约 82%，其余算子合计仅 18%，所以**在集显上继续搬算子没有收益**（实测 Auto
      输出与 CPU 逐字节相同，却慢 8.4–9.25×，瓶颈是权重过内存总线）。真要做应在独显上
      做，优先滑窗 attention、q/k norm+RoPE 合并、RMSNorm 融合、sigmoid head gate，
      shortconv 最后。
- [ ] **ComputePool 的空闲 worker 在 GPU 模式下空转，抢占驱动线程** —
      `src/core/thread_pool.rs` 的 `worker_loop` 空闲分支是「自旋 `WAIT_SPINS` 次 →
      `yield_now()`」。`yield_now()` 只让出调度，线程仍是 runnable，会被立刻重新调度，
      所以持有 ComputePool 的模型在 GPU 模式下会持续烧掉 N-1 个核。
      2026-10-25 在 Timestral 上量化过：Vulkan 跑 4096-token 长上下文，
      `--threads 4` 时 driver 侧 `wait_for_fences` 每次提交要 18.5 ms，
      同一份 GPU 命令流（dispatch 序列、push constants、descriptor 绑定逐条比对全同）
      在不持有 ComputePool 时只要 14.5 ms（快 30%）；`--threads 1` 时两者完全一致
      （14.73 vs 14.78 ms），说明差异与并发线程数成正比而非 GPU 工作量。
      CPU 侧同步观察到 `user` 时间 0.63 s → 5.24 s、占用率 64% → 353%。
      修法有两个方向：(a) GPU 活跃时让 idle worker 真正 park（`Condvar` 或
      `park_timeout`）而不是 yield —— 但这会改变 LLM 热路径的唤醒延迟，
      需要按模型分别评估；(b) 不在 GPU 场景持有 ComputePool。
      Timestral 目前走 (b)：CPU 投影与 attention 继续用 rayon global pool
      （`app::init_rayon_global_pool` 已按 `--threads N` 对齐，语义等价），
      transformer 图本身则由 `src/models/timestral/ops.rs` 的 op 列表统一描述，
      CPU / Vulkan 两个执行器消费同一份列表。
- [ ] **`src/models/diffusion/auk/mod.rs` 的单测泄漏 `RMI_PARITY_TRACE=1`** —
      该测试 `set_var("RMI_PARITY_TRACE", "1")` 后没有 `remove_var`，同一测试进程里
      后续任何会打 trace 的用例都会把 `<name>.f32` 写到**当前工作目录**，
      仓库根目录因此被塞进几百个 `1.timestral.*.f32` 之类的垃圾文件
      （`parity_trace::full_f32` 用 `format!("{trace}.{name}.f32")`，
      trace 路径是 "1" 时就落在 cwd）。修法：测试结束时恢复原值，
      或改用 `TempDir` 下的绝对路径。
- [ ] **Q8_0 与 Q8_K 量化路径按需量化（消除冗余计算，保留两份 buffer）** — dispatch 按 layer 权重格式，省一次量化 pass
- [ ] **Qwen3.5：借用权重与 FFN gate/up 输入量化复用的取舍** — 中期重构，不阻塞局部 FFN 优化
- [ ] **删掉三个纯转发的无参 `new()`** — `CollectSink`（`src/ops/generation_runtime.rs:158`）、
      `DacState`（`src/models/qwen3/tts/codec/dac.rs:115`）、
      `Lfm2MoeSampler`（`src/ops/sampling.rs:376`）都是
      `pub fn new() -> Self { Self::default() }`，而类型本身 `#[derive(Default)]`。
      同一个零值因此有两个名字，调用点又只用其中一个（`new()` 1/3/2 处，
      `default()` 0 处），`Default` derive 只是为了给 `new()` 转发而留。
      `Lfm2MoeSampler` 的动机可见：同族 `LlamaSampler::new(40, 0.95)` 必须带参，
      于是给无参版本也补了个 `new()` 让两行看起来一致 —— 代价是多一个名字。
      仓库分工应是 `new(args)` = 真构造函数（可带参、可 `Result`，如 `VulkanContext::new()`），
      `::default()` = derive-`Default` 类型的零值（`CliOptions` 17 处、
      `ExportOptions` 16 处均如此），两者不互换。已核实：三者均无
      `Default::default()` / `..Default::default()` 残留，也没有任何外层结构体
      依赖它们的 `Default`，故清理是每处 3 行（删 `new()` + 删 derive 里的
      `Default`）。
      反向教训见 `docs/develop/OPTIMIZATION.md:215` 记的 `KvFormat::default() = F32`
      （与 CLI 实际想要的 F16 相反）：**`Default` 的零值不成立时才真该用 `new()`**
      或干脆不给 `Default`。

### K-quant multi-row tile（vec_dot_q4k_q8k_avx2 / vec_dot_q6k_q8k_avx2）

目标：Q4_K_M 从 ~76 t/s → 120-150 t/s。详见 TODO-001 关联。

### BitNet `I2_S` GGML type 的 provenance 与上游兼容性

`GGMLType::I2_S = 36` 和 `general.file_type = 40` (LLAMA_FTYPE_MOSTLY_I2_S) **不是上游
ggml-org/llama.cpp 的一部分**——它们是 `microsoft/BitNet` 私有 fork 在 `ggml.h` 中加的扩展，
转换脚本 `utils/convert-hf-to-gguf-bitnet.py` 用 `gguf.GGMLQuantizationType.I2_S`。后果：

- 这两种 BitNet GGUF **只能**在 BitNet-patched llama.cpp / `microsoft/BitNet` build / 本仓库 (`I2_S` 已注册) 里
  跑，stock `llama.cpp`、`ollama`、`koboldcpp` 等社区 GGML 引擎会在加载时报
  "unknown tensor type" 直接 abort。
- 跨生态分发 BitNet 模型时需要在 README / GGUF 描述里明确这一点，避免用户误以为
  `ollama run bitnet-embedding-0.6b` 能跑（实际上 GGUF 类型会被拒）。
- 长期：上游 PR 把 `I2_S` 合并进 ggml-org/ggml（需要 bitnet.cpp 团队与 ggml 维护者协调），
  或者在 GGUF 规范里加一个 `quantization_version` 字段，让 stock llama.cpp 把
  `file_type=40 + GGMLType=36` 解释成 "需要 BitNet-patched build" 错误并清晰报错。

当前会话的处理：commit message + `src/core/tensor.rs::GGMLType::I2_S = 36` 旁的注释明确写了
"Microsoft BitNet b1.58 I2_S extension; not upstream ggml"。`docs/usage/bitnet_embedding.md` 第一节
"已下载并验证的 GGUF" 提示兼容性受限。**没有自动对齐上游的工作；等上游 merge 或社区分流后再调整。**


- [ ] **模式互斥白名单是手写枚举，每加一个共享 flag 都要改多处** —
      `src/app/cli/parse.rs` 的主循环统一解析所有 flag（`--gpu` 在第 389 行），
      但每个模式在解析之后另有一段手写白名单，声明该模式接受哪些 flag，并对带值
      参数补一次取值校验。`--enhance`、`--system-one-request`、`--decision-request`
      各有一段，`--gpu` 已经是第 3 个模式接受共享 compute flag —— Vela3 混合投影
      放开 `--system-one-request` 的 GPU 支持时，就得手改系统模式的白名单，那行
      `| "--gpu"` 看着像特例，实际是共享 flag 列表的又一份副本。
      方向：把「模式 → 允许的 flag 集」集中成一份声明，或反过来给 flag 标注归属
      模式；两者都是**纯重构**，不改行为。`parse.rs` 是所有模型的入口，要**独立成
      PR**，别混进模型适配。
      详见 [TODO-064](#todo-064-模式互斥白名单手写枚举每加一个共享-flag-都要改多处)。
- [ ] **Vela3 的 F32 权重路径每次投影都整份拷贝** — `models/vela3/tensor.rs` 的
      `linear()` 用 `QuantizedTensor::from_bytes` 交权重：F16/BF16 借用 mmap 字节切片，
      **F32 会把整份权重 `collect()` 成 `Vec<f32>`**（`ops/kernel/quantized_tensor.rs`）。
      而 `linear()` 在 `d3.rs` 的 `project()` 里每层每投影调一次 → F32 诊断变体
      （`vela3-d3-lite-f32` 等）反复拷贝整份权重。发布的 Vela3 权重是 BF16，走零拷贝，
      **生产默认不受影响**，只有 F32 变体慢。
      方向：让 `QuantizedTensor::F32` 也借用字节切片（共享算子改动），或在 Vela3 侧
      加载期转一次并复用。详见
      [TODO-065](#todo-065-vela3-的-f32-权重路径每次投影都整份拷贝)。
- [ ] **严格 Vulkan 抓不到「投影静默回落 CPU」** — `--compute vulkan` 的语义是
      「请求真的在设备上」。共享投影器在若干情况下静默回落 CPU 且**不**把设备标记为
      故障：输出元素数小于 `RUST_GPU_MIN_OUTPUTS`（默认 16K）时、`RUST_GPU_MAX_ROWS`
      限流时、形状检查失败时。请求结束时的严格档检查只看「设备是否因队列故障被禁用」，
      抓不到这些 → 一次 `--compute vulkan` 请求可能全在 CPU 上跑完仍返回 200。
      「设备根本不存在」已由 `configure_cli()` 的前置检查挡住（实测两种情形都成立）；
      还没覆盖的是「设备在，但投影因形状或阈值回落」。
      这是**语义决策而非 bug**：把回落升级成错误，等于承诺严格档的每次投影都在设备上，
      而小投影留在 CPU 正是当前设计意图。
      详见 [TODO-066](#todo-066-严格-vulkan-抓不到投影静默回落-cpu)。

- [ ] **PiVo 编码器的投影换用共享的 F32 分块 GEMM** — 手写的 `pivo_linear` 是
      「一 lane 一输出列」的朴素版，边际成本 0.126 ms/token，比宿主的 0.071 ms/token
      还贵 1.8×；共享层已有 `float_matmul_tiled`（分块 + 合并读 + FMA，三组权重
      一次 dispatch），换过去**不需要新 shader**。**不要做块级融合**：融合只砍固定
      开销、边际成本不变，仍慢于宿主，而 92% 的时间在声码器（上不了设备）。详见
      [TODO-067](#todo-067-pivo-编码器的投影换用共享的-f32-分块-gemm不是融合)。

## Low Priority

- [ ] 更多量化格式支持（Q4_K, Q5_K 等）
- [ ] 完善 GGUfRS 导出功能
- [ ] GGUF 导出支持
- [ ] **ZDTaichu5.0 mmproj 的位置嵌入与归一化和上游 PyTorch 不一致** — 上游把 128x128
      位置表用 `align_corners=False` 插值到 32x32，而第三方 GGUF 转换用的是
      `align_corners=True`（实测该插值与 GGUF 的 `v.position_embd.weight` **逐位一致**，
      相对 L2 0.000000）；两张表相差相对 L2 **0.629774**，传到投影特征就是 **0.2015**。
      另外上游**双重归一化**（image processor 用 ImageNet 统计 → 塔内 conditioner 再用
      CLIP 统计），而 llama.cpp 与本引擎按 GGUF metadata 只归一化一次，忠实上游路径与
      llama.cpp 相差相对 L2 **4.54**。
      现状：本引擎实现的是 **GGUF 契约**（与 llama.cpp 一致，视觉投影 relative L2
      0.00036），**不是 Rust 的 bug**；但与上游自己的数字不一致。
      选项：① 写一个从原始 safetensors 转 mmproj 的转换器，按上游的
      `align_corners=False` 插值并只归一化一次，让引擎与上游对齐；② 维持现状并在
      文档里保留差异说明（当前做法）；③ 向上游/GGUF 转换作者反馈。
      证据：[oracle 记录](../../tools/oracle/zdt/README.md)。

---

## Detailed TODOs (merged from legacy docs/TODO.md)

### TODO-001: Q4_0 AVX2 kernel 重写（已完成，2026-11-21）

#### 原状

`src/ops/kernel/q4_0/avx2.rs` 的 Q4_0 matmul kernel：每 block 先把整个 i32 点积
`hsum_epi32` 成标量，再把**一个标量**塞进 8-lane f32 累加器的 lane 7 做
`_mm256_fmadd_ps`——8 个 FMA lane 只用了 1 个，外加每 block 一次水平归约；
无行分块（TODO 原文写的"不使用 FMA"已过时，代码其实早已引入 fmadd）。

#### 处理

按 `q8_0/avx2.rs` 的结构重写：

1. **逐 lane 累加**：每 block 的 i32 点积保持 8-lane 竖向（每 lane 恰好 4 个乘积，
   精确），`dot_lane = nib_lane - 8 * y_lane`（y_lane 每 block 算一次、4 行共享），
   直接 `_mm256_fmadd_ps(ds, cvtepi32(dot_lane), acc)`；hsum 只在每行结束时做一次。
2. **4 行分块**：Q8 输入 block 每块载入一次、4 个权重行共享（与 Q8_0 kernel 同款）。

#### 验证与语义

- 该累加结构与 llama.cpp 自家 AVX2 `ggml_vec_dot_q4_0_q8_0` **同构**
  （llama.cpp 同样是逐 lane fmadd + 末尾 hsum；两种"每 lane 精确 i32"写法——
  减 8 后符号展开 vs nib−8·y 校正——整数上逐 lane 相等），因此默认路径与
  llama.cpp AVX2 语义对齐；旧 kernel 是既不匹配 scalar 也不匹配 llama.cpp AVX2
  的自定义混合体，已删除。
- parity：per-block 整数点积两边都精确，仅 f32 累加顺序不同 → 与 scalar 的
  对比从"≤4 ULP"放宽为仓库 Q4_K / Q8_0 kernel 同款 **rel < 1e-3**（实测随机
  4×512 最大 7 ULP / 相对 4.5e-7；测试保留 ULP 打印）。
- E2E：Qwen3-0.6B-Q4_0 greedy 输出正常（"The capital of France is **Paris**."）。
- 注意：`tests/inference_parity.rs::q4_0_matches_pinned_scalar_oracle_bit_for_bit`
  （env 门控）把默认路径与 llama.cpp oracle 逐位对拍——若 oracle 是 AVX2 构建，
  本次改动后应保持逐位一致（新结构即 llama.cpp 结构）；若 oracle 是标量/FMA-disabled
  构建，逐位一致原本就不成立（旧 kernel 带 fmadd），需按其 oracle 模式重验。

#### 性能

kernel 级（release，Qwen3-0.6B attn_q 1024 值/次）：旧 kernel 771 µs → 新
**372 µs（2.07×）**，对 scalar 7.32×。

#### 关联文件

- `src/ops/kernel/q4_0/avx2.rs` — 重写后的 kernel + parity 测试
- `src/ops/kernel/q4_0/scalar.rs` — scalar reference（未动）

---

### TODO-002: SIMD GeGLU `tanh_approx` 未启用

#### 现状

`src/models/gemma4/trunk/forward.rs` 的 `ggml_geglu_fp16_inplace` 仍然走 scalar 路径。
AVX2+F16C SIMD 版本原型写过又回退了，注释详细记录在函数体内。

#### 影响

- **性能**：scalar GELU 算 10240 元素 × 2 个 GELU/层（FFN + per-layer）× 35 层 = ~717k scalar ops/token。
  scalar ~12ns/element = **~8.6 ms/token**（当前 E2B ~80ms/token 的 ~10%）。
- **正确性**：scalar 版本 bit-exact 同 llama.cpp。所有 gemma4 测试通过。

#### 选项

1. **接受 ±1-2 ULP drift 启用 SIMD**（推荐）：Padé [7/6] tanh 近似（~0.04% max error in |x| ≤ 4.5）；
   加 `#[cfg(feature = "experimental-geglu-simd")]` 默认关；
   用 `cargo run --features experimental-geglu-simd` 启用。
   预期：+5-8% ETE。风险：gemma4_reference.rs 中 token Oracle 偶尔 ±1 分叉。
2. **Schraudolph fast exp + tanh = (exp(2x)-1)/(exp(2x)+1)**：5-10% tanh error on |x|>2 → 必分叉。不推荐。
3. **保持 scalar**：零风险，零加速。

#### 推荐

短期：选项 3。中期：跑 gemma4_reference.rs Oracle 确认 Padé [7/6] 漂移可控后，切到选项 1。

#### 关联文件

- `src/models/gemma4/trunk/forward.rs:858` — 当前 scalar + 详细 doc-comment

---

### TODO-003: 统一 5 个 `load_weight` / `load_weight_any` 到单一 `core::load_weight_any`

#### 现状

仓库目前每个模型自维护一份 GGML weight loader，签名 / 白名单 / dims 拆解 / 错误返回都不一致：

| 实现 | 签名 | 接受类型 | dims 校验 | 错误返回 |
|---|---|---|---|---|
| `dots/weights.rs::load_weight` | `(source, name, dims)` | F32/F16/BF16/Q8_0 | 严格 | `Result<_, String>` |
| `qwen35/trunk/weights.rs::load_weight` | `(source, name)` | 8 种 | 不校 | `Option<Weight>` |
| `qwen35/trunk/weights.rs::load_weight_f32` | `(source, name) → Vec<f32>` | F32/F16/BF16 | n_elements | `Option<Vec<f32>>` |
| `gemma4/trunk/weights.rs::load_weight` | `(source, name, dims, ggml_type)` | 单一类型 | 严格 dims+type | `Result<_, String>` |
| `gemma4/trunk/weights.rs::load_weight_any` | `(source, name, dims, &[GGMLType])` | 白名单 | 严格 dims+白名单 | `Result<_, String>` |

`qwen3` 干脆没有 `load_weight` —— 直接展开 `Weight::from_quantized(QuantizedTensor::from_bytes(...))`，
天然支持 `QuantizedTensor::from_bytes` 接受的全部 20+ 类型。

**核心公共部分**（≈ 5 行）：

```rust
let info = source.tensor_info(name)?;
let bytes = source.tensor_slice(name)?;
let weight = Weight::from_quantized(QuantizedTensor::from_bytes(
    bytes, info.ggml_type, n_in, n_out,
));
```

#### 影响

- **未来加 IQ2/IQ4/Q6_K/Q5_K 等新量化类型**——每个 `load_weight` 都要扩白名单。
- **dims 拆解不一致**：`dims.last()` vs `dims[0]/dims[1]` vs `dims[..-1]` 累乘
- **白名单粒度不一**：写死 vs 参数化 vs 无
- **错误返回风格不一**：`Result<_, String>` vs `Option<Weight>`

#### 选项

1. **最小**：扩 `dots::load_weight` 接受 Q4_0/Q4_1。维持命名遗留。
2. **中度**：breeze 摆脱 dots 命名，改直接展开 `Weight::from_quantized(...)`（学 qwen3）。
3. **完整**：5 个 `load_weight` 全部迁到 `crate::core::tensor::load_weight_any`：
   - 签名：`(source, name, dims: &[u64], allowed: Option<&[GGMLType]>) -> Result<Weight, String>`
   - `dims` 校验可选；`allowed=None` 表示 `QuantizedTensor::from_bytes` 接受全部类型
   - 所有模型调用方改 import；`qwen35::load_weight_f32` 与 `core::load_f32_tensor` 合并去重
   - 工作量：~200 行；彻底统一

#### 推荐

**方案 3**。`load_weight` 不是 dots 私有的，它是项目 GGML weight loader 的事实标准入口——
`qwen3` 已经证明**不抽象也活得很好**，所以抽象必须带来明显收益。

#### 关联文件

- `src/models/dots/weights.rs:5-67`、`qwen35/trunk/weights.rs:80-124`、`gemma4/trunk/weights.rs:226-309`
- `src/core/tensor.rs:288-330` — `load_f32_tensor`

---

### TODO-004: Breeze Q4_0 不可用 — 走 per-tensor 混合精度路径

#### TL;DR

**Q4_0 单层全量化对 Breeze 不可用**（128 frames vs BF16 59 frames = 117% 偏差）。
短期路线不是"再换一种 4-bit 格式"，而是 **per-tensor 混合精度**：embedding / lm_head / codebook 保留 BF16 或 Q8_0，
hidden-attn / mlp 主矩阵走 Q4_0 或 Q4_K。长期再评估 Q4_K / Q6_K 全替换。

#### 现状

实测推理结果：

| 精度 | 帧数（prompt "你好。"，seed 42） | 文件大小 | 备注 |
|---|---|---|---|
| BF16 | 29 | 6.6 GB | 与原字节等价 |
| F16 | 35 | 6.6 GB | 同源 |
| F32 | 29 | 13 GB | md5 与 BF16 bit-exact |
| Q8_0 | 37 | 3.6 GB | **3.6× 加速，推荐部署** |
| **Q4_0** | **128** | 2.1 GB | **输出退化** |

#### 关联文件

- `tools/converter/breeze/convert_breeze.py` — `_must_keep_source` / `_source_ggml_type`
- `src/ops/kernel/quantized_tensor.rs:275-...` — `QuantizedTensor::from_bytes`
- `models/Breeze-TTS-2-gguf/breeze-tts-2-Q4_0.wav` — 当前 Q4_0 输出（退化证据）

详见原 `docs/TODO.md` §TODO-004 完整分析（256 行），含 per-tensor 风险表、推荐方案、实施步骤。

---

### TODO-005: F32 x86_64 matmul 缺 SIMD kernel — ✅ 已完成

`src/ops/kernel/f32/avx2.rs` 实现完成，F32 / F16 NEON kernel 完成，AVX2 matmul macro 共享代码。
BF16 / F16 / F32 三档精度 matmul 共享 `avx2_matmul_packed!` 宏。

实测：Breeze `--quant f32` scalar 234s → AVX2 54s (4.3×)，`--quant f16` 47s → 31s (1.5×)。
F32 与 BF16 md5 bit-exact (a7c5e3f5...)，三档精度 md5 都已记录 baseline。

---

### TODO-006: Q8_0 Breeze output drifts ~1 ULP under SIMD activation/matmul

**v3 baseline `b051f3c1...` / 29 frames 已选定**。BF16 / F16 / F32 bit-exact 是硬约束；
Q8_0 没有 ground truth（量化本身有损），后续每次 SIMD 改动会再次漂移——只要 ≤ 1 ULP 就接受，记录新 baseline。

详见 `tools/converter/README.md:40` Q8_0 行。

---

### TODO-007: Breeze `rope()` SIMD 路径精度破坏 — ✅ 已修复

`src/ops/rope/neox.rs::rope_neox_inplace_with_table` 新增，BF16 round-trip 严格按 scalar op 顺序
（mul → round → mul → round → add → round），AVX2 内核在每个乘加后立即 bf16 round。
`src/models/breeze/transformer.rs::rope` 一行调用。Breeze 三档精度 (BF16/F16/F32/Q8_0) bit-exact。

剩余：NEON 版本（aarch64）暂未实现，不阻塞 Breeze 路径。

---

### TODO-008: Breeze F16 合成听感更合理 / F32-BF16 需对齐

人工听审发现：Breeze TTS 2 在 BF16 / F16 / F32 下比特级输出一致（BF16 ↔ F32 md5 `c19502ff...`），
但听感上 F16 更合理（F16 md5 `4dd19b6d...`）。

数学上 BF16 = F32（高 16 位），两者 bit-exact 等价是数学正确；F16 与它们差异是 mantissa 精度 + exponent bias 不同。
这是**听感 vs 数值精度**的张力：Breeze 上游训练 / 参考实现大概率用 BF16 模拟精度，
F16 路径数学上不对但听感更"干净"——需要在 BF16 / F32 路径上找到听感偏差的根因。

详见 `models/Breeze-TTS-2-gguf/README.md` Alignment 段落。

---

### TODO-010: Prefill runtime 通用化，避免每个模型重复实现 chunk 调度

#### 现状

`src/core/prefill.rs` 目前只提供 `prefill_chunks()` 和 batch size 校验。
Qwen3、Qwen3.5、Gemma4 仍分别维护自己的 chunk loop、CPU/Vulkan fallback、KV/state snapshot、commit/rollback 和 logits 输出处理。

#### 目标

参考 llama.cpp 的 `llama_batch_allocr`、`llama_memory_context_i` 和 `process_ubatch` 分层：

- 公共 `PrefillRunner` 负责输入校验、chunk 切分、`base_position`、输出顺序、失败重试、backend fallback 和 chunk 级事务；
- 公共 `PrefillContext` / `ChunkTxn` 统一 row shape、状态快照、commit/rollback；
- 模型只实现 `PrefillKernel::forward_chunk`，保留各自的 attention、SSM/conv、MoE 和多模态 position 语义；
- 不支持多行 batch 的模型声明 singleton fallback，不阻塞模型接入。

#### 实施顺序

1. 抽 `PrefillRunner` 包裹现有 Qwen3/Qwen3.5/Gemma4 loop，不改变数值行为。
2. 抽 `PrefillContext` / `ChunkTxn`，统一 KV/state commit、rollback 和 CPU/GPU fallback。
3. 模型接口收敛为 `PrefillKernel` + capability；无 batch 能力时自动走 batch=1。
4. 复用 shape 兼容的 scratch/graph，补充 batch=1、跨 chunk、失败回滚和真实 GGUF parity/benchmark。

#### 关联文件

- llama.cpp：`src/llama-batch.h`、`src/llama-memory.h`、`src/llama-context.cpp`、`src/llama-graph.h`
- 当前 Rust：`src/core/prefill.rs`、`src/ops/kernel/mod.rs::PreparedRows`、`src/models/qwen3/trunk/prefill.rs`、`src/models/qwen35/trunk/session.rs`、`src/models/gemma4/trunk/forward.rs`

---

### TODO-013: CLI interactive mode — multi-turn history + slash commands

#### 现状

CLI interactive mode **已存在**（`src/app/text/generation.rs:218` 的
`run_interactive` 和 `:267` 的 `run_interactive_qwen35`），触发条件是
`options.prompt.is_empty()`（`src/main.rs:495` 的 else 分支）。REPL
loop 完整，从 stdin 读行、Ctrl+C / EOF 退出。

但**核心功能缺失**：

| 功能 | 状态 |
|---|---|
| REPL loop（stdin → stdout） | ✅ |
| Ctrl+C / EOF 退出 | ✅ |
| 每次 input 独立生成 reply | ✅ |
| **multi-turn history** | ❌ 每行独立，模型看不到前文 |
| **system turn（CLI 配置）** | ❌ 无 CLI 接口；模型不知道角色 |
| **`/clear` / `/system "..."` / `/exit` 命令** | ❌ |
| **streaming token-by-token 输出** | ❌ 整段生成完才 print |
| **`--thinking` / `--no-thinking` 生效** | ❌ `run_inference` 在 interactive 路径 hardcode `thinking=false` |
| **KV cache 增量（不重 prefill）** | ❌ 每轮从 0 开始 |

**最大痛点**：
```bash
> Hi, I'm Alice.
Hello Alice!

> What is my name?
I don't know your name.  # ← 不记得 "Alice"
```

根因：`lfm2::run_inference` 和 `lfm25::run_inference` 都签名是
`(..., prompt: &str, ...)`，内部 `build_lfm2_chat_prompt_with_thinking`
的 `&[Lfm2Message]` 被 hardcode 为 `[Lfm2Message { role: "user",
content: prompt }]`，没有 history / system 概念。

HTTP 路径**已经完整**（PR #118 + `da6bbcb` + 后续）—— 缺的是 CLI。

#### 实施 cost（粗算）

| 步骤 | LOC | 风险 |
|---|---|---|
| `lfm2::run_inference` 加 `turns: &[Lfm2Message]` 替代 `prompt: &str` | ~15 | 低（纯重命名 + 调整） |
| `lfm25::run_inference` 同上 | ~15 | 低 |
| `run_interactive` 重写：维护 `Vec<(role, content)>` history + system + slash commands | ~80 | 低 |
| `run_interactive_qwen35` 同上（multimodal path） | ~80 | 低 |
| Streaming token 输出（可选） | ~50 | 中（要 token sink 接口） |
| 思考 flag 透传 | ~5 | 低 |
| 测试（stdin mock + 历史一致性） | ~80 | 中 |
| **总计** | **~325 LOC** | **触及多个 model forward 入口；可能影响 Oracle 测试 baseline** |

#### 为什么不立即实施

1. **interactive mode 是 partial feature，不是 bug**。当前 1-shot-per-line
   满足 smoke test / benchmark / debug 三大主要 CLI 场景。
2. **HTTP 路径已完整**（multi-turn, system, thinking, streaming）。
   真正需要 multi-turn 对话的用户走 HTTP。
3. **改 `lfm2::run_inference` 签名**会触动 CLI / benchmark / Oracle
   test 的多个调用点，需要回归测试全套 llm 模型的 greedy baseline。
4. **ROI 偏低**：LFM2.5-8B-A1B 等思考模型在 CLI 上调试时确实希望有 history，
   但生产场景主要是 HTTP API 调用方。

#### 触发条件

满足下列**任一**条件即应启动：

- 用户公开 issue / Discord 报告"interactive mode 不支持 multi-turn"（≥3 次）
- LFM2.5-Thinking 系列在 CLI 上成为常见调试 / 微调工作流
- 决定给仓库加一个"开发者本地 chat loop"功能，作为测试 / 微调工具

#### 关联文件（实施时）

- `src/models/lfm2/trunk/forward.rs` — `run_inference` 接受 turns
- `src/models/lfm25/trunk/forward.rs` — 同上
- `src/app/text/generation.rs` — `run_interactive` 重写 + 加 streaming 接口
- `src/app/cli/parse.rs` — 加 `--system` CLI flag
- `src/main.rs` — interactive dispatch 把 `options.thinking` 透传
- `tests/cli_history.rs`（新）— 多轮 history 一致性测试

---

### TODO-015: arch 接入的"公共层报到"问题 — 设计记录，暂不实施

#### 起因

`Falcon-H1` 适配 CLI 后用 `/v1/chat/completions` 失败：

```
Architecture "falcon-h1" is not supported by the server text endpoints
```

排查发现一个 arch 要为**三个公共入口各报到一次**，且漏了不报错：

| 入口 | 位置 | falcon 状态 |
|---|---|---|
| CLI | `src/app/text/generation.rs` `} else if arch ==` | ✅ 已接 |
| HTTP runtime | `src/app/text/runtime.rs::build_text_runtime` | ❌ 无 adapter |
| HTTP prompt | `src/app/server/api/tools.rs::is_qwen35` / `llama_family` | ❌ 无分支 |
| JEV | `src/app/jev/single.rs` `match &*arch` | ❌ 落进 `other => Err` |

维护者的比喻：**公共层是"插口/门牌"，用户只面对任务（传文本/图片 → 拿结果），不关心里面是谁**。而当前 registry 全是 `match arch`——**按施工队组织，不按门牌组织**，导致：

1. 新 arch 不知道该去哪几处报到（falcon 漏了 2 处，且静默）；
2. 不支持某能力的 arch（如 TTS）要"证明自己不该在 JEV 里"，需要维护
   "故意不支持"白名单；
3. 同一个能力在三个入口重复出现。

#### 三层拆解（讨论结论）

| 层 | 现状 | 是否要动 |
|---|---|---|
| **执行接口** | `TextRuntime` / `JevScorer` trait 已存在且健康（JEV 9/9 trunks 统一走 trait） | 不用动 |
| **公共 API** | HTTP 的 `/v1/chat/completions` / `/v1/audio/speech` 本来就是按任务分、不关心背后 arch | 已经对了 |
| **登记处** | 3 个手写 `match arch`，新 arch 报到 3 次、漏了不报错 | **唯一真痛点** |

关键认知：用户问"falcon 为什么 HTTP 不能用"，答案是 **adapter 没写**（工作量问题），
不是"架构缺插口"——接口在，只是没插。

#### 评估过的方案

**A. distributed slice（`inventory` crate）+ capability 门牌**

```rust
pub mod capability {
    #[distributed_slice] pub static TEXT_GENERATE: [CapabilityFactory];
    #[distributed_slice] pub static JEV: [CapabilityFactory];
    #[distributed_slice] pub static TTS: [CapabilityFactory];
    // ...
}
// trunk 内自己挂门；不写就是不支持 → TTS 天然不进 JEV，无需白名单
// 公共层写完永久冻结
```

- 语义最干净："不声明 = 不支持"，新 arch 零公共层改动。
- **否决（现在）**：为"省 30 行 registry"引入 linker-magic 依赖不值；且
  I/O 组合仍在快速扩张（chat: 纯文本 → +图 → +音频/视频；omni 会把
  chat/score/embed 的边界糊掉），**预定义 capability slot 是在流沙上钉钉子**——
  真到 omni 那天，是 I/O 抽象重塑 capability，不是反过来。

**B. Plugin trait + 单注册表（~500 LOC 重构）**
- 否决：公共层仍要每 arch 加一行；且 TTS 不支持 JEV 要靠 `default fn` 或按
  capability 拆 4 个 trait，比 A 更脏。

**C. loader arch 列表提成 `SUPPORTED_ARCHES` const + coverage 测试（~80 LOC）**
- **设计已定稿，暂不实施**。见下。

#### TODO-015 的落地设计（将来启用时直接用）

1. `src/core/loader.rs`：把 `model_config_from_source` 里的 `matches!(...)`
   改成 `pub const SUPPORTED_ARCHES: &[&str]` + `contains`。新 arch 必须先加进
   const → 测试自动覆盖它。
2. `tests/arch_coverage.rs`（或 lib test `arch_coverage`）：遍历 `SUPPORTED_ARCHES`，
   断言每个 arch 在 CLI / HTTP / JEV 三个 dispatch 都有**显式**答案
   （`Supported` 或在带原因注释的"故意不支持"白名单里）。
3. CI 加**独立 job**（不能并进 `cargo test --lib`——那 19 个 pre-existing
   failures 会把它染红）：
   ```yaml
   arch-coverage:
     steps:
       - run: cargo test --profile release-fast --lib arch_coverage
   ```

效果：falcon 这类"静默漏注册"以后 CI 立刻红；本地
`cargo test --lib arch_coverage` 秒级回报。零新依赖、零架构改动。

#### 触发条件（满足任一才启动实施）

- arch 数量 > 25，或
- 新 arch 接入频率 > 每月 1 个（"报到 3 次"成为真瓶颈），或
- omni 模型（原生统一多模态 I/O）落地——届时 I/O 抽象自然重塑 capability 定义，
  **那时做 A 才是顺势而为**，今天钉死的 slot 只会碍事。

在触发前，处理方式是：**新 arch 接入时手工检查三处 dispatch + 补实现**
（正如本次给 falcon 补 HTTP/JEV）。

---

### TODO-016: Jina v5 Omni audio encoder — oracle 验证待补

本仓库 `encode_audio()` 已按 jina-ai `feat-v5-omni` llama.cpp fork 设计切 30 s Whisper 块（见 commit `9717d17`），但**未跑过 llama.cpp 端的逐位对照**：

1. **per-chunk byte-equal 对齐 llama.cpp `b96806d`**
   - 现存 `tests/jina_audio_projection_matches_llama_cpp_bits`（#[ignore]）已经覆盖单 30 s 块的 750 × 1024 投影 F32 对照，但每次跑需要：
     - `LLAMA_DIR` 是固定 commit `b96806d96061049a5b574269b049bf6241d63d46` 的独立副本
     - 应用 `tools/oracle/jina_audio/mtmd-audio-projection.patch` + `tools/oracle/qwen35/qwen35-scalar-softmax.patch`
     - `cmake -B build-rmi-jina-audio -DGGML_ACCELERATE=OFF -DGGML_METAL=OFF -DCMAKE_CXX_FLAGS=-DRMI_QWEN35_SCALAR_SOFTMAX`
     - 跑 `llama-mtmd-cli` dump oracle F32，再跑 `cargo test ... -- --ignored`
   - macOS ARM CPU / 单线程 / 标量 softmax / 关 Flash Attention 的严格 CPU 路径
   - 触发条件：新增 jina v5 audio encoder 任何改动时必跑

2. **跨块拼接 oracle 对齐**
   - 上一步只覆盖单个 30 s 块。我们的 `encode_audio()` 把多个 30 s 块的 post-conv tokens 拼给 LLM，让 LLM cross-attend 跨块融合
   - 上游 `feat-v5-omni` fork 也是同样设计（mtmd split 30s + LLM cross-attend），所以两边的最终 LLM embedding 应等价
   - 当前只在仓库端跑过 30s / 60s / 90s 端到端语义验证（cos 关系正确），**未跟 llama.cpp 跨块拼接对比**
   - 验证方法：取 jina-omni + llama.cpp fork + 一段 >30s 音频（e.g. 60s），分别跑两端的最终 embedding，断言 cos ≥ 0.99 + max abs diff 在量化噪声内（Q8_0 文本 + F16 mmproj 路径下应该是 ~1e-3）
   - 触发条件：commit `9717d17` 之后任何 jina v5 改动、Q4_K / Q5_K / Q6_K 等不同量化、新 mmproj 版本

**2026-10-01 per-chunk oracle 实跑结果**（commit `fe4b99c` 之前 / 之后）：
- **mel extraction**：✅ 与 llama.cpp bitwise-aligned（max abs diff 0.378 主要来自帧 2999 末尾 40 个 reflect-vs-zero 反射差异，符合预期）
- **位置嵌入**：❌→✅ commit `1201166` 修了 `token % self.config.window` bug，固定到 `token`（1500 个唯一 pos）。见 commit message
- **conv1d 输出**：✅ **Rust 端正确**——numpy 参考实现和 Rust `conv2_only` 在 ULP 级吻合（max abs 3.4e-3，AVX2 vs scalar 重排差），意味着 Rust 的 `conv1d_same_f16` patch 构造 + `dot_f16_f16_bytes` 实现都没问题
- **oracle dump 不可信**：之前以为是 Rust conv1d 有大 bug（cos=0.18 with oracle），但 2026-10-01 bisect 时发现 llama.cpp 的 `inp_raw` dump 两次值不一样（v0[0]=4.92, v1[0]=1.35），且 dump 出的值和我们 mel 也对不上——说明 ggml 的 input tensor 在 `ggml_set_input` 之后 buffer 被 allocator 复用了，`ggml_backend_tensor_get` 读到的是被覆盖的脏数据。同理 `after_conv1d` dump 也不可信
- **结论**：当前 oracle 对比无法作为 ground truth。Rust 端的 correctness 由 numpy 参考独立验证了。TODO-016 暂时挂起，等 oracle dump 机制修好后重新跑

**当前不做**：
- 触发条件没到：jina v5 audio encoder 改动少（一次 30s 切块重构 + 一次位置嵌入修复），且端到端语义验证已经覆盖了跨块拼接的功能正确性
- 仓库的 cargo test 已经能在没有 llama.cpp 副本的 CI 上跑出 934/19/67 baseline，oracle 失败不会阻塞
- **触发再做**：
  1. llama.cpp dump 机制修了之后（用 `ggml_backend_tensor_get_and_copy` 或者 `ggml_set_input` 后立即读），重新跑 per-chunk 对齐
  2. 用户加新的 jina-omni 量化、或者上游 llama.cpp fork 有新 audio 相关改动需要重新对齐

调试现场保留在 `/tmp/audio-fix-test/`（`our-trace-conv1.jsonl.*`、numpy 参考脚本 `conv2_only=0.054`/`-0.008` 验算过程）和 `/tmp/rmi-jina-audio-*-oracle*.f32`。
下次继续走：先修 oracle dump（直接读 `t->data` 改用 `ggml_backend_tensor_copy` 或者改在 `set_input_f32` 之后立刻 dump），再用可信的 oracle 验证 Rust 端对齐。

**实施步骤**（任一触发条件满足时）：
1. `cd $LLAMA_DIR && git checkout b96806d` 起固定 commit
2. apply 现有两个 patch
3. cmake + build（macOS ARM / Linux x86_64 各一次）
4. 跑现有 `jina_audio_projection_matches_llama_cpp_bits` 对 30s 单块
5. 新增 `jina_audio_projection_matches_llama_cpp_bits_concat`（#[ignore]）跑 60s 跨块拼接
6. 把对照数据 commit 到 `tools/oracle/jina_audio/`（或外置 datum）

### TODO-017: BERT 家族 + EmbeddingGemma encoder — llama.cpp 位级 oracle 未做

`src/models/bert_family/`（bert / jina-bert-v2 / jina-bert-v3 / nomic-bert / nomic-bert-moe）
与 `src/models/gemma_embedding/` 共 7 个 arch，全部**只做了语义验证，没做 llama.cpp 位级 oracle**。

每个测试文件自己都写明了这一点，例如 `tests/jina_v2_base_en.rs`：

> Oracle: local read-only `references/llama.cpp/src/models/bert.cpp`.
> bit-level parity with llama.cpp requires the oracle binary and is tracked separately.

而 `tools/oracle/` 下**没有** `bert/` 也没有 `gemma_embedding/` 目录——对比 gliner2 / clm / laya /
jina_audio / qwen3_tts 都有完整的 `dump_*.py` + `fixtures/` + `compare.py`。

| arch | 上游源文件（本地 `references/llama.cpp` 存在） | 现有验证 |
|---|---|---|
| `bert` | `src/models/bert.cpp` | 语义 + 元数据 + 张量清单 |
| `jina-bert-v2` | `src/models/bert.cpp` + `jina-bert-v2.cpp` | 同上 + WPM pinned ids + 语义排序 |
| `nomic-bert` | `src/models/bert.cpp` | 同上 |
| `nomic-bert-moe` | `src/models/bert.cpp` + `nomic-bert-moe.cpp` | 同上 + MoE 张量清单 |
| `bge-m3` | `src/models/bert.cpp` | 同上 |
| `gemma-embedding` | `src/models/gemma-embedding.cpp` | 同上 + 语义排序 |

已验证的部分（2026-09-30 对 `jina-embeddings-v2-base-en` 复核）：4/4 集成测试通过、CLI 与 HTTP
输出 768/768 元素 `%.9f` 文本完全一致、线程数 1/4/8/0 输出确定、L2 范数 1.000000、语义排序正确。
**这些都不等价于"与 llama.cpp 逐位一致"。**

#### 为什么值得做

这个区域已经出过两个只有 oracle 能早发现的真 bug，都是 port 完成后靠别暴露的：

1. **`attn_proj` 从未加回 `hidden`**（PR #125 记）：每层对 `inpL` 都是恒等映射，attention stack
   整体空转。jina-bert-v2 / nomic-bert / bert 三个变体全中，只是前两个 ERSS 相似度够高掩盖了；
   bge-small 因 CLS pooling 直接输出退化（相似度全 ~1.0）才暴露。修复后 rel→unrel 差距
   jina 0.170→0.417、nomic 0.258→0.443。
2. **MoE `per_expert_bytes` 算成 44 应为 34**（nomic-embed-text-v2-moe）：每个 expert 多读
   0.73 MB，第二个 expert 之后产出 NaN。

共同点：都能被一个位级 oracle 在 port 当天抓住，而不是等用户撞上。

#### 需要的产出

1. `tools/oracle/bert/` 与 `tools/oracle/gemma_embedding/`：`dump_golden.py`（从 llama.cpp
   `llama-embedding` dump 向量 + token ids）、`fixtures/*.json`（覆盖每个变体的开关组合：
   ALiBi vs pos_embd、GELU vs geglu vs SwiGLU、mean vs CLS vs last pooling、fused vs split QKV、
   MoE 偶奇层）、`compare.py`（严格查 token ids / 形状 / F32 原始位，报告首个分叉）。
2. 复用 `tools/oracle/shared/` 的 llama.cpp 构建 recipe（标量、单线程、关 Accelerate/
   Flash Attention），与 gliner2 的 `scalar.c` 模式一致。
3. 每个 arch 至少一条 `#[ignore]` 的 `matches_llama_cpp_bits` 测试，`RMI_*` 环境变量 gating。

#### 触发条件（满足任一才启动）

- 改动 `src/models/bert_family/` 或 `src/models/gemma_embedding/` 的任何数值路径；
- 换量化档（目前 7 个 arch 只有 Q8_0 验证过）；
- 用户报告 embedding 与官方实现不一致；
- 新接 BERT 家族变体（如 `jina-bert-v3`，源文件已在但没接）。

在触发前，`docs/MODEL_LIST.md` 对应行的口径保持"语义验证 + CLI/HTTP bit 一致"，**不写**
"逐位对齐 llama.cpp"。

### TODO-018: server 的两池超售（vision/audio 仍走 rayron）

核对 jina-embeddings-v5-omni 的 image 路径时发现两件事，都还没修。共同背景是
`src/core/thread_pool.rs:45-72` 记录的双池设计：LLM 走 `ComputePool`（显式分区、
无 work-stealing），vision/audio/qwen35 走 **rayron 全局池**（`into_par_iter` / `par_chunks_mut`）。
文档说两池靠 `src/main.rs:130` 的 `init_rayon_global_pool(n)` 对齐线程数，且
"never run concurrently ... so oversubscription is not an issue"。

#### 1. server 从不调用 `init_rayon_global_pool` → 线程数不可控 + 与 ComputePool 超售

CLI 在 main.rs 解析 `--threads` 后就建好 rayron 池；**`src/app/server/mod.rs::run_server`
完全没有这一步**。于是 HTTP 进程里第一个用 rayron 的请求会懒建池，线程数取 rayron
自己的默认（本机 `num_cpus`=18），而 ComputePool 是 `resolve_thread_count(--threads, 18)`
= clamp 到 `DEFAULT_THREAD_CAP=8`。

在 CLI 里这无害（vision 与 LLM 不并发）。在 server 里并发请求可以让 vision（rayron）
与 LLM（ComputePool）真正重叠，于是 18+8 个线程抢 8 核。

曾试过在 server 里补 `init_rayon_global_pool(8)`（2 行），**已撤回**，原因是：
  (a) 它不解决第 2 条的分歧（实测差异数仍是 153/1024，一点没变）；
  (b) 它只把超售从 18+8=26 降到 8+8=16，没有消掉超售本身；
  (c) `rayon::ThreadPoolBuilder::build_global` 只能成功一次，若别处先建了池则我的
      调用静默失效，属于脆弱写法。
根治手段是第 3 条。

#### 2. 〔已结案：误报〕jina-v5-omni image 路径 CLI/HTTP "1 ULP 分歧"是测量假象

一度记录为未决 bug，**实为比对方法错误，两侧逐位一致**。保留此节是因为那个坑
很容易再踩，且我踩了两次。

现象：CLI 打印（`print_embedding` 用 `{:.9}`）与 HTTP JSON（serde_json 对 f32 用
ryu 最短往返）逐元素比，**153/1024 个 `%.9f` 文本不同**，max abs diff 5.0e-10。

错在比法：把 HTTP 的十进制字符串解析成 **f64** 之后直接 format。该字符串是
"能 round-trip 回同一个 f32 的最短表示"，但它对应的 f64 落在 f32 精确值之上，
二次取整后末位进一。元素 0 实证：

    f32 精确值 = 0.0010754974791780114   (bits 3a8cf7b5)
    Rust {:.9}            -> 0.001075497   (正确：第 10 位是 4，舍)
    JSON 最短往返字符串    -> 0.0010754975
    Python format(f64)    -> 0.001075498   (错误：f64 落在 .5 上，进)

正确比法：先 `struct.pack('<f', ...)` 还原成 f32，再 format。改用后
**0/1024 不一致**。旁证：`embedding.final` trace 二进制两份 md5 相同、
HTTP JSON 与其 trace 1024/1024 位一致、CLI 文本与其 trace 1024/1024 一致。

教训（写给你我他）：**跨 `%.9f` 文本与 serde_json 浮点比较，必须先过 f32。**
本轮先是在 BERT/Gemma encoder 上正确绕过（0/1024），随后在 jina image 路径上
又直接 format f64，顺着错方向排查了很多轮（rayron 线程数 / `--threads` /
`--max-context` / 图片路径 / vision 逐位比对都做了，全是白工）。

#### 3. vision 迁到 ComputePool（`thread_pool.rs` 已登记的优选方向）

`src/core/thread_pool.rs:66-72`：未来统一方向是"把 audio/vision/qwen35 迁到
ComputePool，不是把 LLM 迁去 rayron"。迁移面：
- rayron 用点：`src/models/qwen3/vision/mod.rs` 3 处（1094 attention 的
  `into_par_iter`、213/278 两个 `par_chunks_mut`）、`src/models/qwen35/vision/mod.rs` 1 处；
- 需要穿 `pool` 的公共 API：`encode_pair` / `encode_image`（13 个调用点，分布在
  `app/qwen_drive.rs`、`app/omni.rs`、`app/text/multimodal.rs`、`app/text/vision.rs`）
  与 `VisionEncoder::from_source` / `VisionEncoder35::from_source`（10 个构造点）；
- 迁移后 rayron 依赖可整体从 Cargo.toml 移除，第 1 条的"两池对齐"问题随之消失。

注意：**这一条不修第 2 条**（vision 输出已证逐位相同）。它是第 1 条的根治手段。

### TODO-019: token 维寄存器分块，消除 per-token 权重重读

**目标**：让一次权重载入服务多个 token，把权重流量除以分块宽度 NR。图像编辑的文本
编码与去噪合计占端到端 ~95% 的时间，而算力利用率只有峰值的 ~11%——不是算不动，是
权重搬不过来。

#### 现状

`BF16Kernel::forward_batched`（`src/ops/kernel/bf16/mod.rs`）与
`F16Kernel::forward_batched`（`src/ops/kernel/f16/mod.rs`）都是：

```rust
for token in 0..n_tokens {
    self.forward(&input[token * n_in..], &mut output[token * n_out..], ...);
}
```

每个 token 独立做一次 matmul，**整个权重矩阵每个 token 从 DRAM 重读一遍**。
`qwen3::trunk::text_encode_forward` 的 `project_tokens`（本分支新加，按 token 分区到
`ComputePool`）虽然把并行度做上去了，但每个 token 内部仍是
`forward_prepared(..., 0, 1)` 的单行调用，权重流量一分没省。
`longcat::dit::Layer::forward` 是同样的 `chunks_exact` 逐 token 结构。
`qwen35::vision::matmul_weight_batch_pooled` 同理——分区解决并行度，不解决流量。

以 LongCat 文本编码（Qwen2.5-VL-7B，512 token）估算：每层 7 个投影、单投影权重
3584×3584×2 B ≈ 25.7 MB，512 token × 7 × 25.7 MB ≈ **2.6 TB** 权重流量。

#### 影响

本机（20 核 aarch64，`target-cpu=native`，release-fast）Turbo 32×32 / seed 42 / 20 线程：

| 阶段 | 耗时 |
|---|---|
| 文本编码（512 tok） | 158 s |
| 去噪 | 291 s/步 |
| vision | 22.6 s |
| VAE | 0.10 s |

文本编码 158 s 跑约 7.3 TFLOP ≈ **46 GFLOP/s**，相对 20 核 NEON 峰值（~400 GFLOP/s）
只有 **~11%**。分块宽度 NR=4 理论上把权重流量除以 4。

#### 何时触发

任何 prefill / 多 token 场景（图像编辑的文本编码与去噪、embedding、batched prefill）。
**autoregressive 单 token decode 不受影响**（n_tokens=1，没有可复用的维度），
所以这条不影响生成速度，只影响 prefill。

#### 选项

1. **在 kernel 内加 token 分块入口**（推荐）。新增
   `forward_batched_tiled(&self, input, output, n_in, n_out, ith, nth)`：对每个权重行，
   一次载入后用 NR 组累加器算 NR 个 token，然后 `ith/nth` 按**输出行**分区。
   调用方把现有的"按 token 分区"改成"按行分区 + kernel 内部分块"。
2. **调用方自建分块缓冲**。不改 kernel，在模型层把 NR 个 token 的输入打包成一个
   `n_in*NR` 的行——但这会改变 matmul 的数学形状（N 那维变成假的），权重布局不匹配，
   实际等于把权重复制 NR 份。**不可行**。
3. **转置权重为 token-major**。一次性重排权重让 token 维连续。改动面大、加载变慢，
   且对 mmap 的 GGUF 不友好。

#### 推荐

选项 1。理由：分块是纯粹的访存优化，不改权重精度、不改输入精度、不改累加顺序之外的
任何数学，因此**不触碰任何逐位对齐契约**——与本分支已建立的"重结合可以、近似不行"
原则相容。

实施注意：
- `project_tokens` 当前按 token 分区，改成按行分区后 dispatch 次数会变多
  （QKV 从 1 次变 3 次）。需要权衡 dispatch 开销与流量节省，NR=4 时预计净收益为正。
- Q8_0 路径（`q8_0::dispatch`）已经按 NRC4 一次算 4 个**输出行**，思路相同；
  缺的是复用同一权重的**多个输入行**，两者是正交的分块维度。
- 与 [TODO-018](#todo-018-server-的两池超售visionaudio-仍走-rayron) 第 3 条
  （vision 迁到 ComputePool）相关但**不同**：那条解决线程池归属，这条解决权重流量。
  建议先做这条——它不依赖线程池迁移。
- AuK 那边已记了同类问题（"Per-token matmul batching（`linear_into` 一调一次 token，
  瓶颈）"），两条应共用同一个 kernel 侧分块实现。

#### 关联文件

- `src/ops/kernel/bf16/mod.rs`（`forward_batched`）
- `src/ops/kernel/f16/mod.rs`（`forward_batched`）
- `src/ops/kernel/q8_0/dispatch.rs`（已有的 NRC4 输出行分块，可作参照）
- `src/models/qwen3/trunk/forward.rs`（`project_tokens`、`RowsWrap`、`TokenProjection`）
- `src/models/diffusion/longcat/dit.rs`（`Layer::forward` 逐 token 结构）
- `src/models/qwen35/vision/mod.rs`（`matmul_weight_batch` / `..._pooled`）
- `src/core/thread_pool.rs`（`compute` / `row_range`）

---

## Archived / completed work

### ERNIE-Image / Turbo (8B, single-stream DiT) — CPU 实验支持 (2026-10-05)

本次真实文件为 `unsloth/ERNIE-Image-GGUF/ernie-image-Q4_K_M.gguf`，
5,019,124,416 bytes，SHA256 `ed43d36ab45df0ef24e55d88e72bea241d0d54e4006ce0442386aba071b04c2a`。
该 GGUF 错标 `general.architecture=wan`，入口根据完整 ERNIE 张量签名识别。
参考固定在 stable-diffusion.cpp `3f8527a46c54ecf4cb4ed6003da8e8982283c73c`，
ggml `89c4413f5da6fb20cc796f16033d37f129be81fd`。

- DiT：36 层，hidden=4096，32×128 heads，FFN=12288，`GELU(gate) * up`。
  共享 6 路 AdaLN、RMSNorm、image-first/text-last 联合注意力、3D RoPE [32,48,48] / theta=256。
- 文本：Ministral-3-3B-Instruct-2512 Q4_K_M，26 层 / 3072 hidden，Tekken，原始 prompt + BOS；取 hidden_states[-2]，只执行前 25 blocks。
  参考图使用普通 full-width NeoX RoPE / base=1e6、causal attention，输出倒数第二层且不执行 final RMSNorm。
- VAE：`Comfy-Org/ERNIE-Image/vae/flux2-vae.safetensors`，转换入口
  `python -m tools.converter.ernie_image.convert_vae INPUT OUTPUT`。
  Conv 权重 F16，Attention 的四个 Linear 与向量保持 F32。
  128 packed channels 反归一化 → pixel shuffle → 32 channels → post_quant_conv → 共享 Flux decoder。
  latent 边长为输出分辨率 / 16；1024×1024 对应 4096 个 image tokens。
- 采样：discrete flow schedule / shift=4，Euler 积分 velocity；普通模型默认 32 steps / CFG 5，Turbo 默认 8 / CFG 1。
- 已有主 CLI 入口、真实组件契约测试、RoPE/调度/残差/Flux2 unpack/CFG 检查。
  64×64、1 步、CFG 1、seed 42、单线程的完整生成链路 93 个检查点 / 6,440,960 个 F32 原始位一致，
  包含文本编码、噪声、36 层 DiT、Euler latent 和完整 VAE RGB F32。
- 2 步 / CFG 5 的条件/空提示词两路、完整 DiT、采样和 VAE：261 个检查点 / 16,132,096 个 F32 原始位一致。
- 远端 256×256 / 32 步 / CFG 5 / seed 42 / 16 线程运行成功，1027.489 s；样图有桌面和窗景但漏猫，质量验证未通过。
- 可复现工具：`tools/oracle/ernie_image/`，固定版本、隔离插桩、禁用 SIMD/FMA/融合、严格 u32 比较。
- 待完成：Turbo 独立权重、SIMD/FMA 数值对齐及 512/1024 生成质量验证。
  CPU 路径；尚无 ERNIE Vulkan/GPU 支持验证，不能继承 Z-Image 的加速结论。

### AuK-Base (1.5B, Flux2Edit 音频 DiT) 适配 — 🚧 进行中 (2026-10)

`general.architecture = auk`（audio.cpp 的 community model；GGUF 通过 `transformer.transformer_blocks.0.img_attn.qkv.weight` 这类 tensor 命名探测，参考 `references/audio.cpp/src/community_models/auk/{flow,conditioning,vae,audio_conditioning,session}.cpp` 116 KB 参考实现；HuggingFace 镜像源 `audio-cpp/AuK-Base-and-Flash-GGUF`）

核心架构（`audio.cpp/docs/community_models/auk.md` + `config/auk-base.yaml`）：
- Backbone = **Flux2Edit**（Flux 风格双流 DiT，跟 `pig` (Z-Image) 同一家族）
- `dim=1536, heads=24, head_dim=64`，`ff_mult=2 → ffn=3072`
- 10 个 double blocks（并行 img/txt 注意力 + MLP）+ 20 个 single blocks（仅 img 注意力）
- `text_hidden_dim=2048`（Qwen2.5-Omni-3B n_embd）
- VAE = `BigVGANFlowVAE`，64-dim latent，480× 下采样，24 kHz 输出
- CFMEdit = flow-matching 编辑变体，logistic_normal schedule（P_mean=-0.8, P_std=0.8）
- AuK-Base：32 Euler 步 + guidance 2.0
- AuK-Flash：固定 4 步 + guidance 0（蒸馏版；本轮暂不 port）

GGUF 重量（从 `audio-cpp/AuK-Base-and-Flash-GGUF`）：
- `auk-base-f16.gguf` 2.9 GB（F16 DiT，本轮首选）
- `auk-base-q8_0.gguf` 1.5 GB（Q8_0 量化）
- `auk-vae-f32.gguf` 608 MB
- `qwen2.5-omni-3b-q8_0.gguf` 4.0 GB（**本地已有** `models/Qwen2.5-Omni-3B-GGUF/Qwen2.5-Omni-3B-Q8_0.gguf`）

进度：
1. ✅ MODEL_LIST.md 加行 + TODO.md 占位
2. ✅ 下 AuK GGUF + inspect tensor 命名（vs audio.cpp 期望），自校
3. ✅ `src/models/diffusion/auk/{mod,dit,vae,text}.rs` scaffold（参照 pig 的 Flux 双流 + AuK 专有 CFMEdit 逻辑）
4. ✅ CLI dispatch + arch detection + 测试钉 contract（6/6 auk_di_t_q4_k_m 通过）
5. ✅ 端到端冒烟：`--model auk-base-f16.gguf --text-encoder Qwen2.5-Omni-3B-Q8_0.gguf --vae auk-vae-f32.gguf --prompt "..." --out speech.wav`（4 步 ~8.5 min，生成 0.13s 24 kHz mono WAV）

实现要点（commit `899602c..dfd74b9` on `Auk` branch，共 12 commits ~3600 LoC）：
- DiT：Flux2Edit 10 double + 10 single blocks，6-way AdaLN modulation，joint img+text attention，per-head RMS QK-norm + RoPE，packed gate+up SwiGLU FF
- Text encoder：Qwen2.5-Omni-3B（直接复用 `qwen3::trunk::Qwen3Model`）
- Diffusion：linear sigma schedule，Euler step `x_next = x + v * dt`，velocity clipping (band-aid)，CFG with zero-text unconditional pass
- VAE：BigVGANFlow decoder — conv_pre (64→1536) + 6× upsample stages + 18 resblocks (3/stage) + conv_post (24→1)；weight-norm decomposition；SnakeBeta with GGUF-stored alpha/beta
- WAV：16-bit PCM mono writer

待优化（follow-up，no rush）：
- [ ] **Vulkan 路径的渲染不是逐位可复现的**（2026-11-04 定位）— 同一进程内同一份 prompt + seed 连着渲染：CPU 路径 8/8 逐位相同，Vulkan 路径约 6 次里有 1 次不同（latent rms 1.315 vs 1.162，更差时 rms 2.33）。用 `RMI_AUK_DIT_DUMP` 逐张量比对定位到：`text_conditioning.f32`、`latent_in_0/1.f32`、`velocity_0/1.f32` 都逐位相同，**差异从第 3 个 Euler 步（`velocity_2.f32`）开始**，`velocity_cond_2.f32` 同样分叉 —— 采样参数、随机数、文本条件都是确定的，不稳定性在 DiT 的设备路径内部（嫌疑：tiled F16 kernel 的尾块 / scratch 处理，或共享 arena 的复用）。CLI 每次进程只渲染一次，用户拿到的是"抽签"结果；HTTP 服务连续渲染时更容易撞见。回归哨兵 `tests/auk_pipeline_repeat.rs`（`#[ignore]`，CPU 过 / Vulkan 偶发失败）。**与 HTTP 接入无关**，是 GPU 路径的既有问题。
- [ ] 数值校准 oracle-diff vs `references/audio.cpp/src/community_models/auk/flow.cpp` — block forward magnitudes drift without the velocity clip
- [ ] Per-token matmul batching（`linear_into` 一调一次 token，瓶颈；需新增 batched Q8_0 matmul helper 或 thread-local Q8Scratch per rayon worker）
- [ ] VAE resblocks 的 causal padding（当前用 symmetric，lossy 一点；low impact 因 upsample 已用 linear interp）
- [ ] VAE SnakeBeta 的 FIR up/down filters（audio.cpp 用 `build_activation`，我们 skip；lossy 一点，minor impact）

性能现状（4 steps + CFG=2.0 + 4 threads + Intel MTL + `--gpu`，**release 模式**）：
- text_encode: 1.6s（CPU；Qwen2.5-Omni qwen2vl arch 还没 Vulkan）
- denoise: 40s（F16→Q8_0 pre-quant + Vulkan Q8 GPU matmul；4.1x vs 168s CPU F16）
- vae_decode: 50s（ComputePool par_chunks over resblocks，已 2.86x vs 166s serial）
- total: 93s（5.5x vs 515s baseline）

vs 起点（515s/0.13s）：5.5x end-to-end 加速。

注：audio.cpp 的 C++ 代码是 **参考** 而非金标准 — tensor 命名、参数化约定可能跟实际 GGUF 有微小差异（参考 ERNIE-Image 跟 unsloth GGUF 的踩坑先例），以 **GGUF 实际 tensor 名 + shape** 为准。

### Spark-X2.5-1.7B / 4B-GGUF 适配 — ✅ 已完成（功能）+ ⚠️ 性能待优化

`arch=spark2_5`，Xunfei Spark 2.5 讯飞星火。适配点：
- **Fused QKV**：`attn_qkv [n_embd, n_embd_qkv]` 单一投影 → split 为 Q/K/V。
- **Per-layer heterogeneous RoPE**：full-attn `n_rot=64, freq_base=5M`；SWA `n_rot=256, freq_base=10K`。新增 `rope_neox_partial`。
- **Sliding window mask**：SWA 层只看最后 `sliding_window=512` 个位置。
- **Per-head gating**：`attn_gate` pre-attention norm → sigmoid → 逐 head 乘 attention output。
- **GeGLU FFN**：`gelu(gate(x)) * up(x)` 后 down。
- **Tied embeddings**：GGUF 无 `output.weight` → 复用 `token_embd.weight`。
- **Tokenizer pre=`spark2_5`**：与 qwen2 正则不同，新增 `PreTokenizer::Spark2_5`。

冒烟通过。1.7B ≈ 1.1 tok/s, 4B ≈ 0.4 tok/s（4 线程；生产路径未优化）。

待优化：
- [ ] Per-token prefill O(N²) 注意力 → batched prefill
- [ ] 每次 `decode_step` 全量 `Vec::new` → ExecutionScratchpad
- [ ] Generate 阶段没有 KV cache 预热
- [ ] FFN gate 输入源（pre-attention norm vs post-attention hidden）
- [ ] GeGLU 用 `0.5 * x * (1 + tanh(...))` 近似 vs llama.cpp 默认精确 GELU
- [ ] chat template 整段手写，未走 `tokenizer.chat_template` 的 Jinja 引擎
- [ ] 精度对齐 XFllama.cpp oracle
- [ ] `get_f32_tensor` 重复 6 处（暂不抽，等待 `RequiredTensor` trait 落地）

---

## SIMD extension inventory

This project targets the following SIMD instruction sets:

| Architecture | Extensions used | Detection | Files |
|---|---|---|---|
| **x86_64** (Intel Core Ultra 5 125H, AMD Zen 4, etc.) | SSE2 / SSE4.2 / AVX / **AVX2** / **FMA** / **F16C** / **BMI2** | `is_x86_feature_detected!` | `src/ops/simd_avx2.rs`, `src/ops/kernel/*/avx2*.rs` |
| x86_64 (Skylake-X, Ice Lake, Sapphire Rapids, Zen 5) | + **AVX-512F** / **AVX-512BW** / **AVX-512VNNI** | runtime gate | (not yet wired) |
| x86_64 (Intel Alder Lake+, Zen 5) | + **AVX-VNNI** (`vpdpbusd`/`vpdpwssd`) | CPUID `avx_vnni` | `src/ops/kernel/q8_0/avxvnni.rs`（仅参考实现，实测更慢，dispatch 未接线——见 TODO-AVX-VNNI） |
| x86_64 (Arrow Lake, Lunar Lake, Granite Rapids, Zen 5) | + **AVX-VNNI-INT8** (`vpdpbssd`) | CPUID `avxvnniint8` | (not yet wired — see TODO-AVX-VNNI below) |
| **aarch64** | NEON / NEON-F16 / NEON-F32 | `is_aarch64_feature_detected!` | `src/ops/kernel/*/neon*.rs` |
| aarch64 (Apple M1+) | + NEON-DOT / SME | runtime | (not yet wired) |

### AVX-VNNI / AVX-512 background

**AVX-VNNI**（CPUID `avx_vnni`，Intel Tiger Lake 2020 / AMD Zen 5 起支持；dev box
Core Ultra 5 125H, Meteor Lake 有此位）提供 **`vpdpbusd`（unsigned×signed int8 dot，
`_mm256_dpbusd_epi32`）** 与 `vpdpwssd`（i16×i16）。每条指令把 32 个 int8 乘积按
8 个 int32 lane 水平累加。**注意**：`_mm256_dpbssd_epi32`（signed×signed）、
`_mm256_dpbuud_epi32` 等属于 **AVX-VNNI-INT8**（CPUID `avxvnniint8`，Granite Rapids /
Zen 5），是另一个更少见的 CPUID 位——在只有 AVX-VNNI 的机器上执行直接 SIGILL。

**AVX-512 VNNI** (`vpdpbusd` with 64-lane / 32-int32 ops on zmm registers) is the server-class
extension of the same idea: 2× the throughput per instruction and the only way to reach
> 50 tok/s on Qwen3-0.6B decode. Sapphire Rapids, Zen 5, and Zen 6 (Medusa Ridge) support it;
our dev CPU does not.

### TODO-AVX-VNNI: int8 dot 加速

> 状态：**开放项**。原条目前提已修正——`_mm256_dpbssd_epi32` 属 AVX-VNNI-INT8（非
> AVX-VNNI），plain AVX-VNNI（`vpdpbusd`）本机实测更慢已否证；真正的目标是
> `avxvnniint8` 的 `vpdpbssd`（见下方"待做"）。

**原始目标**：在 `q8_0/avx2.rs` 的 int8 dot（`_mm256_maddubs_epi16` → `_mm256_madd_epi16`）
基础上用 AVX-VNNI 缩成 1 条指令，预期 1.3-1.5×。

**调查结论（2026-11-21，Core Ultra 5 125H / Meteor Lake 实测）**：

1. **原条目指令名写错了**。`_mm256_dpbssd_epi32`（signed×signed int8 dot）属于
   **AVX-VNNI-INT8**（CPUID `avxvnniint8`，Granite Rapids / Zen 5），不是 AVX-VNNI；
   在 dev box 上按原计划实现直接 `SIGILL`（gdb 定位到 `vpdpbssd %ymm`）。
   AVX-VNNI（CPUID `avx_vnni`）只有 `vpdpbusd`（unsigned×signed）/ `vpdpwssd`（i16×i16）。
2. **AVX-VNNI 无法去掉 sign fixup**。Q8_0 权重和 activation 都是有符号 int8，`vpdpbusd`
   要求第一操作数无符号，`_mm256_sign_epi8` ×2 必须保留；能省的只有
   `maddubs → madd` 两条里的 1 条，还要多 1 条 `vpxor` 清零三操作数累加器。
3. **实测更慢**。release 单线程、Qwen3-0.6B `blk.0.attn_q`（1024×2048）交替测量 5 轮：
   AVX2 kernel 5.4-5.8 ms / 100 次 vs AVX-VNNI kernel 12.5-13.2 ms / 100 次
   （**0.42-0.47×**）。Meteor Lake 上 `vpdpbusd ymm` 的 uop 成本高于流水良好的
   `vpmaddubsw` + `vpmaddwd` 组合。

**落地产物**：`src/ops/kernel/q8_0/avxvnni.rs`（`vpdpbusd` 版 kernel + 3 个测试：uniform、
±127 极端值、Qwen3-0.6B 真实权重）。kernel 与 AVX2 路径**逐位一致**（两者每 block 的
int32 点积都精确、无 lane 可饱和），测试守住了这个断言。**dispatch 保持 AVX2 默认**；
该文件仅作测量记录，任何机型启用前必须重测 A/B。

**顺手修的契约边界**：`_mm256_sign_epi8(-128, …)` 在 int8 里无法取负（|-128| 仍是 -128），
权重或 activation 含 -128 时 sign fixup 会算错。仓库 Q8_0 量化器
（`quantize_q8_0_into_scalar_range`）clamp 到 ±127，真实权重文件同样是 ±127，生产不会
触发；`avx2.rs` / `avxvnni.rs` real-model 测试的输入生成已改到契约内（原来 `(i as i8)`
会造出 -128，且此前因模型路径 `../models/...` 在本机从未真正跑过）。

**待做（开放项，非否决）**：`avxvnniint8` 路径（`_mm256_dpbssd_epi32`，signed×signed
直接点积）——去掉 2 条 sign fixup、horizontal add 缩 1 条、无 -128 边角，结构上就是
原条目想要的收益。按机型分两档：

| 机型 | CPUID | 应做的 kernel |
|---|---|---|
| **Arrow Lake / Lunar Lake**（消费端，无 AVX-512） | `avxvnniint8` | ymm `_mm256_dpbssd_epi32` —— 那里**唯一可用的单指令 int8 dot**，原条目的预期收益在这档成立 |
| Granite Rapids / Zen 5 | `avxvnniint8` + AVX-512 | zmm `_mm512_dpbssd_epi32`（2× lane 宽），优于 ymm 版；即上方 SIMD 表中 AVX-512VNNI 未接线行的延伸 |

前置条件：本机（Meteor Lake）无法验证 `avxvnniint8`，需要一台目标机型实跑 parity
测试（复用 `avxvnni.rs` 的 3 测试模板：uniform / ±127 极端值 / 真实权重）通过后才能接
dispatch——`vpdpbssd` 在有该 CPUID 的机器上语义更简单（无 sign fixup），但**没实测过的
kernel 不接线**的纪律不变。

### TODO-LLAMA-PER-TOKEN-SIMD化

**目标**：扫描 `src/models/llama/trunk/forward.rs` 仍为 scalar 的 per-token 小循环，
逐个替换为 SIMD 版本。AVX2/FMA/NEON 已覆盖 `vec_scale_f32` / `vec_add_into` / `vec_mad_f32` 等基础算子。

**已审计的热点**（per token × per layer）：

| 操作 | 当前 | 行号 | 优化 |
|---|---|---|---|
| RMS norm grouped | ✅ SIMD via `rms_norm_grouped` → `scale_mul_avx2` | 462, 750 | - |
| Q8_0 quantize | ✅ SIMD via `quantize_q8_0_into_avx2` | 464, 766 | - |
| Q8_K quantize | ✅ SIMD via `quantize_row_q8_k_avx2` | 472, 772 | - |
| BF16 × Q8 matmul | ✅ AVX2 (`matmul_bf16_vs_q8_avx2`) | 484, 702, 804 | - |
| **Embedding scale** | ❌ scalar `for v in x.iter_mut() { *v *= scale; }` | 419-422 | ✅ **已替换为 `vec_scale_f32`**（commit 见 K2check 分支） |
| RoPE (k2-horizon neox) | ✅ SIMD `rope_neox_inplace_avx2` | 528-544 | - |
| KV cache write (F32) | ✅ `copy_from_slice` (memcpy → SIMD 路径) | 577-583 | - |
| KV cache write (F16) | ✅ `f32_slice_to_f16_avx2` (F16C) | 559-568 | - |
| Attention (F16) | ✅ SIMD `dot_f16_f32`, `vec_mad_f16_f32`, `vec_scale_f32` | 609-630 | - |
| Attention (F32) | ✅ SIMD `dot_f32`, `vec_mad_f32` | 636+ | - |
| Residual add | ✅ `vec_add_into`, `vec_mad_f32` | 720-724, 910-914 | - |
| silu_mul | ✅ SIMD `silu_mul_approx_inplace` | 831, 840 | - |
| **KV cache / scores buffer size** | ❌ hardcoded `512.min(cfg.n_ctx)` + `[0.0f32; 512]` | 117, 941 | ✅ **已替换为 `cfg.n_ctx.min(max_context)` + `vec![0.0f32; max_ctx]`**（`--max-context` CLI 参数，默认 8192，见 [`docs/usage/hunyuan.md`](../usage/hunyuan.md)） |

**结论**：除 `embedding_scale` 外 llama trunk 已是 SIMD 化。`embedding_scale` 修复已落地 K2check 分支
（每个 token 节省约 1.3 µs @ n_embd=1536，对总推理时间影响 < 0.1%）。

剩余 per-row attention 在 autoregressive decode 时是 sequential（依赖前面 token 的 KV），
SIMD 不能跨 token 加速——这部分要看 [TODO-010](#todo-010-prefill-runtime-通用化避免每个模型重复实现-chunk-调度) batched prefill。

### LLama trunk matmul 真实瓶颈（per-token）

解码时 n_out=1，所有 matmul kernel 的内并行（按 n_out 切 rows）失效——线程利用率 12.5%（1/8）。
**这是 autoregressive decode 慢的真凶**，不是 SIMD 缺位。

修复路径见 TODO-010（batched prefill）+ 自动接受单 token decode 慢的现实（llama.cpp 同样慢）。

### Standard gemma3 trunk (`src/models/gemma3/trunk/forward.rs`) — DONE for 270M-it

`src/models/gemma3/` 现在服务 **BitNet 270M** (file_type=40) **和**
**standard gemma3-270m-it** (file_type=15, Q4_K_M mixed) — 两条 forward
path 共享同一个 `text_encode` 函数体，`cfg.is_bitnet` 在 7 个投影
（attn_q/k/v/output + ffn_gate/up/down）处分发：

- `is_bitnet=true`  → `bitlinear_projection(...)` (BitNet path，SIMD hot path)
- `is_bitnet=false` → `standard_projection(...)` (Q4_K/Q5_0/Q6K/Q8_0 matmul，
  Q8_0-quantized 激活 × mixed-quant 权重，跟 qwen3 trunk 同款)

新增 `Gemma3Config.sliding_window: usize` 字段（默认 0 = full causal），
`causal_self_attention` 在 mask `j ∈ [0, i + 1 - sliding_window)`。
Standard gemma3-270m-it 声明 `sliding_window=512`（hybrid local/global attn）。

`Gemma3LayerWeights` 同时存 `bitlinear: BitLinearSlot` 和
`wq/wk/.../w_down: Option<StdProjection>`（owned `Vec<u8>` + ggml type + shape）。
Forward 路径在调用 `standard_projection` 时按需构造 `Weight` kernel
（cheap; kernel 只是 byte buffer 的 view + 维度）。

`static_weight` (`src/models/gemma3/trunk/weights.rs`) 现支持 F16 (BitNet 270M)
和 Q8_0 (standard gemma3-270m-it，mixed-quant conversion) 两种 token
embedding。Q8_0 dequant 用 34 bytes / 32-element block layout
(2-byte F16 scale + 32 signed int8)。

`build_config` (`src/models/gemma3/embedding.rs`) 改：
- vocab 从 `tokenizer.ggml.tokens` array length 读（standard GGUF 没有
  `gemma3.vocab_size`；BitNet 仍用 `gemma3.vocab_size` fallback）。
- `gemma3.context_length` 改 unwrap_or(0)（standard GGUF 总是有，BitNet 也总有；fallback
  仅为防御）。
- `gemma3.pooling_type` 改 unwrap_or(1)（standard GGUF 不设，BitNet 设 1）。
- `gemma3.attention.sliding_window` 新字段。

`app/mod.rs::compute_embedding` / `run_embedding` dispatch 加 `"gemma3"` case
→ `crate::models::gemma3::compute_embedding` / `run_embedding`（之前默认
fall through 到 qwen3，不识别 arch）。`models::gemma3` 现在 pub mod。

`models::gemma3::trunk::BitLinearWeights` 引用从 `crate::ops::bitlinear` 改到
`crate::ops::bitnet`（旧的 `bitlinear/` 模块已不再 pub；commit e011536
之后一直未更新 gemma3 trunk 的 import，这是 e011536 的 holdout）。

**e2e** (`tests/gemma3_270m_it_e2e_embed.rs`) 5/5：
shape=640、finite、value range `[-300, 300]`、L2 norm `∈ (0.1, 1000)`、
byte-deterministic、longer prompt 32 tokens 仍 finite、cooking vs
software cosine sim < 0.99（IT 模型 + last-token pooling discrimination
较弱；0.99 threshold 是宽口径，只 catch 完全相同 embedding）。

**性能**：standard gemma3-270m-it Q4_K_M 单 token 4 线程 ~1.0 s；BitNet-270M
packed SIMD 路径 ~480 ms。两者 forward 时间在同一数量级 — Q4_K matmul
是 AVX2 + Q8_0-quantized activation，跟 SIMD int8 × int8 dot 在内存带宽上
类似。

**未做**（4B/12B/27B standard gemma3 的前置）：

1. KV cache + decode loop（生成模式；当前只有 embedding extraction = last-token）。
2. IT chat template（`tokenizer.chat_template` 已在 contract test 里锁定含
   `<start_of_turn>{role}
...<end_of_turn>
`，但 `run_shared_inference` 仍
   返回 error）。
3. **Bit-exact 对照 oracle**：本机没有 cmake/PyTorch/bitnet.cpp build；任何
   "对得上 llama.cpp b96806d --temp 0 --top-k 1 golden log" 的工作需 cmake
   工具链，目前不可达。Forward 通过有限 golden test 验证（shape / finite /
   range / determinism / discrimination）— 跟 BitNet-270M 的 e2e 套件同级别。

针对 gemma-3-270m-it 的 contract test (`tests/gemma3_270m_it_q4_k_m.rs`)
已 8/8：metadata (arch, file_type=15, dims)、sliding_window=512、
SPM tokenizer (`model=llama`, no merges, EOS=106)、mixed-quant tensor
inventory (109 F32 + 81 Q5_0 + 27 Q4K + 9 Q6K + 10 Q8_0 = 236 张量, no I2_S, no norm_in)。

历史：

1. `causal_self_attention` 加 sliding window mask（`gemma3.attention.sliding_window=512`，
   屏蔽 `|i - j| > sliding_window` 的 K/V）。
2. 在 `gemma3/trunk/weights.rs` 加非 BitLinear 路径：`attn_q/k/v/output` +
   `ffn_gate/up/down` 用 `Weight::from_quantized`（Q4_K/Q5_0/Q6K/Q8_0 混合，按张量名匹配），
   走 `matmul_q8_0_quantized_parallel_rows`（现有的 standard matmul，已在 qwen3 trunk 用过）。
3. `Gemma3Config::is_bitnet` field 保留作 dispatcher，但加一个
   `pub fn new_from_source(&self) -> ...` 的 standard gemma3 入口（`text_encode_v2` 或
   重命名）；`embedding.rs::compute_embedding` 走 `cfg.is_bitnet` 分发两个 path。
4. IT tokenizer：`tokenizer.chat_template`（含 `<start_of_turn>{role}\n…<end_of_turn>\n`）
   + EOS=106（不是 BitNet 的 EOS=1）已经在 contract test 锁定；`BPETokenizer` 已经能处理
   SPM `tokens + scores + token_type`（这是 BitNet-270M 复用过的），IT 模型跟 base 模型
   tokenizer layout 完全一致，理论上零增量工作量。
5. KV cache + decode loop（用于 IT generation 而非仅 embedding extraction）。

rough scope: ~400-500 LOC（取决于要不要 KV cache / generation），
与 `tests/gemma3_270m_it_q4_k_m.rs` 的 8 条 contract test + 端到端 forward parity test 配对
（参 llama.cpp `--temp 0 --top-k 1` greedy 输出做 32→1 字符串对齐；oracle 见 llama.cpp `b96806d`
在 `unsloth/gemma-3-270m-it-GGUF` 上的 golden log `main-0319c65`）。

---

### TODO-020: Vulkan SiLU 的 f32 次正规数缺陷 — ✅ 已修复，但饱和边界不可回退

#### 现象

`record_silu_mul` 一旦被写进同一 command buffer 的 dispatch 链（Z-Image DiT 的 FFN
合批），整幅渲染产出全 `-inf`：

```
Inference error: Non-finite Z-Image predicted flow: len=65536, nan=65536, -inf=65536
```

而同样的算子在**分开提交**时完全正确。历史上这个失败被归因为「descriptor set 与
pipeline 布局在跨 dispatch 复用时的状态」，并标注「根因未定位」——**那个归因是错的**。

#### 根因

`shaders/glsl/silu_mul.comp` 的 `exp_approx` 把 exp 输入 clamp 在 `±88.376`
（exp 的溢出点）。但 silu 的补偿算法需要 `1/denominator` 落在 f32 **正规数**范围：

```glsl
den = 1 + exp(-gate);
act = gate / den;
res = fma(-den, act, gate);     // 灾难性抵消，res 是残差
act = fma(res, 1/den, act);     // 用 1/den 修正
```

| clamp | `den` | `1/den` | 状态 |
|---|---|---|---|
| 88.376 | 2.406e38 | **4.157e-39** | **低于正规数下限 1.175e-38** → 修正项退化 |
| 80 | 5.541e34 | 1.805e-35 | 正规数 → 精确 |

`gate ≤ -88` 时修正项产生 NaN。Z-Image FFN 的 gate 实测 amax 到 962、
min 到 -197，所以必然踩中。

**为什么藏了这么久**：自检 `check_operator_chain` 用 `N_FF = 96` 覆盖这个 shader，
而真实 FFN 是 `3840 × 9728 = 37 324 800`——**38.9 万倍**，且小 fixture 的输入从未
接近边界。**这个算子在生产规模下从未被验证过**，任何模型在 GPU 上调用它都会中招
（`record_silu_mul` 是共享算子，不只 Z-Image 用）。

#### 修复

clamp 从 `±88.376` 改为 `±80`，让 `1/den` 保持正规数。新增
`silu_ffn_scale_probe`（`src/vulkan/ops.rs`）用 FFN 真实规模跑，断言零非有限，
并挂进 `run_qwen3_operator_check` 作为常规门禁。

修复前 `16 009 509 / 37 324 800` 非有限（43% 损坏），修复后为 0。

```bash
SILU_ONLY=1 ./target/release/examples/vk_ops_check   # 只跑这一项
```

#### ⚠️ 风险 1：与 CPU SiLU 不再逐位一致（已发生）

分两个区间，不是统一说法：

| 区间 | CPU | GPU（clamp=80）| 一致性 |
|---|---|---|---|
| `\|gate\| < 80`（绝大多数激活）| `x/(1+exp(-x))` | 同式，无 clamp | **逐位一致**，相对差 0 |
| `gate ≤ -88`（极少数）| `1+exp(103)` 溢出 f32 → `x/inf` = **-0** | 饱和在 `exp(80)` → **≈ -1.9e-33** | **绝对差 ~1e-33，相对差不成立** |

**CPU 在这个区间自己也是错的**（数学上 `silu(gate) → 0`，`-0` 是"意外对"），两边只是以
不同方式失效。1e-33 在 f32 里是噪声底，不会污染下游 matmul，但**任何拿 GPU 与 CPU
比对这个区间的测试必须用绝对容差**，用相对容差会误报。

实测代价（全部来自边界区间）：Z-Image 512×512 与 CPU 路径的 PSNR 从
**33.34 dB 降到 31.52 dB**（mean|Δ| 2.81 → 3.61）。人眼核查差异不可见。

**若要两边数学同解**：对 `gate < -80` 直接返回 0（真值就是 0），`|gate| < 80` 的
精确路径不动，这样边界区变成逐位的 `-0`。**未做**——当前状态已足够，
且这是额外改动而非修 bug。

#### ⚠️ 风险 2：clamp 值不可随手调高

看到「88.376 是 exp 溢出点，调回去更精确」是很自然的直觉——**那正是这个 bug**。
`1/den` 必须是正规数，这是硬约束，不是保守取值。

#### ⚠️ 风险 3：任何模型在 GPU 上用 SiLU 都继承这个饱和

`record_silu_mul` 是共享算子。新增模型或新路径调用它时，**不能假设它与 CPU SiLU
等价**。代码注释（`silu_mul.comp` 的 PARITY NOTE）已写明。

#### 关联文件

- `shaders/glsl/silu_mul.comp` — `exp_approx` 的 clamp + SATURATION BOUNDARY 注释
- `src/vulkan/ops.rs` — `silu_ffn_scale_probe`（生产规模门禁）
- `src/models/diffusion/z_image/dit_gpu.rs` — `record_ffn`（FFN 合批，依赖本修复）
- `docs/usage/z-image.md` — 「FFN 合批的 `-inf`：已定位」记录排查全过程

---

### TODO-021: Z-Image FFN 合批 `record_ffn` — 已合入，依赖 TODO-020

一次 command buffer 录 w1 → w3 → SiLU → w2，取代原来「3 次 submit + 2 次 readback +
host 侧 3700 万次 exp」。denoise 52.7 s → 42.5 s，端到端 59.9 s → 49.7 s
（512×512，8 步，seed 42，aarch64 Vulkan 主机）。

**与 TODO-020 是同一件事的两端**：旧 clamp + 合批必然 `-inf`，两者必须一起看。

排查时已排除（别重复）：barrier 覆盖范围、提交粒度、Q8 暂存别名（w3 已加独立
`q8_up`）、region 重叠（bump allocator 不重叠）、validation layer（零警告，代码合规）。
真正的根因在 silu 的数值，不在调度。

`FFN_BATCH_MAX=<n>` 可限制合批生效的层数（默认全部），用于二分定位。

**遗留**：合批后与 CPU 基线 PSNR 31.61 dB（改动前 33.34 dB），差异来自 TODO-020
风险 1 的 silu 饱和。quantize reciprocal 的 1-ulp 修复（见
[VULKAN.md](VULKAN.md)「已知问题与调试开关」）已把该值从 31.52 拉回
32.21 dB（mean|Δ| 3.44 → 3.01），剩余差距仍主要来自 silu 饱和边界。

### TODO-024: attention 相关自检的容差是照 x86 标定的 — ✅ 由 TODO-025 一并解决

**根因**：`cpu_attention_scores` / `cpu_attention_values` /
`check_attention_scores_match_cpu_reduction` 拿 `dot_f16` 当 oracle，而

| 架构 | `dot_f16` | 累加 |
|---|---|---|
| x86_64 | `_mm256_cvtph_ps` + `_mm256_fmadd_ps` | **f32** |
| aarch64 | `fmla .8h` | **f16** |

容差是照 **x86 的 f32 参考**标定的（`attention_scores` 用 3e-5），在 ARM 上参考自带
`sqrt(head_dim) * eps_f16` 误差（head_dim=128 时约 5e-3），与实测 6.6e-3 吻合——
**原理上过不了**。

**解法不在放宽容差，在换 oracle**：统一改用 `reference_dot_f16`（f64 精确求和）。
oracle 是精确算术后，容差自然回到 3e-5 并且真的成立。详见 TODO-025。

**教训**：一个有损的参考值，会让「校验 GPU」这件事悄悄变成「比较两个有损实现」。
前者能发现 bug，后者只能发现巧合——而且它同时把容差推大到连真正的 GPU 回归都测不出来。

### TODO-025: `dot_f16` 在 aarch64 上用 f16 累加 — ✅ 已改，`ggml-f16-dot-parity` 保留旧行为

新增 Cargo feature `ggml-f16-dot-parity`（默认关闭）：

| | 累加 | 指令 | 吞吐(n=4096) | 相对误差 |
|---|---|---|---|---|
| 默认 | **f32** | `FMLAL` | 31 GFLOP/s | **8.5e-08** |
| `--features ggml-f16-dot-parity` | f16 | `fmla .8h` | **122 GFLOP/s** | 2.3e-03 ~ 3.4e-03 |

**吞吐代价实测约 4 倍**，这台机器的 FP16 单元确实比 f32 宽。所以这不是免费的
正确性提升，而是一个需要按调用点分别权衡的取舍——feature 就是为此留的：
校验/参考路径用默认档，吞吐敏感的推理路径可以显式打开旧行为。

（测量用的是逐点测试，n=4096、2 万次迭代；FMLAL 档每 32 元素用 8 次 `.4h`
装载而非 4 次 `.8h`，装载指令是 2 倍，因此其中一部分差距可能来自装载而非
FMLAL 本身吞吐。）

#### 「保留 16 精度」不等于「保留 16 位累加」

现状把两件事捆在一起了，但它们可以分开：

- **f16 存储**：模型格式与 ggml 对齐的要求，**任何配置下都不变**，带宽与显存
  收益完全保留。
- **f16 累加**：实现细节，**不是对齐要求**。决定性证据是 x86 路径
  （`_mm256_cvtph_ps` + `_mm256_fmadd_ps`）**从来不**做 f16 累加。同一份 ggml
  语义，两个架构行为不一致，说明这从来不是「对齐 ggml」，而是各自挑了个实现，
  再被一个逐位测试冻住。把 `fmla .8h` 当作对齐的代价保留下来，等于主动接受一个
  和 x86、和 GPU shader 都不一致的数值行为，只因为它先被写下来了。

`FMLAL` 是 Armv8.2 里「f16 乘、f32 累加」的正规指令，与 `fmla .8h` 同属 FEAT_FP16，
**不引入新的架构要求**。

#### 顺带修好的：自检参考不再依赖 build flag

原先 `cpu_attention_scores` / `cpu_attention_values` / `check_attention_scores_match_cpu_reduction`
拿 `dot_f16` 当 oracle——也就是拿一个「快速实现」当标准答案。累加器宽度一旦变成
构建选项，**打开 flag 就会让检查自动变弱**，而且它问的本来就不是「shader 对不对」，
而是「两个有损实现会不会碰巧对上」。

现在统一改用 `reference_dot_f16`（f64 精确求和）。结果：

| 检查 | 改前 | 改后 |
|---|---|---|
| attention_score_reduction | 容差被迫放大到 2e-3（现收回 3e-5）| **max_rel 0.000e0** |
| attention_scores | 容差 3e-5，实测 6.6e-3 失败 | **max_rel 3.4e-07** |
| attention_values | — | **max_rel 0.000e0** |
| quantize_tie_even | 自 621c622 起一直红 | exact=true |

`vk_ops_check` 现在 **exit 0、25 个算子全过**（连跑多次稳定）。

#### 被推翻的 pinned 测试

以下测试原本钉死 ggml 的 f16 累加结果，现改为按 feature 钉两套值，并补上
**对 f64 参考的精度断言**（只钉 bit 只能证明可复现，证明不了正确）：

- `neon_f16_attention_dot_matches_ggml_four_accumulator_reduction`
  → 仅在 `ggml-f16-dot-parity` 下编译；新增
  `neon_f16_attention_dot_accumulates_in_f32`（默认档，对 f64 参考 rel < 1e-6）
- `f16_dot_dispatch_matches_platform_reduction_contract` → 两套值 + f64 精度断言
- `f16_kernel_matches_ggml_scaled_input_contract`、
  `f16_kernel_scaled_path_matches_ggml_f16_input_and_native_accumulation`
  → 两套值 + 新增 `assert_f32_accumulated` helper
- `z_image::vae::conv_f16_matches_pinned_ggml_im2col_dot_and_post_bias`
  → 两套值（该测试**只**钉了值未补 f64 断言；精度由上一层的 dot 级测试保证）

#### 回归

`cargo test --lib`：默认与 `ggml-f16-dot-parity` 两个配置均为
**1137 passed / 29 failed**，与父提交**完全一致**，零新增失败。

#### 端到端代价（512×512 8 步，Z-Image）

| 阶段 | 改动前 | 改动后 |
|---|---|---|
| CPU vae_decode | 11.6 s | **15.2 / 15.9 s（+34%）** |
| CPU denoise | 波动极大 | 无关（本就不走 `dot_f16`）|
| GPU 端到端 | 48.99 s | 48.59 s（不变）|

**VAE 的 +34% 是真实回退**，两次复测都稳定复现：VAE 的 f16 conv 走
`F16Kernel::forward_scaled` → `dot_f16`，正好落在 FMLAL 慢 4 倍的那一档。
GPU 端到端不受影响，因为 DiT 走 Q8_0 matmul。

同时 **GPU/CPU 一致性改善**：PSNR 32.21 → **33.50 dB**，mean|Δ| 3.01 → **2.87**。
CPU 侧少了一个 f16 累加的误差源，自然更靠近 GPU。

所以这一档的取舍是明确的：**CPU VAE 多花 34% 时间，换 CPU/GPU 差 1.3 dB**。
两次 CPU 渲染逐字节一致，结果可复现。

#### 尚未验证

- ASR（gemma4 / qwen3 mel encoder）与 MLX LoRA 的端到端数值未跑（缺模型）。
  这些是 `dot_f16` 的真实调用方，其中 ASR 的 mel encoder 可能落在算力敏感区，
  4 倍差距是否可接受需要实测决定。
- FMLAL 档的装载指令数是否为瓶颈（见上）。

### TODO-026: 测试套件在满负载下有 flaky，且 CI 根本不跑测试

#### 现象

同一个未改动的提交，重复跑全量 `cargo test --lib --features vulkan`，失败集合会变。
2026-10-13 在 `ZImage3`（`8297e19`）上实测三次：

| 轮次 | 相对首次基线多出的失败 |
|---|---|
| 1 | 无 |
| 2 | `models::dots::vocoder::q8_convolution_matches_scalar_layout_bias_stride_dilation_and_batch_tail` |
| 3 | `models::qwen3::trunk::qwen3_cpu_prefill_matches_batch_one_at_chunk_boundaries` |

两个测试**单独跑时 13/13 全过**（改动前后各 8 次与 5 次），只有全量并行时才偶发。
已确认至少这两个，**完整 flaky 集合尚未枚举**。

#### 为什么这条比看起来严重

全量回归目前只能给出「与基线比对」这种相对结论，而这个基线本身在漂。于是：

- 任何一次单轮全量回归的结论都不可靠 —— 改坏了分不清是真回归还是 flaky，没坏也可能是 flaky 抵消了真回归
- 已经在实践中造成误判：本次差点把 `--features vulkan` 的结果（1220 passed / 39 failed）
  去比**不带** vulkan 的基线（1137 / 29），把 10 个正常差异当成回归；之后又差点把
  `qwen3_cpu_prefill` 的偶发失败归到自己的改动上，实际它在基线上同样会飘
- 新增测试的通过率也不可信：`silu_probe_uses_the_production_ffn_width` 这类断言，
  理论上「一次通过」不构成证据

#### 为什么一直没被发现

`.github/workflows/ci.yml` 只有两个 job：`fmt`（`cargo fmt --check`）和
`vulkan`（shader 校验 + `cargo check` 若干 target）。**没有任何 job 跑测试。**
所以这类问题只能靠本地人工回归撞上，而人工回归恰恰是最容易漏判的场景。

#### 推测成因（未验证）

两者都涉及线程池与分块边界：一个是 `ComputePool` 下的分块量化卷积，一个是
chunk 边界的 prefill。满负载时线程调度/分块顺序变化可能触发只在特定顺序下
才成立的假设。需要看代码确认，暂不下结论。

#### 待办

1. **枚举 flaky 集合**：对全量套件连续跑 N 轮（如 10），统计每个测试的失败频次，
   产出「偶发失败」名单
2. **定位根因**：逐个看名单上的测试是否依赖线程调度/分块顺序；能修则修
3. **暂时隔离**：修不了的先用 `#[ignore]` 或加显式注释说明已知偶发，
   并在名字/注释里写清触发条件，避免下一个人当回归查
4. **CI 补测试 job**：`cargo test --lib`（带/不带 `--features vulkan` 各一），
   失败重跑一次再判定——先把「有测试」这件事建立起来，重试策略随后再加
5. **方法论固化**：在 `docs/develop/OPTIMIZATION.md` 或本文档里写明
   **禁止用单轮全量回归做回归判定**，需要至少 3 轮且失败集合稳定才可下结论

### TODO-027: FFN 尾部的 norm/gate/residual 融合上设备 — ❌ 未完成，实现已丢弃，但四个坑要记住

目标：`readback_norm_and_residual_rows` 在每个 block 末尾把投影结果下载到 host、
做 rms_norm + 门控残差加、再传回设备。512×512 下每 block 每 site 是 59 MB 下行 +
59 MB 上行，一个 step 走两次。融合成单个 kernel 可以省掉其中一次。

**实现已完成到「自检通过」这一步，但接进主路径后成图是坏的（PSNR 10.03 dB，
改动前 GPU↔CPU 是 33.5 dB），因此没有提交。** kernel 本身在生产规模下是逐位正确的，
自检钉住了这一点；错的是它周围的接线。下面四个坑每一个都真实造成过可观测的失败，
而且**每一个的失败方式都不是「报错了」，而是「继续跑出一个看起来合理的错结果」**。

#### 坑 1：等长的两个 bound buffer 会被 glslang 折叠成一个

gamma 和 gate 都是 `HIDDEN` 宽、从 slot 1 和 slot 2 绑定、offset push 常量都是 0、
读取长度相同——编译器因此证明两次读等价，把 gate buffer 从 SPIR-V 里整个删掉：

    $ spirv-dis shaders/bin/fused_norm_gate_residual.spv | grep Binding
    OpDecorate %arena Binding 0
    OpDecorate %weight Binding 1        # Gate 整个不见了

于是 shader 用 gamma 当 gate 读，产出一张 PSNR 9.9 dB 但**结构完整**的图。
试过把索引写成不同表达式（`channel = offset % pc.width` 而不是 `offset`），
**无效**——折叠与表达式无关。最后把两个都改成从 arena 读、用各自的 word offset 才解决。

> **教训**：新增一个 bound buffer 后，用 `spirv-dis ... | grep Binding` 确认它还在。
> 这一步不能省，也不能靠肉眼 review shader。

#### 坑 2：`ArenaRegion.size` 是字节，不是元素数

`Layout::build` 里写 `gate_tanh: take(HIDDEN)?` 而其他行宽区域写的是
`rows * HIDDEN * 4`。`take` 收的是元素数，所以 3840 元素被登记成 3840 字节，
而 `write_f32` 要 15360 字节。结果每层都失败并 fallback：

    [GPU] Z-Image block 0 fell back to the CPU: ... host write region
          offset=8785664 size=3840 cannot hold 15360 bytes

denoise 从 41 s 变成 **3176 s**（77 倍）。自检里没暴露，因为自检用了一个
`region(slot, elements) -> size: elements * 4` 的闭包，**而 session 里是手写的字面量**。

> **教训**：这类错误自检抓不到，因为自检和生产用了两套不同的写法。让它们共用同一个分配函数。

#### 坑 3：`bind_buffers` 从一个永不回收的 pool 分配

在每 block 的调用里 `bind_buffers(&[])` 拿一个新 descriptor set。pool 是固定大小、
分配后不释放，所以第 19 层开始 `ERROR_OUT_OF_POOL_MEMORY`，之后每一层都 fallback。
profile 里表现为「前 18 层正常、之后全崩」。

现有的 `bind_norm` / `bind_modulation` 就是 per-(layer, kind) 缓存的，照抄即可。

> **教训**：`bind_*` 一律 per-layer 缓存。这在仓库里已有先例，不必重新发现。

#### 坑 4：`take_gpu_session` 会把 session **取走**

每步开头需要把残差上传到 arena，我这样写：

```rust
if let Some(mut session) = take_gpu_session(scratch, total_tokens, &mut gpu_enabled) {
    session.upload_residual(...)?;      // session 在这里被 drop
}                                       // 没有放回 scratch
```

`take_gpu_session` 的最后一句是 `scratch.gpu.remove(&rows)`——它**移除**而非借用。
作用域结束 session 被丢弃，循环里每次 `take_gpu_session` 都新建一个，于是
**残差永不跨层累积**，每层都从初始 latent 重新开始。

> **教训**：`take_gpu_session` / `put_gpu_session` 这对「借出」语义的函数，调用后必须归还。
> 名字里的 `take` 已经暗示了，但很容易读成「获取一个引用」。

#### 还有一个没定位的错误

修完上面四个，PSNR 仍是 10.03 dB。**没有找到根因**，不排除还有第五个接线问题。
没有继续猜下去——按前面 TODO-025 的教训，猜测式修复在这里已经失败过一次。

已确认与改动无关的对照：CPU 路径正常（459.7 s 出图正常）。

#### 已经确认正确、可以复用的部分

这些是这次调查的确定结论，即使实现被丢弃也有价值：

1. **参考必须是 `rms_norm.comp`，不是 `rms_norm_inplace`。**
   `rms_norm_inplace` 经由 `sum_sq_f32` 归约，而它返回 **f64** 并保持 64 位累加器
   （NEON 路径）。这是比任何 f32 GPU kernel 都更准的归约，**逐位一致根本达不到**——
   现有的 `rms_norm.comp` 从第一个元素就和它不一致。
   真正该对齐的是 `rms_norm.comp`：它是未融合路径实际跑的 kernel，
   也是残差流的来源。对齐它，融合才是数值上的 no-op。
   （本条修正了最初的假设，最初以为可以对齐 CPU 参考，结果自检永远过不了。）

2. **`tanh` 必须留在 host。** gate 是逐通道的（`HIDDEN` 宽，不是每行一个），
   而 GLSL 的 `tanh` 不保证和 Rust 的 `f32::tanh` 同位模式。在 host 算好再上传即可，
   代价只有 3840 次 tanh。

3. **无 AdaLN gate 的 refiner 用全 1 填充是数学等价的**：`add_modulated_residual`
   的 `None` 分支是 `tokens[i] += residual[i]`，而 `residual * 1.0 == residual`。

4. **数据流是自洽的**：`record_ffn` 之前会把 `attention` 重新上传到 `layout.x`，
   所以 FFN 输入不是 block 开头的那个陈旧值。

5. **normed 输出对主栈是死的**：`record_ffn` 直接吃 GPU 上的 `layout.x`，
   两个 site 写进 `attention` / `qkv` 的 normed 值，主栈都不读。
   真正需要回 host 的只有残差流。这决定了融合的价值上限。

#### 性能（实现本身是有效的，只是没跑对）

`w1 readback` 从 **1109 ms → 346 ms**。denoise 净持平（41.2 → 41.3 s），
因为省下的投影回读换成了残差回读——**除非残差也能跨层常驻**，
而那要求 session 在整个 step 生命周期内不被取走（见坑 4）。

所以这条路线的真实收益上限，取决于能否把残差真正留在设备上跨 34 层累积。
这需要一个比「每层 take 一次 session」更清晰的 session 所有权模型。

### TODO-028: f32 的 aarch64 NEON kernel 名不副实，BGE 在 ARM 上拿不到 SIMD 加速

在 `codex/bge-embedding-gguf` 上审 BGE embedding 支持时发现。BGE 的 decoder 变体
（Mistral / Qwen2 / Gemma2）走 `LlamaSession`（`src/models/decoder_embedding.rs`），
f32 权重落到 `src/ops/kernel/f32/neon.rs::matmul_f32_vs_f32_neon`。

**SIMD 路径是有的**（`forward_f32_rows_dispatch` 会分派过去，`dot_f32` 也有 NEON
实现），所以不是「缺 SIMD 路径」，而是**那个 kernel 基本不是 SIMD kernel**：

```rust
let product = vmulq_f32(
    vld1q_f32(weight_ptr.add(row_off + index)),
    vld1q_f32(input_ptr.add(index)),
);
total += vget_lane_f32(vget_low_f32(product), 0);   // 立刻拆回标量
total += vget_lane_f32(vget_low_f32(product), 1);
total += vget_lane_f32(vget_high_f32(product), 0);
total += vget_lane_f32(vget_high_f32(product), 1);
```

文件头写着「4-wide FMA per inner iteration」，但实现里**没有向量累加器、没有 FMA**：
每 4 个元素做一次 SIMD 乘，紧接着 4 次 `vget_lane` 把结果抽回标量相加。
相对纯标量只省了乘法，省不掉抽取开销和标量累加。

同仓库的姊妹 kernel 是对的：

| kernel | 向量累加器 |
|---|---|
| `f32/avx2.rs`（`avx2_matmul_packed!` 宏）| `_mm256_fmadd_ps` + 双累加器 |
| `bf16/neon.rs` | `vdupq_n_f32` + `vfmaq_f32` + `vaddvq_f32` |
| **`f32/neon.rs`** | **无**，逐 lane 抽回标量 |

`f32/neon.rs` 自己写着：

> `TODO-005: once a CI target exposes aarch64, validate against the scalar reference`

而本仓库 CI（见 TODO-026）**没有任何 aarch64 runner**，所以这个 kernel 从未被验证过。

#### 为什么写成这样（不能直接当 bug 改）

代码里有明确的动机：

> Keep the scalar left-to-right accumulation contract. A vector accumulator
> changes the rounding order enough to cross BF16 boundaries in the autoregressive
> depth decoder.

改成 `vfmaq_f32` 会改变求和结合律 → 数值不再与标量一致。所以这是**有意的精度选择
换来的性能放弃**，不是疏忽。

#### 影响面

- **BGE embedding 在 aarch64 上拿不到 f32 SIMD 加速**，性能接近标量
- 同样影响任何 f32 权重走这个 kernel 的 decoder 模型（本机是 aarch64）

#### 三个方案（未实施）

1. **照 `bf16/neon.rs` 用 `vfmaq_f32` + 4 累加器** —— 最快，但求和顺序改变，
   结果**不再 bit-exact**。embedding 的 cosine 相似度通常不受影响，但这是需要
   实测确认的判断，不是可以假定的。
2. **维持现状** —— 数值契约不动，aarch64 上没有 f32 SIMD 收益。
3. **分路径**：保留严格标量路径给自回归解码，另加一条 embedding 专用的 SIMD 路径。
   **这个方案最稳**——embedding 的容差天然比逐 token 解码宽，
   且不需要牺牲 decoder 的精度契约。

若要推进，先做实测：同一批文本比对三种方案的 embedding 余弦相似度与端到端
检索一致性，再决定。**不要在没有实测的情况下放宽 decoder 的求和契约。**

### TODO-029: BGE decoder embedding 路径的 logits 投影是死代码，且穿透读取内部 scratch

审 `codex/bge-embedding-gguf` 的 `bafc23c` 时发现。该提交把
`LlamaSession::forward_one_token` 的函数体抽成 `forward_input`，加了三个入口：

```rust
forward_one_token        → forward_input(token_id, None, None, /*project_logits*/ true)
forward_embedding        → forward_input(0, Some(emb), None, false)
forward_embedding_mrope  → forward_input(0, Some(emb), Some(mrope), false)
```

重构本身是对的（参数化优于复制三份），`mrope` 的 section 之和必须等于
`n_embd_head_k / 2` 这个校验也写对了。以下两点是「可以更好」，都不影响正确性。

#### 问题 1：`project_logits=false` 对 embedding 路径是死代码

`decoder_embedding.rs` 走的是 `session.prefill(&tokens, 1)`，而 `prefill` 是：

```rust
last_logits = self.forward_chunk(input, rows, base, is_last)?;
```

**不是** `forward_one_token`。`project_logits` 的实参是 `is_last`，即**最后一个 token
仍然会投影出完整 logits**（vocab 通常 32K–150K 宽），而 `decoder_embedding` 只要
`session.scratch.x`，这批 logits 算完直接丢弃。

新加的 `if project_logits { ... }` 包裹只对 `forward_one_token` 有意义，而那条路恒传
`true`。所以 embedding 场景省不下任何东西。

**要真正省掉**：需要让 `prefill` 的最后一个 chunk 也能传 `project_logits=false`，
即 `prefill` 增加一个「只要 hidden state」的入口。当前 API 表达不了这个意图。

**收益可量化**：每次 embedding 请求省一次 `n_embd × vocab` 的投影。0.6B 级模型
vocab 约 150K、n_embd 1024，即 1.5 亿次乘加，而 embedding 的实际计算量只有几层 decoder。

#### 问题 2：`decoder_embedding` 穿透读取 `session.scratch.x`

```rust
rms_norm_grouped(
    &session.scratch.x[..n_embd],   // 跨模块读 impl 的内部字段
    &session.weights.output_norm,
    ...
);
```

两处问题：

1. **封装**：`scratch` 是 `LlamaSession` 的内部状态，从 `decoder_embedding.rs` 直接
   穿透访问，读的是「forward 恰好停在某个中间态」的隐含约定。
2. **重复计算**：`forward_input` 内部已经算过 `rms_norm_grouped`（结果在 `normed`），
   embedding 侧又从 `scratch.x` 重算一遍同样的 norm。

**更干净的做法**：session 暴露一个 `last_hidden(&self) -> &[f32]`，
或者在 `forward_input` 尾部把 pooled 后的结果落到一个专门的字段。
既不用穿透 `scratch`，也不用重算 norm。

#### 与 TODO-028 的关系

`TODO-028` 记录了 f32 在 aarch64 上拿不到 SIMD 加速。
本条的「省掉 logits 投影」和它是同一个方向的两端：一个是**别做多余的计算**，
一个是**把要做的算快**。对 BGE 这类 embedding 负载，前者的收益可能更直接——
vocab 投影通常比 decoder 层本身更贵。

### TODO-030: Visualized BGE 的 embedding 构造复制了 `populate_embeddings`，且 CLS 位置是隐含假设

`bafc23c` 给 `bert_family` 加了视觉分支（`run_embedding_tokens_with_image`）。
核心重构是**对的**：`ec62356` 把 embedding LayerNorm 从 `populate_embeddings`
搬到 `forward_bert_layers` 开头，因为视觉路径要自己构造 `hidden`，而 LayerNorm 是
两条路共用的。搬完之后 text 与 visual 共享同一份 LayerNorm，并保留了
`bert.cpp:92` 的溯源。同一个提交里 `n_ctx_train` 的长度检查从 `token_ids.len()`
改成含图像行的 `n_tokens`，也是必要修正 —— 原来只查文本 token 数，加了图像会漏检。

以下两点是维护隐患，不影响当前正确性。

#### 问题 1：embedding 构造有两份实现

`populate_embeddings`（text）与 visual 分支的内联代码，各自实现了一遍
「token embedding + segment + position」：

```rust
// visual 分支内联手写
embedding_lookup(weights.token_embd, *id, h, weights.token_embd_ggml_type, row);
for j in 0..h {
    row[j] += segment.as_ref().map(|s| s[j]).unwrap_or(0.0);
    row[j] += position[t * h + j];
}
```

风险不在于现在会算错，而在于**改了一处另一处不会跟着变，而且不会有编译错误**。
不同 BGE 变体的 position/segment 规则未必相同（图像 token 是否加 segment、
segment 是否只加给文本），这类差异正是最容易只改一边的。

**建议**：抽出一个共用 helper，visual 只额外传「哪些行是图像」：

```rust
fn populate_embeddings_with_overrides(
    cfg: &BertConfig,
    weights: &BertWeights<'_>,
    token_ids: &[u32],
    image: Option<&[f32]>,
    image_rows: usize,
    hidden: &mut [f32],
) -> Result<(), String>
```

#### 问题 2：token 位置映射依赖未被断言的 CLS 假设

```rust
for (i, id) in token_ids.iter().enumerate() {
    let t = if i == 0 { 0 } else { i + image_rows };
    ...
}
```

`i == 0` 的特殊处理隐含假设 **token 0 永远是 CLS**，但代码里没有任何地方检查。
布局本身自洽（图像占 `1..=image_rows`，文本从 `image_rows + 1` 起），可一旦某个
tokenizer 不加 CLS、或加了别的 prefix token，映射就会静默错位 —— 不报错，
embedding 全错，且这类错误通常要到检索结果变差时才被发现。

**建议**：显式校验 token 数与图像行数的对应关系，并把「token 0 是 CLS」写成
一条可读的断言或注释约定，而不是只靠 `if i == 0` 这个分支表达。

#### 一个正面观察

`position` 表的校验写得很严，值得保留：

```rust
if info.dims.len() != 2 || info.dims[0] != h as u64 || info.dims[1] < n_tokens as u64 {
    return Err("invalid Visualized BGE position table".into());
}
```

dims 数量、宽度、以及**行数是否覆盖 `n_tokens`** 三者都查了 —— 这正是
「不查就会静默越界」的地方。

### TODO-031: Qwen3 resident Vulkan decode 在 aarch64 上打不过 CPU，且默认根本不启用

排查「Qwen3 为什么没有 GPU 加速」时挖出三层问题，从表面到根因。

#### 1. 默认配置下 `--gpu` 是静默空操作（已修，a66705f）

Qwen3 唯一的 GPU 路径是常驻 Vulkan 解码器 `Qwen3VulkanSession`
（`src/vulkan/qwen3.rs`），它**要求 F16 KV**。而 CLI 默认是 F32
（`src/app/text/runtime.rs:50` 的 `defaults::KV_FORMAT = KvFormat::F32`），
于是 `src/models/qwen3/trunk/session.rs` 里 `compute_policy.context()` 那段
直接落到 `None` 分支，**不报错、不提示**，全程 CPU 跑完，日志里还照样打印
`[GPU] Vulkan device: NVIDIA <GPU>`。

```rust
} else if compute_policy == ComputePolicy::Vulkan {
    return Err("Qwen3 Vulkan currently requires F16 KV".into());
} else {
    None          // Auto + F32：请求被静默丢弃
}
```

`a66705f` 改成明确提示并给出正确 flag。**在此之前，本文件之前所有关于
「Qwen3 GPU 加速」的结论都是无效的** —— 那些数字测的是 CPU。

#### 2. 即使显式 `--kv-cache f16`，GPU 也赢不了

在 aarch64 Vulkan 主机（int8 dot product: supported）上实测，
同一台机器、同一 prompt（220 词）、`--threads 8`、等待 load<3、
多次重复取中位数：

| 模型 | 大小 | 生成 ratio (GPU/CPU) | 端到端 ratio | wall CPU | wall GPU |
|---|---|---|---|---|---|
| Qwen3-0.6B-Q8_0 | 0.6G | 0.46x | 0.34x | 2.8s | 8.7s |
| Qwen3-0.6B-Q4_K_M | 0.4G | 0.11x | 0.11x | 3.9s | 33.8s |
| Qwen3-0.6B-BF16 | 1.1G | 0.38x | 0.43x | 5.9s | 15.7s |
| Qwen3-4B-Q4_K_M | 2.3G | 1.08x | 0.14x | 15.7s | 113.4s |
| Qwen3-4B-Q8_0 | 4.0G | **1.08x** | 0.56x | 16.8s | 31.2s |
| Qwen3-8B-Q4_K_M | 4.7G | 0.19x | 0.11x | 35.1s | 398.5s |
| Qwen3-8B-Q8_0 | 8.1G | **1.02x** | 0.54x | 49.0s | 100.1s |
| Qwen3-14B-Q4_K_M | 8.4G | 0.97x | 0.56x | 72.7s | 138.0s |

两个独立的结论：

- **生成（steady state）在 4B 以上才勉强追平**，逐次 ratio 紧密聚在 1.0 附近
  （4B Q8_0: 1.28 / 1.00 / 0.98；8B Q8_0: 1.00 / 1.02 / 0.94），
  **不是加速，是持平**。
- **端到端在所有尺寸上都是输的**（0.11x ~ 0.59x）。因为常驻解码器要先
  付一次性启动成本：~0.4s 的 pipeline JIT + 把整个权重上传进显存。

启动成本拆解（Qwen3-0.6B-Q8_0，`RMI_SETUP_TRACE`）：

```
Qwen3Ops::new (pipeline/descriptor)  211.0 ms
权重上传+bind  633.5 MB               169.0 ms
core_contract 校验                     31.5 ms   ← 已是 OnceLock memoize
q8_0_activation_contract 校验           6.4 ms   ← 同上
```

契约校验那两项**已经**是 `OnceLock::get_or_init`，只跑一次，没有优化空间；
pipeline JIT 和权重上传是一次性 CLI 进程无法摊销的成本。

#### 3. 根因：dense GEMM 在 M 很小时被饿死

用 `RUST_GPU_SUBMIT_TRACE` 测 Qwen3-8B-Q8_0 的 prefill：
一次 prefill 只产生 ~36 次 submission（**分块是合理的**，不是提交过碎），
但每次 `wait` 平均 **188 ms**，而驱动调用 `submit` 只有 35 µs。
即整条链路的开销不在提交，而在 GPU 执行本身。

折算下来整趟 forward 对 8GB 权重只跑到约 **1.2 GB/s 有效带宽**；
而同一块卡上 LM head 那一类（`rows=1`, `n_out` 很大）能跑到 22 GB/s。
差别在于 **M（行数）**：prefill M≈60、decode M=1，对 GPU 来说都太小，
填不满 SMs，既算不出强度也吃不到带宽。

对比同机 CPU：8B prefill 11.3 t/s vs GPU 5.7 t/s —— **CPU 快一倍**。
这台机器的 CPU（含 AVX2/FMA 路径）在这个工作负载上就是比这条 Vulkan
dense 路径强。

#### 为什么先不写「按模型大小决定是否上 GPU」的阈值

本来想用上面的表定一个尺寸阈值，但数据不支持：**4B 以上生成只是持平，
没有出现真正的交叉点**，而且持平的区间里 e2e 依然全输。也就是说
「小模型不上 GPU」这个结论虽然成立，但真正的问题是
**常驻 dense 路径本身在这个硬件上就没赢过**，加阈值只能掩盖症状。

要真正修好，得动 dense GEMM kernel 本身（M 小的时候的 tiling / split-K），
那是重写 shader 的量级，不适合在没有回归基线的情况下顺手做。

#### 附带记录

- **Q4_K_M 在 GPU 上尤其糟**（8B: 0.19x，wall 398s）。
  `GpuMatmulMode::Dot`（GEMV/split-K 路径）只对 F16/BF16 存在，
  Q4_K 只能走 `Prepared` -> `Q4_K_MATMUL`，没有 rows=1 专用 kernel。
  这和 `7750eeb` 里 LFM2.5 MoE 的结论是同一件事的两个入口。
- **8B-Q8_0 的 GPU 运行峰值 RSS 约 18.7 GB**，超过 15 GiB 预算；
  一次性 CLI 同时持有 mmap 权重和显存副本是原因之一。

### TODO-032: Q4_K/Q5_K/Q6_K 在 GPU 上没有 DP4A kernel，8B Q4_K_M 比 CPU 慢 17.6 倍

接着 TODO-031 往下挖，发现 Q4_K 家族在 dense Vulkan 路径上缺少向量化 kernel，
这是 Q4_K_M 特别慢的**结构性**原因。

在 `src/vulkan/ops.rs` 的 operator 表里：

- **Q8_0** 有两个 kernel：
  - `Q8_MATMUL_GROUPED`，其 index 会被 `Q8_MATMUL_GROUPED_DP4A` 顶替
    （`force_dp4a && context.supports_integer_dot_product()`，约 1423 行）；
  - 外加 `Q8_MATMUL_TILED_DP4A`（`rows > 1` 时 `record_weight_matmul_rows`
    自动切到 `record_weight_matmul_tiled_rows`，约 1960 行）。
  两者都吃设备 int8 dot-product，8B Q8_0 生成能持平 CPU。

- **Q4_K / Q5_K / Q6_K** 只有**一个**标量 kernel（`q4_k_matmul.comp` 等，
  表项见约 392 行），既没有 dp4a 变体，也没有 tiled 变体。
  8B Q4_K_M 实测 prefill 0.4 t/s、生成 1.3 t/s，wall 45.8s，
  同机 CPU 2.6s —— **慢 17.6 倍**，是所有测过的组合里最差的。

`q4_k_matmul.comp` 主循环（每 block）：

```glsl
int value = int((group & 1u) == 0u ? packed & 15u : packed >> 4u);  // 解 4-bit nibble
int activation = q8(block * 256u + local);
partial[(local & 31u) >> 2u] += int(scale(slot, offset, group)) * activation * value;
```

单块累加值量级 = `scale(≤63) * activation(≤127) * value(≤15)`，
再在 8 个 lane 上求和，落到 int32。这种**块缩放的 int32 乘加不能直接套
Q8_0 的 dp4a 路径**（Q8_0 的量化把乘积压在 int8 内）。所以 Q4_K 想吃
DP4A，得写一个自己的「块缩放 + int32 提升」的 int8 dot kernel，或者
把激活缩放到能进 int8 的范围 —— 这是一个完整的 shader 项目，不适合随手做。

结论与 TODO-031 合并看：**GPU dense 路径目前只有 Q8_0 勉强可用（持平 CPU），
Q4_K 家族完全不能用（慢 17.6 倍）。** 在补出 Q4_K 的向量化 kernel 之前，
Q4_K 模型应该明确走 CPU，而不是靠一条经验阈值去猜。

### TODO-033: Q4_K split-K GEMV 试过，行度不够；真正的瓶颈是单字节权重访问效率

接着 TODO-032，真的去写了一个 rows==1 的 split-K GEMV（`q4_k_matmul.comp`
里新增分支：rows==1 时不让 64 个 lane 各领一个输出行，而是让 64 个 lane
合作同一个输出行，按 32 元素一个 scale/min 小组切 K，最后树形归约）。
256 元素的 Q4_K block 正好切成 8 个 32 元素组共享一个 scale 和一个 minimum，
所以 lane 之间不需要交换 scale，每个 lane 内部就能把该组的 dot 算完。

结果：

- **数值**：对真实权重（Qwen3-8B-Q4_K_M 的 blk.0.ffn_gate），
  GEMV 输出与已提交（对 CPU 位精确）的 batched 路径 **逐字节一致**。
- **性能**：rows=1 只从 2.33ms 降到 2.11ms（~10%），LFM2.5 MoE 端到端
  1.1 t/s → 1.2 t/s，**基本没动**。因此这个分支没有保留（回退了）。

**为什么并行度不是瓶颈。** 8B prefill 每次提交 GPU 等 188ms（vs 驱动
35µs），整趟 forward 对 8GB 权重只跑出 ~1.2 GB/s；而同卡 LM head
（rows=1、n_out 很大）能到 22 GB/s。差异在于**单次权重字节的访问效率**，
不是 lane 数：

```
Q4_K rows=1:  2.06 MB 权重，实测 2.11 ms  →  0.98 GB/s
理论下限:     若能到 25 GB/s，只需 0.08 ms   →  差 25 倍
```

根源在 shader 的 `weight_byte()` / `q8()`：一次取一个字节
（`w.data[index>>2] >> ((index&3)*8)` 再掩码），nibble 再逐个拆
（`packed & 15` / `packed >> 4`）。加多少 lane 都还是在做字节级提取。

**真正要做的**：把权重读取向量化——用 `uint4`/`int4` 一次搬 16 字节，
在寄存器里用位运算拆 nibble 和 scale，让一次内存事务服务 16 个元素。
这是重写 Q4_K 读取路径的量级，和 TODO-032 是同一个大工程，建议合并做。
在那之前：
- batched（rows>1）的 Q4_K 已经被 `7747f07` 的 shared-memory staging +
  scale 提取外提提速 2.3~2.6 倍（位精确）；
- rows=1 的 Q4_K（MoE decode、单 token）仍然只有 0.98 GB/s，
  继续走 `7750eeb` 的 CPU 门控。

### TODO-034: Z-Image GPU 路径罕见地不可复现（同 seed 约 1/20 次输出不同）

在 aarch64 上交替跑同 seed 的 Z-Image 256×256 八步，**大约每 20 次有一次输出不同**。
幅度很小：mean|Δ| 1.2/255、PSNR 41.5 dB（不是坏图，是舍入级差异），
其余各次逐字节一致。main 与 unified 分支都出现过，unified 修复前后都出现，
所以**不是本次 push constant 特化引入的**。

`main` 的历史记录里已经有同类现象（`docs/usage/z-image.md`「host 写 → device 读的
barrier」一节）：arena 是 CPU 直接写、shader 直接读的映射内存，缺失
HOST→COMPUTE barrier 时能否看见写入只取决于速度；旧 kernel 每次 dispatch 约 700 ms，
写入总来得及被「看见」，tiled kernel 约 64 ms 就可能错过。该记录同时说 512×512 才会
复现、256×256 逐位一致——**现在 256 也会出现，说明 tiled 化之后窗口更窄了**。

需要做的事：

1. 在 `VulkanContext::host_write_barrier` 之外，找出**哪一类** host 写入没有对应的
   barrier。`zimage_projection_kernel_costs`（`src/vulkan/ops.rs`，`#[ignore]`）
   可以先复现到单 kernel：同一形状连续录制多次，比较读回的激活是否逐字节一致。
2. 目标是让 256 与 512 都逐字节可复现，再谈「GPU 与 CPU 逐位一致」。
   在此之前，任何「GPU 输出逐字节一致」的测试结论都要注明运行次数。

### TODO-035: 补偿 exp 的价值只在 decoder 侧，DiT 侧已确认不值得

试过给 `softmax.comp` 加 `exact_exp` push 位、让 Z-Image DiT 走单精度 Horner
（`exp_cheap`），结论是**回退**：

- 性能几乎没赚：attention 相位只从 17.6 s 到 16.7–17.1 s（512，4 步）。同一个
  shader 里编进两套实现后，热循环里那个 select 把省下的算术吃掉了。
- 数值代价可测：DiT GPU 输出与同分支 CPU 参考的 PSNR 从 37.66 dB 降到 31.62 dB
  （mean|Δ| 1.76 → 2.47，256×256 八步 seed 42）。

所以 DiT 保留补偿版。剩下的开放问题是 decoder 侧：`exp_accurate` 的注释说
「1 ULP 的 exp 误差可能翻转 F16 概率平局」，而 `attention_values` 在两种模式下都会把
概率 `round_f16_rte` 之后才累加——补偿把 exp 做到 F32 最后一位，紧接着就被 F16
舍入吃掉了。要回答的是：**decoder 路径真正需要的是 exp 的精度，还是「舍入到 F16
之后不再变」这个性质**。如果只需要后者，把 exp 的误差压到远小于 F16 间隔（例如
1e-4 相对）就够了，补偿项可以砍掉一半。

复现数据与完整推导见 `docs/develop/ZIMAGE_OPTIMIZATION.md`。

### TODO-036: Qwen3-4B Q8_0 过不了 model contract，而 0.6B 与 8B 能过

`test-unified3`（unified-compute）上用 `--gpu --kv-cache f16` 跑 Qwen3 的三个
Q8_0 尺寸，model contract（`src/models/qwen3/trunk/session.rs`，
abs 2e-3 + rel 2e-3）的结果**按尺寸分裂**：

| 模型 | contract | 实测（Generation t/s，同会话） |
|---|---|---|
| Qwen3-0.6B Q8_0 | 通过 | 25.6 / 25.7 / 26.6 |
| Qwen3-4B Q8_0 | **失败**：`model logit mismatch at 0: gpu=9.235686 cpu=9.315228 rel=0.008538938` | 走 CPU，10.6 / 11.6 t/s |
| Qwen3-8B Q8_0 | 通过 | 4.7 / 5.0 / 5.1 / 5.3 / 5.5 |

4B 的相对偏差 8.5e-3 是容差的 4.3 倍，于是整条 GPU 路径被拒并回退 CPU。
**同一个模型家族、同一种量化，只换尺寸就翻脸**，说明这不是「Q8 路径整体不精确」，
而是某个尺寸触发了特定形状的偏差（4B 的 hidden 2560 / 36 层与 8B 的 4096 / 36 层
在 n_in 上不整除某些 chunk 边界，是第一个该查的方向）。

补充两点，避免重复劳动：

- 这个失败**与 push constant 特化无关**：修复前后的 contract 数值逐位相同
  （gpu=9.235686 cpu=9.315228），greedy 输出也逐字相同。
- `main` 没有这道 contract 门，4B 直接跑 GPU，但只有 1.4 t/s generation，
  比 CPU 回退的 10.6 t/s 慢 7 倍——**门是合理的**，问题在于 4B 的 GPU 数值
  偏差太大而不是 GPU 太慢。

同时记录：unified 分支的 decoder 路径比 `main` 快 6 倍以上
（8B Q8_0 generation 5.1 t/s vs 0.8 t/s），这是该分支的目标，回归要盯住这条线。

### TODO-037: `--threads` 等于核数时 CPU 吞吐塌 29 倍

Qwen3-0.6B Q8_0，20 核 aarch64，greedy，64 token：

| `--threads` | prefill t/s | generation t/s |
|---|---|---|
| 12 | 280–335 | 82.8–88.4 |
| 16 | 303–347 | 67.5–97.2 |
| 17 | 281 | 52.2 |
| 18 | 295 | 69.3 |
| 19 | 312 | 69.0 |
| **20** | **46–48** | **2.4** |

悬崖正好落在 `threads == nproc`，可复现 3/3，且不是偶发：20 线程的 prefill 也只有
1/6。默认线程数是 8，所以平时碰不到，只有显式 `--threads $(nproc)` 才会踩到。

机制（`src/core/thread_pool.rs`）：`ComputePool::new(n)` 起 `n - 1` 个 worker，
调用线程也参与，总参与者 = n。worker 在空闲时是 `spin_loop()` 忙等（CPU 模式，
GPU 模式才 yield），而 `compute()` 的收尾是「自旋等 `n_complete == n_threads`」。
在 20 核机器上再叠一个别的租户进程占着核，参与者一多，barrier 就开始等调度器，
一个 token 被放大到 0.4 s。

**已修**（`src/core/thread_pool.rs`）：等待循环与空闲 worker 都不再硬自旋。

1. 新增 `wait_while`：先自旋 `WAIT_SPINS` 次（覆盖一次 hand-off 的微秒级窗口），
   之后改成 `yield_now()`。原来的 `spin_loop()` 无限自旋会让每个参与者一直处于
   runnable 状态。
2. 空闲 worker 同样改成「有界自旋 + `yield_now()`」。这条更关键：
   `compute_with_chunks` 每次调用都新起 `n_threads - 1` 个线程，20 线程时进程里
   同时有 42 个线程挤在 20 核上，而常驻 worker 的热自旋正好在抢这些线程要用的核。
3. `cargo test --lib core::thread_pool` 的两个用例覆盖了 Drop 与 chunk 语义
   （`single_thread_pool_drops_without_spinning`、`chunks_begin_at_zero_and_execute_exactly_once`）。

修完的效果（Qwen3-0.6B Q8_0，`--threads 20`，同会话交替）：

| | generation t/s |
|---|---|
| 修复前 | 2.5 / 2.4 / 2.9 |
| 修复后 | 65.1 / 53.8 / 41.4 |

**20×**。顺带：`--threads 8` 的 CPU 生成从 41–56 t/s 到 83 t/s，
Z-Image 的 **CPU** 路径 denoise 从 139.7 s 降到 126.4 s（−9.5%），GPU 路径不变
（block profile 20.0 s vs 19.5 s，输出逐字节相同）。

没做的：把 worker 数夹到 `min(requested, nproc - 1)`。修完等待逻辑后
`--threads 20` 已经不再塌，这条就没有必要了——但默认线程数仍是 8，
因为 12–20 线程在 0.6B 上没有稳定收益。

顺带记一条同源结论：**Qwen3 的 GPU 常驻解码在这台机器上全面输给 CPU**
（0.6B：CPU 67–97 t/s vs GPU 26 t/s；8B：CPU 12.6–13.5 t/s vs GPU 5.1 t/s），
原因与 TODO-031 一致，只是这次把尺寸补全了。GPU 真正赢的是高算术强度的大矩阵
（Z-Image DiT 权重 1.3 GB/层、rows 数百），而 decode 是 rows=1 的 GEMV——正是
`docs/usage/z-image.md` 里点名会误导的那类形状。

### TODO-038: CPU/GPU 自动分发应该按实测选后端，而不是按模型大小

在 aarch64 上把 Qwen3 的 Q8_0 从 0.6B 扫到 8B（同 prompt、greedy、16 线程，
线程池修复后的构建）：

| 模型 | 每 token 权重 | CPU gen t/s | GPU gen t/s | CPU prefill t/s | GPU prefill t/s |
|---|---|---|---|---|---|
| 0.6B | 0.54 GB | **121.2** | 52.8 | **346.9** | 25.9 |
| 4B | 3.74 GB | **23.1** | contract 失败→CPU | 1.3 | — |
| 8B | 7.57 GB | **11.1** | 9.2 | **14.0** | 4.3 |

**没有出现 GPU 超越的点**，但差距随尺寸收窄：0.6B 差 2.3×，8B 差 1.2×。

把两边换算成有效带宽（每 token 把整个权重集读一遍）：

| 模型 | CPU GB/s | GPU GB/s |
|---|---|---|
| 0.6B | 65 | 28 |
| 8B | 84 | 70 |

于是结论是：**GPU 解码是带宽受限的，而它只跑到 70 GB/s；CPU 在同一台机器上跑到 84 GB/s。**
`qwen3_decode_gemv_bandwidth`（`src/vulkan/ops.rs`，`#[ignore]`）在真实解码形状上
量到 lm_head 一次 661 MB 的 GEMV 最快 8.9 ms（74 GB/s），其余形状 10–52 GB/s，
小矩阵被每次 dispatch 的固定开销吃掉。

**但这个 GPU 数字是带偏的**：测量期间另一个租户占着 85% 的 GPU（root 的 python，
96 GB 显存），nvidia-smi 采样 79–92%。GPU 这边还没拿到空闲机器，所以
「GPU 在大模型上能不能反超」在这台机器上无法判定。

因此自动分发的策略应当是**实测驱动**，不是按尺寸猜：

1. **先量设备忙不忙**。本机 GPU 常驻 85% 利用率时，GPU 路径在任何尺寸都不该被选中。
   可以用首 token 的实测延迟对比 CPU 基线，落后就回退——现在只有「logit 逐位一致」
   的 contract 门，没有「谁更快」的门。
2. **把 setup 成本算进去**。host→device 上传实测只有 5–9 GB/s（PCIe 路径在这台
   统一内存机器上不快），8B 的 7.5 GB 权重上传就要 1 s 起步；一次性的 CLI 短跑
   永远摊不平，serving 才能。策略需要知道预期 token 数。
3. **contract 门先修**。4B 现在直接被拒（TODO-036），在数值对齐修好之前，
   任何尺寸相关的策略表都是空谈。

顺带记一条实测的坑：这台机器上任何「GPU 更快/更慢」的结论都必须附带
**当次测量的 GPU 利用率**，否则数据没有可比性——TODO-036/037/038 里的数字都受此影响。

### TODO-039: PyTorch 参照证明 GPU 路径不是「实现得不好」，是设备被占着

用原始 safetensors 权重跑 PyTorch（`tools/oracle/qwen3/pytorch_qwen3.py`，
bf16，greedy，64 token），和本仓库的两条路径同台对照：

| 模型 | PyTorch CUDA bf16 | 我们的 Vulkan GPU (Q8_0) | 我们的 CPU (Q8_0) | PyTorch CPU fp32 |
|---|---|---|---|---|
| Qwen3-0.6B | 24.7 t/s | **52.8 t/s** | 121.2 t/s | 2.4 t/s |
| Qwen3-8B | 4.2 t/s（15.6 GB/token） | **9.2 t/s**（7.6 GB/token） | 11.1 t/s | — |

**我们的 Vulkan 路径比同卡 PyTorch 快 2.1–2.2×**（Q8_0 把每 token 的字节数减半，
kernel 也够快），所以「GPU 输给 CPU」**不是实现问题**。

真正的原因是带宽比：GPU 有效读取 70 GB/s，CPU 84 GB/s，而 GPU 这 70 GB/s 是在
**另一个租户占着 85% 设备**的情况下拿到的（nvidia-smi 采样 79–92%）。aarch64 的
DRAM 峰值在 273 GB/s 量级，设备空闲时 GPU 路径在 8B 上应该能到 20 t/s 以上，
也就是**会赢过 CPU 的 11.1 t/s**。

所以自动分发的正确形状是「按实测选」，而不是「按大小猜」：先量设备占用与首 token
延迟，落后 CPU 就回退。0.6B 上 GPU 无论如何都输（小模型权重装不进 cache，
CPU 直接吃 LPDDR），8B 以上在设备空闲时才可能赢——这两个边界都需要在
**独占设备**上重新测一遍。

顺带把 PyTorch 的输出也记下来：Q8_0 的 greedy 文本与 bf16 参考在十几个 token 后
就分叉（CPU 路径与 GPU 路径之间也分叉），这是量化 + 非逐位一致的正常表现，
不是 bug 的证据；要判对错得用同权重的 llama.cpp 参照（TODO-036 的 4B contract
失败同理）。

### TODO-040: Astronex GPU 路径在同 seed 下不可复现（aarch64，2026-10-21）

同一个二进制、同一个 prompt、同一个 `--seed 42`，两次 GPU 跑出来的
256x144 视频逐帧 PSNR 只有 18.4–19.2 dB；把采样后的 latent 导出来
（`RMI_ASTRONEX_LATENT_OUT`）对比，61,440 个元素里有 61,439 个不同，
最大绝对差 1.78（latent 本身最大幅值 8.66）。差异在 VAE 之前就出现，
所以 VAE 和文本编码器都不是源头。

已经排除的：

- online attention kernel 与 BF16 投影 kernel **单独**反复调用都逐位一致
  （`examples/astronex_gpu_determinism.rs`）；
- 文本编码器里第一个出现差异的算子**永远是 GPU 投影**：输入哈希相同、
  权重相同，输出哈希不同（`examples/astronex_text_determinism.rs`）；
- 关掉 GPU matmul（`RUST_GPU_MIN_OUTPUTS`）或把 `ComputePool` 降到单线程，
  文本编码器就完全可复现 → CPU 路径和线程池自身的归约都不是源头；
- 差异是**间歇性**的：有些次编码逐位相同，而且首次出错的算子在几次运行之间会移动。

一个已知但未修的规范问题：`src/vulkan.rs` 的 `host_write_barrier` 是在
`begin_commands` 里记录的，位置在它本该覆盖的 host 写**之前**（权重上传与
输入写都在其后）。把屏障挪到 dispatch 之前能降低发生率但不能根除，
所以 host→device 可见性不是唯一原因。

下一步：在**独占设备**上重测（这台 aarch64 机器有另一个租户占着 65 GB 和 79–92% 利用率），
确认是驱动/内存压力导致的偶发，还是我们自己的同步缺口。

### TODO-041: Astronex 视频输出依赖系统 `ffmpeg`

`src/models/diffusion/dreamx/media.rs` 直接 spawn `ffmpeg`，没有它
`--astronex` 跑完采样和 VAE 解码后在写视频时失败（`Start ffmpeg for DreamX
video: No such file or directory`）。这台机器没有系统 ffmpeg，验证时用的是
`imageio-ffmpeg` 装进仓库 venv 的静态构建（7.0.2，含 libx264）。
要么在 USAGE 里写明这个运行期依赖，要么在 `--out` 不支持时给出更早、更明确的报错。

### TODO-042: Astronex 权重契约的 pinned commit 不公开

转换器的形状契约来自 `references/astronex-world-contract`，钉在
`dec4ec502f7113a1ab07b78ae40a7ac1b88b2eeb`；这个 commit 不在
`github.com/Astronex-Robotics/Astronex-World` 的任何 ref 里，也不在
gouzil/rust-model-inference 的远端分支里。本地只能用下载到的 safetensors
自己生成 header，等于 denoiser/VAE 的形状 pin 变成自证（张量数
898/242/196、dtype、`model.*` 前缀和 UMT5 的硬编码形状仍然生效）。
要么把契约 header 随仓库分发，要么让转换器在缺契约时明确报「未钉住」而不是静默通过。

顺带记一条：官方图的 trace 在本机跑到第 1,438 个 checkpoint（整个文本编码器 +
denoiser 大部分）后停在 `NotImplementedError: Uncovered floating factory:
linspace.default` —— scalar oracle 的算子白名单没覆盖 `linspace`（solver 的
timestep 网格），不是权重或图的问题。要跑完整官方图 trace，得先给 oracle 补
`linspace` 的白名单条目。

### TODO-043: 对拍生成文本必须显式 `--temp 0`（默认是 0.6 采样）

排查 BF16/F32 Prepared 新路径的干扰面时（2026-10-22），前两批实验全部作废，
原因就是这个：CLI 的默认温度是 **0.6**（`src/app/cli/validate.rs:321`），
所以不带 `--temp 0` 的「同 seed 两次对比」其实是在比较两次独立采样。

它产生的假象很有说服力：同一二进制、同一 prompt，CPU 连续两次结果不同，
其中一次把澳大利亚首都答成 Melbourne（错误答案），一次答成 Canberra；
统计「与 CPU 参考的最长公共前缀」也随之变成噪声。补上 `--temp 0` 之后
CPU 侧逐位可复现（同 1 和 8 线程都一致），所有差异才变得可解释。

顺带记第二个坑：选测试对象时必须先静态确认它**真的走到被改的代码路径**。
Qwen3-0.6B-BF16 看起来是最干净的受害者（197 个纯 BF16 二维权重），实际上它
用 `with_bf16_input` 构造 kernel，mode 落在 `RoundedBf16`，根本不进新分支，
于是「输出一致」只证明了没变化。`dump_dispatch_log` 只在测试里调用，CLI 读不到，
只能从 `rounds_bf16_input()` 的来源反推哪些模型会命中。

排查结论：主力文本模型（Qwen3 / Qwen3.5 / Edge0 / ERNIE）全部不命中，
Z-Image 代码零改动且 3/3 逐位复现基线 md5，含 110 个 F32 二维权重的
occamy-1.0 在 `--temp 0` 下 GPU 与 CPU 逐位一致。尚未实测的是少数多模态
projector（LongCat）与几个 MoE gate。ZDTaichu5 mmproj 已在本分支按 CPU 数值
对齐（见上面的 ZDT 条目），但 GPU 路径仍在其覆盖范围内。

### TODO-044: Mage-Flow 的 GPU 路径 run-to-run 不确定，投影的 dispatch 结果偶发不完整

排查 BF16/F32 Prepared 新路径（TODO-043）时顺带发现的独立问题，2026-10-23。

**现象**：同一二进制、同一 seed、同一 prompt，Mage-Flow 在 GPU 上连续跑的结果互相
不同：512×512 对 CPU 参考的 PSNR 在 13.3 / 16.1 / 18.6 / 29.5 dB 之间跳，最差的
一次只剩 11 种颜色；256×256 连续六次跑出六个不同的文件。CPU 侧完全可复现（两次
CPU 跑 md5 相同）。

**定位工具**：`RMI_GPU_CPU_CHECK=1` 让每个 GPU 投影与 CPU 内核对拍（同一个权重、
同一份输入）。两种求和顺序不同，正常投影的相对差在 1e-6 附近；设备交回 arena
的旧内容时是 1.0；dispatch 只跑了一部分时落在两者之间。

**故障模式**：某个投影的 GPU 结果与 CPU 参照毫无关系（相对差 1.0，输出全零或
一半是零），**而在同一个 runtime 上立刻重算一次就对了**。这排除了权重、输入与
kernel 逻辑：三方仲裁（GPU / CPU NEON / f64 参照）显示三者都准，真实 BF16 权重
下 GPU 的相对误差 3.9e-7、CPU 的 1.2e-7。异常与形状无关，出现在 text 流
（rows=20）上，因为它的 dispatch 最短、最快完成。

**已排除**：kernel 逻辑（f64 参照下精确）；单算子非确定（真实权重 400 轮逐位
相同）；形状交替与多形状交错（144 个真实形状 × 4 轮逐位相同）；命令提交顺序
（`TokenCommands` 持 `context.mutex`，`submit_commands` 真的 `wait_for_fences`）；
CPU 并发（`--threads 1` 同样复现）；host 写可见性（屏障、`flush_mapped_memory_ranges`、
`vkInvalidateMappedMemoryRanges`、把 host 写提前到录制之前，四种手段都试过）；
pipeline JIT（共享 pipeline 后仍然复现）。

**已实施的修复**（把异常率从 18/2700 降到 1~2/2700）：

- **投影复用**（`src/models/diffusion/mage_flow/dit.rs`）：Denoiser 之前每个
  linear 都新建 `Linear`，于是每个投影都新建一次 runtime、一次 arena、一整套
  operator pipeline。现在投影建一次就复用。
- **operator pipeline 按设备共享**（`src/vulkan.rs`）：每个 arena 各编译一遍
  全部 shader，denoiser 每步要付一百多次编译的代价。
- **host 写提前到录制之前**（`src/vulkan/ops.rs`）：`begin_commands` 里的
  HOST→COMPUTE 屏障本该给 host 写排序，但权重上传与激活拷贝都落在它之后。
- **`flush_mapped` / `invalidate_mapped`**（`src/vulkan.rs`）：映射内存的写与读
  各自显式刷新/失效，不再只依赖 coherent 语义。

### 第二轮排查（2026-10-26）

**校验层**：机器上有 Khronos 校验层
（`/usr/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json`）。在
`src/vulkan.rs` 里加了 `RMI_VULKAN_VALIDATION=1`（启用层 + `VK_EXT_debug_utils`
messenger + `VK_EXT_validation_features` 打开同步校验）后：

- 校验层能加载、能报消息（证明 messenger 与特性都生效）；
- 我们的 Vulkan 用法**没有报任何违规**；
- 但层开启后 NVIDIA 驱动在 `vkCreateComputePipelines` 里崩了：SIGTRAP 落在
  `libnvidia-gpucomp.so` 的着色器编译器里（GPU-AV 关掉也一样崩）。这是驱动侧的
  编译器缺陷，不是我们的用法问题，所以这条路查不下去。

**窄投影探针的假阳性**：`bf16_prepared_activation_round_is_visible` 一度报
4000 轮里失败 16 次。查下来是**探针自己的归一化 bug**——它用第 0 轮的量级当分母，
而输入每轮都在变大（`base + round * step`），于是健康的 1e-6 被放大成 1e-2。
改成每轮按当轮量级归一、并用精确参考代替外推参考后，**4000 轮 0 失败**
（rows=20 与 rows=256 各测）。结论：单投影的窄路径其实是健康的，之前那批
"失败" 全是探针假阳性。

**真正的漏网之鱼**：分歧不在 DiT，也不在单个投影，而在**VAE 解码器**。
逐阶段校验和（`RMI_MAGE_VAE_CHECKSUM=1`）显示 DiT 的投影与注意力逐位相同，
第一个分歧出现在 `pipeline.dec_net.res_blocks.*.mlp.0` 与
`pipeline.y_embedder_x`——这些是 **VAE 里没走校验的算子**（VAE 的 `linear` 用的是
`forward` 而不是 `forward_verified`，VAE 的卷积完全没有校验）。

**损坏形态**（`RMI_GPU_VERIFY_TRACE=1` 打出的形态表）：

| 形状 | 不同的元素 | 受影响的行 | 最差行的错列数 |
|---|---|---|---|
| rows=6, n_out=3072 | 18432/18432 (100%) | 6/6 | 3072 |
| rows=256, n_out=8192 | 1866527/2097152 (89%) | 256/256 | 7478/8192 |
| rows=65536, n_out=32 | 1635183/2097152 (78%) | 65536/65536 | 32/32 |

即：损坏是**整片**的（一个 dispatch 的结果整体不对），不是零星几个元素。

**已实施的修复**（把 16 次相同结果从 3/10 提到 19/20、14/14、10/10）：

- **VAE 的 `linear` 改走 `forward_verified`**（`src/models/diffusion/mage_flow/vae.rs`）。
- **卷积也做校验**（`src/models/diffusion/dreamx/kernels.rs`）：新增
  `conv3d_matches_cpu`，抽样 512 个输出元素与 CPU 卷积对拍；不对就重跑
  （`GPU_RETRY_LIMIT = 3` 次），仍旧不对就整块交给 CPU 卷积。同时把 CPU 卷积
  主体抽成 `conv3d_cpu` 供两条路径共用。
- **注意力校验加密**：每个 token 的每个 head 查 8 个元素（原来只查 1 个），
  设备只坏掉 head 的一部分时原来会漏。
- **投影校验加密**：每行查 32 个散布列 + 16 个连续列（原来 8 + 0），
  连续块用来抓"一整段连续列错"而散布列用来抓零星错列。
- **两级门禁保持不变**：默认 `--gpu` 仍然拒绝，`RMI_GPU_UNSAFE=1` 放行。

**第三轮：抽样密度才是关键**。上面两轮修复把相同结果从 3/10 提到 19/20、24/30，
剩下的漏网之鱼用逐阶段校验和定位到 `pipeline.dec_net.cond_embed`
（rows=256, n_in=384, n_out=8192）——一个**投影**，而且差异极小
（和的相对差 4.4e-5，平方和的相对差 4.1e-5）。反推出大约**几百个元素**散落在
两百万个元素里，不是整片损坏。

抽样抓不住这种损坏：

- 投影检查每行只看 48 列（32 个散布列 + 16 个连续列）。两百万个元素里散落几百个
  错值，每行命中的概率只有 0.6%，256 行合起来也只有 37% 能抓到。
- 卷积检查抽 4096 个元素。98304 个元素里一个 64 元素的 tile 只有 33% 被命中，
  而乘法步长采样还可能整块整块地跳过。**把卷积抽样从 512 提到 4096 之后，
  24 次跑出 23 次相同**——卷积这一路的口子就是这样补上的。

修投影的办法是**换一种抽样方式**：除了"每行抽几列"，再加"**挑几行把每一列都
看一遍**"。散落的错值只要落在这几行里就一定看得见。`GPU_CHECK_ROWS = 4` 时
每次跑 1 分 10 秒（比加检查之前的 1 分 22 秒还快，因为重试少了），**30 次跑出
30 次相同**；`GPU_CHECK_ROWS = 16` 时 18/18 相同，但一次要 2 分 20 秒，比纯 CPU
（2 分 53 秒）快不了多少，不划算，所以取 4。

注意力检查同样加了一段连续维度（`GPU_ATTENTION_CHECK_BLOCK = 16`），抓"一个 head
里连续几个维度错"的情况。

**校验层那次的结论**：`RMI_VULKAN_VALIDATION=1` 打开校验层后，我们的 Vulkan 用法
**没有报任何违规**；但层一开，NVIDIA 驱动就在 `vkCreateComputePipelines` 里
SIGTRAP（落在 `libnvidia-gpucomp.so` 的着色器编译器），关掉 GPU-AV 也一样崩。
这是驱动侧缺陷，所以"用校验层定位到具体违规"这条路走不通。

**co-tenant 不是原因**：机器上的另一个租户后来退出了（`nvidia-smi` 显示 0% 利用率、
只有 Xorg），损坏照样出现，而且频率没有下降。

**第四轮：抽样的形状比密度更要紧**。把投影检查改成"按输出铺开"之后，正确率反而
掉到 22/24——因为采样被组织成了 1024 个**宽 64 列**的块，块空间只被覆盖了 8.3%。
改成**宽 8 列**的块之后，同样 65536 个样本能覆盖 67% 的块空间，正确率回到
**24/24、24/24（两批共 48/48）**。教训：抽样的**密度**一样时，**样本怎么组织**
决定了能覆盖多大的空间。

顺便量到了两个数（`RMI_GPU_VERIFY_TRACE=1` 会在跑完后打印）：

- 健康投影与 CPU 内核的最大相对差：**5.6e-6**；
- 损坏的投影与 CPU 内核的相对差：**7.8e-2**。

也就是说检查的阈值（1e-4）夹在中间很宽裕，**漏网跟阈值无关，是抽样没盖到**。
另外用这些数字可以把"损坏"和"求和顺序不同"分开：两者差三个数量级。

**检查本身也要并行**。检查原来是一个线程里的循环，一次跑要 2 分 38 秒（比纯 CPU
的 2 分 53 秒快不了多少）。把检查改成用 `ComputePool` 分给 16 个线程以后，
一次 1 分 36 秒。**卷积**的检查同理（`GPU_CONV_CHECK_SAMPLES = 65536`，
同样铺开采样）。

**六个变体在 GPU 上的实测**（256×256，各自的真实步数/CFG）：

| 变体 | 配置 | 结果 |
|---|---|---|
| base | 2 步 CFG 5 | **72/72 逐位相同**（三批 24/24 + 24/24） |
| flow | 8 步 CFG 5 | 3/3 逐位相同；与 CPU 参考的 **PSNR 81.24 dB** |
| turbo | 4 步 CFG 1 | 3/3 逐位相同 |
| edit | 4 步 CFG 1 + 参考图 | 3/3 逐位相同 |
| edit-turbo | 4 步 CFG 1 + 参考图 | 3/3 逐位相同 |
| edit-base | 8 步 CFG 5 + 参考图 | **未测**（CPU 上就要 11 分 40 秒，测的时候机器被别人占满） |

健康样本与 CPU 参考的 PSNR 是 **80 dB 上下**，损坏样本是 26~36 dB，所以 flow 的
81.24 dB 说明 GPU 结果的内容也是对的，不只是"自己跟自己一样"。

单次 1 分 36 秒（CPU 路径 2 分 53 秒）。CPU 路径的输出没有变（md5 与基线一致）。
门禁文案仍然写着 "not reproducible"，因为这不是数学保证，只是"这段时间内没再
看到不一样的"。

#### 第二次排查（2026-10-24）：投影偶发出错已经能复现

这一轮把故障缩小到一个具体形状，并拿到了一个能稳定复现的用例。

**可复现用例**：`src/ops/kernel/vulkan.rs` 里的
`bf16_prepared_narrow_projection_survives_many_rounds`，用真实的
`transformer_blocks.8.attn.to_k.weight`（n_in=3072、n_out=3072、rows=20）反复调用，
每次与 CPU 内核比对。`PROBE_ROUNDS=4000` 时失败率约 1/667，与真实流程里
2749 次调用出现 1 次严重错误同量级。**这是之前所有探针都复现不出来的原因：
400 轮根本命中不了 1/667 的事件。**

**失败形态**：

- 每次失败，20 行全部有误差，但「整行全零」的行是随机子集，且偏向行号大的一侧；
- 错值既不等于「只用输入前一半 k 的结果」，也不等于「只用后一半 k 的结果」
  （匹配数为 0），所以不是输入的某一段坏掉；
- 第一次算出的元素总是对的，说明 dispatch 本身跑起来了。

**GPU 路径本身是确定的**：同一进程里连续四次 256×256 完整流程，第三与第四次
逐位相同，对 CPU 参考的 PSNR 达到 80.13 dB（差异只来自 GPU 与 CPU 的求和
顺序不同）。另外两次分别是 26.51 dB 与 36.05 dB，正好对应投影出错的那一档。
所以状态是「好窗口」与「坏窗口」：好窗口里 GPU 与 CPU 逐位一致，坏窗口里某一个
投影出错，之后整条采样链被污染。

**这一轮排除的**：

- **系统内存压力**：另一租户从 74 GB 降到 36 GB、系统可用内存从 42 GB 升到
  79 GB 时，坏窗口依然出现；
- **GPU 竞争强度**：平均利用率 84% 的窗口里 4000 轮零失败，而利用率更低的
  窗口反而失败 6 到 8 次，两者无对应关系；
- **主机写的可见性**：`begin_commands` 里的 HOST→COMPUTE 屏障本来就盖住了权重
  上传（上传发生在屏障之前），另外还加了 `flush_mapped_memory_ranges` 与
  `invalidate_mapped_memory_ranges`，三重保护下仍然出错；
- **kernel 逻辑与投影复用**：f64 参照下 GPU 精确，复用投影把真实流程的异常率从
  18/2700 降到 1/2749。

**这一轮的处置**：GPU 路径改成两级。

- 默认仍然拒绝 `--gpu`，错误消息直接说明原因与开关名；

**2026-11-11 的复测**（本机 aarch64，`RMI_GPU_UNSAFE=1` 放行，256×256 / 4 步 /
CFG 1 / seed 42）：连跑 **6 次，PNG 逐位相同**（6 个 md5 一致），这是 TODO-044
记下「相同结果」之后的又一次观察。门禁**仍然保留**：TODO-044 自己写明这
不是数学保证，只是「这段时间内没再看到不一样的」，而本条的根因（设备投影
偶发整片出错）至今没有定位到具体成因。6/6 不能替代那个定位。

对照：同一台机器上 ERNIE-Image 的设备路径（[TODO-073](#todo-073-ernie-image-的设备路径k-quant-慢q8_0-快-28-倍)）
也是连跑 3 次逐位相同 —— 说明「可复现」这条门禁判据本身是可测的，
Mage-Flow 缺的仍然是根因，不是证据。
- `RMI_GPU_UNSAFE=1` 放行 GPU，速度约为 CPU 的两倍（256×256 约 1.5 分钟对 2 分
  53 秒），代价是结果不保证可复现；
- 采样器每步检查 latent 是否还可用（非有限或整体塌成零就拒绝），被拒绝时
  自动用 CPU 重采样（`ComputePool::disable_gpu_matmul_for_scope`），因此最严重的
  坏窗口会变成「慢但正确」而不是一张平涂图。CPU 路径逐位不变。

**下一步**：

- 用 `examples/mage_conditioning_probe.rs` 可以把两半分开验：Qwen3-VL 的 prefill
  与 DiT 的一步各自连跑三次都逐位相同，所以问题不在条件编码，也不在单步；
- 复现依赖「窗口」，因此要在无邻居的机器上连续跑 `PROBE_ROUNDS=4000` 若干轮，
  抓住一个坏窗口再抓现场；一旦能在受控环境里复现，就可以在 `float_matmul_tiled`
  里加一个校验和来定位是哪些 workgroup 没写；
- `RMI_GPU_CPU_CHECK=1` 现在会打印每个投影与 CPU 的相对差（健康值在 1e-6 附近），
  两次运行对比即可找到第一个出错的投影。

**下一步**：

- 这台机器是共享的：另一租户的进程占 74 GB 统一内存、GPU 利用率 93%、
  系统内存可用仅 42 GB。**先在没有邻居的时段复测**，再判断剩下的是驱动问题
  还是代码问题。
- 若要继续查，需要一个能在真实流程里稳定复现的用例：探针（144 形状 × 4 轮）
  始终不复现，只有完整的 Mage-Flow 流程会。下一步应把「文本编码器的 GPU
  prefill 之后」这一段纳入复现范围——这是真实流程与探针之间最后没被覆盖的差异。
- `src/models/diffusion/dreamx/kernels.rs` 里的 `RMI_GPU_CPU_CHECK=1` 就是为
  继续查这件事留的开关。

**顺带**：`float_matmul_tiled` 的 GPU 求和顺序与 CPU 的 vec4 累加器不同，真实
权重下差 1e-3，被 8 步扩散放大到约 29.5 dB。f64 仲裁显示 GPU 更接近真值
（7.79e-5 vs 1.18e-4），所以这不是错误，不要为了「和 CPU 一致」去改 kernel。

### TODO-045: LFM3 的 GPU 路径只搬了 matmul；剩余算子占 CPU 时间 18%，集显上 Auto 比 CPU 慢 8.4–9.25×

**现状**：`src/models/lfm3/trunk/session.rs:102-127` 把每层的 `w_gate/w_up/w_down`、
`wq/wk/wv/wo/attn_gate`（attention 层）或 `shortconv_in/shortconv_out`（shortconv 层）
加上最后的 `model.output` 注册成 `LinearBinding`，这就是 GPU 上的全部负载。诊断行
`LFM3 projections: requested=Auto backend=Vulkan; attention/shortconv/KV=Cpu` 描述得准确：
**GPU 只做线性投影，attention / shortconv / KV cache 全在 host。**

仍在 CPU 上的算子与仓库现成 shader 的对应关系：

| 算子 | 次数 | 仓库里有没有现成 shader |
| --- | --- | --- |
| RMSNorm（attn/q/k/ffn/post_*/out_norm） | ~5/层 | ✅ `shaders/glsl/rms_norm.comp` |
| q_norm+k_norm+RoPE（双 freq_base 1e6 / 1e4） | 1/层 | ✅ `shaders/glsl/qk_norm_rope.comp`（freq_base 走系数缓冲，`fill_rope_neox` 已有） |
| SwiGLU（silu*up） | 1/层 | ✅ `shaders/glsl/silu_mul.comp` |
| 残差加 | 2/层 | ✅ `shaders/glsl/add.comp` |
| softmax | 1/层 | ✅ `shaders/glsl/softmax.comp` |
| KV 写入 + F32/F16 转换 | 1/层 | ✅ `shaders/glsl/kv_write.comp` |
| attention（full + **SWA**） | 1/层 | ⚠️ 有 `attention_online/scores/values`，但**整个 `shaders/glsl/` 里没有任何支持滑窗的 shader**；LFM3 有 5 个 SWA 层（layer 4/8/12/16/20） |
| head gate（sigmoid） | 1/层 | ❌ **没有 sigmoid shader**；`silu_mul.comp` 有 `exp_mode` 字段可改 |
| shortconv（`l_cache=7` + b/c 三路门控） | 1/层 | ❌ LFM3 特有结构；`qwen35_recurrent_conv.comp` 的布局不同 |

**但"算子没上 GPU"不是主要矛盾。** LFM3 每 token 的投影工作量与本机实测：

- 配置：24 层，d=2048，16 head × 128，n_ff=7680，vocab=128000，11 个 attention 层 +
  13 个 shortconv 层，5 个 SWA 层，`lfm3.attention.head_count_kv` 为 0 的层即 shortconv 层。
- 每 token 投影 **1.728 G MAC = 3.46 GFLOP**（含 lm_head 2048×128000）。
- 本机 Q8_0 GEMM 实测 **80 GFLOPS**（`target/release-fast/micro-bench`，AVX2/FMA）→
  投影理论 **43.2 ms/token**。
- 实测 CPU 解码 **52.6 ms/token**（`--profile`：947.02 ms / 18 token）。

→ **投影占 CPU 解码约 82%，其余所有算子加起来约 18%。**

**影响**：

1. **在这类集成显卡上继续往上搬算子收益有限**。本机（Intel Meteor Lake 集显）
   实测 Auto 路径输出与 CPU 路径逐字节相同，但比 CPU **慢 8.4×**（32 步 greedy：
   2.65 s 对 22.34 s）、**慢 9.25×**（216 token prompt + 8 步 decode：13.80 s 对
   127.63 s）。原因是集显的设备内存就是系统内存，1.85 GB 权重"放进显存"后每次投影
   都要过内存总线，比 CPU 直接读 RAM 更慢。把剩下 18% 的算子搬上 GPU，还要付 GPU
   往返开销，净亏。
2. **在独显上结论相反**：投影一旦真的快起来，host 端那 18% 会立刻变成主要瓶颈，
   那时搬 attention / norm / RoPE 才有意义。
3. **`--gpu` 与 `--compute auto` 都不提示"这次比 CPU 慢"**。用户看到的是"GPU 好慢"，
   而不是"这个后端在这个设备上不划算"。这与 TODO-038 的结论同源：自动分发应当
   **按实测选后端**（首 token 实测延迟对比 CPU 基线，落后就回退），而不是按模型
   大小或"用户要求了 GPU"。

**下一步（建议顺序，仅当有独显可测时）**：

1. **滑窗 attention** —— 给 `attention_online.comp` 加 window 参数。影响 5/11 个
   attention 层，且 SWA 层是最吃 attention 的；这是唯一一处"必须新写"的 attention 逻辑。
2. **q/k norm + RoPE 合并** —— `qk_norm_rope.comp` 现成，push constant 已有
   `q_heads/k_heads/head_dim/norm_flags/eps`，只需喂两组 freq_base 的系数缓冲
   （`fill_rope_coefficients` / `fill_rope_neox` 已在 `src/vulkan/ops.rs`）。
3. **RMSNorm 融合进投影后处理** —— 照抄 `silu_mul.comp` 的 `exp_mode` 模式。
4. **head gate 的 sigmoid** —— 每层只有 16 个标量，最便宜，放哪儿都行。
5. **shortconv** —— 13 层 × 6144 × 7 tap，FLOPs 最小但结构最 bespoke，最后做。

**尚未验证**：以上任何一项都没有实机数据；本机只有集显，而集显上 GPU 路径本就不该
作为加速路径。需要一台有独立显存的机器重测，否则做了也证明不了收益。

**关联**：[TODO-038](#todo-038-cpugpu-自动分发应该按实测选后端而不是按模型大小)、
`docs/develop/SUPPORTED_MODELS.md` 的 LFM3 行（状态 `Verified`（CPU/Q8_0 运行）；
`Experimental`（Vulkan 投影））、`tools/oracle/lfm3/README.md` 的
「同一台机器上的 Auto/Vulkan 投影路径」一节。

---

### TODO-046: qwen3 的 CPU 与 Vulkan 输出差一个 token（arch 白名单里的模型也不对齐）

**现状**：`qwen3` 在 `ComputePolicy::has_complete_device_path()` 的 arch 白名单里
（`llama|qwen3|qwen35|qwen35moe|dory`），所以 `--compute vulkan` 一直放行，走的是
`Qwen3VulkanSession` 那条「整个解码驻留设备」的路径。

但同一条 prompt 下，**CPU 与 Vulkan 的 greedy 输出不一致**。本机（Qwen3-0.6B-Q8_0，
Intel 集显，`--kv-cache f16`）：

| 路径 | 输出 |
| --- | --- |
| CPU | `Hello! How can I assist you today? 😊`（11 token） |
| Vulkan | `Hello! How can I assist you today?`（9 token） |

差的是末尾那个 emoji token —— **典型的数值边界翻转**：两条路径的 logits 排序在
某一个 token 上几乎持平，谁赢取决于累加顺序与舍入。

**这不是「去掉白名单」那次改动引入的**：把改动 stash 掉、退回改动前的 HEAD 重编译，
同样的命令输出**完全相同**（同样少一个 emoji）。而且在 `--prompt hello` 这种单
token prompt 下差异更大（CPU 出英文、Vulkan 出韩文），说明**偏离不止在最后一步**。

**仓库里其实已经记过一次同样的事**：`docs/develop/UNIFIED_COMPUTE_VALIDATION.md`
写的是远程 AMD RADV NAVI31 上 `Qwen3-0.6B-Q8_0.gguf` 的**模型准入门禁在 token 0
就失败**（GPU `4.476106`、CPU `4.4538846`、abs `0.022221565`，门槛
`MODEL_LOGIT_ABS_TOLERANCE = 2e-3`）→ `--compute vulkan` 以此 exit 1。
**本机（Intel 集显）过得了 token 0 的门禁，却在后面对齐时翻了一个 token** ——
同一类数值偏离，两台机器上表现不同。

**JEV 也踩到同一件事**：qwen3 的 JEV 在 Vulkan 下被模型自己的契约门禁拦下
（`model logit mismatch at 0: gpu=5.6980243 cpu=3.4995377 abs=2.1984866`）→ HTTP 500；
而 llama-1B 的 JEV 在 Vulkan 下与 CPU **逐位相同**。**同一个 Vulkan 策略下，两个
白名单里的 arch 一个能过门禁、一个过不了** —— 又一次说明白名单不是能力证明。

**一个可能相关的实现细节**：`src/models/qwen3/trunk/prefill.rs:195` 的
`validate_gpu_contract` 要求 `need_logits` 为真，而常规 prompt prefill
（`need_logits = false`）**不会**跑那个 16 步 decode 契约探针
（`MODEL_CONTRACT_DECODE_STEPS = 16`）。所以**真实生成路径上的 token 偏离没有被
任何门禁看着**。这条线索没验证，只是探针的触发条件摆在那里。

**为什么值得单列一条**：

1. **白名单的含义被高估了。** `has_complete_device_path()` 现在只用来决定「要不要
   打印警告」，但名字读起来像「这些架构的设备路径是对的」。实际上 `qwen3` 的设备
   路径与 CPU 不对齐 —— 白名单里至少有一个成员不满足它字面上的意思。
2. **LFM3 与 OCR 的同项对比是干净的**（Auto/Vulkan 与 CPU 逐字节相同），所以这不是
   「设备算错了」，而是 **qwen3 这条驻留式 Vulkan 解码路径自己的累加顺序**与 CPU
   kernel 不同。

**待查**：

1. `src/models/qwen3/trunk/session.rs` 的 `Qwen3VulkanSession` 与 CPU kernel 在
   **RMSNorm / softmax / silu** 上的归约顺序是否一致（设备端若用并行归约，累加顺序
   就和 CPU 的顺序归约不同）。
2. 差异是**只在边界处翻转**，还是会在长输出里放大？本机只看了 24 token 的短输出。
3. `src/models/llama/trunk/session.rs` 有同构的 `dense Vulkan currently requires F16 KV`
   守卫，llama 也在白名单里 —— **llama 是否也有同样的 token 差异**，本轮没测
   （没有在本机跑 llama 的 CPU/Vulkan 对比）。

**关联**：[TODO-038](#todo-038-cpugpu-自动分发应该按实测选后端而不是按模型大小)、
[TODO-045](#todo-045-lfm3-的-gpu-路径只搬了-matmul剩余算子占-cpu-时间-18集显上-auto-比-cpu-慢-84925)、
`docs/develop/UNIFIED_COMPUTE_VALIDATION.md`（远程 AMD 上 qwen3 token 0 门禁失败）、
`docs/develop/UNLIMITED_OCR_2_REVIEW.md` §7.5。

### TODO-047: WeVisOne 的「是否本模型」判定散在 6 处，且 `enable_thinking` 被静默覆盖

**现状**：WeVisOne-VL-4B 的适配在 6 个位置各自判断「这是不是 WeVisOne」，写法完全一样：

```rust
source.metadata("wevisone_vl.source_revision").is_some()
```

| 位置 | 用途 |
| --- | --- |
| `src/app/server/api/tools.rs:505` | 强制 thinking（HTTP prompt 构造） |
| `src/app/text/qwen3.rs:65` | 强制 thinking（CLI 文本路径） |
| `src/models/qwen3/trunk/forward.rs:758` | 强制 thinking（generate 路径） |
| `src/app/jev/single/qwen3.rs:171` | 强制 jinja 渲染 |
| `src/app/server/mod.rs:1464` | 决定是否构造 `wevisone_options` |
| `src/models/qwen3/vision/mod.rs:612` | 视觉预计算保留 F32 |

**强制 thinking 这件事本身是对的**：WeVisOne 自带的 `chat_template.jinja` 里**完全没有
thinking 机制**（没有 `enable_thinking` 变量、没有 `<think>` 标记，助手轮就是字面
`assistant\n`），官方实现走 `AutoProcessor.apply_chat_template`，所以官方 prompt
永远不含空 think 块。不强制的话 HTTP/CLI 都会插入 `<think>\n\n`，prompt 与官方不一致。

**问题一（判定重复）**：同一判定散 6 处，已经吃过一次亏——thinking 阈值在
forward.rs / qwen3.rs / jev 三处各写一遍。将来加 WeVisOne 变体（或改名 revision
标记）时漏改一处就是静默的行为分叉。**修复方向**：抽一个
`qwen3::is_wevisone(source) -> bool`（或把它做成 `Qwen3Config` 的一个字段），
6 处收敛到 1 处，并补一条单测锁住「哪些路径必须按 WeVisOne 处理」。

**问题二（对外可见的静默覆盖）**：`src/app/server/api/protocol.rs:52-58` 的
`enable_thinking` 是**客户端字段**（顶层 `"enable_thinking": false` 或嵌套
`chat_template_kwargs`）。`tools.rs:505` 的
`enable_thinking.unwrap_or(false) || is_wevisone` 会把客户端**显式传的 false 也
覆盖成 true**。功能上无害（WeVisOne 本来就不 think，客户端拿到的仍是「无思考」
输出），但 API 语义上是「显式请求被静默忽略」。
**修复方向**：要么在强制时打一条 warning，要么在 `protocol.rs` 的字段注释里写明
「WeVisOne 忽略该开关，模型没有 think 模式」。后者零风险，前者更友好。
注意本条与已有的
[enable_thinking 默认值条目](#high-priority) 是**不同**的问题：那条讲的是
`unwrap_or(false)` 这个默认值本身，那条讲的是 WeVisOne 的强制覆盖。

**关联**：`tools/oracle/wevisone_vl/README.md`、`tools/oracle/wevisone_vl/validation.md`。

### TODO-048: `--mmproj` + 纯文本仍会把 qwen2vl / qwen3vlmoe 送进 hybrid 分支报错

**现状**：`src/main.rs` 的多模态分支条件原本是「有没有 `--mmproj`」，而 qwen3vl 的
多模态路径**要求同时有 image/video/audio**。只有 `--mmproj`、没有媒体的请求因此
掉进通用分支里的 `HybridTextModel::from_source`，而它只认
`qwen35 | qwen35moe | edge0` → 报
`Failed to parse hybrid text model: Unsupported hybrid architecture: Some("qwen3vl")`。

WeVisOne 适配这条分支时已把 **`qwen3vl`** 修好：条件改成

```rust
image.is_some() || video.is_some() || audio.is_some()
    || (explicit_mmproj.is_some() && arch != "qwen3vl")
```

无媒体时 qwen3vl 走文本路（`run_shared_inference` / `run_qwen3_inference`），
实测 Qwen3-VL-4B-Instruct 纯文本从报错变成正常生成。

**还没修的**：`qwen2vl` 与 `qwen3vlmoe` 同样带 `--mmproj` + 纯文本时，仍会掉进
hybrid 分支报同一个错。**这是改动前的既有行为，不是回归**，但与已修好的 qwen3vl
不一致。

**为什么不顺手放开**：把条件扩成
`!matches!(arch.as_str(), "qwen2vl" | "qwen3vl" | "qwen3vlmoe")` 需要先确认这两个
arch 的**文本路**真的可用：

- `qwen2vl` 走 qwen3 trunk（`Qwen3Config::from_source` 支持该 arch，rope 走
  `Qwen3Rope::Mrope` 分支），理论可行，**未实测**；
- `qwen3vlmoe` 是 MoE（`config.moe`），`run_qwen3_inference` 是否正确处理共享专家
  **未实测**。

在没有这两个 arch 的权重时放开，会把「明确的报错」变成「静默走错路」——后者更难
排查。

**待办**：

1. 找一个 `qwen2vl` GGUF（+ mmproj），跑 `--mmproj` + 纯文本，确认文本路可用后再
   把 `qwen2vl` 加进豁免名单；
2. 同样验证 `qwen3vlmoe` 的 MoE 文本路；
3. 两者的媒体路（`--image` / `--video` / `--audio`）保持不变，不在豁免范围内。

**关联**：WeVisOne 适配分支（`codex/wevisone-vl-4b`）的 `src/main.rs` 改动与
`docs/usage/qwen3.md` §3 的 M-RoPE 记录。

---

### TODO-049: `GpuLinear` 的 runtime 缓存按**权重指针**分桶，多权重模型按权重数付管线创建费

> **2026-11-06 补充（EmbeddingGemma-2）**：同一根因在 EG2 上更明显。EG2 的文本塔
> 有 24 层 × 约 6 个权重 = 约 145 个投影，每个都重建一次 `BatchedLinearRuntime`，
> 于是**文本塔上设备一律是亏的**：媒体路径（260 token）慢约 0.7 s（6.97 s vs
> 6.26 s），短提示（7 token）慢得多（0.49 s vs 0.08 s）—— 固定开销摊不开。
> EG2 的自动策略因此把文本塔与音频塔的投影都留在主机 ✓
> （`src/models/embeddinggemma2/text.rs` ✓、`audio.rs` ✓），只让视觉塔上设备 ✓
> （它的投影与注意力都是净赚 ✓）。修好这个 TODO（runtime 按形状共享 ✓）后，
> 文本塔与音频塔就能一起上设备 ✓。


**现象**：ZDTaichu5.0 的视觉塔上设备后，投影只快 25%（CPU 36.93s → 设备 27.62s）。
一开始以为是激活往返，逐项量下来不是：

| 项 | 实测 | 占比 |
| --- | --- | --- |
| runtime 创建 | **213ms × 131 个权重 = 27.9s** | ~100% |
| fence 延迟 | 2.5ms × 229 次 submission = 0.57s | 2% |
| host↔device 传输 | 写 1.21GB + 读 1.55GB ≈ 30ms | 0.1% |
| 权重上传 | 1.64GB ≈ 1.3ms | ~0% |

`BatchedLinearRuntime::new` → `Qwen3Ops::new_with_size` 每次都要建管线/描述符，
**实测 213ms/次**（`BatchedLinearRuntime::new(ctx, 1034, 1280, 3840, 2)` 连建 10 次 = 2.13s）。

**根因**：`src/ops/kernel/vulkan.rs` 的 `GpuLinear` 持有
`state: Mutex<Option<LinearState>>`，而 `LinearState.weight` 是**权重指针**：

```rust
if state.as_ref().is_none_or(|cached| {
    cached.weight != key || cached.spec != spec || cached.shape != shape
})
```

每个 `Weight` 的 `VulkanKernel` 各有一个 `GpuLinear` → 每个权重一份缓存 → 第一次用这个
权重就重建一次管线。视觉塔 131 个权重**每个只用一次** → 131 次全价创建。
而塔的权重只有约 6 种形状（每层 4 种 × 32 + head 2 种）→ 同一形状被重复建了几十次。

**影响面**：任何**权重多、每个权重用得少**的模型都中招（视觉塔、DiT、多投影塔）。
权重少但复用多的模型（LLM 的每层权重被很多 token 复用）看不出来，因为第一次创建后
命中缓存了。

**待办**：

1. 把 runtime 缓存的键从权重指针改成 `(format, n_in, n_out, tile_rows)`，进程级共享
   （管线与权重无关，权重是 dispatch 时绑定的 buffer）。预计 ZDT 视觉投影
   27.6s → **~2s**，创建次数 131 → 个位数。
2. 缓存要挂在比 `GpuLinear` 更外层的地方（`GpuLinear` 是 per-weight 的），且要考虑
   `descriptor_capacity` / `output_slots` 也进键。
3. 改完要复核：共享 runtime 的并发安全性（多个 `Weight` 同时用同一 runtime）、显存
   里权重 buffer 的生命周期、以及现有 GPU 路径的数值不变。
4. 之后再看 fence 批处理（229 次 × 2.5ms = 0.57s）：`BatchedLinearRuntime::with_output_slots`
   + `run_group_gpu` 已支持把一层的多个 matmul 记进同一个 command buffer。

**不做**：激活常驻。当前传输只有 ~30ms，**不是瓶颈**；等 1、4 做完再谈。

**关联**：`src/ops/kernel/vulkan.rs`（`GpuLinear::state` / `LinearState`）、
`src/vulkan/ops.rs`（`BatchedLinearRuntime::new` → `Qwen3Ops::new_with_size`）、
`tools/oracle/zdt/README.md` 的 device path 一节。

---

### TODO-050: `cargo build --all-targets` 在 main 上就失败，`cargo test`（不带 `--lib`）跑不起来

**原现象**（已修）：

```
error[E0063]: missing fields `sway_sampling_coef` and `time_grid` in initializer of `AukOptions`
error: could not compile `rust-model-inference` (test "auk_flash_smoke") due to 1 previous error
```

`cargo build --all-targets` 退出码 101 → **不带 `--lib` 的 `cargo test` 会先在编译期
失败**，整套集成测试（`tests/*.rs`）跑不到。

**已修（AukOptions 那条）**：`tests/auk_*` 的构造补回了 recipe（AuK-Base 用
`sway_sampling_coef: Some(-1.0)`，AuK-Flash 用 `time_grid: Some(FLASH_TIME_GRID)`）。

**复发**（2026-11-09，PR #217 修）：AukOptions 修好后，`--all-targets` 又被**新落地的目标**打断 ——
5 个 example 与 1 个集成测试用了 feature-gated 的 API，却没声明 `required-features`：

```
error[E0433]: cannot find `parity_trace` in `rust_model_inference`          (test "nemotron3_nano_moe")
error[E0601]: `main` function not found in crate `astronex_text_determinism`
error[E0432]: unresolved import `rust_model_inference::ops::get_vulkan_context`  (example "vk_mem_info")
```

**已修（PR #217）**：`Cargo.toml` 给这 6 个目标补上 `required-features`
（5 个 example 要 `vulkan` —— 其中 4 个文件顶部虽有 `#![cfg(feature = "vulkan")]`，
但那样整个 crate 会被 cfg 空、`main` 消失，反而报 E0601；`nemotron3_nano_moe`
要 `parity-trace`，它在不带 cfg 门的地方调 `parity_trace::token_ids`）。
`.github/workflows/ci.yml` 的 Vulkan job 加了两条
`cargo check --locked --all-targets`（默认特性 + 全特性各一条），这类断裂从此当场变红。

**仍待办**：`AukOptions` 的 recipe 字段没有默认值，容易漏填 —— 考虑给
`sway_sampling_coef` / `time_grid` 一个与 pipeline 默认一致的 `Default`，
或改成构造期必填的 builder。

**关联**：`tests/auk_flash_smoke.rs`、`src/models/auk*` 的 `AukOptions`、
`Cargo.toml` 的 `[[example]]`/`[[test]]`、`.github/workflows/ci.yml`。
### TODO-051: Mellum 的 GPU 路径能跑但打不过 CPU——decode 是单行投影，设备带宽只有 NEON 的一半

Mellum 12B-A2.5B（Q8_0，12.9 GB）在 aarch64 上接 Vulkan device 路径，
**功能上没问题**：`RMI_PARITY_TRACE` 双后端对拍 15466 个检查点**逐位一致**
（sum/min/max/head 全同），JEV 单问也一致（A 0.2219 / B 0.7781）。
但 `--compute auto` 比 `--compute cpu` **慢 40%**。

#### 已做：批量 prefill（真正的形状修复）

`MellumRunner::forward_tokens`（`src/models/mellum/trunk/batch.rs`）把 prompt
作为一批求值，每个投影一次拿到全部行，权重矩阵每批只读一次而不是每 token 一次。
**逐位一致**：投影是逐行乘加，Q8_0 kernel 每行独立累加，一行的结果与同批有多少行无关
（实测输出头 2304→98304 在 2/4/8 行下与单行设备路径、与 CPU 路径都是 **0 mismatch**）；
attention、MoE 路由、专家折叠都按行、按 slot 顺序走，与逐 token 路径同序。
哨兵 `tests/mellum_batched_prefill.rs`（42 token / 98304 logits 逐位相同）。

收益（权重流量，不是设备速度）：

| 测量（8 线程，各 3 次） | 逐 token | 批量 |
|---|---|---|
| JEV prefill（21 token prompt） | 2585 ms | 1014–1102 ms |
| 端到端（21 token prompt + 8 output token） | 1006–1119 ms | 594–643 ms |
| 端到端（183 token prompt + 8 output token） | 7558–8211 ms | 2911–3078 ms |

单 token 要流过 2.1 GB 权重（每层 22.5 MB attention + 24 专家 × 3 × 2.19 MB，28 层），
批量后每个权重每批只读一次。

#### 新证据（2026-11-04，PyTorch CUDA 对照跑通）

之前"Mellum 的 CUDA 对照跑不了"（另一个租户占 82.7 GB 统一内存，CUDA 路径要
~48 GB 峰值）这一条作废：机器空出来之后 `tools/bench_mellum_pytorch.py --device cuda`
跑通了（PyTorch 2.15.0+cu132，23.13 GB peak，空闲 GPU）：

| | bytes/token | decode | 有效带宽 |
|---|---|---|---|
| PyTorch CUDA bf16 | 4.42 GB | 29.4 tok/s（33.9 ms/token） | **130 GB/s** |
| 我们的 CPU Q8_0 | 2.35 GB | 22.7 tok/s（44 ms/token） | 53 GB/s |
| 我们的 Vulkan Q8_0（实测只有输出头） | 2.35 GB | — | 66 GB/s |

**这台机器上的设备能跑到 130 GB/s，我们的 Q8_0 device kernel 只到 66 GB/s
——一半。** 所以"设备赢不了"不是设备的天花板，是我们 kernel 的访问模式：
`q8_matmul_dp4a.comp` 让**每个线程独占一个输出行**，一个 warp 的 64 个线程
因此读 64 个相距 2448 字节的扇区，而不是连续字节；PyTorch/cuBLAS 的 GEMV
是让 warp 协同读连续权重再做归约。

如果 Q8_0 kernel 打到 130 GB/s，Mellum 的 decode 是 2.35 GB / 130 GB/s =
18 ms/token = **55 tok/s，对 CPU 的 22.7 tok/s 赢 2.4×**——那才是 Mellum
设备路径真正该有的样子。仓库里已经有转置权重的机制（BF16/F32 走
`record_transposed_float_matmul_rows`，"上传时转置，让 tiled float kernel
沿归约轴 unit stride 走"），**Q8_0 没有走这条路**。

顺带纠正一条旧结论：我们的 CPU decode 只跑到 53 GB/s，而输出头单独测能到
115 GB/s → **CPU decode 不是带宽受限，是 785 次 pool dispatch 受限**。

#### 现状：设备仍然赢不了，原因是上传而不是形状

批量形状对设备**友好**（设备一次读完权重，CPU 每行重读）：
输出头 2304→98304 在 8 行时设备 3.53 ms vs CPU 17.55 ms（5×），
32 行时 5.32 ms vs 75.60 ms（14×）。但 prefill 只需要**最后一个位置**的 logits，
所以这个形状用不上；而能用的形状（`wq`/`wo` 各 10 MB）设备只有 ~12 GB/s
（`n_out=4096` 只有 64 个 workgroup，喂不饱设备），CPU 是 ~100 GB/s。

更关键的是**上传**：host→device 只有 ~10 GB/s（DRAM 读是 ~100 GB/s），
prefill 里每个权重只用一次，上传就是纯开销。实测 JEV prefill：
CPU 1073 ms → 设备接管 `wq`/`wo` 后 1460 ms。
所以**批量 prefill 默认走 CPU**，`RMI_MELLUM_DEVICE_PREFILL=1` 可以打开设备路径
（serving 场景权重常驻，上传可摊销，这条值得在服务器上复测）。

#### 已做：Q8_0 转置权重布局 + 新 kernel（设备带宽 60 → 163 GB/s）

新增 `shaders/glsl/q8_matmul_transposed.comp` 和 `src/vulkan/q8_layout.rs`：
上传时把 Q8_0 权重按 **64 行一组**重排（每个 block 的 64 行 payload 连续，
scale 紧跟其后），于是一个 warp 读到的 32 行是**连续字节**，4 字节 load 也对齐。

| 设备 kernel（输出头 2304→98304，单行） | 时间 | 有效带宽 |
|---|---|---|
| ggml 原生布局 | 4.0 ms | 60 GB/s |
| 转置布局 | 1.47 ms | **163 GB/s**（2.71×） |
| 设备纯流式（element-wise，240 MB 读+写） | — | 171 GB/s（设备天花板） |

**逐位一致**：转置版与原生版输出 0 mismatch（哨兵
`q8_0_transposed_layout_matches_and_faster`）。

这条路只用在**单行**（decode）。多行走 tiled kernel，而 tiled kernel 的瓶颈是
**激活的 shared-memory 暂存**（每个 block 每 lane 要做 32 token × 8 次 LDS），
不是权重布局——转置版实测只有 0.60×，所以多行仍用原生布局。
（中间试过"按 block 排序"的布局，tiled 仍然更慢；按 64 行分组才让单行版
保持 2.7×。）

host 端转置 240 MB 约 26 ms（9 GB/s），一次性，直接写进设备 buffer
（`upload_static_filled`），所以上传只比原生多约 9 ms。

**默认关闭**（`RMI_GPU_Q8_TRANSPOSED=1` 才开）。转置只搬字节不算数，
数值契约一字未改（round-trip 测试证明布局可逆、黄金位测试证明三种
reduction mode 下与原生 kernel 逐位相同、边界形状 n_out=1/63/64/65/127/129
逐位相同），但寻址写错会**越界读 → GPU fault**（进程崩，不是静默错值），
而目前 Mellum 还没有任何投影真的走设备，所以先不开。

#### 新证据：设备的一次性开销在驱动初始化，不在 kernel

| 一次性开销 | 时间 |
|---|---|
| Vulkan 上下文（instance + device，驱动初始化） | **279 ms** |
| 50 个 compute pipeline 编译 | 11.7 ms |
| 输出头权重上传（240 MB） | 17 ms（转置后 26 ms） |

这 279 ms 之前算在生成时间里，是"`--compute auto` 慢 40%"的主要来源。
现在 `compute_policy().configure_cli()` 在**后台线程**初始化设备，和 12 GB
模型加载重叠 → 21 token prompt + 16 output 的生成时间 1009 ms → 685–750 ms。

#### 现状：kernel 不再是瓶颈，但端到端仍然打平

（下面的"设备转置"是**打开** `RMI_GPU_Q8_TRANSPOSED=1` 测的；默认关闭时
设备走原生布局，输出头 4.0 ms/token，比 CPU 的 2.6 ms 慢，scope 必须留着。）

打开后设备端 Q8_0 kernel 超过 PyTorch CUDA 的 130 GB/s，但 128 token 的生成
CPU 5672–5742 ms、设备转置 5134–5951 ms，**打平**。原因：

1. 单行投影只占每 token 2.35 GB 权重里的 240 MB（10%）；
2. CPU decode 不是带宽受限，是 785 次 pool dispatch 受限（整模型 53 GB/s，
   单个投影 115 GB/s）；
3. 设备每 token 还要付一次提交往返（~0.4 ms）和 logits 读回。

所以 `project()` 的 CPU scope 仍然保留——打平不是赢。

#### 剩下的

1. **prefill 和 decode 之间来回重传权重**（最高优先级）：权重在**同一形状**下
   是常驻的（`GpuLinear` 是 per-weight 的，它缓存自己那一份），但 `GpuLinear::state`
   里的 runtime 是按 (权重, 形状) 建的，而 prefill 是多行、decode 是单行
   → 形状一变 runtime 就重建，权重跟着重传。实测（Mellum 全投影走设备）：
   2 token 时 1047 次上传 / 2.92 GB，8 token 时 2250 次 / 5.56 GB
   （第二份是 decode 重建 runtime 后的重传），32 token 时 2268 次 / 5.60 GB
   —— 即 **每次 prefill 都要重传 2.9 GB**。显存不是问题（旧副本随 runtime 释放，
   峰值 4.36 GB，开关两档一样），问题是**时间和带宽**。
   要让 serving 场景能摊销，得让 runtime 跨形状存活（缓存按 (权重, 形状) 分开）。
2. **把一层的投影合并成一次提交**：每 token 785 次提交，每次 ~0.4 ms = 314 ms，
   比 kernel 本身还贵。仓库里已有 `record_group_matmul`（多权重共用一个
   command buffer），Mellum 的 trunk 还没用。
3. **decode 阶段仍然是单行**（每 token 785 次 pool dispatch，20 线程比 8 线程慢：
   1426–1711 ms vs 1006–1119 ms）。TODO-037 已经改过一次等待逻辑（自旋 → yield），
   但 Mellum 这个规模仍然不划算。**小投影不该扇出整个线程池**——1 行的 matmul 分给
   8 个 worker，唤醒成本可能超过计算本身。
2. **设备端 Q8_0 kernel 的带宽（最高优先级）**。PyTorch CUDA 在同一台机器上跑到
   **130 GB/s**，我们的 Q8_0 device kernel 只有 **66 GB/s**。差在访问模式：
   `q8_matmul_dp4a.comp` 让每个线程独占一个输出行，warp 的 64 个线程读 64 个
   相距一行的扇区；而 BF16/F32 已经有转置权重的路子
   （`record_transposed_float_matmul_rows`，上传时转置让 kernel 沿归约轴
   unit stride 走），**Q8_0 没走**。给 Q8_0 加一条转置权重 + warp 协同归约的
   路径，是让 Mellum（以及所有 Q8_0 模型）设备路径能赢 CPU 的前提。
   另有一个次要因素：输出头能跑 66 GB/s（`n_out=98304`，1536 个 workgroup），
   `wq` 只有 12 GB/s（`n_out=4096`，64 个 workgroup）——小 `n_out` 喂不饱设备。
3. **每 token 的输出头 240 MB**。decode 每 token 都要读一遍 240 MB 的输出头，
   CPU 2.09 ms（115 GB/s）vs 设备 3.6 ms（66 GB/s）。设备 kernel 的带宽只有 CPU 的
   57%，这是设备端 Q8_0 kernel 本身的问题。
4. 设备空闲时重测。TODO-038/039 指出这台机器上的 GPU 数字都受另一个租户占用影响
   （nvidia-smi 采样 79–92%），上面的对比在 GPU 0% 利用率下测得，但仍需在独占时复测。


### TODO-052: decode 每个投影一次 pool dispatch，784 次里 26 次的开销大于活

#### 现象与账单

Mellum 12B Q8_0（8 线程）在 aarch64 上 decode 44 ms/token（22.7 tok/s）。拆开：

- 纯计算下界 **23 ms/token**（每 token 读 2.35 GB 权重，CPU 多线程有效带宽 ~102 GB/s；
  单投影实测 115 GB/s）
- pool dispatch 固定开销 **~21 ms/token（48%）**

每 token **785 次** `ComputePool::compute`（28 层 × 28 个投影）。每次的固定成本约 26 µs：
等所有 worker 空闲 → `notify_all()` 唤醒 7 个线程 → 主线程自己也做一份 → 等最后一个
完成（尾部效应）。

而每个投影**每 worker**的活只有 12–98 µs：

| 投影 | 每 worker 读 | 每 worker 算 |
|---|---|---|
| `wq` (2304→4096) | 1.25 MB | ~98 µs |
| `wk` / `wv` (2304→512) | 154 KB | **~12 µs** |
| `wo` (2304→2304) | 693 KB | ~54 µs |
| 专家 up/gate/down (2304→896 / 896→2304) | 258 KB | **~20 µs** |
| 输出头 (2304→98304) | 10 MB | ~784 µs |

→ 784 次里 **26 次的活比 dispatch 开销还小**（wk/wv + 24 个专家投影）。

#### 证据：线程曲线

1→8 线程只有 3.17× 加速（不是 8×），16/20 线程直接倒退 —— 唤醒成本随线程数增长，
8 线程已经吃完并行收益（64 token prompt，best-of-2）：

| threads | 64 token | 相对 1 线程 |
|---|---|---|
| 1 | 8986 ms | 1× |
| 2 | 5050 ms | 1.78× |
| 4 | 3309 ms | 2.72× |
| **8** | **2833 ms** | **3.17×** |
| 16 | 3669 ms | 倒退 |
| 20 | 4212 ms | 更慢 |

#### 改法：把能并行的投影合并进一次 dispatch

一层的依赖结构（`forward.rs` 的单 token 路径）：

```
{wq, wk, wv}        共享同一输入 ✓ 同组
   ↓ attention（逐 query 行）
{wo}                单独一个 ✗
   ↓
{8 专家的 up+gate}  16 个投影，专家之间彼此独立 ✓ 同组
   ↓ silu(gate) * up
{8 专家的 down}     8 个投影，彼此独立 ✓ 同组
```

→ 每层 **4 次** dispatch（现在 28 次）→ 112 次/token（现在 785）→ 开销
20.4 → 2.9 ms/token → **44 → 26.5 ms/token（1.66×，22.7 → 39 tok/s）**。

实现上**不需要新的 pool 原语**：现有的 `compute` 就够，closure 里让每个 worker 顺序做
多个投影（各自按 `ith/nth` 切分）：

```rust
pool.compute(|ith, nth| {
    wq.forward_prepared(..., ith, nth);   // 98 µs
    wk.forward_prepared(..., ith, nth);   // 12 µs ← 被 wq 掩盖，不再单独付一次 dispatch
    wv.forward_prepared(..., ith, nth);   // 12 µs
});
```

「每层 1 次」需要阶段 barrier + 动态领活（新 pool 原语；`compute_with_chunks` 因为
barrier 与常驻 worker 不兼容而不可用），收益 1.66× → 1.8×，先不做。

#### 风险

1. **位序 / 依赖**（真风险，可控）：组的顺序不变 → 依赖天然满足；必须用现有 Mellum 哨兵
   验证（`tests/mellum_batched_prefill.rs` 逐位 + `RMI_PARITY_TRACE` 的 15466 个检查点）。
2. **DRAM 局部性**（真风险，5–15%，需实测）：现在是 8 个 worker 读**同一矩阵的相邻区间**
   （地址接近，命中同一批 DRAM 行），合并后 8 个 worker 读 8 个相距很远的矩阵 → 行缓冲
   命中率可能下降 → 有效带宽可能从 102 GB/s 掉一点。
3. **「`wo` 单独一次」的浪费**（已知，可接受）：7 个 worker 空转 54 µs × 28 层
   = 0.7 ms/token。
4. **专家阶段的输入准备**：8 个专家的 `group_input` 要同时在位（`batch.rs` 里
   `group_input` 是每专家一份，单 token 路径要新建）。
5. **不是风险**：带宽总量（2.35 GB/token 不变）、计算量、显存。

#### 跨机器稳健性

省掉的 dispatch 次数与机器无关（672 次 ≈ 17.5 ms/token），但**相对收益**取决于 CPU
带宽 —— CPU 越强，纯计算被压得越短，dispatch 开销的占比越高：

| 机器 | CPU 有效带宽 | 纯计算/token | 现在 | 合并后 | 加速 |
|---|---|---|---|---|---|
| aarch64（18 核共享内存） | ~102 GB/s | 23 ms | 44 ms | 26.5 ms | 1.66× |
| Intel Ultra 5（LPDDR5X-8533） | ~110 GB/s | 21 ms | 42 ms | 24.5 ms | 1.7× |
| i7-12700（DDR4-3200 双通道） | ~45 GB/s | 52 ms | 72 ms | 55 ms | 1.3× |
| 老 i7 / 单通道 DDR4 | ~25 GB/s | 94 ms | 114 ms | 97 ms | 1.18× |

（这些带宽是量级估计，不是实测；TODO-038/039 记的那台机器就是共享内存的 aarch64。）

#### 顺带

这条对**所有模型**都成立（任何走 `pool.compute` 做逐投影 dispatch 的 trunk），不只 Mellum。
改的时候顺带把接口设计成能同时喂给 CPU 和设备（照 `record_group_matmul` 的形状：一层一次、
多个权重共用一个 command buffer），以后接设备就只是换个 backend，而不是再改一遍 trunk。

### TODO-053: Nemotron-3-Nano 30B-A3B（`nemotron_h_moe`）—— 已支持，剩按类型家族抽样复验

**状态**：✅ B1–B7 已全部完成——Q5_1 dequant kernel（B1）、arch 与 config 前缀（B2）、
MoE 层分支（B3）、共享专家与平方 ReLU（B4）、oracle 对拍全 52 层 × 6 步逐位一致（B5）、
CLI / HTTP / JEV 入口（B6，2026-11-09）、Vulkan 设备路径（B7，2026-11-10 阶段 3；Q8_0
档设备路径实测逐位，2026-11-11）。完整契约、量化档对照表与逐项记录都在
[NEMOTRON3_NANO_30B_MOE.md](NEMOTRON3_NANO_30B_MOE.md)。

**剩余（低优先）**：按类型家族抽样 3 档做真实权重复验——`Q6_K`（k-quant 家族，设备
可跑）、`Q4_0`（无 k-quant 的 4-bit 家族）、`IQ4_XS`（i-quant 家族，CPU-only 路径）。
其余维持「类型层面的契约」：上游 22 个量化档的 CPU 路径类型全部支持（`Q4_K_M` 与
`Q8_0` 已下载验证），但未下载权重；10 个 i-quant / Q2_K / Q3_K 档没有设备 kernel，
设备 session 建不起来（`auto` 回退 CPU，`--compute vulkan` 严格报错），给它们做
「设备 vs CPU」评估没有意义。

### TODO-054: Dory 的默认提示在 CLI / HTTP / 模型自带模板之间三分；JEV 的「backend=partial」提示对 Dory 是假的

两条都是**文档/提示与实测不符**，实现在数值上没错。
（原标题里的「pixtral 缺强制 BOS」已改为 **F2 强制 BOS 的优先级写反**，
且那条已修；见下面的 F2。）

#### F1：CLI 与 HTTP 给 Dory 的默认提示不一致，三条路径三个形态

`src/models/dory/forward.rs:773` 走 `prompt::legacy::format_chat("dory", …)`，
`src/app/server/api/tools.rs:554` 走 `prompt::build_qwen_chat_prompt(…)`，
两者都把 `dory` 映射到 `ChatTemplate::ChatML`，但渲染结果不同。
用仓库自己的 tokenizer 把 token id 打出来（`What color is the sky?`）：

| 路径 | 结尾形态 | token 数 |
|---|---|---|
| CLI 内置 | `<\|im_start\|>user\n{q}<\|im_end\|>\n<\|im_start\|>assistant\n` | **15** |
| HTTP 内置 | 同上，再补 `<think>\n</think>\n` | **19** |
| 模型自带模板（thinking=false） | 多一个空 system 轮，结尾 `<think></think>` | 22 |
| 模型自带模板（thinking=true） | 同上，结尾 `<think>\n` | 22 |

- CLI 一个 think 标记都没有（模型直接开答）；
- HTTP 补一个**空** think 块（`append_qwen_assistant_prefix(…, enable_thinking=false)`）；
- 模型自带模板多一个**空 system 轮**（`{%- set system_message = "" %}` 让
  `{%- if system_message is defined %}` 成立）并且 think 块**开着**。

贪心解码下三条路径给出不同文本（CLI `The sky appears blue during the day due to
Rayleigh scattering` / HTTP `The sky is most commonly perceived as blue during the
day because the Earth's atmosphere scatteres sunlight (`）。

`tests/cli_http_agreement.rs` 只覆盖 llama / qwen3 / qwen35，**没有 dory**，
所以这条分叉落在覆盖矩阵的盲区里。

修法：让 dory 的两条入口都走模型自带模板，并把 dory 加进
`tests/cli_http_agreement.rs` 锁住两条入口逐 token 一致。

#### F2：强制 BOS 的**优先级**与 llama.cpp 相反 ✅ 已修

> **更正（2026-11-05，做 30B 适配时重新核对）**：本条原先写「pixtral 的强制 BOS
> 没有实现 → Dory 的提示少了 `<s>`」，**结论是错的**。Dory 的 GGUF 明写
> `add_bos_token = false`，参考实现因此**不加** BOS，仓库也没加 —— 两边一致。
> 真正错的是**优先级**。

`src/core/tokenizer/mod.rs` 的注释写着 llama.cpp 对 llama3 预分词家族
（含 **pixtral**）强制 `add_bos = true`，但实现里 `force_add_bos = true` **只给了
falcon-h1**，而且取值写成 `bool_meta(add_bos_token)? || forced_add_bos` ——
**强制值赢过 GGUF**。

llama.cpp 的真实语义（上游 master 与 NVIDIA 的 Dory fork `98a74c97`
**完全一致**，`src/llama-vocab.cpp`）是两步：

1. pre 分词器分支给**默认值**（pixtral / tekken → `add_bos = true`）；
2. GGUF 的 key **覆盖**默认值，连 `false` 都赢：
   ```c
   bool temp = true;
   if (ml.get_key(LLM_KV_TOKENIZER_ADD_BOS, temp, false)) { add_bos = temp; }
   ```

所以应该是「**key 存在则用 key，key 缺失则用 pre 默认值**」。

| GGUF | `pre` | `add_bos_token` | llama.cpp | 修正前仓库 |
|---|---|---|---|---|
| Dory 1.0 4B | `pixtral` | 有，`false` | `false` | `false` ✅ |
| Nemotron-3-Nano 30B | `pixtral` | **无** | `true` | `false` ❌ |
| Nemotron-3-Nano 4B | — | `true` | `true` | `true` ✅ |

**缺 BOS 的是 key 缺失的情况**，30B 正是这一类。顺带修正前对 **falcon-h1**
也是错的（仓库无条件强制 BOS，参考实现里 GGUF 的 `false` 会盖掉它）。

修法（已实施）：`add_bos` 改成「key 存在则用 key，否则用 `forced_add_bos`」；
`forced_add_bos` 补齐 llama.cpp 的表（本仓库支持的 pre 里是 `llama-bpe` /
`pixtral` / `tekken` / `lfm2` / `falcon-h1`；`exaone` / `exaone4` /
`exaone-moe` **不在**表里）。回归测试 `core::tokenizer::tests` 三条：
`pixtral_defaults_to_adding_bos` / `gguf_add_bos_token_overrides_the_pretokenizer_default` /
`gguf_add_bos_token_true_adds_bos_without_a_default`。

Dory 的 oracle 对拍是**喂 token id** 的，验的是前向不是提示构造，所以这条一直
没被抓到；30B 这边同样的道理，提示构造要单独对 token id。

#### F3：`--compute vulkan --jev` 对 Dory 打出的「没有完整设备路径」提示是错的

`src/app/cli/types.rs:240` 的 `vulkan_partial_mode()` 只看**模式**不看 **arch**，
而 `ComputePolicy::has_complete_device_path()` 里 `dory` 是在列的。实测 Dory 的 JEV
前向**整条在设备上**：

| 路径 | prefill | 概率 |
|---|---|---|
| Vulkan（aarch64） | 5886 ms | A 0.9309 / B 0.0166 / C 0.0505 |
| CPU | 24646 ms | A 0.9309 / B 0.0166 / C 0.0505 |

4.2× 的差距 + 完全一致的概率说明整条前向走了设备。服务端
`src/app/server/mod.rs:1911` 打的是同一句，而且注释描述的是 qwen3 那种部分 offload
的模型（「the projections move to the device, the rest stays on the host」），
对 Dory 也不成立。

修法：打印前先问 `has_complete_device_path(arch)`。
---

### TODO-055: Occamy-1.0 的 logits 包络不跨主机（step 6 越界），且 CPU 与设备的 greedy 不同

**现象一：包络越界。** `tools/oracle/pr169` 的 Occamy 默认路径界限是 relative L2 ≤ 0.1，
记录那次（Apple Silicon）逐步为 0.0848/0.0775/0.0390/0.0426/0.0497/0.0707/0.0617/0.0580
—— 自身最大值 0.0848 已经贴边。2026-11-05 在 aarch64 上用同一份权重复验，逐步为
0.0873/0.0491/0.0243/0.0240/0.0336/0.0684/**0.1198**/0.0805 → **step 6 越界**。

不是实现问题：argmax 与 llama.cpp **每一步都一致**（8/8 greedy 逐 token 相同，且与
记录那次相同）；视觉投影 relative L2 0.00070（记录 0.00336）；tokenizer 精确一致。
脆弱点是 step 4 —— llama.cpp 的 top-2 margin 是 0.247，Rust 侧只有 0.039，
host 之间的近似差异足以翻这一步的 token（记录那次就翻了）。

**现象二：CPU 与设备的 greedy 不同。** 同一提示、同为 `--temp 0`：

- CPU：`2 + 2 = 4`
- Vulkan：`2 + 2 equals **4**.`

**在 `origin/main`（不含任何 Occamy 改动）上完全复现** → 不是这条分支引入的。
而 TODO-043 那段写的「occamy-1.0 在 `--temp 0` 下 GPU 与 CPU 逐位一致」用的是**另一份
权重**（含 110 个 F32 二维权重），对这份 Q8_0 checkpoint 不成立。

**待办**：

1. Occamy 的逐步界限改成**两台主机都如实记录**的形式（像 Intern-Decision 那条：把
   界限放宽到能容纳两台主机的实测值，并保留逐 step 数值），不要让界限只对一台主机成立。
2. 现象二要么定位到具体算子（MoE 路由在设备上的专家选择、还是 Q8_0 量化路径），
   要么把 TODO-043 里那句「逐位一致」限定清楚（哪份权重、哪个 host、哪个 arch）。
   设备侧 MoE 路由目前没有逐层专家 id 的对拍手段 —— `src/models/qwen35/trunk/moe.rs`
   有 `ffn_moe_topk` 的 trace，但 llama.cpp 参考侧没有对应 dump。

**关联**：`tools/oracle/pr169/{README.md,verification.json}`（`occamy_reverification_2026_11_05`）、
`src/models/qwen35/trunk/moe.rs`、`src/ops/kernel/qtensor_owned.rs`（设备 Q8_0 路径）。

---

### TODO-056: `--mmproj` + `--interactive` 会静默丢掉交互 REPL，按空提示跑一次

**现象**：

```sh
printf 'What is 2+2?\n' | rust-model-inference --model occamy-Q8_0.gguf \
    --mmproj mmproj-occamy-F16.gguf --interactive
```

不报错、不进 REPL，直接按**空的** `--prompt` 生成一次。模型回
`Hello! How can I help you today?` —— 因为它根本没看到问题。
同样的命令去掉 `--mmproj` 就正常（进 REPL、提示 19 token、答 `2 + 2 = 4`）。

**根因**：`src/main.rs` 的分支顺序。带 `--mmproj`（且 arch 不是 `qwen3vl`）时，
`else if !options.jev && (image.is_some() || … || (explicit_mmproj.is_some() && arch != "qwen3vl"))`
先命中 → 走 `run_multimodal_with_video(..., prompt = options.prompt, ...)`，
而 `--prompt` 为空 → 空提示跑一次就结束。交互分支在后面的 `else` 里，
永远轮不到。`run_interactive_qwen35` 里也把 `mmproj_path` 写死成 `None`，
所以即使进了 REPL 也用不上 mmproj。

**归属**：main 既有（该分支与 Occamy 工作无关），影响所有带 mmproj 的
qwen35 / qwen35moe / edge0 / qwen3vlmoe 家族。ZDTaichu5.0、Intern-Decision、
PhysBrain 的入口矩阵都没有覆盖"交互 + mmproj"这一格，所以没被发现。

**待办**：

1. `--mmproj` 与 `--interactive` 同时出现时，要么明确报错（"交互模式暂不支持 mmproj"），
   要么把 mmproj 传进 `run_interactive_qwen35` 并让它走 qwen35 多模态路径
   （main.rs:1179-1182 的注释本来就写着"qwen35 多模态栈处理无图的情况"，
   只是分支顺序把它挡住了）。
2. 给入口矩阵补一格"交互 + mmproj"，防止再次静默降级。

**关联**：`src/main.rs`（多模态分支与交互分支的顺序）、
`src/app/text/generation.rs::run_interactive_qwen35`（mmproj 写死 None）。

---

### TODO-057: qwen35moe（Occamy-1.0）的设备 decode 比 CPU 慢 4 倍 —— 每 token 320 次专家投影太碎

**实测**（aarch64，Q8_0，8 线程，`--temperature 0`）：

| 场景 | | CPU | Vulkan | 设备/CPU |
| --- | --- | --- | --- | --- |
| 文本 32 token | prefill | 1.088 s | **0.632 s** | **0.58×** ✅ |
| | decode 32 token | 2.034 s | 8.164 s | **4.0×** ❌ |
| | 生成速率 | **15.2 t/s** | 3.8 t/s | 0.25× ❌ |
| | 端到端 | 10.2 tok/s | 3.6 tok/s | 0.35× ❌ |
| 图像（1 token） | 视觉编码 | 0.540 s | 0.539 s | 1.00×（塔的注意力/SSM 仍在 CPU） |
| | prefill | 1.596 s | **1.032 s** | 0.65× ✅ |

**原因**：Occamy 每 token 每层要算 **8 个路由专家**（256 选 8）+ 1 个共享专家，
40 层 → 每 token **360 次极小的投影**（1×2048 → 1×512）。设备的 prefill（一次几百行）
划算 ✓，但 decode 是 360 次单行 dispatch ✗ → 每次的固定开销大于那点算力 ✗。

与仓库里已有的结论同源：TODO-052（每投影一次 pool dispatch，784 次里 26 次的开销大于活）、LFM2.5 的
`min_gpu_matmul_outputs` 阈值（单行投影留在 CPU）。**qwen35moe 的 decode 目前没有
类似的"留在 CPU"策略** —— 单行专家投影在设备上必然亏。

**待办**：

1. 给 qwen35moe 的 decode 加阈值：专家投影的行数太少时留在 CPU（与
   `min_gpu_matmul_outputs` 同思路），至少让 `--compute vulkan` 的 decode 不比 CPU 慢。
2. 或者让专家权重常驻设备、按层合并成一次 dispatch（`BatchedLinearRuntime::with_output_slots`
   + `run_group_gpu` 已支持一个 command buffer 里记多个 matmul ✓）。
3. 视觉塔那边：只有 matmul 上了设备，注意力/SSM 仍在 CPU → 视觉编码在设备上没有收益
   （0.539 s vs 0.540 s）。要么补齐，要么在文档里说明。

**关联**：`src/models/qwen35/trunk/moe.rs`、`src/models/qwen35/vision/mod.rs`、
`src/ops/kernel/vulkan.rs`（`min_gpu_matmul_outputs`）、TODO-052。

---

### TODO-058: qwen35 的 CLIP 视觉塔（mmproj）没有设备路径

**现状**：文本塔在设备上，视觉塔不在。

- 文本塔（`qwen35moe` / Occamy-1.0）：**完整在设备上** —— trace 显示
  `scope=resident_hybrid_decoder backend=Vulkan`（prefill rows=19、decode rows=1），
  733 个权重张量一次性上传 36.35 GB，之后每 token 41 次 submission；但比 CPU 慢 4 倍
  （3.8 vs 15.2 t/s）→ 见 TODO-055。
- 视觉塔（mmproj）：**不上设备** —— `src/models/qwen35/vision/mod.rs` 的
  `matmul_weight_batch` 直接调 `weight.kernel.forward_batched(...)`，而
  `VulkanKernel::forward_batched` 只是转发给内部的 **CPU** kernel。视觉编码两个后端
  耗时完全一样（CPU 0.539 s vs Vulkan 0.540 s），CLI 因此如实报
  `backend=partial mode=mmproj`。
- ZDTaichu5.0 的 RADIO 塔**已经**接上设备（`src/models/zdt/vision/mod.rs:115` 先试
  `weight.try_vulkan_rows(...)`）→ 同一仓库里两种做法并存。

**影响面**：qwen35 的 CLIP 塔是多模型共用的底座（Qwen3.5-0.8B/2B、Qwen3.8-27B、
OvisOCR3、Sage-A1.5、Holo4-27B、LensVLM-9B、Occamy-1.0、Intern-Decision-0.8B、
PhysBrain1.5-2B、Qwen-Drive、LongCat 图像、Omni、BGE-VL-Screenshot，≥12 个），
改它会改掉所有这些模型的图像输出 → **先测量再决定，不要顺手改**。

**F16 是第二道闸**：文本路径明确把 F16 权重排除在设备之外（`gpu_compatible`：
"Vulkan has no matching F16 activation/reduction contract"），而 mmproj 正是 F16 →
要先量出 F16 的设备/CPU 数值差（Occamy 的 oracle 显示同权重 F32 激活 vs F16 激活
的投影差 0.0268）。

**待办**：

1. 先测量：给塔加 `try_vulkan_rows` 分支（不动默认路径），用现成的 llama.cpp dump
   量 F16 的差并定界。
2. 差在界内 → 再谈铺开到各模型（每个模型跑各自的 oracle）；不可接受 → 写下
   "F16 的 CLIP 塔留在 CPU"并关掉本条。
3. 顺带评估收益上限：塔的 attention/softmax/norm 仍在 CPU，只搬 matmul 未必有墙钟收益。

**关联**：`src/models/qwen35/vision/mod.rs`、`src/models/zdt/vision/mod.rs`、
`src/ops/kernel/vulkan.rs`、`src/compute/linear.rs`、TODO-055。

---

### TODO-059: EmbeddingGemma-2 的文本塔 forward 不走线程池 —— 媒体 embedding 比文本慢 50 倍

**实测**（aarch64，8 线程可用，`--embedding`，同一个 1.29 GB GGUF）：

| 路径 | token 数 | 耗时 | user/real |
| --- | --- | --- | --- |
| 文本 | ~5 | **1.0 s** | 0.88 / 1.00 |
| 图像（媒体路径） | ~300（280 个视觉 soft token + 提示） | **50.8 s** | 50.75 / 50.97 |

`user ≈ real` → **只用了一个核** ✗。视觉塔本身不慢
（`tests/embeddinggemma_2_vision_smoke` 48×48 输入 0.76 s ✓）。

**根因**：`src/models/embeddinggemma2/text.rs` 的 `EmbeddingGemmaTextModel` 持有
`pool: Arc<ComputePool>`（第 295 行 ✓、构造于第 306 行 ✓），但 `forward_inputs`
**从不通过它分发** ✓ —— 全文只有字段、构造和一句 "mean pool" 注释引用 `pool` ✓：

- Step 2 的 `per_layer_model_proj`（768 → 24×per_layer_width）**逐 token** 直接调
  `Weight::forward` ✓
- Step 3 的 24 层 transformer ✓
- Step 4 的 `text_embproj` 也是**逐 token** ✓

其它模型（qwen35 / llama / gemma4 …）都在这些循环里用 `ComputePool` ✓。

**待办**：

1. 把 `forward_inputs` 的逐 token 循环按 token 分块交给 `ComputePool::compute` ✓
   （每线程一个输出切片 ✓，照 `src/models/qwen35/trunk/weights.rs` 的
   `matmul_weight_batch_pooled` 写法 ✓）。
2. Step 2 的 `per_layer_model_proj` 是逐 token 独立的 ✓，可以**一次算完整批**
   （`PreparedRows::matmul_group` 或批量 kernel ✓）。
3. 改完复测：媒体路径应从 50 s 降到个位数秒 ✓，并用
   `tests/embeddinggemma_2_multimodal_oracle`（2 passed ✓）确认数值不变 ✓。

**同一处的第三个后果：`--compute vulkan` 下 EG2 一个算子都不上设备。**

实测（`--embedding --prompt 'The sky is blue'`，同一份 GGUF）：

| 后端 | 耗时 | trace |
| --- | --- | --- |
| CPU | 779 ms | — |
| Vulkan | 762 ms | **没有任何 `scope=` 行** ✓ |

CLI 接受 `--compute vulkan` 并如实报 `backend=partial mode=embedding` ✓，但耗时与 CPU 一样 ✓ →
塔的 matmul 直接调 `Weight::forward` ✗，没有 `try_vulkan_rows` 分支 ✗（与 TODO-056 的
qwen35 CLIP 塔同一类缺口 ✓）。权重是 Q8_0/BF16/F32 ✓，设备侧格式都支持 ✓ → 补分支即可 ✓。

**收益预估**：文本路径本来就快（0.8 s ✓ → 收益小 ✓）；媒体路径 50.8 s ✗ → 收益大 ✓，
但要先做上面的批量化 ✓（否则单线程 + 设备提交开销会互相拖累 ✓）。

**关联**：`src/models/embeddinggemma2/text.rs`、`src/models/embeddinggemma2/mod.rs`
（`compute_media_embedding`）、`tests/embeddinggemma_2_vision_smoke.rs`、TODO-056。

---

### TODO-060: EmbeddingGemma-2 的设备路径与 CPU 差 0.033 —— 设备的量化归约契约未对齐

**背景**：EG2 的文本塔已改为持 `Weight`（GGUF 契约 ✓）并批量化 ✓，
`--compute vulkan` 下设备确实接活 ✓（trace：`backend=Vulkan rows=263` ✓、
媒体路径 1083 次 submission ✓）。但实测：

| 对比 | relative L2 | cosine |
| --- | --- | --- |
| CPU kernel vs Vulkan（同一提示、768 维输出） | **0.03330** | 0.99948 |
| CPU kernel vs 上游 HF transformers 参考 | 0.05069 | 0.99873 |

即：**设备的量化 matmul 与 CPU 的量化 matmul 本身就不一致** ✗（0.033 ✓），
这与 TODO-056 的 F16 契约缺口同类 ✓ —— 设备侧的 Q8_0/BF16 归约（shader 的
累加顺序/中间精度 ✓）与 CPU kernel 不同 ✓。

**影响**：嵌入模型对向量敏感 ✓ —— 同一提示在 CPU 与 GPU 上会得到不同的 768 维向量 ✓
（cosine 0.9995 ✓，看起来接近但不是同一个向量 ✓）。仓库的惯例是
「GPU 路径必须与 CPU 一致 ✓」（ZDTaichu5.0 实测差 ~5e-7 ✓），这里差 0.033 ✗。

**2026-11-06 定位结果**（视觉塔注意力上设备 ✓）：

| 对比 | relative L2 | cosine |
| --- | --- | --- |
| 视觉塔：设备 vs CPU | **0.00000858** | 1.00000000 ✓ |
| 媒体最终向量：设备 vs CPU（自动策略） | **0.006413** | 0.99998 |
| 媒体最终向量：设备 vs CPU（`--compute vulkan`） | 0.006492 | 0.99998 |
| 纯文本：设备 vs CPU | 0.008853 | 0.99996 |
| 纯音频：设备 vs CPU | 0.018612 | 0.99983 |

差异**不在注意力** ✓ —— 视觉塔的设备注意力走 F32 的融合双向 kernel ✓
（`AttentionRuntime` ✓、`shaders/glsl/attention_online.comp` ✓，无 mask ✓、
Gemma4 的单位 scale ✓），两边一致到 1e-5 ✓。

差异在 **F16 matmul 的激活精度** ✓：设备的 `shaders/glsl/f16_matmul.comp` ✓
把激活舍入到 F16（`round_f16_rte` ✓，对齐 ggml ✓），而 CPU 的 F16 kernel ✓
（`src/ops/kernel/f16/neon.rs` ✓）只把**权重**转成 F32 ✓、激活保持 F32 ✓。
也就是说**设备侧才是 ggml 语义 ✓，CPU 侧更精确 ✓** —— 差 0.0089 的方向与
TODO-056 的 F16 契约缺口一致 ✓。

**待办**：

1. 决定契约方向 ✓：要么让 CPU 也把激活舍入到 F16 ✓（对齐 ggml/llama.cpp ✓，
   但会拉大 CPU 与上游 HF 的差 ✓ —— 目前 CPU vs HF 是 0.0507 ✓），
   要么让设备保留 F32 激活 ✓（偏离 ggml ✓）。
2. 对齐之前如实记录数值 ✓（已写进 `tools/oracle/pr169/verification.json` ✓
   的 `device_attention_2026_11_06` ✓ 与 README ✓）。
3. 媒体最终向量仍差 0.0064 ✓（自动策略）/ 0.0065 ✓（`--compute vulkan` ✓），
   都在 oracle 的 0.02 界内 ✓；CPU 路径是 0.0147 ✓。

**关联**：`src/ops/kernel/vulkan.rs`（`try_matmul` ✓）、
`shaders/glsl/f16_matmul.comp`、`src/ops/kernel/f16/neon.rs`、
`src/models/embeddinggemma2/text.rs`、`src/models/embeddinggemma2/vision.rs`
（设备注意力 ✓）、`tests/embeddinggemma_2_device_parity.rs` ✓、TODO-056。
---

### TODO-061: Nemotron-3 Nano 4B 的 Mamba-2 SSM 层还没上设备

**状态：阶段 2 已完成（2026-11-09）** —— 42 层全部在设备上，`nemotron_h` 已列入
`ComputePolicy::has_complete_device_path`。下面是这一阶段的记录与踩坑清单；
30B-A3B（`nemotron_h_moe`）仍是另一条 TODO（TODO-053）。

**阶段 1（2026-11-06）**：GPU 跑 17 个 FFN 层 + 4 个 attention 层；CPU 跑 21 个 SSM 层。
  两者以 `scratch.hidden` 为唯一真相源：设备段读它、写它，CPU 段原地读写它。
  42 层按「连续的非 SSM 层段」切分，段与段之间没有多余的同步。

**阶段 2（2026-11-09）**：21 个 Mamba-2 SSM 层也搬上设备，hidden state 不再往返 host。
  SSM 分支在 token 维必须串行（conv history 与 scan state 都是递归的），所以
  一层里的顺序是：先对所有行做一次 RMS norm → in_proj matmul（所有行一起）
  → 逐 token 串行跑 conv/dt/scan/gate/norm → out_proj matmul（所有行一起）
  → 残差加。

- **新增的 5 个 shader**（`shaders/glsl/nemotron_h_ssm_*.comp`）：
  `conv`（depthwise causal conv1d + SiLU）、`dt`（softplus + decay）、
  `scan`（选择性 scan + D skip）、`gate`（`silu(z)*y`）、`norm`（group RMSNorm）。
  语义与 qwen35 的 SSM **不同**（qwen35 是 recurrent 的 beta/alpha + `state_key` 点积），
  没有复用它的 shader。
- **数值**：GPU↔CPU **逐位一致**（`relative_l2` 与 `scaled_max_abs` 全为 0），
  1 / 2 / 3 token 都过。双 1% 门限只是保守下限，实际达到了逐位一致。
- **F32 KV cache 是必需的**（阶段 1 的结论仍然成立）：llama.cpp 默认把 KV cache 存 F16，
  而本模型的参考 harness 用 F32。F16 的舍入在 attention 输出上留下 ~1e-5 的相对误差，
  会被随后的 Mamba-2 递归（`state = state·exp(dt·A) + B·x·dt`，`|A|` 约 10–30）
  放大约 4000 倍，最终在 logits 上变成 1% 的相对误差。为此新增了
  `kv_write_f32` / `attention_scores_f32` / `attention_values_f32_kv` / `softmax_f32`
  四个 shader，只给本模型用，不改动 llama/qwen3/qwen35 共用的 F16 路径。

**踩过的坑（阶段 1）**：

- 阶段 1 最初让设备段和 CPU 段各自持有一份 hidden，结果第 0 层（SSM 层）读到
  未初始化的全 0 —— 设备输出在第一次 `forward_layers` 之前就没人写。
  **教训**：混合路径必须指定唯一真相源，不能各持一份。
- 误差归因用环境变量 `RMI_NH_CPU_ATTENTION` / `RMI_NH_CPU_FFN` 把某一条分支切回 CPU
  才发现问题在 attention 上；逐层 dump CPU/GPU 的 hidden 才定位到第一个分叉层
  （layer 12，第一个 attention 层）。

**踩过的坑（阶段 2）**：

1. **GLSL 的数组输出参数默认是 `in`**。`void f(float x[4], float result[4])` 里的
   `result` 写回会被丢弃，于是 conv/gate 读到未初始化的值 → NaN。必须写
   `out float result[4]`。
2. **GLSL 的字面量**：hex float 没有 `f` 后缀（`0x1p23f` 非法）；double 常量要加
   `lf` 后缀，否则被舍入成 f32；`half` 是保留字。
3. **conv ring 的 tap 顺序**：CPU 的 `mamba2_conv_step` 是**先算后左移** history，
   所以计算时 `history[0]` 是**最老**的 tap（t-3）、`history[2]` 是最新的（t-1）。
   设备的三角环因此是「tap k 读槽 `(t + k) % 3`」，而不是 `(t - 1 - k) % 3`。
   写成后者时第 0 个 token 仍然正确（history 全 0），第 2 个 token 起才发散。
4. **group RMSNorm 必须用 F64 累加**：CPU 的 `sum_sq_f32` 在 aarch64 是 F64 NEON。
   共享的 `rms_norm.comp` 用 F32 补偿和，和 F64 和会差 1–2 ULP，而 `mean_sq` 正好
   落在这个粒度上，于是 `rstd` 差 1 ULP、输出差 2 ULP。新写的
   `nemotron_h_ssm_norm.comp` 用 `double` 累加（F32 平方、F64 求和、最后 narrow
   到 F32），`mean_sq` 就是两边正确舍入后的同一个值。共享 shader 没有改动。
5. **glibc 的 `logf` / `expf` 必须精确复刻**：CPU 的 softplus 用的是 `f32::exp` /
   `f32::ln`（glibc），不是 ggml。仓库的 `host_libm.glsl` 里的 `libm_log` 是
   **x87 80-bit** 版、`libm_exp` 是**正确舍入**版，两者都与 glibc 不同：
   `libm_log` 在 softplus 上差 1 ULP，`libm_exp` 大约每 100 个值差 1 ULP。
   新写的 `nemotron_h_ssm_libm.glsl` 逐条转写 glibc 的
   `sysdeps/ieee754/flt-32/e_logf.c` 与 `e_expf.c`（ARM optimized-routines 同源）。
6. **ggml 的 SiLU 用的是 `ggml_v_expf`**，仓库自带的 `exp_approx_neon.glsl`
   （求值顺序不同）和 `exp_accurate.glsl`（更高阶多项式）都不是它。
   新写的 `shaders/glsl/exp_ggml.glsl` 精确复刻，且复刻了 ggml ARM 路径的一个
   关键细节：**fast / overflow 路径是按 4 个 lane 一组选的**
   （`vaddvq_u32(c) == 0`），所以调用方每个 invocation 处理 4 个通道。
7. **设备端的 Mamba-2 状态跨 prefill 保留**，必须和 host 的 scratch 对齐。
   设备 session 记录 `committed_position`，与 scratch 的 `next_position` 不一致时
   （新的 scratch = 新的序列）清零 conv ring 与 scan state。否则第一个 token 就用
   上了上一条序列的 history。
8. **SSM 的中间量必须按行存放**。一开始 conv/dt/scan/gate 共用一个 scratch，
   于是 3 个 token 时最后一行覆盖了前两行；改成 per-row 后仍然错：scan 读的
   `dt` / `decay` 必须是**当前行**的槽位，误用整块区域的基址时，第 2 个 token 起
   用的是第 0 行的 dt —— **单 token 测不出来**，必须用 2~3 token 的 prefill 才能发现。
9. **`SsmShape` 的尺寸**：conv 宽度是 `d_inner + 2*n_group*d_state`（加法），
   in_proj 宽度是 `2*d_inner + 2*n_group*d_state + dt_rank`（也是加法）。
   写成乘法会让 session 创建直接失败。

**性能（aarch64，8 线程，`Hello` + 32 token）**：设备路径 prefill 97 t/s、生成 21.4 t/s、
端到端 **17.8 tok/s**；纯 CPU prefill 31 t/s、生成 26 t/s、端到端 14.7 tok/s。
阶段 1 的端到端是 5.7 tok/s，所以阶段 2 是 **3.1×**。

- prefill 快 3.1×：多行 matmul 让权重读一次、11 行复用。
- **逐 token 生成仍略慢于 CPU**（21.4 vs 26 t/s）。原因不在本模型的 SSM 路径，
  而在共用的 Q8_0 matmul kernel：单 token 走 `q8_matmul_grouped.comp`，它是
  **标量 int8 乘加**，没有用上 GPU 的 DP4A 单元；多行路径才走
  `q8_matmul_tiled_dp4a_wide`（`dotPacked4x8EXT`）。**两次实测都说明不能简单
  地把单 token 切到 tiled kernel**：
  - 直接用 32 token 的 wide kernel：21.2 → **11.3 t/s**（token tile 固定 32，
    只有 1 个 token 有效时 `accumulator[32]` 几乎全空，而它按运行时下标访问，
    于是落在 local memory 里，每次访问都是一次内存往返）。
  - 另写一个 4 token 的 twin（与 wide kernel 的**每 token 算术完全相同**，
    只是 tile 更窄）：21.2 → **17.5 t/s**，仍然更慢。

  也就是说单 token 的瓶颈**不是** int8 乘加算力，而是权重读取（约 3.6 GB/token：
  SSM 的 in/out 投影 1.66 GB、FFN 1.34 GB、attention 0.15 GB、输出投影 0.41 GB）。
  按 21.4 t/s 折算只有约 80 GB/s，离 aarch64 的峰值还远，**但换 kernel 并不能
  提高它**。要继续优化得先弄清是访存模式（每个 lane 读一行权重，64 条并发
  流）还是别的什么，这是共用 kernel 的问题，影响面覆盖所有模型，本次没有做。

  **后续**：那个问题后来查清了 —— 是**访存模式**，而且解法（transposed
  layout）已经实现，见 [TODO-062](#todo-062-q8_0-权重的-transposed-layout共用-kernel影响所有模型)。
  单 token 生成 21.4 → 28.7 t/s，端到端 17.8 → 24.8 tok/s，逐位一致。

**待办**：

1. 30B-A3B（`nemotron_h_moe`）的 expert routing 还没有设备路径（TODO-053）。
2. 单 token 的 DP4A Q8_0 matmul（共用 kernel，影响所有模型）→ 见
   [TODO-062](#todo-062-q8_0-权重的-transposed-layout共用-kernel影响所有模型)。
3. 更长的上下文与其它显卡/驱动未验收。

**关联**：`src/vulkan/nemotron_h.rs`、`src/models/nemotron_h/trunk/forward.rs`、
`src/vulkan/ops.rs`（`record_kv_write_rows_f32`、`record_attention_scores_f32_rows`、
`record_softmax_f32_rows`、`record_attention_values_f32_kv_rows`）、
`shaders/glsl/nemotron_h_ssm_*.comp`、`shaders/glsl/exp_ggml.glsl`、
`shaders/glsl/nemotron_h_ssm_libm.glsl`、TODO-053（30B 契约）。

---

### TODO-062: Q8_0 权重的 transposed layout（共用 kernel，影响所有模型）

**状态**：⚠️ 已实现（`RMI_GPU_Q8_TRANSPOSED=1` 开启），默认关闭；逐位一致已验证。

**问题**：单 token 生成慢，瓶颈在**权重访存模式**，不在 int8 算力。
`q8_matmul_grouped.comp` 给每个 lane 一行权重，一个 warp 于是读 32 个
「相隔一整行」的扇区，而且每个 4 字节的 payload 读取都跨过 34 字节的 block
边界，要从两个字里拼出来。`src/vulkan/q8_layout.rs` 里记过实测：
这个 kernel 在 aarch64 上只有 **55-59 GB/s**，而设备读同样的字节能到
**138-171 GB/s** —— kernel 输给了自己的访存模式，不是输给内存系统。

**做法**：仓库里早就有 transposed layout（把 64 行 tile 的 block 归到一起，
payload 连续且 4 字节对齐），但**它从来没跑起来过**：

- `Q8_MATMUL_TRANSPOSED` 常量写的是 `OPERATOR_SHADERS.len() - 1`，
  只有在该 shader 排在表尾时才对。Nemotron 的算子追加到表尾之后，
  这个常量就跟着表长走，**「转置 matmul」实际跑的是 SSM norm shader**：
  dispatch 正常完成、什么都不写。已修（见 `8c50c59`），并加了一个
  「每个 pipeline 常量都必须指向它命名的 shader」的测试，防止再漂移。
- transposed layout 原来只服务 one-row kernel。tiled kernel 给每个 lane 一列
  输出、读那一列的 block，64 个 lane 触到的 64 行同样相隔
  `blocks_per_row * 34` 字节 —— **访存模式是一样的**，所以给 tiled kernel
  也补了转置版本（`scripts/transposed-tiled-shader.py` 生成，保证两者同步）。

**结果（aarch64，8 线程）**：

| 场景 | native layout | transposed layout |
|---|---|---|
| one-row Q8_0 matmul（仓库自带 benchmark） | 55-59 GB/s | **138-162 GB/s（2.5-2.7×）** |
| Nemotron-3 Nano 4B 单 token 生成 | 21.4 t/s | **28.7-29.7 t/s（1.35×）** |
| Nemotron-3 Nano 4B 11 token prefill | 113 ms | **66 ms（1.7×）** |
| Nemotron-3 Nano 4B 1211 token prefill | 3.44 s（352 t/s） | **2.66 s（455 t/s，1.29×）** |
| Nemotron-3 Nano 4B 端到端 | 17.8 tok/s | **24.8-25.3 tok/s（1.4×）** |
| Spark-X2.5-1.7B-Q8_0 端到端 | 4.5 tok/s | **11.2 tok/s（2.5×）** |

**逐位一致**：Nemotron-3 Nano 4B 的 1/2/3 token prefill 全部
`relative_l2=0`、`scaled_max_abs=0`；Spark-X2.5 的输出文本逐字相同。
layout 只是把字节换个位置，kernel 的归约顺序不变。

**代价**：每个 Q8_0 权重在上传时转置一次（240 MB 约 26 ms，9 GB/s），
设备显存与 native 相同（只多了最后一个 tile 的 padding）。转置后的 buffer
同时服务 one-row 和 tiled 两条路径，**不需要两份**。

**默认关闭**：地址写错会越界读 device buffer。开启前要在所有架构上跑一遍
硬件测试。

**待办**：

1. 转置 layout 还没有覆盖 `q8_matmul_cooperative_dp4a`（窄投影的单行路径会
   退回 transposed 的 one-row kernel，实测没有变慢，但少了一个并行度来源）。
2. `RUST_GPU_DISPATCH_TRACE` 的日志里还没有区分 native / transposed 的
   一行 matmul，只能靠 shader 名看。
3. 在 AVX2 与标量主机上还没有硬件验收（tiled kernel 的三个变体都补了，
   但只在本机 NEON 路径上跑过）。
4. **开启 flag 后跑全量 `cargo test --lib`，偶尔会多出 1 个失败，而且每次
   多出来的那个都不一样**（8 次全量跑里出现过 vocoder / dit / speaker 各一次）。
   单独跑这三个测试 20 次全过；失败的数值也只差一行（某个 time step 拿到了
   上一个 time step 的结果）。这三个测试都不走转置路径 —— vocoder / speaker
   的 conv 走的是 CPU 的 `forward_batched`，dit 的 GPU session 用的是
   `bind_weight_buffers_mode`（native layout），所以**更像是全量并行跑测试
   时的既有 flake**，而不是转置 layout 引入的。定位方法：并行的 GPU 测试会
   共用一个 Vulkan context，怀疑有别的测试把 host 内存写坏了。下次要查就
   从这里入手（`cargo test --lib -- --test-threads=1` 作为对照）。

**关联**：`src/vulkan/q8_layout.rs`、`src/vulkan/ops.rs`、
`shaders/glsl/q8_matmul_transposed.comp`、
`shaders/glsl/q8_matmul_tiled_dp4a*_transposed.comp`、
`scripts/transposed-tiled-shader.py`、TODO-061（Nemotron 设备路径）。

---

### TODO-063: JEV / 多问题永远拿不到「完整设备路径」（所有架构）

**状态**：⚠️ 已知缺口，未排期。

**现象**：`--compute vulkan` 起 server 时，server 自己会打印

```
compute requested=Vulkan backend=partial mode=jev
note=no complete device path for this mode; only the offloadable parts run on the device
```

（`src/app/server/mod.rs`、`src/app/cli/types.rs`）。JEV 走的
`run_forward_logits_*` 家族**没有**「把整个模型挂到设备」的入口：
Nemotron 的 `attach_gpu` 只有 `src/models/nemotron_h/trunk/runtime.rs`
（文本生成）一个调用点，JEV 的 scoring 路径不走它；其它架构连
`attach_gpu` 都没有，只有按投影的 `ComputePolicy` 卸载，所以 JEV 在它们
身上最多是「部分卸载」，不是完整设备路径。

**实测**（Nemotron-3 Nano 30B-A3B，Q4_K_M，aarch64，8 线程）：CLI `--jev`
两个问题（同一 context）12.3 s；HTTP `/v1/jev/score` 两个问题 42.6 s
（含冷启动）。两者都是纯 CPU 结果 —— 与 CPU 运行逐位相同，因为根本没有
上设备。

**为什么不是 30B 独有的问题**：覆盖矩阵的「JEV × GPU」这一格在所有架构上
都是空的。30B 只是把它显式暴露出来（它的文本路径有完整设备路径，JEV
没有），因此不能靠补 Nemotron 解决。

**方向**（未定）：

1. JEV 的 scoring 本质是「同一 context + N 个 option 的 logits」，设备
   session 已经有 `prefill`，缺的是把 JEV 的 `forward_logits` 接到 session
   上（每个 option 一次短 prefill，或把 option 拼成多行一次算）。
2. 先要定 KV 复用语义：`docs/develop/jev.md` 里 multi-question 目前是
   「每个 question 重新 prefill」，接设备后要不要保留这个语义、还是共享
   KV，会改变数值与耗时。
3. 验收：JEV 的判定概率在设备路径上按误差界验收（不能要求逐位，理由同
   `NEMOTRON3_NANO_30B_MOE.md` §B7.2），并加一条「top-1 option 与 CPU
   相同」的功能门限。

**关联**：`src/app/jev/`、`src/models/*/trunk/forward.rs` 的
`run_forward_logits_*`、`src/models/nemotron_h/trunk/runtime.rs`、
`docs/develop/jev.md`、TODO-053（30B 适配）。

### TODO-064: 模式互斥白名单手写枚举，每加一个共享 flag 都要改多处

**状态**：⚠️ 已知缺口，未排期。

**现象**：`src/app/cli/parse.rs` 的主解析循环统一处理所有 flag
（`--gpu` 在第 389 行就是 `"--gpu" => options.gpu = true`），但每个
「模式」在解析之后另有一段**手写白名单**，用来声明该模式允许哪些
flag、并对带值参数补一次取值校验：

- `options.enhance` → `--enhance | --compute | --gpu | --overwrite | --serve`
- `options.system_one_request` → `--adapter | --system-one-request | --compute | --gpu`
- `options.decision_request` → 自带一段匹配

白名单里出现一个 flag，就等于「这个模式接受它」；其余任何 flag 都会得到
「`<模式>` cannot be combined with `<flag>`」。

**为什么它是缺口**：白名单是**手写枚举**，每有一个新模式接受共享 flag，
就要在三处白名单里各加一次。Vela3 混合 Vulkan 投影
（`codex/vela3-gpu`，`828c8e8a`）把 `--system-one-request` 从「只允许
CPU」放开到可带 `--gpu`，就必须手改 `parse.rs` 的系统模式白名单 ——
那行 `| "--gpu"` 看起来像特例，实际是「共享 flag 列表」的第四份副本。
下次 `--enhance`（RE-USE）也支持 compute 时，已经改过一次了。

**实测**（aarch64，`review/vela3-gpu`）：`--system-one-request --gpu` 走
`parse_cli_options` → `validate_cli_options` 均通过，`compute_policy()`
为 `Auto`；`--gpu` 与 `--compute` 同用仍被拒。

**方向**（未定）：

1. 把「模式 → 允许的 flag 集」集中成一份声明（例如每个模式一个
   `const ALLOWED: &[&str]`），解析后统一查表，取代三处内联白名单。
   这是**纯重构**，不改任何 flag 的接受或拒绝行为。
2. 或者反过来：给 flag 声明它属于哪些模式（`--gpu` 属于 text / enhance /
   decision / system-one / …），模式检查按声明求交集。后者更不容易
   漏，但要求给每个共享 flag 标注归属，改动面更大。
3. 两种方案都要带一条「重构前后行为逐项相同」的回归：把现有
   `src/app/cli/tests.rs` 与 `tests/system_one_cli.rs` 里的模式冲突用例
   扩充成「每个模式 × 每个共享 flag」的矩阵。

**影响面**：`src/app/cli/parse.rs` 是**所有模型**的入口（Qwen3、GLM-4、
llama、Vela3、Idol 等全走它），重构会动到全仓 CLI 解析路径，因此应当
**独立成 PR**，不要混进模型适配的 PR。

**关联**：`src/app/cli/parse.rs`、`src/app/cli/types.rs`、
`src/app/cli/validate.rs`、`tests/system_one_cli.rs`。

### TODO-065: Vela3 的 F32 权重路径每次投影都整份拷贝

**状态**：⚠️ 已知缺口，未排期。

**现象**：`src/models/vela3/tensor.rs` 的 `linear()` 用
`QuantizedTensor::from_bytes` 把权重交给共享投影器。F16 与 BF16 走
`QuantizedTensor::F16`/`BF16` 变体，只借用 mmap 的字节切片；**F32 走
`QuantizedTensor::F32` 变体，会把整份权重 `collect()` 成一份
`Vec<f32>`**（`src/ops/kernel/quantized_tensor.rs` 的 `from_bytes`）。

`linear()` 在 `models/vela3/d3.rs` 的 `project()` 里被调用，而
`project()` 每层每投影调一次（q/k/v/o、MLP gate/up、LoRA 的 A/B 各一次），
于是 F32 变体（`vela3-d3-lite-f32`、`vela3-d3-turbo-f32` 等诊断
产物）会反复拷贝整份权重。

**为什么现在不严重**：发布的 Vela3 权重是 BF16（README 的
`--stream-f32-weights` 一节写明「先检查每个源张量都是 BF16」），走的是
零拷贝路径，所以**生产默认不受影响**；只有 F32 诊断变体会慢。

**方向**（未定）：

1. 让 `QuantizedTensor::F32` 也借用字节切片，kernel 侧按需解释
   little-endian F32；这是共享算子的改动，影响所有用 F32 的模型。
2. 或者在 Vela3 侧把 F32 权重在加载期就转成一次性的 `Vec<f32>` 并
   复用，而不是每次 `linear()` 重新构造。
3. 验收：F32 变体在固定请求上的墙钟不随投影次数线性增长。

**关联**：`src/models/vela3/tensor.rs`、
`src/ops/kernel/quantized_tensor.rs`、`tools/oracle/vela3/README.md`。

### TODO-066: 严格 Vulkan 抓不到「投影静默回落 CPU」

**状态**：⚠️ 已知缺口，未排期。

**现象**：`--compute vulkan` 的语义是「请求真的在设备上」。共享投影器在
若干情况下会**静默**回落 CPU，并且不把设备标记为故障：

- `try_matmul` 在输出元素数小于 `RUST_GPU_MIN_OUTPUTS`（默认 16K）时
  返回 `false`，让投影留在 CPU 池上 —— 这是为避免 fence 延迟拖慢小投影
  的**有意设计**；
- `RUST_GPU_MAX_ROWS` 限流、形状检查、CPU scope 也走同一出口。

这些出口**不调用** `mark_gpu_broken`，所以请求结束时的严格档检查
（`ComputePolicy::context()`，只看「设备是否因队列故障被禁用」）**抓不到
它们**。结果是：一次 `--compute vulkan` 请求可能全部在 CPU 上跑完，
仍然返回 200 和完整响应。

**已经覆盖的场景**：「设备根本不存在」由 `configure_cli()` 的前置检查
挡住 —— 严格档在加载权重之前就报 `requires a usable Vulkan device`；
未编译 `--features vulkan` 时报 `--features vulkan`。Vela3 混合投影
（`codex/vela3-gpu`）实测这两种情形都成立。

**还没覆盖的场景**：设备存在但投影因形状或阈值回落 CPU。仓库现有惯例是
把这种回落当作正常行为（共享投影器对所有模型都这样），因此要不要把它
升级成严格档的错误，**是语义决策而非 bug**：改了就等于承诺「严格档的每
一次投影都在设备上」，而小投影留在 CPU 恰恰是当前的设计意图。

**方向**（未定）：

1. 先定语义：严格档到底承诺「设备可用」还是「每一次投影都在设备上」。
2. 若取后者，需要一个能区分「回落原因」的接口（形状不支持 / 阈值太小 /
   设备故障），并让严格档在出现「非设备故障」的回落时报错或计数上报。
3. 验收：一次全部落回 CPU 的严格档请求，要么被拒，要么在响应里带一个
   可观测的「设备投影数 = 0」标记。

**关联**：`src/ops/kernel/vulkan.rs`（`try_matmul`、
`min_gpu_matmul_outputs`）、`src/compute/mod.rs`（`context`）、
`src/models/vela3/mod.rs`（严格档事后检查）。

---

### TODO-067: PiVo 编码器的投影换用共享的 F32 分块 GEMM（不是融合）

**状态**：📌 待办，未开工。设备路径已经接线且数值等价（帧数与宿主完全相同，
波形在 oracle 门限内），本条只处理它的性能，**不阻塞任何交付**。

**问题**：`shaders/glsl/pivo_linear.comp` 是手写的朴素投影——每个 lane 负责一个
输出列，权重按列跨步读，既不合并读也不分块。结果是**边际成本比宿主还贵**：

| 量 | 设备 | 宿主 |
| --- | --- | --- |
| 编码器边际成本 | 0.126 ms/token | 0.071 ms/token |
| 编码器固定开销（≈70 次 dispatch × ≈50 µs） | ≈3.7 ms | — |
| 编码器，34 token | 8.01 ms | 2.63 ms |
| 编码器，100 token | 14.54 ms | 8.54 ms |
| 编码器，200 token | 17.10 ms | 14.37 ms |
| 端到端 HTTP 请求，短（21 token） | 78.3 ms | 75.8 ms |
| 端到端 HTTP 请求，长（≈130 token） | 630.7 ms | 561.7 ms |

（Intel Core Ultra 5 125H + Intel(R) Graphics (MTL) 核显，`release-fast`，
HTTP 请求中位数。）

**做法**：直接复用共享层已有的 F32 分块路径，**不写新 shader**：

- `src/vulkan/ops.rs` 的 `record_transposed_float_matmul_rows` 走
  `shaders/glsl/float_matmul_tiled.comp`（64 lane、32 token 暂存、权重 vec4
  合并读、FMA），小形状走 `float_matmul_small`。
- **权重布局**：该 kernel 按 `[in][out]` 取权重，PiVo 的 GGUF 是 `[out][in]` →
  用 `src/vulkan/dense.rs` 已有的 `upload_f32_transposed` /
  `transpose_f32_weights` 在加载时转置一次（编码器权重约 1.6 MB，一次性）。
- **权重来源**：该 kernel 从 binding（w0/w1/w2）读权重，不走 arena → PiVo 运行时
  改成「权重常驻独立 buffer」，而不是现在的「常驻 arena」。
- **一次 dispatch 三组权重**：q/k/v（以及 gate/up）合并成一次，kernel 支持 3 个
  slot，每个 attention block 因此少 2 次 dispatch。
- **激活值布局**：输入 `[token][in]`、输出 `[token][out]` 行主序 = PiVo 原生布局，
  不需要转置。
- **没有 bias**：PiVo 编码器的 43 个张量里没有 bias，与该 kernel 一致（它不支持
  bias）。
- `n_in % 2 == 0` 的要求对 PiVo 成立（96 / 192 / 24）。

**不要做的**：**不要做块级融合**（把整个 attention block 融成一个 shader）。
收益分析：融合只砍固定开销（≈70 → ≈7 次 dispatch，省 ≈3.3 ms），边际成本不变，
融合后 34 token 约 4.7 ms、200 token 约 25 ms，**仍慢于宿主**。而融合会让三件事
变复杂：barrier 纪律（barrier 前不得有 lane 相关退出——本条实现时已经踩过一次
死锁）、寄存器与共享内存压力（分数暂存 512 float + Kahan 归约 128 float）、
以及校验结构（要从算子级改成块级）。仓库里融合真正划算的先例是 Z-Image 的 VAE
（1477 次提交 → 39 次，GPU 时间 −56%），PiVo 的 ≈70 次不在那个量级；而且 92% 的
时间在声码器（FARGAN 严格顺序递归，上不了设备），编码器最多占 3–12%，融合最多
换来 2–8%。

**校验要求**：

1. **帧数必须与宿主完全相同**（否则音频长度会变）。现在的余量：编码器相对误差
   4.61e-7、对数时长误差 4.77e-7，最短 oracle 用例的帧数舍入余量是 0.0024 帧
   （约 650× 余量）。
2. `src/vulkan/ops.rs` 里现有的算子级测试要覆盖新核（对 PiVo 的形状做宿主参照）；
   `src/models/pivo/tests.rs` 的帧数对比继续作为端到端门禁。
3. 若换核后 `pivo_linear.comp` 退役，先把它降级为**设备侧参照实现**，用「新核 ==
   旧核，逐位相同」验收，而不是只对宿主。

**待办**：

1. 换核后实测编码器的边际成本与 dispatch 次数，再决定要不要块级融合。
2. 权重常驻位置从 arena 改到独立 buffer 后，`PivoRuntime` 的权重指纹、容量核算
   与 `upload_weight` 要跟着改。
3. 在 AVX2 与标量主机、以及独显上各验收一次（本机只有核显）。
4. 若将来 PiVo 支持批量合成，固定开销与边际成本会被摊薄，届时 GPU 才可能真正
   赢过宿主——比融合更值得等。

**关联**：`shaders/glsl/float_matmul_tiled.comp`、`shaders/glsl/float_matmul_small.comp`、
`src/vulkan/ops.rs`（`record_transposed_float_matmul_rows`、`FLOAT_MATMUL_TILED`、
`PivoRuntime::linear`）、`src/vulkan/dense.rs`（`upload_f32_transposed`、
`transpose_f32_weights`）、`src/models/pivo/gpu.rs`、`shaders/glsl/pivo_linear.comp`、
[PiVo validation](../../tools/oracle/pivo/validation.md)。

### TODO-069: MTP / DSpark / n-gram 投机解码 —— qwen35 的 MTP、DSpark 与 n-gram 已实现，EAGLE3 未做

**状态**：⚠️ 调研完成（2026-11-11）。**`qwen35` 稠密档的 MTP 起草头与验证/接受
回路已实现**（见 TODO-072 与 PR #241：草稿逐 token 与 pinned llama.cpp 一致，
接受率 0.58 vs 0.53，投机解码与 greedy 逐 token 相同）；**DSpark / DFlash
sidecar 的起草侧与验证侧也已端到端实现**（`src/models/dflash/`，目标为
`qwen35` 系：per-layer 输入 hidden 捕获 → DFlash 编码器注入 → 噪声块去噪 →
Markov/conf 头 → 批量验证 + KV/状态回滚；**入口 CLI `--dspark-model` 与 HTTP
`--serve --dspark-model` 都已接线**（`RuntimeOptions::with_dspark_model`，
哨兵 `cli_and_http_agree_on_dspark_speculative_decode` 断言 CLI/HTTP 逐字节
一致且与 greedy 输出相同）），与 llama.cpp `b96806d` 的 `draft-dspark` 对同一
prompt 逐 token 一致，复现记录见
[tools/oracle/dspark/README.md](../../tools/oracle/dspark/README.md)；**n-gram
self-speculative（上游 `ngram-simple`）也已实现**（`src/core/ngram.rs` 移植
+ `HybridSession::{ngram_speculative_step, ngram_generate}` 复用同一验证/回滚
机制，无需第二份权重；CLI `--ngram-draft N` / `--ngram-lookup N` 与 HTTP
`--serve --ngram-draft` 经 `RuntimeOptions::with_ngram_draft/with_ngram_lookup`
接线，哨兵 `cli_and_http_agree_on_ngram_speculative_decode` 断言 CLI/HTTP
一致且与 greedy 相同；oracle 侧 `--spec-type ngram-simple` 的输出与其 plain
greedy byte 级一致——同一契约两侧成立，fresh 跨引擎对拍被 chat 模板差异
挡住，与 DSpark 轮同因）；EAGLE3 仍未实现（standalone draft 决定不做，见下）。

**背景**：[SPECULATIVE_DECODING.md](SPECULATIVE_DECODING.md) 是 2026-09-23 的设计稿。
按上游 llama.cpp 的 pin `b96806d` 重新核对了一遍契约，并盘点了真实模型里
到底哪些带这些头（unsloth 名下 354 个 `-GGUF` 仓库，逐个读 GGUF 头部）。

**上游契约**（`common/speculative.cpp::common_speculative_types_from_gguf` 自动识别）：

| 类型 | 识别条件 | draft 是什么 |
|---|---|---|
| `draft-mtp` | arch 不是 `dflash`，且有 `blk.{block_count-1}.nextn.eh_proj.weight` | 主模型里多出来的 MTP block |
| `draft-dflash` | arch 是 `dflash`，且没有 `markov_w1.weight` | 独立的 DFlash 小模型 |
| `draft-dspark` | arch 是 `dflash`，且有 `markov_w1.weight` | DFlash + Markov 头（+ 可选 Confidence 头） |
| `draft-eagle3` | 独立的小 EAGLE3 模型 | |

- **MTP 计算**（`src/models/deepseek4.cpp::graph_mtp`）：
  `concat(RMSNorm(embed(next_token), nextn.enorm), RMSNorm(h_last, nextn.hnorm))`
  → `eh_proj` → 一层普通 block → `shared_head_head(shared_head_norm(out))`。
  张量：`blk.%d.nextn.{eh_proj,embed_tokens,enorm,hnorm,shared_head_head,shared_head_norm}`；
  metadata：`<arch>.nextn_predict_layers`。
- **DSpark 计算**（`src/models/dflash.cpp`）：DFlash 小模型读主模型若干层的
  hidden state（`target_layers`），内部 conv + selector；**Markov 头**按「上一个
  已提交的 token」给每个位置的 draft logits 加偏置
  （`logits[i] = base[i] + markov_w2 @ markov_w1[prev(i)]`，链式），
  **Confidence 头**预测每个位置的接受概率
  （`sigmoid(conf_proj · [embd[i]; markov_w1[prev(i)]] + b)`）。
- **sidecar 约定**：上游下载器认 draft repo 里的 `MTP/`、`dflash/`、`dspark/`、
  `eagle3/` 子目录（`dspark` 优先于 `dflash`）。

**真实模型**（unsloth 的发布就是按这个约定）：

| 模型 | sidecar | arch / 备注 |
|---|---|---|
| Gemma-4 全系（E2B/E4B/12B/26B-A4B/31B，含 qat 变体） | `MTP/` 0.06–0.95 GB | `gemma4-assistant`，`nextn_predict_layers = 4`，形状不同（`nextn.pre_projection` / `post_projection`） |
| Qwen3.8-27B | `MTP/` 1.37 GB (Q4_0) | `qwen35`，`nextn_predict_layers = 1` |
| Qwen3.8-Flash-Next | `MTP/` 2.79–7.77 GB | `qwen4exp`，带 mHC 头（`nextn.hc_head_{up,down,norm}`） |
| DeepSeek-V4-Flash-0731 / V4.1-Flash-1017 | `dspark/` 8.5–11.3 GB | `dflash`，3 个 DSV4 block，128 专家 top-3，`target_layers=[37,38,39]`，`block_size=5` |
| Mistral-Large-4-1T-A52B | `dspark/` 4.21 GB | `dflash` |
| MiniMax-M3.1-Flash | 根目录 `dspark-*.gguf` 2.09 GB | `dflash` |

**本仓库现状（2026-11-18，MTP 与 DSpark 均已落地）**：

- **MTP（PR #241 / #250）**：`qwen35` 稠密档与 Vulkan 设备路径都已实现，见
  TODO-072。Qwen3.8-27B 主 GGUF 自带的 `blk.64.nextn.*` 头由
  `src/models/qwen35/trunk/mtp.rs` 使用，不再静默忽略。
- **DSpark（PR #252）**：`src/models/dflash/` 起草 + `qwen35` trunk 层循环顶部
  的 hidden 捕获 + `HybridSession` 的验证/回滚。CLI `--dspark-model` 与 HTTP
  `--serve --dspark-model` 都走 `RuntimeOptions::with_dspark_model`：
  `build_text_runtime` 对非 `qwen35` target 显式报错（flag 不静默丢弃），
  设备路径显式报错，temp>0 / 带图请求在请求时显式报错。
- **n-gram（本次）**：`src/core/ngram.rs` 是上游
  `common_ngram_simple_draft` 的逐行移植（含其全部边界：位置 0 是 not-found
  哨兵、倒序搜索跳过当前出现、`copy_max < size_n` 不起草、
  `cur_len <= n+m+1` 不起草），单测钉住每条边界；session 侧
  `ngram_speculative_step` / `ngram_generate` 与 dspark step 共享同一
  greedy 验证与 KV/recurrent 回滚。入口门控与 dspark 同构：非 `qwen35`
  显式报错、Vulkan 显式报错、与 `--mtp-draft`/`--dspark-model` 叠加显式
  报错、`ngram_draft < ngram_lookup` 显式报错（上游在该区间永不起草）；
  Qwen3.5-2B 实测 `--ngram-draft 8 --ngram-lookup 4`：48 token 47 轮
  （1/8 草稿接受——接受率本质上是工作负载相关的，重复文本才起草）。
- **standalone draft（draft-simple）决定不做**：所有可跑 target 已有更强的
  draft 轴（27B 有 MTP、2B 有 DSpark、qwen35 全系有 n-gram）；standalone 是
  谱系最弱形态（无 target 内部状态 + draft 自身 autoregressive），且缺跨 arch
  的 KV 回滚基建。
- **设备路径的 DSpark 仍不可用**：Vulkan forward 无 per-layer 捕获
  （`dspark_gpu_guard` 显式拒绝），是 TODO-072 之外的下一个设备侧活。

**关联**：[SPECULATIVE_DECODING.md](SPECULATIVE_DECODING.md)、
[tools/oracle/dspark/README.md](../../tools/oracle/dspark/README.md)、
`src/models/dflash/`、`src/models/qwen35/trunk/session.rs`、
`src/app/text/runtime.rs`、TODO-072。

### TODO-070: Qwen3.8-27B 的 Vulkan 路径在 Q8_0 档下不可用（FFN 太宽）

**状态**：✅ 已修（#238）。

**现象**：`--compute vulkan` 跑 `Qwen3.8-27B-Q8_0.gguf` 直接报

```
Inference error: Unsupported shape: invalid Vulkan matmul input width/stride
```

**原因**：Vulkan 的 Q8_0 matmul 有 `n_in > 16_384` 的上限
（`src/vulkan/ops.rs` 的 `matmul_rows_push` 与 `src/vulkan.rs` 的
`matmul_q8_0`）。Qwen3.8-27B 的 MLP down 投影是 `blk.N.ffn_down.weight` =
17408×5120 的 Q8_0，即 `n_in = 17408`，超限。Holo4-27B 同样的 FFN 宽度没问题，
因为它是 Q4_K/Q6_K，那个上限只针对 Q8_0。

**根因是过期的守卫**：16384 来自旧 kernel —— 它把整行输入一次性放进 shared
memory（`shared uint s_input[4096]`，512 个 block）。#166 起 Q8_0 的 kernel
都按 64 个 block 分块暂存，shared memory 不再是约束，但 host 侧两处守卫没删。
所以不需要改 shader，删掉守卫即可。

**结果**（#238，本机 aarch64）：设备 Q8_0 kernel 对 CPU 归约（宽度 17408）逐位
一致；16 步 greedy 的 15 个 token（含 EOS）CPU 与设备逐个相同；logits
（248320 维）逐位一致（relative L2 = 0）。详见
[UNIFIED_COMPUTE_VALIDATION.md](UNIFIED_COMPUTE_VALIDATION.md) 的
「Q8_0 宽投影与 Qwen3.8-27B 的设备路径」一节。

**仍未覆盖**：batch=64 的长行预填、图像路径（视觉塔仍在 CPU）、长上下文、
AMD/Apple 驱动，以及 27B 以外的 Q8_0 宽投影模型。

**关联**：#238、`src/vulkan/ops.rs`、`src/vulkan.rs`、
`docs/develop/SUPPORTED_MODELS.md` 的 Qwen3.8 行。

### TODO-071: qwen35 家族的量化档与 llama.cpp 差约 1 ULP（默认 Q8_0 kernel 的累加顺序）

**状态**：⚠️ 已定位，按误差界接受，未改默认 kernel。

**现象**：量化档（Q8_0 / Q4_K_M）与 pinned llama.cpp 不再逐位一致。以
NeoHorse-1-4B 为例：GDN checkpoint 的相对 L2 是 `8.9e-8`…`5.4e-7`（门槛
`1e-5`），greedy token 逐个相同，但深层栈的漂移到 `layer_output-31`
`5.7e-2`…`1.1e-1`；Qwen3.8 的 Q8_0 档是同一形状（GDN `2.2e-7`…`3.5e-7`、
第 63 层 `7.4e-2`、logits `5.4e-2`）。

**根因**：仓库默认的 Q8_0 kernel（`dotprod_nrc4` / `neon_nrc1`）把每个 block
的整数点积先归约成标量再进 F32 乘加；llama.cpp 的
`ggml_vec_dot_q8_0_q8_0` 对每对 block 保留 4 路累加器、最后才归约。
per-lane 的分组方式会改变最后一位。`src/ops/kernel/q8_0/ggml.rs` 记着这件事，
并提供了复刻 llama.cpp 算术的入口（`forward_prepared_ggml`），Nemotron 主干在
用它。

**为什么不改**：qwen35 家族保持默认契约（那个 ggml-exact 入口比默认 kernel
慢，而 Q8_0 档的验收已经是「GDN checkpoint ≤ 1e-5 + greedy token 逐个相同 +
漂移报告」）。这是**全局**取舍，不是逐模型的：若要恢复量化档的逐位一致，
应当给 qwen35 家族一个统一开关（例如 `RMI_QWEN35_Q8_GGML_EXACT=1`），
并重新测性能与各模型的数值。

**顺带**：MoE 路径（`qwen35moe`）默认走参考归约，所以它与标量 Oracle 逐位
一致（Sage-A1.5 的 1,100 个检查点、MiniCPM-V 的 437 个检查点都是 0 误差）；
受影响的只有用默认 Q8_0 SIMD kernel 的稠密路径。

**关联**：`src/ops/kernel/q8_0/ggml.rs`、`src/ops/kernel/q8_0/dispatch.rs`、
`tools/converter/neohorse/README.md`、`docs/develop/SUPPORTED_MODELS.md`。

### TODO-072: Qwen3.8-27B 的 MTP 起草头已与 llama.cpp 对平，CPU 与设备路径上的投机解码都比 greedy 快

**状态**：✅ 起草头 + 验证/接受回路已实现，与 pinned llama.cpp **逐轮同草稿**；
接受率 0.54 vs 0.53。✅ CPU 上也有收益了：批量 Q8_0 kernel（`sdot`）+ 不用重算
投影的回滚 + 按批做词表投影之后，2 个草稿的投机解码比 greedy 快约 27%
（4.28 s vs 5.86 s，只算解码段是 2.81 s vs 4.39 s，1.56 倍）。✅ **设备路径也能
跑投机解码了**：目标前向在 Vulkan 上，设备把每一行的 hidden 与词表投影一次性交
回来（设备一次读一遍词表矩阵，而不是每行一遍）。✅ **起草头也搬上设备了**：它
是主栈之后的一个稠密块，设备侧有它自己的权重、自己的 KV cache 槽位、自己的
norm 与 FFN，catch-up 是一批行。真模型上投机解码产出的 token 与 greedy **逐个
相同**（Q8_0 档接受率 0.54，与 CPU 一致）。设备回滚也改成了「只重放状态更新」
（chunk 捕获每个循环层的投影，回滚时用它重放 conv + SSM scan），代价从一次完整
前向降到一次状态更新重放（Q4_0 档实测 1.54 s → 0.02 s；设备上的 137 ms →
20～26 ms）。⚠ **Q8_0 与 Q4_0 档上设备的投机解码都比 greedy 快约 1.55 倍**
（2 个草稿；Q8_0 的 decode 段 8.1 s vs 12.6 s，Q4_0 是 7.8 s vs 12.1 s）。⚠ 前提
是**多行前向在设备上足够便宜**：Q8_0 与 Q4_0 都有 tiled kernel（权重跨行共享），
所以验证那一遍的 3 行 ≈ 1 行；其余权重档位（Q4_1 / Q5_0 / Q4_K / Q6_K …）还没有，
那些档位的多行前向要付 N 倍权重流量，投机解码在设备上没有收益。

**做了什么**：

- `src/models/qwen35/trunk/mtp.rs`：装载并执行 GGUF 自带的 MTP 头（`blk.64.*`）。
  图与 llama.cpp `b96806d` 的 `graph_mtp` 一致：
  `concat(RMSNorm(embed(t_p), nextn.enorm), RMSNorm(h_{p-1}, nextn.hnorm))`
  `→ nextn.eh_proj → 一个注意力块（自己的权重、自己的 KV cache）→ FFN`
  `→ shared_head_norm → shared_head_head`。
  `nextn.embed_tokens` / `shared_head_head` / `shared_head_norm` 缺省时回落到主
  模型的 embedding / 输出权重 / 输出 norm（与上游相同）。
- 配对规则：MTP 块在位置 `p` 上吃 `(embed(t_p), h_{p-1})`，预测 `t_{p+1}`；`h`
  必须是**输出 norm 之后**的 hidden state（llama.cpp 的 `h_nextn`）。这条规则踩
  过三个坑，三个都定位到了：
  1. `last_hidden_state()` 原来返回的是最后一个 chunk 的**第一**行而不是最后一行
     → 预填之后的第一轮起草全是噪声（接受率 0.36）；
  2. 起草上下文的 catch-up 必须把 hidden 整体**右移一位**（位置 `p` 的 token 配
     `h_{p-1}`），llama.cpp 用 `pending_h` 跨 batch 携带最后一行；配错时起草头的
     KV 内容整体错一位；
  3. **起草块自己的注意力前面少了一次 norm**：`graph_mtp` 把 `eh_proj` 的输出
     同时当作残差（`inpSA`）和注意力的输入，但注意力的输入是它**用
     `nextn.attn_norm` 归一化过的副本**。本仓库的 dense-attention 辅助函数按
     约定只吃「已经归一化」的输入（主栈就是这么调的），起草头直接把 `eh_proj`
     的输出喂了进去 → Q/K/V 全错 → 注意力读的不是 llama.cpp 那个键空间。
     定位方式：把 llama.cpp 的 `graph_mtp` 每个节点的数值 dump 出来（给它的
     scheduler 装一个 eval callback），与本仓库逐节点对拍——`eh_proj` 之前
     **逐位相同**，`attn_norm` 之后开始分叉。补上这次 norm 之后，`attn_norm` /
     `Qcur` / `Kcur` / `Vcur` / 门控后的注意力输出 / FFN / 头 norm **逐节点都对
     上了**（只差最后几位浮点），接受率从 0.18 回到 0.54。
- 验证/接受回路：一次目标前向跑 `[bonus, draft_0..draft_{n-1}]`，用每一行的
  logits 与草稿逐个比对，取最长匹配前缀。被拒的草稿必须回滚：
  - KV：把被拒位置的 key 行清零（本仓库的自由槽位靠「第一行全零」扫描出来）；
  - 循环状态：GDN / shortconv 的状态逐 token 携带，只回滚 KV 不够 → 快照 +
    重放已接受的 token。llama.cpp 用 context checkpoint + replay，语义一致。
    重放只做**状态更新**：前向捕获状态更新要吃的那几行（投影后的 q/k/v 与过了
    激活函数的 beta/alpha），回滚不重算投影，也不算这一层的输出。
- CLI：`--mtp-draft N`（N=每轮起草的 token 数，0=关闭）。只支持 greedy
  （`--temp 0`）、只支持纯文本 prompt（视觉路径的 prompt 行是投影后的 embedding，
  拿不到 token id）。HTTP（`--serve --mtp-draft N`）走同一条路：温度为 0 的请求
  走投机解码，其余请求照旧逐 token 解码。

**设备路径上的投机解码**（PR #250）：

- 设备一次前向只交回**最后一行**的 hidden 与 logits，投机解码需要的是**每一行**
  的：验证要知道目标在每个草稿位置会产出什么（每一行的词表投影），起草头要把
  草稿接在上一行的 hidden 后面（每一行的 hidden），catch-up 要用每个已提交位置
  的 hidden 填起草头的 KV cache。于是设备侧加了两条采集路径：
  - `normed` 区（本来就是 `rows × n_embd`）：整块归一化并把每一行读回来；
  - `logits_rows` 区（`rows × vocab`，只在**有 MTP 头**的 checkpoint 上分配）：
    整批行一次词表投影。投影的输入按 `n_embd` 跨步、输出按 `vocab` 跨步，现有
    的批量 kernel（`q8_matmul_tiled_dp4a.comp` 与 `q8_matmul_transposed.comp`）
    已经是这个写法，所以**不用改 shader**。
  - 两条采集是分开的：catch-up 只要 hidden（预填时不需要为一整批 prompt 行去读
    1.3 GB 的词表矩阵），验证才要 logits。
- **起草头现在也跑在设备上**：起草头是主栈之后的一个稠密块，有自己的注意力、
  自己的 FFN、自己的 norm 和**自己的 KV cache**。设备侧现在也录它：
  - 它的 KV cache 是设备 arena 里主栈之后的那一层槽位（`kv_k` / `kv_v` 多一
    层），起草头的注意力读写的是自己的槽位，不是目标的 key。
  - 它的 `eh_proj` 吃的是「归一化后的 embedding ‖ 归一化后的 hidden state」。
    设备没有 concat shader，所以**两个 norm 各写进同一个 buffer 的两半**，
    投影再按 `2 × n_embd` 的行宽读这一块——不用新增 shader，也不用把权重拆成
    两份上传。
  - 起草头的词表投影复用设备上**已有的** `output.weight`（checkpoint 没有给
    起草头自己的 `shared_head_head` 时，upstream 也是这么做的），所以不用再传
    一份 1.3 GB。
  - **catch-up 是一批行**：验证那一批已提交的位置在设备上是一遍起草头（多行的
    起草头注意力按行做因果遮罩），不是每行一遍。
  - 起草头的权重如果设备放不下（不支持的格式），起草头**留在 CPU 上**，目标仍然
    跑在设备上。
- **设备上的回滚只重放状态更新**：CPU 路径回滚时只重放循环状态（捕获投影后重放
  状态更新，见 PR #248）。设备侧现在也走这条路：chunk 在跑的时候把**每个循环层
  的投影**（投影后的 q/k/v、gate、beta/alpha）留在设备上的一份 capture 区里，回滚
  时把保存的循环状态上传回去，再用捕获的投影**只重放 conv + SSM scan**。
  - 捕获的是**投影**而不是层的输入：投影要读循环层的权重（48 层 × 65 MB ≈ 3.1 GB），
    是状态更新里贵的那一半。捕获输入的话，回滚得把投影重算一遍——实测那一半就占
    回滚的 122 ms（总共 137 ms）。捕获投影之后，回滚只跑状态更新本身。
  - 循环层的状态更新只吃它自己的投影，所以重放它就够了，不必重跑整个模型（更不必
    重跑 FFN 和输出投影）。代价从「一次完整前向」降到「一次状态更新重放」：Q4_0 档
    实测 **1.54 s → 0.02 s**（设备上的 137 ms → **20～26 ms**）。
  - capture 区只给**有 MTP 头**的 checkpoint 分配（每个循环层四个投影各一行槽位），
    Qwen3.8-27B 是 48 × 64 × (10240 + 6144 + 48 + 48) × 4 B ≈ 198 MB。
- **真模型验证**（Qwen3.8-27B Q4_0，`unsloth/Qwen3.8-27B-GGUF`，设备路径）：
  投机解码产出的 token 与 greedy **逐个相同**（16 个生成 token），接受率 0.50。
  同一模型在 CPU 上也逐个相同，接受率 0.60（Q4_0 的设备 kernel 与 CPU kernel
  舍入不同，所以两条路径的 greedy token 之间可能有一两个不同，这是量化档的
  既有现象，与投机解码无关）。
- **tiny 模型的单测**（`qwen35_mtp_speculative_matches_greedy_on_the_cpu` /
  `..._on_the_device`）：一个带 MTP 头、带循环层的极小模型，目标在 CPU 与设备
  两条路径上跑，投机解码的 token 必须与 greedy 逐个相同。fixture 的起草头被
  故意做成「提出一个目标永远不会选的 token」，于是**每一轮都被拒**，回滚与重放
  的分支必然被执行到。

**实测**（Qwen3.8-27B Q8_0，19 token 的 chat prompt，16 个生成 token，
`--mtp-draft 4`）：

- 起草头与 llama.cpp **逐轮同草稿**：每一轮的 4 个草稿都与 pinned llama.cpp
  `b96806d` 的 `llama-server --spec-type draft-mtp --spec-draft-n-max 4` 记的
  候选完全相同（第一轮 `[119721, 103722, 97889, 102768]`，接受计数 4/0/1/2/3
  也逐轮相同）。
- 接受率：本仓库 13/24 = 0.54，llama.cpp 10/19 = 0.53（同一个模型、同一个
  prompt、同一个 n_max）。
- 投机解码产出的 token 与 greedy **逐个相同**——这是回滚做对了的判据。

**CPU 上的性能：比 greedy 快 1.3 倍，还没有倍数收益**。这里有五条独立的账：

1. **批量前向曾经不比逐 token 前向便宜**：Q8_0 的 per-row kernel 每个激活行都要
   把权重矩阵重读一遍，而权重就是流量的主体，于是 `R` 行的批量前向要付 `R` 倍
   的权重流量。新的批量 kernel（`src/ops/kernel/q8_0/neon.rs`）把权重行留在
   寄存器里同时服务一组激活行，**逐位**复刻 per-row kernel 的算术（496/496 个
   投影逐位相同，最大差 0），预填从 5.23 s 降到 2.16 s，整段 greedy 生成从
   10.28 s 降到 6.59 s。两行以上的批次都走这条路径——三行（两个草稿的验证）原来
   因为一个 `n_rows >= 4` 的门槛掉回 per-row 路径，白白多付了一遍权重流量。
2. **批量 kernel 自己曾经是算力瓶颈**：它的整数点积走的是 widening 累加
   （`vmull_s8` + 成对加法，每条指令 4 个 int8 MAC），而 per-row 的 dispatch 在
   这台机器上选的是 `sdot`（每条指令 16 个 MAC）。也就是说，批量路径跑在它替换掉
   的那条路径的 1/4 指令速率上。改用同一个 `sdot` 之后（整数点积精确，分组方式
   不影响结果，批量路径仍然与 per-row 路径逐位相同），预填 2.16 s → 1.47 s。
   ⚠ 这里的判据是**运行时探测**：`is_aarch64_feature_detected!("dotprod")` 在这台
   机器上是 true，尽管 `/proc/cpuinfo` 里没有 `dotprod`（内核不列 ARMv9 隐含的
   扩展），所以不能照抄 cpuinfo 那一行。
3. **前向是访存瓶颈**：这个 checkpoint 的 Q8_0 权重一共 29 GB，一遍前向要把它们
   全读一遍；greedy 每个 token 一遍前向（≈290 ms/token，≈100 GB/s 的有效带宽），
   这已经是这台机器的内存带宽上限。所以批量前向再快也快不过「把权重读一遍」，
   投机解码的收益只能来自**少跑几遍前向**。
4. **回滚曾经把 5.6 GB 权重又读了一遍**：被拒绝的草稿会推进循环状态，回滚要把
   状态退回已接受的位置。原来的回滚拿捕获的层输入重跑整个循环层——包括投影
   （每层 86 MB）和输出投影（每层 31 MB），而后者的结果根本没人用。现在前向
   捕获**状态更新真正要吃的**那几行（投影后的 q/k/v，以及过了激活函数的
   beta/alpha），回滚只重放状态更新本身：一轮的回滚从 89 ms 降到 23 ms。
5. **一轮的固定开销**：起草头的 `N` 次前向（每次都带一次 1.3 GB 的词表投影：
   `output.weight` 是 248320 × 5120）、catch-up 的头前向、验证那一遍的词表行。
   现在只读用得到的词表行（catch-up 的行没人采样，验证在第一个不匹配处就停），
   一轮从 1.35 s 降到 0.85 s，回滚再降 0.07 s/轮。
6. **验证侧的词表投影曾经一行一次**：验证要知道目标在每个草稿位置会产出什么，
   所以要读那一批行的 logits，而每一行的投影都要把 1.3 GB 的 `output.weight`
   重读一遍。现在目标前向按批投影那一批行（批量 kernel 与 per-row kernel 逐位
   相同），一遍只读一次词表矩阵；catch-up 也不再跑草稿头的 FFN（它只写 KV
   cache，FFN 到不了 cache）。一轮省 0.03 s，整段生成省 5%。

合起来的实测（19 token chat prompt，生成 16 token，8 线程；这组数字叠在批量
Q8_0 kernel、`sdot`、便宜回滚与按批词表投影之上）。这台机器的负载会漂，所以对比
一律用**交替跑**的两个二进制，取中位数：

| 路径 | 整段生成 |
| --- | --- |
| greedy | 5.86 s（prefill 1.47 s + decode 4.39 s） |
| 投机，2 个草稿 | **4.28 s**（中位数；采样 4.24–4.29 s；接受率 0.83） |
| 投机，2 个草稿（没有按批词表投影） | 4.53 s（中位数；采样 4.50–4.57 s） |
| 投机，2 个草稿（连便宜回滚都没有） | 5.45 s（中位数；采样 5.38–6.35 s） |
| 投机，3 个草稿 | 4.96 s / 5.01 s（接受率 0.72） |
| 投机，4 个草稿 | 5.39 s / 5.51 s（接受率 0.54） |

即：**2 个草稿时比 greedy 快约 27%**（只算解码段是 2.81 s vs 4.39 s，1.56 倍）；
便宜回滚省掉 14%（5.45 s → 4.67 s），按批词表投影再省 5%（4.53 s → 4.28 s）。
草稿再多反而更慢（一轮的行数是线性涨的，收益是对数涨的，而且每个草稿都要
付一次 1.3 GB 的词表投影）。
llama.cpp 在同一台机器上没有这个收益（它的 MTP decode 262 ms/token，就是单
token decode 的量级）——差别在于它没有把 Q8_0 的权重读取在批次内共享。
一轮 34 GB 的流量里，目标前向占 29 GB（跑不掉），草稿头占 3.4 GB，catch-up
占 0.5 GB，词表投影占 3.9 GB（目标侧那遍已经按批做，一遍只读一次 1.3 GB 的
`output.weight`；草稿头的每个草稿仍要单独读一遍）。还能省的有一处：草稿头的
词表投影（每个草稿一遍 1.3 GB，而草稿之间是串行的——下一个草稿要用上一个草稿的
token，所以没法合并）。

**设备路径上的实测**（18 token 的 chat prompt，32 个生成 token，
`--compute vulkan`，8 线程；交替跑取中位数。这台机器的负载会漂，所以对比
一律交替跑）：

Qwen3.8-27B **Q8_0**（29 GB 权重）：

| 草稿数 | 轮数 | 接受率 | 目标前向次数 | prefill | decode | 合计 |
| --- | --- | --- | --- | --- | --- | --- |
| 0（greedy） | 31 | – | 31 | 0.7 s | 12.6 s | 13.3 s |
| 2 | 14 | 64.3% | 14 | 0.7 s | **8.1 s** | **8.8 s** |

Qwen3.8-27B **Q4_0**（16 GB 权重）：

| 草稿数 | 轮数 | 接受率 | 目标前向次数 | prefill | decode | 合计 |
| --- | --- | --- | --- | --- | --- | --- |
| 0（greedy） | 31 | – | 31 | 1.8 s | 12.1 s | 13.9 s |
| 2 | 13 | 73.1% | 13 | 1.8 s | **7.8 s** | **9.6 s** |
| 4 | 12 | 41.7% | 12 | 1.8 s | 9.3 s | 11.2 s |

即：两个权重档上，**2 个草稿的设备投机解码都比 greedy 快约 1.55 倍**
（decode 段）。4 个草稿接受率掉到 41.7%，多出来的轮次把收益吃掉一些。

（这台机器的背景负载会漂，所以上表的绝对值随时间变；对比一律交替跑，
取同一轮里的相对值。同一台机器上，起草头搬上设备之前是 12.3 s vs 9.1 s
（1.35 倍），起草头搬上设备之后是 12.3 s vs 8.4 s（1.45 倍）——起草头每步的
24 ms 词表投影是那 0.7 s 差的主要来源；再把回滚从 137 ms 降到 20 ms 之后是
12.1 s vs 7.8 s（1.55 倍）。）

**一轮里各部分的实测**（Q4_0 档，2 个草稿，13 轮，decode 7.8 s = 600 ms/轮）：

| 部分 | 每轮 | 占比 |
| --- | --- | --- |
| 目标前向（3 行） | 530 ms | 88% |
| 起草头 2 步（设备上） | 30 ms | 5% |
| catch-up（3 行） | 3 ms | 0.5% |
| 回滚（9/13 轮有拒绝） | 20～26 ms | 3.5%（按摊薄算） |

所以现在**目标前向本身是瓶颈**：它读 17 GB 权重却只跑出 32 GB/s，而起草头那次
词表投影读 1 GB 能跑出 100 GB/s——前向不是带宽瓶颈，是每层十来个 dispatch 的
串行开销（64 层 × ~0.8 ms/层）。要再往上得动 dispatch 的粒度（把 norm 融进
matmul 的 prologue、把残差 add 融进下一个 matmul……），那是个大重构。

另外，起草头在 CPU 上并不便宜：实测每步 24 ms、catch-up 每行 1.5 ms（Q4_0 档，
8 线程）——每一步都是一遍 1.3 GB 的词表投影。**起草头搬上设备之后**这一步几乎
不要钱了：Q4_0 档的设备投机解码从 9.1 s 降到 8.4 s（decode 段）。

**多行前向在设备上有多贵，取决于权重格式**——这条决定了投机解码在设备上有没有
收益：

- **Q8_0**：多行 matmul 走 tiled kernel（`q8_matmul_tiled_dp4a.comp`），**权重在
  行之间共享**，多一行几乎不花钱。验证那一遍是 3 行 ≈ 1 行的代价，于是「少跑几遍
  前向」的收益完整保留。
- **Q4_0**：原来只有逐行 GEMV（`q4_0_matmul.comp`：每个 `(输出行, token 行)`
  组合自己把权重行重读一遍，而且 32 个 MAC 全是标量），于是多行前向要付 N 倍权重
  流量，投机解码在设备上**没有收益**（打平）。
  现在补上了 Q4_0 的 tiled kernel（`q4_0_matmul_tiled_dp4a.comp`）：权重按
  `(输出列, k-block)` 读一次、在 tile 的多行之间共享，4-bit 解包后用打包点积。
  算术与逐行 kernel **逐位相同**（整数和精确，浮点部分保持 `(和 × 权重 scale) ×
  激活 scale`、逐块相加的顺序），新测试用同一份权重比 1～5 行逐位对拍。
  实测设备前向：1 行 0.39 s（保持逐行 kernel，tile 的 staging 在单行时是纯开销）、
  3 行 0.81 s → **0.51 s**、8 行 2.07 s → **0.82 s**。于是 Q4_0 上的设备投机
  解码从「打平」变成 **1.35 倍**。

**已知边界**：

- 起草头的**词表投影每个草稿仍要单独读一遍词表矩阵**（Q4_0 档的 `output.weight`
  是 Q6_K，约 1.04 GB；Q8_0 档是 1.27 GB）：草稿之间是串行的（下一个草稿要用上一个
  草稿的 token），没法合并成一批。这是 MTP 的算法本身如此，不是实现的取舍；它在
  设备上只占一轮的 5%，不值得动。
- 设备的**多行前向**对 Q8_0 与 Q4_0 都便宜了（两者都有 tiled kernel，权重跨行
  共享）；但 Q4_1 / Q5_0 / Q5_1 / Q4_K / Q5_K / Q6_K 档还没有，那些档位的多行
  前向仍然要付 N 倍权重流量，投机解码在设备上没有收益。
- 设备的**单行 decode 比 CPU 慢**：Q4_0 档实测 404 ms/token vs CPU 的 187 ms
  （8 线程，17 GB 权重），Q8_0 档 406 ms vs 274 ms。设备的优势在**多行**（3 行
  175 ms/行、8 行 103 ms/行，都比 CPU 快，tiled kernel 让多行共享一次权重读取），
  所以设备路径的价值在批量（预填、投机解码的验证），不在单行解码。
- 剩下的三块都是**大活**，按性价比排在后面：
  1. **per-row GEMV 的访存效率**：设备单行只跑到 44 GB/s（Q4_0）/ 74 GB/s
     （Q8_0），而 CPU 是 91 / 106 GB/s，设备自己的大词表投影能跑 104 GB/s——
     模型的 matmul 只跑到了设备自身能力的三四成。现在的 per-row kernel 是
     「一个 lane 负责一个输出列」，于是 64 个 lane 各读不同的权重行，访存不合并；
     tiled kernel 用共享内存 staging 解决了它，但只在多行时启用。要让单行也快，
     得换个读法，同时**保住每列的累加顺序**（否则破坏与 CPU 的逐位一致）。
  2. **算子融合**：把 norm 融进 matmul 的 prologue、把残差 add 融进下一个
     norm。前向里每层十来个 dispatch，64 层就是六百多个；但融合要么改归约顺序
     （norm 的 Kahan 修正树形归约是这个仓库与 CPU 逐位一致的基础），要么只融
     不改数值的那部分（add+norm），收益减半。风险与收益不对称，放最后。
  3. **其它权重档位的 tiled kernel**（Q4_1 / Q5_0 / Q5_1 / Q4_K / Q5_K /
     Q6_K）：纯覆盖面，加速不了手上的 checkpoint。
- CPU 上前向是访存瓶颈，投机解码的收益只有 1.4 倍（见上面的实测）；要多草稿
  才有倍数收益的话，得先让每个草稿的词表投影不再单独读一遍 1.3 GB。
- 设备路径的投机解码要求整批草稿（`1 + N` 行）装进一个 chunk（默认 64 行），
  否则报错：每一行的 hidden 与 logits 都得留在前向之后。
- 只有稠密 `qwen35` 装载 MTP 头；`qwen35moe` 的 nextn 走专家，是另一张图。
- TODO-069（MTP / DSpark 契约）在 PR #235 的分支上，
  `docs/develop/SPECULATIVE_DECODING.md` 由那个 PR 拥有，本条不重复它的内容。

**复现**：

```sh
# 本仓库：接受率 + 与 greedy 逐 token 相同
RMI_QWEN35_MODEL=<Qwen3.8-27B-Q8_0.gguf> \
RMI_QWEN35_PROMPT_TOKENS=248045,846,198,95826,110827,98419,108829,113672,119721,1710,248046,198,248045,74455,198,248068,271,248069,271 \
RMI_QWEN35_MTP_STEPS=16 RMI_QWEN35_MTP_DRAFTS=4 \
cargo test --locked --profile release-fast --features parity-trace --lib qwen35_mtp -- --ignored --nocapture

# 同一件事跑在设备上：加 --features vulkan 与 RMI_QWEN35_MTP_DEVICE=1
RMI_QWEN35_MODEL=<Qwen3.8-27B-Q4_0.gguf> \
RMI_QWEN35_PROMPT_TOKENS=248045,846,198,95826,110827,98419,108829,113672,119721,1710,248046,198,248045,74455,198,248068,271,248069,271 \
RMI_QWEN35_MTP_STEPS=16 RMI_QWEN35_MTP_DRAFTS=4 RMI_QWEN35_MTP_DEVICE=1 \
cargo test --locked --profile release-fast --features "parity-trace,vulkan" \
  --lib qwen35_mtp_speculative -- --ignored --nocapture

# 设备上的性能：交替跑 greedy 与 MTP，看 [pipeline-timing] 与 [mtp] 两行
target/release-fast/rust-model-inference --model <gguf> --prompt "..." \
  --max-tokens 32 --temp 0 --compute vulkan --threads 8 --mtp-draft 0
target/release-fast/rust-model-inference --model <gguf> --prompt "..." \
  --max-tokens 32 --temp 0 --compute vulkan --threads 8 --mtp-draft 2

# llama.cpp 参照：同一模型、同一 prompt、同一 n_max
llama-server -m <gguf> -md <gguf> --spec-type draft-mtp --spec-draft-n-max 4 -lv 5
# 然后用同样的 token 数组打 /completion，日志里的 "draft candidate" 就是草稿
```

**关联**：`src/models/qwen35/trunk/mtp.rs`、`src/models/qwen35/trunk/session.rs`、
`src/models/qwen35/trunk/forward.rs`、`src/models/qwen35/trunk/weights.rs`、
`src/models/qwen35/trunk/scratch.rs`、`src/ops/kernel/q8_0/neon.rs`、
`src/vulkan/qwen35.rs`、`src/compute/state.rs`、
`docs/develop/SUPPORTED_MODELS.md`。

### TODO-073: ERNIE-Image 的设备路径：K-quant 慢，Q8_0 快 2.8 倍

**状态**：✅ 已解决（2026-11-12）。K-quant 仍是已知缺口。

**结论**：ERNIE-Image 的设备投影**能不能赢，取决于权重量化格式和投影形状两件事**，
不是取决于模型。

| DiT 权重 | 规格 | CPU denoise | 设备 denoise | 结果 |
|---|---|---|---|---|
| Q4_K_M | 256×256 / 8 步 | 251.8 s | 559.5 s | **慢 2.2 倍** |
| Q8_0 | 256×256 / 8 步 | 219.2 s | **78.4 s** | **快 2.8 倍** |

Q8_0 的结果：与 CPU 的 PNG 差异 PSNR **69.78 dB**（max_abs 1/255），同一 seed
连跑两次逐位相同。

**两件事各占一半**：

**一、权重量化格式**。设备只有 Q8_0 有向量化 kernel —— `Q8_MATMUL_GROUPED_DP4A`
与 `Q8_MATMUL_TILED_DP4A` 吃设备 int8 dot-product ✓✓。K-quant 家族只有一个标量
kernel：它的**块缩放 int32 累加塞不进 int8 dot product** ✓✓（Q8_0 的量化把乘积
压在 int8 内 ✓✓），要吃 DP4A 得写一个「块缩放 + int32 提升」的 int8 dot
kernel —— 那是 TODO-032 量过的完整 shader 项目 ✓✓。

⇒ **模型层按权重量化格式选后端** ✓✓：Q8_0 的 DiT 走设备 ✓✓，K-quant 的 DiT 留在
CPU 范围 ✓✓。

**二、投影形状**（这一半是本条自己踩出来的）。ERNIE 的 DiT 原本**逐 token** 调用
投影 —— `text_proj`、`final_linear`、qkv、mlp 都是一行一次 ✓✓。一行一次在设备上
是「每行一次权重上传 + 一次 fence」✗✗，fence 的开销淹没计算 ✗✗：

| 路径 | Q8_0 DiT denoise |
|---|---|
| 单行投影允许上设备 | 624.6 s |
| 单行投影强制 CPU、整批上设备 | **78.4 s** |

⇒ ⇒ **单行投影强制 CPU** ✓✓（`disable_gpu_matmul_for_scope` ✓✓，底下的多线程
CPU kernel 不变 ✓✓），**整批投影走设备** ✓✓ ✓✓。

**仍然剩下的缺口**：K-quant 的 DiT 在设备上没有优势 ✓✓ ⇒ 要么给 K-quant 写设备
kernel（TODO-032 ✓✓），要么这类模型就该留在 CPU ✓✓。

**踩过的两个坑，记下来**：

- **共享内核自带设备 dispatch，模型层拦不住** ✗✗：`src/ops/kernel/q4_k.rs` 与
  `q8_0/parallel.rs` 在 `ops::gpu_matmul_active()` 为真时自己就走设备 ✗✗。
  模型层的门控只覆盖模型自己发起的投影 ✗✗。⇒ **单行路径要显式关掉** ✓✓。
- **64×64 / 1 步这类小规格测不出设备路径的真实表现** ✗✗：20 个 token 时，
  权重上传和 252 次 dispatch 的固定成本占主导 ✗✗，设备必然输 ✗✗。⇒ 设备路径的
  测量要用真实尺寸（256×256 / 8 步 ✓✓）✓✓。

**关联**：`src/models/diffusion/ernie_image/{gpu.rs,dit.rs,mod.rs}`、
`src/models/diffusion/ernie_image/text.rs`（文本编码器留在 CPU ✓✓）、
`src/ops/kernel/q4_k.rs`、`src/ops/kernel/q8_0/parallel.rs`、
[TODO-032](#todo-032-q4_kq5_kq6_k-在-gpu-上没有-dp4a-kernel8b-q4_k_m-比-cpu-慢-176-倍)。

### TODO-074: qwen3 视觉塔按字节长度猜权重格式 —— BF16 会被当成 F16 读

**状态**：⚠️ 已知缺口，未排期。2026-11-11 定位。

**现象**：`src/models/qwen3/vision/mod.rs` 的 `decode_linear_weight` **不看张量自己声明的
格式**，而是看字节长度：

- 每元素 4 字节 → F32
- 每元素 2 字节 → **F16**
- `n_out * (n_in / 32) * 34` → Q8_0

⇒ **F16 与 BF16 都是每元素 2 字节** ✓✓ —— **长度分不出两者** ✓✓，而两者解出来的值
不同：1.0 的 BF16 位模式按 F16 读出来是 **1.875** ✓✓。

**为什么现在没炸**：目前走 qwen3 视觉塔的模型，视觉权重都是 F32
（Mage-Flow 的 `mmproj-mage-flow-F32.gguf` 全部 314 个张量都是 F32）——F32 是
4 字节，猜得对 ✓✓。**一旦有一个 BF16 的视觉塔进这个路径，就是静默的错值** ✓✓：
不报错、不崩，只是每个权重都读错 ✓✓。

**对照**：qwen35 的视觉塔**已经是对的** ✓✓ —— 它的 `load_source_weight` 按
`ggml_type` 分派（F32 / F16 / BF16 / Q8_0 ✓✓），而不是靠长度猜 ✓✓ ✓✓。
qwen3 塔是这一族里剩下的那个例外 ✓✓。

**为什么没有直接改**：qwen3 塔的 `VisionLayer` 每个权重字段是 `&[u8]` ✓✓，
**同一个层里的不同权重可以是不同格式** ✓✓（比如 BF16 的主干 + F32 的 norm ✓✓），
所以要修就得给每个字段各记一个 dtype ✓✓ —— 那是 `VisionLayer` 的结构变更 ✓✓，
会牵动所有构造它的调用方（zdt ✓✓、wevisone_vl ✓✓、mage_flow ✓✓）✓✓。

**方向**（未定）：

1. 在 `core::tensor` 加一个按声明格式取矩阵的共享入口
   （`load_f32_matrix` 已经在 `4c77690f` 里加好了 ✓✓ —— 它就是给这一类用的 ✓✓）✓✓，
   然后把 qwen3 塔的权重路径换成它 ✓✓ —— 顺带把「按格式取」这件事收进 core ✓✓，
   符合 §6 的单一事实源原则 ✓✓。
2. 或者把 qwen3 塔的 `VisionLayer` 权重字段从 `&[u8]` 改成解码后的值 ✓✓，
   跟刚修完的 bias 一样 ✓✓ —— 代价是所有构造方都要跟 ✓✓。

**验收**：给 qwen3 塔一个 BF16 的视觉塔，权重解码结果与按声明格式解码的结果
逐位相同 ✓✓。

**关联**：`src/models/qwen3/vision/mod.rs`（`decode_linear_weight`）、
`src/models/qwen35/vision/mod.rs`（`load_source_weight`，dtype-aware 的参考 ✓✓）、
`src/core/tensor.rs`（`load_f32_matrix` ✓✓）。

### TODO-075: Qwen-Image-2.1 的基础版设备路径 —— 现在能跑了，但与 CPU 的差异和那个偶发的首轮差异都还没定位

**状态**：⚠️ 部分完成（2026-11-16）。设备路径能跑、能出图、能复现；与 CPU 的数值差异
和那个偶发的首轮差异待定位。

**发布权重是混合格式**：Q8_0 的 DiT GGUF 一共 265 个张量 = 192 个 Q8_0 + 73 个 BF16。

- **Q8_0**：每个 block 的 `attn.to_q/to_k/to_v/to_out.0`、`img_mlp.gate_up`、
  `img_mlp.out`。
- **BF16**：`img_in`、`modulation.1`、`norm_out.linear`、`proj_out`、
  `time_text_embed.timestep_embedder.linear_1/2`、`txt_in.in_layer`、
  `txt_in.out_layer`。
- **全部 65 个一维 norm 向量也是 BF16**：每个 block 的 `attn.norm_q.weight` 与
  `attn.norm_k.weight`，加上 `txt_in.text_norm.weight`。

⇒ 加载器原来把「二维矩阵」和「一维向量」分成两条路，**两条都硬编码了格式** ✓✓：
一维向量按 F32 读（长度对不上 ⇒ `Invalid ... weight length` ✓✓），
`norm_out.linear` 按 F32 读（切片越界 panic ✓✓）。

⇒ 现在**格式由张量自己声明**（`tensor_info().ggml_type` ✓✓），调用点不再假设格式 ✓✓。

**分派器自己写错过一次** ✓✓：把 14 个 `q8_linear` 调用点换成按 dtype 分派的 `linear()`
时，分派器的默认分支写成了 `self.linear(...)` ⇒ **无限自递归** ✓✓。release 构建把尾递归
优化成了循环 ⇒ **模型永远出不来**（单核 100% 死转，30 分钟无输出 ✓✓）。
`rustc` 的 `unconditional_recursion` lint 报了，但淹没在 148 条 warning 里 ✓✓。

**实测**（CFG 4 / seed 42 / 16 线程 ✓✓）：

| 规格 | 路径 | 时间 | 复现性 |
|---|---|---|---|
| 128×128 / 2 步 | CPU | 22 s | 4/4 逐位相同 ✓✓ |
| 128×128 / 2 步 | 设备（`--gpu`） | 15 s | 4/4 逐位相同 ✓✓ |
| 256×256 / 4 步 | CPU | 106.8 s | 2/2 逐位相同 ✓✓ |
| 256×256 / 4 步 | 设备（`--gpu`） | 58.3 s | 合并 main 前 **4 次里 3 个结果不同** ✗✗；合并后 4/4 逐位相同 ✓✓ |

⇒ 设备路径**快 1.83 倍**（256×256 / 4 步 ✓✓）✓✓。

**CPU 与设备的 PNG 差异**（PSNR ✓✓）：

| 规格 | CPU vs 设备 | 设备自己 vs 设备自己 |
|---|---|---|
| 128×128 / 2 步 | 36.30 dB（max_abs 57 ✓✓） | 逐位相同 ✓✓ |
| 256×256 / 4 步 | 42.96 dB（max_abs 70 ✓✓） | 逐位相同 ✓✓（合并 main 后 ✓✓） |

⇒ 合并 main 前，设备自己跟自己之间的差异（44.42 dB ✓✓）和它跟 CPU 的差异
（41.51 / 42.58 dB ✓✓）是同一个量级 ⇒ 那时的抖动**不是**「设备算得不准」，而是**结果
不确定** ✓✓；合并 main 后这个抖动没有了 ✓✓。

**仍未定位的两件事**：

1. ~~**设备路径在 256×256 下不可复现**~~ ✅ **已解决**（2026-11-17）✓✓。

   原来的现象：4 次跑出 3 个不同的 PNG ✓✓，而 128×128 / 2 步是 4/4 逐位相同 ✓✓、
   CPU 在 256×256 下也是 2/2 逐位相同 ✓✓ ⇒ 与规模相关，且只在设备上出现 ✓✓。

   **根因是 main 的 Q8_0 内核改动，不是这个模型** ✓✓。合并 main（含 **#244「在批内共享
   Q8_0 权重读」** ✓✓ 与 **#247「Q8_0 内核用 sdot」** ✓✓）之后，同样的
   256×256 / 4 步跑 4 次**全部逐位相同** ✓✓，CPU 的结果也没有变 ✓✓（md5 `5cf7aae0`
   ✓✓）⇒ 之前那 3/4 不同确实是随机的 ✓✓。

   ⇒ 教训：**设备路径的可复现性要在最新的 main 上测** ✓✓ —— Q8_0 内核换实现之后，
   旧分支上的设备结果就不可比了 ✓✓。

   **当时记下的两条线索**（都留档 ✓✓）：Qwen-Image 用到而 ERNIE 不用的是 **BF16 投影**
   （`GpuMatmulMode::RoundedBf16` ✓✓）与 **VAE 的设备卷积矩阵** ✓✓；共享的 `ops::nn`
   路径里有一道 `verify` 防护 ✓✓，注释写明**这个设备曾经返回过前一次调用的数字** ✓✓
   （陈旧缓冲 / 未同步的上传 ✓✓），而 Qwen-Image 的 `gpu.run` 没有这道防护 ✗✗。
   ⇒ 上面的根因就落在这两条里（**具体是哪一条没有单独验证** ✓✓）。

2. **偶发的首轮差异** ⚠️ ⚠️：换了二进制之后的**头一两轮**跑出来的 PNG 与之后的不一样
   （`261e954d` vs `d3b846d3` ✓✓），之后就 4/4 逐位相同 ✓✓；加满 CPU 负载也复现不出来 ✓✓。
   单线程、2 / 4 / 16 线程的结果都与「稳定值」一致 ⇒ **与线程数无关** ✓✓。
   **用固定的 `--qwen-context-file` 绕过文本编码器后两次逐位相同** ✓✓ ⇒ 差异来自
   **Qwen3-VL 的文本编码器** ✓✓，不是 DiT、不是 VAE、不是采样 ✓✓。

   ⇒ Qwen3-VL 的文本编码器在多线程下偶发不可复现 ⇒ 这是**共享代码**的问题
   （`src/models/qwen3/trunk/` ✓✓），**不是本 PR 引入的**
   （`git diff origin/main -- src/models/qwen3/trunk/` 为空 ✓✓）。

3. **设备与 CPU 的数值契约不同** ⚠️：设备路径的 Q8_0 投影把激活量化到 int8 ✓✓，
   BF16 投影用 `RoundedBf16` 模式在设备上做 f32 累加 ✓✓，而 CPU 的 BF16 路径是 f64
   累加 ✓✓ ⇒ **数值契约不同** ✓✓。哪一条占主导没有量 ✓✓。

**方向**（未定）：

1. 给 Qwen3-VL 的文本编码器补一条「同 seed 连跑逐位相同」的门禁 ✓✓，再定位那个偶发差异
   （先怀疑线程池的归约顺序与 KV cache 的写入顺序 ✓✓）。
2. 设备路径的数值对齐按仓库既有做法：设备对齐用 `RMI_SCALAR=1` ✓✓，或者把 CPU 的
   BF16 累加也降到 f32 ✓✓。

**验收**：Qwen-Image-2.1 基础版在 256×256 / 4 步下，CPU 连跑 3 次逐位相同 ✓✓、
**设备连跑 3 次逐位相同** ✓✓，且 PSNR 与同规格的 Turbo 水平相当 ✓✓。

**关联**：`src/models/diffusion/qwen_image_2_1/dit.rs`（`linear()` / `vector()` 的 dtype
分派 ✓✓）、`src/models/diffusion/qwen_image_2_1/mod.rs`（`PrefixedSource` ✓✓、类型集合
校验 ✓✓）、`src/models/qwen3/trunk/`（文本编码器的复现性，待定位 ✓✓）、
[TODO-073](#todo-073-ernie-image-的设备路径k-quant-慢q8_0-快-28-倍)（同一套「按权重量化
格式选后端」的结论 ✓✓）。

### TODO-076: Z-Image 的 F16 DiT 档在设备路径上第二步全 NaN —— 根因是设备端没有 CPU 那份 1/128 的预缩放

**状态**：✅ **本轮的缺口已修**（2026-11-18）。F16 档的 DiT 现在能上设备；
**共享算子还差一份支持**（设备端要能对 FFN 中间量做缩放）。

**现象**（`models/z-image-gguf-mine/z-image-turbo-f16.gguf` ✓✓，128×128 / seed 42 ✓✓）：

| 规格 | CPU | 设备（`--gpu`） |
|---|---|---|
| 128×128 / 1 步 | 13.0 s，出图 ✓✓ | 3.9 s，出图，与 CPU 差 **58.29 dB** ✓✓ |
| 128×128 / 2 步 | 22.0 s，出图 ✓✓ | ✗✗ `Non-finite Z-Image predicted flow: len=4096, nan=4096` |
| 64×64 / 1 步 | ✓✓ | ✓✓ |
| 64×64 / 2 步 | ✓✓ | ✗✗ `len=1024, nan=1024` |

⇒ **不是分辨率相关，是步数相关** ✓✓：`RUST_GPU_DIAG=1` 把分界点钉死了 ✓✓ ——
第 1 步 `sigma=1.000` 正常，第 2 步 `sigma=0.003` 全 NaN ✓✓。

**怎么找到的** ✓✓（探针临时加在 `z_image/dit.rs` 上，测完已删 ✓✓）：

1. 第 1 步 CPU 与设备逐项**同量级** ✓✓：残差流 43.483765 两边一样 ✓✓、
   qkv 368.420319 对 368.420532 ✓✓、out_proj 112102.141 对 112102.156 ✓✓；
   **但 velocity 差 0.4%** ✓✓（4.892777 对 4.872861 ✓✓）。
2. 逐 block 对比 out_proj：**refiner（block 0/1）完全一致（0.00%）** ✓✓，
   **主栈第一层（block 2）就差 -7.35%** ✓✓ ⇒ 分叉在主栈 ✓✓。
3. `RUST_GPU_ATTENTION=0`（注意力回到 CPU）时 **qkv 与 out_proj 都一致** ✓✓
   ⇒ 分叉在**设备的 attention** ✓✓；attention 输出逐 block 对比：refiner 差 0.02% ✓✓，
   主栈第一层差 **-1.13%** ✓✓。
4. 第 2 步的逐层序列：**layer 4（主栈第一层）入口 tokens 有限（726.5）** ✓✓，
   调制有限 ✓✓、attention 输出有限（131.9）✓✓、out_proj 有限（30730）✓✓，
   **而 `ffn_w2` 是 `max_abs=279398, inf=53760`** ✗✗ ⇒ **inf 产生在 FFN 的下投影** ✓✓。

**根因** ✓✓：CPU 的 FFN 下投影写的是 `linear_into_scaled(.., 1.0 / 128.0)` ✓✓ ——
激活**先缩小再舍入到 F16** ✓✓，而**舍入是有范围的** ✓✓；
设备端 `record_ffn` 记 W2 时**没有这份预缩放** ✓✓ ⇒ 激活按原样舍入 ⇒
第 2 步的 FFN 中间量约 **3.6e6** ✓✓，越过 F16 的 **65504** ✓✓ ⇒ **inf** ✓✓ ⇒
下一层的 tokens 带着 inf ✓✓ ⇒ 再一步变成 NaN ✓✓ ⇒ flow 全 NaN ✓✓。

⇒ **矩阵乘法是线性的** ✓✓，缩放输入不改变结果、只改变舍入前的量级 ✓✓ ——
而量级正是要改的东西 ✓✓。

**本轮改了什么** ✓✓：主栈的 **W2 投影不绑设备** ✓✓，落回 CPU 行路径
（CPU 路径自带那份 1/128 ✓✓）；其余四个投影照常上设备 ✓✓。
⇒ **F16 档现在 128×128 / 2 步 CPU 22.0 s → 设备 7.4 s（快 3.0×）** ✓✓，
与 CPU 的 PNG 差 **58.33 dB**（max_abs 2 ✓✓），不再有 NaN ✓✓；
严格 `--compute vulkan` 档同样出图 ✓✓（`backend=partial mode=vae` ✓✓）。
⇒ Q8_0 档不受影响 ✓✓（128×128 / 2 步 denoise **1.85~1.87 s** ✓✓，三次一致 ✓✓）。

**还差什么** ✓✓：

1. **共享算子**：设备的浮点 matmul 要能接住「带预缩放的中间量」✓✓ ⇒
   具体缺口是**设备端没有缩放激活的入口** ✓✓（`record_ffn` 记 W2 时只能原样上传 ✓✓），
   补上以后 FFN 的下投影也能留在设备上 ✓✓。
2. **F32 档的加载缺口已修** ✓✓：`z-image-turbo-f32.gguf` 也在 `models/z-image-gguf-mine/` ✓✓，
   但它原先**加载不了** —— `validate_dit` 把张量的格式**写死**了 ✓✓（bias/norm 一组要 F32 ✓✓、
   矩阵一组要 F16 ✓✓），于是加载时就报错：`Invalid tensor cap_embedder.1.weight:
   shape [2560, 3840] type F32; expected [2560, 3840] F16` ✓✓。
   ⇒ **那是加载器缺口，不是设备缺口** ✓✓，与 Qwen-Image-2.1 那个「格式该由张量自己声明」
   是同一类问题 ✓✓（[TODO-075](#todo-075-qwen-image-21-的基础版设备路径--现在能跑了但与-cpu-的差异和那个偶发的首轮差异都还没定位) ✓✓）。
   ⇒ **已修** ✓✓：校验按张量自己声明的格式来 ✓✓（`validate_dit` 的矩阵组接受 F16|F32 ✓✓、
   `require_matrix` 接受 F16|Q8_0|F32 ✓✓），加载按 dtype 分派 ✓✓
   （`linear_into` 新增 F32 分支，走 `F32Kernel` + `forward_f32_rows_dispatch` ✓✓，
   scale 的约定与 F16/Q8_0 两支一致：先缩放输入、输出再除回来 ✓✓；
   `force_f32_linear_into` 也按 dtype 读，F32 直接按 f32 读而不是从 F16 放宽 ✓✓）。
   ⇒ **但这个「F32 档」其实是混合格** ✓✓：453 个张量 = **F32 271 + Q8_0 180 + F16 2** ✓✓ ——
   180 个 Q8_0 是**主栈** 30 层 × 6 个矩阵（qkv/out/w1/w2/w3/adaLN_modulation.0 ✓✓），
   F32 的是 **refiner**（`context_refiner.*` ✓✓）与两个 embedder（`cap_embedder.1` / `x_embedder` ✓✓），
   2 个 F16 是 `cap_pad_token` / `x_pad_token` ✓✓
   ⇒ **没有哪一档的主栈是 F32** ✓✓ ⇒ 「F32 档的主栈上设备」这件事压根不存在 ✓✓
   （Q8_0 档与 F32 档的主栈是同一份 Q8_0 ✓✓，F32 档只把 refiner 与 embedder 换成了 F32 ✓✓）。
   ⇒ 实测（128×128 / 2 步 / seed 42 ✓✓）：CPU denoise **13.4~15.4 s** ✓✓，
   设备（`--gpu`）denoise **5.18~5.37 s** ✓✓，严格 `--compute vulkan` 同样出图 ✓✓
   （`backend=partial mode=vae` ✓✓）；CPU 与设备的 PNG 差 **33.10 dB** ✓✓，
   与 Q8_0 档 CPU 的差 **29.17 dB** ✓✓（两份文件只差 refiner/embedder 的精度 ✓✓）。
   ⇒ **设备端还差 F32 的绑定** ✓✓（`bind_block_weights` 的 `_ => continue` ✓✓）：
   它的主栈是 Q8_0 所以照常上设备 ✓✓（`RUST_GPU_DIAG=1` 量到 `projections=150` ✓✓），
   但 refiner 的 F32 矩阵留在 CPU ✓✓；设备其实有 BF16/F32 的转置浮点 matmul ✓✓，
   要用它需要主机侧把 F32 权重转置后再传 ✓✓（共享算子的布局约定 ✓✓）。
3. **设备 attention 与 CPU 差 ~1%** ✓✓：这是**与 F16 无关**的差异 ✓✓ ——
   Q8_0 档的设备路径同样有它（PSNR 49.7 dB ≈ 1% 相对误差 ✓✓），
   根因在共享的 attention kernel ✓✓，本轮没有动它 ✓✓。

**顺带记一条方法论** ✓✓：**设备路径的耗时波动很大** ✓✓ —— 同一规格、同一二进制，
Q8_0 档 128×128 / 2 步的 denoise 量到过 **1.86 s / 3.37 s / 8.35 s** ✓✓ 三次 ⇒
⇒ **性能数字必须多次测量** ✓✓，单次读数不能写进文档 ✓✓
（本轮所有引用的数字都重测过三次 ✓✓）。

**验收**：Z-Image 的 F16 档在 `--gpu` 下出图且与 CPU 的 PSNR ≥ 40 dB ✓✓（58.33 dB ✓✓）；
F16 主栈的四个投影在设备上跑完一步且结果有限 ✓✓；FFN 的下投影留在 CPU ✓✓；
F32 档能加载并出图 ✓✓（CPU 13.4~15.4 s ✓✓ / 设备 5.2~5.4 s ✓✓，
与 Q8_0 档 CPU 的差 29.17 dB ✓✓ —— 只差 refiner/embedder 的精度 ✓✓）。

**关联**：`src/models/diffusion/z_image/dit.rs`（`bind_block_weights` ✓✓、
`f16_projections_enabled` ✓✓）、`src/models/diffusion/z_image/dit_gpu.rs`
（`record_ffn` ✓✓、缺预缩放的地方 ✓✓）、
[TODO-073](#todo-073-ernie-image-的设备路径k-quant-慢q8_0-快-28-倍)（同一套「按权重量化格式选后端」的结论 ✓✓）。

### TODO-077: Z-Image 的设备路径不可复现 —— 同一 seed 的两次设备运行会出不同的图

**状态**：🔴 **未定位**（2026-11-18 新发现）。已定位到「设备的 attention 链」这一层，
再往里（哪一段、还是屏障不够）还没定。

**现象**（Q8_0 档 ✓✓，`models/z-image-gguf-mine/z-image-turbo-q8_0.gguf` ✓✓，
128×128 / 2 步 / seed 42 ✓✓，`--gpu` ✓✓）：

| 连跑次数 | 不同的 PNG | 说明 |
|---|---|---|
| 8 次 | **5 个** | 多数输出占 4/8 ✓✓ |
| 6 次（1 步） | **2 个** | ⇒ 不是跨步的状态 ✓✓ |
| 8 次（`RUST_GPU_ATTENTION=0`） | **1 个** | ⇒ **分叉在设备的 attention** ✓✓ |
| 8 次（`RUST_GPU_F16=0`） | **5 个** | ⇒ 与 refiner 走 CPU 行路径无关 ✓✓ |
| 6 次（`RUST_GPU_TILED=0`） | **4 个** | ⇒ 与 tiled matmul 无关 ✓✓ |
| 8 次（attention 链三段各自单独提交） | **4 个** | ⇒ **段与段之间的屏障不是问题** ✓✓ |
| 8 次（`SPLIT_FFN=1`） | **5 个** | ⇒ 不是 FFN 链的提交粒度 ✓✓ |
| 8 次（`FFN_BATCH_MAX=0`，FFN 不批量） | **5 个** | ⇒ 不是批量 FFN 链 ✓✓ |

⇒ **把 attention 链的三段（`kv_write` → `attention_scores_tiled` → `softmax` →
`attention_values`）改成各自单独提交并等待，不确定性不变** ✓✓；把 FFN 链拆开也不变 ✓✓
⇒ **竞态在单个 dispatch 内部** ✓✓ ⇒ 只剩 Z-Image 专有的那几个着色器 ✓✓：
`attention_scores_tiled.comp` ✓✓ 与 `qk_norm_rope_interleaved.comp` ✓✓
（ERNIE 走的是标准的 `attention_scores.comp` + 同一份 softmax/values ✓✓，而它是确定的 ✓✓）。

⇒ 差异幅度从**末位舍入**（PSNR 65.6 dB、max_abs 1 ✓✓）到**整张图都不同**
（PSNR 23.1 dB、max_abs 154 ✓✓）都有 ✓✓ ⇒ 不是固定的某个系数写错 ✓✓，
更像**读到了竞态下的数据** ✓✓。

**已排除** ✓✓：

1. **CPU 路径是确定的** ✓✓：F32 档 CPU 连跑 4 次逐位相同 ✓✓。
2. **文本编码器不在设备上** ✓✓：`RUST_GPU_TEXT` 默认关 ✓✓，编码在 CPU ✓✓。
3. **不是设备路径的通病** ✓✓：ERNIE-Image Q8_0 档的设备路径（256×256 / 2 步）
   连跑 4 次**逐位相同** ✓✓（它走的是标准的 `record_attention_rows` ✓✓，
   而 Z-Image 走自己的 `record_diy_attention_full` ✓✓）。
4. **VAE 不是源头** ✓✓：关掉设备 attention 之后 VAE 照旧在设备上 ✓✓，而那时是确定的 ✓✓。

⇒ **所以分叉在 Z-Image 特有的那条 attention 链上** ✓✓：
`kv_write` → `attention_scores_tiled` → `softmax` → `attention_values` ✓✓
（`record_diy_attention_full` ✓✓，全部录进**同一个** command buffer ✓✓，
中间只有 `compute_barrier` ✓✓）。

**算子级测试：三段都单独跑过，都确定** ✓✓（新增在 `src/vulkan/zimage_probe.rs` ✓✓，
`RUST_ZIMAGE_VK_PROBE=1` 才跑 ✓✓）：

| 单跑的一段 | 同一输入连跑 8 次 |
|---|---|
| `attention_scores_tiled`（含 kv_write ✓✓） | **8/8 逐位相同** ✓✓ |
| 融合的 Q/K norm + RoPE ✓✓ | **8/8 逐位相同** ✓✓ |
| 整条 attention 链（kv_write → scores → softmax → values）✓✓ | **8/8 逐位相同** ✓✓ |

⇒ **三个着色器单独跑都没有竞态** ✓✓ ⇒ 竞态在**它们之外** ✓✓。

**逐 block 探针查到了竞态的样子** ✓✓（临时加在 `z_image/dit.rs` 上，测完已删 ✓✓）：
每次运行打出每个 block 的 attention 输出指纹 ✓✓（求和 + 最大绝对值 + 前 4 个值 ✓✓），
两次运行对比 ⇒ **分叉从某一个 block 开始** ✓✓（实测分别落在 layer 4 / 5 / 8 / 15 / 31 ✓✓
⇒ 与时序有关 ✓✓，不是固定的那一层 ✓✓）。

⇒ **而且不是「算错」，是「读早了」** ✓✓：出问题的那个 block，它的 attention 输出
**前 4 个值和上一个 block 一模一样** ✓✓（例：某次运行的 layer 8 输出
`head=[3.9762678, -0.04423639, 1.6983759, -1.7022396]` ✓✓，
与另一次运行的 layer 7 **逐位相同** ✓✓），
而求和与前一个 block 不同 ✓✓ ⇒ **缓冲区是「新旧混在一起」的** ✓✓
⇒ **读回拷贝和设备的 values dispatch 撞上了** ✓✓
（有时整块都是旧的 ✓✓，有时只有一部分是旧的 ✓✓ ⇒ 拷贝发生在 dispatch 写到一半的时候 ✓✓）。

⇒ **这与「屏障」无关** ✓✓：三个着色器内部、段与段之间、链与链之间都各有屏障 ✓✓，
而单独跑都确定 ✓✓ ⇒ **要么某次提交里根本没有那个 dispatch** ✓✓，
要么**读回的拷贝没有排在那个 dispatch 之后** ✓✓。
（提交机制本身看起来是对的：`submit_commands` 会 `end_command_buffer` → `reset_fences` →
`queue_submit` → `wait_for_fences` ✓✓，命令缓冲与 fence 都在互斥锁下 ✓✓。）

**下一步** ✓✓：

1. **在提交层查**：给 `submit_commands` 加一条断言 —— 提交前检查命令缓冲里
   确实记录了该有的 dispatch ✓✓（`recorded_dispatches` 计数已经有 ✓✓，
   但它是全局的 ✓✓，要按提交分组才有用 ✓✓）；同时在 `record_diy_attention` 之后
   立刻读回 `attention_out` 的前几个值 ✓✓，与提交后的值比对 ✓✓ ——
   若提交后立刻读是对的、而几微秒后 `run_block_gpu` 再读却变了 ✓✓，
   那就是**读回路径**的问题 ✓✓（`read_f32_into` 的 staging 拷贝 ✓✓），
   否则就是**提交丢 dispatch** ✓✓。
2. 若提交确实带了那个 dispatch ✓✓，查 `read_f32_into` 的 staging 拷贝路径 ✓✓：
   它把 arena 拷到 staging 再由主机读 ✓✓，拷贝本身也在同一个队列上 ✓✓，
   理论上排在 compute 之后 ✓✓ —— 但**读到的「新旧混合」说明它没有** ✓✓。
3. 修好之前，**文档里不能再写「同一 seed 两次设备运行逐位相同」** ✓✓ ——
   那句话是**样本太少**得出的 ✓✓（同一份二进制连跑 3 次恰好相同 ✓✓），
   已更正 ✓✓（`docs/MODEL_LIST.md` ✓✓、`docs/develop/SUPPORTED_MODELS.md` ✓✓）。

**验收**：Z-Image 的设备路径在同一规格下连跑 8 次**逐位相同** ✓✓；
CPU 与设备的 PSNR 维持 ≥ 40 dB ✓✓。

**关联**：`src/models/diffusion/z_image/dit_gpu.rs`（`record_diy_attention` ✓✓）、
`src/vulkan/ops.rs`（`record_diy_attention_full` ✓✓、`record_diy_attention_scores` ✓✓、
`record_softmax_rows` ✓✓、`record_attention_values_rows` ✓✓、`compute_barrier` ✓✓、
`read_f32_into` ✓✓）、`src/vulkan.rs`（`submit_commands` ✓✓、`compute_barrier` ✓✓）、
`src/vulkan/zimage_probe.rs`（本轮新增的三个算子级探针 ✓✓：
`tiled_attention_scores_are_reproducible` ✓✓、
`fused_qk_norm_rope_is_reproducible` ✓✓、
`diy_attention_chain_is_reproducible` ✓✓）、
`shaders/glsl/attention_scores_tiled.comp` ✓✓、`shaders/glsl/softmax.comp` ✓✓、
`shaders/glsl/attention_values.comp` ✓✓、
[TODO-076](#todo-076-z-image-的-f16-dit-档在设备路径上第二步全-nan--根因是设备端没有-cpu-那份-1128-的预缩放)（同一套设备路径的其它缺口 ✓✓）。

### TODO-078: 图像编辑模型的 HTTP 入口未实现 —— LongCat-Image-Edit 与 Qwen-Image-2.1

**状态**：📌 未实现（本轮未接）。

**结论**：本轮把文生图的 HTTP 入口接上了（Z-Image、ERNIE-Image 共用
`POST /v1/images/generations`，与 CLI 逐位相同），但**图像编辑类**的两个模型
还没有入口 —— 这是**缺实现**，不是「不适用」。

| 模型 | 缺什么 | 为什么不能套用现成的路由 |
|---|---|---|
| LongCat-Image-Edit / Edit-Turbo | 参考图入口 | 请求要带一张待编辑的图加一段指令，`/v1/images/generations` 的「prompt → 一张图」形状装不下 |
| Qwen-Image-2.1 | 参考图 + 自有编排 | 除了参考图，它还驱动正负 CFG 两路、Euler 调度、Qwen3-VL 视觉塔与 DeepStack，编排不在 `DiffusionPipeline` 里 |

**已就位的基础**：

- 路由本身是通用的（`src/app/server/image_generation.rs`），请求体是
  `prompt` / `size` / `width` / `height` / `steps` / `seed` / `guidance` /
  `n` / `response_format`；要接编辑模型，需要加 `image`（base64 或路径）字段。
- 模型层已经有可复用的切分：`src/models/diffusion/longcat/`（Transformer 与
  图文条件）与 `src/models/diffusion/qwen_image_2_1/`（DiT / 组件契约）。
  缺的只是**把它们编排成一次生成**，再映射到请求 —— 这部分属于 models 层
  （模型管线），HTTP 层仍然只做输入映射。

**下一步**：

1. 定请求形状：编辑类模型用 `/v1/images/edits`（multipart 还是 JSON+base64），
   还是复用 `/v1/images/generations` 加 `image` 字段。按 OpenAI 的惯例是前者。
2. 把 LongCat 的完整编辑入口下沉成 models 层的管线（现在是 CLI 编排），
   再让 CLI 与 HTTP 共用 —— 现在的 CLI 编排住在 `src/app/diffusion.rs`，
   按分层红线它应该搬进 `src/models/diffusion/longcat/`。
3. Qwen-Image-2.1 同理：把 `src/app/diffusion.rs` 里的 `run_qwen_image_2_1`
   编排下沉成 models 层管线。

**验收**：两个模型各有一个 HTTP 请求，返回的 PNG 与同参数的 CLI 输出逐位相同；
请求缺参考图、参考图非方形、参考图尺寸不符这三种情况各自报 400。

**关联**：`src/app/server/image_generation.rs`、`src/app/server/diffusion.rs`、
`src/app/diffusion.rs`（`run_longcat_image_edit` / `run_qwen_image_2_1`）、
`src/models/diffusion/longcat/`、`src/models/diffusion/qwen_image_2_1/`、
[SUPPORTED_MODELS.md](SUPPORTED_MODELS.md)（两行的 HTTP 状态）。
