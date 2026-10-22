# P5-2 重做 + 两条支线开发计划（2026-10-20）

动机与数据背景见 [throughput-analysis-261020.md](throughput-analysis-261020.md)：官方 FP8 grouped kernel 对照下，我方 decode 的 MoE 段 57.3ms/step 由「320 次专家迭代 × ~10 个 Python/torch 调度小算子」主导（launch 本身 ~10ms 不是大头）；prefill 由小 kernel SM 填充不足（1.51 波）主导。旧 P5-2 v2 合并 kernel 在 opt-p5-2 分支、B=32 旧配置下 eval 变慢被否；本次按当前 P5-3 tile 配置重做，判定基准换成 bench_throughput.py（prefill/decode 分档），eval wall 与 ppl 作为正确性副指标。

开发顺序：**工作 1（主线，最大杠杆）→ 工作 2 → 工作 3**。每条都按既有习惯：先写测试程序对齐主流程逻辑（含 GPU 填充度/并行 SM 分析）→ 自测全绿 → 本人手动跑主流程。

## 工作 1：P5-2 重做——按 bit 池合并多 expert 的 grouped kernel

**参考**：opt-p5-2 分支 `moe_gate_up_grouped_gf_v2`/`moe_down_grouped_gf_v2`（block_info 方案：1D grid，每 block 读 `(num_blocks,3) int32` 的 `[expert_idx, block_b, block_n]`；expert_info 记 token_cursor 行段；扁平 (B_total, N) 布局）；当前 main 的 `quantization/wxa8/bit_partitioned_moe.py::_gate_up_matmul/_down_matmul`（含 hoisted 旋转/量化分支、P5-3 tile 表）。

**步骤**：
1. **对拍测试先行**：`test/test_moe_merged_kernel.py`（新）——按主流程 ctx 结构构造真实形状（35B 4bpw 各 bit 池：gate_up/down、bit 1/2/4、混合 inter_size），合并 kernel vs 逐 expert 结果对拍；并做 **SM 填充度分析**（单 expert 1.51 波 vs 合并后波形，复用 backlog §4.3 方法）。
2. **kernel 实现**：`turboquant_utils/triton_kernels_a8.py` 加 wxa8 合并版 gate_up + down（block_info 1D grid + P5-3 当前 tile 表 + 扁平布局，按当前主流程逻辑重写，不搬旧分支代码）；随后 wxa16/wxfp8 同型。先 wxa8（decode 主力）。
3. **接入**：`quantization/wxa8/bit_partitioned_moe.py` 增加合并路径——按 bit 池一次性构建 block_info/expert_info、flat act、SiLU 融合一次、scatter 按 token_cursor 切片；与逐 expert 路径并存，用开关控制（对拍对照用）。新路径默认开，逐 expert 仅作对照，**不降级回退**。**验收标准：热循环里不许剩任何逐 expert 的 torch 算子**——decode 每步的调度小算子从 ~3200 个（320 迭代 × ~10）降到 ~150 个；down 路径的逐 expert 旋转+量化 v1 先保留（每 expert 一次 launch，~2-3ms/步），v2 再评估折进合并 kernel。
4. **自测**：kernel 对拍全绿 + 小形状吞吐微测 → 本人手动跑 `bench_throughput.py`（4bpw + 2bpw 全档，check PASS）与 ppl eval（ppl 必须与逐 expert 逐位一致）。

**风险与口径**：旧否决（269.4→288.6s）在 B=32 旧 tile 配置下做出；本次带 P5-3 配置重测，prefill/decode 分档判定，eval wall 仅参考。若合并后某档变慢，按档位分析填充度/尾波而不是整体回退。

## 工作 2：4bpw 配置的 attention 4-bit 路径（Lloyd-Max LUT → 直读）

4bpw 的 qscheme attn=[4]，attention 走 4-bit Lloyd-Max 码本的 in-kernel LUT（W4A16），是 4bpw 特有开销；2bpw 的 attention 是 8-bit 均匀码本 identity 直读（免查表）。两条候选，实测择优（**先 B 后 A**：B 零重量化成本可快速验证；A 需重新量化 ~3h/份，B 精度不可接受时再上）：

- **B. load-time fold：4-bit Lloyd-Max → int8 网格**：复用 WFF-1 的 fold 模式（turboquant_utils/triton_kernels_fp4.py::fold_indices_to_e2m1 的分块实现可作模板），load 期把 LUT 折进 int8 权重，attention 直接走现有 W8 identity kernel。**现有 261016 checkpoint 直接可用**；4-bit 仅 16 级、级间距远大于 int8 网格步长（max/127），fold 损失预期远小于 8-bit 时代「256 级塌 61 级」的坑，具体数值测试给出。
- **A. 均匀 4-bit 码本 + identity 直读 kernel**：量化侧把 attention 的码本改成均匀（量化时生成），inference 侧加 bit_width=4 的 identity 分支（indices 即权重值 × scale）。需重新量化。

**步骤**：1) 定位量化侧 attention 码本生成点（qscheme attn 的码本选择逻辑）；2) `test/` 对拍：B 的 fold int8 与 LUT 的数值差异（含 4-bit 上 int8 网格损失的等效评估）、A 的 uniform-4bit identity vs LUT 数值/吞吐；3) ppl 影响实测（本人手动 eval）；4) 接入 + 全档 bench。

## 工作 3：激活量化链——先测量再优化（P8-1 式）

出发点：分析 §2 第 3 条差距来源——我们 wxa8 每层对激活做 per-token-per-group 旋转+量化（`rotate_quantize_fused`，hoist 后仍按 bit 池来一遍），官方 FP8 是 per-tensor 一次缩放直进 tensor core，这条链每层每个 B 都付费；现有 profile 都是 B=32 口径，B=1 没测过。

**步骤**：1) 测量：沿用 `test/test_wxa8_loop_profile.py` 思路做 **B=1 decode** 口径的逐 stage 拆分（rotate_quantize/hoisted 在 57.3ms MoE 段里的占比）；2) 依测量定优化点（候选：量化融合进 gather、按 bit 批量量化、B=1 轻量路径）；3) 对拍 + 吞吐验证（check PASS + ppl 一致）。注意在工作 1 落地后的新基线上测（gather+量化形态会变）。

## 共用验证口径

- 吞吐：`bench_throughput.py --quant-dir models/qwen3.5-35B-4bpw-261016-{gptq,turboquant}/ --mode wxa8/wxa16/wxfp8`（1018 口径基线：dec 11.26-11.39 tok/s、pf2048 1603-1611 tok/s）；2bpw checkpoint 同样回归。
- 正确性：`eval_qwen35.py` ppl（wikitext2/c4 与现数字逐位一致）+ bench 的 cache-vs-full 一致性守卫 check PASS。
- 环境：dart312-t38（triton 3.8 + ptxas 12.8，勿动）；kernel 对拍测试用 `conda run -n dart312-t38 python test/test_*.py` 自测。

## 变更记录

- 2026-10-20：立项。工作 1 为主线（P5-2 重做），工作 2/3 支线；参考与步骤如上。本人拍板后开工。
- 2026-10-20 晚：**工作 1 步骤 1-4 完成（待本人主流程验证）**——kernel（`_wxa8_merged_kernel_gf` 双模式 + 两个 wrapper + block_info builder，triton_kernels_a8.py 新节）、对拍测试（test_moe_merged_kernel.py：kernel 级 6 组 + 模块级 A/B 6 组全 bit-exact）、接入（wxa16 forward 抽出 `_expert_compute` 方法，wxa8 覆盖为合并路径，`DARTMOQ_MERGED_MOE=0` 回退逐 expert）、自测预览（64 expert 合成模块：decode 3.24× / prefill 2.69×）。bring-up 修复：wrapper 参数顺序与 kernel 签名错位、down 的 N tile 数误用 group 数、norms_prescaled 需用 ctx 标志（硬编码 True 会在未预乘路径上错 1/√gs）。待本人跑 bench_throughput.py 全档（check PASS）+ ppl eval（与逐 expert 逐位一致）。
- 2026-10-20 深夜：**回归修复**——WxFP4 继承 WxA8 会误继承合并路径（e2m1 打包激活喂 int8 kernel → warmup gather 越界 assert，1018.thrput.4 t38 实测）；已改为合并路径仅 WxA8 本类生效（`self.__class__ is WxA8BitPartitionedGroupMoE`），WxFP4 回退逐 expert 并 A/B 验证 bit-exact。待 fp4 合并 kernel 落地后单独接线。注意本次 wxfp4 在 t38 已越过 WFF-5 的 fold OOM 点（分块 fold 生效），修复后若再挂即为驻留副本问题（方案留档待拍板）。
- 2026-10-20 深夜：**t38 正式数字（1018.thrput.4/5，均为 dart312-t38，check 全 PASS）**——4bpw wxa8 合并路径：pf2048 1603→3619.4（2.26×）、pf4096 2692→4898.8（1.82×）、decode 11.27→13.34（+18%）、ttft 0.794→0.322、MoE 段 57.3→41.3ms/step；prefill 已反超官方 GPTQ-Int4（3298）。2bpw wxa8 合并路径：pf2048 1611→3648.8（2.26×）、pf4096 2766→5186.8（1.88×）、decode 9.78→11.69（+19.5%）、MoE 段 63.2→46.6ms。同环境确认：4bpw decode 快 2bpw 14%、2bpw 长 prefill 快 5.9%（合并后带宽收益可见度从 +3% 提高到 +5.9%）。wxa16/wxfp8 未接线（「随后同型」）。
- 2026-10-20 深夜：**WFF-5 修复落地**——`_FOLD_BYTES_BUDGET`（64MB/层/方向）小池折叠、大池跳过走 in-kernel LUT 分支（bit-exact A/B 验证，maxdiff 0）；2bpw wxfp4 全绿（小池折叠不变），4bpw wxfp4 重跑验证中。
- 2026-10-20 深夜：**4bpw wxfp4 首个成功数字**（1018.thrput.6，check PASS）——pf2048 1567.3 / dec 11.12 / ttft 0.806 / MoE 58.8ms。为「逐 expert 路径 + 预算跳过 fold 的 in-kernel LUT」双重旧路径叠加，显著慢于合并版 wxa8（3619/13.34）；待 wxfp4 合并 kernel 同型接线后重测。注意 wxfp4 合并版的自然设计就是 in-kernel LUT + 合并 grid（不需要 fold 副本，4bpw 无 +10GB 代价）。
- 2026-10-20 深夜（续）：**wxfp8 合并同型落地（待本人主流程验证）**——
  `triton_kernels_fp8.py` 增 `_wxfp8_merged_kernel_gf` + 两个 merged wrapper
  （block_info/expert_info 骨架复用 a8 的 `_build_merged_block_info`，import
  不重复实现；fp8 语义三处差异：激活 tile e4m3 且 masked load 不给 other、
  码本 e4m3 LUT、group 内 fp32 dot 直接累加）；wxa16 基类抽出模式无关共享
  工具（`_merged_pools_ok` / `_merged_pools` / `_merged_expert_info_rows` /
  `_merged_expert_info_groups`），wxa8 编排改调共享工具（行为零变化，模块
  A/B 与 test_quant_io wxa8 段回归 bit-exact / relerr 0.01336 与记录一致），
  WxFP8 加 `_expert_compute` 合并覆盖（精确类门控 + `DARTMOQ_MERGED_MOE`
  开关，编排与 wxa8 逐点对应）。对拍（test_moe_merged_kernel.py，t38 自测）：
  kernel 级 f8 decode/prefill × bit1/2/4 全 **bit-exact**；模块级 fp8 A/B
  bit-exact（⚠ 形状必须 gs=128——fp8 kernel 的 masked load 不能给 other，
  K 尾部未定义 lane 会污染有效输出，真实模型 gs=128 恒满足）；test_quant_io
  WxFP8 段全绿（relerr 0.05042，合并路径默认开）。perf 预览（64 expert 真实
  形状）：decode 3.16× / prefill 2.60×（wxa8 同批 3.16×/2.72×）。待本人
  bench_throughput.py 全档（check PASS）+ ppl eval（与逐 expert 逐位一致）。
- 2026-10-20（续）：**wxfp8 合并路径正式数字**（1018.thrput.7，git 11e7d11+，
  check 全 PASS，t38）——4bpw：pf16 46.4→84.7–85.1、pf128 181–184→445–446.2、
  pf2048 1606–1609→**3605.4–3620（2.25×）**、pf4096 2680–2686→**4932.2–4959.9
  （1.84×）**、decode 11.30–11.43→**13.93–14.03（+23%）**、ttft 0.791–0.797→
  0.36–0.361、MoE 段 39.9ms；2bpw：pf2048 1628.2→3590.9（2.20×）、pf4096
  2785→5188.2（1.86×）、decode 9.92–9.93→11.62–11.64（+17%）、ttft 0.819→
  0.404、MoE 段 46.8ms。判读：**4bpw wxfp8 decode 反超 wxa8 合并版**
  （13.98 vs 13.34，+5%）——逐 expert 时代两模式 decode 持平（11.3 vs 11.3），
  调度链消失后 FP8 Tensor Core 算力优势才兑现（MoE 段 39.9 vs 41.3ms）；
  prefill 持平（3620 vs 3619.4），小 prefill（pf16 84.7 vs 111.5、pf128 445
  vs 492.7）wxa8 仍更快；2bpw 两模式全档持平（bit2 主导、kernel 形态相同）；
  与官方 FP8 decode 差距收窄至 **1.12×**（15.77/14.03）。ppl eval 待本人跑
  （bit-exact 预期：合并 kernel 同 tile 分解同累加顺序，模块 A/B 已零差）。
- 2026-10-21：**wxa16 合并同型落地（待本人主流程验证）**——
  `triton_kernels.py` 增 `_turboquant_fused_matmul_merged_kernel_gf` + 两个
  merged wrapper（block_info/expert_info 骨架复用 a8 的
  `_build_merged_block_info`，import 不重复实现；wxa16 语义差异：无激活
  量化/无 xs、acc_g/total_acc 用 tl.float16（`out_dtype=tl.float16`）、
  norms masked other=1.0、旋转在 wrapper 外——gate_up 整池一次
  batch_rotate_input（hoisted 或现算）、down 逐 expert `seed + start`）；
  WxA16 基类 `_expert_compute` 加合并门控（精确类 + env + pools_ok +
  **groups_ok** + gf 双侧标志）。groups_ok 是 bring-up 补的：test_quant_io
  小形状 inter/gs=1 触发合并 wrapper 的 num_groups>=2 拒绝（逐 expert
  wrapper 有 ==1 回退分支，合并版设计上要求调用侧回退）——真实模型
  gate_up 16 组 / down 4 组恒满足。对拍（t38 自测）：kernel 级 a16
  decode/prefill × bit1/2/4 全 bit-exact，模块级 a16 A/B bit-exact
  （B=12/2048）；test_quant_io 全绿（wxa8 0.01336 / wxfp8 0.05042 与记录
  一致，子类链零扰动）。**a16 特有口径注记：tile 表两档 BLOCK_K 不同
  （gate_up small 128 / large 32），fp16 累加顺序跨档会变——合并
  （B_total 选档）与逐 expert（B_e 选档）跨档时非位级一致（舍入级，
  两者都正确）；真实 decode（双 small）/ eval bs=16（双 small）/ 大
  prefill（双 large）均同档，ppl 位级校验仍成立**。perf 预览（64 expert
  真实形状）：decode 2.22× / prefill 2.43×。待本人 bench（默认 --mode
  wxa16）+ ppl（run.evalppl.sh wxa16 行）。
- 2026-10-21（续）：**勘误官方 FP8 双卡口径**（详见
  throughput-analysis-261020.md 变更记录）——bench 走 device_map=auto，
  是 accelerate 连续层块 naive 切分：单请求流严格串行、无 TP、无流水线
  重叠、无 EPLB（transformers 无 expert parallelism 概念）；双卡只解决
  容量（35GB>31.4GB）不折算力。原「单卡等价 ~5000 / per-GPU 3.1×→1.4×」
  作废，per-GPU prefill 差距按全量口径：合并前 6.2×、合并后 2.77×
  （10009 vs 3619）。
