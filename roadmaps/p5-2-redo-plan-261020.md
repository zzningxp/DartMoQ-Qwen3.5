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
