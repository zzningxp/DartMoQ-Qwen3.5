# WxFP4 Fold 路线规划（2026-09-27）

> **定位**：WxFP4 保守版（`wxfp4-plan-260924.md` WF4-4，已 eval 投产）的 W 侧开销消解。
> bit1/4 expert 的 W 当前以「packed 码本索引 + kernel 内 nib LUT 查表」形态进 GEMM——
> 本路线把 idx→e2m1 映射挪到 ctx 构建期一次完成（fold），显存常驻 e2m1，kernel 零转换，
> 数值 bit-exact 不变；checkpoint、bit2 路径、attention 路径完全不动。
> **来源**：2026-09-27 与本人架构讨论定调——扩张到 4-bit 是必须的，把扩张位置做到最优；
> W 是静态数据，kernel 内逐 tile 重做扩张是性价比亏损。

---

## 一、背景与证据链

1. **kernel 内转换的开销结构**（`_wxfp4_fused_matmul_kernel_swapped_gf` 的 W 侧
   unpack→LUT→拼包块）：
   - LUT gather 逐元素不可向量化（BN×BK=16K 次/tile）；idx 走 int32 展开域，寄存器压力大；
   - 转换结果须 reg→smem 写回才能喂 tcgen05 MMA（dot_scaled 操作数在 smem），
     直存路径 value 可 TMA 直达 smem——LUT 路径多一整趟往返；
   - 转换链在 k 循环关键路径上，CUDA 核与 tensor 核无法重叠该部分工作。
2. **实测锚点**（同形状 B2048 N1024 K2048）：
   | W 侧形态 | kernel | 吞吐 |
   |---|---|---|
   | packed 索引 + kernel 内 LUT | grouped swapped | 177 TF |
   | 直存 e2m1（指针） | expert 版 | 319 TF |
   | 直存 e2m1（TMA+persistent） | expert 版 | 389 TF（dense 492） |
   | 纯 TMA 参照上限 | — | 546 TF |
3. **真实 per-expert 形态**（WF4-2 迭代 1 实测）：gate_up Be32/64 37-39us vs
   fp8 13-15us（**0.33-0.40x，转换开销是主因**——WF4-2 迭代 1 的遗留优化项
   "gate_up 需转换开销消解"即由本路线承接）；down 已持平 fp8。
4. **预期管理**：当前 MoE wall ~80% 是 per-expert Python 循环（P4 profile），
   fold 的直接收益兑现在 gate_up per-expert 微秒级、大 B 场景与 kernel 简化；
   MoE wall 层面的提升受循环开销压制（wall 层消解属 P5 系，不在本路线）。

## 二、与历史决策的关系（显存账澄清）

- WF4-P1 曾否决「W 侧离线展开为 e2m1」——**该否决针对 bit2 大池**：
  inter=130304（每层 1018/1024 sub-expert），显存 2× 膨胀（9.5GB→19GB）不可接受。
- 本路线折叠对象是 **fp4 线内的 bit1/bit4 小池**（每层 6/1024 sub-expert，
  2bpw-260831-u8 checkpoint 的 DP 分配 bit1=4 块 / bit4=2 块）：

  | bit | 池 inter | 折叠前后 B/elem | 每层显存变化 |
  |---|---|---|---|
  | 4 | 256 | 0.5 → 0.5 | **零膨胀**（gate_up 0.5MB + down 0.25MB 尺寸不变） |
  | 1 | 512 | 0.125 → 0.5 | +1.18 MB（gate_up +0.79 + down +0.39） |

  - 全模型（40 层）上限 **+47 MB**（旧 indices buffer 保留不释放；替换共享 key
    可省这笔，但为杜绝 wxfp4↔其他模式类切换读到污染 buffer 的风险，选择保留）；
  - bit2 维持 wxa8 int8 原路径，**3-4 激进版入口条件 2/3 不受影响、继续挂起**。
- **精度**：fold 与现行 LUT 是同一映射，仅执行时机/位置不同（每 forward 逐 tile →
  ctx 构建一次），输出 **bit-exact**；码本 v2 relerr（1-bit 0 / 2-bit 0.0305 /
  4-bit 0.076）不因折叠改变。保守版 eval 基线（wiki 7.7954 / c4 11.2711，
  WF4-4 2026-09-25）应逐位复现。

## 三、方案设计

数据流：checkpoint packed 索引（磁盘不动）→ ctx 构建期一次 fold →
显存常驻 packed e2m1（group-first 同布局）→ GEMM 直读 + MMA。

1. **fold 工具**（`turboquant_utils/triton_kernels_fp4.py`，紧挨 `build_e2m1_codebook`）：
   `fold_indices_to_e2m1(indices_packed_gf, nib, bit_width, group_size)`
   → `(G, N, GS//2) uint8`。复用 `turboquant_utils/quantize.py` 的 `unpack_nbit`
   （byte 内 low-bit-first，与 kernel `>> (pos*bit)) & mask` 约定一致，
   由测试 [1] byte-exact 验证）+ `nib[idx]` gather + `lo | hi<<4` 打包，纯 torch。
2. **kernel 分支**（`_wxfp4_fused_matmul_kernel_swapped_gf`）：新 constexpr
   `W_FOLDED`——True 时 W tile 从 folded buffer `(G, N, GS//2)` 直接 load
   （byte 域 mask，写法同 A 侧），跳过 unpack/LUT/拼包整块；packed_per_group
   固定 GS//2，row_base 与 legacy（GS//(8/bit)）分开计算。
   **legacy LUT 分支原样保留**（对照路径 + 项目规范禁删）。
3. **wrapper 透传**：`_launch_swapped` + 两个 slice wrapper 加可选参
   `w_e2m1_gf=None`——非 None 按同布局切片（rows `[:, r0:r1, :]` /
   in_features `[g0:g1]`）+ W_FOLDED=True，nib_ptr 传 folded buffer 占位
   （constexpr 分支不读）。顺手补模块级 `import math`
   （`triton_kernels_fp4.py` slice wrapper 的潜伏 NameError，生产未触发）。
4. **ctx 构建**（`quantization/wxfp4/bit_partitioned_moe.py` `_get_bit_context`）：
   FP4_BITS 循环内算出 nib 后，新增 `gu/dn["indices_e2m1_gf"] = fold(...)`。
   **新增 key，不替换共享的 `indices_packed_gf`**（比 norms_gf 替换先例更保守，
   模式切换零风险）；`_gate_up_matmul` / `_down_matmul` 调用点改传 folded key。
5. **明确不做（本轮）**：TMA/persistent 版 grouped kernel（折叠后 W 已可 TMA，
   bench 若显示 load-bound 再立后续项，参照 expert 版 492 TF）；bit2 迁移；
   checkpoint/state_dict 格式改动；删除 legacy LUT 路径。

## 四、实施步骤

| 阶段 | 内容 | 前置 |
|---|---|---|
| WFF-0 | 本规划文件 | — |
| WFF-1 | fold 工具 + kernel W_FOLDED 分支 + wrapper 透传（含 math import 修复） | WFF-0 |
| WFF-2 | `test/test_wxfp4_fold_gemm_align.py`：[1] fold byte-exact（bit1/2/4）[2] folded vs legacy kernel bit-exact + vs torch 契约 relerr（对齐 0.00021 口径）[3] bench（bit1 N1024K2048 / bit4 N512K2048 gate_up 实际形状，B∈{64,2048}，打印 TF + CTA/SM；锚点 177/319/389/546） | WFF-1 |
| WFF-3 | ctx 接线 + 调用点切换 + `test_quant_io.py` 既有 WxFP4 段回归 | WFF-2 |
| WFF-4 | 本人手动 eval（t38 env）：ppl 应与 0925 基线逐位持平 + gate_up/down kernel 阶段耗时对照 | WFF-3 |

> 遵守项目规范：禁止降级回退（fold 若有问题须在当前版本迭代修复，legacy 分支
> 仅作对照与回退保险，不作为回退目标）；不自定版本号；执行日志同 wxfp4-plan
> 规则——每完成一步追加一条，数据抄实测输出。

## 五、验收标准

1. WFF-2 [1]：fold 输出与「legacy kernel 构造路径的 torch 复算」**逐字节相等**（bit1/2/4）。
2. WFF-2 [2]：folded kernel vs legacy kernel 输出**逐元素相等**；
   vs torch 契约 relerr 与现行 0.00021 同量级。
3. WFF-2 [3]：bench 实测抄入执行日志；重点观察 gate_up per-expert 形状
   （37-39us → ?，对照 fp8 13-15us）的改善幅度。
4. WFF-4：eval ppl 与 2026-09-25 保守版基线逐位一致（bit-exact 的端到端背书）；
   速度记录 gate_up/down kernel 阶段（预期 wall 变化受循环开销压制，如实记录）。

## 六、风险表

| 风险 | 等级 | 缓解 |
|---|---|---|
| fold 位序约定与 kernel unpack 不一致 | 中 | 测试 [1] byte-exact 直接暴露；low-bit-first 复核 |
| W_FOLDED 分支布局编址 bug | 中 | 测试 [2] bit-exact 对拍兜底（legacy 保留作对照） |
| MoE wall 收益低于预期（循环占 80%） | 高（预期内） | 本路线目标定位 kernel 层消解；wall 层归 P5 系；不因 wall 无感回退 |
| 未来 checkpoint bit1/4 池变大 → 显存账恶化 | 已发生（WFF-5，2026-10-19） | 4bpw 全 4bit 池实测 OOM；分块 fold 已修瞬时分配，驻留副本方案见 WFF-5（legacy LUT 跳过 / kernel 内算术化 待拍板） |

## 变更记录

- 2026-09-27 WFF-0：初版规划（fold 直存方案与显存账，与本人讨论定调）。
- 2026-09-27 WFF-1/2/3：fold 工具 + W_FOLDED kernel 分支 + wrapper/ctx 接线 +
  对拍/吞吐/集成回归全绿——folded vs legacy **bit-exact**（gate_up/down 两种
  切片 + test_quant_io 集成 relerr 0.21473 与折叠前完全一致）；吞吐 gate_up
  形状 **1.99-2.43×**（B64 40.2→16.6us；B2048 bit1 185→367 TF）。
- 2026-09-27 WFF-4 eval 闭环（本人手动）：**ppl 逐位一致**（wiki 7.7972 /
  c4 11.2708）；wall **-9.8%**（110.53→100.79s），**wxfp4 成为最快模式**
  （vs wxa8 -6.5%）。WFF-x 全部完成。
- 2026-10-19 WFF-5：4bpw（全 4bit）checkpoint 上 wxfp4 首层 OOM，见执行日志
  WFF-5（风险表「bit1/4 池变大」条目兑现）。

## 执行日志

> 条目格式：`- 日期 事项：做了什么 → 实测数据 → 结论/下一步`。数据必须抄实测输出，不写主观估计。

### WFF-0：规划 ✅ 已完成

- 2026-09-27：本文件创建；证据链汇总自 wxfp4-plan WF4-2/WF4-4 执行日志；
  显存账基于 2bpw-260831-u8 的 DP 分配（bit1=4 / bit4=2 sub-expert 块每层）。

### WFF-1：fold 工具 + kernel 分支 + wrapper 透传 ✅ 已完成

- 2026-09-27：`turboquant_utils/triton_kernels_fp4.py` 落地
  `fold_indices_to_e2m1`（复用 `unpack_nbit` 纯 torch 实现，ctx 构建期执行
  一次）+ `_wxfp4_fused_matmul_kernel_swapped_gf` 新增 `W_FOLDED` constexpr
  分支（W tile 从 folded buffer (G,N,GS//2) 直读，byte 域寻址/掩码写法同
  A 侧 packed 加载；legacy LUT 分支原样保留作对照）+ `_launch_swapped` 与
  两个 slice wrapper 加 `w_e2m1_gf` 可选参（nib_ptr 传占位，constexpr 分支
  不读）。顺手补模块级 `import math`（slice wrapper 的潜伏 NameError，
  生产路径全传 norms_prescaled=True 从未触发）。

### WFF-2：对拍与吞吐 ✅ 已完成

- 2026-09-27（`test/test_wxfp4_fold_gemm_align.py`，t38 env，RTX 5090）：
  - [1] fold byte-exact：bit1/2/4 全 True（`unpack_nbit` 位序与 kernel
    unpack 一致，fold 输出 = legacy 构造路径 torch 复算逐字节相等）
  - [2] gate_up 行切片（B256 N512 K2048）：folded vs legacy **bit-exact
    True**（bit1/bit4）；契约 relerr legacy 0.00021 / folded 0.00021
    （同值，对齐既有口径）
  - [2b] down in_features 切片（B256 N2048 K512 bit4）：bit-exact True
  - [3] 吞吐（slice wrapper 默认 cfg，锚点 177/319/389/546 TF）：
    | 形状 | legacy(LUT) | folded | speedup |
    |---|---|---|---|
    | bit1 B64 N1024 K2048 | 40.2us（7 TF） | 16.6us（16 TF） | **2.42×** |
    | bit1 B2048 N1024 K2048 | 46.5us（185 TF） | 23.4us（367 TF） | **1.99×** |
    | bit4 B64 N512 K2048 | 40.0us（3 TF） | 16.5us（8 TF） | **2.43×** |
    | bit4 B2048 N512 K2048 | 46.3us（93 TF） | 21.8us（197 TF） | **2.13×** |
  - 判读：B64 per-expert 形状 16.5-16.6us 已贴近 fp8 参照 13-15us——
    WF4-2 迭代 1 遗留的 "gate_up 0.33x" 追到 **~1.1x，转换开销消解达成**；
    B2048 bit1 367 TF 反超 expert 指针版锚点 319 TF（grouped 结构带
    norms/groups 仍胜出）。B64 绝对值受 launch 与占用限制（16 CTA/0.1 SM，
    单 tile），预期内；MoE wall 层收益受 per-expert 循环占 80% 压制（§一.4）。

### WFF-3：ctx 接线 ✅ 已完成

- 2026-09-27：`quantization/wxfp4/bit_partitioned_moe.py` `_get_bit_context`
  对 FP4_BITS 新增 `gu/dn["indices_e2m1_gf"] = fold(...)`（**新 key，不替换
  共享 indices_packed_gf**）+ `_gate_up_matmul`/`_down_matmul` 调用点传
  `w_e2m1_gf=...get("indices_e2m1_gf")`。回归 `test/test_quant_io.py`
  （t38 env）：**WxFP4 保守版 moe forward relerr vs A16 = 0.21473，与折叠前
  WF4-3 记录（wxfp4-plan 执行日志）完全一致**（同种子确定性测试）——
  端到端 bit-exact 背书；wxa8（0.01336）/wxa16/wxfp8 段同时全绿。

### WFF-4：本人手动 eval ✅ 已完成（2026-09-27）

- 2026-09-27（logs/0923.4.fp4.eval.log，git 64c84d0+，本人手动跑）：
  | 运行 | git | wiki ppl | c4 ppl | wall | t_wiki | t_c4 |
  |---|---|---|---|---|---|---|
  | 折叠前基线（0923.3） | 64c84d0 | 7.7972 | 11.2708 | 110.53 | 51.83 | 58.70 |
  | **折叠后（0923.4）** | 64c84d0+ | **7.7972** | **11.2708** | **100.79** | **44.94** | **55.85** |
  - **精度：ppl 逐位一致**（wiki 7.7972 / c4 11.2708）——bit-exact 端到端
    复现，WFF-4 验收达标，本迭代闭环
  - 速度：wall **-9.8%**（110.53→100.79）；t_wiki **-13.3%**、t_c4 -4.9%
  - 模式排序变化（同 0923.3 批次）：**wxfp4 100.79 < wxa8 107.82
    < wxfp8 112.24 < wxa16 141.53**——折叠后 wxfp4 成为最快模式
    （vs wxa8 -6.5%、vs wxfp8 -10.2%），且 ppl 最优地位不变
  - 判读：wall 收益超 §一.4 的保守预期（"循环占 80%"的 P4 旧 profile
    低估了 fp4 路径 kernel 阶段的 wall 占比）→ kernel 时间在 wall 中
    仍占可观比例，**§三.5 挂起的 TMA/persistent grouped（367→~490 TF
    空间）立项条件已被数据支持**，是否排期待本人定

### WFF-5：4bpw 全 4bit 池下 OOM（2026-10-20 已修复落地）

- ~~驻留副本（本人已拍板：先不动，2026-10-19）~~ **2026-10-20 修复落地**：
  `quantization/wxfp4/bit_partitioned_moe.py::_FOLD_BYTES_BUDGET`（64MB/层/
  方向）——小池折叠（2bpw 的 bit1/4 长尾池，全模型 +47MB 不变）、大池跳过
  fold 走 kernel 内 LUT 分支（`W_FOLDED=False`，同一 LUT 同一映射，bit-exact
  A/B 验证 maxdiff 0）。4bpw wxfp4 首次可跑通（1018.thrput.6：pf2048 1567.3
  / dec 11.12 / check PASS，为「逐 expert + 未折叠 LUT」双重旧路径叠加数字，
  待 fp4 合并 kernel 接线后重测——合并版天然不需要 fold 副本）。

- 2026-10-19（logs/1018.thrput.1.w4.log，bench git 99336b8+）：
  4bpw checkpoint（global-a4s4m4bpw，256 expert 全部 bit4）上 wxfp4 两次
  运行均在首层 `_get_bit_context` → `fold_indices_to_e2m1` 报
  `CUDA out of memory. Tried to allocate 4.00 GiB`。风险表「bit1/4 池变大」
  条目兑现——§二 的 +47MB 上限以「小池」为前提，全 4bit 池下不成立：
  - **瞬时分配**：整矩阵一次物化 `idx.long()`（8B/元素）。已修：按 G 分块
    （~32M 元素/块，峰值 ~数百 MB），CPU 对拍 bit 1/2/4 与整矩阵路径
    bit-exact（test 见 throughput-bench-261019.md §二）。
  - **驻留副本（本人已拍板：先不动，2026-10-19）**：`indices_e2m1_gf` 在全
    4bit 池下是全量 MoE 索引副本（40 层 ctx 缓存累计 ≈ +10GB），31.4GB 卡装
    不下。候选留档：① 池大小预算内跳过 fold，kernel 走 legacy in-kernel LUT
    分支（`w_e2m1_gf=None`，分支仍在）——内存受限的 bit 级自适应；② 风险表
    原方案的 kernel 内算术化（w1 select / w2 where 链，bit4 无对应算术化
    需另行论证）。当前 4bpw 上 wxfp4 不跑，2bpw 路径不受影响。
- 详情：[roadmaps/throughput-bench-261019.md](roadmaps/throughput-bench-261019.md) §二
