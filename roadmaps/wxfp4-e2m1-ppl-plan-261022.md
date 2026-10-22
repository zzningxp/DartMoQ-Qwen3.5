# WxFP4 e2m1 激活 ppl 代价研究路线（2026-10-22）

> **定位**：4bpw（bit 池 100% bit4）下 wxfp4 模式 ppl 比同池三模式高
> +0.19~+0.23，turboquant/gptq 双 W 源复现、加性 → **e2m1 激活格式固有代价**。
> 本路线不追速度（速度线归 P5-2/P8），只回收这笔质量税：目标 = wxfp4 在 4bpw
> 「吞吐登顶 且 ppl 与 wxa8/wxfp8 同簇」。
> **来源**：logs/1020.1.evalppl.all.log（git c87966f / c87966f+，2026-10-22），
> 同日 12 组 eval（2bpw-s1 ×4 + 4bpw-turboquant ×4 + 4bpw-gptq ×4）首次给出
> 4bpw 全量 ppl。数字已录 README「性能」节 4bpw 小节。

---

## 一、背景与证据链

1. **ppl 偏离（4bpw，wiki 145 / c4 256 samples，sequential bs=16）**：

   | ckpt | WxA16 | WxA8 | WxFP8 | **WxFP4** | WxFP4 增量 |
   |---|---|---|---|---|---|
   | turboquant wiki | 6.8037 | 6.8070 | 6.7959 | **6.9909** | +0.19 |
   | turboquant c4 | 9.9446 | 9.9469 | 9.9606 | **10.1548** | +0.21 |
   | gptq wiki | 7.0322 | 7.0342 | 7.0197 | **7.2582** | +0.22 |
   | gptq c4 | 10.1355 | 10.1348 | 10.1482 | **10.3541** | +0.22 |

   三模式（a16/a8/fp8 激活）聚在彼此 ±0.01 内，唯独 e2m1 激活独高一档。

2. **误差分解（vs fp16 基线 6.5852 / 9.6725）——加性、与 W 源无关**：

   | ckpt | W-only（wxa16） | 加 e2m1（wxfp4） | e2m1 增量 |
   |---|---|---|---|
   | turboquant wiki | +0.219 | +0.406 | **+0.187** |
   | gptq wiki | +0.447 | +0.673 | **+0.224** |

   e2m1 一项 ≈ 整个 4bit W 量化的代价再付一遍；W 质量变差（gptq）增量不涨
   → 不是「W 误差与 A 误差交互」，是格式固有、校准方法换不掉（turboquant 与
   gptq 已是两种极端不同的 W 校准）。

3. **bit 池覆盖面（meta.json 统计，2026-10-22 实测）——2bpw 不显形的真因**：

   | checkpoint | bit1 | bit2 | bit4 | wxfp4 模式下 e2m1 覆盖面 |
   |---|---|---|---|---|
   | 2bpw-260930-s1 | 1.4% | **97.9%** | 0.7% | **2.1%**（98% expert 走 WxA8 int8 激活） |
   | 4bpw-261016-tq | — | — | **100%** | 全部 MoE 计算 |
   | 4bpw-261016-gptq | — | — | **100%** | 全部 MoE 计算 |

   修正早期单一「W 误差掩蔽」解释：2bpw 四模式 ppl 持平的主因是 **bit2 池根本
   不走 e2m1**（分派设计使然），掩蔽是次因。

4. **relerr 量级对照**：e2m1 网格 {0,±0.5,±1,±1.5,±2,±3,±4,±6} 仅 1 位尾数，
   worst relerr ~20–33%（[0.5,1) 区间 33%、[4,6] 区间 20%）；对照 e4m3 实测
   2.57%（wxfp8 计划时代研究，ppl 代价≈0）、int8 ~0.4%（wxa8 代价≈0）。
   **疑点（待 EA-1 解释）：relerr 只有 e4m3 的 3~4 倍，ppl 代价却 ~20 倍**——
   误差逐层传播/累积的放大机制本身是研究对象。

5. **锁死约束**：`tl.dot_scaled` / FP4 MMA 要求 A=e2m1 + per-32 e8m0（2 的幂），
   不能加尾数位、不能改组大小 → 缓解只能落在 **scale 选择、舍入、分派、W 侧
   补偿** 四处（下节任务）。

## 二、量化链现状（代码锚点）

- 参考实现 `turboquant_utils/triton_kernels_fp4.py` ~L126（`quantize_act` 参考）：
  per-32 amax → `e = floor(log2(amax/E2M1_MAX))` → 2 的幂 scale → RTN + clamp。
- 运行时量化（`rotate_quantize_fused_fp4` 每 forward 执行），**W 侧与 checkpoint
  均不动**——scale 取法与舍入是两个显式可改点，改动只落在激活量化一处。
- e8m0 限 2 的幂 + floor 取法：组内动态范围利用最多差 √2（等效 1 bit 增益
  空间），是「可修部分」的上界候选。

## 三、任务分解（EA 系列）

### EA-1 relerr 实测与归因（测量先行；kernel/test 级自测）

- **目标**：真实 35B 激活上量化 relerr 实测 + 把 +0.2 ppl 拆成
  「网格粒度（不可修）」vs「scale-floor + RTN（可修）」两块占比。
- **方法**：仿 wxfp8 的 2.57% 研究口径，真激活统计；对照组：
  现行 e2m1 / e2m1+MSE 最优 2 的幂 scale / e2m1+假想 fp16 scale（不可实现，
  仅作上界）；× 有无旋转 × gate_up vs down 输入分布；顺带逐层 relerr 曲线
  （喂 EA-3 敏感层定位 + 解释 §一.4 的放大现象）。
- **判据**：可修占比 ≥1/3 → EA-2 值得做；逐层 relerr 方差大 → EA-3 分派收益高。
- **成本**：低（test 级自测，不动主流程）。

### EA-2 scale 选择与舍入策略（零 checkpoint 重出）

- **内容**：floor → 逐组 MSE 最优的 2 的幂（候选仅 1–2 个，逐组试选）；
  RTN → stochastic rounding（对称网格下通常省 5–15% 误差）。
- **改动面**：仅激活量化 kernel/包装（`rotate_quantize_fused_fp4` 及参考实现），
  W 侧、checkpoint、GEMM kernel 全不动；对拍判据从 bit-exact 换成 relerr/ppl。
- **成本**：低–中（含一次 ppl 复跑验证）。

### EA-3 敏感层 A 位宽分派（结构性解）

- **依据**：`tl.dot_scaled` 允许 lhs/rhs 格式分别指定——**W 保 e2m1 不丢 FP4
  MMA 速度，仅敏感层/池的 A 升 e4m3**（W=e2m1 + A=e4m3 的混合格式组合需先
  验证 tcgen05 双格式路径吞吐不掉，这是前置判据）。
- **方法**：逐层 ablation eval 定位敏感层（40 层，二分或全扫，每层一次 ppl）；
  假设：少数层贡献大头（router 对激活扰动敏感、误差逐层放大不均匀）。
- **成本**：中；复用现有逐层/逐 expert 分派机制。

### EA-4 A-error-aware W 校准（科研增量最大，放最后）

- **内容**：把「A 将被 e2m1 量化」纳入校准目标——最优 W 不再是逼近原 W，
  而是在 A 量化后的 GEMM 里逼近原输出。turboquant / gptq 两条 W 线都兼容。
- **成本**：高（改校准流程 + 重出 checkpoint，与 checkpoint 版本管理衔接）。

## 四、附带发现与口径注意

- **4bpw 质量侧 turboquant 明显占优**：全模式 ppl 比 gptq 好 +0.22（wiki）/
  +0.19（c4），两份 bit 池结构相同（100% bit4）→ 纯量化算法差异；gptq eval
  快 3–6% 疑后跑 JIT 热缓存（turboquant 先跑付编译），**速度口径以吞吐表为准
  （两者 ≤0.6% 持平），勿用本日志比速度**。
- bit 分布统计口径：`meta.json['layers']` 各 MoE 层 `inter_size_by_bit[bit] ÷
  intermediate_size` = 该 bit expert 数（2026-10-22 验证可用，3 份 checkpoint）。
- 4bpw eval 比 2bpw 同模式慢 9–38%：bit 池更宽、每层搬运字节与 kernel 负载
  更重（已录 README）。

## 五、与其他路线的关系

- 与 P5-2（合并 kernel）、P8（attention）速度线**并行不冲突**；EA-1 方法论同
  P8-1（测量先行）。
- `wxfp4-plan-260924`（保守版落地）、`wxfp4-fold-plan-260927`（W 侧 fold）
  均只动 W 形态且 bit-exact，与本路线（激活侧）正交，可叠加。
- 触发条件：本人批准后 EA-1 先行（GPU 空闲时 kernel 级自测）；EA-2 依赖
  EA-1 归因结论；EA-3 前置验证混合 dot_scaled 吞吐；EA-4 排最后。
- 状态（2026-10-22）：EA-1 ~ EA-4 均未动工。
