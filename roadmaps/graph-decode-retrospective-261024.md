# CUDA Graph Decode 开发复盘（261023–261024）

## 〇、背景与目标

decode 阶段是 launch 密集型负载：每步数千次 host 端 kernel 发射，GPU 大量时间耗在 launch 队列而不是计算。CUDA graph 的原理是把一步的全部 kernel 录成 DAG（capture），之后每步一次提交重放（replay），host 开销从"每 kernel 一次"降为"每步一次"。

- 目标：官方 FP8 基线 16 tok/s → 60–80 tok/s。
- 结果先行：官方 FP8 达 **98.64 tok/s（6.0×，git c4e2831，双卡）**；我方量化模型
  四模式全落地（静态派发 + 空 expert 早退，261025），4bpw 18.5–20.1ms/步
  （45–54 tok/s 单卡）、2bpw 21.9–23.7ms；保真门两类分叉已定性（4bpw 状态
  漂移 / 2bpw 闭环 NaN），见 §二 问题 10/11 与 §四。

---

## 一、官方 FP8（双卡分段路径，已全通）

### 问题 1：device_map 与整步 capture 互斥

- **现象**：双卡 device_map 加载的模型，整步 capture 直接失败。
- **技术原因**：框架按 device_map 在模块边界做跨设备搬运，跨设备拷贝在录制期非法（录制期 kernel 不执行、无数据可搬）。
- **解决途径**：整步图改为分段图——前半层一张图（卡 0）、后半层+norm+lm_head 一张图（卡 1），段间状态搬运放在图外显式执行。附带教训：手写字典 device_map 时量化权重落卡不生效，需 auto + max_memory 加载后反推真实切分点。

### 问题 2：对拍软门的漂移误判风险

- **现象**：图路径与 wrapper eager 对拍，logits 出现 ~1.5 量级漂移，疑似正确性问题。
- **技术原因**：静态 KV cache 满槽对齐 vs 动态 cache 精确长度，产生偶发 1 ulp 差；GDN（线性注意力）prefill 状态吸收该 ulp 后，MoE top-8 路由在真平局处翻转，逐层放大。属于两条合法解码轨迹的分叉，不是图本身的错。
- **解决途径**：软门判据 = 漂移有界（阈值 3.0）+ 翻转处双方 top-2 gap 均为真平局（≤0.5）；另设硬门（图重放 vs 手动同路径逐步 bitwise）兜底正确性。两层门缺一不可：只看软门会漏真 bug，只看硬门无法解释合法分叉。

### 问题 3：rotary cos/sin 缓冲设备错位（根因一）

- **现象**：先后表现为 hooked capture 失败、无 hook capture 设备不匹配、更早一轮的"rotary 模块非法"误诊——同一根因三种表象。
- **技术原因**：加载框架给 rotary 挂的对齐 hook 按 dispatch 时记录的 execution_device 路由调用，而该设备与所在段的设备不一致，导致 cos/sin 缓冲生在另一张卡上。
- **解决途径**：rotary 绕过 hook 直连原始实现、在段内本卡计算，缓冲落位后显式搬运兜底。教训：hook 的路由目标在 dispatch 时固化，事后单独搬参数不改变路由。

### 问题 4：聚合报错掩盖真实失败点（调试方法论）

- **现象**：capture 结束时抛出的聚合错误只指向 capture_end，真实失败环节不可见，排查一度靠猜（产生问题 3 的误诊）。
- **技术原因**：CUDA 录制期错误在结束时刻统一上报，Python 异常链里最内层（__context__ 链尾）才是真实失败点。
- **解决途径**：异常链解包工具，逐层打印链尾 traceback 帧。加入后每轮定位一击必中。

### 问题 5：capture 流单例导致空图、relaxed 假成功（根因二）

- **现象**：第二张卡上的图在 relaxed 录制模式下"成功"，但重放输出恒定 token；global 模式则报空图。
- **技术原因**：torch 的默认 capture 流是类级懒单例，在首设备创建后被第二设备复用，第二设备的工作落在 legacy 默认流上、没有进图——得到空图。global 模式如实判死；relaxed 模式静默放过（假成功）。空图的签名 = 结束时空图警告 + 重放输出恒定。
- **解决途径**：每设备显式创建独立 capture 流传入。教训：relaxed 模式只用于定位、不作为放行依据；"录制成功"必须配合重放对拍才有意义。

### 问题 6：加载框架 hook 的双刃性

- **现象**：hook 是 capture 毒药（跨设备搬运进录制区），但尝试永久摘除后 wrapper 的正常前向又死（跨卡搬运依赖 hook）。
- **技术原因**：同一机制承担两个职责——推理路由与跨设备搬运。
- **解决途径**：只在录制窗口内临时摘除（保存原 forward 与状态，录制结束立即恢复），并保留三级重试（global → relaxed → 录制窗口摘 hook）作为保险。缓冲设备修好后一级直过，摘除路径未再启用。

### 结果

- capture 两段均一次通过（global 模式、hook 在位、零回退）；硬门 teacher-forced bitwise 全等 + 闭环 16 token 一致；10.1ms/步 = 98.64 tok/s（基线 60.8ms）。
- 对 eager 的逐步分叉出现在第 14 步，按问题 2 判据认定为良性平局翻转。
- 单卡大显存场景无需分段，整步图即可（分段是 device_map 的无奈拆分）。

---

## 二、我方量化模型（单卡整步路径，静态化已落地）

### 问题 7：不同 checkpoint 死点不同 → 全量审计而非逐个修

- **现象**：4bpw 死在 MoE 派发，2bpw 死在 attention 线性层——同一 forward 路径、不同死点。
- **技术原因**：层内 attention 先于 MoE 执行；2bpw 的 attention 走 A8 量化路径（带常量重算问题），4bpw 走 A16 路径（干净），所以先撞死点不同。
- **解决途径**：对 forward 全路径做 capture 非法点静态审计（DtoH 同步、输出形状随数据变的算子、host 驱动的控制流、录制期 H2D），出全量清单后分类处理。验证闭环：修掉常量类后 2bpw 死点如预测推进到与 4bpw 相同的 MoE 派发点——两路汇合，证明清单无遗漏，没有陷入"按下葫芦浮起瓢"。

### 问题 8：常量类非法点——每步重算的量化常量（已修）

- **技术原因**：A8 注意力线性层每步从码本张量反推量化步长，读取回 host 做分支判断与传参——每层每次前向一次 DtoH 同步，录制期非法，eager 期也是纯开销。
- **解决途径**：常量提升（加载/首次构建布局时一次算好，缓存为标量传入 kernel）。数学零变化。
- **附带收益**：2bpw eager decode 91.5 → 77.6 ms/步（−15.2%），仅此一项。
- **同类排除项**（有设备常驻缓存、录制安全）：分组旋转矩阵缓存、码本 buffer、占位码本。

### 问题 9：结构类非法点——MoE 派发的 host 编排（静态化已落地 261025）

- **技术原因**：MoE 前向的派发链是 host 驱动的：token 统计读回 host 得到每专家行偏移 → Python 字典按路由组织"bit 池" → 变长张量拼接 → kernel 的专家调度表在 host 构建后传设备。录制会把"捕获那一刻的路由"烤死进图，重放换路由即错。这不是 bug 而是 eager 时代的编排方式与图模式的根本冲突。
- **静态化设计**（三条支柱，kernel 零改动）：其一，合并 kernel 本身已是设备驱动——专家行段与块调度表运行时从设备张量读取，调度表可按最坏情况静态化（空专家的块全掩码安全退出）；其二，decode 每层总行数恒等于 top-k，只有各 bit 池间的分布动态——固定容量池缓冲 + 设备端统计（index_add）/池内游标（cumsum）/行内位次（定形比较算子）组装，padding 行置零、权重为零的 scatter 对合法 token 恒等加零；其三，gate_up 旋转在 decode 下无条件预提升（token 数为 1，数学等价），down 向逐专家旋转改为按 bit 全局预建旋转矩阵库 + 设备索引选取（依赖旋转 seed 跨层共享，实现时验证）。
- **测试设计要点**：对拍用例必须包含"录制用路由 A、重放换路由 B"——这是唯一能抓住烤死路由类错误的用例；另做静态路径 vs eager 容差对拍、图重放 vs 手动同路径 bitwise 硬门、SM 填充度分析。

---

### 问题 10：静态 GEMM 空块不早退——每步 2–4 TFLOP 废 MMA（已修，261025 实测）

- **现象**：四模式全模型 capture 通过但 graph 49–63ms/步（15.8–20.4 tok/s），
  远逊官方 10.1ms；模式间差异小且与权重字节量弱相关（4bpw≈2bpw）。
- **技术原因**：静态派发按最坏情况铺 grid（每池 E×N-tile），空 expert 的块仅被
  mask_b 掩 token 行，权重 load（w_mask 不含 mask_b）与 tl.dot 照跑——每层全
  256 expert 满 tile 计算，每步废 MMA ≈2–4 TFLOP（40 层 × E×(2·512·2048+
  2048·512)×2×BLOCK_B），小 tile + 码本 gather 有效吞吐仅 55–90 TFLOPS，MoE
  静态路径独占 40–50ms。host 编排路径成员恒非空，缺陷只在静态路径显形——
  故既有 eval/ppl 对它全然无感。
- **解决途径**：四个 merged kernel（a16/a8/fp8/fp4）读 expert_info 后
  `token_count <= 0` 整块早退（对齐 pooled rotate kernel 写法）。toy 四模式
  parity 仍 bitwise 0，merged/integration 回归全绿。
- **结果（8023c96+，logs/1023.graph.seg.5.ours.log，全表见 §四）**：全模式
  18.5–23.7ms/步（早退前 49–63，再 2.4–2.7×；vs eager 合计 3.3–4.0×）。
  **模式排名翻转**：早退前排名由空块废 flop 决定（fp4 BLOCK_B=32 废最少 →
  49ms 最快）；早退后由活跃 expert kernel 效率决定——量化三模式 BLOCK_N
  32/32/128 与 18.5/18.7/20.1ms 单调对应（decode 活跃 8 expert 的
  gate_up/down 块数 256/512 vs 64/128，5090 有 170 SM，fp4 大 tile 填充
  不足；权重字节减半在 B=1 非主导，活跃权重流仅 ~0.3–0.7ms/步）。2bpw
  三量化模式持平 21.9–22.0ms（bit≤2 池共享 int8 码本 kernel 族），wxa16
  2bpw 最慢 23.7（低 bit 混合池双格式激活链固定成本，与吞吐小节结论一致）。

### 问题 11：保真门两类分叉——4bpw 状态漂移与 2bpw 闭环 NaN（B2 诊断定性，261025）

- **现象**：4bpw 四模式 pass C parity 13.5–16.6（wxa16 闭环却 MATCH 全部
  1024 步）；2bpw 四模式闭环全部 DIFF@1/2，其中 wxfp4/wxfp8 pass C
  parity=0、wxa8/wxa16 pass C parity=nan。
- **技术原因（B2 闭环诊断 + NaN 捕获/prefill_tok_match 标记定性）**：
  - **4bpw = state divergence**：prefill parity 0.75–1.05（StaticCache vs
    DynamicCache 的 GDN prefill 状态差，与官方 FP8 DIFF@14 同机制），漂移
    随步数累积（早期步 ~1.3 → 1024 步 max 13–17），首个 argmax 翻转步因
    模式/实例而异（DIFF@2/12/19/∞）。wxa16 闭环 MATCH 说明漂移未翻转实际
    轨迹的 argmax——parity 大 ≠ token 流必分叉，两个口径各测各的（旧
    「parity 16.7 却 MATCH」矛盾由此解释）。官方「漂移有界 3.0」判据下
    我方 4bpw 超界（13–17），需 prefill 对齐从源头压缩。
  - **2bpw = 闭环 step1 logits 含 NaN（真 bug，非平局翻转）**：四模式签名
    完全一致（logits_diff=nan、graph_tok=0、eager_tok=279、eager 侧
    top-2 gap ~2.1 非平局）⇒ 与模式无关，指向低位池公共路径（bit≤2 池
    共享 int8 码本 kernel 族与同一 dispatch/rotate 链）。喂入 token 已被
    排除（wxa8/wxa16 prefill_tok_match=True，喂的就是 eager token，仍
    NaN，且其 pass C 同样 nan）；而 wxfp4/wxfp8 的 pass C bitwise 0 与
    B2 闭环 NaN 并存 ⇒ 图重放与静态 kernel 本身可精确保真，NaN 经特定
    prefill 状态实例进入（每次 static_prefill 独立执行，host 路径
    index_add_ 原子序非确定 → 1–2 ulp 不同的状态命中某个数据依赖边界）。
    host 路径 eval/ppl 从未出现该现象 ⇒ 边界在静态路径独有代码（候选：
    pooled rotate-quantize 的 amax=0 → scale 0/0、低位池 scale 转换、
    trash 槽读取），需逐层溯源。
- **解决途径（进行中）**：B2 已加 graph_logits_has_nan / prefill_tok_match /
  parity 计算不再被 `max(0.0, nan)` 吞 NaN（旧 cl_parity 0.0 实为 NaN 被吞；
  logs_parser 的 parity 正则同步支持 nan）。下一步逐层溯源：静态派发路径
  eager 化（DARTMOQ_STATIC_DISPATCH=1、不 capture）重试 prefill 直到复现
  NaN 实例，再逐层 isnan 二分定位层/池/kernel，最后查激活幅值与量化
  scale 的 0/0 或溢出边界。
- **对既有 ppl/eval 结论无影响**：eval 走 host 编排路径 + DynamicCache
  （DARTMOQ_STATIC_DISPATCH 仅 graph_decode 设置），与静态路径不相交；
  早退分支 host 路径不触发。修复前 2bpw 图路径只可测速、不可出文本。



## 三、方法论沉淀

1. **异常链解包**：capture 聚合报错只信链尾，不信表层 traceback。
2. **先全量审计再动手**：多死点场景先出非法点清单并分类（常量类/结构类），修复顺序按清单推进，死点汇合即闭环。
3. **常量提升与结构改造两分法**：每步重算的常量是"免费修复+免费提速"；数据依赖的 host 编排必须设备端重设计，两者不可混淆。
4. **双刃依赖的处理模式**：窗口化禁用+必复，不做永久摘除。
5. **假成功识别**：relaxed 录制通过≠成功；空图签名（结束警告+重放输出恒定）要当一等公民症状对待。
6. **双层正确性门**：硬门（bitwise，图 vs 手动同路径）保证录制无失真；软门（有界漂移+平局判定，图 vs 旧路径）区分真 bug 与合法分叉。图加速的正确性验证两者都要。

## 四、当前状态与下一步（261025 更新）

- 官方 FP8：全通收官，6.0×（98.64 tok/s）。
- 我方模型：静态化（问题 9）+ 空 expert 早退（问题 10）全落地，四模式 8 档
  全出数（graph_decode 口径 plen=128 / steps=1024 / batch=1 / 单卡，
  8023c96+，logs/1023.graph.seg.5.ours.log；单位 ms/步，parity 为 pass C
  teacher-forced max_abs，nan = logits 含 NaN）：

| ckpt | 模式 | eager | graph | replay | 加速 | tok/s | 闭环 tokens | parity |
|---|---|---|---|---|---|---|---|---|
| 4bpw | wxa8 | 73.8 | **18.5** | 18.4 | 3.99× | **54.01** | DIFF@19 | 14.23 |
| 4bpw | wxfp8 | 74.6 | 18.7 | 18.4 | 3.99× | 53.47 | DIFF@12 | 16.62 |
| 4bpw | wxa16 | 76.7 | 19.7 | 19.2 | 3.90× | 50.82 | MATCH | 13.47 |
| 4bpw | wxfp4 | 72.1 | 20.1 | 19.6 | 3.58× | 49.68 | DIFF@2 | 14.98 |
| 2bpw | wxa8 | 74.2 | 21.9 | 21.5 | 3.39× | 45.65 | DIFF@1 | nan |
| 2bpw | wxfp8 | 74.2 | 21.9 | 21.5 | 3.38× | 45.58 | DIFF@1 | 0 |
| 2bpw | wxfp4 | 76.6 | 22.0 | 21.4 | 3.48× | 45.43 | DIFF@1 | 0 |
| 2bpw | wxa16 | 78.5 | 23.7 | 23.0 | 3.31× | 42.12 | DIFF@2 | nan |

- **与官方 98.64 tok/s 的公平口径（修正）**：官方是双卡 device_map 分段
  （[0:18] cuda:0 / [18:40] cuda:1 + head），但闭环贪心每步
  `argmax().item()` 同步 ⇒ 两段串行、跨步无重叠——**第二张卡贡献显存
  容量而非吞吐**（FP8 35GB 单卡 32GB 放不下，双卡是容量所迫；与吞吐
  基准「串行切分、无算力红利」结论一致，98.64 不能按 ÷2 折算单卡）。
  公平对比 = 官方 10.1ms/步 vs 我方 18.5ms/步（wxa8 4bpw），**差 1.83×，
  未追平**；目标 10–12ms（83–100 tok/s，单卡追平乃至超过官方双卡总吞吐）。
- **下一步（顺序）**：① nsys 逐段拆 18.5ms（graph 内 kernel 全署名，一次
  看清 MoE 静态路径 / attention / lm_head / dispatch 链占比，避免拍脑袋
  排序）；② fp4 池 decode 档 tile 重扫（BLOCK_N 128→32/64）与空块数削减
  （decode 档 BLOCK_B 64/32→16 同方向）；③ dispatch 链融合（每层 ~25 个
  小 kernel × 40 层 ≈ 图内 1000 节点）；④ 2bpw NaN 逐层溯源（问题 11
  路线，修复前 2bpw 图路径只可测速）；⑤ 4bpw prefill 对齐（压漂移源头）。

## 关于混合精度静态图的优化特点：

静态图优化的不是混合精度本身，而是“每 token 一次的 host 派发开销”，而我们模型混合精度的特点恰好让这件事变得可行：

优化对象：decode 每 token、每层 MoE 都要做一次“路由分发”——现在的方式是 host 读回 token 统计（DtoH 同步，GPU 等着）→ Python 建池字典 → Python 循环切变长张量 → host 算好调度表再传回 GPU（H2D）。乘 40 层，每 token 数百次 host 操作 + 数十次同步/上传，这才是 decode 慢的大头之一。静态图把这一整套换成图内设备端定形算子，重放一次发射，host 操作归零。

混合精度的特点为什么反而是优势：每层的 (N4, N2, N1) 组合在量化时定死、推理期不变 → 每层有几个池、每池几个 expert、用什么 kernel 配置和旋转库，全是加载期常量，直接固化成该层的那段图节点。图不要求层同构——40 层各不相同毫无影响，各层录各的 kernel 链。这就是你上次问“每层都不一样怎么办”的答案：不一样恰恰意味着这些差异是常量，常量正适合烤进图。

真正动态的只剩路由：8 行 token 怎么分到 4/2/1-bit 池、分给哪个 expert——这变成 expert_info 缓冲里的设备端数据，kernel 运行时自己读。烤死的是形状和常量，活的是数据。

decode 行数恒等于 top_k 是整个方案成立的支点：容量上界固定 → 池缓冲、调度表、旋转库全都静态化，padding 行空转即可。