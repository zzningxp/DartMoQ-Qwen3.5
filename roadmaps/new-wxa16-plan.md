进一步速度优化方向
基于对所有分支和当前实现的分析，以下是按优先级排序的优化方向：

1. cuTile 路径集成与对比
问题：cutile_kernels.py 已经实现了但没有集成到主推理流程
方案：把 WxA16Linear 和 MoE 的推理路径加上 cuTile 后端选项，实测对比 Triton vs cuTile 的性能
预期收益：cuTile 的 MMA 可能比 Triton 的 tl.dot 更高效（尤其是 TF32/FP16 Tensor Core 利用率）

⭐ 高优先级（预期收益大，实现难度适中）
1. Group 级融合：一个 kernel 处理所有 group
问题：main 分支中 num_groups 次 kernel launch（如 hidden_size=4096, group_size=128 → 32 次 launch），launch 开销 + 多次写入/读取中间结果
方案：参考 triton-temp 分支的 3D grid 思路，但改为 单 kernel 内循环处理所有 group 并累加（而不是输出 (B,N,G) 再 sum）
关键：旋转矩阵的计算也要融合进 kernel（用 Hadamard 旋转替代 QR，因为 Hadamard 可以在寄存器内高效计算）
预期收益：kernel launch 减少 30×+，中间结果显存大幅减少
2. MoE 的 grouped GEMM 化：同一 bit 的所有 expert 一次 kernel
问题：main 分支逐 expert 循环，每个 expert 单独 launch kernel；expert 数量多但每个 expert 的 token 少时，GPU 利用率低
方案：参考 false-grouped 的 gather/scatter + bad-triton 的 3D grid 思路，但用 grouped GEMM 方式：
把同一 bit 的所有 expert 权重打包好（已经是 bit-partitioned 布局）
一个 kernel 内用 expert_info_ptr 索引各 expert 的 token_start/token_count/weight_range
关键是 每个 SM 动态分配给某个 expert，避免负载不均衡
预期收益：MoE 部分 kernel launch 从 O(experts × bits × groups) 降到 O(bits × groups)
3. Gate-Up 全融合：dequant + matmul + silu + mul
问题：gate_up 的 matmul 结果写出到显存，然后读回来做 silu + 逐元素乘
方案：参考 moe-whole-kernel-very-slow 分支的 _clustered_fused_up_gate_kernel，把 silu(gate) * up 融合到 kernel epilogue 中
关键：不要像 moe-whole-kernel 那样做整层融合，只做 gate_up 的 epilogue fusion；保持 kernel 简洁
预期收益：减少一次中间结果 (M, 2I) 的读写，约省 20-30% gate_up 部分时间
4. Triton autotune 或手动调优 block size
问题：当前所有 kernel 的 BLOCK_B=16, BLOCK_N=64, BLOCK_K=64 是硬编码的，对不同形状（如 8-bit vs 2-bit，大 batch vs 小 batch）未必最优
方案：针对 MoE 推理中常见的形状（如 B=32256 tokens, N=变化的 expert 神经元数, K=128(group_size)），手动测试几组配置并选择最优；或者用 Triton 的 autotune 但限制搜索空间（510 个候选）
注意：commit eff97a3 说"remove @triton.autotune, better speed"，是因为 autotune 首次调用太慢。可以做离线调优 + 硬编码最优配置
预期收益：5-20% 取决于形状
⭐⭐ 中优先级（收益中等或实现难度较高）
5. 旋转融合进 kernel（用 Hadamard 矩阵）
问题：当前旋转在 Python 端用 PyTorch matmul 完成，需要额外 kernel launch 和中间结果显存
方案：如果 rotation == "hadamard"，可以把 Hadamard 变换融合进 Triton kernel（Hadamard 是 butterfly 结构，可以在寄存器内高效计算）
注意：需要确认量化精度是否受旋转方式影响
预期收益：减少 group 数量级的 PyTorch matmul 开销
6. Split-K 优化
问题：当 M 很小（如单个 expert 只有几十 token）但 K 很大时，GEMM 的并行度不够
方案：对 K 维度做 split-K，多个 block 计算同一输出 tile 的不同 K 段，最后原子加
适用场景：MoE 中 token 数少的 expert
预期收益：小 batch 场景的 GPU 利用率提升
7. FP16 输出 + FP16 激活
问题：当前 kernel 输出固定 float32（output = torch.empty(B, N, dtype=torch.float32)），但激活实际上是 fp16/bf16
方案：让 kernel 直接输出 fp16，累加器仍用 fp32；输入也用 fp16
预期收益：减少输出带宽 50%；tl.dot 用 fp16 Tensor Core 可能更快
8. Bit-packing 布局优化：按 32-bit / 128-bit 对齐
问题：当前 uint8 逐字节加载，对 memory coalescing 不是最优
方案：把 packed indices 按 32-bit（或 128-bit vector）存储和加载，一次加载多个字节
预期收益：内存访问效率提升
⭐⭐⭐ 探索性方向（高风险高回报）
9. Flash Attention 风格的在线反量化 Attention
问题：Attention 的 Q/K/V 投影层被量化了，但 attention 计算本身是全精度的
方案：如果把 K/V cache 也做量化（TurboQuant 风格），在 Flash Attention kernel 内部实时反量化，可大幅减少 KV cache 带宽
难度：很高，需要深度修改 Flash Attention kernel

10. 权重预排序：让 codebook lookup 更友好
问题：码本查找是 gather 操作，虽然 codebook 小但索引是随机的
方案：按激活数值排序权重行/列，使相邻位置访问相近的码本条目，提升 L1 缓存命中率
难度：需要配合量化算法一起改


六、各分支的经验教训总结
3D grid 按 expert 并行（bad-triton）不行：因为不同 expert 的 token 数差异大（MoE 的固有特性），导致严重的负载不均衡，边界检查开销也大

整层 MoE kernel（moe-whole-kernel-very-slow）不行：kernel 太复杂（gate_up + silu + down 全融合），Triton 编译器优化效果差，寄存器压力大

3D grid 按 group 并行（triton-temp）有潜力但显存爆炸：输出 (B, N, num_groups) 的中间结果显存太大。正确做法应该是单 kernel 内循环处理所有 group 并直接累加，而不是输出 3D tensor

Shared memory codebook（shared-mem-codebook）可能无收益：码本只有 2^bit_width 个 float32（最大 256 个 = 1KB），Triton 的 L1 cache 已经能很好地缓存，手动管理 shared mem 反而增加 barrier 开销

Gather/scatter 策略（false-grouped）思路正确但需要精化："每个 group 一次 kernel 处理所有 token" 是对的，但不应该算所有行然后 scatter，而应该用 expert_info 指针方式，只计算需要的行（类似 grouped GEMM）

去掉 autotune 是对的（commit eff97a3）：autotune 首次调用开销太大，对于 MoE 这种动态形状场景不友好。但应该做离线调优后硬编码最优配置

七、推荐的迭代路线
如果要继续优化，建议按以下顺序推进：

第一步：先做 group 级融合（单 kernel 处理所有 group），这是最大的 launch 开销来源
第二步：做 gate_up epilogue fusion（silu + mul 融合到 kernel 尾部），减少中间显存读写
第三步：做 MoE grouped GEMM 化（同一 bit 的所有 expert 一次 kernel），进一步减少 launch 并提升 GPU 利用率
第四步：手动调优 block size + num_warps + num_stages
第五步：探索 FP16 输入/输出 + Hadamard 旋转融合