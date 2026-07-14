好，现在开始做一件大事。
现在的代码是这样的流程：
每一层，前向 attention 和 shared expert ，到 router expert 部分，计算 inner product 得到 loss （GPTQ 模式下就是 mse）根据 loss 进行排序并组织原始 weights，获得 scheme，这里不能叫 fake quant，这里是 simulation quant，这里是不产生 weights 变化的。然后根据 scheme 进行指定量化，以前是量化完，马上根据量化结果反量化回去，存在内存里面。所有需要量化的层量化好之后，进行 ppl test 前向推理。因为这时已经量化的参数和没有量化的层的参数，都是 fp16 就在原来实现的 类里面直接进行前向推理就行了。
但是这时不行的，这样的话量化过程只是一个 fake quant 的过程，只是一个模拟的过程，我们现在要用实际的方法进行改进。这里我们起名叫做 wxa16 方法，也就是weights 是 x bit 的 int，在这里主要是 1、2、4、8 bit，3bit 因为不好 packed 所以这里不用了。8bit 主要用于 attention 和 shared expert 部分。

好，我先说一下我期望的流程：
每一层在进入实际量化阶段的时候，attention 和 shared expert 会被量化为 int8 可以采用 w8a16 方法。router expert ，和以前一样，计算 inner product 得到 loss 根据 loss 进行排序并组织原始 weights，获得 scheme。这里是不产生 weights 变化的。然后根据 scheme 进行指定量化。好这里的量化过程是最大的变化，现在是把量化结果（packed weights，量化 scale，codebook等）存下来放在内存了，不反量化回去，同时删除原来的 fp16 weights（attention 和 shared expert 同理，这里就是根据 scheme 指定 1、2、4 bit 进行 w1a16，w2a16，w4a16 的量化存储方法）。然后是根据量化结果前向得到这层输出结果 hidden state 。这和 ppl test 时候一样了。后边一起说。

然后就是推理时候 ppl test eval 时候，这时候存的量化结果再现场进行反量化，得到 fp16 再算结果。整理来说相当于就是把原来前面量化时候的反量化放到了推理前向时候再进行反量化，如果把这两部分拆开的话，中间传递的数据则是 fp16 和 wx 的区别，后者压缩效果显著。

量化的时候，还需要同步输出日志，显示，量化前的存储空间占比和量化后的存储空间占比，attention 一行，up/gate/down 分开 shared expert 一行，router expert 1、2、4 按占比汇总一个空间变化情况。
具体前向推理的时候，也要在可以插入日志的地方进行日志输出，记得加 flush=True 参数。主要是在比较费时的位置，来确定是在什么时候跑到什么地方了。


然后就遇到最困难的地方了，如果我们在前向时候用最简单的反量化+计算设计的话，会非常慢，这是因为其实当前主流的量化前向推理方法，都是将反量化算子和后续的 gemm 融为一体统计进行大矩阵并行计算的。哪怕不进行融合算子，也会通过等价公式将反量化过程变成一个并行算子进行，这里还没有完全设计好，我觉得你可以进行一定的发挥。

59e56412e：已经处理好前面的基本逻辑，下一步就开始优化反量化+计算设计的加速了。
阶段1：先不要搞太复杂，先优化 "反量化" 函数本身
先看看当前 turboquant_dequantize_packed_rows 和 turboquant_dequantize_packed_cols 有没有可以优化的微观层面：

有没有不必要的 tensor 创建/拷贝？
有没有可以原地操作的？
有没有可以用 view 而不是 copy 的？
阶段2：然后再做部分融合
比如把"码本查找 + 逆缩放 + 逆旋转"这几步融合成一个 kernel，避免中间结果。

阶段3：最后再做完全融合
把反量化和 GEMM 完全融合在一起。

阶段1：微观优化点分析

        🔴 优化点1：旋转矩阵重复生成（最大的问题！）
        当前问题：


        在反量化循环里，每次都重新生成！
        for g_start in range(0, N, group_size):
            ...
            Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(device)
            W_g_approx = Y_unscaled @ Pi
        而旋转矩阵 Pi 只和 g_dim 和 seed 有关，完全可以缓存！

        而且我们有768次反量化调用（256 experts × 3 bits），每次都要生成 N/group_size 个旋转矩阵... 这太浪费了！

        优化方案：

        方案A：在 WxA16BitPartitionedGroupMoE.from_build_block 阶段，就把所有需要的旋转矩阵预生成好，存起来
        方案B：实现一个简单的 LRU 缓存
        🟡 优化点2：Python 循环开销
        当前问题：


        for g_start in range(0, N, group_size):  # Python 循环
            ...
        如果 group_size 小（比如 128），这个循环次数会很多。

        优化思路：看看能不能用 torch 的向量化操作，或者把一些步骤移出循环。

        🟢 优化点3：小的优化点
        dtype 转换：当前先 float32，最后再转 orig_dtype。能不能直接用 orig_dtype？
        码本查找：codebook[indices_g] 这个操作本身很快，但看看能不能和后续的操作融合
        不必要的 tensor 创建：W_approx = torch.zeros(...) 能不能直接原地写入？