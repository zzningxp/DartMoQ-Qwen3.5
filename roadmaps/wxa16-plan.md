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


阶段 1 已经做完了，现在基于这个状态点，来做融合工作，
    首先明确，我们的目标是引入 Triton 融合 kernel 来替代"先反量化再做 GEMM"的两步走方案，理论上应该更快。conda 环境是 dart312，你后续可以自己测试。

    我们的基本思路是复用 triton_kernels.py 中的 kernel 实现。
    但是现在我不清楚这里的 kernel 实现细节，需要先写个测试程序来测试，放在 turboquant utils 里面。

    1）目前 里面只有 4bit 的实现，我们就测这个，你全程不要改动 triton_kernels.py 中的 kernel 实现。
    2）写一个矩阵乘法测试程序，输入 w1 是 1024 * 512 的 fp16 矩阵，量化为 q1 是 4bit 的矩阵，1024 * 512 ，128 group 一组的 turboquant packed 形式，带缩放因子和码本，输入 x 是 16 * 1024 的 fp16 矩阵。输出就是 16 * 512 的 fp16 矩阵。
    3）将上一个测试结果的输出，作为下一个输入 x，输入 q2 是另一个 4 bit 的矩阵（q2 是由同尺寸的 w2 量化获得的），尺寸都是 512 * 1024，输出就是 16 * 1024 的 fp16 矩阵。这里的两个 w1 和 w2 是两个形状转置的矩阵，但是都是按行量化成 q1 和 q2 的，都是 128 group。
    4）以上矩阵尺寸要可配置测试更多情况。这里主要是要测试速度，尤其是要测试出来他确实是一次发生并行执行的。有三个测试方法：i）原本的 w1 ，w2， 和原本 x 他们正常流程做矩阵乘，ii）反量化，然后 fp16 运算的，iii）triton kernel 实现。速度要更快。
    之前面临的问题，triton_fused_matmul kernel 不支持分组量化（group_size=128），当 group_size=128 时，我们需要对每个分组分别调用 triton_fused_matmul，然后把结果累加。
    目前已经实现好了一个正确的 triton_fused_matmul_grouped 函数，串行调用分组量化的情况。
    f93f2c5, DONE.
    
    好，/home/daodao/Develop/DartMoQ-Qwen3.5/turboquant_utils/test_triton_simple.py 中已经写好了，一个 fp16 的输入 x 矩阵（batchsize * hidden size），与 fp16 的 w1、w2 的量化版本 q1、q2 分别进行乘法的调用方法。相当于已经有一个 "码本查找 + 逆缩放 + 逆旋转" 等完整流程了。
    1）目前这里面只有 4bit 的实现，而我们的 wxa16 是要支持 1bit 2bit 4bit 8bit 的，所以需要先实现一个 nbit 来根据输入实现的。实现细节应该是，4bit 就是两个 4bit packed 成一个 8bit 然后通过位运算来进行unpack，8bit 就是直接 unpack。1bit 是 8 个 1bit packed 成一个 8bit。2bit 是 4 个 2bit packed 成一个 8bit。中间都要通过位运算来并行执行。
    2）更新 test_triton_simple.py 的方法，按 4bit 的方法支持 4 个 bit 位的测试。

    
