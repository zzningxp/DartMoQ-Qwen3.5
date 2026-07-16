好，现在开始做一件大事。
现在的代码是这样的流程：
每一层，前向 attention 和 shared expert ，到 router expert 部分，计算 inner product 得到 loss （GPTQ 模式下就是 mse）根据 loss 进行排序并组织原始 weights，获得 scheme，这里不能叫 fake quant，这里是 simulation quant，这里是不产生 weights 变化的。然后根据 scheme 进行指定量化，以前是量化完，马上根据量化结果反量化回去，存在内存里面。所有需要量化的层量化好之后，进行 ppl test 前向推理。因为这时已经量化的参数和没有量化的层的参数，都是 fp16 就在原来实现的 类里面直接进行前向推理就行了。
但是这时不行的，这样的话量化过程只是一个 fake quant 的过程，只是一个模拟的过程，我们现在要用实际的方法进行改进。这里我们起名叫做 wxa16 方法，也就是weights 是 x bit 的 int，在这里主要是 1、2、4、8 bit，3bit 因为不好 packed 所以这里不用了。8bit 主要用于 attention 和 shared expert 部分。

好，我先说一下我期望的流程：
每一层在进入实际量化阶段的时候，attention 和 shared expert 会被量化为 int8 可以采用 w8a16 方法。router expert ，和以前一样，计算 inner product 得到 loss 根据 loss 进行排序并组织原始 weights，获得 scheme。这里是不产生 weights 变化的。然后根据 scheme 进行指定量化。好这里的量化过程是最大的变化，现在是把量化结果（packed weights，量化 scale，codebook等）存下来放在内存了，不反量化回去，同时删除原来的 fp16 weights（attention 和 shared expert 同理，这里就是根据 scheme 指定 1、2、4 bit 进行 w1a16，w2a16，w4a16 的量化存储方法）。然后是根据量化结果前向得到这层输出结果 hidden state 。这和 ppl test 时候一样了。后边一起说。

然后就是推理时候 ppl test eval 时候，这时候存的量化结果再现场进行反量化，得到 fp16 再算结果。整理来说相当于就是把原来前面量化时候的反量化放到了推理前向时候再进行反量化，如果把这两部分拆开的话，中间传递的数据则是 fp16 和 wx 的区别，后者压缩效果显著。

量化的时候，还需要同步输出日志，显示，量化前的存储空间占比和量化后的存储空间占比，attention 一行，up/gate/down 分开 shared expert 一行，router expert 1、2、4 按占比汇总一个空间变化情况。
具体前向推理的时候，也要在可以插入日志的地方进行日志输出，记得加 flush=True 参数。主要是在比较费时的位置，来确定是在什么时候跑到什么地方了。



59e56412e：已经处理好前面的基本逻辑，下一步就开始优化反量化+计算设计的加速了。

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
    7fe9fa9, DONE.
    
我们要再更新一版 test 程序。主要目的是测试混合精度时候的 down 。
   
    现在还有个新问题，test_triton_simple.py 中的 w2 是简单的情况，但是我们实际项目中的 moe down 是要分组量化成不同精度的，down shape: (out_features=H, in_features=sub_set_neurons)，这里有两个问题：第一，量化分组时候的朝向是按行还是按列？第二，输入 size 问题。
    首先，我们本来就是分组量化时，量化的分组是在神经元这一维度上的，所以不会有同一个分组量化跨越神经元分组。
    其次，输入原来如果是 batchsize * intermedia size，现在要分组量化，输入 size 就是 batchsize * sub_set_neurons。输出 size 就是 batchsize * out_features。最后多个 out 直接数值相加，多个相同尺寸的 out 变成一个原尺寸的 out。
    我觉得这块挺清晰的逻辑。
    然后 test 中需要增加对应的测试项目：1）按 2:1:1 切三份的方式完成前面描述的方式，都是 4bit。2）按 2:1:1 切三份方式，使用 4bit：2bit：1bit 再测试一遍。
    9b3f04d, DONE. test_triton_mixed_precision.py



    好，我们现在完成实际量化推理工作，首先明确，我们的目标是引入 Triton 融合 kernel 来替代当前推理时"先反量化再做 GEMM"的两步走方案，速度应该会更快。conda 环境是 dart312，你后续可以自己测试。

    我们的基本思路是复用 triton_kernels.py 中的 kernel 实现。
    目前已经实现好了一个正确的 triton_fused_matmul_grouped 函数，串行调用分组量化的情况。
    /home/daodao/Develop/DartMoQ-Qwen3.5/turboquant_utils/test_triton_simple.py 中已经写好了，一个 fp16 的输入 x 矩阵（batchsize * hidden size），与 fp16 的 w1、w2 的量化后矩阵 q1、q2 （带量化参数）分别进行乘法的调用方法，支持 1bit、2bit、4bit、8bit。相当于已经有一个 "码本查找 + 逆缩放 + 逆旋转" 等完整流程了。并已经完成了基本的小矩阵乘法验证，有轻微性能损耗暂不深究。
    
    好，当前就是把我们实际上进行的量化完之后进行 forward 的过程换成 kernel 方法，弄完之后现在的分步反量化再 GEMM 就可以扔掉了。
    attention 和 shared expert 部分和也要量化成 8bit （现在的 scheme 一般都是 8bit）
    我认为可能存在的问题，测试程序的 w1、w2 分别对应 up/gate 和 down 专家的参数矩阵，你可以照猫画虎套进去。因为我们是混合精度方法呢，一个专家下面应该是有 3 套方案分别有三组矩阵，这时候就不要再拆分他们了。然后测试程序里面的码本、缩放和 packed 方法，最好不要动，如果不合适，可能需要改量化过程中的 packed 和量化过程。。让前续方法来适配调好的 kernel 。如果没有冲突那最好了。
    注意，你可能会遇到 down proj 因为 in_features 维度的切片需要更复杂的处理的问题，这里我按 down 的方式完成了 test_triton_mixed_precision.py 中的 w2 的测试
    还需要在项目中增加丰富的日志，方便调试和分析。
    eff97a3，DONE. 

| git      | model            | sli | qsch              | rank                     | qmode      | qlayers | wiki   | c4     | status | time   | t_quant | t_ppl  | t_wiki | t_c4   | err |
|----------|------------------|-----|-------------------|--------------------------|------------|---------|--------|--------|--------|--------|--------|---------|--------|--------|--------|
| fc2eec0  | Qwen3.5-35B-A3B  | 4   | global-a8s8m2bpw  | turboquant_innerproduct  | turboquant | [0]     | 6.594  | 9.6828 | ok     | 489.53 | 223.26  | 266.27 | 117.56 | 148.71 |     |
| eff97a3  | Qwen3.5-35B-A3B  | 4   | global-a8s8m2bpw  | turboquant_innerproduct  | turboquant | [0]     | 6.5933 | 9.6822 | ok     | 423.61 | 212.73  | 210.88 | 91.62  | 119.26 |     |  


  [Layer 0] time: 9.24s (move: 0.02s, forward: 9.20s), mlp_type=WxA16BitPartitionedGroupMoE | Memory: CUDA 0: 4.30GB/6.70GB | CUDA 1: 0.00GB/0.00GB | CPU: 8.44GB
  [Layer 1] time: 1.86s (move: 0.41s, forward: 1.07s), mlp_type=Qwen3_5MoeSparseMoeBlock | Memory: CUDA 0: 4.30GB/30.59GB | CUDA 1: 0.00GB/0.00GB | CPU: 11.01GB

  [Layer 0] time: 8.57s (move: 0.02s, forward: 8.52s), mlp_type=WxA16BitPartitionedGroupMoE | Memory: CUDA 0: 5.16GB/7.62GB | CUDA 1: 0.00GB/0.00GB | CPU: 15.98GB
  [Layer 1] time: 7.87s (move: 0.02s, forward: 7.83s), mlp_type=WxA16BitPartitionedGroupMoE | Memory: CUDA 0: 5.16GB/7.58GB | CUDA 1: 0.00GB/0.00GB | CPU: 15.98GB
  [Layer 2] time: 7.73s (move: 0.05s, forward: 7.65s), mlp_type=WxA16BitPartitionedGroupMoE | Memory: CUDA 0: 5.16GB/7.56GB | CUDA 1: 0.00GB/0.00GB | CPU: 15.98GB
  [Layer 3] time: 10.63s (move: 0.02s, forward: 10.59s), mlp_type=WxA16BitPartitionedGroupMoE | Memory: CUDA 0: 5.16GB/7.58GB | CUDA 1: 0.00GB/0.00GB | CPU: 15.98GB
  [Layer 4] time: 8.40s (move: 0.05s, forward: 8.33s), mlp_type=WxA16BitPartitionedGroupMoE | Memory: CUDA 0: 5.16GB/7.57GB | CUDA 1: 0.00GB/0.00GB | CPU: 15.98GB
  [Layer 5] time: 1.67s (move: 0.15s, forward: 1.14s), mlp_type=Qwen3_5MoeSparseMoeBlock | Memory: CUDA 0: 5.16GB/29.48GB | CUDA 1: 0.00GB/0.00GB | CPU: 18.56GB
  [Layer 6] time: 1.78s (move: 0.15s, forward: 1.12s), mlp_type=Qwen3_5MoeSparseMoeBlock | Memory: CUDA 0: 5.16GB/29.48GB | CUDA 1: 0.00GB/0.00GB | CPU: 21.62GB


不过这里 9.2s 还是挺慢的。。

## 精度问题：
并且现在精度还有下降！
积累到 40 层的话 ppl 已经压不住了，，

| git      | model            | sli | qsch              | rank                     | qmode      | qlayers | wiki    | c4      | status | time    | t_quant  | t_ppl   | t_wiki | t_c4   | err |
|----------|------------------|-----|-------------------|--------------------------|------------|---------|---------|---------|---------|--------|---------|----------|---------|--------|--------|
| aac6342  | Qwen3.5-35B-A3B  | 4   | global-a8s8m2bpw  | turboquant_innerproduct  | turboquant | all     | 7.6864  | 11.2656 | ok      | 8654.59 | 8253.99  | 400.6   | 154.49 | 246.11 |     |
| efb4122  | Qwen3.5-35B-A3B  | 4   | global-a8s8m2bpw  | turboquant_innerproduct  | turboquant | all     | 11.2955 | 15.7725 | ok      | 8456.37 | 7493.08  | 963.29  | 358.03 | 605.26 |     |  

<!-- 1. 单独 Linear (triton_fused_matmul_grouped):
  FP16:                 0.01 ms
  反量化+GEMM:          0.70 ms
  Triton:               0.84 ms

  误差对比:
    Triton vs FP16: max=19.556366, mean=3.470817
    反量化+GEMM vs Triton: max=0.053061, mean=0.010133

2.1 MoE up_gate (triton_fused_matmul_grouped_slice_rows):
  FP16:                 0.01 ms
  反量化+GEMM:          1.62 ms
  Triton:               3.40 ms

  误差对比:
    Triton vs FP16: max=104.514130, mean=18.042759
    反量化+GEMM vs Triton: max=0.089306, mean=0.016234

2.2 MoE down (triton_fused_matmul_grouped_slice_in_features):
  FP16:                 0.01 ms
  反量化+GEMM:          0.96 ms
  Triton:               0.88 ms

  误差对比:
    Triton vs FP16: max=65.019997, mean=12.635082
    反量化+GEMM vs Triton: max=0.057957, mean=0.011487 -->

场景1: 单独 Linear
    FP16:                 0.01 ms
    反量化+GEMM:          0.71 ms
    Triton:               0.83 ms
  误差对比 (vs FP16):
    Triton:               max=19.556366, mean=3.470817
    反量化+GEMM vs Triton: max=0.053061, mean=0.010133

场景2.1: MoE up_gate (slice_rows)
    FP16:                 0.02 ms
    反量化+GEMM:          1.37 ms
    Triton:               1.62 ms
  误差对比 (vs FP16):
    Triton:               max=25.554825, mean=4.987385
    反量化+GEMM vs Triton: max=0.081268, mean=0.014249

场景2.2: MoE down (slice_in_features)
    FP16:                 0.01 ms
    反量化+GEMM:          0.70 ms
    Triton:               0.23 ms
  误差对比 (vs FP16):
    Triton:               max=9.238243, mean=1.756374
    反量化+GEMM vs Triton: max=0.026306, mean=0.005040

现在可能不是反量化+gemm 和 triton 造成的了，
上面这几个误差，其实都不影响最终结果，测了几次，w/o-triton 反量化+gemm  、triton kernel 的最终误差都差不多。
而是这里的反量化+gemm 和之前的虚拟量化+fp16 gemm 之间的关系。。
  现在的测试程序，对比了 没有 triton 的反量化+ gemm ，和 triton kernel的。我现在需要增加一个对比项，这个对比项是之前的模拟量化时候的方法（可以参考 git 6477b82），就是现场量化，量化完马上就反量化存回 fp16 去，存好的方法。我现在需要你在当前的测试程序中实现这个方法，起名叫 simu_quant ，和 w/o-triton 反量化+gemm  、triton kernel 比精度和速度。

## 额外
另外就是可以考虑要保存量化后的参数了。


## 加速：
1. Kernel 调用次数太多 → 优化方案
方案：把旋转 + 多个 group 的 fused matmul 合并到一个 kernel
不是每个 group 调用一次 triton_fused_matmul
而是创建一个新 kernel，内部遍历处理所有 group
或者：先在 Python 层把所有旋转做好，然后调用一个 kernel 处理所有 group

这个数学上可行吗？
混合精度时：按 slice 分别处理（每个 slice 内统一精度，一个 kernel 处理它内部的所有 group）


2. 每次循环都做旋转矩阵计算和旋转 → 优化方案
方案A：预计算所有旋转矩阵 + 一次性旋转所有输入

不要在循环里：
for g in groups:
    Pi = generate_rotation_matrix(...)
    x_rot_g = x_g @ Pi.T

而是：
x_rot_all = torch.zeros_like(x)
for g in groups:
    Pi = generate_rotation_matrix(...)
    x_rot_all[:, g_start:g_end] = x[:, g_start:g_end] @ Pi.T
然后把 x_rot_all 传给 kernel

方案B（更激进）：把旋转直接做进 Triton kernel 里

在 kernel 内部生成旋转矩阵（或者传进去）
在 kernel 内部做旋转

3. 每次循环都做 packed indices 的切片和 clone → 优化方案
方案：不要切片，直接传完整 indices_packed + 偏移信息

kernel 内部根据 group_idx 计算需要的位置
不需要 clone，直接用索引访问


4.多次 output 累加 → 优化方案
方案：在 kernel 内部累加，或者直接一次输出

把累加逻辑放到 kernel 里
或者在 Python 层只做最后一次写入
