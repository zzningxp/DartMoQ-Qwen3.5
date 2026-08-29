#!/usr/bin/env python3
"""P8-2: chunk gated delta rule 的仓库内优化实现（P2 fp16 化方法迁移）。

主流程实际执行 transformers 内置版的 `torch_chunk_gated_delta_rule`
（modeling_qwen3_5_moe.py:245，仓库根副本与内置版逐字节相同）。
按实施载体 (a)：优化版放在仓库内，模型加载时 monkeypatch 替换，不碰 site-packages。

优化项（每项独立、可逐项验证）：
  P8-2a  零风险：seq 整除 chunk 时 pad_size=0，跳过 5 次 F.pad 全量拷贝
         （P8-1 实测 5.7ms/层 ≈ 模块 4%，跳过后数值逐位一致）
  P8-2b  fp16 化：wy_prep 的 3 个大 bmm + chunk 循环内的 bmm 用 fp16 输入、
         cuBLAS fp32 累加（.half() 输入、.float() 输出），Tensor Core 加速；
         WY 递归 / decay / 状态递推保持 fp32（P8-1 精度敏感区清单）。
         实测 torch 层无收益（cublas 批量 launch 开销为主），默认关闭，
         fp16 留到 P8-4 融合 kernel 内部。
  P8-3   WY 递归向量化：63 次 Python 循环的三角递归 L_new = L + L·L_new
         闭式解 L_new = (I−L)⁻¹L，用一次批量 solve_triangular 替代
         （数学等价，浮点舍入顺序不同）
  P8-4a  bmm 合并：wy_prep 2 个共享左操作数的 bmm 沿右维拼接为 1 个；
         chunk 循环 2 对共享右操作数的 bmm 沿左维拼接（5 个 → 3 个）

⚠️ 函数体以 transformers 内置版为基准逐行对齐，主流程升级时需同步 diff。
数值验证：pad-skip 单独开时与原函数逐位一致；fp16 开后误差 ~1e-2 量级
（同 P6-1 bf16 提升的舍入噪声级别），ppl 为最终判据。

用法（模型加载后，由本人手动接线或测试程序调用）:
    from turboquant_utils.delta_rule import patch_delta_rule, unpatch_delta_rule
    patch_delta_rule(model)      # 替换模型内所有 GatedDeltaNet 的 chunk 路径
    ...
    unpatch_delta_rule(model)
"""

from contextlib import nullcontext

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

# 优化开关（测试/消融用）
ENABLE_PAD_SKIP = True
# ⚠️ fp16 bmm 在 torch 层实测**无收益**（+9.5ms 回归，见 test_p82_delta_rule_fp16.py）：
# 这些 bmm 的成本是 16384 个 64×64 小 GEMM 的 cublas 批量开销，不是 FLOPs，
# fp16 只省 FLOPs 不省 launch，还额外付 cast 成本。fp16 的正解位置在
# P8-4 的融合 Triton kernel 内部（tl.dot fp16 无 cublas 批量开销）。默认关闭。
ENABLE_FP16_BMM = False

# P8-3: WY 递归向量化。原 63 次 Python 循环是三角递归 L_new = L + L·L_new，
# 闭式解 L_new = (I−L)⁻¹L，用一次批量三角求解（cuBLAS trsm）替代整个循环。
ENABLE_WY_SOLVE = True

# P8-4a: bmm 合并（P4-7 思路）。wy_prep 的 attn@v_beta 与 attn@(k_beta*exp)
# 共享左操作数；chunk 循环的 q@state 与 k_cumdecay@state 共享右操作数、
# attn@v_new 与 k^T@v_new 共享右操作数——沿共享维拼接一次 bmm 替代两次。
# ⚠️ torch 层实测**回归 +7.5ms**（cat 拷贝 + 合并后 cublas 调度反而更差），
# 与 fp16 同理：小批量 GEMM 的 launch 开销是主成本，合并概念留到 P8-4
# 融合 Triton kernel 内部用。默认关闭，仅作消融参考。
ENABLE_BMM_MERGE = False

# P8-4: chunk 循环融合 Triton kernel。每块（64 token）的 5 个小 bmm +
# elementwise 全塞进一个 kernel（一个 program 负责一个 (batch, head)），
# 块间 32 次串行递推留在 Python 循环里，每块一次 launch（原 5 次 cublas +
# 多次 elementwise）。tl.dot 无 cublas 小批量 GEMM 开销，后续可换 fp16。
# 默认先开（数值对拍与性能见 test/test_p84_triton_chunk.py）。
ENABLE_TRITON_CHUNK = True

# P8-4 第三步：wy_prep 融合 kernel。wy_bmm + WY 递归（63 步在寄存器内）+
# 2 个后置 bmm（attn@v_beta、attn@(k_beta*exp(g))）全在一个 kernel 里完成，
# attn 矩阵不出寄存器（省 536MB 写 + 两次读），solve_triangular 也被替代
# （WY 用原 63 步递归语义，与 solve 的差异 ~5e-4 已知可接受）。
ENABLE_TRITON_WY = True


def _l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def fast_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    prof=None,
    **kwargs,
):
    """torch_chunk_gated_delta_rule 的优化版（签名与原函数一致）。

    Args:
        prof: 可选 CudaStageProfiler（测量用）。传入时按 cast / pad / decay /
              wy_bmm / wy_solve / chunk_loop / finalize 分阶段计时，
              不传时零开销（原函数调用方式完全不受影响）。
    """
    def _stage(name):
        return prof.stage(name) if prof is not None else nullcontext()

    initial_dtype = query.dtype
    _qk_bf16 = False
    with _stage("cast"):
        if use_qk_l2norm_in_kernel:
            query = _l2norm(query, dim=-1, eps=1e-6)
            key = _l2norm(key, dim=-1, eps=1e-6)
        # P8-4 第二步：q/k 保持 bf16 直通 Triton kernel（transpose 只搬一半字节，
        # 省掉 .to(fp32) 转换）。bf16→fp32 转换是精确的，scale 乘法移进 kernel 内
        # 以 fp32 完成，数值与旧路径一致。仅在 Triton 路径必然生效的形状下启用。
        if (ENABLE_TRITON_CHUNK and chunk_size == 64
                and query.shape[-1] == 128 and value.shape[-1] == 128):
            query = query.transpose(1, 2).contiguous()  # bf16
            key = key.transpose(1, 2).contiguous()      # bf16
            value = value.transpose(1, 2).contiguous().to(torch.float32)
            beta = beta.transpose(1, 2).contiguous().to(torch.float32)
            g = g.transpose(1, 2).contiguous().to(torch.float32)
            _qk_bf16 = True
        else:
            query, key, value, beta, g = [
                x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
            ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    with _stage("pad"):
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        # P8-2a: pad_size=0 时 F.pad 只做全量拷贝，跳过（数值逐位一致）
        if pad_size > 0 or not ENABLE_PAD_SKIP:
            query = F.pad(query, (0, 0, 0, pad_size))
            key = F.pad(key, (0, 0, 0, pad_size))
            value = F.pad(value, (0, 0, 0, pad_size))
            beta = F.pad(beta, (0, pad_size))
            g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    if not _qk_bf16:
        query = query * scale

    with _stage("beta_reshape"):
        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        # reshape to chunks
        query, key, value, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    with _stage("decay"):
        # chunk decay（decay 的 cumsum/exp 链保持 fp32，P8-1 精度敏感区）
        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

    # ---- wy_prep：WY 递归 + 三个大 bmm ----
    # P8-4 第三步：Triton 融合路径（attn 不出寄存器，WY 用 63 步递归语义）
    if ENABLE_TRITON_WY:
        with _stage("wy_fused"):
            _wy_res = _triton_wy_prep(k_beta, key, v_beta, decay_mask, g, chunk_size=chunk_size)
        if _wy_res is not False:
            value, k_cumdecay = _wy_res
        else:
            value, k_cumdecay = _torch_wy_prep(
                k_beta, key, v_beta, decay_mask, g, chunk_size, mask, v_head_dim)
    else:
        value, k_cumdecay = _torch_wy_prep(
            k_beta, key, v_beta, decay_mask, g, chunk_size, mask, v_head_dim)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    if value.dim() != 5:
        print(f"[DEBUG wy] attn={tuple(attn.shape)} v_beta={tuple(v_beta.shape)} "
              f"k_beta={tuple(k_beta.shape)} value={tuple(value.shape)} "
              f"key={tuple(key.shape)} chunk_size={chunk_size}", flush=True)
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk（状态递推保持 fp32）
    with _stage("chunk_loop"):
        last_recurrent_state, core_attn_out = _fast_chunk_loop(
            query, key, value, k_cumdecay, decay_mask, g,
            chunk_size=chunk_size, total_sequence_length=total_sequence_length,
            core_attn_out=core_attn_out, last_recurrent_state=last_recurrent_state,
            scale=scale)

    with _stage("finalize"):
        if not output_final_state:
            last_recurrent_state = None
        if core_attn_out.shape[2] != total_sequence_length // chunk_size:
            print(f"[DEBUG finalize] core_attn_out={tuple(core_attn_out.shape)} "
                  f"seq_len={sequence_length} chunk={chunk_size} "
                  f"total={total_sequence_length}", flush=True)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1],
                                              -1, core_attn_out.shape[-1])
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


@triton.jit
def _delta_chunk_kernel(
    q_ptr, k_ptr, v_ptr, kcd_ptr, dmask_ptr, g_ptr, g_last_ptr,
    state_ptr, o_ptr, new_state_ptr,
    q_rs, k_rs, v_rs, kcd_rs, dm_rs, g_rs, gl_rs, o_rs,
    SCALE: tl.constexpr,
    CS: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    """P8-4: 单个 chunk 的 delta rule 融合 kernel。

    每个 program 负责一个 (batch, head)（grid = B*H）：
      attn        = q @ k^T * decay_mask          (CS, CS)
      attn_inter  = (q * exp(g)) @ state          (CS, BLOCK_V)
      v_prime     = k_cumdecay @ state            (CS, BLOCK_V)
      o           = attn_inter + attn @ (v - v_prime)
      new_state   = state * exp(g_last) + (k * exp(g_last - g))^T @ (v - v_prime)
    K 与 V 维均分块（BLOCK_K × BLOCK_V）以控制共享内存与寄存器压力。
    各张量是 (BH, NC, ...)[:, i] 的 strided 视图，行间 stride 由 *_rs 传入。
    """
    pid = tl.program_id(0)
    offs_cs = tl.arange(0, CS)
    offs_bk = tl.arange(0, BLOCK_K)

    dmask_tile = tl.load(dmask_ptr + pid * dm_rs + offs_cs[:, None] * CS + offs_cs[None, :])
    g_vec = tl.load(g_ptr + pid * g_rs + offs_cs)
    g_last = tl.load(g_last_ptr + pid * gl_rs)
    g_exp = tl.exp(g_vec)                      # (CS,)
    decay_factor = tl.exp(g_last - g_vec)      # (CS,)
    exp_g_last = tl.exp(g_last)

    # attn = q @ k^T（K 分块累加）。q 可能为 bf16（cast 缩减路径）：
    # bf16→fp32 转换精确，scale 在 fp32 下乘——与旧路径数值一致；k 不乘 scale
    attn_acc = tl.zeros((CS, CS), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + offs_bk
        qk = tl.load(q_ptr + pid * q_rs + offs_cs[:, None] * K + offs_k[None, :]).to(tl.float32) * SCALE
        kk = tl.load(k_ptr + pid * k_rs + offs_cs[:, None] * K + offs_k[None, :]).to(tl.float32)
        attn_acc += tl.dot(qk, tl.trans(kk), out_dtype=tl.float32)
    attn = attn_acc * dmask_tile  # (CS, CS)

    # 状态相关计算：K 分块累加 attn_inter / v_prime，再算输出与状态更新
    for v0 in range(0, V, BLOCK_V):
        offs_v = v0 + tl.arange(0, BLOCK_V)
        v_tile = tl.load(v_ptr + pid * v_rs + offs_cs[:, None] * V + offs_v[None, :])

        ai_acc = tl.zeros((CS, BLOCK_V), dtype=tl.float32)  # attn_inter
        vp_acc = tl.zeros((CS, BLOCK_V), dtype=tl.float32)  # v_prime
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + offs_bk
            qk = tl.load(q_ptr + pid * q_rs + offs_cs[:, None] * K + offs_k[None, :]).to(tl.float32) * SCALE
            kcdk = tl.load(kcd_ptr + pid * kcd_rs + offs_cs[:, None] * K + offs_k[None, :])
            state_k = tl.load(state_ptr + pid * K * V + offs_k[:, None] * V + offs_v[None, :])
            qg = qk * g_exp[:, None]                        # (CS, BLOCK_K)
            ai_acc += tl.dot(qg, state_k, out_dtype=tl.float32)
            vp_acc += tl.dot(kcdk, state_k, out_dtype=tl.float32)

        v_new = v_tile - vp_acc
        o_part = tl.dot(attn, v_new, out_dtype=tl.float32)  # (CS, BLOCK_V)
        tl.store(o_ptr + pid * o_rs + offs_cs[:, None] * V + offs_v[None, :],
                 ai_acc + o_part)

        # 状态更新：new_state[k0 段] = state 段 * exp(g_last) + k_decay^T @ v_new
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + offs_bk
            kk = tl.load(k_ptr + pid * k_rs + offs_cs[:, None] * K + offs_k[None, :]).to(tl.float32)
            state_k = tl.load(state_ptr + pid * K * V + offs_k[:, None] * V + offs_v[None, :])
            k_decay_k = kk * decay_factor[:, None]          # (CS, BLOCK_K)
            state_part = tl.dot(tl.trans(k_decay_k), v_new, out_dtype=tl.float32)  # (BK, BLOCK_V)
            tl.store(new_state_ptr + pid * K * V + offs_k[:, None] * V + offs_v[None, :],
                     state_k * exp_g_last + state_part)


@triton.jit
def _delta_wy_kernel(
    kb_ptr, key_ptr, vb_ptr, dmask_ptr, g_ptr,
    value_ptr, kcd_ptr,
    kb_rs, key_rs, vb_rs, dm_rs, g_rs, value_rs, kcd_rs,
    CS: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """P8-4 第三步：wy_prep 融合 kernel（每个 program 一个 (batch, head, chunk)）。

    在一个 kernel 内完成：
      attn       = -(k_beta @ key^T * decay_mask) 严格下三角（K 分块）
      WY 递归    = 原 63 步循环（寄存器内，无 launch）
      value      = (attn + I) @ v_beta
      k_cumdecay = (attn + I) @ (k_beta * exp(g_cum))
    attn 矩阵不出寄存器，替代 solve_triangular 的 536MB 内存往返。
    WY 用原 63 步递归语义（与 solve 闭式解的差异 ~5e-4，已知可接受）。
    """
    pid = tl.program_id(0)
    offs_cs = tl.arange(0, CS)
    offs_bk = tl.arange(0, BLOCK_K)

    dmask_tile = tl.load(dmask_ptr + pid * dm_rs + offs_cs[:, None] * CS + offs_cs[None, :])
    g_vec = tl.load(g_ptr + pid * g_rs + offs_cs)  # 已 cumsum 的 g
    g_exp = tl.exp(g_vec)

    # attn = -(k_beta @ key^T) * decay_mask，严格下三角
    attn = tl.zeros((CS, CS), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + offs_bk
        kb_k = tl.load(kb_ptr + pid * kb_rs + offs_cs[:, None] * K + offs_k[None, :])
        key_k = tl.load(key_ptr + pid * key_rs + offs_cs[:, None] * K + offs_k[None, :]).to(tl.float32)
        attn += tl.dot(kb_k, tl.trans(key_k), out_dtype=tl.float32)
    attn = -attn * dmask_tile
    attn = tl.where(offs_cs[None, :] < offs_cs[:, None], attn, 0.0)  # 严格下三角

    # WY 递归 63 步（寄存器内）
    # 原 torch 语义：attn[i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    #   row = attn[i, :i] (i,), sub = attn[:i, :i] (i, i)
    #   row.unsqueeze(-1) 是 (i, 1) 列向量，广播乘 sub → (i, i)
    #   result[p, q] = row[p] * sub[p, q]（row 第 p 个元素 × sub 第 p 行所有列）
    #   .sum(-2) 对行维求和 → result[q] = sum_p row[p] * sub[p, q] = row @ sub
    # Triton 中用 tl.expand_dims(row, 1) 得到 (CS,1) 列向量，与 sub(CS,CS) 广播乘即列向量语义。
    for i in tl.static_range(1, CS):
        row = tl.sum(tl.where(offs_cs[:, None] == i, attn, 0.0), axis=0)  # (CS,) 第 i 行
        sub = tl.where(offs_cs[:, None] < i, attn, 0.0)                   # (CS, CS) 前 i 行
        row_col = tl.expand_dims(row, axis=1)                             # (CS, 1) 列向量
        weighted = row_col * sub                                          # (CS, CS) 列广播：[r,c]=row[r]*sub[r,c]
        update = tl.sum(weighted, axis=0)                                 # (CS,) 对行求和 = row @ sub
        update = tl.where(offs_cs < i, update, 0.0)
        attn = attn + tl.where(offs_cs[:, None] == i, update[None, :], 0.0)

    attn = attn + tl.where(offs_cs[:, None] == offs_cs[None, :], 1.0, 0.0)  # + I

    # value = attn @ v_beta（V 分块 64）
    for v0 in range(0, V, 64):
        offs_v = v0 + tl.arange(0, 64)
        vb_tile = tl.load(vb_ptr + pid * vb_rs + offs_cs[:, None] * V + offs_v[None, :])
        val_tile = tl.dot(attn, vb_tile, out_dtype=tl.float32)             # (CS, 64)
        tl.store(value_ptr + pid * value_rs + offs_cs[:, None] * V + offs_v[None, :], val_tile)

    # k_cumdecay = attn @ (k_beta * exp(g_cum))（K 分块，k_beta 重载自 L2）
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + offs_bk
        kb_k = tl.load(kb_ptr + pid * kb_rs + offs_cs[:, None] * K + offs_k[None, :])
        kbg_k = kb_k * g_exp[:, None]
        kcd_tile = tl.dot(attn, kbg_k, out_dtype=tl.float32)               # (CS, BK)
        tl.store(kcd_ptr + pid * kcd_rs + offs_cs[:, None] * K + offs_k[None, :], kcd_tile)


def _triton_wy_prep(k_beta, key, v_beta, decay_mask, g, chunk_size=64):
    """P8-4 第三步：wy_prep 的 Triton 融合版。

    返回 (value, k_cumdecay)；形状不满足（CS=64, K=128, V%64==0）时返回 False。
    g 必须是已 cumsum 的 g（decay 阶段完成）。attn 矩阵全程不出 kernel。
    """
    B, H, NC, CS, K = k_beta.shape
    V = v_beta.shape[-1]
    if CS != 64 or K != 128 or V % 64 != 0:
        return False

    BHNC = B * H * NC
    kb2 = k_beta.reshape(BHNC, CS, K)
    key2 = key.reshape(BHNC, CS, K)
    vb2 = v_beta.reshape(BHNC, CS, V)
    dm2 = decay_mask.reshape(BHNC, CS, CS)
    g2 = g.reshape(BHNC, CS)
    value = torch.empty(B, H, NC, CS, V, dtype=torch.float32, device=k_beta.device)
    kcd = torch.empty(B, H, NC, CS, K, dtype=torch.float32, device=k_beta.device)
    val2 = value.reshape(BHNC, CS, V)
    kcd2 = kcd.reshape(BHNC, CS, K)

    # P8-6 自动调优最优配置（B=32, H=16, seq=2048, CS=64, K=V=128）：
    # BLOCK_K=128（K 维不分块）+ stages=2，wy 2.53→1.82ms（+28%）
    _delta_wy_kernel[(BHNC,)](
        kb2, key2, vb2, dm2, g2, val2, kcd2,
        kb2.stride(0), key2.stride(0), vb2.stride(0), dm2.stride(0), g2.stride(0),
        val2.stride(0), kcd2.stride(0),
        CS=CS, K=K, V=V, BLOCK_K=128,
        num_warps=4, num_stages=2,
    )
    return value, kcd


def _torch_wy_prep(k_beta, key, v_beta, decay_mask, g, chunk_size, mask, v_head_dim):
    """wy_prep 的 torch 版（原路径，作回退与消融基准）。"""
    if ENABLE_FP16_BMM:
        attn = -((k_beta.half() @ key.transpose(-1, -2).half()).float() * decay_mask).masked_fill(mask, 0)
    else:
        # key 在 cast 缩减路径下是 bf16（k_beta 已因 beta fp32 提升为 fp32），
        # bmm 要求同类型，显式转 fp32（bf16→fp32 精确，数值不变）
        attn = -((k_beta @ key.transpose(-1, -2).to(torch.float32)) * decay_mask).masked_fill(mask, 0)
    if ENABLE_WY_SOLVE:
        # P8-3: 三角递归 L_new = L + L·L_new 的闭式解 L_new = (I−L)⁻¹L。
        # attn 为严格下三角 L；I−L 单位下三角，批量三角求解一次完成。
        eye = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
        i_minus_l = eye - attn
        attn = torch.linalg.solve_triangular(i_minus_l, attn,
                                             upper=False, unitriangular=True) + eye
    else:
        # WY 递归保持 fp32（动态范围大）
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    if ENABLE_BMM_MERGE:
        # P8-4a: attn@v_beta 与 attn@(k_beta*exp) 共享左操作数，沿右维拼接一次 bmm
        rhs = torch.cat([v_beta, k_beta * g.exp().unsqueeze(-1)], dim=-1)
        if ENABLE_FP16_BMM:
            merged = (attn.half() @ rhs.half()).float()
        else:
            merged = attn @ rhs
        value = merged[..., :v_head_dim]
        k_cumdecay = merged[..., v_head_dim:]
    else:
        if ENABLE_FP16_BMM:
            value = (attn.half() @ v_beta.half()).float()
            k_cumdecay = (attn.half() @ (k_beta * g.exp().unsqueeze(-1)).half()).float()
        else:
            value = attn @ v_beta
            k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    return value, k_cumdecay


def _triton_chunk_loop(query, key, value, k_cumdecay, decay_mask, g,
                       total_sequence_length, core_attn_out, last_recurrent_state,
                       scale=None):
    """P8-4: chunk 串行循环的 Triton 融合版。

    与 _fast_chunk_loop 的 torch 版数学一致，块间递推留在 Python 循环，
    每块一次 kernel launch（一个 program 负责一个 (batch, head)）。
    形状要求（本模型满足）：CS=64, K=V=128 且为 2 的幂；不满足时回退 torch 版。
    q/k 可为 bf16（cast 缩减路径），scale 在 kernel 内以 fp32 乘到 q 上。
    """
    B, H, NC, CS, K = query.shape
    V = value.shape[-1]
    if CS != 64 or K != V or (K & (K - 1)) != 0 or V % 32 != 0:
        return False  # 形状不满足，回退 torch 版
    if scale is None:
        scale = 1.0 / (K ** 0.5)

    BH = B * H
    q2 = query.reshape(BH, NC, CS, K)
    k2 = key.reshape(BH, NC, CS, K)
    v2 = value.reshape(BH, NC, CS, V)
    kcd2 = k_cumdecay.reshape(BH, NC, CS, K)
    dm2 = decay_mask.reshape(BH, NC, CS, CS)
    g2 = g.reshape(BH, NC, CS)
    o2 = core_attn_out.reshape(BH, NC, CS, V)
    state = last_recurrent_state.reshape(BH, K, V).contiguous()
    new_state = torch.empty_like(state)
    g_last_all = g2[:, :, -1].contiguous()  # (BH, NC)

    # P8-6 自动调优最优配置（B=32, H=16, seq=2048, CS=64, K=V=128）：
    # BLOCK_V=128（V 维不分块）+ warps=4 + stages=2，chunk 循环 6.34→2.92ms（+54%）
    BLOCK_K = 32
    BLOCK_V = 128
    for i in range(NC):
        _delta_chunk_kernel[(BH,)](
            q2[:, i], k2[:, i], v2[:, i], kcd2[:, i], dm2[:, i], g2[:, i],
            g_last_all[:, i], state, o2[:, i], new_state,
            q2[:, i].stride(0), k2[:, i].stride(0), v2[:, i].stride(0),
            kcd2[:, i].stride(0), dm2[:, i].stride(0), g2[:, i].stride(0),
            g_last_all[:, i].stride(0), o2[:, i].stride(0),
            SCALE=scale,
            CS=CS, K=K, V=V, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
            num_warps=4, num_stages=2,
        )
        state, new_state = new_state, state  # 交换缓冲，避免每块新分配

    last_recurrent_state = state.reshape(B, H, K, V)
    return last_recurrent_state, core_attn_out


def _fast_chunk_loop(query, key, value, k_cumdecay, decay_mask, g, chunk_size,
                     total_sequence_length, core_attn_out, last_recurrent_state, **kwargs):
    """fast_chunk_gated_delta_rule 的 chunk 串行循环（独立函数便于消融）。

    返回更新后的 (last_recurrent_state, core_attn_out)。
    """
    scale = kwargs.get("scale", None)
    if ENABLE_TRITON_CHUNK:
        res = _triton_chunk_loop(
            query, key, value, k_cumdecay, decay_mask, g,
            total_sequence_length, core_attn_out, last_recurrent_state,
            scale=scale)
        if res is not False:
            return res
        # 形状不满足（非常规配置）时回退 torch 版
    # 回退路径：q/k 若还是 bf16（cast 缩减路径），先转 fp32 并补 scale，
    # 保持与原 torch 版数值行为一致
    if query.dtype == torch.bfloat16:
        query = query.to(torch.float32)
        key = key.to(torch.float32)
        if scale is not None:
            query = query * scale
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        # g 最后一项作 decay 基底。注意形状语义：
        #   - 减法用 (B,H,1)（与 (B,H,CS) 对齐广播，勿用 (B,H,1,1)——会与 H 维错位）
        #   - 状态乘法用 (B,H,1,1)（与 (B,H,K,V) 对齐）
        g_last_s = g[:, :, i, -1, None]  # (B,H,1) 原始值
        if ENABLE_BMM_MERGE:
            # P8-4a: 共享右操作数的两对 bmm 沿左维拼接（chunk 循环 5 个 bmm → 3 个）
            #   [q_i*g_exp; k_cumdecay_i] @ state → attn_inter + v_prime
            #   [attn; k_i_decay^T] @ v_new   → attn@v_new + 状态更新项
            attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
            lhs_state = torch.cat(
                [q_i * g[:, :, i, :, None].exp(), k_cumdecay[:, :, i]], dim=-2)
            state_rhs = lhs_state @ last_recurrent_state
            attn_inter = state_rhs[:, :, :chunk_size]
            v_prime = state_rhs[:, :, chunk_size:]
            v_new = v_i - v_prime
            k_i_decay_t = (k_i * (g_last_s - g[:, :, i]).exp()[..., None]).transpose(-1, -2)
            lhs_vnew = torch.cat([attn, k_i_decay_t], dim=-2)
            vnew_rhs = lhs_vnew @ v_new
            core_attn_out[:, :, i] = attn_inter + vnew_rhs[:, :, :chunk_size]
            last_recurrent_state = (
                last_recurrent_state * g_last_s[..., None].exp() + vnew_rhs[:, :, chunk_size:]
            )
        else:
            attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
            v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
            core_attn_out[:, :, i] = attn_inter + attn @ v_new
            last_recurrent_state = (
                last_recurrent_state * g_last_s[..., None].exp()
                + (k_i * (g_last_s - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
            )

    return last_recurrent_state, core_attn_out


_PATCHED_RULE = {}  # id(instance) -> 原函数


def patch_delta_rule(module):
    """把 module（或整个模型）内所有 GatedDeltaNet 的 chunk 路径换成 fast 版。

    仅替换实例属性 `chunk_gated_delta_rule`（模块 init 时赋的是内置版函数），
    seq_len==1 的 recurrent 路径不动（eval 用不到）。
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet

    targets = [m for m in module.modules() if isinstance(m, Qwen3_5MoeGatedDeltaNet)]
    for m in targets:
        if id(m) in _PATCHED_RULE:
            continue
        _PATCHED_RULE[id(m)] = m.chunk_gated_delta_rule
        m.chunk_gated_delta_rule = fast_chunk_gated_delta_rule
    return targets


def unpatch_delta_rule(module):
    for m in module.modules():
        if id(m) in _PATCHED_RULE:
            m.chunk_gated_delta_rule = _PATCHED_RULE.pop(id(m))
    return module
