# -*- coding: utf-8 -*-
"""WxFP8 路线的 Triton kernel 与工具（roadmaps/wxfp8-plan-260924.md）。

阶段内容：
  WF-1  build_fp8_codebook（本文件，码本 e4m3 化 + 最优 scale 搜索）
  WF-2  _rotate_quantize_kernel fp8 版（激活量化，待加）
  WF-3  wxfp8 GEMM kernels（attention 全矩阵 + MoE gate_up/down，待加）

与 triton_kernels_a8.py 的关系：结构对称平移。int8 版的契约是
`codebook[i] ≈ cb_i8[i] * cb_step`（cb_step 折进激活 extra_scale），
fp8 版保持同一契约：`codebook[i] ≈ cb_e4m3[i] * cb_scale`。
"""

import math

import torch
import triton
import triton.language as tl

# 合并 kernel 的 host 端 block_info 构建器与 int8 版逐字节同构（纯 torch、
# dtype 无关），直接复用，不重复实现
from turboquant_utils.triton_kernels_a8 import _build_merged_block_info  # noqa: E402

# e4m3 (float8_e4m3fn) 最大有限值；超过它 torch eager 转换会出 NaN（不饱和），
# Triton 侧 .to(tl.float8e4nv) 才是 satfinite —— 所有 eager 侧转换前必须 clamp。
E4M3_MAX = 448.0


def _cb_relerr(cb: torch.Tensor, lut: torch.Tensor, scale: float) -> float:
    """码本相对误差 ‖lut*scale - cb‖ / ‖cb‖（与 build_int8_codebook 文档口径一致）。"""
    rec = lut.float() * scale
    return ((rec - cb).norm() / cb.norm()).item()


def build_fp8_codebook(codebook: torch.Tensor, n_coarse: int = 512):
    """把 fp16/fp32 码本转成 e4m3 LUT，返回 (cb_e4m3, cb_scale)。

    满足 `codebook[i] ≈ cb_e4m3[i] * cb_scale`（与 build_int8_codebook 同契约，
    cb_scale 同样折进激活 extra_scale 通道）。

    与 int8 版的关键差异：e4m3 是浮点网格（~3-bit 尾数，转换误差尺度不变，
    直接转换 RMS ~2-3%），但正因如此存在 int8 没有的优化自由度——
    **单一 cb_scale 可搜索**，让低 bit 码本的少量质心尽量都落在网格点上：

        1-bit  ±a            → 取 s = a/6 即精确（误差 0）
        2-bit  ±a, ±b        → 近精确（a/b 比值贴近两个网格值之比）
        4-bit  8 个幅度      → 预计 <1%（实测见 test_wxfp8_codebook_prec.py）
        8-bit  uniform 256 级 → 结构性塌级（e4m3 总共仅 254 个有限幅度，
                               均匀等差网格映射后等效 ~W7；attention 敏感，
                               混合部署决策见 roadmap 精度表）

    搜索在 load 时一次性完成（coarse 对数网格 + 局部细化），代价可忽略。
    """
    cb = codebook.float()
    cb_max = cb.abs().max()
    if cb_max <= 0:
        raise ValueError("码本全零，无法转 fp8")

    def lut_at(scale: float) -> torch.Tensor:
        # 除以 scale 后 clamp 到 ±448 再转 e4m3（模拟 Triton satfinite 语义；
        # torch eager .to() 越界出 NaN，必须先 clamp）
        return (cb / scale).clamp_(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)

    # coarse：默认 scale = cb_max/448（满量程）附近 ±1 个二倍程的对数网格
    s0 = (cb_max / E4M3_MAX).item()
    coarse = torch.logspace(
        torch.log10(torch.tensor(s0)) - 1.0,
        torch.log10(torch.tensor(s0)) + 1.0,
        n_coarse,
    ).tolist()
    best_s, best_err = s0, float("inf")
    for s in coarse:
        err = _cb_relerr(cb, lut_at(s), s)
        if err < best_err:
            best_s, best_err = s, err

    # fine：最优 coarse 点附近对数细化两轮
    span = (coarse[1] / coarse[0]) if len(coarse) > 1 else 1.0
    for ratio in (span ** 0.5, span ** 0.25):
        fine = [best_s * ratio ** (i / 64.0 - 0.5) for i in range(64)]
        for s in fine:
            err = _cb_relerr(cb, lut_at(s), s)
            if err < best_err:
                best_s, best_err = s, err

    cb_e4m3 = lut_at(best_s).contiguous()
    return cb_e4m3, best_s


# ===========================================================================
# WF-2：rotate + quantize 融合 kernel（fp8 版）
# ===========================================================================

def quantize_act_per_token_group_fp8(x_rot: torch.Tensor, group_size: int,
                                     extra_scale: float = 1.0):
    """per-token per-group e4m3 量化（torch 参考实现，对应 int8 版
    quantize_act_per_token_group）。

    Args:
        x_rot: (B, K) 已旋转的激活，K 必须是 group_size 的整数倍
        group_size: 分组大小
        extra_scale: 额外折进激活 scale 的常量，传 `build_fp8_codebook`
            返回的 `cb_scale`（与 int8 版传 cb_step 完全同机制，
            让 WxA16 的 fp16 norms_gf buffer 原封不动沿用）。

    Returns:
        x_f8:    (B, K) float8_e4m3fn，contiguous
        x_scale: (B, K // group_size) fp32，contiguous（已含 extra_scale）

    ⚠ eager 侧除法结果理论上 ≤448 但 fp32 除法可到 448*(1+ε)，torch
    .to() 越界出 NaN，所以这里必须 clamp（kernel 侧 satfinite 不需要）。
    """
    B, K = x_rot.shape
    if K % group_size != 0:
        raise ValueError(f"K ({K}) 必须是 group_size ({group_size}) 的整数倍")
    G = K // group_size

    xg = x_rot.reshape(B, G, group_size).float()
    amax = xg.abs().amax(dim=-1)                       # (B, G)
    scale = (amax / E4M3_MAX).clamp(min=1e-8)
    q = (xg / scale.unsqueeze(-1)).clamp_(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    if extra_scale != 1.0:
        scale = scale * extra_scale
    return q.reshape(B, K).contiguous(), scale.contiguous()


@triton.jit
def _rotate_quantize_kernel_fp8(
    x_ptr,            # (B, K_total) 未旋转激活
    rot_ptr,          # (NUM_GROUPS, G0_STRIDE) 每 group 一个旋转矩阵，
                      # 布局与 rotation.generate_batch_rotation_matrices 一致：
                      # 第 g 个矩阵的基址 = g * G0_STRIDE，行 stride = GROUP_SIZE
    x_f8_ptr,         # (B, K_total) e4m3 输出
    x_s_ptr,          # (B, NUM_GROUPS) fp16 输出 scale（已折入 EXTRA_SCALE）
    B, K_total,
    ROT_G0_STRIDE,    # rot_ptr 第 0 维 stride
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EXTRA_SCALE: tl.constexpr,    # 折进 scale 的常量（cb_scale）
    BLOCK_B: tl.constexpr = 32,
):
    """分组旋转 + per-token-per-group e4m3 量化的融合 kernel（WxFP8 WF-2）。

    与 _rotate_quantize_kernel（int8 版，triton_kernels_a8.py:428）逐行对应，
    数学等价，只有量化尾不同：
        int8: q = round(acc / (amax/127))，clamp ±127，存 int8
        fp8:  q = (acc / (amax/448)).to(tl.float8e4nv)   # RTNE + satfinite

    satfinite 语义：acc/scale_unf 理论上 ≤448，但 fp32 除法可能得 448*(1+ε)，
    .to(tl.float8e4nv) 会饱和到 448 而不是出 NaN —— 这正是 kernel 内量化
    天然安全、torch eager .to() 必须先 clamp 的原因。

    注意（与 int8 版相同的约定）：EXTRA_SCALE 只折进**存出去的** scale；
    量化除法必须用未折的 amax/448，否则 q = acc/scale 会偏小 cb_scale 倍。
    存出的 scale 是 fp16：2-bit 的 cb_scale (~0.006) 比 int8 的 cb_step 小
    ~3.5x，典型 scale 会落进 fp16 次正规区附近（~6e-5），那里仍有 ~10 bit
    精度（relerr ~0.05%，与正常区相当）；只有 amax 极小的 group（贡献本身
    可忽略）才明显失真。若 WF-5 ppl 异常，此处是排查点之一。
    """
    pid_g = tl.program_id(0)
    pid_b = tl.program_id(1)
    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = rb < B

    # x 块: (BLOCK_B, GROUP_SIZE) —— 每个 program 处理一个 (group, row-block)
    x_off = rb[:, None] * K_total + pid_g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    x_tile = tl.load(x_ptr + x_off, mask=mask_b[:, None], other=0.0)  # fp16

    # 旋转: acc = x_tile @ P_g^T（对齐 batch_rotate_input 的 x @ P_batch^T 约定）
    acc = tl.zeros((BLOCK_B, GROUP_SIZE), dtype=tl.float32)
    rj = tl.arange(0, GROUP_SIZE)
    rk = tl.arange(0, GROUP_SIZE)
    # P_g 以 (j 行, k 列) 布局加载，dot 里转置回来
    p_off = pid_g * ROT_G0_STRIDE + rj[:, None] * GROUP_SIZE + rk[None, :]
    p_tile = tl.load(rot_ptr + p_off)                          # (G, G)
    acc += tl.dot(x_tile, tl.trans(p_tile), out_dtype=tl.float32)

    # per-row max → scale（除法用未折 EXTRA_SCALE 的值）
    amax = tl.max(tl.abs(acc), axis=1)                         # (BLOCK_B,)
    scale_unf = tl.where(amax < 1e-8 * 448.0, 1e-8, amax / 448.0)

    q_f8 = (acc / scale_unf[:, None]).to(tl.float8e4nv)
    tl.store(x_f8_ptr + x_off, q_f8, mask=mask_b[:, None])
    tl.store(x_s_ptr + rb * NUM_GROUPS + pid_g,
             (scale_unf * EXTRA_SCALE).to(tl.float16), mask=mask_b)


def rotate_quantize_fused_fp8(x: torch.Tensor, rot: torch.Tensor,
                              group_size: int, num_groups: int,
                              extra_scale: float = 1.0):
    """分组旋转 + e4m3 量化（融合 kernel 的 python wrapper，对应 int8 版
    rotate_quantize_fused）。

    Args:
        x: (B, K_total) 未旋转激活，K_total = num_groups * group_size
        rot: (num_groups, group_size, group_size) 每 group 一个旋转矩阵，
             与 generate_batch_rotation_matrices 的返回布局一致
        group_size / num_groups: 分组参数
        extra_scale: 折进 scale 的常量（build_fp8_codebook 的 cb_scale）

    Returns:
        x_f8:    (B, K_total) float8_e4m3fn
        x_scale: (B, num_groups) fp16
    """
    B, K_total = x.shape
    if K_total != num_groups * group_size:
        raise ValueError(f"K_total ({K_total}) != num_groups*group_size "
                         f"({num_groups * group_size})")
    if rot.shape != (num_groups, group_size, group_size):
        raise ValueError(f"rot 形状应为 {(num_groups, group_size, group_size)}, "
                         f"得到 {tuple(rot.shape)}")

    x_f8 = torch.empty((B, K_total), dtype=torch.float8_e4m3fn, device=x.device)
    x_scale = torch.empty((B, num_groups), dtype=torch.float16, device=x.device)

    # BLOCK_B 固定 32：tl.dot 要求 M >= 16，极小 B（长尾 expert）不能缩块，
    # 靠 mask 丢掉多余行 —— 与 int8 版相同的取舍。
    BLOCK_B = 32
    grid = (num_groups, triton.cdiv(B, BLOCK_B))
    _rotate_quantize_kernel_fp8[grid](
        x, rot, x_f8, x_scale, B, K_total, rot.stride(0),
        GROUP_SIZE=group_size, NUM_GROUPS=num_groups,
        EXTRA_SCALE=extra_scale, BLOCK_B=BLOCK_B,
        num_warps=4, num_stages=3,
    )
    return x_f8, x_scale


@triton.jit
def _rotate_quantize_kernel_fp8_pooled(
    x_ptr,            # (B_cap, K_total) fp16 —— 池缓冲（行序 = 成员序）
    rot_ptr,          # 旋转矩阵库（TOTAL, G0_STRIDE），第 k 个矩阵基址 = k*stride
    expert_info_ptr,  # (E, 4) int32 [token_start, token_count, rot_base, g_count]
    block_info_ptr,   # (num_blocks, 2) int32 [member_idx, block_b]
    x_f8_ptr,         # (B_cap, K_total) e4m3 输出
    x_s_ptr,          # (B_cap, NUM_GROUPS) fp16 输出 scale
    K_total,
    ROT_G0_STRIDE,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EXTRA_SCALE: tl.constexpr,
    BLOCK_B: tl.constexpr = 32,
):
    """池内逐成员旋转 + e4m3 量化（静态派发 down 方向）。

    与 `_rotate_quantize_kernel_fp8` 同一数学、同一量化尾；差别只在旋转
    矩阵来源：down 的 seed 含 expert 偏移，整批不能共用一个矩阵。按"成员
    × 行块"的最坏情况静态调度表分发程序，行段与矩阵基序号由 kernel 运行时
    从 expert_info 读（列 2 = start // group_size，与 GEMM 的 group 切片
    语义一致，共用同一份缓冲）。token_count=0 的成员整块早退。
    """
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)
    bi = block_info_ptr + pid * 2
    midx = tl.load(bi + 0).to(tl.int32)
    blk_b = tl.load(bi + 1).to(tl.int32)

    ei = expert_info_ptr + midx * 4
    token_start = tl.load(ei + 0).to(tl.int32)
    token_count = tl.load(ei + 1).to(tl.int32)
    rot_base = tl.load(ei + 2).to(tl.int32)
    if token_count <= 0:
        return

    rb = token_start + blk_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = rb < token_start + token_count

    x_off = rb[:, None] * K_total + pid_g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    x_tile = tl.load(x_ptr + x_off, mask=mask_b[:, None], other=0.0)

    acc = tl.zeros((BLOCK_B, GROUP_SIZE), dtype=tl.float32)
    rj = tl.arange(0, GROUP_SIZE)
    rk = tl.arange(0, GROUP_SIZE)
    p_off = (rot_base + pid_g) * ROT_G0_STRIDE + rj[:, None] * GROUP_SIZE + rk[None, :]
    p_tile = tl.load(rot_ptr + p_off)
    acc += tl.dot(x_tile, tl.trans(p_tile), out_dtype=tl.float32)

    amax = tl.max(tl.abs(acc), axis=1)
    scale_unf = tl.where(amax < 1e-8 * 448.0, 1e-8, amax / 448.0)

    q_f8 = (acc / scale_unf[:, None]).to(tl.float8e4nv)
    tl.store(x_f8_ptr + x_off, q_f8, mask=mask_b[:, None])
    tl.store(x_s_ptr + rb * NUM_GROUPS + pid_g,
             (scale_unf * EXTRA_SCALE).to(tl.float16), mask=mask_b)


def rotate_quantize_fused_fp8_pooled(x: torch.Tensor, rot_bank: torch.Tensor,
                                     expert_info: torch.Tensor,
                                     block_info: torch.Tensor,
                                     group_size: int, num_groups: int,
                                     extra_scale: float = 1.0):
    """池内逐成员旋转 + e4m3 量化的 python wrapper（静态派发用）。

    参数语义与 triton_kernels_a8.rotate_quantize_fused_pooled 一一对应；
    extra_scale 传 cb_scale。
    """
    B_cap, K_total = x.shape
    if K_total != num_groups * group_size:
        raise ValueError(f"K_total ({K_total}) != num_groups*group_size "
                         f"({num_groups * group_size})")
    x_f8 = torch.empty((B_cap, K_total), dtype=torch.float8_e4m3fn,
                       device=x.device)
    x_scale = torch.empty((B_cap, num_groups), dtype=torch.float16,
                          device=x.device)
    grid = (block_info.shape[0], num_groups)
    _rotate_quantize_kernel_fp8_pooled[grid](
        x, rot_bank, expert_info, block_info, x_f8, x_scale,
        K_total, rot_bank.stride(0),
        GROUP_SIZE=group_size, NUM_GROUPS=num_groups,
        EXTRA_SCALE=extra_scale, BLOCK_B=32,
        num_warps=4, num_stages=3,
    )
    return x_f8, x_scale


# ===========================================================================
# WF-3：融合反量化 + FP8 matmul kernel（group-first 布局）
# ===========================================================================

@triton.jit
def _wxfp8_fused_matmul_kernel_grouped_gf(
    # Input
    x_ptr,            # (B, K_total) e4m3   —— 已旋转并量化（rotate_quantize_fused_fp8）
    xs_ptr,           # (B, NUM_GROUPS) fp16 —— per-token per-group 激活 scale（含 cb_scale）
    # Quantized weight (group-first 布局)
    indices_ptr,      # (NUM_GROUPS_TOTAL, N, PACKED_PER_GROUP) uint8
    codebook_ptr,     # (n_levels,) e4m3 —— build_fp8_codebook 的 LUT
    norms_ptr,        # (NUM_GROUPS_TOTAL, N) fp16 —— 与 WxA16/WxA8 的 norms_gf 同源
                      #   （1/sqrt(gs) 已预乘；码本的 cb_scale 折在激活 scale 里）
    # Output
    output_ptr,       # (B, N) fp16
    # Shape
    B, N,
    K_total,              # 总 K = num_groups * group_size
    INDICES_G0_STRIDE,    # indices 第 0 维 stride = N * PACKED_PER_GROUP
    NORMS_G0_STRIDE,      # norms 第 0 维 stride = N
    XS_ROW_STRIDE,        # xs 第 0 维 stride = NUM_GROUPS_TOTAL
    # Constexpr config
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BIT_WIDTH: tl.constexpr,
    N_LEVELS: tl.constexpr,
    BLOCK_B: tl.constexpr = 256,
    BLOCK_N: tl.constexpr = 32,
    BLOCK_K: tl.constexpr = 128,
):
    """WxFP8 multi-group fused dequant + FP8 matmul kernel（group-first 布局）。

    与 _wxa8_fused_matmul_kernel_grouped_gf（triton_kernels_a8.py:35）逐行对应，
    数学结构相同，三处不同：
      1. 激活 tile 是 e4m3（1 byte，与 int8 同访存量）
      2. 码本是 e4m3 LUT（build_fp8_codebook 产物）。无 IDENTITY_CB 路径——
         e4m3 网格对 idx 非线性，8-bit 均匀码本也必须查表
         （attention W8 的 LUT 塌级问题见 roadmap §1.3，混合部署为备选）
      3. 累加器：group 内直接 fp32 dot 累加（int8 版是 int32 组内 + 出组转 fp32），
         出 group 乘 (xs_g × norm_g) 的 epilogue 结构不变

    注: indices_ptr / norms_ptr 指向切片后的起始位置（第 0 个待处理 group 的
    起点），kernel 内 g 从 0 到 NUM_GROUPS-1；xs_ptr 指向完整 (B, NUM_GROUPS_TOTAL)
    起点（同 int8 版约定，见其 docstring）。
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_b = rb < B
    mask_n = rn < N

    total_acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    ELEMENTS_PER_BYTE = 8 // BIT_WIDTH
    PACKED_PER_GROUP = GROUP_SIZE // ELEMENTS_PER_BYTE

    row_base = rn * PACKED_PER_GROUP  # (BLOCK_N,)

    for g in range(NUM_GROUPS):
        g_start = g * GROUP_SIZE
        g_base = g * INDICES_G0_STRIDE

        norm_g = tl.load(norms_ptr + g * NORMS_G0_STRIDE + rn, mask=mask_n, other=0.0)
        xs_g = tl.load(xs_ptr + rb * XS_ROW_STRIDE + g, mask=mask_b, other=0.0)

        acc_g = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, GROUP_SIZE, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)
            mask_k = rk < GROUP_SIZE

            # 激活 tile：e4m3。masked load 不给 other（int 字面量无法转 fp8e4nv；
            # 被 mask 的行/列只影响不会写出的输出行列，K 方向 gs=128 整除 BLOCK_K 无尾部）
            inp_off = rb[:, None] * K_total + (g_start + rk)[None, :]
            x_tile = tl.load(x_ptr + inp_off,
                             mask=mask_b[:, None] & mask_k[None, :])

            w_mask = mask_n[:, None] & mask_k[None, :]
            if BIT_WIDTH == 8:
                byte_off = g_base + row_base[:, None] + rk[None, :]
                idx = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.int32)
            else:
                BIT_MASK = (1 << BIT_WIDTH) - 1
                byte_col = rk // ELEMENTS_PER_BYTE
                pos_in_byte = rk % ELEMENTS_PER_BYTE
                byte_off = g_base + row_base[:, None] + byte_col[None, :]
                packed = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
                shift = pos_in_byte * BIT_WIDTH
                idx = ((packed >> shift[None, :]) & BIT_MASK).to(tl.int32)

            # 码本查表 → e4m3 权重 tile（不给 other，理由同上）
            w_f8 = tl.load(codebook_ptr + idx, mask=w_mask)
            acc_g += tl.dot(x_tile, tl.trans(w_f8))

        # 出 group：乘 (激活 scale × 权重 scale)，累加进总累加器
        total_acc += acc_g * xs_g[:, None] * norm_g[None, :]

    tl.store(
        output_ptr + rb[:, None] * N + rn[None, :],
        total_acc.to(output_ptr.dtype.element_ty),
        mask=mask_b[:, None] & mask_n[None, :],
    )


# ---------------------------------------------------------------------------
# tile 配置表：先用 WxA8 表作为种子（结构同、访存量同量级），标注待扫。
# 格式: (BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages)
# ⚠ fp8 dot 与 int8 dot 的 MMA 吞吐不同（~0.79x），最优点可能偏移，
#   接线后按 kernel_autotune 流程补扫（对齐 wxa8 表的实测注释风格）。
# ---------------------------------------------------------------------------
_WXFP8_CONFIG_ATTN = {
    "small": {
        8: (64, 128, 128, 8, 3),
    },
    "large": {
        8: (128, 128, 128, 8, 3),
    },
}
_WXFP8_CONFIG_GATE_UP = {
    "small": {
        1: (64, 32, 128, 8, 3), 2: (64, 32, 128, 8, 4), 4: (64, 32, 128, 8, 3),
    },
    "large": {
        1: (256, 32, 128, 4, 2), 2: (256, 32, 128, 4, 2), 4: (256, 32, 128, 4, 2),
    },
}
_WXFP8_CONFIG_DOWN = {
    "small": {
        1: (64, 32, 128, 8, 3), 2: (64, 32, 128, 8, 3), 4: (64, 32, 128, 8, 2),
    },
    "large": {
        1: (128, 32, 128, 4, 2), 2: (256, 32, 128, 4, 2), 4: (256, 32, 128, 4, 2),
    },
}
_B_THRESHOLD_SMALL = 256
_WXFP8_DEFAULT_CONFIG = (64, 32, 128, 4, 2)


def get_wxfp8_config(bit_width: int, direction: str = "gate_up", B: int | None = None):
    """取指定 bit-width / 方向 / B 档位的 fp8 tile 配置（当前为 WxA8 种子值，待扫）。"""
    if direction == "attn":
        table = _WXFP8_CONFIG_ATTN
    elif direction == "down":
        table = _WXFP8_CONFIG_DOWN
    else:
        table = _WXFP8_CONFIG_GATE_UP
    size_key = "small" if (B is not None and B <= _B_THRESHOLD_SMALL) else "large"
    return table[size_key].get(bit_width, _WXFP8_DEFAULT_CONFIG)


def _launch_fp8(x_f8, x_scale, indices_slice, cb_f8, norms_slice,
                group_size, num_groups, bit_width, direction, cfg=None):
    """kernel 启动公共入口（对应 int8 版 _launch）。"""
    B = x_f8.shape[0]
    N = indices_slice.shape[1]
    K_total = x_f8.shape[1]

    out = torch.empty(B, N, dtype=torch.float16, device=x_f8.device)

    BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages = (
        cfg if cfg is not None
        else get_wxfp8_config(bit_width, direction=direction, B=B))

    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(N, BLOCK_N))
    _wxfp8_fused_matmul_kernel_grouped_gf[grid](
        x_f8, x_scale, indices_slice, cb_f8, norms_slice, out,
        B, N, K_total,
        indices_slice.stride(0), norms_slice.stride(0), x_scale.stride(0),
        GROUP_SIZE=group_size, NUM_GROUPS=num_groups,
        BIT_WIDTH=bit_width, N_LEVELS=cb_f8.shape[0],
        BLOCK_B=BLOCK_B, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def wxfp8_matmul_grouped_gf(
    x_f8, x_scale, indices_packed_gf, cb_f8, norms_gf,
    group_size, num_groups, bit_width: int, cfg=None,
):
    """attention 路径：完整矩阵（无切片）——对应 wxa8_matmul_grouped_gf。

    Args:
        x_f8: (B, K_total) float8_e4m3fn（rotate_quantize_fused_fp8 产物）
        x_scale: (B, num_groups) fp16（已折 cb_scale）
        indices_packed_gf: (num_groups, N, packed_per_group) uint8
        cb_f8: (n_levels,) float8_e4m3fn（build_fp8_codebook 产物）
        norms_gf: (num_groups, N) fp16，需已预乘 1/sqrt(group_size)

    Returns:
        output: (B, N) fp16
    """
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")
    return _launch_fp8(
        x_f8, x_scale, indices_packed_gf, cb_f8, norms_gf,
        group_size, num_groups, bit_width, "attn", cfg=cfg,
    )


def wxfp8_matmul_grouped_slice_rows_gf(
    x_f8, x_scale, indices_packed_gf, cb_f8, norms_gf,
    group_size, in_features, row_start, row_end, bit_width: int = 2,
    norms_prescaled: bool = False,
):
    """gate_up 路径：行切片（输出维度切片）——对应 wxa8_matmul_grouped_slice_rows_gf。

    Args:
        x_f8: (B, in_features) e4m3，已旋转并量化（rotate_quantize_fused_fp8）
        x_scale: (B, in_features // group_size) fp16（已折入 cb_scale）
        indices_packed_gf: (num_groups, N_total, packed_per_group) uint8
        cb_f8: (n_levels,) e4m3（build_fp8_codebook 产物）
        norms_gf: (num_groups, N_total) fp16，与 WxA16 的 norms_gf 同源
        row_start, row_end: 行切片范围
        norms_prescaled: norms 是否已预乘 1/sqrt(group_size)。主路径恒为 True
            （P6-2 预乘）；False 时这里补除，对齐 WxA16 语义。

    Returns:
        output: (B, row_end - row_start) fp16
    """
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")
    if in_features % group_size != 0:
        raise ValueError(
            f"WxFP8 要求 in_features ({in_features}) 对齐 group_size ({group_size})")

    num_groups = in_features // group_size
    norms_slice = norms_gf[:, row_start:row_end]
    if not norms_prescaled:
        norms_slice = norms_slice.float() / math.sqrt(group_size)
    return _launch_fp8(
        x_f8, x_scale,
        indices_packed_gf[:, row_start:row_end, :],
        cb_f8,
        norms_slice,
        group_size, num_groups, bit_width, "gate_up",
    )


def wxfp8_matmul_grouped_slice_in_features_gf(
    x_f8, x_scale, indices_packed_gf, cb_f8, norms_gf,
    group_size, original_start, original_end, bit_width: int = 2,
    norms_prescaled: bool = False,
):
    """down 路径：in_features 切片（group 维切片，整块连续）
    ——对应 wxa8_matmul_grouped_slice_in_features_gf。

    x_f8 已经是切片后的输入（每个 expert 自己的 act_out），且已用
    seed_base = seed + original_start 旋转过再量化。

    Args:
        x_f8: (B, original_end - original_start) e4m3
        x_scale: (B, num_groups_in_slice) fp16（已折入 cb_scale）
        original_start, original_end: 原始全权重的 in_features 切片范围
        norms_prescaled: 同 slice_rows_gf

    Returns:
        output: (B, N) fp16
    """
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")
    if original_start % group_size != 0:
        raise ValueError(
            f"original_start ({original_start}) 必须对齐 group_size ({group_size})")

    slice_in_features = original_end - original_start
    if slice_in_features % group_size != 0:
        raise ValueError(
            f"切片宽度 ({slice_in_features}) 必须对齐 group_size ({group_size})")

    g_start = original_start // group_size
    g_end = original_end // group_size

    norms_slice = norms_gf[g_start:g_end]
    if not norms_prescaled:
        norms_slice = norms_slice.float() / math.sqrt(group_size)
    return _launch_fp8(
        x_f8, x_scale,
        indices_packed_gf[g_start:g_end],
        cb_f8,
        norms_slice,
        group_size, g_end - g_start, bit_width, "down",
    )


# ===========================================================================
# P5-2 同型：按 bit 池合并多 expert 的 grouped kernel（2026-10-20，wxfp8 版）
#
# 结构与 triton_kernels_a8.py 的 _wxa8_merged_kernel_gf / 两个 merged wrapper
# 逐点对应（block_info 1D grid + expert_info 扁平布局 + 双模式切片），只有
# 三处按 fp8 语义换（同 _wxfp8_fused_matmul_kernel_grouped_gf 的差异清单）：
#   1. 激活 tile 是 e4m3；masked load 不给 other（fp8 字面量无法转换）
#   2. 码本是 e4m3 LUT（cb_scale 折在激活 scale 里，xs 为 fp16）
#   3. group 内直接 fp32 dot 累加（无 int8 版的 int32 中间累加器）
# 与逐 expert 的 wxfp8_matmul_grouped_slice_* 期望 bit-exact：同一 tile
# 分解、同一累加顺序。⚠ kernel 前提与逐 expert 版一致：group_size 整除
# BLOCK_K（真实模型 gs=128）——否则 K 尾部的未定义 fp8 lane 会污染有效输出。
# ===========================================================================


@triton.jit
def _wxfp8_merged_kernel_gf(
    # Input
    x_ptr,            # (B_total, K_total) e4m3 —— gather 后扁平连续
    xs_ptr,           # (B_total, NUM_GROUPS) fp16 —— 行与 x 对齐（含 cb_scale）
    # Quantized weight (group-first 布局，本 bit 池共享)
    indices_ptr,      # (NUM_GROUPS_TOTAL, N_total, PACKED_PER_GROUP) uint8
    codebook_ptr,     # (n_levels,) e4m3 —— build_fp8_codebook 的 LUT
    norms_ptr,        # (NUM_GROUPS_TOTAL, N_total) fp16（1/sqrt(gs) 已预乘）
    # Expert / block 布局
    expert_info_ptr,  # (E, 4) int32: [token_start, token_count, slice_start, slice_count]
    block_info_ptr,   # (num_blocks, 3) int32: [expert_idx, block_b, block_n]
    # Output
    output_ptr,       # (B_total, N_slice) fp16 扁平连续，行 stride = OUT_ROW_STRIDE
    # Shape / strides
    B_total, K_total, N_LIMIT,
    INDICES_G0_STRIDE, NORMS_G0_STRIDE, XS_ROW_STRIDE, OUT_ROW_STRIDE,
    # Constexpr config
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,     # 本 expert 的有效 group 数
    BIT_WIDTH: tl.constexpr,
    N_LEVELS: tl.constexpr,
    SLICE_IS_GROUPS: tl.constexpr = False,
    BLOCK_B: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 32,
    BLOCK_K: tl.constexpr = 128,
):
    """按 bit 池合并多 expert 的 WxFP8 grouped GEMM（双模式）。

    SLICE_IS_GROUPS=False（gate_up / 行切片）/ True（down / in_features 切片）
    的语义与 _wxa8_merged_kernel_gf 完全一致（见其 docstring）。
    """
    pid = tl.program_id(0)
    bi = block_info_ptr + pid * 3
    eid = tl.load(bi + 0).to(tl.int32)
    blk_b = tl.load(bi + 1).to(tl.int32)
    blk_n = tl.load(bi + 2).to(tl.int32)

    ei = expert_info_ptr + eid * 4
    token_start = tl.load(ei + 0).to(tl.int32)
    token_count = tl.load(ei + 1).to(tl.int32)
    slice_start = tl.load(ei + 2).to(tl.int32)
    slice_count = tl.load(ei + 3).to(tl.int32)

    # 静态派发下 expert_info 覆盖池内全部 expert（decode 时绝大多数
    # token_count=0）：空成员整块早退，权重流带与 dot 全免（对齐 pooled
    # rotate kernel 的写法）。host 编排路径成员恒非空，此分支不触发
    if token_count <= 0:
        return

    rb = token_start + blk_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = rb < token_start + token_count

    if SLICE_IS_GROUPS:
        # down：N 维完整（从 0 起），group 维按 slice_start 偏移
        rn = blk_n * BLOCK_N + tl.arange(0, BLOCK_N)
    else:
        # gate_up：N 维是共享权重的列切片，从 slice_start 起
        rn = slice_start + blk_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rn < N_LIMIT

    total_acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    ELEMENTS_PER_BYTE = 8 // BIT_WIDTH
    PACKED_PER_GROUP = GROUP_SIZE // ELEMENTS_PER_BYTE
    row_base = rn * PACKED_PER_GROUP

    for g in range(NUM_GROUPS):
        g_global = (slice_start + g) if SLICE_IS_GROUPS else g
        g_start = g * GROUP_SIZE
        g_base = g_global * INDICES_G0_STRIDE

        norm_g = tl.load(norms_ptr + g_global * NORMS_G0_STRIDE + rn,
                         mask=mask_n, other=0.0)
        xs_g = tl.load(xs_ptr + rb * XS_ROW_STRIDE + g, mask=mask_b, other=0.0)

        acc_g = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, GROUP_SIZE, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)
            mask_k = rk < GROUP_SIZE

            # 激活 tile：e4m3。masked load 不给 other（fp8 字面量无法转
            # fp8e4nv；被 mask 的行只影响不会写出的输出行；K 方向 gs=128
            # 整除 BLOCK_K 无尾部——见节注释的前提）
            inp_off = rb[:, None] * K_total + (g_start + rk)[None, :]
            x_tile = tl.load(x_ptr + inp_off,
                             mask=mask_b[:, None] & mask_k[None, :])

            w_mask = mask_n[:, None] & mask_k[None, :]
            if BIT_WIDTH == 8:
                byte_off = g_base + row_base[:, None] + rk[None, :]
                idx = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.int32)
            else:
                BIT_MASK = (1 << BIT_WIDTH) - 1
                byte_col = rk // ELEMENTS_PER_BYTE
                pos_in_byte = rk % ELEMENTS_PER_BYTE
                byte_off = g_base + row_base[:, None] + byte_col[None, :]
                packed = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
                shift = pos_in_byte * BIT_WIDTH
                idx = ((packed >> shift[None, :]) & BIT_MASK).to(tl.int32)

            # 码本查表 → e4m3 权重 tile（不给 other，理由同上）
            w_f8 = tl.load(codebook_ptr + idx, mask=w_mask)
            acc_g += tl.dot(x_tile, tl.trans(w_f8))

        # 出 group：乘 (激活 scale × 权重 scale)，累加进总累加器
        total_acc += acc_g * xs_g[:, None] * norm_g[None, :]

    # 输出列坐标回到扁平缓冲（down 模式 rn 已从 0 起，rows 模式回退 slice_start）
    out_col = (rn - slice_start) if not SLICE_IS_GROUPS else rn
    out_off = rb[:, None] * OUT_ROW_STRIDE + out_col[None, :]
    tl.store(output_ptr + out_off, total_acc.to(output_ptr.dtype.element_ty),
             mask=mask_b[:, None] & mask_n[None, :])


def _launch_wxfp8_merged_gf(
    x_f8, x_scale, indices_packed_gf, cb_f8, norms_used,
    expert_info_dev, block_info, out, n_extent,
    group_size, num_groups, bit_width, slice_is_groups, cfg,
):
    """wxfp8 merged kernel 统一发射入口（host 编排路径与静态派发路径共用）。

    参数序与 _wxfp8_merged_kernel_gf 形参一一对应；expert_info_dev 必须是
    (E,4) int32 设备张量（kernel 运行时读行段/列段）。
    """
    BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages = cfg
    _wxfp8_merged_kernel_gf[(block_info.shape[0],)](
        x_f8, x_scale, indices_packed_gf, cb_f8, norms_used,
        expert_info_dev, block_info, out,
        out.shape[0], x_f8.shape[1], n_extent,
        indices_packed_gf.stride(0), norms_used.stride(0), x_scale.stride(0),
        out.stride(0),
        GROUP_SIZE=group_size, NUM_GROUPS=num_groups,
        BIT_WIDTH=bit_width, N_LEVELS=cb_f8.shape[0],
        SLICE_IS_GROUPS=slice_is_groups,
        BLOCK_B=BLOCK_B, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )


def wxfp8_matmul_merged_rows_gf(
    x_f8, x_scale, indices_packed_gf, cb_f8, norms_gf,
    expert_info, group_size, in_features, bit_width: int = 2,
    norms_prescaled: bool = True, cfg=None,
):
    """gate_up 合并版（fp8）：同一 bit 池所有 expert 一次 launch（行切片语义）。

    与逐 expert 的 wxfp8_matmul_grouped_slice_rows_gf 逐段等价（数学
    bit-exact 期望）。x_f8 已按 sorted_tokens gather 成 (B_total, K) 扁平
    布局，x_scale 行对齐；expert_info 里 slice_start/slice_count 是该
    expert 在共享 indices/norms 上的列切片（主流程为 e*2*inter / 2*inter）。

    Returns:
        output: (B_total, 2*inter) fp16 扁平连续（按 expert 顺序行拼接）
    """
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")
    if in_features % group_size != 0:
        raise ValueError(
            f"WxFP8 要求 in_features ({in_features}) 对齐 group_size ({group_size})")

    expert_info = [tuple(int(v) for v in row) for row in expert_info]
    B_total = x_f8.shape[0]
    num_groups = in_features // group_size
    # 同 bit 池内 inter_size 一致 → N_slice = slice_count 统一
    n_slice = expert_info[0][3] if expert_info else 0
    if any(row[3] != n_slice for row in expert_info):
        raise ValueError(
            f"merged rows kernel 要求池内列切片一致（inter_size 相同），"
            f"得到 slice_count 集合 {sorted({row[3] for row in expert_info})}")

    norms_full = norms_gf if norms_prescaled else (
        norms_gf.float() / math.sqrt(group_size))

    BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages = (
        cfg if cfg is not None
        else get_wxfp8_config(bit_width, direction="gate_up", B=B_total))

    block_info = _build_merged_block_info(expert_info, BLOCK_B, BLOCK_N,
                                          n_slice, x_f8.device)
    out = torch.empty(B_total, n_slice, dtype=torch.float16, device=x_f8.device)
    n_limit = max(row[2] for row in expert_info) + n_slice

    _launch_wxfp8_merged_gf(
        x_f8, x_scale, indices_packed_gf, cb_f8, norms_full,
        torch.tensor(expert_info, dtype=torch.int32, device=x_f8.device),
        block_info, out, n_limit,
        group_size, num_groups, bit_width, slice_is_groups=False,
        cfg=(BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages))
    return out


def wxfp8_matmul_merged_in_features_gf(
    x_f8, x_scale, indices_packed_gf, cb_f8, norms_gf,
    expert_info, group_size, bit_width: int = 2,
    norms_prescaled: bool = True, cfg=None,
):
    """down 合并版（fp8）：同一 bit 池所有 expert 一次 launch（in_features 切片语义）。

    x_f8 已按 expert 顺序拼接成 (B_total, inter) 扁平布局（每 expert 的
    激活已用 seed+original_start 旋转并量化，v1 由调用方逐 expert 调
    rotate_quantize_fused_fp8）；expert_info 里 slice_start/slice_count 是
    group 维切片（主流程为 e*inter/gs / inter/gs），列 N 是共享的完整 hidden。

    Returns:
        output: (B_total, N) fp16 扁平连续（按 expert 顺序行拼接）
    """
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")

    expert_info = [tuple(int(v) for v in row) for row in expert_info]
    B_total = x_f8.shape[0]
    g_count = expert_info[0][3] if expert_info else 0
    if any(row[3] != g_count for row in expert_info):
        raise ValueError(
            f"merged in_features kernel 要求池内 group 切片一致（inter_size "
            f"相同），得到集合 {sorted({row[3] for row in expert_info})}")

    norms_slice = norms_gf if norms_prescaled else (
        norms_gf.float() / math.sqrt(group_size))
    N = norms_slice.shape[1]

    BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages = (
        cfg if cfg is not None
        else get_wxfp8_config(bit_width, direction="down", B=B_total))

    block_info = _build_merged_block_info(expert_info, BLOCK_B, BLOCK_N,
                                          N, x_f8.device)
    out = torch.empty(B_total, N, dtype=torch.float16, device=x_f8.device)

    _launch_wxfp8_merged_gf(
        x_f8, x_scale, indices_packed_gf, cb_f8, norms_slice,
        torch.tensor(expert_info, dtype=torch.int32, device=x_f8.device),
        block_info, out, N,
        group_size, g_count, bit_width, slice_is_groups=True,
        cfg=(BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages))
    return out


def wxfp8_matmul_merged_rows_gf_static(
    x_f8, x_scale, indices_packed_gf, cb_f8, norms_gf,
    expert_info_dev, block_info, n_slice, n_limit,
    group_size, in_features, bit_width: int = 2,
    norms_prescaled: bool = True, cfg=None,
):
    """gate_up 合并版·静态派发（CUDA graph decode 用）。

    与 wxfp8_matmul_merged_rows_gf 同一 kernel、同一 tile 数学；差别仅在
    编排来源：expert_info 与 block_info 均为设备常驻张量（行段每步设备端
    重写）。函数内零 host 读、零 H2D、零形状变化——capture 期合法。
    """
    norms_full = norms_gf if norms_prescaled else (
        norms_gf.float() / math.sqrt(group_size))
    out = torch.empty(x_f8.shape[0], n_slice, dtype=torch.float16,
                      device=x_f8.device)
    if cfg is None:
        cfg = get_wxfp8_config(bit_width, direction="gate_up", B=x_f8.shape[0])
    _launch_wxfp8_merged_gf(
        x_f8, x_scale, indices_packed_gf, cb_f8, norms_full,
        expert_info_dev, block_info, out, n_limit,
        group_size, in_features // group_size, bit_width,
        slice_is_groups=False, cfg=cfg)
    return out


def wxfp8_matmul_merged_in_features_gf_static(
    x_f8, x_scale, indices_packed_gf, cb_f8, norms_gf,
    expert_info_dev, block_info, group_size, g_count,
    bit_width: int = 2, norms_prescaled: bool = True, cfg=None,
):
    """down 合并版·静态派发（CUDA graph decode 用），契约同 rows 静态版。"""
    norms_slice = norms_gf if norms_prescaled else (
        norms_gf.float() / math.sqrt(group_size))
    N = norms_slice.shape[1]
    out = torch.empty(x_f8.shape[0], N, dtype=torch.float16,
                      device=x_f8.device)
    if cfg is None:
        cfg = get_wxfp8_config(bit_width, direction="down", B=x_f8.shape[0])
    _launch_wxfp8_merged_gf(
        x_f8, x_scale, indices_packed_gf, cb_f8, norms_slice,
        expert_info_dev, block_info, out, N,
        group_size, g_count, bit_width, slice_is_groups=True, cfg=cfg)
    return out
