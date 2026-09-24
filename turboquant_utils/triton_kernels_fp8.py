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

import torch
import triton
import triton.language as tl

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
