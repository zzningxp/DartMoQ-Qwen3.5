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
