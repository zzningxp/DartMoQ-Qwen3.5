#!/usr/bin/env python3
"""Norm 融合：RMSNorm / gated RMSNorm 的 Triton 单 pass 实现 + 类级 monkeypatch。

背景（2026-09-03，实测见 test/test_attn_targets_profile.py 与 test_delta_chunk_profile.py）：
Qwen3.5 MoE 每层的 norm 是纯带宽浪费的大单块：
  - Qwen3_5MoeRMSNorm（40 层 × input/post norm + attn q/k norm + 末层 norm）
    torch 版：x.float() 扩 fp32 + pow2 中间态 + mean + rsqrt + 权重乘 + type_as，
    同一份数据要过 ~6 遍（B=32 seq=2048 时单次 ~3.5ms，attn q_norm 7.0ms）。
  - Qwen3_5MoeRMSNormGated（delta net 输出 + z gate，11.6ms/层 × 30 层）
    torch 版还要多两遍 gate 的 fp32 cast 与 silu。
Triton 版：sumsq 与 normalize 两遍逻辑都在 CTA 内完成（行数据静态展开留寄存器），
每行只从 DRAM 读一次、写一次 → 每层 ~2ms 量级。

数值语义：逐 step 镜像 torch 的 dtype 转换（见各 kernel docstring），
唯一不可避免的差异是 mean 的归约树序（tl.sum 蝶形 vs torch 向量化分段），
fp32 下相对误差 ~1e-7，对拍标准见 test/test_norm_fusion.py。

用法（主流程，接线见 eval_qwen35.py 的 patch_delta_rule 旁边）：
    from turboquant_utils.norm_kernels import patch_norms
    patch_norms(model)   # 类级 monkeypatch，对已加载模型内全部实例生效
    ...                  # unpatch_norms() 复原
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

__all__ = [
    "rms_norm",
    "rms_norm_gated",
    "patch_norms",
    "unpatch_norms",
    "norm_config",
]

# N -> (ROWS_PER_PROG, BLOCK_N, num_warps)
# 初始值由形状经验给出，test/test_norm_fusion.py 的配置扫描负责校准
norm_config = {
    128:  (8, 128, 4),    # delta net gated norm（M = B*S*32 ≈ 2M 行）
    256:  (8, 256, 4),    # attn q/k norm（M = B*S*16 / B*S*2）
    2048: (2, 512, 8),    # 层输入/输出 norm（M = B*S）
}

_TL_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def _triton_ok(x):
    """Triton 路径的适用条件：CUDA、行内元素连续、行距嵌套一致、行宽在配置表内、
    浮点 dtype。

    注意不用 is_contiguous()：attention 的 q_norm 输入是 torch.chunk 产生的
    strided view（行距 512 > 行宽 256），torch 原版照常处理。kernel 把前导维度
    拍平成单一行索引，要求地址对行号线性：stride(i) ==
    prod(size(i+1..-2)) × stride(-2)。chunk view 满足（外层 stride 继承父张量），
    不满足的奇异布局回退 torch。
    """
    if not (x.is_cuda and x.dim() >= 2 and x.stride(-1) == 1
            and x.size(-1) in norm_config and x.dtype in _TL_DTYPE):
        return False
    prod = x.stride(-2)
    for d in range(x.dim() - 3, -1, -1):
        prod *= x.size(d + 1)
        if x.stride(d) != prod:
            return False
    return True


# ===========================================================================
# kernel
# ===========================================================================

@triton.jit
def _rmsnorm_kernel(
    x_ptr, w_ptr, out_ptr,
    M,
    row_stride,           # x 的行 stride（元素数；连续时 = N）
    out_row_stride,       # out 的行 stride（元素数）
    eps,
    ROWS_PER_PROG: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """Qwen3_5MoeRMSNorm 的 Triton 等价：out = (x*rstd) * (1+w)，末尾一次 cast。

    镜像 torch 语义（modeling_qwen3_5_moe.py Qwen3_5MoeRMSNorm.forward）：
      x.float() → pow2.mean → x*rsqrt(v+eps) → *(1+weight.float()) → type_as(x)
    权重乘与归一化都在 fp32 完成，仅最终一次 cast 到 OUT_DTYPE。
    与 torch 的差异仅 mean 归约树序（fp32 ~1e-7 相对误差）。
    """
    pid = tl.program_id(0)
    rows = pid * ROWS_PER_PROG + tl.arange(0, ROWS_PER_PROG)
    rmask = rows < M
    cols = tl.arange(0, BLOCK_N)

    sumsq = tl.zeros((ROWS_PER_PROG,), dtype=tl.float32)
    for c in tl.static_range(0, N, BLOCK_N):
        x = tl.load(x_ptr + rows[:, None] * row_stride + (c + cols)[None, :],
                    mask=rmask[:, None], other=0.0).to(tl.float32)
        sumsq += tl.sum(x * x, axis=1)
    # libdevice.rsqrt（__nv_rsqrtf）与 torch CUDA 的 rsqrt 同源，tl.rsqrt 是另一近似
    rstd = libdevice.rsqrt(sumsq / N + eps)

    for c in tl.static_range(0, N, BLOCK_N):
        x = tl.load(x_ptr + rows[:, None] * row_stride + (c + cols)[None, :],
                    mask=rmask[:, None], other=0.0).to(tl.float32)
        w = tl.load(w_ptr + c + cols).to(tl.float32)
        y = x * rstd[:, None] * (1.0 + w)[None, :]
        tl.store(out_ptr + rows[:, None] * out_row_stride + (c + cols)[None, :],
                 y.to(OUT_DTYPE), mask=rmask[:, None])


@triton.jit
def _rmsnorm_gated_kernel(
    x_ptr, gate_ptr, w_ptr, out_ptr,
    M,
    row_stride,           # x 的行 stride（元素数；连续时 = N）
    out_row_stride,       # out 的行 stride（元素数）
    gate_row_stride,
    eps,
    ROWS_PER_PROG: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    WEIGHT_LOWPREC: tl.constexpr,
):
    """Qwen3_5MoeRMSNormGated 的 Triton 等价：out = (x*rstd)*w*silu(gate)。

    镜像 torch 语义（modeling_qwen3_5_moe.py Qwen3_5MoeRMSNormGated.forward）：
      x32  = x.float(); x32 = x32 * rsqrt(mean(x32²)+eps)
      y    = weight * x32.to(input_dtype)      # weight 也是低精度时 → 低精度（RN 舍入）
      out  = (y * silu(gate.float())).to(input_dtype)   # fp32 乘，末尾 cast
    注意 torch 的 dtype 提升：weight 为 fp32 时 step2 结果保持 fp32（无中间舍入），
    weight 与输入同为 bf16/fp16 时才在 bf16/fp16 域乘（真实模型是这种情况）。
    WEIGHT_LOWPREC 由包装函数按此规则传入；kernel 内显式 .to(OUT_DTYPE)
    复现每个 step 的舍入，与 torch 逐 step 一致。
    """
    pid = tl.program_id(0)
    rows = pid * ROWS_PER_PROG + tl.arange(0, ROWS_PER_PROG)
    rmask = rows < M
    cols = tl.arange(0, BLOCK_N)

    sumsq = tl.zeros((ROWS_PER_PROG,), dtype=tl.float32)
    for c in tl.static_range(0, N, BLOCK_N):
        x = tl.load(x_ptr + rows[:, None] * row_stride + (c + cols)[None, :],
                    mask=rmask[:, None], other=0.0).to(tl.float32)
        sumsq += tl.sum(x * x, axis=1)
    rstd = libdevice.rsqrt(sumsq / N + eps)

    for c in tl.static_range(0, N, BLOCK_N):
        x = tl.load(x_ptr + rows[:, None] * row_stride + (c + cols)[None, :],
                    mask=rmask[:, None], other=0.0).to(tl.float32)
        g = tl.load(gate_ptr + rows[:, None] * gate_row_stride + (c + cols)[None, :],
                    mask=rmask[:, None], other=0.0).to(tl.float32)
        w = tl.load(w_ptr + c + cols).to(tl.float32)
        xn = x * rstd[:, None]
        if WEIGHT_LOWPREC:
            # step2（低精度 weight）：xn 与 w 先在 OUT_DTYPE 域乘（RN 舍入）
            y = (xn.to(OUT_DTYPE).to(tl.float32) * w).to(OUT_DTYPE)
        else:
            # step2（fp32 weight）：torch 提升到 fp32，只对 xn 有一次舍入
            y = xn.to(OUT_DTYPE).to(tl.float32) * w
        # silu 保持 fp32，用 torch CUDA silu 的原公式 x/(1+exp(-x))；
        # libdevice.exp（= __nv_expf，与 torch 的 expf 同源）比 tl.exp 更贴近
        # （实测 fp32 silu 差 2 ulp vs 8 ulp）
        sg = g / (1.0 + libdevice.exp(-g))
        out = y.to(tl.float32) * sg
        tl.store(out_ptr + rows[:, None] * out_row_stride + (c + cols)[None, :],
                 out.to(OUT_DTYPE), mask=rmask[:, None])


# ===========================================================================
# 包装函数
# ===========================================================================

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3_5MoeRMSNorm.forward 的 Triton 等价（形状不适用时回退 torch 语义）。

    调用方保证 x 满足 _triton_ok；不满足时本函数不做 Triton 计算，
    走 torch 等价式（与 modeling 文件里 _norm + (1+weight) 完全一致）。
    """
    if not _triton_ok(x):
        # 回退：与 Qwen3_5MoeRMSNorm.forward 完全一致的 torch 公式
        #（必须先 x.float() 再 pow2，bf16 域平方会丢精度）
        out = x.to(torch.float32)
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
        return (out * (1.0 + weight.float())).type_as(x)

    N = x.size(-1)
    M = x.numel() // N
    rows_per_prog, block_n, num_warps = norm_config[N]
    # 输出用连续 buffer（torch elementwise 对 strided 输入也产出连续张量，行为一致）
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(M, rows_per_prog),)
    _rmsnorm_kernel[grid](
        x, weight, out, M,
        row_stride=x.stride(-2), out_row_stride=N, eps=float(eps),
        ROWS_PER_PROG=rows_per_prog, N=N, BLOCK_N=block_n,
        OUT_DTYPE=_TL_DTYPE[x.dtype],
        num_warps=num_warps,
    )
    return out


def rms_norm_gated(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor,
                   eps: float) -> torch.Tensor:
    """Qwen3_5MoeRMSNormGated.forward 的 Triton 等价（形状不适用时回退 torch 语义）。"""
    if not (_triton_ok(x) and _triton_ok(gate) and gate.shape == x.shape):
        x32 = x.to(torch.float32)
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
        x32 = weight * x32.to(x.dtype)
        return (x32 * torch.nn.functional.silu(gate.to(torch.float32))).to(x.dtype)

    N = x.size(-1)
    M = x.numel() // N
    rows_per_prog, block_n, num_warps = norm_config[N]
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(M, rows_per_prog),)
    _rmsnorm_gated_kernel[grid](
        x, gate, weight, out, M,
        row_stride=x.stride(-2), out_row_stride=N,
        gate_row_stride=gate.stride(-2), eps=float(eps),
        ROWS_PER_PROG=rows_per_prog, N=N, BLOCK_N=block_n,
        OUT_DTYPE=_TL_DTYPE[x.dtype],
        WEIGHT_LOWPREC=(weight.dtype == x.dtype and x.dtype in (torch.bfloat16, torch.float16)),
        num_warps=num_warps,
    )
    return out


# ===========================================================================
# 类级 monkeypatch（eval 接线用）
# ===========================================================================

_ORIG_FORWARDS = {}
_PATCH_CALL_COUNT = 0   # 测试确认「接线生效」用（test/test_norm_fusion.py 断言递增）


def _triton_rmsnorm_forward(self, x):
    if _triton_ok(x):
        global _PATCH_CALL_COUNT
        _PATCH_CALL_COUNT += 1
        return rms_norm(x, self.weight, self.eps)
    return _ORIG_FORWARDS[type(self)](self, x)


def _triton_gated_forward(self, hidden_states, gate=None):
    if gate is not None and _triton_ok(hidden_states):
        global _PATCH_CALL_COUNT
        _PATCH_CALL_COUNT += 1
        return rms_norm_gated(hidden_states, gate, self.weight, self.variance_epsilon)
    return _ORIG_FORWARDS[type(self)](self, hidden_states, gate)


def patch_norms(module=None):
    """类级 monkeypatch：Qwen3_5MoeRMSNorm / Qwen3_5MoeRMSNormGated 的 forward → Triton 版。

    与 patch_delta_rule 的实例级替换不同，这里按类替换：eval 的 sequential 模式
    会把层在 CPU/GPU 间搬移，类级替换不受搬移影响，对已加载模型内全部实例生效。
    幂等；unpatch_norms() 复原。module 参数仅为与 patch_delta_rule 调用习惯对齐。
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeRMSNorm,
        Qwen3_5MoeRMSNormGated,
    )

    if Qwen3_5MoeRMSNorm not in _ORIG_FORWARDS:
        _ORIG_FORWARDS[Qwen3_5MoeRMSNorm] = Qwen3_5MoeRMSNorm.forward
        Qwen3_5MoeRMSNorm.forward = _triton_rmsnorm_forward
    if Qwen3_5MoeRMSNormGated not in _ORIG_FORWARDS:
        _ORIG_FORWARDS[Qwen3_5MoeRMSNormGated] = Qwen3_5MoeRMSNormGated.forward
        Qwen3_5MoeRMSNormGated.forward = _triton_gated_forward
    return module


def unpatch_norms():
    """复原 patch_norms 的类级替换（幂等）。"""
    for cls, orig in list(_ORIG_FORWARDS.items()):
        cls.forward = orig
    _ORIG_FORWARDS.clear()
