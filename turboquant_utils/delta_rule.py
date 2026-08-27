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
    with _stage("cast"):
        if use_qk_l2norm_in_kernel:
            query = _l2norm(query, dim=-1, eps=1e-6)
            key = _l2norm(key, dim=-1, eps=1e-6)
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
    with _stage("wy_bmm"):
        if ENABLE_FP16_BMM:
            attn = -((k_beta.half() @ key.transpose(-1, -2).half()).float() * decay_mask).masked_fill(mask, 0)
        else:
            attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    if ENABLE_WY_SOLVE:
        with _stage("wy_solve"):
            # P8-3: 三角递归 L_new = L + L·L_new 的闭式解 L_new = (I−L)⁻¹L。
            # attn 为严格下三角 L；I−L 单位下三角，批量三角求解一次完成。
            # 数学与原 63 次循环等价（浮点舍入顺序不同，误差 ~1e-4~1e-3 量级）。
            eye = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
            i_minus_l = eye - attn
            attn = torch.linalg.solve_triangular(i_minus_l, attn,
                                                 upper=False, unitriangular=True) + eye
    else:
        with _stage("wy_loop"):
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
            core_attn_out=core_attn_out, last_recurrent_state=last_recurrent_state)

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


def _fast_chunk_loop(query, key, value, k_cumdecay, decay_mask, g, chunk_size,
                     total_sequence_length, core_attn_out, last_recurrent_state, **kwargs):
    """fast_chunk_gated_delta_rule 的 chunk 串行循环（独立函数便于消融）。

    返回更新后的 (last_recurrent_state, core_attn_out)。
    """
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
