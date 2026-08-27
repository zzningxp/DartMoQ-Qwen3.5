#!/usr/bin/env python3
"""P8-1: attention 分阶段 CUDA event 测量（P6-0 方法移植）。

对 transformers 内置的 Qwen3.5MoE attention 模块做运行时包装（monkeypatch），
在既有计算调用前后打 CUDA event，**不改动任何计算逻辑**。
包装可逆（unpatch），未启用时零开销。

背景：仓库根的 modeling_qwen3_5_moe.py 与 transformers 内置版逐字节相同，
主流程实际执行的是内置版；为避免改 site-packages，测量走包装方式。

用法（测试，见 test/test_p81_attn_profile.py）:
    patch_attn_profiling(attn_module)
    out = attn_module(x)
    print(CudaStageProfiler.format_stats(attn_module.last_attn_stats))

用法（真实模型，由本人手动在 eval 里加一行）:
    from turboquant_utils.attn_profile import patch_attn_profiling
    patch_attn_profiling(model)  # 对模型内全部 attention 实例生效
    注意：测量开启后每次 forward 首尾会 synchronize，绝对耗时不可与正式 eval 比，
    只看各 stage 占比。测完 unpatch_attn_profiling(model) 复原。
"""

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeAttention,
)

from turboquant_utils.cuda_profiler import CudaStageProfiler

# id(instance) -> [(kind, ...)] 复原记录
# kind == "submodule": (attr, sub, orig_forward)
# kind == "callable" : (attr, orig_fn)
_PATCHED = {}


def _stage_call(prof: CudaStageProfiler, name: str, fn):
    """把任意 callable 包一层 stage 计时。prof 关闭时是零开销透传。"""

    def wrapped(*args, **kwargs):
        with prof.stage(name):
            return fn(*args, **kwargs)

    return wrapped


def _make_profiled_forward(prof: CudaStageProfiler, module, orig_forward):
    """替换模块 forward，只在首尾补 begin_round / end_round，不改内部逻辑。"""

    def profiled(*args, **kwargs):
        prof.begin_round()
        try:
            return orig_forward(*args, **kwargs)
        finally:
            if prof.enabled:
                module.last_attn_stats = prof.end_round()

    return profiled


def patch_attn_profiling(module, enabled=True):
    """给 module（或整个模型）里所有 Qwen3.5MoE attention 实例挂 CUDA event 计时。

    包装对象：
      GatedDeltaNet: in_proj_qkv / in_proj_z / in_proj_b / in_proj_a /
                     conv1d（torch fallback）或 causal_conv1d_fn / causal_conv1d_update /
                     chunk_gated_delta_rule / recurrent_gated_delta_rule /
                     norm / out_proj
      MoeAttention : q_proj / k_proj / v_proj / q_norm / k_norm / o_proj
    """
    targets = [
        m for m in module.modules()
        if isinstance(m, (Qwen3_5MoeGatedDeltaNet, Qwen3_5MoeAttention))
    ]

    for m in targets:
        if id(m) in _PATCHED:
            continue  # 已包装过，避免双重包装
        prof = CudaStageProfiler(enabled=bool(enabled))
        m.attn_prof = prof
        m.attn_prof_enabled = bool(enabled)
        m.last_attn_stats = {}

        records = []
        orig_forward = m.forward
        records.append(("forward", None, m, orig_forward))
        m.forward = _make_profiled_forward(prof, m, orig_forward)

        # 子模块 forward 包装（注意：包装的是 forward 而不是 __call__——
        # nn.Module._call_impl 动态查 self.forward，包装 __call__ 不生效）
        if isinstance(m, Qwen3_5MoeGatedDeltaNet):
            for attr in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                         "conv1d", "norm", "out_proj"):
                sub = getattr(m, attr)
                records.append(("submodule", attr, sub, sub.forward))
                sub.forward = _stage_call(prof, attr, sub.forward)
            # 快速路径函数（当前环境 causal_conv1d_fn 为 None，chunk 走 torch 实现）
            for attr, stage in (
                ("causal_conv1d_fn", "conv1d_fast"),
                ("causal_conv1d_update", "conv1d_update_fast"),
                ("chunk_gated_delta_rule", "delta_rule_chunk"),
                ("recurrent_gated_delta_rule", "delta_rule_recurrent"),
            ):
                fn = getattr(m, attr)
                if fn is not None:
                    wrapped = _stage_call(prof, stage, fn)
                    records.append(("callable", attr, fn))
                    setattr(m, attr, wrapped)
        elif isinstance(m, Qwen3_5MoeAttention):
            for attr in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj"):
                sub = getattr(m, attr)
                records.append(("submodule", attr, sub, sub.forward))
                sub.forward = _stage_call(prof, attr, sub.forward)

        _PATCHED[id(m)] = records
    return targets


def unpatch_attn_profiling(module):
    """复原 patch_attn_profiling 的所有包装。"""
    for m in module.modules():
        if id(m) not in _PATCHED:
            continue
        records = _PATCHED.pop(id(m))
        m.attn_prof_enabled = False
        for kind, *rest in records:
            if kind == "submodule":
                attr, sub, orig_forward = rest
                sub.forward = orig_forward
            elif kind == "forward":
                _, m_self, orig_forward = rest
                m_self.forward = orig_forward
            else:
                attr, orig_fn = rest
                setattr(m, attr, orig_fn)
    return module
