"""官方 GPTQ-Int4 基线的 MoE 专家派发优化（bench_throughput.py 外部基线路径专用）。

背景（2026-10-27 定位，logs/1026.thrput.1.int4.log）：
gptqmodel 加载 GPTQ MoE 检查点时，经 defuser 把 transformers 融合 3D 专家
权重（Qwen3_5MoeExperts.gate_up_proj/down_proj）拆成 256 个 _ExpertContainer
（per-expert gate/up/down qlinear），并在每个 experts 实例上绑定 defuser 的
linear_loop_experts_forward。该 forward 的 `for e in range(num_experts)` 循环
里每个专家一次 `mask.any()` 的 bool() DtoH 同步——40 层 × 256 = 每步 10240
次同步 ≈ 232ms/步（decode 4.31 tok/s），且与 GEMM kernel 后端无关（1026 实测
marlin / exllama_v2 / triton 三后端 decode 全部 4.31）。

unique_loop 派发：与 linear_loop 数值完全一致（同样的逐专家 boolean mask
gather/写回、同样的 act_fn 门控、最后同样 reshape+sum 确定性累加），只是把
「每专家一次同步」换成「每层一次同步」：torch.unique(expert_ids).tolist()
取活跃专家集合（decode top-8 下 ≤8 个）再循环 host 整数。同步数 256/层 →
1/层；prefill（几乎全专家活跃）同样受益（每层少 255 次 any() 同步）。
"""

from types import MethodType

import torch

# 派发实现名（对齐 defuser 词汇：linear_loop = 原生行为）
LINEAR_LOOP_IMPL = "linear_loop"
UNIQUE_LOOP_IMPL = "unique_loop"


# 运行期探针：unique_loop forward 是否真的被调用过（bench warmup 后校验，
# 防「patch 打了 0 个模块」这类静默失效——1027 首版实踩）
_UNIQUE_LOOP_PROBE = {"called": False}


def reset_unique_loop_probe():
    """清零探针（bench 在 warmup 前调用，随后用 unique_loop_active 校验）。"""
    _UNIQUE_LOOP_PROBE["called"] = False


def unique_loop_active() -> bool:
    return _UNIQUE_LOOP_PROBE["called"]


def unique_loop_experts_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """linear_loop 的低同步版：活跃专家集合一次 .tolist() 取回再循环。

    与 defuser.linear_loop_experts_forward 逐算子对齐（bitwise 同结果）：
    token_idx 展开 → 一次性 gather → 逐专家 boolean mask →
    out_per_sample[mask] 写回 → 权重乘 → view(num_tokens, k, H).sum(1)。
    唯一行为差异：专家循环顺序来自 torch.unique（升序，与 linear_loop 的
    range 升序一致），且跳过 ≥ num_experts 的 sentinel id（linear_loop 靠
    range 上界天然跳过，这里在 host 侧过滤，未命中的样本行保持 0，两种
    实现的 sentinel 语义等价）。
    """
    if not _UNIQUE_LOOP_PROBE["called"]:
        _UNIQUE_LOOP_PROBE["called"] = True
        print(f"[ext] unique_loop experts forward active: "
              f"cls={type(self).__name__} num_experts={self.num_experts}",
              flush=True)
    # 兼容 [batch, seq, H] 输入（与 linear_loop 相同的展平逻辑）
    if hidden_states.dim() == 3:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        top_k_index = top_k_index.view(-1, top_k_index.size(-1))
        top_k_weights = top_k_weights.view(-1, top_k_weights.size(-1))
    else:
        batch_size, seq_len = None, None
        hidden_dim = hidden_states.size(-1)

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    num_experts = self.num_experts

    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(
        -1, num_top_k).reshape(-1)                     # (S,)
    sample_weights = top_k_weights.reshape(-1).to(hidden_states.dtype)
    expert_ids = top_k_index.reshape(-1)               # (S,)

    selected_hidden_states = hidden_states[token_idx]  # (S, hidden_dim)

    out_per_sample = torch.zeros(
        token_idx.size(0), hidden_dim, device=device, dtype=hidden_states.dtype)

    # 每层唯一一次 DtoH 同步：活跃专家 id 集合（linear_loop 为每专家一次）
    active_experts = [e for e in torch.unique(expert_ids).tolist()
                      if e < num_experts]

    # 延迟导入：与 defuser 共用同一份门控逻辑（_apply_expert_gate /
    # act_fn 回退），避免复制实现
    from defuser.modeling.moe_experts_interface import _apply_expert_gate

    for expert_idx in active_experts:
        mask = expert_ids == expert_idx
        expert_input = selected_hidden_states[mask]

        expert = getattr(self, str(expert_idx))
        if hasattr(expert, "gate_proj"):
            gate_out = expert.gate_proj(expert_input)
            up_out = expert.up_proj(expert_input)
        else:
            gate_out = None
            up_out = expert.up_proj(expert_input)
        gated_out = _apply_expert_gate(self, gate_out, up_out)
        expert_out = expert.down_proj(gated_out)

        out_per_sample[mask] = expert_out.to(out_per_sample.dtype)

    out_per_sample = out_per_sample * sample_weights.unsqueeze(-1)
    final_hidden_states = out_per_sample.view(
        num_tokens, num_top_k, hidden_dim).sum(dim=1)

    if batch_size is not None:
        final_hidden_states = final_hidden_states.view(
            batch_size, seq_len, hidden_dim)
    return final_hidden_states


_IMPL_TABLE = {}


def register_unique_loop_experts() -> bool:
    """把 unique_loop 注册进 transformers 的 ALL_EXPERTS_FUNCTIONS 注册表。

    对齐 defuser register_linear_loop_experts 的做法（写 _global_mapping）。
    Qwen3_5MoeExperts 这类带 @use_experts_implementation 装饰器的类，运行期
    由类级 forward 按 self.config._experts_implementation 从注册表取实现
    （defuser 在加载时只设了这个 config，没有实例 forward——1027 首版补丁
    按实例 MethodType 匹配，实跑 0 命中即此因）。幂等，重复调用无副作用。
    """
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    if UNIQUE_LOOP_IMPL not in ALL_EXPERTS_FUNCTIONS._global_mapping:
        ALL_EXPERTS_FUNCTIONS._global_mapping[UNIQUE_LOOP_IMPL] = \
            unique_loop_experts_forward
    return True


def patch_gptq_moe_dispatch(model, implementation: str = UNIQUE_LOOP_IMPL) -> int:
    """把 GPTQ MoE experts 的派发切到 implementation（默认 unique_loop）。

    覆盖 defuser 的两种装载形态：
    1. 实例绑定（_install_instance_forward，非装饰器 spec 路径）：forward 是
       MethodType 且 __func__ 指向已知派发 → 直接重绑；
    2. 装饰器路径（Qwen3_5MoeExperts 等）：experts 模块带 .config，运行期
       按 config._experts_implementation 从 ALL_EXPERTS_FUNCTIONS 派发 →
       注册实现后改写 config。
    implementation="linear_loop" 时等价回原生对照。返回处理的 experts 模块数。
    """
    from defuser.modeling.moe_experts_interface import (
        LINEAR_LOOP_IMPL as DEFUSER_LINEAR_LOOP,
        linear_loop_experts_forward,
    )

    impls = {
        DEFUSER_LINEAR_LOOP: linear_loop_experts_forward,
        UNIQUE_LOOP_IMPL: unique_loop_experts_forward,
    }
    if implementation not in impls:
        raise ValueError(
            f"unknown moe dispatch implementation: {implementation!r} "
            f"(expected one of {sorted(impls)})")
    register_unique_loop_experts()

    # GPTQModel 包装器 → 内层 HF 模块树（wrapper 本身可能不递归子模块）
    root = getattr(model, "model", model)
    target = impls[implementation]
    # 当前可能已绑定的已知派发（linear_loop 或已换过的 unique_loop），
    # 保证 unique_loop ⇄ linear_loop 双向切换都能找到实例
    known = set(impls.values()) | {unique_loop_experts_forward}
    n_patched = 0
    _patched_classes = set()
    for _, mod in root.named_modules():
        # 形态 1：实例绑定
        fn = getattr(mod, "forward", None)
        if fn is not None and getattr(fn, "__func__", None) in known:
            mod.forward = MethodType(target, mod)
            n_patched += 1
            _patched_classes.add(type(mod).__name__)
            continue
        # 形态 2：装饰器路径（experts 容器特征 + 带 config 的模块）
        if (type(mod).__name__.endswith("Experts")
                and isinstance(getattr(mod, "num_experts", None), int)
                and getattr(mod, "config", None) is not None):
            mod.config._experts_implementation = implementation
            n_patched += 1
            _patched_classes.add(type(mod).__name__)
    # 顶层 config 同步（有些校验/日志读顶层；派发本体只看 experts.config）
    top_cfg = getattr(root, "config", None)
    if top_cfg is not None:
        top_cfg._experts_implementation = implementation
    # 0 命中大声告警：首版补丁曾在装饰器路径下静默 0 命中，实跑白跑一轮
    if n_patched == 0 and implementation != DEFUSER_LINEAR_LOOP:
        print(f"[ext][WARN] patch_gptq_moe_dispatch({implementation!r}) "
              f"matched 0 experts modules -- dispatch NOT changed; "
              f"decode will stay on linear_loop sync path", flush=True)
    elif _patched_classes:
        print(f"[ext] patched experts classes: "
              f"{sorted(_patched_classes)}", flush=True)
    return n_patched
