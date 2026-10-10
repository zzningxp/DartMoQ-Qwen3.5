"""triton 多卡 device dispatch shim。

问题（triton 3.8 + 驱动 570，二卡实测）：triton 驱动按"进程内首次
launch"建立指针可访问集，之后用另一张卡的张量 launch 会误报
`ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)`
——实际不是 CPU 张量，是驱动没刷新设备上下文（顺序依赖：先 launch 过
cuda:0，再 launch cuda:1 必崩；反之亦然）。

绕过：给 JITFunction.run 打补丁，launch 前把 torch 当前设备切到第一个
CUDA 张量实参所在设备（`torch.cuda.device()` 上下文，launch 后自动恢复）。
实测 set_device 后同参数 launch 成功。对单卡/同卡连续 launch 是零开销
no-op。triton 版本升级修复后可整体移除（shim 幂等，重复安装无害）。
"""

import torch
import triton.runtime.jit as _tjit

_ORIG_RUN = _tjit.JITFunction.run


def _run_with_device(self, *args, **kwargs):
    # triton 内部存在多种调用形态（grid 位置传/keyword 传、warmup 亦同），
    # 全捕获签名避免绑定冲突；设备从第一个 CUDA 张量实参推断
    device = None
    for a in args:
        if isinstance(a, torch.Tensor) and a.is_cuda:
            device = a.device
            break
    if device is None:
        for v in kwargs.values():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                device = v.device
                break
    if device is not None:
        with torch.cuda.device(device):
            _ORIG_RUN(self, *args, **kwargs)
            return
    _ORIG_RUN(self, *args, **kwargs)


_tjit.JITFunction.run = _run_with_device
print("[shim] triton multi-GPU device dispatch installed", flush=True)