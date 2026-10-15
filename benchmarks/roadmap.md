# 20261010 lm-eval 下游任务评测开发 roadmap（P3 落地）

> 用途：P3（对齐 BitsMoE 七任务）的实现计划与落地记录。
> 计划在 2026-10-10 与 Claude 结对制定并实现完成；本文为其存档 + 手动验证指引。
> 关联文档：[20261010_论文构思_评测方法与基准对照计划.md](20261010_论文构思_评测方法与基准对照计划.md) §三 P3。

## 一、背景与决策

- 目标：量化 checkpoint（WxA16/WxA8/WxFP8/WxFP4 同一份）直接跑 lm-eval-harness
  下游任务，对齐 BitsMoE 七任务口径（five-shot）：
  hellaswag / arc_easy / mathqa / mmlu / openbookqa / winogrande / gsm8k。
- 上游 DartMoQ 的 `eval_zero_shot()` 走 HFLM 透传，但其日志可证**从未在
  `--load-quantized` checkpoint 上跑通过**，gsm8k 生成在旧 modeling 上崩过
  （DynamicCache.seen_tokens）。故本仓库不复用该路径。
- 决策（本人确认）：
  1. 独立脚本 `eval_tasks.py`，不集成进 eval_qwen35.py；
  2. **评测路径不依赖 HFLM**——自定义 `DartMoQLM`（lm-eval `TemplateLM` 子类）
     直接驱动量化模型；HFLM 仅在 parity 测试里作对照真值；
  3. lm-eval 版本「都试试」：先 dart312-t38 装 **0.4.12** 跑通拿数字；
     上游实际钉的是 **0.4.9.2**（注意：论文构思文档里写的 0.4.9.1 有误，
     上游 environment.yml 实为 0.4.9.2），后续克隆环境做版本敏感性对照。
     已验证 0.4.12 与 transformers 5.13.0 / torch 2.9 / triton 3.8 共存无冲突。

## 二、实现要点（已落地）

新文件两个，未改动任何现有文件：

- `eval_tasks.py`（仓库根）
  - `DartMoQLM(TemplateLM)`：实现 `tok_encode` / `_loglikelihood_tokens` /
    `generate_until` / `loglikelihood_rolling` 五个抽象成员；上下文编码与
    空上下文 prefix 处理继承自 TemplateLM（源码核实于 0.4.12）。
  - **关键设计**：`model.model(...)` 只取 hidden，仅在 continuation 预测位
    过 lm_head（词表 ~248K，全量 logits 约 600MB/样本不可行）；按
    `tuple(inp)` 分组去重，mmlu/arc/obqa 的单 token 选项共享上下文 →
    每 doc 一次 forward；右填充 + attention_mask（Qwen3.5 建模按 mask
    cumsum 生成 position_ids，padding 正确性已在 tiny 模型实测）。
  - lm_head 评分分块（默认 256 位/块）+ fp32 CE；OOM 对半降批重试。
  - generate_until：左填充、`stop_sequences_criteria` + 解码后截断兜底、
    `use_cache=True` try/finally 翻转。
  - 守卫：standby-CPU 回退（模型未整驻 GPU）直接 SystemExit；两卡切分
    自动 bs→8；patch_delta_rule/patch_norms 与 eval_qwen35.py 同款
    `DARTMOQ_SKIP_*` 环境阀；逐层 `mlp.warmup_kernels` Triton JIT 预热。
  - 逐任务 try/except 隔离（gsm8k 失败不阻塞前六任务）；
    结果 JSON 落 `results/tasks_<ckpt>_<mode>_<k>shot_<githash>_<ts>.json`
    （含 git hash、lm-eval/transformers/torch 版本、口径参数）。
- `test/test_eval_tasks_lm.py`
  - tiny 随机 Qwen3.5 MoE（4 层混合注意力，~16M 参数，CPU fp32，
    必须用 `Qwen3_5MoeTextConfig` 直构，外层 VL config 不转发关键字段）
    与 HFLM 对拍；不依赖量化/Triton 组件。
  - Stage A：手造 Instance（空上下文/单 token/多 token/尾空格/>8 条多批）
    loglikelihood allclose + is_greedy 相等；generate_until 精确串相等
    （我方 bs=8/4 带 padding vs HFLM bs=1 无填充真值）。
  - Stage B：`simple_evaluate` 端到端 arc_easy（0-shot, limit=10），
    acc/acc_norm 必须精确相等。

## 三、验证状态

- [x] `conda run -n dart312-t38 pip install lm_eval==0.4.12`（钉死版本未被动）
- [x] parity 自测全绿（2026-10-10，dart312-t38）：
  - Stage A loglikelihood：10/10 对上，max |Δlogprob| = 1.43e-05
  - Stage A generate_until：4/4 精确相等（KV-cache 生成 + 左填充正确）
  - Stage B arc_easy 端到端：acc=0.2000 / acc_norm=0.1000，与 HFLM 精确相等
- [ ] 35B 真实 checkpoint 冒烟（手动，见下）
- [ ] 七任务正式数字（手动）
- [ ] gsm8k 首跑（本仓库首次走 `.generate`：量化 MoE 的 T=1 解码形状、
      混合注意力 KV cache 均未在 decode 下验证过——失败不阻塞前六任务）

## 四、手动运行指引（主流程，占大量显存）

```bash
# 冒烟：单任务小样本（建议先跑这个）
python eval_tasks.py models/qwen3.5-2bpw-260930-s1 \
    --inference-quant-mode wxa8 --tasks arc_easy --limit 20 --num-fewshot 5

# 六个 loglikelihood 任务（每个任务独立 simple_evaluate，失败隔离）
python eval_tasks.py models/qwen3.5-2bpw-260930-s1 \
    --inference-quant-mode wxa8 \
    --tasks hellaswag arc_easy mathqa mmlu openbookqa winogrande \
    --num-fewshot 5 --batch-size 8

# gsm8k 单独最后跑（生成任务首测；--gen-batch-size 保持 1）
python eval_tasks.py models/qwen3.5-2bpw-260930-s1 \
    --inference-quant-mode wxa8 --tasks gsm8k --num-fewshot 5

# 全默认（七任务，gsm8k 排最后）
python eval_tasks.py models/qwen3.5-2bpw-260930-s1 --inference-quant-mode wxa8
```

环境注意：

- wxfp4 模式必须 dart312-t38 环境（triton 3.8 + ptxas 12.8）。
- 122B（40GB checkpoint）需双卡：自动两卡切分，bs 自动收敛 8；
  单卡会回退 standby CPU 并直接报错退出（任务评测需整模型驻留）。
- **本机 HF hub 访问会挂起**（元数据刷新不超时）。数据集缓存现状：
  arc/mmlu/gsm8k/winogrande 已就绪；hellaswag/mathqa 仅 hub 快照；
  **openbookqa 完全没有，需联网下载一次**。建议：
  1. 先联网跑 `--tasks openbookqa hellaswag mathqa --limit 2` 把数据拉全；
  2. 正式测全部加 `HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1` 前缀（parity
     测试实测离线变量下可正常命中缓存）。
- 首轮跑新形状仍会触发 Triton JIT（预热只覆盖 1024/2048 档），
  端到端首跑偏慢属正常；正式计时不影响（任务评测只看精度不看速度）。

## 五、后续（未做，按需推进）

- lm-eval 0.4.9.2 版本敏感性对照（克隆 dart312-t38 环境装 0.4.9.2，
  与 0.4.12 对比 1-2 个任务分数；若与 transformers 5.13 冲突则记录并放弃，
  论文口径注明 0.4.12）。
- fp16 基线七任务（35B fp16 ≈70GB，需流式/offload 方案，另行立项）。
- 结果回填论文构思文档 §2.4 我们行 + 与 BitsMoE 表同列对照。
