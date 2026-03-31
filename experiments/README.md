# TinyZero Qwen3 GRPO 实验记录

## 项目目标

用 Qwen3-1.7B-Base 复现 TinyZero 的 grokking 现象 —— 模型在 Countdown 算术任务上通过 GRPO 训练，从随机猜测突然跃迁到正确求解。

## 实验时间线

### 2026-03-30: 初次尝试

| 实验 | 引擎 | 策略 | verl | 结果 |
|------|------|------|------|------|
| `2026-03-30_qwen3_grpo` | HF | fsdp | 0.7.1 | 训练启动但 outputs 日志为空（多次尝试均未成功写入） |
| `2026-03-30_baseline` | HF | fsdp | 0.7.1 | 同上，添加了 `reward.custom_reward_function.path` |
| `2026-03-30_vllm_grpo` | vLLM | fsdp2 | 0.7.1 | **失败**: vLLM 要求 transformers<5，Qwen3 需要 transformers>=5.2 |

### 2026-03-31: 调试与迭代

| 实验 | 引擎 | 策略 | verl | 结果 |
|------|------|------|------|------|
| `2026-03-31_grpo_baseline` | HF | fsdp + CPU offload | 0.7.1 | 添加了 `use_legacy_worker_impl=enable`，仍在调试中 |
| `2026-03-31_countdown_grpo` | - | - | - | 空目录（未实际运行） |
| `2026-03-31_grpo_countdown` | HF | fsdp2 | 0.7.1 | 最新配置，添加了 dynamic batch + TensorBoard 补丁 |

## 关键发现

### 有效方案

1. **verl v0.7.1 + HF rollout**: 当前服务器环境（Python 3.12 + torch 2.10 + transformers 5.4）下，HF rollout 是唯一可行的推理引擎
2. **FSDP2 策略**: 比 FSDP 更高效，支持 `use_dynamic_bsz=True`
3. **自定义 Reward 注入**: 通过 `reward.custom_reward_function.path` 指向 `src/reward/countdown.py`
4. **TensorBoard 指标补丁**: `entry.py` monkey-patch 了 `compute_data_metrics`，记录 `format_success_rate` 和 `result_success_rate`
5. **ChatML 模板**: 使用 `apply_chat_template` + `enable_thinking=True` 生成 Qwen3 格式的 prompt

### 无效/失败的尝试

1. **vLLM Rollout**: vLLM (截至 0.18.0) 与 transformers>=5 不兼容，无法用于 Qwen3
   - 追踪: https://github.com/vllm-project/vllm/issues/30466
2. **verl v0.4.0 方案**: CoLA-RL 教程的配置（torch 2.6 + vllm 0.8.5 + Python 3.10）与当前服务器环境不匹配
   - flash-attn/flashinfer 预编译 wheel 与 torch 2.10 不兼容
   - 降级整个环境风险太大
3. **Outputs 日志为空**: 所有 20+ 次训练运行的 `outputs/*/main_ppo.log` 都是空文件
   - 可能原因: verl Hydra 配置问题，或训练在 log 写入前崩溃
   - TensorBoard 日志目录有数据，但内容待确认

## 当前状态

- **环境**: AutoDL RTX 5090 32GB, Python 3.12, torch 2.10, verl 0.7.1
- **最新配置**: `experiments/2026-03-31_grpo_countdown/run.sh`
- **未解决**: 训练是否能正常完成并产生收敛结果尚未验证
- **下一步**:
  1. 确认 TensorBoard 日志中是否有有效训练指标
  2. 如果训练未成功，检查 console 输出日志排查错误
  3. 考虑降级到 verl v0.4.0 + torch 2.6 方案（需要新建 conda 环境）

## 技术要点备忘

### Reward 函数签名 (verl v0.7.1)
```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """返回 dict: {"score": float, "format_success": int, "result_success": int}"""
```

### 数据格式
```python
{
    "data_source": "countdown",
    "prompt": [{"content": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n", "role": "user"}],
    "reward_model": {"style": "rule", "ground_truth": {"target": 98, "numbers": [44, 19, 35]}}
}
```

### 显存管理 (RTX 5090 32GB)
- 1.7B 模型 ~3.4GB (bf16)
- FSDP + CPU offload 可将训练占用控制在 ~16GB
- vLLM 需要额外 ~8GB 用于 KV cache，单 GPU 勉强不够
