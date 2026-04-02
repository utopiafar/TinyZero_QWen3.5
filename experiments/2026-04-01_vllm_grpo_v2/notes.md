# v2: save_freq 优化 + 采样日志

## 变更点

| 项目 | v1 | v2 |
|------|----|----|
| save_freq | 50 | 300 |
| TensorBoard 目录 | `/root/tf-logs/` (扁平) | `/root/tf-logs/vllm-GRPO-v2/` (子目录) |
| experiment_name | vllm-GRPO | vllm-GRPO-v2 |
| 采样日志 | 无 | 每 25 步采样 2 条 → TensorBoard text tab `train/samples` |

## v1 实验回顾 (step 1-499)

- **score**: 卡在 0.1 (format reward)，从未达到 1.0
- **format_success**: 24% → 100%（快速学会 `<answer>` 格式）
- **result_success**: ≈0%（基本没学会正确计算）
- **entropy**: 2.18 → 0.70（输出模式坍缩）
- **response_length**: 215 → 94 tokens（越来越短）
- **崩溃原因**: step 500 保存 checkpoint 时磁盘 IO 错误（95% 磁盘使用率）
- **checkpoint 总量**: 182GB (9 × 20GB)

## 训练超参（未变）

- lr: 1e-6, KL coef: 0.001, batch_size: 32, rollout n=8
- response_length: 512, max_prompt_length: 256
