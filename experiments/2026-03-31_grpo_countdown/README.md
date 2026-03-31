# GRPO Countdown Qwen3-1.7B-Base 实验

## 目的
参照 CoLA-RL 教程（verl v0.4.0）的完整配置，训练 Qwen3-1.7B-Base 完成 Countdown 任务。

## 关键变更（从旧配置迁移）
- **verl**: v0.7.1 → v0.4.0（源码方式，复制到项目）
- **torch**: 2.9.0 → 2.6.0
- **vllm**: 无 → 0.8.5.post1（用于 rollout）
- **Python**: 3.12 → 3.10
- **Rollout**: HF → vllm
- **Strategy**: fsdp → fsdp2
- **Reward**: 通过 verl/utils/reward_score/ 注册
- **数据模板**: 手动 ChatML → apply_chat_template + enable_thinking=True
- **Dynamic batch**: 关闭 → 开启
