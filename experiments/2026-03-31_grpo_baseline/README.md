# GRPO Baseline 实验

## 目的

首次 GRPO 训练基线实验，验证整个训练流程是否正常工作。

## 环境

| 项目 | 值 |
|------|------|
| GPU | NVIDIA RTX 5090 (32GB) |
| PyTorch | 2.10.0+cu128 |
| flash-attn | 2.8.1 |
| verl | 0.7.1 |
| 模型 | Qwen3-1.7B-Base |
| 数据 | Countdown-Tasks-3to4 |

## 关键参数

- Rollout: HF (vLLM 与 transformers>=5 不兼容)
- 策略: FSDP + CPU offload
- 采样数: n=8 (GRPO 分组)
- KL 正则: coef=0.001, low_var_kl
- 学习率: 1e-6
- 总轮次: 15 epochs
- max_prompt_length: 512
- max_response_length: 512

## 结果

(训练后填写)

## 观察和结论

(训练后填写)
