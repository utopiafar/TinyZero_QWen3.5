# CLAUDE.md — 项目指南

## 项目简介

GRPO 学习项目，目标是用 Qwen3-1.7B-Base 复现 TinyZero 的 grokking 现象。

最初参照 CoLA-RL 教程（verl v0.4.0 + vLLM rollout），但因服务器环境兼容性问题，
最终使用 verl v0.7.1 + HF rollout + FSDP2 的方案。

详细实验记录见 `experiments/README.md`。

## 项目结构

```
TinyZero_QWen3.5/
├── CLAUDE.md                # 本文件 — 项目指南
├── pyproject.toml           # 包配置
├── requirements.txt         # 依赖版本
├── entry.py                 # 训练入口（注入 TensorBoard 指标补丁）
├── docs/
│   └── setup_guide.md       # 服务器环境安装指南
├── src/                     # 核心代码
│   ├── reward/countdown.py  # Countdown 奖励函数 (1.0/0.1/0.0)
│   ├── templates/countdown.py  # ChatML 模板
│   └── utils/tokenizer.py   # Qwen3 tokenizer 工具
├── scripts/
│   ├── setup_env.sh         # 环境安装脚本
│   ├── prepare_data.py      # 数据预处理（apply_chat_template + enable_thinking=True）
│   └── train_grpo.sh        # 通用训练脚本（支持 HF/vLLM 切换）
└── experiments/             # 实验记录
    ├── README.md            # 全部实验总结
    └── <date>_<name>/       # 各实验配置和笔记
```

## 环境要求

**当前实际环境: AutoDL base conda (Python 3.12)**

| 包 | 版本 | 说明 |
|---|------|------|
| python | 3.12 | 服务器默认 |
| torch | 2.10.0+cu128 | |
| verl | 0.7.1 | PyPI 安装 |
| transformers | 5.4.0 | Qwen3 需要 >=5.2 |
| flash-attn | 2.8.1 | GitHub 预编译 wheel |

> **注意**: 原计划的 `tinyzero` conda 环境 (Python 3.10 + torch 2.6 + verl v0.4.0 + vLLM 0.8.5)
> 因预编译 wheel 兼容性问题未能完成。如果需要使用 vLLM rollout，
> 需要等待 vLLM 支持 transformers>=5，或搭建独立的 Python 3.10 环境。

## 关键技术信息

### Reward 规则 (data_source='countdown')

- `1.0`: 等式正确（数字正确 + 结果正确）
- `0.1`: 格式正确（有 `<answer>` 标签）但答案错误
- `0.0`: 无效格式（无 `<answer>` 标签）

Reward 通过 `reward.custom_reward_function.path` 注入:
```
reward.custom_reward_function.path=src/reward/countdown.py
reward.custom_reward_function.name=compute_score
```

### 数据格式

使用 `apply_chat_template` + `enable_thinking=True` 生成 prompt：
```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
```

每个样本结构：
```python
{
    "data_source": "countdown",
    "prompt": [{"content": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n", "role": "user"}],
    "reward_model": {"style": "rule", "ground_truth": {"target": 98, "numbers": [44, 19, 35]}}
}
```

### 训练配置（GRPO + HF Rollout）

当前最佳配置（`experiments/2026-03-31_grpo_countdown/run.sh`）：
- **Rollout**: `hf`（vLLM 与 transformers>=5 不兼容）
- **训练策略**: `fsdp2`
- **KL 正则**: `use_kl_loss=True`, `kl_loss_type=low_var_kl`, `kl_loss_coef=0.001`
- **动态 batch**: `use_dynamic_bsz=True`, `ppo_max_token_len_per_gpu=4096`
- **采样数**: `rollout.n=8`
- **学习率**: 1e-6
- **Reward**: 通过 `reward.custom_reward_function.path` 注入

### entry.py TensorBoard 补丁

`entry.py` monkey-patch 了 verl 的 `compute_data_metrics`，额外记录:
- `reward/format_success_rate`: 模型输出包含 `<answer>` 标签的比例
- `reward/result_success_rate`: 答案完全正确的比例

### AutoDL 服务器路径

| 用途 | 路径 | 说明 |
|------|------|------|
| 持久化 | `/root/autodl-fs` | 模型、checkpoint，重启不丢 |
| 临时文件 | `/root/autodl-tmp` | 日志、数据，重启消失 |
| 模型 | `/root/autodl-fs/models/Qwen3-1.7B-Base` | |
| 数据 | `/root/autodl-tmp/data/countdown` | |

## 服务器操作流程

```bash
# 1. 环境安装（首次）
bash scripts/setup_env.sh

# 2. 数据预处理
python scripts/prepare_data.py --local_dir /root/autodl-tmp/data/countdown

# 3. 启动实验
bash experiments/2026-03-31_grpo_countdown/run.sh
```

## 已知问题

1. **vLLM 与 transformers>=5 不兼容**: 无法使用 vLLM rollout，只能用 HF
   - 追踪: https://github.com/vllm-project/vllm/issues/30466
2. **Outputs 日志为空**: Hydra 生成的 `main_ppo.log` 全部为空文件
3. **verl v0.4.0 方案受阻**: flash-attn/flashinfer 预编译 wheel 与 torch 2.10 不兼容
