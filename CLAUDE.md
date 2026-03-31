# CLAUDE.md — 项目指南

## 项目简介

GRPO 学习项目，目标是用 Qwen3-1.7B-Base 复现 TinyZero 的 grokking 现象。

技术栈：verl v0.7.1 + vLLM 0.18.0 rollout + FSDP2，运行在 AutoDL 单 GPU 服务器上。

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
│   ├── prepare_data.py      # 数据预处理（手拼 ChatML 模板，零依赖）
│   └── train_grpo.sh        # 通用训练脚本（支持 HF/vLLM 切换）
└── experiments/             # 实验记录
    ├── README.md            # 全部实验总结
    └── <date>_<name>/       # 各实验配置和笔记
```

## 环境要求

**当前实际环境: AutoDL base conda (Python 3.12)**

| 包 | 版本 | 说明 |
|---|------|------|
| python | 3.12.3 | 服务器默认 |
| torch | 2.10.0+cu128 | PyPI 安装 |
| verl | 0.7.1 | PyPI 安装 |
| transformers | 4.57.6 | vLLM 要求 <5 |
| vllm | 0.18.0 | PyPI 安装 |
| flash-attn | 2.8.3 | 预编译 wheel，匹配 cu128+torch2.10 |
| numpy | 1.26.4 | |

### 环境安装流程

```bash
# 1. 基础依赖（使用 base conda 环境）
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install verl==0.7.1 vllm==0.18.0 "transformers==4.57.6"

# 2. flash-attn（必须用预编译 wheel，不能从源码编译）
pip install /autodl-fs/data/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl

# 3. 数据预处理（手拼 ChatML 模板，无需升级 transformers）
python scripts/prepare_data.py --local_dir /root/autodl-tmp/data/countdown
```

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

手拼 Qwen3 ChatML 模板生成 prompt（不依赖 `apply_chat_template`）：
```
<|im_start|>user\n...task...<|im_end|>\n<|im_start|>assistant\n
```

每个样本结构：
```python
{
    "data_source": "countdown",
    "prompt": [{"content": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n", "role": "user"}],
    "reward_model": {"style": "rule", "ground_truth": {"target": 98, "numbers": [44, 19, 35]}}
}
```

### 训练配置（GRPO + vLLM Rollout）

最佳配置（`experiments/2026-03-31_vllm_grpo/run.sh`）：
- **Rollout**: `vllm` + `mode=async`
- **训练策略**: `fsdp2`
- **KL 正则**: `use_kl_loss=True`, `kl_loss_type=low_var_kl`, `kl_loss_coef=0.001`
- **动态 batch**: `use_dynamic_bsz=True`, `ppo_max_token_len_per_gpu=4096`
- **采样数**: `rollout.n=8`
- **学习率**: 1e-6
- **vLLM 显存**: `gpu_memory_utilization=0.5`（根据 GPU 显存调整）
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
| TensorBoard | `/root/autodl-tmp/tf-logs` | |

## 服务器操作流程

```bash
# 1. 数据预处理（首次或数据丢失后）
python scripts/prepare_data.py --local_dir /root/autodl-tmp/data/countdown

# 2. 启动实验（vLLM rollout）
bash experiments/2026-03-31_vllm_grpo/run.sh

# 3. 查看 TensorBoard
tensorboard --logdir /root/autodl-tmp/tf-logs --port 6006
```
