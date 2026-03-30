# CLAUDE.md — 项目指南

## 项目简介

GRPO 学习项目，目标是用 Qwen3.5-2B-Base 复现 TinyZero 的 grokking 现象。

## 项目结构

```
TinyZero_QWen3.5/
├── CLAUDE.md                # 本文件 — 项目指南
├── entry.py                 # 训练入口（monkey-patch reward → 启动 verl）
├── pyproject.toml           # 包配置
├── requirements.txt         # 依赖版本（已验证）
├── docs/
│   └── setup_guide.md       # 远程服务器环境安装指南
├── src/                     # 核心代码
│   ├── reward/countdown.py  # 奖励函数 (1.0/0.1/0.0)
│   ├── templates/countdown.py # 中文 Prompt 模板 (ChatML + few-shot)
│   └── utils/tokenizer.py   # Qwen3.5 特殊 token 处理
├── scripts/                 # 通用脚本
│   ├── setup_env.sh         # 环境安装
│   ├── prepare_data.py      # 数据预处理
│   └── train_grpo.sh        # 训练模板脚本
└── experiments/             # 实验记录（每次实验一个目录）
    └── YYYY-MM-DD_name/
        ├── run.sh           # 当次实验的启动脚本
        └── README.md        # 实验笔记（结果、观察、结论）
```

## 关键技术信息

### Qwen3.5-2B-Base 特殊 Token

| Token | ID | 用途 |
|-------|-----|------|
| `` | 248044 | eos_token |
| `<\|im_start\|>` | 248045 | ChatML 消息开始 |
| `<\|im_end\|>` | 248046 | ChatML 消息结束 |

stop_token_ids = [248044, 248046]

### Qwen3.5 混合架构

Qwen3.5 使用 Gated DeltaNet (线性注意力, 18层) + Gated Attention (标准注意力, 6层) 混合架构。
需要安装 `flash-linear-attention` 和 `causal-conv1d` 来加速线性注意力层，否则会回退到慢速 PyTorch 实现。

### Reward 规则

- `1.0`: 等式正确（数字正确 + 结果正确）
- `0.1`: 格式正确（有 `<answer>` 标签）但答案错误
- `0.0`: 无效格式（无 `<answer>` 标签）

### 依赖版本（已验证）

安装顺序: PyTorch → vLLM(可选) → flash-attn → verl → flash-linear-attention → 其他

| 包 | 版本 | 注意事项 |
|---|------|----------|
| torch | 2.9.0+cu128 | 先装，用 cu128 index |
| transformers | >=5.2.0 | **Qwen3.5 必须**, 不兼容 vLLM |
| verl | 0.7.1 | |
| flash-attn | 2.8.3 | 必须用预编译 wheel |
| flash-linear-attention | >=0.4.0 | Qwen3.5 DeltaNet 加速, 需从源码编译 |
| causal-conv1d | >=1.6.0 | Qwen3.5 DeltaNet 依赖, 需从源码编译 |
| numpy | >=1.26,<2.0 | 必须 < 2.0 |
| liger-kernel | >=0.7.0 | |

**重要**: vLLM (截至 0.18.0) 要求 `transformers<5`，与 Qwen3.5 的 `transformers>=5.2.0` 不兼容。
训练使用 HF rollout (`rollout.name=hf`)，这是 verl 的默认且推荐方式。
详见: https://github.com/vllm-project/vllm/issues/30466

### AutoDL 服务器路径

| 用途 | 路径 | 说明 |
|------|------|------|
| 持久化 | `/root/autodl-fs` | 模型、checkpoint，重启不丢 |
| 临时文件 | `/root/autodl-tmp` | 日志、数据，重启消失 |
| 模型 | `/root/autodl-fs/models/Qwen3.5-2B-Base` | |
| 数据 | `/root/autodl-tmp/data/countdown` | |
| 日志 | `/root/autodl-tmp/tf-logs` | TensorBoard |
| 实验 | `/root/autodl-tmp/experiments` | 每次实验独立目录 |

网络加速: `source /etc/network_turbo`
HF 镜像: `export HF_ENDPOINT=https://hf-mirror.com`
PyPI 镜像: `-i https://pypi.tuna.tsinghua.edu.cn/simple/`

## 实验规范

每次实验在 `experiments/` 下创建目录，命名格式: `YYYY-MM-DD_简短描述`

实验目录必须包含:
1. **`run.sh`**: 当次实验使用的完整启动脚本（含具体参数），可直接重新运行
2. **`README.md`**: 实验笔记，包括:
   - 目的 / 假设
   - 关键参数变更
   - 结果摘要
   - 观察和结论

创建新实验: 复制 `scripts/train_grpo.sh` 到实验目录，修改参数和名称后运行。

## 服务器操作流程

```bash
# 1. 环境安装（首次）
bash scripts/setup_env.sh

# 2. 数据预处理
python scripts/prepare_data.py --local_dir /root/autodl-tmp/data/countdown

# 3. 启动实验
bash experiments/YYYY-MM-DD_name/run.sh

# 4. TensorBoard
tensorboard --logdir /root/autodl-tmp/tf-logs --bind_all
```
