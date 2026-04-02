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
├── entry.py                 # 训练入口（直接调用 verl main_ppo）
├── docs/
│   └── setup_guide.md       # 服务器环境安装指南
├── src/                     # 核心代码
│   ├── reward/countdown.py  # Countdown 奖励函数 (1.0/0.1/0.0)
│   ├── templates/countdown.py  # ChatML 模板（含 few-shot 中文示例）
│   └── utils/tokenizer.py   # Qwen3 tokenizer 工具
├── scripts/
│   ├── setup_env.sh         # 环境安装脚本
│   ├── prepare_data.py      # 数据预处理（手拼 ChatML 中文模板）
│   └── train_grpo.sh        # 通用训练脚本（支持 HF/vLLM 切换）
└── experiments/             # 实验记录（按日期命名）
    ├── README.md            # 全部实验总结
    └── <date>_<name>/       # 各实验配置（run.sh + notes.md）
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

Reward 通过 `reward.custom_reward_function` 注入:
```
reward.custom_reward_function.path=src/reward/countdown.py
reward.custom_reward_function.name=compute_score
```

reward 函数返回 dict: `{"score": float, "format_success": int, "result_success": int}`

### TensorBoard 指标

verl 的 `metric_utils.py` 已集成 reward 细分指标（无需 monkey-patch）：
- `reward/format_success_rate`: 模型输出包含 `<answer>` 标签的比例
- `reward/result_success_rate`: 答案完全正确的比例
- `critic/score/mean`: 平均 reward 分数
- `response_length/clip_ratio`: 输出被截断的比例

### 数据格式

手拼 Qwen3 ChatML 中文模板生成 prompt（不依赖 `apply_chat_template`）：
```
<|im_start|>user\n...中文任务描述...<|im_end|>\n<|im_start|>assistant\n
```

每个样本结构：
```python
{
    "data_source": "countdown",
    "prompt": [{"content": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n", "role": "user"}],
    "reward_model": {"style": "rule", "ground_truth": {"target": 98, "numbers": [44, 19, 35]}}
}
```

### 训练配置（最新: v2 实验）

最新配置见 `experiments/2026-04-01_vllm_grpo_v2/run.sh`：
- **Rollout**: `vllm` + `mode=async`
- **训练策略**: `fsdp2`
- **KL 正则**: `use_kl_loss=True`, `kl_loss_type=low_var_kl`, `kl_loss_coef=0.001`
- **动态 batch**: `use_dynamic_bsz=True`, `ppo_max_token_len_per_gpu=4096`
- **采样数**: `rollout.n=8`
- **学习率**: 1e-6
- **vLLM 显存**: `gpu_memory_utilization=0.5`
- **max_response_length**: 1024（从 v2 开始，原 512 截断率高）
- **max_num_seqs**: 256（配合更长 response，避免 vLLM sampler warmup OOM）
- **Reward**: 通过 `reward.custom_reward_function.path` 注入

### AutoDL 服务器路径

| 用途 | 路径 | 说明 |
|------|------|------|
| 持久化 | `/root/autodl-fs` | 模型、代码，重启不丢 |
| 临时文件 | `/root/autodl-tmp` | 数据、日志，重启消失 |
| 模型 | `/root/autodl-fs/models/Qwen3-1.7B-Base` | |
| 数据 | `/root/autodl-tmp/data/countdown` | 重启后需重新生成 |
| TensorBoard | `/root/tf-logs` | 各实验子目录 |
| Checkpoint | `./checkpoints/` | 项目内，每个 ~20GB |

## 实验组织

### 命名规则

`experiments/<日期>_<描述>/`，每个实验目录包含：
- `run.sh` — 完整的训练启动脚本（可独立运行）
- `notes.md` 或 `README.md` — 实验笔记和结果

### 历史实验

| 实验 | 日期 | 说明 |
|------|------|------|
| `2026-03-30_*` | 03-30 | 早期探索（baseline, HF rollout 等） |
| `2026-03-31_vllm_grpo` | 03-31 | v1: vLLM rollout + 英文 prompt + 512 response |
| `2026-04-01_vllm_grpo_v2` | 04-01 | v2: 中文 prompt + 1024 response + max_num_seqs=256 |

### 新建实验

1. 复制最近的 `run.sh` 到新目录：`cp -r experiments/2026-04-01_vllm_grpo_v2 experiments/<新日期>_<新名>`
2. 修改 `experiment_name`（决定 TensorBoard 子目录名）
3. 修改配置参数
4. 在 `run.sh` 头部注释中记录变更

## 服务器操作流程

### 启动训练

```bash
# 1. 数据预处理（首次或服务器重启后，/root/autodl-tmp 会清空）
python scripts/prepare_data.py --local_dir /root/autodl-tmp/data/countdown

# 2. 启动实验（后台运行）
nohup bash experiments/2026-04-01_vllm_grpo_v2/run.sh &
# 或前台运行（可看实时输出）
bash experiments/2026-04-01_vllm_grpo_v2/run.sh
```

### 监控训练

```bash
# TensorBoard
tensorboard --logdir /root/tf-logs --port 6006

# 查看最新 step
grep "step:" /root/tf-logs/<experiment_name>_console.log | tail -1

# 查看关键指标趋势
grep "step:" /root/tf-logs/<experiment_name>_console.log | grep -oP "step:\d+.*?reward/format_success_rate:[0-9.]+.*?reward/result_success_rate:[0-9.]+"
```

### 停止训练

```bash
# 停止 Ray 集群（会 kill 所有 worker）
ray stop --force

# 如有残留 vLLM 进程
pkill -9 -f "vllm"
```

### 清理空间

```bash
# 删除 checkpoint（每个 ~20GB）
rm -rf ./checkpoints/

# 删除 TensorBoard 日志
rm -rf /root/tf-logs/<experiment_name>/
```

### 常见问题

- **vLLM OOM (sampler warmup)**: 降低 `max_num_seqs`（默认 1024 太大）或增大 `gpu_memory_utilization`
- **GPU 残留进程**: `ray stop --force` + `pkill -9 -f "vllm"`，然后 `nvidia-smi` 确认清空
- **服务器重启后数据丢失**: `/root/autodl-tmp` 是临时目录，需重新 `python scripts/prepare_data.py`
