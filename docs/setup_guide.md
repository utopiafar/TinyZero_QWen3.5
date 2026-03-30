# 远程服务器环境安装指南

## 环境信息

| 项目 | 值 |
|------|------|
| 服务器 | AutoDL |
| GPU | NVIDIA RTX 5090 (32GB VRAM) |
| CUDA Toolkit | 12.8 (V12.8.93) |
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.12.3 (miniconda base) |
| RAM | 754 GB |

## 已验证的依赖版本

| 包 | 版本 | 安装方式 |
|---|------|----------|
| torch | 2.9.0+cu128 | PyTorch cu128 index |
| transformers | 5.4.0 | 清华 PyPI 镜像 |
| verl | 0.7.1 | 清华 PyPI 镜像 |
| flash-attn | 2.8.3+cu128torch2.9 | 预编译 wheel (本地传到服务器) |
| flash-linear-attention | 0.4.2 | 源码编译 (`--no-build-isolation`) |
| causal-conv1d | 1.6.1 | 源码编译 (`--no-build-isolation`) |
| numpy | 1.26.4 | 清华 PyPI 镜像 |
| liger-kernel | 0.7.0 | 清华 PyPI 镜像 |
| peft | 0.18.1 | 清华 PyPI 镜像 |
| accelerate | 1.13.0 | 清华 PyPI 镜像 |

## 安装步骤（按顺序执行）

### 前置条件

- AutoDL 服务器已启动，SSH 可连接
- 本地已下载 Qwen3.5-2B-Base 模型（git lfs pull 完成）

### 1. 上传模型到服务器

```bash
# 先确认本地模型文件是真实的（不是 LFS 指针）
ls -lh /path/to/Qwen3.5-2B-Base/model.safetensors-00001-of-00001.safetensors
# 应该是 ~4.3G，如果只有 135B 说明需要 git lfs pull

# 上传到服务器（排除 .git 目录）
rsync -avz --exclude='.git' -e "ssh -p <端口号>" \
    /path/to/Qwen3.5-2B-Base/ \
    root@<服务器地址>:/root/autodl-fs/models/Qwen3.5-2B-Base/
```

### 2. 安装 PyTorch

```bash
ssh <服务器> "source /etc/network_turbo; source /root/miniconda3/etc/profile.d/conda.sh; conda activate base; \
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128"
```

验证:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 应输出: 2.9.0+cu128 True
```

### 3. 下载并上传 flash-attn 预编译 wheel

在本地下载（服务器网络可能不稳定）:
```bash
# 本地 macOS
curl -L -o /tmp/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl \
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl"

# 传到服务器
scp -P <端口号> /tmp/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl \
    root@<服务器地址>:/root/autodl-tmp/

# 在服务器上安装
pip install /root/autodl-tmp/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl
```

来源: https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/tag/v0.9.0

### 4. 安装 verl 和核心依赖

```bash
# 用清华镜像（阿里云镜像包不全）
MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple/"

pip install "numpy>=1.26.0,<2.0.0" $MIRROR
pip install verl==0.7.1 $MIRROR
pip install tensorboard liger-kernel peft accelerate $MIRROR
```

### 5. 安装 transformers 5.x（Qwen3.5 必需）

```bash
pip install "transformers>=5.2.0" $MIRROR
```

**重要**: Qwen3.5 模型类型是 `qwen3_5`，只有 transformers >= 5.2.0 才识别。4.x 版本会报 `KeyError: 'qwen3_5'`。

### 6. 安装 flash-linear-attention + causal-conv1d

Qwen3.5 使用混合架构 (Gated DeltaNet + Gated Attention)，DeltaNet 层需要这两个库加速。

```bash
# 必须加 --no-build-isolation，否则 pip 会在隔离环境中重新下载 torch（极度缓慢）
pip install flash-linear-attention causal-conv1d --no-build-isolation $MIRROR
```

编译约需 5-10 分钟（nvcc 编译 CUDA 内核）。

### 7. Clone 项目并安装

```bash
cd /root/autodl-tmp
git clone https://github.com/utopiafar/TinyZero_QWen3.5.git
cd TinyZero_QWen3.5
pip install -e .
```

### 8. 准备训练数据

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/prepare_data.py --local_dir /root/autodl-tmp/data/countdown
# 生成: train.parquet (327680 条) + test.parquet (1024 条)
```

## 验证

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = '/root/autodl-fs/models/Qwen3.5-2B-Base'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16, device_map='auto'
)
inputs = tokenizer('Hello', return_tensors='pt').to(model.device)
out = model.generate(**inputs, max_new_tokens=5)
print(tokenizer.decode(out[0]))
# 应输出类似: Hello, I'm a ...
"
```

## 已知问题

### vLLM 与 Qwen3.5 不兼容

**问题**: vLLM (截至 0.18.0) 要求 `transformers<5`，而 Qwen3.5 需要 `transformers>=5.2.0`。

**影响**: 无法使用 vLLM 作为推理引擎。

**解决方案**: 使用 HF rollout (`rollout.name=hf`)，这是 verl 的默认方式，也是 TinyZero 参考项目的默认配置。对于 2B 模型，HF 推理速度完全够用。

**追踪**: https://github.com/vllm-project/vllm/issues/30466

### flash-linear-attention 编译慢

**问题**: `pip install flash-linear-attention` 会在隔离环境中重新下载 PyTorch (~2GB)。

**解决**: 使用 `--no-build-isolation` 跳过隔离环境。

### 阿里云 PyPI 镜像缺包

**问题**: `pip install vllm==0.12.0` 从阿里云镜像找不到包。

**解决**: 使用清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple/`。
