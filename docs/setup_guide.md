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
| torch | 2.10.0+cu128 | PyTorch cu128 index |
| transformers | 5.4.0 | 清华 PyPI 镜像 |
| verl | 0.7.1 | 清华 PyPI 镜像 |
| flash-attn | 2.8.1+cu12torch2.10 | GitHub Releases 预编译 wheel |
| numpy | 1.26.4 | 清华 PyPI 镜像 |
| liger-kernel | 0.7.0 | 清华 PyPI 镜像 |
| peft | 0.18.1 | 清华 PyPI 镜像 |
| accelerate | 1.13.0 | 清华 PyPI 镜像 |

## 安装步骤（按顺序执行）

### 前置条件

- AutoDL 服务器已启动，SSH 可连接
- 本地已下载 Qwen3-1.7B-Base 模型（git lfs pull 完成）

### 1. 上传模型到服务器

```bash
# 先确认本地模型文件是真实的（不是 LFS 指针）
ls -lh /path/to/Qwen3-1.7B-Base/model.safetensors
# 应该是合理大小，如果只有 135B 说明需要 git lfs pull

# 上传到服务器（排除 .git 目录）
rsync -avz --exclude='.git' -e "ssh -p <端口号>" \
    /path/to/Qwen3-1.7B-Base/ \
    root@<服务器地址>:/root/autodl-fs/models/Qwen3-1.7B-Base/
```

### 2. 安装 PyTorch

```bash
ssh <服务器> "source /etc/network_turbo; source /root/miniconda3/etc/profile.d/conda.sh; conda activate base; \
pip install torch==2.10.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128"
```

验证:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 应输出: 2.10.0+cu128 True
```

### 3. 安装 flash-attn 预编译 wheel

**注意**: flash-attn 必须使用与 PyTorch 版本匹配的预编译 wheel，不可从源码编译（耗时极长且容易失败）。

当前环境: PyTorch 2.10.0 + CUDA 12.8 + Python 3.12 + CXX11_ABI=TRUE

```bash
# 从官方 GitHub Releases 下载预编译 wheel
source /etc/network_turbo  # AutoDL 学术加速
wget -O /tmp/flash_attn-wheel.whl \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# 安装（--no-deps 避免重装 torch）
pip install /tmp/flash_attn-wheel.whl --no-deps
```

验证:
```bash
python -c "import flash_attn; print(f'flash_attn {flash_attn.__version__} OK')"
# 应输出: flash_attn 2.8.1 OK
```

**Wheel 选择要点**:
- 来源: [flash-attention 官方 Releases](https://github.com/Dao-AILab/flash-attention/releases)
- 版本: v2.8.1 是目前唯一有 torch 2.10 预编译 wheel 的版本（v2.8.2/2.8.3 最高只到 torch 2.9）
- CXX11_ABI: 需与 PyTorch 一致，`python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"` 查询
- 命名规则: `flash_attn-{版本}+cu12torch{torch版本}cxx11abi{TRUE/FALSE}-cp{py版本}-linux_x86_64.whl`

**常见问题**: 如果导入报 `undefined symbol` 错误，说明 wheel 与 PyTorch 版本不匹配，需重新选择对应的 wheel。

### 4. 安装 verl 和核心依赖

```bash
# 用清华镜像（阿里云镜像包不全）
MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple/"

pip install "numpy>=1.26.0,<2.0.0" $MIRROR
pip install verl==0.7.1 $MIRROR
pip install tensorboard liger-kernel peft accelerate $MIRROR
```

### 5. 安装 transformers 5.x

```bash
pip install "transformers>=5.2.0" $MIRROR
```

**重要**: Qwen3 模型类型是 `qwen3`，需要 transformers >= 5.2.0 才识别。4.x 版本会报 `KeyError: 'qwen3'`。

### 6. Clone 项目并安装

```bash
cd /root/autodl-tmp
git clone https://github.com/utopiafar/TinyZero_QWen3.5.git
cd TinyZero_QWen3.5
pip install -e .
```

### 7. 准备训练数据

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

model_path = '/root/autodl-fs/models/Qwen3-1.7B-Base'
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

### vLLM 与 transformers>=5 不兼容

**问题**: vLLM (截至 0.18.0) 要求 `transformers<5`，而 Qwen3 需要 `transformers>=5.2.0`。

**影响**: 无法使用 vLLM 作为推理引擎。

**解决方案**: 使用 HF rollout (`rollout.name=hf`)，这是 verl 的默认方式，也是 TinyZero 参考项目的默认配置。对于 1.7B 模型，HF 推理速度完全够用。

**追踪**: https://github.com/vllm-project/vllm/issues/30466

### 阿里云 PyPI 镜像缺包

**问题**: `pip install vllm==0.12.0` 从阿里云镜像找不到包。

**解决**: 使用清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple/`。
