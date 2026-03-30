#!/bin/bash
# =============================================================================
# AutoDL 服务器环境安装脚本
# =============================================================================
# 用法: bash scripts/setup_env.sh
#
# 测试环境: AutoDL RTX 5090 (32GB), CUDA 12.8, Ubuntu 22.04
# Python: 3.12.x (miniconda base 环境)
#
# 重要说明:
#   - Qwen3.5 需要 transformers >= 5.2.0，与 vLLM (要求 transformers<5) 不兼容
#   - 训练使用 HF rollout (verl 默认)，不需要 vLLM
#   - PyPI 请使用清华镜像，阿里云镜像包不全
# =============================================================================
set -euo pipefail

MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple/"

echo "============================================"
echo "TinyZero QWen3.5 环境安装"
echo "============================================"

# AutoDL 学术加速
echo "[1/8] 启用学术加速..."
source /etc/network_turbo 2>/dev/null || echo "  (非 AutoDL 环境，跳过)"

# 确认环境
echo "[2/8] 环境检查..."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
echo "  Python: $(python --version)"
echo "  CUDA: $(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# 安装 PyTorch (CUDA 12.8)
echo "[3/8] 安装 PyTorch 2.9.0+cu128..."
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 安装 Flash Attention (预编译 wheel)
echo "[4/8] 安装 Flash Attention 2.8.3 (预编译 wheel)..."
PY_VER=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
FLASH_WHEEL="flash_attn-2.8.3+cu128torch2.9-${PY_VER}-${PY_VER}-linux_x86_64.whl"
FLASH_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/${FLASH_WHEEL}"

echo "  下载: ${FLASH_WHEEL}"
if ! pip install ${FLASH_URL}; then
    echo "  预编译 wheel 下载失败，尝试本地安装..."
    if [ -f "/root/autodl-tmp/${FLASH_WHEEL}" ]; then
        pip install "/root/autodl-tmp/${FLASH_WHEEL}"
    else
        echo "  请手动下载 wheel 并安装:"
        echo "  https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/tag/v0.9.0"
        exit 1
    fi
fi

# 安装 verl 和核心依赖
echo "[5/8] 安装 verl 0.7.1 及依赖..."
pip install "numpy>=1.26.0,<2.0.0" $MIRROR
pip install verl==0.7.1 $MIRROR
pip install tensorboard liger-kernel peft accelerate $MIRROR

# 安装 transformers (Qwen3.5 需要 >= 5.2.0)
echo "[6/8] 安装 transformers >= 5.2.0 (Qwen3.5 必需)..."
pip install "transformers>=5.2.0" $MIRROR

# 安装 Qwen3.5 DeltaNet 加速库 (需要从源码编译 CUDA 内核)
echo "[7/8] 安装 flash-linear-attention + causal-conv1d (编译中，约 5-10 分钟)..."
pip install flash-linear-attention causal-conv1d --no-build-isolation $MIRROR

# 安装项目
echo "[8/8] 安装项目..."
pip install -e .

# 验证
echo ""
echo "============================================"
echo "验证安装..."
python -c "
import torch; print(f'  torch: {torch.__version__}, CUDA: {torch.version.cuda}')
import transformers; print(f'  transformers: {transformers.__version__}')
import verl; print(f'  verl: {verl.__version__}')
import numpy; print(f'  numpy: {numpy.__version__}')
import flash_attn; print(f'  flash_attn: {flash_attn.__version__}')
try:
    import fla; print(f'  flash_linear_attention: OK')
except: print('  flash_linear_attention: 未安装 (DeltaNet 将使用慢速路径)')
try:
    import causal_conv1d; print(f'  causal_conv1d: OK')
except: pass
import liger_kernel; print(f'  liger_kernel: OK')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print('验证通过!')
"

echo ""
echo "============================================"
echo "环境安装完成!"
echo ""
echo "下一步:"
echo "  1. python scripts/prepare_data.py --local_dir /root/autodl-tmp/data/countdown"
echo "  2. bash scripts/train_grpo.sh"
echo ""
echo "注意: vLLM 与 transformers>=5 不兼容，训练使用 HF rollout"
echo "============================================"
