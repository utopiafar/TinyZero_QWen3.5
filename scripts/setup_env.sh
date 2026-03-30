#!/bin/bash
# =============================================================================
# AutoDL 服务器环境安装脚本
# =============================================================================
# 用法: bash scripts/setup_env.sh
#
# 前置条件:
#   - Python 3.12.x (miniconda)
#   - CUDA 12.8
#   - 至少 24GB GPU 显存 (Qwen3.5-2B-Base + GRPO)
# =============================================================================
set -euo pipefail

echo "============================================"
echo "TinyZero QWen3.5 环境安装"
echo "============================================"

# AutoDL 学术加速
echo "[1/7] 启用学术加速..."
source /etc/network_turbo 2>/dev/null || echo "  (非 AutoDL 环境，跳过)"

# 创建 conda 环境 (如果不存在)
ENV_NAME=${ENV_NAME:-tinyzero}
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "[2/7] 创建 conda 环境: ${ENV_NAME}..."
    conda create -n $ENV_NAME python=3.12 -y
else
    echo "[2/7] conda 环境 ${ENV_NAME} 已存在，跳过"
fi

echo "请运行: conda activate ${ENV_NAME}"
echo ""

# 安装 PyTorch (CUDA 12.8)
echo "[3/7] 安装 PyTorch 2.9.0+cu128..."
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 安装 vLLM
echo "[4/7] 安装 vLLM 0.12.0..."
pip install vllm==0.12.0

# 安装 Flash Attention (预编译 wheel)
echo "[5/7] 安装 Flash Attention 2.8.3 (预编译 wheel)..."
PYTHON_VERSION=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
FLASH_ATTN_WHEEL="flash_attn-2.8.3+cu128torch2.9-${PYTHON_VERSION}-${PYTHON_VERSION}-linux_x86_64.whl"
FLASH_ATTN_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/${FLASH_ATTN_WHEEL}"

echo "  尝试下载预编译 wheel: ${FLASH_ATTN_WHEEL}"
if ! pip install ${FLASH_ATTN_URL}; then
    echo "  预编译 wheel 下载失败，请手动安装:"
    echo "  1. 访问 https://flashattn.dev/install/prebuilt-wheels 找到对应的 wheel"
    echo "  2. 或访问 https://github.com/mjun0812/flash-attention-prebuild-wheels/releases"
    echo "  3. 下载后执行: pip install <wheel文件路径>"
fi

# 安装 verl 和其他依赖
echo "[6/7] 安装 verl 0.7.1 及依赖..."
pip install "numpy>=1.26.0,<2.0.0"
pip install verl==0.7.1
pip install tensorboard liger-kernel

# 安装项目依赖
echo "[7/7] 安装项目依赖..."
pip install -e .

echo ""
echo "============================================"
echo "环境安装完成!"
echo ""
echo "下一步:"
echo "  1. conda activate ${ENV_NAME}"
echo "  2. python scripts/prepare_data.py --local_dir /root/autodl-tmp/data/countdown"
echo "  3. bash scripts/train_grpo.sh"
echo ""
echo "查看 TensorBoard:"
echo "  tensorboard --logdir /root/autodl-tmp/tf-logs --bind_all"
echo "============================================"
