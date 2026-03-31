#!/bin/bash
# =============================================================================
# TinyZero Qwen3 GRPO 环境安装脚本（参照 CoLA-RL 教程 verl v0.4.0）
# =============================================================================
# 测试环境: AutoDL RTX 5090 (32GB), CUDA 13.0, Ubuntu 22.04
# Python: 3.10 (conda env: tinyzero)
# =============================================================================
set -euo pipefail

MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple/"

echo "============================================"
echo "TinyZero Qwen3 环境安装 (verl v0.4.0)"
echo "============================================"

# AutoDL 学术加速
echo "[1/6] 启用学术加速..."
source /etc/network_turbo 2>/dev/null || echo "  (非 AutoDL 环境，跳过)"

# 创建 conda 环境
echo "[2/6] 创建 conda 环境 (Python 3.10)..."
source /root/miniconda3/etc/profile.d/conda.sh
conda config --set solver classic
if conda env list | grep -q "^tinyzero "; then
    echo "  环境 tinyzero 已存在"
else
    conda create -n tinyzero python=3.10 -y
fi
conda activate tinyzero
echo "  Python: $(python --version)"

# 安装 PyTorch + vllm + tensordict
echo "[3/6] 安装 torch 2.6.0 + vllm 0.8.5..."
pip install --no-cache-dir \
    "vllm==0.8.5.post1" \
    "torch==2.6.0" \
    "torchvision==0.21.0" \
    "torchaudio==2.6.0" \
    "tensordict==0.6.2" \
    torchdata \
    -i https://pypi.mirrors.ustc.edu.cn/simple/

# 安装 Flash Attention（需要从 GitHub 下载预编译 wheel）
echo "[4/6] 安装 Flash Attention 2.7.4.post1..."
PY_VER=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
FLASH_WHEEL="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-${PY_VER}-${PY_VER}-linux_x86_64.whl"
GH_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${FLASH_WHEEL}"

# 尝试多个 GitHub 代理
DOWNLOADED=false
for proxy in "https://gh-proxy.com/" "https://ghproxy.cc/" "https://mirror.ghproxy.com/"; do
    echo "  尝试 ${proxy}..."
    if wget -q --timeout=120 "${proxy}${GH_URL}" -O "/tmp/${FLASH_WHEEL}"; then
        if [ -s "/tmp/${FLASH_WHEEL}" ]; then
            DOWNLOADED=true
            break
        fi
    fi
done

if [ "$DOWNLOADED" = true ]; then
    pip install --no-cache-dir "/tmp/${FLASH_WHEEL}"
else
    echo "  自动下载失败，请手动下载安装:"
    echo "  ${GH_URL}"
fi

# 安装 FlashInfer
echo "[5/6] 安装 FlashInfer 0.2.2.post1..."
FI_WHEEL="flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl"
FI_URL="https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/${FI_WHEEL}"

DOWNLOADED=false
for proxy in "https://gh-proxy.com/" "https://ghproxy.cc/" "https://mirror.ghproxy.com/"; do
    echo "  尝试 ${proxy}..."
    if wget -q --timeout=120 "${proxy}${FI_URL}" -O "/tmp/${FI_WHEEL}"; then
        if [ -s "/tmp/${FI_WHEEL}" ]; then
            DOWNLOADED=true
            break
        fi
    fi
done

if [ "$DOWNLOADED" = true ]; then
    pip install --no-cache-dir "/tmp/${FI_WHEEL}"
else
    echo "  自动下载失败，请手动下载安装:"
    echo "  ${FI_URL}"
fi

# 安装 verl v0.4.0
echo "[6/6] 安装 verl v0.4.0..."
if [ ! -d "/tmp/verl_v040" ]; then
    git clone --depth 1 --branch v0.4.0 https://github.com/volcengine/verl.git /tmp/verl_v040
fi
cd /tmp/verl_v040
pip install --no-cache-dir -e . $MIRROR

# 复制 verl/ 和 recipe/ 到项目目录（如果还没有的话）
PROJECT_DIR=/autodl-fs/data/TinyZero_QWen3.5
if [ ! -d "${PROJECT_DIR}/verl" ]; then
    cp -r /tmp/verl_v040/verl ${PROJECT_DIR}/
    cp -r /tmp/verl_v040/recipe ${PROJECT_DIR}/
    echo "  已复制 verl/ 和 recipe/ 到项目目录"
fi

# 回到项目目录
cd ${PROJECT_DIR}

# 验证安装
echo ""
echo "============================================"
echo "验证安装..."
python -c "
import torch; print(f'  torch: {torch.__version__}, CUDA: {torch.version.cuda}')
import transformers; print(f'  transformers: {transformers.__version__}')
import verl; print(f'  verl: {verl.__version__}')
import vllm; print(f'  vllm: {vllm.__version__}')
import flash_attn; print(f'  flash_attn: {flash_attn.__version__}')
import flashinfer; print(f'  flashinfer: OK')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print('验证通过!')
"

echo ""
echo "============================================"
echo "环境安装完成!"
echo ""
echo "下一步:"
echo "  1. conda activate tinyzero"
echo "  2. python scripts/prepare_data.py --local_dir /root/autodl-tmp/data/countdown"
echo "  3. bash experiments/2026-03-31_grpo_countdown/run.sh"
echo "============================================"
