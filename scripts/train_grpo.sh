#!/bin/bash
# =============================================================================
# TinyZero QWen3.5 GRPO 训练脚本
# =============================================================================
# 适配 AutoDL 服务器环境
#
# 用法:
#   bash scripts/train_grpo.sh                    # 默认使用 HF 推理
#   bash scripts/train_grpo.sh --use-vllm         # 使用 vLLM 推理
# =============================================================================
set -euo pipefail

# =============================================================================
# 环境变量
# =============================================================================
# AutoDL 学术加速 (取消注释以启用)
# source /etc/network_turbo

# HuggingFace 镜像 (AutoDL 网络环境差时使用)
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

# CUDA / NCCL 配置
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1

# =============================================================================
# 路径配置 (根据 AutoDL 环境调整)
# =============================================================================
# 持久化目录: 模型、checkpoint 等重启不丢失
PERSIST_DIR=${PERSIST_DIR:-/root/autodl-fs}

# 临时目录: 日志、数据等 (重启会消失)
TMP_DIR=${TMP_DIR:-/root/autodl-tmp}

N_GPUS=${N_GPUS:-1}
BASE_MODEL=${BASE_MODEL:-$PERSIST_DIR/models/Qwen3.5-2B-Base}
DATA_DIR=${DATA_DIR:-$TMP_DIR/data/countdown}
LOG_DIR=${LOG_DIR:-$TMP_DIR/tf-logs}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_countdown_qwen35_2b}

mkdir -p $LOG_DIR

# =============================================================================
# 推理引擎选择
# =============================================================================
ROLLOUT_NAME="hf"
TP_SIZE=1

if [[ "${1:-}" == "--use-vllm" ]]; then
    echo "使用 vLLM 推理引擎"
    ROLLOUT_NAME="vllm"
    TP_SIZE=${VLLM_TP_SIZE:-1}
else
    echo "使用 HF 推理引擎 (默认)"
fi

# =============================================================================
# 启动训练
# =============================================================================
echo "============================================"
echo "实验: $EXPERIMENT_NAME"
echo "模型: $BASE_MODEL"
echo "数据: $DATA_DIR"
echo "GPU数: $N_GPUS"
echo "推理引擎: $ROLLOUT_NAME"
echo "日志: $LOG_DIR"
echo "============================================"

python3 entry.py \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.response_length=512 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.micro_batch_size=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger='[console, tensorboard]' \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.project_name=TinyZero \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    2>&1 | tee $LOG_DIR/${EXPERIMENT_NAME}_console.log
