#!/bin/bash
# =============================================================================
# 实验: 2026-03-30_baseline
# 基线 GRPO 训练 - Qwen3-1.7B-Base Countdown 任务
# =============================================================================
set -euo pipefail

# 环境变量
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1

# 路径
PERSIST_DIR=/root/autodl-fs
TMP_DIR=/root/autodl-tmp

N_GPUS=1
BASE_MODEL=$PERSIST_DIR/models/Qwen3-1.7B-Base
DATA_DIR=$TMP_DIR/data/countdown
LOG_DIR=$TMP_DIR/tf-logs
EXPERIMENT_NAME=grpo_countdown_qwen3_1.7b_baseline

mkdir -p $LOG_DIR

echo "============================================"
echo "实验: $EXPERIMENT_NAME"
echo "模型: $BASE_MODEL"
echo "数据: $DATA_DIR"
echo "GPU数: $N_GPUS"
echo "推理引擎: hf (默认)"
echo "日志: $LOG_DIR"
echo "============================================"

cd /autodl-fs/data/TinyZero_QWen3.5

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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.response_length=512 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
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
