#!/bin/bash
# =============================================================================
# 实验: 2026-03-30_vllm_grpo
# 使用 vllm Rollout + FSDP2 的 GRPO 训练 (Qwen3-1.7B-Base Countdown)
# 参考: veRL 官方 GRPO 文档 + CoLA-RL 成功案例
# =============================================================================
set -euo pipefail

# 环境变量
source /etc/network_turbo 2>/dev/null || true
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1
export VLLM_USE_V1=1

# 路径
PERSIST_DIR=/root/autodl-fs
TMP_DIR=/root/autodl-tmp

N_GPUS=1
BASE_MODEL=$PERSIST_DIR/models/Qwen3-1.7B-Base
DATA_DIR=$TMP_DIR/data/countdown
LOG_DIR=$TMP_DIR/tf-logs
EXPERIMENT_NAME=countdown_vllm_grpo

mkdir -p $LOG_DIR

# 动态 batch size (性能调优)
use_dynamic_bsz=True
actor_ppo_max_token_len=$((1024 * 4))
infer_ppo_max_token_len=$((1024 * 4))

echo "============================================"
echo "实验: $EXPERIMENT_NAME"
echo "模型: $BASE_MODEL"
echo "数据: $DATA_DIR"
echo "GPU数: $N_GPUS"
echo "推理引擎: vllm (V1)"
echo "策略: FSDP2"
echo "日志: $LOG_DIR"
echo "============================================"

cd /autodl-fs/data/TinyZero_QWen3.5

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward.custom_reward_function.path=src/reward/countdown.py \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
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
