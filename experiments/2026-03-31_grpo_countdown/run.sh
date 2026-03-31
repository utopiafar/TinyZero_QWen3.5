#!/bin/bash
# =============================================================================
# TinyZero Countdown GRPO 训练脚本
# =============================================================================
# verl v0.7.1 + HF rollout + FSDP2, 1x RTX 5090 32GB, Qwen3-1.7B-Base
# 注意: 单 GPU 上 vllm async rollout 显存不够，改用 HF rollout
# =============================================================================
set -x

export HF_ENDPOINT=https://hf-mirror.com
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
# vllm 0.18 的 memory pool 与 expandable_segments 不兼容
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# TensorBoard 日志目录
export TENSORBOARD_DIR=/root/tf-logs

CHECKPOINT_PATH=/root/autodl-fs/models/Qwen3-1.7B-Base
DATA_DIR=/root/autodl-tmp/data/countdown
LOG_DIR=/root/tf-logs
SAMPLE_DIR=/root/tf-logs/samples

gpu_nums=1
project_name='TinyZero-Countdown'
experiment_name='Qwen3-1.7B-GRPO'

mkdir -p $LOG_DIR $SAMPLE_DIR

python3 entry.py \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=${CHECKPOINT_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    reward.custom_reward_function.path=src/reward/countdown.py \
    reward.custom_reward_function.name=compute_score \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=False \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${gpu_nums} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.resume_mode=auto \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    trainer.rollout_data_dir=${SAMPLE_DIR} \
    $@ 2>&1 | tee ${LOG_DIR}/${experiment_name}_console.log
