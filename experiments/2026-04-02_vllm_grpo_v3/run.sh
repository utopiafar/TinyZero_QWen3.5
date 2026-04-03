#!/bin/bash
# =============================================================================
# TinyZero Countdown GRPO — vLLM Rollout v3
# =============================================================================
# verl v0.7.1 + vLLM 0.18 + FSDP2, 2x H800 80GB PCIe, Qwen3-4B-Base
# =============================================================================
# 变更记录 (对比 v2):
#   - 模型: Qwen3-1.7B → Qwen3-4B-Base
#   - GPU: 1x RTX 5090 32GB → 2x H800 80GB PCIe (无 NVLink, SYS 互连)
#   - 并行策略: FSDP2 全切片 + NCCL 超时容错
#   - rollout.n: 8 → 16
#   - train_batch_size: 32 → 64 → 96
#   - ppo_mini_batch_size: 16 → 32 → 48
#   - ppo_max_token_len_per_gpu: 4096 → 8192 → 12288
#   - save_freq: 300 → 10 → 25, max_actor_ckpt_to_keep=3
#   - NCCL_TIMEOUT: 5h 容错
#   - 新增 critic/advantages/std 到 TensorBoard
#   - log_val_generations=4（采样文本到 TensorBoard text tab）
# =============================================================================
set -xe

source /root/miniconda3/bin/activate base

export HF_ENDPOINT=https://hf-mirror.com
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=18000000
export TOKENIZERS_PARALLELISM=true
export VLLM_LOGGING_LEVEL=WARN

CHECKPOINT_PATH=/autodl-fs/data/models/Qwen3-4B-Base
DATA_DIR=/root/autodl-tmp/data/countdown
LOG_DIR=/root/tf-logs

gpu_nums=2
project_name='TinyZero-Countdown'
experiment_name='vllm-GRPO-v3'

# TensorBoard 日志输出到实验专属子目录
export TENSORBOARD_DIR=${LOG_DIR}/${experiment_name}

mkdir -p ${TENSORBOARD_DIR}

python3 entry.py \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=96 \
    data.max_prompt_length=256 \
    data.max_response_length=1024 \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=48 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.checkpoint.save_contents="['model', 'optimizer']" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.load_format=dummy \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
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
    trainer.log_val_generations=4 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${gpu_nums} \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.resume_mode=auto \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    $@ 2>&1 | tee ${LOG_DIR}/${experiment_name}_console.log
