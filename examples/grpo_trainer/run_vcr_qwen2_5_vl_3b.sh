#!/bin/bash
# VCR (Vision-SR1) GRPO training for Qwen2.5-VL-3B on 4×GPU
# Usage:
#   bash examples/grpo_trainer/run_vcr_qwen2_5_vl_3b.sh           # default REWARD_MODE=vcr
#   bash examples/grpo_trainer/run_vcr_qwen2_5_vl_3b.sh grpo      # pure GRPO ablation
#   REWARD_MODE=self_reward bash examples/grpo_trainer/run_vcr_qwen2_5_vl_3b.sh

set -e
set -x

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_PROJECT=qwen2_5_vl_3b_vcr
export WANDB_ENTITY=chenjunwei-shenzhen-university

# progression reward mode: vcr | grpo (custom scorer supports these)
REWARD_MODE=${1:-${REWARD_MODE:-vcr}}
export REWARD_MODE
echo "Running with REWARD_MODE=${REWARD_MODE}"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# paths (edit as needed)
DATA_TRAIN=/home/chenjunwei/work/Vision-SR1/outputs/vision_sr1_vcr_verl_chat.parquet
DATA_VAL=/home/chenjunwei/data/vcr_parquet/test_verl.parquet
MODEL_PATH_SFT=/home/chenjunwei/work/LLaMa-Factory/saves/sft_models/VCR-R1-SFT-3B-last-5e6-epoch3-batch16/checkpoint-1500
MODEL_PATH_BASE=/home/chenjunwei/work/Qwen2.5-VL-3B-Instruct/Qwen2.5-VL-3B-Instruct

# default uses base model; override by setting MODEL_PATH in env, e.g.
# MODEL_PATH=${MODEL_PATH_SFT} bash examples/grpo_trainer/run_vcr_qwen2_5_vl_3b.sh
MODEL_PATH=${MODEL_PATH:-${MODEL_PATH_BASE}}
# Optional: set this to inject rollout instruction into the last user message.
# If omitted, trainer keeps VERL original prompt behavior.
export VCR_ROLLOUT_INSTRUCTION="You are tasked with analyzing an image/video to generate a detailed description to help you answer the question. First analyze the image/video and produce a self-contained description—detailed enough that can lead to the correct answer. Wrap the entire description in <description> </description> tags.\n Next, engage in an internal dialogue and include self-reflection or verification in your reasoning process. Provide your detailed, step-by-step reasoning based on the image/video description information and image/video, and enclose this part within <think> </think> tags.\n Finally, provide a single word or phrase answer to the question in \\boxed{}.\nThe output format should be: <description> image/video description here </description> <think> reasoning process here </think> \\boxed{FINAL ANSWER here}."

RUN_NAME=qwen2_5_vl_3b_base_${REWARD_MODE}_kl001_st10
CKPT_DIR=./saves/${RUN_NAME}

python3 -m verl.trainer.main_ppo \
  trainer.project_name=${WANDB_PROJECT} \
  trainer.experiment_name=${RUN_NAME} \
  trainer.default_local_dir=${CKPT_DIR} \
  trainer.save_freq=20 \
  trainer.total_epochs=1 \
  trainer.val_before_train=false \
  trainer.val_only=false \
  trainer.test_freq=10 \
  trainer.n_gpus_per_node=4 \
  \
  data.train_files=${DATA_TRAIN} \
  data.val_files=${DATA_VAL} \
  data.prompt_key=problem \
  data.image_key=images \
  data.reward_fn_key=data_source \
  data.max_prompt_length=12800 \
  data.max_response_length=2048 \
  data.train_batch_size=128 \
  data.return_raw_chat=false \
  data.filter_overlong_prompts=true \
  \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.use_fused_kernels=true \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.rollout.max_model_len=14848 \
  actor_rollout_ref.rollout.max_num_batched_tokens=14848 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  \
  custom_reward_function.path=verl/utils/reward_score/vcr_reward.py \
  custom_reward_function.name=compute_score \
  algorithm.use_kl_in_reward=false \
  algorithm.adv_estimator=grpo \
  \
  reward_manager.name=vcr \
  reward_manager.source=register \
  \
  +trainer.reward_mode=${REWARD_MODE} \
  +trainer.score_batch_size=512 \
  +trainer.score_temperature=1.0 \
  +trainer.score_max_model_len=14848 \
  +trainer.debug_ans_tokens=5 \
  "$@"
