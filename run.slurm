#!/bin/bash

#SBATCH --job-name=RunReAlign
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=128G
#SBATCH --output=logs/%j.log

ml purge
ml CUDA/12.1.1
ml GCCcore/13.3.0
ml Miniconda3/23.10.0-1
source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate re-align

cd $SCRATCH/Re-Align

echo "Starting Main!!!"
mkdir -p /scratch/user/ali.a/triton_cache
mkdir -p /scratch/user/ali.a/hf_cache
mkdir -p /scratch/user/ali.a/hf_home
mkdir -p /scratch/user/ali.a/hf_datasets
mkdir -p /scratch/user/ali.a/wandb

export HOME=/scratch/user/ali.a
export TRITON_CACHE_DIR=/scratch/user/ali.a/triton_cache

export HF_HOME=/scratch/user/ali.a/hf_home
export HF_DATASETS_CACHE=/scratch/user/ali.a/hf_datasets
export HF_HUB_OFFLINE=1
export HF_DEBUG=1

export WANDB_DIR=/scratch/user/ali.a/wandb
export WANDB_CACHE_DIR=/scratch/user/ali.a/wandb
export WANDB_CONFIG_DIR=/scratch/user/ali.a/wandb
export WANDB_SETTINGS_DIR=/scratch/user/ali.a/wandb
export WANDB_MODE=offline


ls -lh
deepspeed --include=localhost:0,1 --master_port 60000 train_rdpo.py \
    --model_name_or_path liuhaotian/llava-v1.6-vicuna-7b \
    --data_path "./preference_data/pref_data.json" \
    --deepspeed "./deepspeed/zero2.json" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --bf16 True \
    --lora_enable True \
    --beta 0.1 \
    --output_dir "./output/llava-vicuna-7b-rdpo-lora-1e-6-beta-0.1" \
    --cache_dir "$HF_HOME/hub" \
    --sft_weight 0.05 \


echo "Done!!!"
