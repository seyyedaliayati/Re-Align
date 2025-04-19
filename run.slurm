#!/bin/bash

#SBATCH --job-name=RunReAlign
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:rtx:2
#SBATCH --mem=8G
#SBATCH --output=logs/%j.log

ml purge
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
export TRANSFORMERS_CACHE=/scratch/user/ali.a/hf_cache
export HF_HOME=/scratch/user/ali.a/hf_home
export HF_DATASETS_CACHE=/scratch/user/ali.a/hf_datasets
export WANDB_DIR=/scratch/user/ali.a/wandb
export WANDB_CACHE_DIR=/scratch/user/ali.a/wandb
export WANDB_CONFIG_DIR=/scratch/user/ali.a/wandb
export WANDB_SETTINGS_DIR=/scratch/user/ali.a/wandb
export WANDB_API_KEY=f5115de6923afe0cac4579bff8845e57245d9f8e


ls -lh
wandb login --relogin
# deepspeed --master_port 60000 train_rdpo.py \
#     --model_name_or_path liuhaotian/llava-v1.6-vicuna-7b \
#     --data_path "./preference_data/pref_data.json" \
#     --deepspeed "./deepspeed/zero2.json" \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --save_strategy "epoch" \
#     --save_total_limit 1 \
#     --learning_rate 1e-6 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --bf16 True \
#     --lora_enable True \
#     --beta 0.1 \
#     --output_dir "./output/llava-vicuna-7b-rdpo-lora-1e-6-beta-0.1" \


echo "Done!!!"
