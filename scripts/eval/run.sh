#!/bin/bash

# HPRC
ml purge
ml Anaconda3/2021.11
cd $SCRATCH/Re-Align
source venv/bin/activate
python -V
pwd


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

export CUDA_VISIBLE_DEVICES=0

# echo "Running SQA Evaluation"
# python scripts/eval/sqa.py > ./logs/sqa.log 2>&1
# echo "Finished SQA Evaluation"

# echo "Running VQA Evaluation"
# python scripts/eval/vqa.py > ./logs/vqa.log 2>&1
# echo "Finished VQA Evaluation"

# echo "Running MM-Vet Evaluation"
# python scripts/eval/mm_vet.py > ./logs/mm_vet.log 2>&1
# echo "Finished MM-Vet Evaluation"

echo "Running VizWiz Evaluation"
python scripts/eval/vizwiz.py > ./logs/vizwiz.log 2>&1
echo "Finished VizWiz Evaluation"