#!/bin/bash

# HPRC Only
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

# GPUs
export CUDA_VISIBLE_DEVICES=0

# SicenceQA Base
echo "ScienceQA Base..."
python -m llava.eval.model_vqa_science \
    --model-path liuhaotian/llava-v1.6-vicuna-7b \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/base.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/base.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/base_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/base_result.json

# ScienceQA rDPO
echo "ScienceQA rDPO..."
python -m llava.eval.model_vqa_science \
    --model-path /scratch/user/ali.a/Re-Align/output/llava-vicuna-7b-rdpo-lora-1e-6-beta-0.1 \
    --model-base liuhaotian/llava-v1.6-vicuna-7b \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/rdpo.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/rdpo.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/rdpo_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/rdpo_result.json

# ScienceQA SimPO
echo "ScienceQA SimPO..."
python -m llava.eval.model_vqa_science \
    --model-path /scratch/user/ali.a/Re-Align/output/llava-vicuna-13b-rsimpo-lora-1e-6-beta-0.1-new-qkvo \
    --model-base liuhaotian/llava-v1.6-vicuna-7b \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/simpo.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/simpo.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/simpo_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/simpo_result.json
# -----------------------------------------------------------------------------------------

# TextVQA Base
echo "TextVQA Base..."
python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.6-vicuna-7b \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/base.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/base.jsonl

# TextVQA rDPO
echo "TextVQA rDPO..."
python -m llava.eval.model_vqa_loader \
    --model-path /scratch/user/ali.a/Re-Align/output/llava-vicuna-7b-rdpo-lora-1e-6-beta-0.1 \
    --model-base liuhaotian/llava-v1.6-vicuna-7b \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/rdpo.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/rdpo.jsonl

# TextVQA SimPO
echo "TextVQA SimPO..."
python -m llava.eval.model_vqa_loader \
    --model-path /scratch/user/ali.a/Re-Align/output/llava-vicuna-13b-rsimpo-lora-1e-6-beta-0.1-new-qkvo \
    --model-base liuhaotian/llava-v1.6-vicuna-7b \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/simpo.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/simpo.jsonl
# -----------------------------------------------------------------------------------------