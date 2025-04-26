lr=1e-6
beta=0.1

# export TRANSFORMERS_CACHE=/data/wendi/hf_cache
# export HF_HOME=/data/wendi/hf_cache
# export HUGGINGFACE_HUB_CACHE=/data/wendi/hf_cache

# HPRC
ml purge
ml Anaconda3/2021.11
cd $SCRATCH/Re-Align

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

deepspeed --include=localhost:0,1,2,3 --master_port 60000 train_rdpo.py \
    --model_name_or_path liuhaotian/llava-v1.6-vicuna-7b \
    --data_path "./preference_data/pref_data.json" \
    --deepspeed "./deepspeed/zero2.json" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --bf16 True \
    --lora_enable True \
    --beta $beta \
    --loss "dpo" \
    --output_dir "./output/llava-vicuna-7b-rdpo-lora-$lr-beta-$beta" \
