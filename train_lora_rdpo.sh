lr=1e-6
beta=0.1

export TRANSFORMERS_CACHE=/data/wendi/hf_cache
export HF_HOME=/data/wendi/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/wendi/hf_cache

deepspeed --include=localhost:0,1,2,3 --master_port 60000 train_rdpo.py \
    --sft_weight 0.05 \
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
    --output_dir "./output/llava-vicuna-13b-rdpo-lora-$lr-beta-$beta-new-qkvo" \
