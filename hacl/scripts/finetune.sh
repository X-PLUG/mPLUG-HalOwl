#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
# PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################
export model_name=llava_7b
export save_name=${model_name}_sft
export work_space=/
export PYTHONPATH=$PYTHONPATH:${work_space}llava
export root_path=/
deepspeed llava/train/train.py \
    --deepspeed  scripts/zero2.json\
    --model_name_or_path  ${root_path}ckpt/llama-7b-hf  \
    --data_path data/LLaVA-Instruct-150K/llava_instruct_150k.json \
    --image_folder  data/LLaVA-Instruct-150K/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ${root_path}ckpt/LLaVA-Pretrained-Projectors/LLaVA-7b-pretrain-projector-v0-CC3M-595K-original_caption.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir output/${save_name} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
