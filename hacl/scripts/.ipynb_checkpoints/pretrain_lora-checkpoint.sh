#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################
# NCCL_DEBUG=INFO 
# deepspeed llava/train/train_mem.py \
#     --deepspeed /tf/project/LLaVA/scripts/zero2.json  \
#     --lora_enable True \
#     --model_name_or_path /tf/ckpt/${MODEL_VERSION} \
#     --version ${PROMPT_VERSION} \
#     --data_path /tf/data/LLaVA-Instruct-150K/llava_instruct_150k.json \
#     --image_folder /tf/data/LLaVA-Instruct-150K/train2017 \
#     --vision_tower /tf/ckpt/clip-vit-large-patch14 \
#     --tune_mm_mlp_adapter True \
#     --tune_vision_tower True \
#     --calculate_contrastive_loss True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir ./checkpoints/${MODEL_VERSION}-finetune \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True \
#     --dataloader_num_workers 4 \
#     --report_to wandb
# export PYTHONPATH=$PYTHONPATH:/tf/project/LLaVA_itc_v2/llava
# deepspeed llava/train/train_mem.py \
#     --deepspeed /tf/project/LLaVA_itc_v2/scripts/zero2.json  \
#     --lora_enable True \
#     --model_name_or_path /tf/ckpt/${MODEL_VERSION} \
#     --version ${PROMPT_VERSION} \
#     --pretrain_mm_mlp_adapter /tf/project/LLaVA_itc_v2/checkpoints/vicuna-7b-v1.3-pretrain/checkpoint-39/mm_projector.bin \
#     --pretrain_vision_tower /tf/project/LLaVA_itc_v2/checkpoints/vicuna-7b-v1.3-pretrain/checkpoint-39/vision_tower.bin \
#     --data_path /tf/data/LLaVA-CC3M/chat.json \
#     --image_folder /tf/data/LLaVA-CC3M/images \
#     --vision_tower /tf/ckpt/clip-vit-large-patch14 \
#     --tune_mm_mlp_adapter True \
#     --tune_vision_tower True \
#     --calculate_contrastive_loss True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir ./checkpoints/${MODEL_VERSION}-pretrain2 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True \
#     --dataloader_num_workers 4 \
#     --report_to wandb
export PYTHONPATH=$PYTHONPATH:/shd/jcy/project/LLaVA_itc_v3/llava

# CUDA_VISIBLE_DEVICES=4,5,6,7
deepspeed  --include localhost:4,5,6,7 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json  \
    --lora_enable False \
    --model_name_or_path  /shd/jcy/ckpt/vicuna-7b-v1.5\
    --version ${PROMPT_VERSION} \
    --data_path /shd/jcy/data/llava_data/pretrain/chat.json \
    --image_folder /shd/jcy/data/llava_data/pretrain/LLaVA-CC3M-Pretrain-595K \
    --vision_tower /shd/jcy/ckpt/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --tune_vision_tower True \
    --calculate_contrastive_loss True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --use_queue False \
    --freeze_model True \
    --output_dir ./output/${MODEL_VERSION}-pretrain \
    --num_train_epochs 3 \
    --per_device_train_batch_size 256 \
    --only_contrastive_loss True \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to wandb
