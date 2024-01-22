#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.3"
################## VICUNA ##################

export PYTHONPATH=$PYTHONPATH:/shd/project/LLaVA_itc_v3/llava

 
deepspeed  --include localhost:1,2,3,4 --master_port 29500 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json  \
    --lora_enable False \
    --model_name_or_path  /shd/ckpt/vicuna-7b-v1.3 \
    --version ${PROMPT_VERSION} \
    --data_path /shd/data/llava_data/sft/llava_instruct_150k.json \
    --image_folder /shd/data/coco/2017/train2017 \
    --vision_tower /shd/ckpt/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter output/vicuna-7b-v1.3-pretrain_coco_cc_no_eos_with_generation/mm_projector.bin  \
    --tune_mm_mlp_adapter False \
    --tune_vision_tower False \
    --calculate_contrastive_loss False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --use_queue False \
    --queue_size 30720 \
    --freeze_model False \
    --output_dir ./output/${MODEL_VERSION}-sft_itc_coco_no_eos \
    --num_train_epochs 2 \
    --per_device_train_batch_size 48 \
    --gather_all False \
    --add_eos_token_to_image False \
    --add_eos_token_to_caption False \
    --only_contrastive_loss False \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
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

