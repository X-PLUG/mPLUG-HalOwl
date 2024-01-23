# Hallucination Augmented Contrastive Learning for Multimodal Large Language Model



**Codebase for "Hallucination Augmented Contrastive Learning for Multimodal Large Language Model"** [[Paper](https://arxiv.org/abs/2312.06968)] <br>


## Release
- Release MiniGPT + HACL code. &#10008;
- Release hallucinationay caption dataset.  &#10008;
- Release LLaVA1.5 + HACL code.&#10004;



### Pretrain (feature alignment)


```Shell
MODEL_VERSION=vicuna-7b-v1.5 
PROMPT_VERSION=plain
 
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path  ./ckpt/vicuna-7b-v1.5 \
    --version $PROMPT_VERSION \
    --data_path  ./data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./data/LLaVA-Pretrain/images \
    --vision_tower ./ckpt/clip-vit-large-patch14-334 \
    --pretrain_mm_mlp_adapter ./ckpt/llava_use/llava-1.5-vicuna-v1.5-7b-pretrain-mmprojector/mm_projector.bin  \
    --tune_mm_mlp_adapter True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./output/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 96 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
 

```
</details>


### Visual Instruction Tuning

1. Prepare data

Please download the annotation of LLaVA instruction tuning data [llava_instruct_158k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json), and download the COCO train2017 images [here](https://cocodataset.org/#download).

2. Start instruction Finetuning.
```Shell
deepspeed  llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./ckpt/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./data/LLaVA-Instruct-150K/llava_v1_5_mix665k_new.json \
    --image_folder ./data/ \
    --contrastive_data_path ./data/LLaVA-Pretrain/sft_from_pretrain.json \
    --vision_tower ./ckpt/clip-vit-large-patch14-334 \
    --pretrain_mm_mlp_adapter ./project/LLaVA/output/llava-vicuna-7b-v1.5-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./output/llava-v1.5-13b-SFT \
    --num_train_epochs 1 \
    --data_chunk_num 16 \
    --do_generation True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --group_by_length False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

```
</details>

## Citation

If you find LLaVA useful for your research and applications, please cite using this BibTeX:
```bibtex
@misc{jiang2023hallucination,
      title={Hallucination Augmented Contrastive Learning for Multimodal Large Language Model}, 
      author={Chaoya Jiang and Haiyang Xu and Mengfan Dong and Jiaxing Chen and Wei Ye and Ming Yan and Qinghao Ye and Ji Zhang and Fei Huang and Shikun Zhang},
      year={2023},
      eprint={2312.06968},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```

## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon.
