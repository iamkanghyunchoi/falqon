#!/bin/bash

OUTPUT_DIR="./falqon_output/13b-$1-$4-b$2-r$3-step$5-seed$6-lr$7-topk$8"

python falqon_prequantized.py \
    --model_name_or_path huggyllama/llama-13b \
    --output_dir ${OUTPUT_DIR} \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 500 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 8 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --lora_r $3 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type $1 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset $4 \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size $2 \
    --gradient_accumulation_steps 1 \
    --max_steps $5 \
    --eval_steps 187000 \
    --learning_rate $7 \
    --adam_beta2 0.999 \
    --max_grad_norm 3.0 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed $6 \
    --torch_compile True \
    --save_safetensors False \
    --num_topk $8 \
    --preq_path ./llama-13b-falqon

lm_eval --model hf-fp8lora \
    --model_args pretrained=huggyllama/llama-13b,device_map={"":0},max_memory={"":"24GB"},trust_remote_code=True,quant_type=$1,adapter_path=${OUTPUT_DIR}/checkpoint-$5/,lora_rank=$3,lora_alpha=16 \
    --tasks mmlu,hellaswag,piqa,winogrande,arc_easy,arc_challenge,boolq,openbookqa \
    --device cuda:0 \
    --batch_size auto \
    --model_path ${OUTPUT_DIR} \
    --num_fewshot 0

lm_eval --model hf-fp8lora \
    --model_args pretrained=huggyllama/llama-13b,device_map={"":0},max_memory={"":"24GB"},trust_remote_code=True,quant_type=$1,adapter_path=${OUTPUT_DIR}/checkpoint-$5/,lora_rank=$3,lora_alpha=16 \
    --tasks mmlu,hellaswag,piqa,winogrande,arc_easy,arc_challenge,boolq,openbookqa \
    --device cuda:0 \
    --batch_size auto \
    --model_path ${OUTPUT_DIR} \
    --num_fewshot 5