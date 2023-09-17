lr=1e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=../pretrain/chinese-alpaca-2-7b-hf
chinese_tokenizer_path=../pretrain/chinese-alpaca-2-7b-hf
dataset_dir=ZJQA/train/
per_device_train_batch_size=4
per_device_eval_batch_size=4
gradient_accumulation_steps=8
output_dir=output-ZJQA/alpaca-7b
validation_file=ZJQA/dev.json

deepspeed_config_file=ds_zero2_no_offload.json

torchrun --master_port 29501 --nnodes 1 --nproc_per_node 4 run_clm_sft_with_peft-chinese-7b.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --num_train_epochs 10 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --evaluation_strategy epoch \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length 1024 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --validation_file ${validation_file} \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --load_best_model_at_end True
