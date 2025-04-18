BASE_MODEL="meta-llama/Llama-2-7b-hf"
RES_MODEL="output/PiSSA-Llama-rank128"
DATA_PATH="pissa-dataset"
export HF_ENDPOINT=https://hf-mirror.com

#huggingface-cli download --token hf_*** --resume-download $RES_MODEL --local-dir $RES_MODEL
if [ -e $RES_MODEL ]; then
    echo "Use pre-initialized residual model."
else
    echo "Perform PiSSA initialization by my self."
    python utils/init_pissa.py --base_model_path $BASE_MODEL --output_dir $RES_MODEL --init_weights pissa_niter_16 --lora_r 128 --lora_alpha 128 --lora_dropout 0 --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
fi


for GAS in 1
do
    TOTAL_BS=$((2 * GAS * 4))
    OUTPUT_PATH="output/PiSSA_script_bs${TOTAL_BS}"

    # batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
    deepspeed --master_port=16971 --include=localhost:4,5,6,7 train_wandb.py \
        --deepspeed configs/ds_config_zero2_no_offload.json \
        --model_name_or_path $RES_MODEL \
        --full_finetune False \
        --bf16 \
        --adapter_name_or_path "pissa_init" \
        --data_path $DATA_PATH \
        --sub_task metamath:100000 \
        --dataset_split train \
        --dataset_field instruction output \
        --output_dir $OUTPUT_PATH \
        --num_train_epochs 1 \
        --model_max_length 512 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps $GAS \
        --save_strategy "steps" \
        --save_steps 10000000 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --logging_steps 1 \
        --lr_scheduler_type "cosine" \
        --report_to "tensorboard" \
        --merge True \

done

# CUDA_VISIBLE_DEVICES=4,5 python utils/test_acc.py \
#                 --input_file /home/mhlee/PiSSA/output/PiSSA_script_bs128_ckpt1/metamath_response.jsonl \
#                 --ckpt_step 1 \
#                 --wandb_project "PiSSA_batch_test"