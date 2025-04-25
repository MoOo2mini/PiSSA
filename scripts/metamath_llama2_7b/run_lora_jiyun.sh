#!/bin/bash

#SBATCH -J PiSSA_16_128
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=24G
#SBATCH -p RTX6000ADA
#SBATCH -t 3-0
#SBATCH -o /home/jiyunbae/coursework/PiSSA/lora_train.out

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


for GAS in 8 4 1 #128 64 16
# for GAS in 16 8 2 1
do
    for lr in 1e-3 3e-4 5e-5 2e-5 5e-6 1e-6
    do
        TOTAL_BS=$((4 * GAS * 4))
        # OUTPUT_PATH="output/PiSSA_script_bs${TOTAL_BS}"
        OUTPUT_PATH="output/PiSSA_script_bs${TOTAL_BS}_${lr}"
        MAX_STEP=$((100000 / TOTAL_BS + 1))


        #batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
        deepspeed --master_port=16971 --include=localhost:0,1,2,3 train_wandb.py \
            --deepspeed configs/ds_config_zero2_no_offload.json \
            --model_name_or_path $BASE_MODEL \
            --full_finetune False \
            --bf16 \
            --init_weights True \
            --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
            --lora_rank 128 \
            --lora_alpha 128 \
            --lora_dropout 0 \
            --data_path $DATA_PATH \
            --sub_task metamath:100000 \
            --dataset_split train \
            --dataset_field instruction output \
            --output_dir $OUTPUT_PATH \
            --num_train_epochs 1 \
            --model_max_length 512 \
            --per_device_train_batch_size 4 \
            --gradient_accumulation_steps $GAS \
            --save_strategy "steps" \
            --save_steps 10000000 \
            --save_total_limit 1 \
            --learning_rate $lr \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --logging_steps 1 \
            --lr_scheduler_type "cosine" \
            --report_to "tensorboard" \
            --merge True

        for i in $(seq 1 1 29)
        do
            CKPT_NAME="postupdate_adapter_step$i"
            CKPT_PATH="${OUTPUT_PATH}/${CKPT_NAME}"
            OUTPUT_PATH_MER="${OUTPUT_PATH}_ckpt$i"

            if [ -d "$CKPT_PATH" ]; then

                CUDA_VISIBLE_DEVICES=0,1,2,3 python utils/merge_adapter.py \
                    --base_model $RES_MODEL \
                    --adapter $CKPT_PATH \
                    --output_path $OUTPUT_PATH_MER

                CUDA_VISIBLE_DEVICES=0,1,2,3 python utils/gen_vllm.py \
                    --model $OUTPUT_PATH_MER \
                    --sub_task metamath \
                    --output_file $OUTPUT_PATH_MER/metamath_response.jsonl

                CUDA_VISIBLE_DEVICES=0,1,2,3 python utils/test_acc.py \
                    --input_file $OUTPUT_PATH_MER/metamath_response.jsonl \
                    --ckpt_step $i \
                    --wandb_project "PiSSA_batch_test" \
                    --bs $TOTAL_BS \
                    --lr $lr

            fi
        done
        for i in $(seq 30 30 780)
        do
            CKPT_NAME="postupdate_adapter_step$i"
            CKPT_PATH="${OUTPUT_PATH}/${CKPT_NAME}"
            OUTPUT_PATH_MER="${OUTPUT_PATH}_ckpt$i"

            if [ -d "$CKPT_PATH" ]; then

                CUDA_VISIBLE_DEVICES=0,1,2,3 python utils/merge_adapter.py \
                    --base_model $RES_MODEL \
                    --adapter $CKPT_PATH \
                    --output_path $OUTPUT_PATH_MER

                CUDA_VISIBLE_DEVICES=0,1,2,3 python utils/gen_vllm.py \
                    --model $OUTPUT_PATH_MER \
                    --sub_task metamath \
                    --output_file $OUTPUT_PATH_MER/metamath_response.jsonl

                CUDA_VISIBLE_DEVICES=0,1,2,3 python utils/test_acc.py \
                    --input_file $OUTPUT_PATH_MER/metamath_response.jsonl \
                    --ckpt_step $i \
                    --wandb_project "PiSSA_batch_test" \
                    --bs $TOTAL_BS \
                    --lr $lr

            fi
        done
        if [[ "$GAS" == 4 ]]; then
            for i in $(seq 780 120 $MAX_STEP)
            do
                CKPT_NAME="postupdate_adapter_step$i"
                CKPT_PATH="${OUTPUT_PATH}/${CKPT_NAME}"
                OUTPUT_PATH_MER="${OUTPUT_PATH}_ckpt$i"

                if [ -d "$CKPT_PATH" ]; then

                    CUDA_VISIBLE_DEVICES=0,1,2,3 python utils/merge_adapter.py \
                        --base_model $RES_MODEL \
                        --adapter $CKPT_PATH \
                        --output_path $OUTPUT_PATH_MER

                    CUDA_VISIBLE_DEVICES=0,1,2,3 python utils/gen_vllm.py \
                        --model $OUTPUT_PATH_MER \
                        --sub_task metamath \
                        --output_file $OUTPUT_PATH_MER/metamath_response.jsonl

                    CUDA_VISIBLE_DEVICES=0,1,2,3 python utils/test_acc.py \
                        --input_file $OUTPUT_PATH_MER/metamath_response.jsonl \
                        --ckpt_step $i \
                        --wandb_project "PiSSA_batch_test" \
                        --bs $TOTAL_BS \
                        --lr $lr

                fi
            done
        elif [[ "$GAS" == 1 ]]; then
            for i in $(seq 780 480 $MAX_STEP)
            do
                CKPT_NAME="postupdate_adapter_step$i"
                CKPT_PATH="${OUTPUT_PATH}/${CKPT_NAME}"
                OUTPUT_PATH_MER="${OUTPUT_PATH}_ckpt$i"

                if [ -d "$CKPT_PATH" ]; then

                    CUDA_VISIBLE_DEVICES=0,1,2,3 python utils/merge_adapter.py \
                        --base_model $RES_MODEL \
                        --adapter $CKPT_PATH \
                        --output_path $OUTPUT_PATH_MER

                    CUDA_VISIBLE_DEVICES=0,1,2,3 python utils/gen_vllm.py \
                        --model $OUTPUT_PATH_MER \
                        --sub_task metamath \
                        --output_file $OUTPUT_PATH_MER/metamath_response.jsonl

                    CUDA_VISIBLE_DEVICES=0,1,2,3 python utils/test_acc.py \
                        --input_file $OUTPUT_PATH_MER/metamath_response.jsonl \
                        --ckpt_step $i \
                        --wandb_project "PiSSA_batch_test" \
                        --bs $TOTAL_BS \
                        --lr $lr

                fi
            done
        fi
    done
done

exit 0
