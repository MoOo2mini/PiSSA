#!/bin/bash

BASE_MODEL="meta-llama/Llama-2-7b-hf"
RES_MODEL="output/PiSSA-Llama-rank128"
DATA_PATH="pissa-dataset"

for GAS in 16 8
do
    TOTAL_BS=$((2 * GAS * 4))
    OUTPUT_PATH="output/PiSSA_script_bs${TOTAL_BS}"

    MAX_STEP=$((100000 / TOTAL_BS + 1))

    for i in 0 2 4 6 8
    do
        CKPT_NAME="postupdate_adapter_step$i"
        CKPT_PATH="${OUTPUT_PATH}/${CKPT_NAME}"
        OUTPUT_PATH_MER="${OUTPUT_PATH}_ckpt$i"

        if [ -d "$CKPT_PATH" ]; then

            CUDA_VISIBLE_DEVICES=4,5 python utils/merge_adapter.py \
                --base_model $RES_MODEL \
                --adapter $CKPT_PATH \
                --output_path $OUTPUT_PATH_MER

            CUDA_VISIBLE_DEVICES=4,5 python utils/gen_vllm.py \
                --model $OUTPUT_PATH_MER \
                --sub_task metamath \
                --output_file $OUTPUT_PATH_MER/metamath_response.jsonl

            CUDA_VISIBLE_DEVICES=4,5 python utils/test_acc.py \
                --input_file $OUTPUT_PATH_MER/metamath_response.jsonl \
                --ckpt_step $i \
                --wandb_project "PiSSA_batch_test" \
        
        fi

    done

    for i in $(seq 10 20 $MAX_STEP)
    do
        CKPT_NAME="postupdate_adapter_step$i"
        CKPT_PATH="${OUTPUT_PATH}/${CKPT_NAME}"
        OUTPUT_PATH_MER="${OUTPUT_PATH}_ckpt$i"

        if [ -d "$CKPT_PATH" ]; then

            CUDA_VISIBLE_DEVICES=4,5 python utils/merge_adapter.py \
                --base_model $RES_MODEL \
                --adapter $CKPT_PATH \
                --output_path $OUTPUT_PATH_MER

            CUDA_VISIBLE_DEVICES=4,5 python utils/gen_vllm.py \
                --model $OUTPUT_PATH_MER \
                --sub_task metamath \
                --output_file $OUTPUT_PATH_MER/metamath_response.jsonl

            CUDA_VISIBLE_DEVICES=4,5 python utils/test_acc.py \
                --input_file $OUTPUT_PATH_MER/metamath_response.jsonl \
                --ckpt_step $i \
                --wandb_project "PiSSA_batch_test" \
        
        fi

    done

done
