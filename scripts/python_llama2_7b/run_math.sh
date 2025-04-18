BASE_MODEL="meta-llama/Llama-2-7b-hf"
RES_MODEL="output/PiSSA-Llama-2-7b-r128"
OUTPUT_PATH="output/python-PiSSA-Llama-2-7b-r128"
DATA_PATH="pissa-dataset"
export HF_ENDPOINT=https://hf-mirror.com
if [ -e $RES_MODEL ]; then
    echo "Use pre-initialized residual model."
else
    echo "Perform PiSSA initialization by myself."
    python utils/init_pissa.py --base_model_path $BASE_MODEL --output_dir $RES_MODEL --init_weights pissa_niter_16 --lora_r 128 --lora_alpha 128 --lora_dropout 0 --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
fi

python utils/gen_vllm.py --model $OUTPUT_PATH --sub_task metamath --output_file $OUTPUT_PATH/metamath_response.jsonl
python utils/test_acc.py --input_file $OUTPUT_PATH/metamath_response.jsonl