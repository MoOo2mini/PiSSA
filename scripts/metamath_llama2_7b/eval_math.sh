BASE_MODEL="meta-llama/Llama-2-7b-hf"
RES_MODEL="output/PiSSA-Llama-2-7b-r128"
OUTPUT_PATH="output/metamath-PiSSA-Llama-2-7b-r128"
DATA_PATH="pissa-dataset"
export HF_ENDPOINT=https://hf-mirror.com

python utils/merge_adapter.py --base_model $RES_MODEL --adapter $OUTPUT_PATH/checkpoint-781/ --output_path ${OUTPUT_PATH}_jj
echo "Output path: $OUTPUT_PATH"
python utils/gen_vllm.py --model ${OUTPUT_PATH}_jj --sub_task metamath --output_file ${OUTPUT_PATH}_jj/metamath_response.jsonl

python utils/test_acc.py --input_file ${OUTPUT_PATH}_jj/metamath_response.jsonl