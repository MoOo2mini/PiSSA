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
        echo $i

    done

    for i in $(seq 10 20 $MAX_STEP)
    do
        echo $i

    done

done
