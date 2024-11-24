#!/bin/bash

session=$1
config=$2
csv_path=$3
data_path_audio=$4
data_path_roberta=$5
checkpoint_path=$6

python src/run_iemocap_infer.py \
    --session "$session" \
    --config_path "$config" \
    --csv_path "$csv_path" \
    --data_path_audio "$data_path_audio" \
    --data_path_roberta "$data_path_roberta" \
    --checkpoint_path "$checkpoint_path"
