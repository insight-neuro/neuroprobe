#!/bin/bash
# Multi-class GNN benchmark — GNN + linear with binary_tasks=false
# WithinSession only, all 15 tasks, all 12 Lite pairs.
#
# Usage (from examples/):
#   export ROOT_DIR_BRAINTREEBANK=/storage/eg99/braintreebank_data
#   export NEUROPROBE_FEATURES_FILE=features.csv
#   nohup bash run_multiclass_gnn.sh > eval_results/multiclass_gnn/run.log 2>&1 & echo $!

set -e

export ROOT_DIR_BRAINTREEBANK="${ROOT_DIR_BRAINTREEBANK:-/storage/eg99/braintreebank_data}"
export NEUROPROBE_FEATURES_FILE="${NEUROPROBE_FEATURES_FILE:-features.csv}"

ALL_TASKS="onset,speech,volume,delta_volume,pitch,word_index,word_gap,gpt2_surprisal,word_head_pos,word_part_speech,word_length,global_flow,local_flow,frame_brightness,face_num"
SAVE_BASE="eval_results/multiclass_gnn"
mkdir -p "$SAVE_BASE"

SUBJECT_TRIALS=(
    "1 1" "1 2"
    "2 0" "2 4"
    "3 0" "3 1"
    "4 0" "4 1"
    "7 0" "7 1"
    "10 0" "10 1"
)

echo "========================================================"
echo "Multi-class GNN + Linear Benchmark"
echo "  binary_tasks=false, WithinSession, all 15 tasks"
echo "  Output: $SAVE_BASE"
echo "========================================================"

for ST in "${SUBJECT_TRIALS[@]}"; do
    S=$(echo "$ST" | cut -d' ' -f1)
    T=$(echo "$ST" | cut -d' ' -f2)
    echo "  [gnn] sub${S} trial${T}"
    python eval_population.py \
        --classifier_type gnn \
        --gnn_variant gnn_v1_stgcn \
        --gnn_graph functional \
        --subject_id "$S" \
        --trial_id "$T" \
        --eval_name "$ALL_TASKS" \
        --split_type WithinSession \
        --only_1second \
        --binary_tasks false \
        --verbose \
        --if_exists skip \
        --save_dir "${SAVE_BASE}/gnn/sub${S}_trial${T}"

    echo "  [linear] sub${S} trial${T}"
    python eval_population.py \
        --classifier_type linear \
        --preprocess.type laplacian-stft_abs \
        --subject_id "$S" \
        --trial_id "$T" \
        --eval_name "$ALL_TASKS" \
        --split_type WithinSession \
        --only_1second \
        --binary_tasks false \
        --verbose \
        --if_exists skip \
        --save_dir "${SAVE_BASE}/linear/sub${S}_trial${T}"
done

echo "========================================================"
echo "Done. Results in: $SAVE_BASE"
echo "========================================================"
