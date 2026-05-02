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

# Split pairs across 3 GPUs (1/2/3) for 3x speedup
GPU1_PAIRS=("1 1" "1 2" "2 0" "2 4")
GPU2_PAIRS=("3 0" "3 1" "4 0" "4 1")
GPU3_PAIRS=("7 0" "7 1" "10 0" "10 1")

run_pair() {
    local S=$1
    local T=$2
    echo "  [gnn] sub${S} trial${T} (GPU $CUDA_VISIBLE_DEVICES)"
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

    echo "  [linear] sub${S} trial${T} (GPU $CUDA_VISIBLE_DEVICES)"
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
}

run_gpu_worker() {
    local GPU=$1
    shift
    export CUDA_VISIBLE_DEVICES=$GPU
    for ST in "$@"; do
        S=$(echo "$ST" | cut -d' ' -f1)
        T=$(echo "$ST" | cut -d' ' -f2)
        run_pair "$S" "$T"
    done
}

echo "========================================================"
echo "Multi-class GNN + Linear Benchmark"
echo "  binary_tasks=false, WithinSession, all 15 tasks"
echo "  Parallelized across GPUs 1/2/3"
echo "  Output: $SAVE_BASE"
echo "========================================================"

run_gpu_worker 1 "${GPU1_PAIRS[@]}" > "${SAVE_BASE}/log_gpu1.txt" 2>&1 &
run_gpu_worker 2 "${GPU2_PAIRS[@]}" > "${SAVE_BASE}/log_gpu2.txt" 2>&1 &
run_gpu_worker 3 "${GPU3_PAIRS[@]}" > "${SAVE_BASE}/log_gpu3.txt" 2>&1 &

echo "3 GPU workers running in background (GPUs 1/2/3)."
echo "Monitor with:"
echo "  tail -f ${SAVE_BASE}/log_gpu1.txt"

wait
echo "========================================================"
echo "Done. Results in: $SAVE_BASE"
echo "========================================================"
