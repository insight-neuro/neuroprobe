#!/bin/bash
# Full Neuroprobe competition run — GNN across all tasks and splits
#
# Covers:
#   - WithinSession: train/test from same subject+trial
#   - CrossSession:  train on one trial, test on another (same subject)
#   - CrossSubject:  train on subject 2 trial 4, test on all other Lite pairs
#
# All 15 official Neuroprobe tasks, all 12 Lite subject/trial pairs.
#
# Usage:
#   cd examples/
#   ROOT_DIR_BRAINTREEBANK=/storage/czw/braintreebank_data/all_subject_data \
#     conda run -n gait_fixed bash run_gnn_competition.sh
#
# Expected runtime: ~2-6 hours on a GPU (Devon/Somerset)
# Resume safely: uses --if_exists skip to skip already-completed evals

set -e

# ---------------------------------------------------------------------------
# Configuration — update GNN_VARIANT/GNN_GRAPH after running the benchmark
# ---------------------------------------------------------------------------
# Best variant from local benchmark (sub3 trial0, 3 tasks):
#   gnn_v1_stgcn coords    → AUC ~0.821
#   gnn_v2_gat   coords    → AUC ~0.830
#   gnn_v1_stgcn functional→ AUC ~0.843  ← default
#   gnn_v2_gat   functional→ run benchmark first (run_gnn_benchmark.sh)
GNN_VARIANT="${GNN_VARIANT:-gnn_v1_stgcn}"
GNN_GRAPH="${GNN_GRAPH:-functional}"
SAVE_BASE="${SAVE_BASE:-eval_results/gnn_competition}"

# Data root — override via env var or edit here
export ROOT_DIR_BRAINTREEBANK="${ROOT_DIR_BRAINTREEBANK:-/storage/eg99/braintreebank_data}"
export NEUROPROBE_FEATURES_FILE="${NEUROPROBE_FEATURES_FILE:-features.csv}"

# All 15 official Neuroprobe tasks (scene_onset excluded — added in fork, not part of benchmark)
ALL_TASKS="onset,speech,volume,delta_volume,pitch,word_index,word_gap,gpt2_surprisal,word_head_pos,word_part_speech,word_length,global_flow,local_flow,frame_brightness,face_num"

# Neuroprobe Lite subject/trial pairs (neuroprobe/config.py NEUROPROBE_LITE_SUBJECT_TRIALS)
# CrossSubject train pair is (2, 4) — defined as DS_DM_TRAIN_SUBJECT_ID/TRIAL_ID in config.py
SUBJECT_TRIALS=(
    "1 1"
    "1 2"
    "2 0"
    "2 4"
    "3 0"
    "3 1"
    "4 0"
    "4 1"
    "7 0"
    "7 1"
    "10 0"
    "10 1"
)

# ---------------------------------------------------------------------------
run_eval() {
    local SUBJECT_ID=$1
    local TRIAL_ID=$2
    local SPLIT=$3
    local SAVE_DIR="${SAVE_BASE}/${SPLIT}/sub${SUBJECT_ID}_trial${TRIAL_ID}"

    echo "  sub${SUBJECT_ID} trial${TRIAL_ID} → ${SAVE_DIR}"
    python eval_population.py \
        --classifier_type gnn \
        --gnn_variant "$GNN_VARIANT" \
        --gnn_graph "$GNN_GRAPH" \
        --subject_id "$SUBJECT_ID" \
        --trial_id "$TRIAL_ID" \
        --eval_name "$ALL_TASKS" \
        --split_type "$SPLIT" \
        --only_1second \
        --verbose \
        --if_exists skip \
        --save_dir "$SAVE_DIR"
}

# ---------------------------------------------------------------------------
echo "========================================================"
echo "GNN Competition Run"
echo "  Variant : $GNN_VARIANT"
echo "  Graph   : $GNN_GRAPH"
echo "  Data    : $ROOT_DIR_BRAINTREEBANK"
echo "  Output  : $SAVE_BASE"
echo "========================================================"
echo ""

# --- WithinSession ---
echo "--- WithinSession (12 subject/trial pairs) ---"
for ST in "${SUBJECT_TRIALS[@]}"; do
    SUBJECT_ID=$(echo "$ST" | cut -d' ' -f1)
    TRIAL_ID=$(echo "$ST" | cut -d' ' -f2)
    run_eval "$SUBJECT_ID" "$TRIAL_ID" "WithinSession"
done
echo ""

# --- CrossSession ---
echo "--- CrossSession (12 subject/trial pairs) ---"
for ST in "${SUBJECT_TRIALS[@]}"; do
    SUBJECT_ID=$(echo "$ST" | cut -d' ' -f1)
    TRIAL_ID=$(echo "$ST" | cut -d' ' -f2)
    run_eval "$SUBJECT_ID" "$TRIAL_ID" "CrossSession"
done
echo ""

# --- CrossSubject ---
# Fixed training pair: subject 2 trial 4 (DS_DM_TRAIN_SUBJECT_ID/TRIAL_ID)
# Skip (2, 4) as test — it is the training data
echo "--- CrossSubject (train: sub2 trial4 — 11 test pairs) ---"
for ST in "${SUBJECT_TRIALS[@]}"; do
    SUBJECT_ID=$(echo "$ST" | cut -d' ' -f1)
    TRIAL_ID=$(echo "$ST" | cut -d' ' -f2)
    if [ "$SUBJECT_ID" = "2" ] && [ "$TRIAL_ID" = "4" ]; then
        echo "  Skipping sub2 trial4 (used as CrossSubject training data)"
        continue
    fi
    run_eval "$SUBJECT_ID" "$TRIAL_ID" "CrossSubject"
done
echo ""

echo "========================================================"
echo "Competition run complete."
echo "Results saved to: $SAVE_BASE"
echo ""
echo "To use a different variant, run:"
echo "  GNN_VARIANT=gnn_v2_gat GNN_GRAPH=functional bash run_gnn_competition.sh"
echo "========================================================"
