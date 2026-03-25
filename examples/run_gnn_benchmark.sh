#!/bin/bash
# Quick benchmark: compare GNN variants on 1 subject/trial, 3 tasks.
# Run this first to identify the best architecture before the full competition run.
#
# Usage: bash run_gnn_benchmark.sh
# Expected runtime: ~10-30 min per variant depending on hardware

set -e

SUBJECT_ID=1
TRIAL_ID=1
EVAL_NAMES="onset,gpt2_surprisal,speech"
SPLIT="WithinSession"
SAVE_DIR="eval_results/gnn_benchmark"

export NEUROPROBE_FEATURES_FILE="${NEUROPROBE_FEATURES_FILE:-features.csv}"
export ROOT_DIR_BRAINTREEBANK="${ROOT_DIR_BRAINTREEBANK:-/storage/eg99/braintreebank_data}"

for VARIANT in gnn_v1_stgcn gnn_v2_gat; do
    for GRAPH in coords functional; do
        echo "========================================"
        echo "Running variant: $VARIANT  graph: $GRAPH"
        echo "========================================"
        python eval_population.py \
            --classifier_type gnn \
            --gnn_variant "$VARIANT" \
            --gnn_graph "$GRAPH" \
            --subject_id "$SUBJECT_ID" \
            --trial_id "$TRIAL_ID" \
            --eval_name "$EVAL_NAMES" \
            --split_type "$SPLIT" \
            --only_1second \
            --verbose \
            --save_dir "$SAVE_DIR/${VARIANT}_${GRAPH}"
        echo ""
    done
done

echo "========================================"
echo "Benchmark complete. Results in: $SAVE_DIR"
echo "Compare test_roc_auc across variants."
echo "========================================"
