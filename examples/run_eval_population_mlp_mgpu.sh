#!/bin/bash
#SBATCH --job-name=e_p_lite
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH --array=1-6
#SBATCH --output data/logs/%A_%a.out # STDOUT
#SBATCH --error data/logs/%A_%a.err # STDERR
#SBATCH --open-mode=append  # Append to output files instead of overwriting
#SBATCH --requeue
#SBATCH -p ou_bcs_low

nvidia-smi

export PYTHONUNBUFFERED=1
export ROOT_DIR_BRAINTREEBANK=/orcd/data/fiete/001/zaho/braintreebank/ # Engaging
source .venv/bin/activate

# Use the BTBENCH_LITE_SUBJECT_TRIALS from btbench_config.py
declare -a subjects=(1 1 2 2 3 3 4 4 7 7 10 10)
declare -a trials=(1 2 0 4 0 1 0 1 0 1 0 1)

declare -a eval_names=(
    "frame_brightness"
    "global_flow"
    "local_flow"
    "face_num"
    "volume"
    "pitch"
    "delta_volume"
    "speech"
    "onset"
    "gpt2_surprisal"
    "word_length"
    "word_gap"
    "word_index"
    "word_head_pos"
    "word_part_speech"
)
# to make it sequential, just aggregate the eval_names separating with a comma
eval_names=(
    $(IFS=,; echo "${eval_names[*]}")
)

declare -a preprocess=(
    # 'none' # no preprocessing, just raw voltage
    #'stft_absangle', # magnitude and phase after FFT
    #'stft_realimag' # real and imaginary parts after FFT
    # 'stft_abs' # just magnitude after FFT ("spectrogram")
    'laplacian-stft_abs' # just magnitude after FFT ("spectrogram")

    #'remove_line_noise' # remove line noise from the raw voltage
    #'downsample_200' # downsample to 200 Hz
    #'downsample_200-remove_line_noise' # downsample to 200 Hz and remove line noise
)

declare -a splits_type=(
    # "WithinSession"
    "CrossSession"
    # "CrossSubject"
)

declare -a classifier_type=(
    # "linear"
    #"cnn"
    #"transformer"
    "mlp"
)

# Calculate two array task IDs to run in parallel
TASK_ID_1=$(( ($SLURM_ARRAY_TASK_ID - 1) * 2 + 1 ))
TASK_ID_2=$(( ($SLURM_ARRAY_TASK_ID - 1) * 2 + 2 ))

# Function to run a single task
run_task() {
    local TASK_ID=$1
    
    # Calculate indices for this task
    EVAL_IDX=$(( ($TASK_ID-1) % ${#eval_names[@]} ))
    PAIR_IDX=$(( ($TASK_ID-1) / ${#eval_names[@]} % ${#subjects[@]} ))
    PREPROCESS_IDX=$(( ($TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} % ${#preprocess[@]} ))
    SPLITS_TYPE_IDX=$(( ($TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} % ${#preprocess[@]} % ${#splits_type[@]} ))
    CLASSIFIER_TYPE_IDX=$(( ($TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} / ${#preprocess[@]} / ${#splits_type[@]} % ${#classifier_type[@]} ))
    
    # Get subject, trial and eval name for this task
    EVAL_NAME=${eval_names[$EVAL_IDX]}
    SUBJECT=${subjects[$PAIR_IDX]}
    TRIAL=${trials[$PAIR_IDX]}
    PREPROCESS=${preprocess[$PREPROCESS_IDX]}
    SPLITS_TYPE=${splits_type[$SPLITS_TYPE_IDX]}
    CLASSIFIER_TYPE=${classifier_type[$CLASSIFIER_TYPE_IDX]}
    save_dir="data/eval_results_lite_${SPLITS_TYPE}"
    
    echo "Running eval for TASK_ID $TASK_ID: eval $EVAL_NAME, subject $SUBJECT, trial $TRIAL, preprocess $PREPROCESS, classifier $CLASSIFIER_TYPE"
    echo "Save dir: $save_dir"
    echo "Split type: $SPLITS_TYPE"
    
    # Check if we're trying to evaluate subject 2 with DS_DM split (which is invalid)
    if [[ "$SPLITS_TYPE" == "DS_DM" && "$SUBJECT" == "2" ]]; then
        echo "Cannot evaluate the cross subject split on subject 2; exiting"
        return 0
    fi
    
    # Add the -u flag to Python to force unbuffered output
    # Set thread limits for this process (use 2 threads per task)
    # SLURM's cgroup will handle CPU isolation
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
    PYTORCH_NUM_THREADS=2 python -u examples/eval_population.py \
        --eval_name $EVAL_NAME \
        --subject_id $SUBJECT \
        --trial_id $TRIAL \
        --preprocess.type $PREPROCESS \
        --verbose \
        --save_dir $save_dir \
        --split_type $SPLITS_TYPE \
        --classifier_type $CLASSIFIER_TYPE \
        --only_1second
}

# Run both tasks in parallel on the same GPU
# Each task limited to 2 threads; SLURM cgroup handles CPU allocation
echo "Starting parallel execution of TASK_ID $TASK_ID_1 and TASK_ID $TASK_ID_2 (2 threads each)"
run_task $TASK_ID_1 &
PID1=$!
run_task $TASK_ID_2 &
PID2=$!

# Wait for both tasks to complete
wait $PID1
EXIT_CODE_1=$?
wait $PID2
EXIT_CODE_2=$?

echo "TASK_ID $TASK_ID_1 finished with exit code $EXIT_CODE_1"
echo "TASK_ID $TASK_ID_2 finished with exit code $EXIT_CODE_2"

# Exit with error if either task failed
if [ $EXIT_CODE_1 -ne 0 ] || [ $EXIT_CODE_2 -ne 0 ]; then
    exit 1
fi