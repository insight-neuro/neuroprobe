#!/bin/bash
#SBATCH --job-name=e_p_lite
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2  
#SBATCH --mem=48G
#SBATCH -t 2:40:00
#SBATCH --array=1-108
#SBATCH --output data/logs/%A_%a.out # STDOUT
#SBATCH --error data/logs/%A_%a.err # STDERR
#SBATCH --open-mode=append  # Append to output files instead of overwriting
#SBATCH --requeue
#SBATCH -p mit_preemptable

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
    'none' # no preprocessing, just raw voltage
    #'stft_absangle', # magnitude and phase after FFT
    #'stft_realimag' # real and imaginary parts after FFT
    'stft_abs' # just magnitude after FFT ("spectrogram")
    'laplacian-stft_abs' # just magnitude after FFT ("spectrogram")

    #'remove_line_noise' # remove line noise from the raw voltage
    #'downsample_200' # downsample to 200 Hz
    #'downsample_200-remove_line_noise' # downsample to 200 Hz and remove line noise
)

declare -a splits_type=(
    "WithinSession"
    "CrossSession"
    "CrossSubject"
)

declare -a classifier_type=(
    "linear"
    #"cnn"
    #"transformer"
)

# Calculate indices for this task
EVAL_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) % ${#eval_names[@]} ))
PAIR_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} % ${#subjects[@]} ))
PREPROCESS_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} % ${#preprocess[@]} ))
SPLITS_TYPE_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} / ${#preprocess[@]} % ${#splits_type[@]} ))
CLASSIFIER_TYPE_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} / ${#preprocess[@]} / ${#splits_type[@]} % ${#classifier_type[@]} ))

# Get subject, trial and eval name for this task
EVAL_NAME=${eval_names[$EVAL_IDX]}
SUBJECT=${subjects[$PAIR_IDX]}
TRIAL=${trials[$PAIR_IDX]}
PREPROCESS=${preprocess[$PREPROCESS_IDX]}
SPLITS_TYPE=${splits_type[$SPLITS_TYPE_IDX]}
CLASSIFIER_TYPE=${classifier_type[$CLASSIFIER_TYPE_IDX]}
save_dir="data/eval_results_lite_${SPLITS_TYPE}_multiclass"

echo "Running eval for eval $EVAL_NAME, subject $SUBJECT, trial $TRIAL, preprocess $PREPROCESS, classifier $CLASSIFIER_TYPE"
echo "Save dir: $save_dir"
echo "Split type: $SPLITS_TYPE"

# Check if we're trying to evaluate subject 2 with DS_DM split (which is invalid)
if [[ "$SPLITS_TYPE" == "DS_DM" && "$SUBJECT" == "2" ]]; then
    echo "Cannot evaluate the cross subject split on subject 2; exiting"
    exit 0
fi


# Add the -u flag to Python to force unbuffered output
python -u examples/eval_population.py \
    --eval_name $EVAL_NAME \
    --subject_id $SUBJECT \
    --trial_id $TRIAL \
    --preprocess.type $PREPROCESS \
    --verbose \
    --save_dir $save_dir \
    --split_type $SPLITS_TYPE \
    --classifier_type $CLASSIFIER_TYPE \
    --binary_tasks False \
    --only_1second