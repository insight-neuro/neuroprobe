#!/bin/bash
#SBATCH --job-name=e_p_lite
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2  
#SBATCH --mem=64G
#####SBATCH --gres=gpu:1
#SBATCH -t 1:20:00
#####SBATCH --constraint=24GB
#SBATCH --exclude=dgx001,dgx002
#SBATCH --array=1-2052
#SBATCH --output data/logs/%A_%a.out # STDOUT
#SBATCH --error data/logs/%A_%a.err # STDERR
#SBATCH -p use-everything

nvidia-smi

export PYTHONUNBUFFERED=1
source .venv/bin/activate

# Use the BTBENCH_LITE_SUBJECT_TRIALS from btbench_config.py
declare -a subjects=(1 1 2 2 3 3 4 4 7 7 10 10)
declare -a trials=(1 2 0 4 0 1 0 1 0 1 0 1)

declare -a eval_names=(
    "frame_brightness"
    "global_flow"
    "local_flow"
    "global_flow_angle"
    "local_flow_angle" 
    "face_num"
    "volume"
    "pitch"
    "delta_volume"
    "delta_pitch"
    "speech"
    "onset"
    "gpt2_surprisal"
    "word_length"
    "word_gap"
    "word_index"
    "word_head_pos"
    "word_part_speech"
    "speaker"
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
    "SS_SM"
    "SS_DM"
    "DS_DM"
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
save_dir="data/eval_results_lite_${SPLITS_TYPE}"

echo "Running eval for eval $EVAL_NAME, subject $SUBJECT, trial $TRIAL, preprocess $PREPROCESS, classifier $CLASSIFIER_TYPE"
echo "Save dir: $save_dir"
echo "Split type: $SPLITS_TYPE"

# Add the -u flag to Python to force unbuffered output
python -u eval_population.py \
    --eval_name $EVAL_NAME \
    --subject_id $SUBJECT \
    --trial_id $TRIAL \
    --preprocess.type $PREPROCESS \
    --verbose \
    --save_dir $save_dir \
    --split_type $SPLITS_TYPE \
    --classifier_type $CLASSIFIER_TYPE \
    --only_1second