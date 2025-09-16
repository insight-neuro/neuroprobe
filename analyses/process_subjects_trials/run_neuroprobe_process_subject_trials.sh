#!/bin/bash
#SBATCH --job-name=neuroprobe_process_subject_trials          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=2    # Request 8 CPU cores per GPU
#SBATCH --mem=4G
#SBATCH --exclude=dgx001,dgx002
#SBATCH -t 1:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=1-26  # 285 if doing mini btbench
#SBATCH --output data/logs/%A_%a.out # STDOUT
#SBATCH --error data/logs/%A_%a.err # STDERR
#SBATCH -p use-everything

export PYTHONUNBUFFERED=1
source .venv/bin/activate
# Use the BTBENCH_LITE_SUBJECT_TRIALS from btbench_config.py
declare -a subjects=(1 1 1 2 2 2 2 2 2 2 3 3 3 4 4 4 5 6 6 6 7 7 8 9 10 10)
declare -a trials=(0 1 2 0 1 2 3 4 5 6 0 1 2 0 1 2 0 0 1 4 0 1 0 0 0 1)

id=$((SLURM_ARRAY_TASK_ID-1))

python -u run_scripts/process_subject_trials.py --subject ${subjects[$id]} --trial ${trials[$id]}