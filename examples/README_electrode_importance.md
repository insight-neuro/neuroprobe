# Electrode Importance Analysis

This script extracts electrode importance from trained classifiers to identify which electrodes are most informative for a given classification task.

## Usage

### Basic Example

```bash
python examples/eval_electrode_importance.py \
    --subject_id 3 \
    --trial_id 0 \
    --eval_name onset \
    --split_type WithinSession \
    --plot_3d
```

### With Custom Time Window

```bash
python examples/eval_electrode_importance.py \
    --subject_id 3 \
    --trial_id 0 \
    --eval_name scene \
    --bin_start 0.0 \
    --bin_end 1.0 \
    --preprocess.type laplacian-stft_abs \
    --plot_3d \
    --verbose
```

### Cross-Session Evaluation

```bash
python examples/eval_electrode_importance.py \
    --subject_id 3 \
    --trial_id 0 \
    --eval_name onset \
    --split_type CrossSession \
    --plot_3d
```

## Arguments

### Required
- `--subject_id`: Subject ID (integer)
- `--trial_id`: Trial ID (integer)

### Optional
- `--eval_name`: Evaluation task name (default: 'onset')
- `--split_type`: Type of train/test splits - WithinSession, CrossSession, or CrossSubject (default: 'WithinSession')
- `--bin_start`: Time bin start in seconds relative to word onset (default: 0.0)
- `--bin_end`: Time bin end in seconds relative to word onset (default: 1.0)
- `--preprocess.type`: Preprocessing type (default: 'laplacian-stft_abs')
- `--plot_3d`: Generate 3D scatter plot of electrodes colored by importance
- `--verbose`: Print detailed progress information
- `--save_dir`: Directory to save results (default: 'eval_results')

## Output

The script generates:

1. **CSV file**: `electrode_importance_<preprocess>_<subject>_<trial>_<eval>_bin<start>to<end>.csv`
   - Contains electrode labels and their importance scores
   - Sorted by importance (highest first)

2. **3D Plot** (if `--plot_3d` is used): 
   - 3D scatter plot showing electrode locations colored by importance
   - Saved as PNG file

## How It Works

1. Loads the subject and trial data
2. Generates train/test splits based on `--split_type`
3. For each fold:
   - Trains a linear classifier on all electrodes
   - Extracts coefficients (weights) from the classifier
   - Reshapes coefficients to match electrode structure
   - Computes mean absolute weight per electrode
4. Averages importance across all folds
5. Saves results and optionally generates plots

## Notes

- Only linear classifiers are supported (coefficients are directly interpretable)
- The importance is computed as the mean absolute value of classifier coefficients
- For multi-class tasks, uses the first class's coefficients
- Results are averaged across all cross-validation folds

## Example Output

```
Top 10 most important electrodes:
  F3aOFa2: 0.0234
  F3aOFa3: 0.0218
  F2Ia1: 0.0195
  ...
```





