"""
Script to extract and visualize electrode importance for classification tasks.

This script trains a classifier on all electrodes and extracts the importance
of each electrode based on the classifier coefficients. Results are saved to
CSV and optionally plotted on brain hemispheres.
"""

from neuroprobe.braintreebank_subject import BrainTreebankSubject
import neuroprobe.train_test_splits as neuroprobe_train_test_splits
import neuroprobe.config as neuroprobe_config

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import torch, numpy as np
import argparse, json, os, time
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from eval_utils import *

preprocess_options = [
    'none', # no preprocessing, just raw voltage
    'stft_absangle', # magnitude and phase after FFT
    'stft_realimag', # real and imaginary parts after FFT
    'stft_abs', # just magnitude after FFT ("spectrogram")
    'laplacian', # Laplacian rereference
    'remove_line_noise', # remove line noise from the raw voltage
    'downsample_200', # downsample to 200 Hz
]
splits_options = [
    'WithinSession', # same subject, same trial
    'CrossSession', # same subject, different trial
    'CrossSubject', # different subject, different trial
]

parser = argparse.ArgumentParser(description='Extract electrode importance for classification tasks')
parser.add_argument('--eval_name', type=str, default='onset', help='Evaluation name (e.g. onset, scene, gpt2_surprisal)')
parser.add_argument('--split_type', type=str, choices=splits_options, default='WithinSession', 
                    help=f'Type of splits to use ({", ".join(splits_options)})')
parser.add_argument('--subject_id', type=int, required=True, help='Subject ID')
parser.add_argument('--trial_id', type=int, required=True, help='Trial ID')

parser.add_argument('--verbose', action='store_true', help='Whether to print progress')
parser.add_argument('--save_dir', type=str, default='eval_results', help='Directory to save results')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

parser.add_argument('--only_1second', action='store_true', 
                    help='Whether to only evaluate on 1 second after word onset')
parser.add_argument('--full', action='store_true', 
                    help='Whether to use the full eval for Neuroprobe (NOTE: Lite is the default!)')
parser.add_argument('--nano', action='store_true', 
                    help='Whether to use Neuroprobe Nano for faster evaluation')
parser.add_argument('--binary_tasks', type=lambda x: x.lower() == 'true', default=True, 
                    help='Whether to use binary classification for tasks that support it')

parser.add_argument('--preprocess.type', type=str, default='laplacian-stft_abs', 
                    help=f'Preprocessing to apply to neural data ({", ".join(preprocess_options)})')
parser.add_argument('--preprocess.stft.nperseg', type=int, default=512, 
                    help='Length of each segment for FFT calculation')
parser.add_argument('--preprocess.stft.poverlap', type=float, default=0.75, 
                    help='Overlap percentage for FFT calculation')
parser.add_argument('--preprocess.stft.window', type=str, choices=['hann', 'boxcar'], default='hann', 
                    help='Window type for FFT calculation')
parser.add_argument('--preprocess.stft.max_frequency', type=int, default=150, 
                    help='Maximum frequency (Hz) to keep after FFT calculation')
parser.add_argument('--preprocess.stft.min_frequency', type=int, default=0, 
                    help='Minimum frequency (Hz) to keep after FFT calculation')

parser.add_argument('--classifier_type', type=str, choices=['linear'], default='linear', 
                    help='Type of classifier (only linear supported for importance extraction)')
parser.add_argument('--bin_start', type=float, default=0.0, 
                    help='Time bin start (seconds relative to word onset)')
parser.add_argument('--bin_end', type=float, default=1.0, 
                    help='Time bin end (seconds relative to word onset)')
parser.add_argument('--plot', action='store_true', 
                    help='Whether to generate brain plots')
parser.add_argument('--plot_3d', action='store_true', 
                    help='Whether to generate 3D scatter plot')

args = parser.parse_args()

eval_name = args.eval_name
splits_type = args.split_type
subject_id = args.subject_id
trial_id = args.trial_id

verbose = bool(args.verbose)
save_dir = args.save_dir
seed = args.seed

only_1second = bool(args.only_1second)
lite = not bool(args.full)
nano = bool(args.nano)
assert (not nano) or (splits_type != "CrossSession"), "Nano only works with WithinSession or CrossSubject splits"
assert (not nano) or lite, "--nano and --full cannot be used together"
binary_tasks = bool(args.binary_tasks)

preprocess_type = getattr(args, 'preprocess.type')
preprocess_parameters = {
    "type": preprocess_type,
    "stft": {
        "nperseg": getattr(args, 'preprocess.stft.nperseg'),
        "poverlap": getattr(args, 'preprocess.stft.poverlap'),
        "window": getattr(args, 'preprocess.stft.window'),
        "max_frequency": getattr(args, 'preprocess.stft.max_frequency'),
        "min_frequency": getattr(args, 'preprocess.stft.min_frequency')
    }
}

classifier_type = args.classifier_type
if classifier_type != 'linear':
    raise ValueError("Only linear classifiers are supported for electrode importance extraction")

bin_start = args.bin_start
bin_end = args.bin_end
plot_brain = bool(args.plot)
plot_3d = bool(args.plot_3d)

# Set random seeds for reproducibility
np.random.seed(seed)
torch.manual_seed(seed)

bins_start_before_word_onset_seconds = 0.5 if not only_1second else 0
bins_end_after_word_onset_seconds = 1.5 if not only_1second else 1

# Load subject
subject = BrainTreebankSubject(subject_id, cache=True, dtype=torch.float32)
subset_electrodes(subject, lite=lite, nano=nano)
subject.load_neural_data(trial_id)

if verbose:
    log(f"Loaded subject {subject_id}, trial {trial_id}", priority=0)
    log(f"Number of electrodes: {len(subject.electrode_labels)}", priority=0)

# Generate splits
if splits_type == "WithinSession":
    folds = neuroprobe_train_test_splits.generate_splits_within_session(
        subject, trial_id, eval_name, dtype=torch.float32, 
        output_indices=False, 
        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*neuroprobe_config.SAMPLING_RATE), 
        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*neuroprobe_config.SAMPLING_RATE),
        lite=lite, nano=nano, binary_tasks=binary_tasks)
    train_subject = subject
elif splits_type == "CrossSession":
    folds = neuroprobe_train_test_splits.generate_splits_cross_session(
        subject, trial_id, eval_name, dtype=torch.float32, 
        output_indices=False, 
        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*neuroprobe_config.SAMPLING_RATE), 
        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*neuroprobe_config.SAMPLING_RATE),
        lite=lite, binary_tasks=binary_tasks)
    train_subject = subject
elif splits_type == "CrossSubject":
    if verbose: 
        log("Loading the training subject...", priority=0)
    train_subject_id = neuroprobe_config.DS_DM_TRAIN_SUBJECT_ID
    train_subject = BrainTreebankSubject(train_subject_id, allow_corrupted=False, cache=True, dtype=torch.float32)
    train_subject_electrodes = neuroprobe_config.NEUROPROBE_LITE_ELECTRODES[train_subject.subject_identifier] if lite else train_subject.electrode_labels
    train_subject.set_electrode_subset(train_subject_electrodes)
    if verbose: 
        log("Subject loaded.", priority=0)
    folds = neuroprobe_train_test_splits.generate_splits_cross_subject(
        {subject_id: subject, train_subject_id: train_subject}, 
        subject_id, trial_id, eval_name, dtype=torch.float32, 
        output_indices=False, 
        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*neuroprobe_config.SAMPLING_RATE), 
        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*neuroprobe_config.SAMPLING_RATE),
        lite=lite, nano=nano, binary_tasks=binary_tasks)

if verbose:
    log(f"Generated {len(folds)} folds", priority=0)

def extract_electrode_importance(folds, train_subject, subject, preprocess_type, preprocess_parameters, 
                                  bin_start, bin_end, electrode_labels, verbose=False):
    """
    Extract electrode-level importance from trained classifiers.
    
    Parameters:
    - folds: list of train/test splits
    - train_subject: subject used for training
    - subject: subject used for testing
    - preprocess_type: preprocessing type
    - preprocess_parameters: preprocessing parameters
    - bin_start: time bin start (seconds relative to word onset)
    - bin_end: time bin end (seconds relative to word onset)
    - electrode_labels: list of electrode labels
    - verbose: whether to print progress
    
    Returns:
    - electrode_importance: array of importance scores per electrode (averaged across folds)
    """
    all_electrode_importances = []
    
    data_idx_from = int((bin_start + bins_start_before_word_onset_seconds) * neuroprobe_config.SAMPLING_RATE)
    data_idx_to = int((bin_end + bins_start_before_word_onset_seconds) * neuroprobe_config.SAMPLING_RATE)
    
    def get_data_and_label(item):
        if isinstance(item, dict):
            return item["data"], item["label"]
        else:
            return item[0], item[1]
    
    for fold_idx, fold in enumerate(folds):
        train_dataset = fold["train_dataset"]
        test_dataset = fold["test_dataset"]
        
        if verbose:
            log(f"Processing fold {fold_idx+1}/{len(folds)}...", priority=1, indent=1)
        
        # Prepare and preprocess data
        X_train = np.concatenate([
            preprocess_data(
                get_data_and_label(item)[0][:, data_idx_from:data_idx_to].unsqueeze(0),
                train_subject.electrode_labels,
                preprocess_type,
                preprocess_parameters
            ).float().numpy() for item in train_dataset
        ], axis=0)
        y_train = np.array([get_data_and_label(item)[1] for item in train_dataset])
        
        # Flatten for linear classifier
        original_shape = X_train.shape
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        
        # Train classifier
        clf = LogisticRegression(random_state=seed, max_iter=10000, tol=1e-3)
        clf.fit(X_train_scaled, y_train)
        
        # Extract coefficients
        # For binary: clf.coef_ is (1, n_features)
        # For multi-class: clf.coef_ is (n_classes, n_features)
        if len(clf.coef_.shape) == 2:
            # Multi-class: average across classes or use first class
            coef = np.abs(clf.coef_[0])
        else:
            coef = np.abs(clf.coef_)
        
        # Reshape to match electrode structure
        # Need to figure out feature dimensions per electrode
        n_features_total = coef.shape[0]
        n_electrodes = len(electrode_labels)
        
        # Calculate features per electrode from the original shape before flattening
        # original_shape is (n_samples, n_electrodes, ...)
        if len(original_shape) == 4:  # (n_samples, n_electrodes, n_timebins, n_freqs) - STFT
            n_features_per_electrode = original_shape[2] * original_shape[3]
        elif len(original_shape) == 3:  # (n_samples, n_electrodes, n_timebins)
            n_features_per_electrode = original_shape[2]
        elif len(original_shape) == 2:  # (n_samples, n_electrodes)
            n_features_per_electrode = 1
        else:
            # Fallback: divide evenly (shouldn't happen, but just in case)
            n_features_per_electrode = n_features_total // n_electrodes
            if verbose:
                log(f"Warning: Unexpected shape {original_shape}, using {n_features_per_electrode} features per electrode", priority=1, indent=2)
        
        # Verify the math
        if n_features_per_electrode * n_electrodes != n_features_total:
            raise ValueError(f"Feature count mismatch: {n_electrodes} electrodes * {n_features_per_electrode} features = {n_electrodes * n_features_per_electrode}, but got {n_features_total} total features")
        
        # Reshape coefficients to (n_electrodes, n_features_per_electrode)
        coef_reshaped = coef.reshape(n_electrodes, n_features_per_electrode)
        
        # Aggregate across features: mean absolute weight per electrode
        electrode_importance = np.mean(coef_reshaped, axis=1)
        
        all_electrode_importances.append(electrode_importance)
        
        if verbose:
            log(f"Fold {fold_idx+1}: Extracted importance for {len(electrode_labels)} electrodes", priority=2, indent=2)
    
    # Average across folds
    electrode_importance_avg = np.mean(all_electrode_importances, axis=0)
    
    return electrode_importance_avg

# Extract importance
if verbose:
    log(f"Extracting electrode importance for {eval_name}...", priority=0)
    log(f"Time bin: {bin_start:.2f} to {bin_end:.2f} seconds", priority=1, indent=1)

electrode_importance = extract_electrode_importance(
    folds, train_subject, subject, preprocess_type, preprocess_parameters,
    bin_start, bin_end, subject.electrode_labels, verbose=verbose
)

# Create DataFrame with results
results_df = pd.DataFrame({
    'electrode': subject.electrode_labels,
    'importance': electrode_importance
})
results_df = results_df.sort_values('importance', ascending=False)

if verbose:
    log(f"\nTop 10 most important electrodes:", priority=0)
    for idx, row in results_df.head(10).iterrows():
        log(f"  {row['electrode']}: {row['importance']:.4f}", priority=0)

# Save results
preprocess_suffix = f"{preprocess_type}" if preprocess_type != 'none' else 'voltage'
preprocess_suffix += f"_nperseg{preprocess_parameters['stft']['nperseg']}" if 'stft' in preprocess_type else ''
preprocess_suffix += f"_poverlap{preprocess_parameters['stft']['poverlap']}" if 'stft' in preprocess_type else ''
preprocess_suffix += f"_{preprocess_parameters['stft']['window']}" if 'stft' in preprocess_type and preprocess_parameters['stft']['window'] != 'hann' else ''
preprocess_suffix += f"_maxfreq{preprocess_parameters['stft']['max_frequency']}" if 'stft' in preprocess_type else ''
preprocess_suffix += f"_minfreq{preprocess_parameters['stft']['min_frequency']}" if 'stft' in preprocess_type and preprocess_parameters['stft']['min_frequency'] != 0 else ''

file_save_dir = f"{save_dir}/electrode_importance_{preprocess_suffix}"
os.makedirs(file_save_dir, exist_ok=True)

csv_filename = f"{file_save_dir}/electrode_importance_{subject.subject_identifier}_{trial_id}_{eval_name}_bin{bin_start:.2f}to{bin_end:.2f}.csv"
results_df.to_csv(csv_filename, index=False)

if verbose:
    log(f"Results saved to {csv_filename}", priority=0)

# Plot 3D scatter if requested
if plot_3d:
    electrode_coords = subject.get_electrode_coordinates().numpy()
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        electrode_coords[:, 0],  # X (Left-Right)
        electrode_coords[:, 1],  # Y (Posterior-Anterior)
        electrode_coords[:, 2],  # Z (Inferior-Superior)
        c=electrode_importance,
        cmap='viridis',
        s=100
    )
    
    plt.colorbar(scatter, label='Electrode Importance')
    ax.set_xlabel('Left-Right (mm)')
    ax.set_ylabel('Posterior-Anterior (mm)')
    ax.set_zlabel('Inferior-Superior (mm)')
    ax.set_title(f'Electrode Importance: {eval_name} (Subject {subject_id}, Trial {trial_id})')
    
    plot_filename = f"{file_save_dir}/electrode_importance_{subject.subject_identifier}_{trial_id}_{eval_name}_bin{bin_start:.2f}to{bin_end:.2f}_3d.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    if verbose:
        log(f"3D plot saved to {plot_filename}", priority=0)
    plt.close()

# Plot hemispheres if requested
if plot_brain:
    # Set up paths
    base_path = os.path.join(neuroprobe_config.ROOT_DIR, 'localization')
    left_hem_file_name = 'left_hem_clean.png'
    right_hem_file_name = 'right_hem_clean.png'
    coords_file_name = 'elec_coords_full.csv'
    
    # Load hemisphere images and coordinates
    left_hem_img = plt.imread(os.path.join(base_path, left_hem_file_name))
    right_hem_img = plt.imread(os.path.join(base_path, right_hem_file_name))
    coords_df = pd.read_csv(os.path.join(base_path, coords_file_name))
    
    # Process electrode IDs
    split_elec_id = coords_df['ID'].str.split('-')
    coords_df['Subject'] = [t[0] for t in split_elec_id]
    coords_df['Electrode'] = [t[1] for t in split_elec_id]
    
    # Scale coordinates
    matlab_xlim = (-108.0278, 108.0278)
    matlab_ylim = (-72.9774, 72.9774)
    
    x_scale = left_hem_img.shape[1] / (matlab_xlim[1] - matlab_xlim[0])
    y_scale_l = left_hem_img.shape[0] / (matlab_ylim[1] - matlab_ylim[0])
    y_scale_r = right_hem_img.shape[0] / (matlab_ylim[1] - matlab_ylim[0])
    
    def scale(x, s, d):
        return -(x - d) * s
    
    scaled_coords_df = coords_df.copy()
    scaled_coords_df.loc[scaled_coords_df['Hemisphere'] == 1, 'X'] = coords_df.loc[coords_df['Hemisphere'] == 1, 'X'].apply(lambda x: scale(x, x_scale, matlab_xlim[1]))
    scaled_coords_df.loc[scaled_coords_df['Hemisphere'] == 1, 'Y'] = coords_df.loc[coords_df['Hemisphere'] == 1, 'Y'].apply(lambda x: scale(x, y_scale_l, matlab_ylim[1]))
    scaled_coords_df.loc[scaled_coords_df['Hemisphere'] == 0, 'X'] = coords_df.loc[coords_df['Hemisphere'] == 0, 'X'].apply(lambda x: -scale(x, y_scale_r, matlab_xlim[0]))
    scaled_coords_df.loc[scaled_coords_df['Hemisphere'] == 0, 'Y'] = coords_df.loc[coords_df['Hemisphere'] == 0, 'Y'].apply(lambda x: scale(x, y_scale_r, matlab_ylim[1]))
    
    def plot_hemisphere_axis(electrode_importance_dict, ax=None, hemisphere="left", title=None, vmin=None, vmax=None, cmap='viridis'):
        """
        Plot electrodes on a hemisphere colored by importance.
        
        Parameters:
        - electrode_importance_dict: dict mapping electrode_label -> importance_value
        - ax: matplotlib axis
        - hemisphere: "left" or "right"
        - title: title for the plot
        - vmin, vmax: color scale limits
        - cmap: colormap to use
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.set_aspect('equal')
        
        if hemisphere == "left":
            ax.imshow(left_hem_img)
            hem_index = 1
        elif hemisphere == "right":
            ax.imshow(right_hem_img)
            hem_index = 0
        else:
            raise ValueError("hemisphere must be 'left' or 'right'")
        
        ax.axis('off')
        
        # Get coordinates for this subject and hemisphere
        subject_str = f'sub_{subject_id}'
        
        all_x, all_y, all_colors = [], [], []
        
        for electrode_label, importance in electrode_importance_dict.items():
            # Find coordinates for this electrode
            coords = scaled_coords_df[
                (scaled_coords_df.Subject == subject_str) & 
                (scaled_coords_df.Electrode == electrode_label) & 
                (scaled_coords_df.Hemisphere == hem_index)
            ]
            
            if len(coords) > 0:
                x = coords['X'].values[0]
                y = coords['Y'].values[0]
                all_x.append(x)
                all_y.append(y)
                all_colors.append(importance)
        
        if len(all_x) == 0:
            if verbose:
                log(f"No electrodes found for {hemisphere} hemisphere", priority=1)
            return None
        
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        all_colors = np.array(all_colors)
        
        # Set color scale if not provided
        if vmin is None:
            vmin = np.min(all_colors)
        if vmax is None:
            vmax = np.max(all_colors)
        
        # Sort by color so lower values appear on top
        sort_idx = np.argsort(all_colors)[::-1]
        
        sc = ax.scatter(all_x[sort_idx], all_y[sort_idx], 
                       c=all_colors[sort_idx], 
                       vmin=vmin, vmax=vmax, 
                       s=100, 
                       edgecolors='black', 
                       linewidths=0.5,
                       cmap=cmap)
        
        if title:
            ax.set_title(title, fontsize=14, pad=10)
        
        return sc
    
    # Create electrode importance dictionary
    electrode_importance_dict = dict(zip(subject.electrode_labels, electrode_importance))
    
    # Create figure with both hemispheres
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot both hemispheres with shared color scale
    vmin = np.min(electrode_importance)
    vmax = np.max(electrode_importance)
    
    sc1 = plot_hemisphere_axis(electrode_importance_dict, ax=axes[0], hemisphere="left", 
                                title=f"Left Hemisphere\n{eval_name} (Subject {subject_id}, Trial {trial_id})",
                                vmin=vmin, vmax=vmax, cmap='viridis')
    sc2 = plot_hemisphere_axis(electrode_importance_dict, ax=axes[1], hemisphere="right", 
                                title=f"Right Hemisphere\n{eval_name} (Subject {subject_id}, Trial {trial_id})",
                                vmin=vmin, vmax=vmax, cmap='viridis')
    
    # Add colorbar
    if sc1 is not None or sc2 is not None:
        sc = sc1 if sc1 is not None else sc2
        cbar = plt.colorbar(sc, ax=axes, fraction=0.046, pad=0.04)
        cbar.set_label('Electrode Importance', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    plot_filename = f"{file_save_dir}/electrode_importance_{subject.subject_identifier}_{trial_id}_{eval_name}_bin{bin_start:.2f}to{bin_end:.2f}_hemispheres.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    if verbose:
        log(f"Hemisphere plot saved to {plot_filename}", priority=0)
    plt.close()

if verbose:
    log("Done!", priority=0)

