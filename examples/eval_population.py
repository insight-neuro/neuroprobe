from neuroprobe.braintreebank_subject import BrainTreebankSubject
import neuroprobe.train_test_splits as neuroprobe_train_test_splits
import neuroprobe.config as neuroprobe_config

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch, numpy as np
import argparse, json, os, time
import gc  # Add at top with other imports

from eval_utils import *

def _make_unique_path(path: str) -> str:
    """
    If `path` already exists, return a new path by appending `_run{n}` before the extension.
    Example: `foo.json` -> `foo_run1.json`, `foo_run2.json`, ...
    """
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    i = 1
    while True:
        candidate = f"{root}_run{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


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

parser = argparse.ArgumentParser()
parser.add_argument('--eval_name', type=str, default='onset', help='Evaluation name(s) (e.g. onset, gpt2_surprisal). If multiple, separate with commas.')
parser.add_argument('--split_type', type=str, choices=splits_options, default='WithinSession', help=f'Type of splits to use ({", ".join(splits_options)})')
parser.add_argument('--subject_id', type=int, required=True, help='Subject ID')
parser.add_argument('--trial_id', type=int, required=True, help='Trial ID')

parser.add_argument('--verbose', action='store_true', help='Whether to print progress')
parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing results')
parser.add_argument('--save_dir', type=str, default='eval_results', help='Directory to save results')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--if_exists', type=str, choices=['new', 'resume', 'skip'], default='new',
                    help="If the output JSON already exists and --overwrite is not set: "
                         "'new' creates a new file with a different name, "
                         "'resume' loads and continues, "
                         "'skip' skips entirely.")

parser.add_argument('--only_1second', action='store_true', help='Whether to only evaluate on 1 second after word onset') # NOTE: set this to true for the Neuroprobe benchmark
parser.add_argument('--full', action='store_true', help='Whether to use the full eval for Neuroprobe (NOTE: Lite is the default!)')
parser.add_argument('--nano', action='store_true', help='Whether to use Neuroprobe Nano for faster evaluation')
parser.add_argument('--binary_tasks', type=lambda x: x.lower() == 'true', default=True, help='Whether to use binary classification for tasks that support it')

parser.add_argument('--preprocess.type', type=str, default='laplacian-stft_abs', help=f'Preprocessing to apply to neural data ({", ".join(preprocess_options)})')
parser.add_argument('--preprocess.stft.nperseg', type=int, default=512, help='Length of each segment for FFT calculation (only used if preprocess is stft_absangle, stft_realimag, or stft_abs)')
parser.add_argument('--preprocess.stft.poverlap', type=float, default=0.75, help='Overlap percentage for FFT calculation (only used if preprocess is stft_absangle, stft_realimag, or stft_abs)')
parser.add_argument('--preprocess.stft.window', type=str, choices=['hann', 'boxcar'], default='hann', help='Window type for FFT calculation (only used if preprocess is stft_absangle, stft_realimag, or stft_abs)')
parser.add_argument('--preprocess.stft.max_frequency', type=int, default=150, help='Maximum frequency (Hz) to keep after FFT calculation (only used if preprocess is stft_absangle, stft_realimag, or stft_abs)')
parser.add_argument('--preprocess.stft.min_frequency', type=int, default=0, help='Minimum frequency (Hz) to keep after FFT calculation (only used if preprocess is stft_absangle, stft_realimag, or stft_abs)')

# Optional: run only a single time bin (relative to word onset, in seconds).
# Example: --only_bin_start 0.125 --only_bin_end 0.375
parser.add_argument('--only_bin_start', type=float, default=None,
                    help='If set with --only_bin_end, run evaluation only for this time bin start (seconds, relative to word onset).')
parser.add_argument('--only_bin_end', type=float, default=None,
                    help='If set with --only_bin_start, run evaluation only for this time bin end (seconds, relative to word onset).')

parser.add_argument('--classifier_type', type=str, choices=['linear', 'cnn', 'transformer', 'mlp', 'hybrid', 'gnn', 'dae', 'vae', 'brainbert'], default='linear', help='Type of classifier to use for evaluation')
parser.add_argument('--gnn_variant', type=str, default='gnn_v1_stgcn',
                    choices=['gnn_v0_bugfix', 'gnn_v1_stgcn', 'gnn_v2_gat'],
                    help='GNN architecture variant (only used when --classifier_type gnn)')
parser.add_argument('--brainbert.path', type=str, default=None, help='Path to BrainBERT directory (if not provided, will try to auto-detect)')
parser.add_argument('--brainbert.pretrained', type=lambda x: x.lower() == 'true', default=True, help='Whether to use pretrained BrainBERT weights (default: True)')
parser.add_argument('--brainbert.frozen', type=lambda x: x.lower() == 'true', default=True, help='Whether to freeze BrainBERT weights (default: True)')
args = parser.parse_args()

eval_names = args.eval_name.split(',')
splits_type = args.split_type
subject_id = args.subject_id
trial_id = args.trial_id

verbose = bool(args.verbose)
overwrite = bool(args.overwrite)
save_dir = args.save_dir
seed = args.seed
if_exists = getattr(args, 'if_exists', 'new')

only_1second = bool(args.only_1second)
lite = not bool(args.full)
nano = bool(args.nano)
assert (not nano) or (splits_type != "CrossSession"), "Nano only works with WithinSession or CrossSubject splits; does not work with CrossSession."
assert (not nano) or lite, "--nano and --full cannot be used together. Neuroprobe Full and Neuroprobe Nano are different evaluations."
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
gnn_variant = args.gnn_variant

# BrainBERT-specific arguments
brainbert_path = getattr(args, 'brainbert.path', None)
brainbert_pretrained = getattr(args, 'brainbert.pretrained', True)
brainbert_frozen = getattr(args, 'brainbert.frozen', True)

model_name = model_name_from_classifier_type(classifier_type)

# Set random seeds for reproducibility
np.random.seed(seed)
torch.manual_seed(seed)

bins_start_before_word_onset_seconds = 0.5 if not only_1second else 0
bins_end_after_word_onset_seconds = 1.5 if not only_1second else 1
bin_size_seconds = 0.25
bin_step_size_seconds = 0.125

bin_starts = []
bin_ends = []
if not only_1second:
    for bin_start in np.arange(-bins_start_before_word_onset_seconds, bins_end_after_word_onset_seconds-bin_size_seconds, bin_step_size_seconds):
        bin_end = bin_start + bin_size_seconds
        if bin_end > bins_end_after_word_onset_seconds: break

        bin_starts.append(bin_start)
        bin_ends.append(bin_end)
    bin_starts += [-bins_start_before_word_onset_seconds]
    bin_ends += [bins_end_after_word_onset_seconds]
bin_starts += [0]
bin_ends += [1]

# SINGLE_BIN_OVERRIDE (easy to comment out to restore old behavior)
if args.only_bin_start is not None or args.only_bin_end is not None:
    if args.only_bin_start is None or args.only_bin_end is None:
        raise ValueError("Provide both --only_bin_start and --only_bin_end")
    if not (args.only_bin_end > args.only_bin_start):
        raise ValueError("--only_bin_end must be > --only_bin_start")
    bin_starts = [float(args.only_bin_start)]
    bin_ends = [float(args.only_bin_end)]


# use cache=True to load this trial's neural data into RAM, if you have enough memory!
# It will make the loading process faster.
subject = BrainTreebankSubject(subject_id, cache=True, dtype=torch.float32)
subset_electrodes(subject, lite=lite, nano=nano)
neural_data_loaded = False

for eval_name in eval_names:
    start_time = time.time()

    preprocess_suffix = f"{preprocess_type}" if preprocess_type != 'none' else 'voltage'
    preprocess_suffix += f"_nperseg{preprocess_parameters['stft']['nperseg']}" if 'stft' in preprocess_type else ''
    preprocess_suffix += f"_poverlap{preprocess_parameters['stft']['poverlap']}" if 'stft' in preprocess_type else ''
    preprocess_suffix += f"_{preprocess_parameters['stft']['window']}" if 'stft' in preprocess_type and preprocess_parameters['stft']['window'] != 'hann' else ''
    preprocess_suffix += f"_maxfreq{preprocess_parameters['stft']['max_frequency']}" if 'stft' in preprocess_type else ''
    preprocess_suffix += f"_minfreq{preprocess_parameters['stft']['min_frequency']}" if 'stft' in preprocess_type and preprocess_parameters['stft']['min_frequency'] != 0 else ''
    
    file_save_dir = f"{save_dir}/{classifier_type}_{preprocess_suffix}"
    os.makedirs(file_save_dir, exist_ok=True) # Create save directory if it doesn't exist

    file_save_path = f"{file_save_dir}/population_{subject.subject_identifier}_{trial_id}_{eval_name}.json"
    # If a file already exists and we're not overwriting, decide whether to resume, skip, or create a new file.
    if os.path.exists(file_save_path) and not overwrite:
        if if_exists == 'skip':
            log(f"Skipping {file_save_path} because it already exists (--if_exists skip)", priority=0)
            continue
        elif if_exists == 'new':
            new_path = _make_unique_path(file_save_path)
            if verbose:
                log(f"Output exists; writing to new file: {new_path}", priority=0)
            file_save_path = new_path
        elif if_exists == 'resume':
            # Existing resume logic below will load/merge results.
            pass
        else:
            raise ValueError(f"Invalid --if_exists value: {if_exists}")
    # REMOVE THIS EARLY SKIP - let the resume logic handle it instead
    # if os.path.exists(file_save_path) and not overwrite:
    #     log(f"Skipping {file_save_path} because it already exists", priority=0)
    #     continue

    # Load neural data if it hasn't been loaded yet; NOTE: this is done here to avoid unnecessary loading of neural data if the file is going to be skipped.
    if not neural_data_loaded:
        start_time = time.time()
        subject.load_neural_data(trial_id)
        subject_load_time = time.time() - start_time
        if verbose:
            log(f"Subject loaded in {subject_load_time:.2f} seconds", priority=0)
        neural_data_loaded = True

    results_population = {
        "time_bins": [],
    }
    
    # Initialize results structure early so we can save incrementally
    # Build description
    if classifier_type == 'brainbert':
        description = f"BrainBERT ({'pretrained' if brainbert_pretrained else 'untrained'}, {'frozen' if brainbert_frozen else 'trainable'}) using all electrodes ({preprocess_type if preprocess_type != 'none' else 'voltage'})."
    else:
        description = f"Simple {model_name} using all electrodes ({preprocess_type if preprocess_type != 'none' else 'voltage'})."
    
    results = {
        "model_name": model_name,
        "author": "Andrii Zahorodnii",
        "description": description,
        "organization": "MIT",
        "organization_url": "https://azaho.org/",
        "timestamp": time.time(),
        "evaluation_results": {
            f"{subject.subject_identifier}_{trial_id}": {
                "population": results_population
            }
        },
        "config": {
            "preprocess": preprocess_parameters,
            "only_1second": only_1second,
            "seed": seed,
            "subject_id": subject_id,
            "trial_id": trial_id,
            "splits_type": splits_type,
            "classifier_type": classifier_type,
        },
        "timing": {
            "subject_load_time": subject_load_time,
            "regression_run_time": None,  # Will be updated at the end
        }
    }
    
    # Add BrainBERT-specific config if applicable
    if classifier_type == 'brainbert':
        results["config"]["brainbert"] = {
            "path": brainbert_path,
            "pretrained": brainbert_pretrained,
            "frozen": brainbert_frozen
        }
    
    # Load existing results if file exists (for resuming)
    if os.path.exists(file_save_path) and not overwrite and if_exists == 'resume':
        try:
            with open(file_save_path, "r") as f:
                existing_results = json.load(f)
                # Merge existing results
                if "evaluation_results" in existing_results:
                    existing_population = existing_results["evaluation_results"].get(
                        f"{subject.subject_identifier}_{trial_id}", {}
                    ).get("population", {})
                    # Restore completed bins
                    if "time_bins" in existing_population:
                        results_population["time_bins"] = existing_population["time_bins"]
                    if "whole_window" in existing_population:
                        results_population["whole_window"] = existing_population["whole_window"]
                    if "one_second_after_onset" in existing_population:
                        results_population["one_second_after_onset"] = existing_population["one_second_after_onset"]
                    if verbose:
                        log(f"Loaded existing results: {len(results_population.get('time_bins', []))} time bins, whole_window: {'yes' if 'whole_window' in results_population else 'no'}, one_second: {'yes' if 'one_second_after_onset' in results_population else 'no'}", priority=0)
        except Exception as e:
            if verbose:
                log(f"Could not load existing results (starting fresh): {e}", priority=1)

    # Check if all bins are already complete (only skip if everything is done)
    # First, determine what bins we expect
    needs_whole_window = False
    needs_one_second = False
    expected_time_bins = set()
    for bin_start, bin_end in zip(bin_starts, bin_ends):
        if bin_start == -bins_start_before_word_onset_seconds and bin_end == bins_end_after_word_onset_seconds and not only_1second:
            needs_whole_window = True
        elif bin_start == 0 and bin_end == 1:
            needs_one_second = True
        else:
            expected_time_bins.add((float(bin_start), float(bin_end)))
    
    # Check if all expected bins are present
    all_bins_complete = True
    if needs_whole_window and "whole_window" not in results_population:
        all_bins_complete = False
    if needs_one_second and "one_second_after_onset" not in results_population:
        all_bins_complete = False
    
    # Check time bins
    existing_time_bins = set()
    for existing_bin in results_population.get("time_bins", []):
        existing_time_bins.add((existing_bin.get("time_bin_start"), existing_bin.get("time_bin_end")))
    
    missing_time_bins = expected_time_bins - existing_time_bins
    if missing_time_bins:
        all_bins_complete = False
        if verbose:
            log(f"Missing {len(missing_time_bins)} time bins, will resume from last completed bin", priority=0)
    
    if all_bins_complete and not overwrite and if_exists == 'resume':
        log(f"Skipping {file_save_path} because all bins are already complete", priority=0)
        continue

    if splits_type == "WithinSession":
        folds = neuroprobe_train_test_splits.generate_splits_within_session(subject, trial_id, eval_name, dtype=torch.float32, 
                                                                                        output_indices=False, 
                                                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*neuroprobe_config.SAMPLING_RATE), 
                                                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*neuroprobe_config.SAMPLING_RATE),
                                                                                        lite=lite, nano=nano, binary_tasks=binary_tasks)
        train_subject = subject
    elif splits_type == "CrossSession":
        folds = neuroprobe_train_test_splits.generate_splits_cross_session(subject, trial_id, eval_name, dtype=torch.float32, 
                                                                                        output_indices=False, 
                                                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*neuroprobe_config.SAMPLING_RATE), 
                                                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*neuroprobe_config.SAMPLING_RATE),
                                                                                        lite=lite, binary_tasks=binary_tasks)
        train_subject = subject
    elif splits_type == "CrossSubject":
        if verbose: log("Loading the training subject...", priority=0)
        train_subject_id = neuroprobe_config.DS_DM_TRAIN_SUBJECT_ID
        train_subject = BrainTreebankSubject(train_subject_id, allow_corrupted=False, cache=True, dtype=torch.float32)
        train_subject_electrodes = neuroprobe_config.NEUROPROBE_LITE_ELECTRODES[train_subject.subject_identifier] if lite else train_subject.electrode_labels
        train_subject.set_electrode_subset(train_subject_electrodes)
        all_subjects = {
            subject_id: subject,
            train_subject_id: train_subject,
        }
        if verbose: log("Subject loaded.", priority=0)
        folds = neuroprobe_train_test_splits.generate_splits_cross_subject(all_subjects, subject_id, trial_id, eval_name, dtype=torch.float32, 
                                                                                        output_indices=False, 
                                                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*neuroprobe_config.SAMPLING_RATE), 
                                                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*neuroprobe_config.SAMPLING_RATE),
                                                                                        lite=lite, nano=nano, binary_tasks=binary_tasks)


    for bin_start, bin_end in zip(bin_starts, bin_ends):
        # Check if this bin has already been processed
        bin_already_processed = False
        if bin_start == -bins_start_before_word_onset_seconds and bin_end == bins_end_after_word_onset_seconds and not only_1second:
            bin_already_processed = "whole_window" in results_population
        elif bin_start == 0 and bin_end == 1:
            bin_already_processed = "one_second_after_onset" in results_population
        else:
            # Check if this time bin is already in the results
            for existing_bin in results_population.get("time_bins", []):
                if existing_bin.get("time_bin_start") == float(bin_start) and existing_bin.get("time_bin_end") == float(bin_end):
                    bin_already_processed = True
                    break
        
        if bin_already_processed and not overwrite:
            if verbose:
                log(f"Skipping bin {bin_start}-{bin_end} (already processed)", priority=1)
            continue
        
        data_idx_from = int((bin_start+bins_start_before_word_onset_seconds)*neuroprobe_config.SAMPLING_RATE)
        data_idx_to = int((bin_end+bins_start_before_word_onset_seconds)*neuroprobe_config.SAMPLING_RATE)

        bin_results = {
            "time_bin_start": float(bin_start),
            "time_bin_end": float(bin_end),
            "folds": []
        }

        # Loop over all folds
        for fold_idx, fold in enumerate(folds):
            train_dataset = fold["train_dataset"]
            test_dataset = fold["test_dataset"]

            log(f"Fold {fold_idx+1}, Bin {bin_start}-{bin_end}")
            log("Preparing and preprocessing data...", priority=2, indent=1)

            # Convert PyTorch dataset to numpy arrays for scikit-learn
            # X_train = np.concatenate([preprocess_data(item[0][:, data_idx_from:data_idx_to].unsqueeze(0), train_subject.electrode_labels, preprocess_type, preprocess_parameters).float().numpy() for item in train_dataset], axis=0)
            # y_train = np.array([item[1] for item in train_dataset])
            # X_test = np.concatenate([preprocess_data(item[0][:, data_idx_from:data_idx_to].unsqueeze(0), subject.electrode_labels, preprocess_type, preprocess_parameters).float().numpy() for item in test_dataset], axis=0)
            # y_test = np.array([item[1] for item in test_dataset])
            def get_data_and_label(item):
                if isinstance(item, dict):
                    return item["data"], item["label"]
                else:
                    return item[0], item[1]
            
            X_train = np.concatenate([preprocess_data(get_data_and_label(item)[0][:, data_idx_from:data_idx_to].unsqueeze(0), train_subject.electrode_labels, preprocess_type, preprocess_parameters).float().numpy() for item in train_dataset], axis=0)
            y_train = np.array([get_data_and_label(item)[1] for item in train_dataset])
            X_test = np.concatenate([preprocess_data(get_data_and_label(item)[0][:, data_idx_from:data_idx_to].unsqueeze(0), subject.electrode_labels, preprocess_type, preprocess_parameters).float().numpy() for item in test_dataset], axis=0)
            y_test = np.array([get_data_and_label(item)[1] for item in test_dataset])

            gc.collect()  # Collect after creating large arrays

            if splits_type == "CrossSubject":
                if verbose: log("Combining regions...", priority=2, indent=1)
                regions_train = get_region_labels(train_subject)
                regions_test = get_region_labels(subject)
                X_train, X_test, common_regions = combine_regions(X_train, X_test, regions_train, regions_test)

            # Flatten the data after preprocessing in-place (only for linear and mlp)
            original_X_train_shape = X_train.shape
            original_X_test_shape = X_test.shape
            
            # Get electrode coordinates for GNN (before any reshaping)
            electrode_coordinates = None
            if classifier_type == 'gnn':
                # Get electrode coordinates from the dataset
                if hasattr(train_dataset, 'electrode_coordinates'):
                    electrode_coordinates = train_dataset.electrode_coordinates
                elif hasattr(test_dataset, 'electrode_coordinates'):
                    electrode_coordinates = test_dataset.electrode_coordinates
                else:
                    # Fallback: get from subject
                    electrode_coordinates = train_subject.get_electrode_coordinates().numpy()
            
            if classifier_type in ['linear', 'mlp']:
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)
            # For cnn, transformer, hybrid, gnn, brainbert: keep original shape

            log(f"Standardizing data...", priority=2, indent=1)

            # Standardize the data
            if classifier_type in ['linear', 'mlp']:
                # Standardize flattened data
                scaler = StandardScaler(copy=False)
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            elif classifier_type == 'brainbert':
                # BrainBERT handles its own standardization internally
                # Just pass through the data as-is (should be STFT format)
                pass
            else:
                # Standardize multi-dimensional data (cnn, transformer, hybrid, gnn)
                # Standardize across samples but preserve spatial/temporal structure
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_test_flat = X_test.reshape(X_test.shape[0], -1)
                scaler = StandardScaler(copy=False)
                X_train_flat = scaler.fit_transform(X_train_flat)
                X_test_flat = scaler.transform(X_test_flat)
                X_train = X_train_flat.reshape(original_X_train_shape)
                X_test = X_test_flat.reshape(original_X_test_shape)
            gc.collect()  # Collect after standardization

            log(f"Training model...", priority=2, indent=1)

            # Train logistic regression
            if classifier_type == 'linear':
                clf = LogisticRegression(random_state=seed, max_iter=10000, tol=1e-3)
                clf.fit(X_train, y_train)
            elif classifier_type == 'cnn':
                clf = CNNClassifier(random_state=seed)
                clf.fit(X_train, y_train)
            elif classifier_type == 'transformer':
                clf = TransformerClassifier(random_state=seed)
                clf.fit(X_train, y_train)
            elif classifier_type == 'mlp':
                clf = MLPClassifier(random_state=seed)
                clf.fit(X_train, y_train, X_val=X_test, y_val=y_test)
            elif classifier_type == 'hybrid':
                clf = HybridCNNRNNClassifier(random_state=seed)
                clf.fit(X_train, y_train)
            elif classifier_type == 'gnn':
                clf = GNNClassifier(random_state=seed, gnn_variant=gnn_variant)
                clf.fit(X_train, y_train, electrode_coordinates=electrode_coordinates)
            elif classifier_type == 'dae':
                clf = DenoisingAutoencoderClassifier(random_state=seed)
                clf.fit(X_train, y_train)
            elif classifier_type == 'vae':
                clf = VariationalAutoencoderClassifier(random_state=seed)
                clf.fit(X_train, y_train)
            elif classifier_type == 'brainbert':
                # BrainBERT expects STFT input, so ensure preprocessing is done
                if 'stft' not in preprocess_type:
                    raise ValueError("BrainBERT requires STFT preprocessing. Please use a preprocess type that includes 'stft' (e.g., 'stft_abs', 'laplacian-stft_abs')")
                clf = BrainBERTClassifier(
                    random_state=seed,
                    brainbert_path=brainbert_path,
                    pretrained=brainbert_pretrained,
                    frozen=brainbert_frozen
                )
                clf.fit(X_train, y_train)

            torch.cuda.empty_cache()
            gc.collect()

            # Evaluate model
            train_accuracy = clf.score(X_train, y_train)
            test_accuracy = clf.score(X_test, y_test)

            # Get predictions - for multiclass classification
            train_probs = clf.predict_proba(X_train)
            test_probs = clf.predict_proba(X_test)
            gc.collect()  # Collect after predictions

            # Filter test samples to only include classes that were in training
            valid_class_mask = np.isin(y_test, clf.classes_)
            y_test_filtered = y_test[valid_class_mask]
            test_probs_filtered = test_probs[valid_class_mask]

            # Convert y_test to one-hot encoding
            y_test_onehot = np.zeros((len(y_test_filtered), len(clf.classes_)))
            for i, label in enumerate(y_test_filtered):
                class_idx = np.where(clf.classes_ == label)[0][0]
                y_test_onehot[i, class_idx] = 1

            y_train_onehot = np.zeros((len(y_train), len(clf.classes_)))
            for i, label in enumerate(y_train):
                class_idx = np.where(clf.classes_ == label)[0][0]
                y_train_onehot[i, class_idx] = 1

            # For multiclass ROC AUC, we need to calculate the score for each class
            n_classes = len(clf.classes_)
            if n_classes > 2:
                train_roc = roc_auc_score(y_train_onehot, train_probs, multi_class='ovr', average='macro')
                test_roc = roc_auc_score(y_test_onehot, test_probs_filtered, multi_class='ovr', average='macro')
            else:
                train_roc = roc_auc_score(y_train_onehot, train_probs)
                test_roc = roc_auc_score(y_test_onehot, test_probs_filtered)

            fold_result = {
                "train_accuracy": float(train_accuracy),
                "train_roc_auc": float(train_roc),
                "test_accuracy": float(test_accuracy),
                "test_roc_auc": float(test_roc)
            }
            bin_results["folds"].append(fold_result)
            
            # Clean up variables no longer needed
            del X_train, y_train, X_test, y_test, train_probs, test_probs
            del y_test_filtered, test_probs_filtered, y_test_onehot, y_train_onehot
            del clf
            if 'scaler' in locals():
                del scaler
            gc.collect()  # Collect after cleanup

            if verbose: 
                log(f"Population, Fold {fold_idx+1}, Bin {bin_start}-{bin_end}: Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}, Train ROC AUC: {train_roc:.3f}, Test ROC AUC: {test_roc:.3f}", priority=0, indent=0)

        if bin_start == -bins_start_before_word_onset_seconds and bin_end == bins_end_after_word_onset_seconds and not only_1second:
            results_population["whole_window"] = bin_results # whole window results
        elif bin_start == 0 and bin_end == 1:
            results_population["one_second_after_onset"] = bin_results # one second after onset results
        else:
            results_population["time_bins"].append(bin_results) # time bin results
        
        # Save incrementally after each bin
        results["evaluation_results"][f"{subject.subject_identifier}_{trial_id}"]["population"] = results_population
        results["timing"]["regression_run_time"] = time.time() - start_time
        with open(file_save_path, "w") as f:
            json.dump(results, f, indent=4)
        if verbose:
            log(f"Results saved incrementally after bin {bin_start}-{bin_end}", priority=1)
    
    regression_run_time = time.time() - start_time
    if verbose:
        log(f"Regression run in {regression_run_time:.2f} seconds", priority=0)

    # Update final timing (results already saved incrementally)
    results["timing"]["regression_run_time"] = regression_run_time
    with open(file_save_path, "w") as f:
        json.dump(results, f, indent=4)
    if verbose:
        log(f"Final results saved to {file_save_path}", priority=0)

    # Clean up at end of each eval_name loop
    del folds
    gc.collect()