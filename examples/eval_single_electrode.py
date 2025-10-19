from neuroprobe.braintreebank_subject import BrainTreebankSubject
import neuroprobe.train_test_splits as neuroprobe_train_test_splits
import neuroprobe.config as neuroprobe_config

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch, numpy as np   
import argparse, json, os, time

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
]

parser = argparse.ArgumentParser()
parser.add_argument('--eval_name', type=str, default='onset', help='Evaluation name(s) (e.g. onset, gpt2_surprisal). If multiple, separate with commas.')
parser.add_argument('--subject_id', type=int, required=True, help='Subject ID')
parser.add_argument('--trial_id', type=int, required=True, help='Trial ID')
parser.add_argument('--verbose', action='store_true', help='Whether to print progress')
parser.add_argument('--save_dir', type=str, default='eval_results', help='Directory to save results')

parser.add_argument('--preprocess.type', type=str, default='laplacian-stft_abs', help=f'Preprocessing to apply to neural data ({", ".join(preprocess_options)})')
parser.add_argument('--preprocess.stft.nperseg', type=int, default=512, help='Length of each segment for FFT calculation (only used if preprocess is stft_absangle, stft_realimag, or stft_abs)')
parser.add_argument('--preprocess.stft.poverlap', type=float, default=0.75, help='Overlap percentage for FFT calculation (only used if preprocess is stft_absangle, stft_realimag, or stft_abs)')
parser.add_argument('--preprocess.stft.window', type=str, choices=['hann', 'boxcar'], default='hann', help='Window type for FFT calculation (only used if preprocess is stft_absangle, stft_realimag, or stft_abs)')
parser.add_argument('--preprocess.stft.max_frequency', type=int, default=150, help='Maximum frequency (Hz) to keep after FFT calculation (only used if preprocess is stft_absangle, stft_realimag, or stft_abs)')
parser.add_argument('--preprocess.stft.min_frequency', type=int, default=0, help='Minimum frequency (Hz) to keep after FFT calculation (only used if preprocess is stft_absangle, stft_realimag, or stft_abs)')

parser.add_argument('--split_type', type=str, choices=splits_options, default='CrossSession', help=f'Type of splits to use ({", ".join(splits_options)})')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--only_1second', action='store_true', help='Whether to only evaluate on 1 second after word onset') # NOTE: set this to true for the Neuroprobe benchmark
parser.add_argument('--lite', action='store_true', help='Whether to use the lite eval for Neuroprobe (which is the default)')
parser.add_argument('--electrodes', type=str, default='all', help='Electrode labels to evaluate on. If multiple, separate with commas.')
parser.add_argument('--classifier_type', type=str, choices=['linear', 'cnn', 'transformer', 'mlp'], default='linear', help='Type of classifier to use for evaluation')
parser.add_argument('--binary_tasks', type=lambda x: x.lower() == 'true', default=True, help='Whether to use binary classification for tasks that support it')
args = parser.parse_args()

eval_names = args.eval_name.split(',') if ',' in args.eval_name else [args.eval_name]
subject_id = args.subject_id
trial_id = args.trial_id 
verbose = bool(args.verbose)
save_dir = args.save_dir
only_1second = bool(args.only_1second)
electrodes = args.electrodes.split(',') if ',' in args.electrodes else [args.electrodes]
seed = args.seed
lite = bool(args.lite)
splits_type = args.split_type
classifier_type = args.classifier_type
binary_tasks = bool(args.binary_tasks)

# Set up preprocessing parameters structure like in eval_population.py
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

model_name = model_name_from_classifier_type(classifier_type)

bins_start_before_word_onset_seconds = 0.5# if not only_1second else 0
bins_end_after_word_onset_seconds = 1.5# if not only_1second else 1
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

# Set random seeds for reproducibility
np.random.seed(seed)
torch.manual_seed(seed)

# use cache=True to load this trial's neural data into RAM, if you have enough memory!
# It will make the loading process faster.
subject = BrainTreebankSubject(subject_id, allow_corrupted=False, cache=True, dtype=torch.float32)

if lite:
    subject.set_electrode_subset(subject.get_lite_electrodes())
all_electrode_labels = subject.electrode_labels if electrodes[0] == 'all' else electrodes

for eval_name in eval_names:
    # Update save directory to use preprocess_type and parameters like in eval_population.py
    preprocess_suffix = f"{preprocess_type}" if preprocess_type != 'none' else 'voltage'
    preprocess_suffix += f"_nperseg{preprocess_parameters['stft']['nperseg']}" if 'stft' in preprocess_type else ''
    preprocess_suffix += f"_poverlap{preprocess_parameters['stft']['poverlap']}" if 'stft' in preprocess_type else ''
    preprocess_suffix += f"_{preprocess_parameters['stft']['window']}" if 'stft' in preprocess_type and preprocess_parameters['stft']['window'] != 'hann' else ''
    preprocess_suffix += f"_maxfreq{preprocess_parameters['stft']['max_frequency']}" if 'stft' in preprocess_type else ''
    preprocess_suffix += f"_minfreq{preprocess_parameters['stft']['min_frequency']}" if 'stft' in preprocess_type and preprocess_parameters['stft']['min_frequency'] != 0 else ''

    save_dir = f"{save_dir}/{classifier_type}_{preprocess_suffix}_single_electrode"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"electrode_{subject.subject_identifier}_{trial_id}_{eval_name}.json"
    
    # Load existing results if file exists
    results = {
        "model_name": model_name,
        "author": "Andrii Zahorodnii",
        "description": f"Simple {model_name}.",
        "organization": "MIT",
        "organization_url": "https://mit.edu",
        "timestamp": time.time(),
        "evaluation_results": {
            f"{subject.subject_identifier}_{trial_id}": {
                "electrode": {}
            }
        },
        "random_seed": seed
    }
    
    if os.path.exists(f"{save_dir}/{filename}"):
        with open(f"{save_dir}/{filename}", "r") as f:
            results = json.load(f)
    
    results_electrode = results["evaluation_results"][f"{subject.subject_identifier}_{trial_id}"]["electrode"]
    
    for electrode_idx, electrode_label in enumerate(all_electrode_labels):
        # Skip if electrode already processed
        if electrode_label in results_electrode:
            if verbose:
                log(f"Skipping electrode {electrode_label} - already processed", priority=0)
            continue
            
        subject.clear_neural_data_cache()
        subject.set_electrode_subset([electrode_label])
        subject.load_neural_data(trial_id)
        if verbose:
            log(f"Electrode {electrode_label} subject loaded", priority=0)

        results_electrode[electrode_label] = {
            "time_bins": [],
        }

        if splits_type == "WithinSession":
            folds = neuroprobe_train_test_splits.generate_splits_within_session(subject, trial_id, eval_name, dtype=torch.float32, 
                                                                                            output_indices=False, 
                                                                                            start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*neuroprobe_config.SAMPLING_RATE), 
                                                                                            end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*neuroprobe_config.SAMPLING_RATE),
                                                                                            lite=lite, max_samples=3500, binary_tasks=binary_tasks)
        elif splits_type == "CrossSession":
            folds = neuroprobe_train_test_splits.generate_splits_cross_session(subject, trial_id, eval_name, dtype=torch.float32, 
                                                                                            output_indices=False, 
                                                                                            start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*neuroprobe_config.SAMPLING_RATE), 
                                                                                            end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*neuroprobe_config.SAMPLING_RATE),
                                                                                            lite=lite, max_samples=3500, include_all_other_trials=True, binary_tasks=binary_tasks)

        for bin_start, bin_end in zip(bin_starts, bin_ends):
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

                # Convert PyTorch dataset to numpy arrays for scikit-learn - now using proper preprocess_parameters
                X_train = np.array([preprocess_data(item[0][:, data_idx_from:data_idx_to].float().numpy(), all_electrode_labels, preprocess_type, preprocess_parameters) for item in train_dataset])
                y_train = np.array([item[1] for item in train_dataset])
                X_test = np.array([preprocess_data(item[0][:, data_idx_from:data_idx_to].float().numpy(), all_electrode_labels, preprocess_type, preprocess_parameters) for item in test_dataset])
                y_test = np.array([item[1] for item in test_dataset])

                # Flatten the data after preprocessing
                original_X_train_shape = X_train.shape
                original_X_test_shape = X_test.shape
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)

                # Standardize the data
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Train logistic regression
                if classifier_type == 'linear':
                    clf = LogisticRegression(random_state=seed, max_iter=10000, tol=1e-3)
                elif classifier_type == 'cnn':
                    X_train = X_train.reshape(original_X_train_shape)
                    X_test = X_test.reshape(original_X_test_shape)
                    clf = CNNClassifier(random_state=seed)
                elif classifier_type == 'transformer':
                    X_train = X_train.reshape(original_X_train_shape)
                    X_test = X_test.reshape(original_X_test_shape)
                    clf = TransformerClassifier(random_state=seed)
                elif classifier_type == 'mlp':
                    X_train = X_train.reshape(original_X_train_shape)
                    X_test = X_test.reshape(original_X_test_shape)
                    clf = MLPClassifier(random_state=seed)
                else:
                    raise ValueError(f"Invalid classifier type: {classifier_type}")
                clf.fit(X_train, y_train)

                # Evaluate model
                train_accuracy = clf.score(X_train, y_train)
                test_accuracy = clf.score(X_test, y_test)

                # Get predictions - for multiclass classification
                train_probs = clf.predict_proba(X_train)
                test_probs = clf.predict_proba(X_test)

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
                if verbose: 
                    log(f"Electrode {electrode_label} ({electrode_idx+1}/{len(all_electrode_labels)}), Fold {fold_idx+1}, Bin {bin_start}-{bin_end}: Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}, Train ROC AUC: {train_roc:.3f}, Test ROC AUC: {test_roc:.3f}", priority=0)

            if bin_start == -bins_start_before_word_onset_seconds and bin_end == bins_end_after_word_onset_seconds and not only_1second:
                results_electrode[electrode_label]["whole_window"] = bin_results # whole window results
            elif bin_start == 0 and bin_end == 1:
                results_electrode[electrode_label]["one_second_after_onset"] = bin_results # one second after onset results
            else:
                results_electrode[electrode_label]["time_bins"].append(bin_results) # time bin results

        #if electrode_idx % 2 == 0:
        # Save after each electrode is processed
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/{filename}", "w") as f:
            json.dump(results, f, indent=4)
        if verbose:
            log(f"Results saved to {save_dir}/{filename} after processing electrode {electrode_label}", priority=0)

    # Remove final save since we're saving after each electrode
    if verbose:
        log(f"All electrodes processed for {eval_name}", priority=0)