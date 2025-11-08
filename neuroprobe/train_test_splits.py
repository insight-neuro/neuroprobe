import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold
import numpy as np

from .datasets import BrainTreebankSubjectTrialBenchmarkDataset
from .config import *


def generate_splits_cross_subject(all_subjects, test_subject_id, test_trial_id, eval_name, dtype=torch.float32,
                          lite=True, nano=False,
                          
                          # Dataset parameters
                          binary_tasks=True,
                          output_indices=False, 
                          start_neural_data_before_word_onset=int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE), 
                          end_neural_data_after_word_onset=int(END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE),
                          max_samples=None):
    """Generate train/test splits for Cross-Subject evaluation.
    
    This function creates train/test splits by using one subject and movie as the test set,
    and using all other subjects and movies (except the test movie) as the training set.
    This evaluates generalization across both subjects and movie content (i.e. the same subject but different movies).

    Args:
        all_subjects (dict): Dictionary mapping subject IDs to Subject objects
        test_subject_id (int): ID of the subject to use as test set
        test_trial_id (int): ID of the trial/movie to use as test set
        eval_name (str): Name of the evaluation metric to use (e.g. "rms")
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
        lite (bool): if True, the eval is Neuroprobe (the default), otherwise it is Neuroprobe-Full.
        nano (bool): if True, the eval is Neuroprobe-Nano (the default), otherwise it is Neuroprobe (if lite is True)

        # Dataset parameters
        output_indices (bool, optional): Whether to output the indices of the neural data. Defaults to False.
        start_neural_data_before_word_onset (int, optional): Number of seconds before the word onset to start the neural data. Defaults to START_NEURAL_DATA_BEFORE_WORD_ONSET.
        end_neural_data_after_word_onset (int, optional): Number of seconds after the word onset to end the neural data. Defaults to END_NEURAL_DATA_AFTER_WORD_ONSET.
        max_samples (int, optional): the maximum number of samples to include in the dataset (defaults to None, which means default limits: none for Neuroprobe-Full, 3500 for Neuroprobe-Lite, 1000 for Neuroprobe-Nano)
        
    Returns:
        list: A list of dictionaries, each containing:
            - train_dataset (BrainTreebankSubjectTrialBenchmarkDataset): Training dataset
            - val_dataset (BrainTreebankSubjectTrialBenchmarkDataset): Validation dataset
            - test_dataset (BrainTreebankSubjectTrialBenchmarkDataset): Test dataset
    """
    assert test_subject_id != DS_DM_TRAIN_SUBJECT_ID, "Test subject cannot be the same as the training subject."

    test_dataset = BrainTreebankSubjectTrialBenchmarkDataset(all_subjects[test_subject_id], test_trial_id, dtype=dtype, eval_name=eval_name, 
                                                             binary_tasks=binary_tasks, output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                             lite=lite, nano=nano, max_samples=max_samples)
    
    train_subject_id, train_trial_id = DS_DM_TRAIN_SUBJECT_ID, DS_DM_TRAIN_TRIAL_ID
    train_dataset = BrainTreebankSubjectTrialBenchmarkDataset(all_subjects[train_subject_id], train_trial_id, dtype=dtype, eval_name=eval_name, 
                                                                binary_tasks=binary_tasks, output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                                lite=lite, nano=nano, max_samples=max_samples)

    test_size = len(test_dataset)
    val_size = test_size // 2
    val_indices = list(range(val_size))
    test_indices = list(range(val_size, test_size))
    val_dataset = Subset(test_dataset, val_indices)
    test_dataset = Subset(test_dataset, test_indices)
    # copy the electrode coordinates and labels from the test dataset to the val and test datasets
    val_dataset.electrode_coordinates = test_dataset.dataset.electrode_coordinates
    val_dataset.electrode_labels = test_dataset.dataset.electrode_labels
    test_dataset.electrode_coordinates = test_dataset.dataset.electrode_coordinates
    test_dataset.electrode_labels = test_dataset.dataset.electrode_labels

    return [
        {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset
        }
    ]
    

def generate_splits_cross_session(test_subject, test_trial_id, eval_name, dtype=torch.float32,
                          lite=True,
                          
                          # Dataset parameters
                          binary_tasks=True,
                          output_indices=False, 
                          start_neural_data_before_word_onset=int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE), 
                          end_neural_data_after_word_onset=int(END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE),
                          max_samples=None, include_all_other_trials=False):
    """Generate train/test splits for Cross-Session evaluation.
    
    This function creates train/test splits by using one movie as the test set and all other
    movies from the same subject as the training set (trimmed at max_other_trials movies). 
    Unlike Within-Session, this does not perform k-fold cross validation since movies are already naturally separated.

    Args:
        test_subject (Subject): Subject object containing brain recording data
        test_trial_id (int): ID of the trial/movie to use as test set
        eval_name (str): Name of the evaluation metric to use (e.g. "rms")
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
        lite (bool): if True, the eval is Neuroprobe (the default), otherwise it is Neuroprobe-Full.

        # Dataset parameters
        output_indices (bool, optional): Whether to output the indices of the neural data. Defaults to False.
        start_neural_data_before_word_onset (int, optional): Number of seconds before the word onset to start the neural data. Defaults to START_NEURAL_DATA_BEFORE_WORD_ONSET.
        end_neural_data_after_word_onset (int, optional): Number of seconds after the word onset to end the neural data. Defaults to END_NEURAL_DATA_AFTER_WORD_ONSET.
        max_samples (int, optional): the maximum number of samples to include in the dataset (defaults to None, which means default limits: none for Neuroprobe-Full, 3500 for Neuroprobe-Lite, 1000 for Neuroprobe-Nano)
        include_all_other_trials (bool, optional): if True, include all other trials for training (defaults to False). If False, only include the longest trial for training (NOTE: for Neuroprobe, there is no choice).
    Returns:
        list: A list of dictionaries, each containing:
            - train_dataset (BrainTreebankSubjectTrialBenchmarkDataset): Training dataset
            - val_dataset (BrainTreebankSubjectTrialBenchmarkDataset): Validation dataset
            - test_dataset (BrainTreebankSubjectTrialBenchmarkDataset): Test dataset
    """
    assert len(NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT[test_subject.subject_id]) > 1, f"Training subject must have at least two trials. But subject {test_subject.subject_id} has only {len(NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT[test_subject.subject_id])} trials."
    
    test_dataset = BrainTreebankSubjectTrialBenchmarkDataset(test_subject, test_trial_id, dtype=dtype, eval_name=eval_name, 
                                                             binary_tasks=binary_tasks, output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                             lite=lite, max_samples=max_samples)
        
    if include_all_other_trials:
        train_trial_ids = [trial_id for trial_id in NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT[test_subject.subject_id] if trial_id != test_trial_id]

        if max_samples is not None:
            max_samples = max_samples // len(train_trial_ids)

        train_datasets = [BrainTreebankSubjectTrialBenchmarkDataset(test_subject, train_trial_id, dtype=dtype, eval_name=eval_name, 
                                                                    binary_tasks=binary_tasks, output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                                    lite=lite, max_samples=max_samples) for train_trial_id in train_trial_ids]
        train_dataset = ConcatDataset(train_datasets)
    else:
        if not lite:
            train_trial_id = NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT[test_subject.subject_id][0]
            if train_trial_id == test_trial_id:
                train_trial_id = NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT[test_subject.subject_id][1] # If the longest trial is the test trial, use the second longest trial for training
        else:
            train_trial_id = [trial_id for subject_id, trial_id in NEUROPROBE_LITE_SUBJECT_TRIALS if subject_id == test_subject.subject_id and trial_id != test_trial_id][0] # Get the first other trial for the training set (there should only be one)
        

        train_dataset = BrainTreebankSubjectTrialBenchmarkDataset(test_subject, train_trial_id, dtype=dtype, eval_name=eval_name, 
                                                                    binary_tasks=binary_tasks, output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                                    lite=lite, max_samples=max_samples)

    test_size = len(test_dataset)
    val_size = test_size // 2
    val_indices = list(range(val_size))
    test_indices = list(range(val_size, test_size))
    val_dataset = Subset(test_dataset, val_indices)
    test_dataset = Subset(test_dataset, test_indices)
    # copy the electrode coordinates and labels from the test dataset to the val and test datasets
    val_dataset.electrode_coordinates = test_dataset.dataset.electrode_coordinates
    val_dataset.electrode_labels = test_dataset.dataset.electrode_labels
    test_dataset.electrode_coordinates = test_dataset.dataset.electrode_coordinates
    test_dataset.electrode_labels = test_dataset.dataset.electrode_labels

    return [
        {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset
        }
    ]


def generate_splits_within_session(test_subject, test_trial_id, eval_name, dtype=torch.float32,
                          lite=True, nano=False,
                          
                          # Dataset parameters
                          binary_tasks=True,
                          output_indices=False, 
                          start_neural_data_before_word_onset=int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE), 
                          end_neural_data_after_word_onset=int(END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE),
                          max_samples=None):
    """Generate train/test splits for Within Session evaluation.

    This function performs k-fold cross validation on data from a single subject and movie.
    

    Args:
        test_subject (Subject): Subject object containing brain recording data
        test_trial_id (int): ID of the trial/movie to use
        eval_name (str): Name of the evaluation metric to use (e.g. "rms", "word_gap", "pitch", "delta_volume")
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
        lite (bool): if True, the eval is Neuroprobe (the default), otherwise it is Neuroprobe-Full.
        nano (bool): if True, the eval is Neuroprobe-Nano (the default), otherwise it is Neuroprobe (if lite is True)

        # Dataset parameters
        output_indices (bool, optional): Whether to output the indices of the neural data. Defaults to False.
        start_neural_data_before_word_onset (int, optional): Number of seconds before the word onset to start the neural data. Defaults to START_NEURAL_DATA_BEFORE_WORD_ONSET.
        end_neural_data_after_word_onset (int, optional): Number of seconds after the word onset to end the neural data. Defaults to END_NEURAL_DATA_AFTER_WORD_ONSET.
        max_samples (int, optional): the maximum number of samples to include in the dataset (defaults to None, which means default limits: none for Neuroprobe-Full, 3500 for Neuroprobe-Lite, 1000 for Neuroprobe-Nano)

    Returns:
        list: A list of dictionaries, each containing:
            - train_dataset (BrainTreebankSubjectTrialBenchmarkDataset): Training dataset
            - val_dataset (BrainTreebankSubjectTrialBenchmarkDataset): Validation dataset
            - test_dataset (BrainTreebankSubjectTrialBenchmarkDataset): Test dataset
    """

    train_datasets = []
    test_datasets = []
    val_datasets = []

    dataset = BrainTreebankSubjectTrialBenchmarkDataset(test_subject, test_trial_id, dtype=dtype, eval_name=eval_name, 
                                                        binary_tasks=binary_tasks, output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                        lite=lite, nano=nano, max_samples=max_samples)
    
    k_folds = NEUROPROBE_LITE_N_FOLDS if not nano else NEUROPROBE_NANO_N_FOLDS
    kf = KFold(n_splits=k_folds, shuffle=False)  # shuffle=False is important to avoid correlated train/test splits!
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        # Skip empty splits
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        
        train_dataset = Subset(dataset, train_idx)
        train_datasets.append(train_dataset)
        test_dataset = Subset(dataset, test_idx)
        
        # Further split the test dataset into val and test datasets
        test_size = len(test_dataset)
        val_size = test_size // 2
        val_indices = list(range(val_size))
        test_indices = list(range(val_size, test_size))
        val_dataset = Subset(test_dataset, val_indices)
        test_dataset = Subset(test_dataset, test_indices)
        # copy the electrode coordinates and labels from the test dataset to the val and test datasets
        val_dataset.electrode_coordinates = dataset.electrode_coordinates
        val_dataset.electrode_labels = dataset.electrode_labels
        test_dataset.electrode_coordinates = dataset.electrode_coordinates
        test_dataset.electrode_labels = dataset.electrode_labels
        test_datasets.append(test_dataset)
        val_datasets.append(val_dataset)

    return [
        {
            "train_dataset": train_dataset, 
            "val_dataset": val_dataset, 
            "test_dataset": test_dataset
        } for train_dataset, val_dataset, test_dataset in zip(train_datasets, val_datasets, test_datasets)
    ]

# For backwards compatibility
generate_splits_DS_DM = generate_splits_cross_subject
generate_splits_SS_DM = generate_splits_cross_session
generate_splits_SS_SM = generate_splits_within_session

# For flexibility in function naming convention
generate_splits_CrossSubject = generate_splits_cross_subject
generate_splits_CrossSession = generate_splits_cross_session
generate_splits_WithinSession = generate_splits_within_session
