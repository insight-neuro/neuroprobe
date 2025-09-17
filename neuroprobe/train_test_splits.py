import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold
import numpy as np

from .braintreebank_subject import BrainTreebankSubject
from .datasets import BrainTreebankSubjectTrialBenchmarkDataset
from .config import *


def generate_splits_DS_DM(all_subjects, test_subject_id, test_trial_id, eval_name, dtype=torch.float32,
                          lite=True, nano=False,
                          
                          # Dataset parameters
                          output_indices=False, 
                          start_neural_data_before_word_onset=int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE), 
                          end_neural_data_after_word_onset=int(END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE)):
    """Generate train/test splits for Different Subject Different Movie (DS-DM) evaluation.
    
    This function creates train/test splits by using one subject and movie as the test set,
    and using all other subjects and movies (except the test movie) as the training set.
    This evaluates generalization across both subjects and movie content.

    Args:
        all_subjects (dict): Dictionary mapping subject IDs to Subject objects
        test_subject_id (int): ID of the subject to use as test set
        test_trial_id (int): ID of the trial/movie to use as test set
        eval_name (str): Name of the evaluation metric to use (e.g. "rms")
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
        lite (bool): if True, the eval is Neuroprobe-Lite (the default), otherwise it is Neuroprobe-Full.
        nano (bool): if True, the eval is Neuroprobe-Nano (the default), otherwise it is Neuroprobe-Lite (if lite is True)

        # Dataset parameters
        output_indices (bool, optional): Whether to output the indices of the neural data. Defaults to False.
        start_neural_data_before_word_onset (int, optional): Number of seconds before the word onset to start the neural data. Defaults to START_NEURAL_DATA_BEFORE_WORD_ONSET.
        end_neural_data_after_word_onset (int, optional): Number of seconds after the word onset to end the neural data. Defaults to END_NEURAL_DATA_AFTER_WORD_ONSET.

    Returns:
        tuple: A tuple containing:
            - train_datasets (list): List of training datasets
            - test_dataset (Dataset): Dataset for the test subject and trial
    """
    assert test_subject_id != DS_DM_TRAIN_SUBJECT_ID, "Test subject cannot be the same as the training subject."

    test_dataset = BrainTreebankSubjectTrialBenchmarkDataset(all_subjects[test_subject_id], test_trial_id, dtype=dtype, eval_name=eval_name, 
                                                             output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                             lite=lite, nano=nano)
    
    train_subject_id, train_trial_id = DS_DM_TRAIN_SUBJECT_ID, DS_DM_TRAIN_TRIAL_ID
    train_dataset = BrainTreebankSubjectTrialBenchmarkDataset(all_subjects[train_subject_id], train_trial_id, dtype=dtype, eval_name=eval_name, 
                                                                output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                                lite=lite, nano=nano)

    return train_dataset, test_dataset
    

def generate_splits_SS_DM(test_subject, test_trial_id, eval_name, dtype=torch.float32,
                          lite=True,
                          
                          # Dataset parameters
                          output_indices=False, 
                          start_neural_data_before_word_onset=int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE), 
                          end_neural_data_after_word_onset=int(END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE)):
    """Generate train/test splits for Single Subject Different Movies (SS-DM) evaluation.
    
    This function creates train/test splits by using one movie as the test set and all other
    movies from the same subject as the training set (trimmed at max_other_trials movies). 
    Unlike SS-SM, this does not perform k-fold cross validation since movies are already naturally separated.

    NOTE: Neuroprobe-Nano does not support SS-DM because it only contains one movie per subject.

    Args:
        test_subject (Subject): Subject object containing brain recording data
        test_trial_id (int): ID of the trial/movie to use as test set
        eval_name (str): Name of the evaluation metric to use (e.g. "rms")
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
        lite (bool): if True, the eval is Neuroprobe-Lite (the default), otherwise it is Neuroprobe-Full.

        # Dataset parameters
        output_indices (bool, optional): Whether to output the indices of the neural data. Defaults to False.
        start_neural_data_before_word_onset (int, optional): Number of seconds before the word onset to start the neural data. Defaults to START_NEURAL_DATA_BEFORE_WORD_ONSET.
        end_neural_data_after_word_onset (int, optional): Number of seconds after the word onset to end the neural data. Defaults to END_NEURAL_DATA_AFTER_WORD_ONSET.

    Returns:
        tuple: A tuple containing:
            - train_datasets (list): List of training datasets
            - test_dataset (Dataset): Dataset for the test trial
    """
    assert len(NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT[test_subject.subject_id]) > 1, f"Training subject must have at least two trials. But subject {test_subject.subject_id} has only {len(NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT[test_subject.subject_id])} trials."
    
    test_dataset = BrainTreebankSubjectTrialBenchmarkDataset(test_subject, test_trial_id, dtype=dtype, eval_name=eval_name, 
                                                             output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                             lite=lite)
        
    if not lite:
        train_trial_id = NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT[test_subject.subject_id][0]
        if train_trial_id == test_trial_id:
            train_trial_id = NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT[test_subject.subject_id][1] # If the longest trial is the test trial, use the second longest trial for training
    else:
        train_trial_id = [trial_id for subject_id, trial_id in NEUROPROBE_LITE_SUBJECT_TRIALS if subject_id == test_subject.subject_id and trial_id != test_trial_id][0] # Get the first other trial for the training set (there should only be one)
    

    train_dataset = BrainTreebankSubjectTrialBenchmarkDataset(test_subject, train_trial_id, dtype=dtype, eval_name=eval_name, 
                                                                output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                                lite=lite)
    return train_dataset, test_dataset


def generate_splits_SS_SM(test_subject, test_trial_id, eval_name, dtype=torch.float32,
                          lite=True, nano=False,
                          
                          # Dataset parameters
                          output_indices=False, 
                          start_neural_data_before_word_onset=int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE), 
                          end_neural_data_after_word_onset=int(END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE)):
    """Generate train/test splits for Single Subject Single Movie (SS-SM) evaluation.
    
    This function performs k-fold cross validation on data from a single subject and movie.
    If gap_length is specified and not None, it ensures temporal gaps between train and test sets to avoid
    temporal correlation in the data. For example, if gap_length=300, there will be at least
    300 seconds between any training and test samples. If gap_length is None, no temporal gap
    is enforced between train and test sets.

    Args:
        test_subject (Subject): Subject object containing brain recording data
        test_trial_id (int): ID of the trial/movie to use
        eval_name (str): Name of the evaluation metric to use (e.g. "rms", "word_gap", "pitch", "delta_volume")
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
        lite (bool): if True, the eval is Neuroprobe-Lite (the default), otherwise it is Neuroprobe-Full.
        nano (bool): if True, the eval is Neuroprobe-Nano (the default), otherwise it is Neuroprobe-Lite (if lite is True)

        # Dataset parameters
        output_indices (bool, optional): Whether to output the indices of the neural data. Defaults to False.
        start_neural_data_before_word_onset (int, optional): Number of seconds before the word onset to start the neural data. Defaults to START_NEURAL_DATA_BEFORE_WORD_ONSET.
        end_neural_data_after_word_onset (int, optional): Number of seconds after the word onset to end the neural data. Defaults to END_NEURAL_DATA_AFTER_WORD_ONSET.

    Returns:
        tuple: A tuple containing:
            - train_datasets (list): List of k training dataset splits
            - test_datasets (list): List of k test dataset splits, which correspond to the train datasets in the array above
    """

    train_datasets = []
    test_datasets = []

    dataset = BrainTreebankSubjectTrialBenchmarkDataset(test_subject, test_trial_id, dtype=dtype, eval_name=eval_name, 
                                                        output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                        lite=lite, nano=nano)
    
    k_folds = NEUROPROBE_LITE_N_FOLDS if not nano else NEUROPROBE_NANO_N_FOLDS
    kf = KFold(n_splits=k_folds, shuffle=False)  # shuffle=False is important to avoid correlated train/test splits!
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        # Skip empty splits
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        
        train_dataset = Subset(dataset, train_idx)
        train_datasets.append(train_dataset)
        test_datasets.append(Subset(dataset, test_idx))

    return train_datasets, test_datasets