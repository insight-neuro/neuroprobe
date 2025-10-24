"""
Neuroprobe: A benchmark for evaluating intracranial brain responses to naturalistic stimuli.

This package provides tools for analyzing neural data from the BrainTreebank dataset,
including dataset loading, preprocessing, and evaluation utilities.
"""

__version__ = "0.1.6"
__author__ = "Andrii Zahorodnii, Christopher Wang, Bennett Stankovits, Charikleia Moraitaki, Geeling Chau, Andrei Barbu, Boris Katz, Ila R Fiete"
__email__ = "zaho@csail.mit.edu"

# Import main classes and functions
from .braintreebank_subject import BrainTreebankSubject
from .datasets import BrainTreebankSubjectTrialBenchmarkDataset
from . import config
from . import train_test_splits
from .train_test_splits import generate_splits_cross_session, generate_splits_cross_subject, generate_splits_within_session, generate_splits_CrossSession, generate_splits_CrossSubject, generate_splits_WithinSession

# Make key classes available at package level
__all__ = [
    "BrainTreebankSubject",
    "BrainTreebankSubjectTrialBenchmarkDataset", 
    "config",
    "train_test_splits",
    "generate_splits_cross_session",
    "generate_splits_cross_subject",
    "generate_splits_within_session",
    "generate_splits_CrossSession",
    "generate_splits_CrossSubject",
    "generate_splits_WithinSession",
] 
