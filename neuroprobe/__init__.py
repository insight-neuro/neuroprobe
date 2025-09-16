"""
Neuroprobe: A benchmark for evaluating intracranial brain responses to naturalistic stimuli.

This package provides tools for analyzing neural data from the BrainTreebank dataset,
including dataset loading, preprocessing, and evaluation utilities.
"""

__version__ = "1.0"
__author__ = "Andrii Zahorodnii, Bennett Stankovits, Christopher Wang, Charikleia Moraitaki, Geeling Chau, Ila R Fiete, Boris Katz, Andrei Barbu"
__email__ = "zaho@csail.mit.edu"

# Import main classes and functions
from .braintreebank_subject import BrainTreebankSubject
from .datasets import BrainTreebankSubjectTrialBenchmarkDataset
from . import config
from . import train_test_splits

# Make key classes available at package level
__all__ = [
    "BrainTreebankSubject",
    "BrainTreebankSubjectTrialBenchmarkDataset", 
    "config",
    "train_test_splits"
] 