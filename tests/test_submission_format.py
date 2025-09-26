#!/usr/bin/env python3
"""
Test suite to validate Neuroprobe leaderboard submission format.
This test ensures all submissions follow the guidelines in SUBMIT.md.
"""

import os
import json
import re
import pytest
from pathlib import Path
from typing import Dict, List, Set, Any

# Import the task mapping from neuroprobe config
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Set a dummy environment variable to avoid import errors
os.environ.setdefault('ROOT_DIR_BRAINTREEBANK', '/tmp')

from neuroprobe.config import NEUROPROBE_TASKS_MAPPING

class TestSubmissionFormat:
    """Test class to validate submission format requirements."""
    
    @pytest.fixture
    def leaderboard_dir(self):
        """Get the leaderboard directory path."""
        return Path(__file__).parent.parent / "leaderboard"
    
    @pytest.fixture
    def submission_dirs(self, leaderboard_dir):
        """Get all submission directories."""
        if not leaderboard_dir.exists():
            pytest.skip("Leaderboard directory does not exist")
        
        submission_dirs = [d for d in leaderboard_dir.iterdir() if d.is_dir()]
        if not submission_dirs:
            pytest.skip("No submissions found in leaderboard directory")
        
        return submission_dirs
    
    def test_submission_directory_naming_convention(self, submission_dirs):
        """Test that submission directories follow the naming convention: MODELNAME_AUTHORFIRSTNAME_AUTHORLASTNAME_DAY_MONTH_YEAR"""
        # Allow for more complex model names and author names that may contain underscores
        naming_pattern = re.compile(r'^.+_[^_]+_[^_]+_\d{1,2}_\d{1,2}_\d{4}$')
        
        for submission_dir in submission_dirs:
            assert naming_pattern.match(submission_dir.name), \
                f"Directory {submission_dir.name} does not follow naming convention: MODELNAME_AUTHORFIRSTNAME_AUTHORLASTNAME_DAY_MONTH_YEAR"
    
    def test_required_files_exist(self, submission_dirs):
        """Test that all required files exist in each submission directory."""
        required_files = ['metadata.json', 'PUBLICATION.bib', 'ATTESTATION.txt']
        
        for submission_dir in submission_dirs:
            for required_file in required_files:
                file_path = submission_dir / required_file
                assert file_path.exists(), \
                    f"Required file {required_file} missing in {submission_dir.name}"
    
    def test_metadata_json_structure(self, submission_dirs):
        """Test that metadata.json has all required fields with correct types."""
        required_fields = {
            'model_name': str,
            'description': str,
            'author': str,
            'organization': str,
            'organization_url': str,
            'timestamp': (int, float)
        }
        optional_fields = {
            'logo_url': str
        }
        
        for submission_dir in submission_dirs:
            metadata_path = submission_dir / 'metadata.json'
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check required fields
            for field, expected_type in required_fields.items():
                assert field in metadata, \
                    f"Required field '{field}' missing from metadata.json in {submission_dir.name}"
                assert isinstance(metadata[field], expected_type), \
                    f"Field '{field}' has wrong type in metadata.json in {submission_dir.name}. Expected {expected_type}, got {type(metadata[field])}"
            
            # Check optional fields if present
            for field, expected_type in optional_fields.items():
                if field in metadata:
                    assert isinstance(metadata[field], expected_type), \
                        f"Optional field '{field}' has wrong type in metadata.json in {submission_dir.name}. Expected {expected_type}, got {type(metadata[field])}"
    
    def test_split_directories_exist(self, submission_dirs):
        """Test that at least one valid split directory exists."""
        valid_splits = {'Within-Session', 'Cross-Session', 'Cross-Subject'}
        
        for submission_dir in submission_dirs:
            existing_splits = {d.name for d in submission_dir.iterdir() if d.is_dir() and d.name in valid_splits}
            assert existing_splits, \
                f"No valid split directories found in {submission_dir.name}. Expected at least one of: {valid_splits}"
    
    def test_population_json_files_exist(self, submission_dirs):
        """Test that all required population_TASKNAME.json files exist in each split directory."""
        expected_tasks = set(NEUROPROBE_TASKS_MAPPING.keys())
        valid_splits = {'Within-Session', 'Cross-Session', 'Cross-Subject'}
        
        for submission_dir in submission_dirs:
            split_dirs = [d for d in submission_dir.iterdir() if d.is_dir() and d.name in valid_splits]
            
            for split_dir in split_dirs:
                # Get all population JSON files
                population_files = [f for f in split_dir.iterdir() if f.name.startswith('population_') and f.name.endswith('.json')]
                
                # Extract task names from filenames
                found_tasks = {f.name.replace('population_', '').replace('.json', '') for f in population_files}
                
                # Check that all expected tasks are present
                missing_tasks = expected_tasks - found_tasks
                assert not missing_tasks, \
                    f"Missing population files for tasks {missing_tasks} in {submission_dir.name}/{split_dir.name}"
                
                # Check for unexpected tasks
                unexpected_tasks = found_tasks - expected_tasks
                assert not unexpected_tasks, \
                    f"Unexpected population files for tasks {unexpected_tasks} in {submission_dir.name}/{split_dir.name}"
    
    def test_population_json_structure(self, submission_dirs):
        """Test that population JSON files have the correct structure."""
        required_top_level_fields = {
            'model_name': str,
            'description': str,
            'author': str,
            'organization': str,
            'organization_url': str,
            'timestamp': (int, float),
            'evaluation_results': dict
        }
        
        valid_splits = {'Within-Session', 'Cross-Session', 'Cross-Subject'}
        
        for submission_dir in submission_dirs:
            split_dirs = [d for d in submission_dir.iterdir() if d.is_dir() and d.name in valid_splits]
            
            for split_dir in split_dirs:
                population_files = [f for f in split_dir.iterdir() if f.name.startswith('population_') and f.name.endswith('.json')]
                
                for pop_file in population_files:
                    with open(pop_file, 'r') as f:
                        data = json.load(f)
                    
                    # Check top-level fields
                    for field, expected_type in required_top_level_fields.items():
                        assert field in data, \
                            f"Required field '{field}' missing from {pop_file.name} in {submission_dir.name}/{split_dir.name}"
                        assert isinstance(data[field], expected_type), \
                            f"Field '{field}' has wrong type in {pop_file.name} in {submission_dir.name}/{split_dir.name}. Expected {expected_type}, got {type(data[field])}"
                    
                    # Check evaluation_results structure
                    eval_results = data['evaluation_results']
                    assert isinstance(eval_results, dict) and eval_results, \
                        f"evaluation_results must be a non-empty dict in {pop_file.name} in {submission_dir.name}/{split_dir.name}"
                    
                    # Check each session result
                    for session_id, session_data in eval_results.items():
                        self._validate_session_data(session_data, session_id, pop_file, submission_dir, split_dir)
    
    def _validate_session_data(self, session_data: Dict[str, Any], session_id: str, pop_file: Path, submission_dir: Path, split_dir: Path):
        """Validate the structure of session data within evaluation_results."""
        assert isinstance(session_data, dict), \
            f"Session data for {session_id} must be a dict in {pop_file.name} in {submission_dir.name}/{split_dir.name}"
        
        assert 'population' in session_data, \
            f"'population' key missing for {session_id} in {pop_file.name} in {submission_dir.name}/{split_dir.name}"
        
        population_data = session_data['population']
        assert isinstance(population_data, dict), \
            f"'population' must be a dict for {session_id} in {pop_file.name} in {submission_dir.name}/{split_dir.name}"
        
        # Check for time bin data (typically 'one_second_after_onset')
        for time_bin_name, time_bin_data in population_data.items():
            assert isinstance(time_bin_data, dict), \
                f"Time bin data '{time_bin_name}' must be a dict for {session_id} in {pop_file.name} in {submission_dir.name}/{split_dir.name}"
            
            required_time_bin_fields = {
                'time_bin_start': (int, float),
                'time_bin_end': (int, float),
                'folds': list
            }
            
            for field, expected_type in required_time_bin_fields.items():
                assert field in time_bin_data, \
                    f"Required field '{field}' missing from time bin '{time_bin_name}' for {session_id} in {pop_file.name} in {submission_dir.name}/{split_dir.name}"
                assert isinstance(time_bin_data[field], expected_type), \
                    f"Field '{field}' has wrong type in time bin '{time_bin_name}' for {session_id} in {pop_file.name} in {submission_dir.name}/{split_dir.name}. Expected {expected_type}, got {type(time_bin_data[field])}"
            
            # Validate folds structure
            folds = time_bin_data['folds']
            assert len(folds) > 0, \
                f"'folds' must contain at least one fold for {session_id} in {pop_file.name} in {submission_dir.name}/{split_dir.name}"
            
            for i, fold in enumerate(folds):
                self._validate_fold_data(fold, i, session_id, time_bin_name, pop_file, submission_dir, split_dir)
    
    def _validate_fold_data(self, fold: Dict[str, Any], fold_idx: int, session_id: str, time_bin_name: str, pop_file: Path, submission_dir: Path, split_dir: Path):
        """Validate the structure of fold data."""
        required_fold_fields = {
            # 'train_accuracy': (int, float),
            # 'train_roc_auc': (int, float),
            # 'test_accuracy': (int, float),
            'test_roc_auc': (int, float)
        }
        
        assert isinstance(fold, dict), \
            f"Fold {fold_idx} must be a dict for {session_id} in {pop_file.name} in {submission_dir.name}/{split_dir.name}"
        
        for field, expected_type in required_fold_fields.items():
            assert field in fold, \
                f"Required field '{field}' missing from fold {fold_idx} for {session_id} in {pop_file.name} in {submission_dir.name}/{split_dir.name}"
            assert isinstance(fold[field], expected_type), \
                f"Field '{field}' has wrong type in fold {fold_idx} for {session_id} in {pop_file.name} in {submission_dir.name}/{split_dir.name}. Expected {expected_type}, got {type(fold[field])}"
            
            # Validate metric ranges (0-1 for accuracy and ROC AUC)
            value = fold[field]
            assert 0.0 <= value <= 1.0, \
                f"Field '{field}' value {value} out of range [0,1] in fold {fold_idx} for {session_id} in {pop_file.name} in {submission_dir.name}/{split_dir.name}"
    
    def test_attestation_file_format(self, submission_dirs):
        """Test that ATTESTATION.txt contains the required attestations."""
        required_phrases = [
            "I attest that the training and test splits of Neuroprobe were respected",
            "I attest that the submitted model was not pretrained on any data that intersects with any data of Neuroprobe",
            "SIGN"
        ]
        
        for submission_dir in submission_dirs:
            attestation_path = submission_dir / 'ATTESTATION.txt'
            
            with open(attestation_path, 'r') as f:
                content = f.read().strip()
            
            # Allow "N/A" as a placeholder for now
            if content == "N/A":
                continue
                
            for phrase in required_phrases:
                assert phrase in content, \
                    f"Required phrase '{phrase}' not found in ATTESTATION.txt in {submission_dir.name}"
            
            # Check that there are at least 2 SIGN statements
            sign_count = content.count('SIGN')
            assert sign_count >= 2, \
                f"ATTESTATION.txt must contain at least 2 SIGN statements, found {sign_count} in {submission_dir.name}"
    
    def test_publication_bib_format(self, submission_dirs):
        """Test that PUBLICATION.bib is not empty and appears to be a valid bibtex file."""
        for submission_dir in submission_dirs:
            bib_path = submission_dir / 'PUBLICATION.bib'
            
            with open(bib_path, 'r') as f:
                content = f.read().strip()
            
            assert content, \
                f"PUBLICATION.bib cannot be empty in {submission_dir.name}"
            
            # Allow "N/A" as a placeholder, otherwise check bibtex format
            if content != "N/A":
                # Basic bibtex format check - should contain @ and braces
                assert '@' in content and '{' in content and '}' in content, \
                    f"PUBLICATION.bib does not appear to be valid bibtex format in {submission_dir.name}"
    
    # Allow for separate AUTHOR in metadata.json
    # def test_metadata_consistency(self, submission_dirs):
    #     """Test that metadata.json fields are consistent across all population JSON files."""
    #     consistent_fields = ['model_name', 'description', 'author', 'organization', 'organization_url']
    #     valid_splits = {'Within-Session', 'Cross-Session', 'Cross-Subject'}
        
    #     for submission_dir in submission_dirs:
    #         # Read metadata.json
    #         with open(submission_dir / 'metadata.json', 'r') as f:
    #             metadata = json.load(f)
            
    #         split_dirs = [d for d in submission_dir.iterdir() if d.is_dir() and d.name in valid_splits]
            
    #         for split_dir in split_dirs:
    #             population_files = [f for f in split_dir.iterdir() if f.name.startswith('population_') and f.name.endswith('.json')]
                
    #             for pop_file in population_files:
    #                 with open(pop_file, 'r') as f:
    #                     pop_data = json.load(f)
                    
    #                 # Check consistency of fields
    #                 for field in consistent_fields:
    #                     assert metadata[field] == pop_data[field], \
    #                         f"Field '{field}' inconsistent between metadata.json and {pop_file.name} in {submission_dir.name}/{split_dir.name}"


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=long", "--no-header"]) 