import os
import numpy as np
import pickle
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from matplotlib.figure import Figure
import pandas as pd
from pathlib import Path

# Import the function to test
from eeglearn.preprocess.clean import process_file, clean_pipeline

class TestProcessFile:
    """Test class for clean.py functions."""
    # Define class variables instead of using __init__
    default_filepath = "tests/test_data/test_eeg.csv"
    ID = "sub-19694366"
    sessID = "ses-1"
    cond = "EO"
    epochs_length = 2.0
    line_noise = [50, 100, 150]
    sfreq = 500
        
    @pytest.fixture
    def setup_test_environment(self):
        """Create a temporary test environment"""
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        preprocessed_dir = os.path.join(temp_dir, 'cleaned')
        os.makedirs(preprocessed_dir, exist_ok=True)
        
        # Use environment variable if available, otherwise use default
        filepath = os.environ.get('EEG_TEST_FILE_PATH', self.default_filepath)
        # skip if envrionment variable is not found
        if filepath == self.default_filepath:
            pytest.skip("No environment variable found. Skipping test.")
        
        yield filepath, preprocessed_dir, temp_dir  # Return values needed for test
        
        # Cleanup after test
        shutil.rmtree(temp_dir)
        
        # Only remove the test file if we created it
        if filepath == self.default_filepath and os.path.exists(self.default_filepath):
            os.remove(self.default_filepath)

    def test_process_file_basic(self, setup_test_environment):
        """Test basic functionality of process_file"""
        # Set plots to False for simpler testing
        plots = False
        filepath, preprocessed_dir, _ = setup_test_environment
        
        # Create arguments tuple
        args = (filepath, self.ID, self.sessID, self.cond, 
                self.epochs_length, self.line_noise, self.sfreq, 
                plots, preprocessed_dir)
        
        # Call function
        result = process_file(args)
        
        # Check output directory was created
        expected_dir = f'{preprocessed_dir}/{self.ID}/{self.sessID}/eeg'
        assert os.path.exists(expected_dir)
        
        # Check output file was created
        expected_file = f'{expected_dir}/{self.ID}_{self.sessID}_{self.cond}_preprocessed.npy'
        assert os.path.exists(expected_file)
        
        # Verify return value
        assert result == (self.ID, self.sessID, self.cond)

    @pytest.fixture
    def setup_clean_pipeline_test(self):
        """Create a temporary test environment for clean_pipeline tests."""
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        preprocessed_dir = os.path.join(temp_dir, 'cleaned')
        derivatives_dir = os.path.join(temp_dir, 'derivatives')
        os.makedirs(preprocessed_dir, exist_ok=True)
        os.makedirs(derivatives_dir, exist_ok=True)
        
        yield derivatives_dir, preprocessed_dir, temp_dir
        
        # Cleanup after test
        shutil.rmtree(temp_dir)

    @patch('eeglearn.preprocess.clean.Pool')
    def test_participant_processing_with_different_conditions(self, mock_pool, setup_clean_pipeline_test):
        """Test that the correct number of participants are processed regardless of conditions selected."""
        # Setup
        derivatives_dir, preprocessed_dir, _ = setup_clean_pipeline_test
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        
        # Create test data structure with 5 participants, with both EO and EC files
        test_participants = [f"sub-{i}" for i in range(1, 6)]  # 5 participants
        conditions = ["EO", "EC"]
        
        # Create a structure that os.walk would return
        mock_walk_data = [
            (derivatives_dir, [], [f"{sub}_ses-1_{cond}_eeg.csv" 
                                 for sub in test_participants 
                                 for cond in conditions])
        ]
        
        # Expected files to process for different condition selections
        expected_eo_only_files = 5  # 5 participants with EO only
        expected_ec_only_files = 5  # 5 participants with EC only
        expected_both_cond_files = 10  # 5 participants with both EO and EC
        
        # Define mock imap results - just return the input args
        mock_pool_instance.imap.side_effect = lambda func, args: args
        
        # Test with EO condition only
        with patch('os.walk', return_value=mock_walk_data):
            with patch('os.listdir', return_value=test_participants):
                # Process only EO condition
                clean_pipeline(
                    derivates_dir=derivatives_dir,
                    preprocessed_dir=preprocessed_dir,
                    sfreq=self.sfreq,
                    epochs_length=self.epochs_length,
                    line_noise=self.line_noise,
                    n_processes=1,
                    conditions=['EO'],
                    sessions=['ses-1'],
                    plots=False
                )
                
                # Check that the correct number of files was processed
                # Get the second call to imap (after the first test)
                args_list = mock_pool_instance.imap.call_args[0][1]
                assert len(args_list) == expected_eo_only_files, \
                    f"Expected {expected_eo_only_files} files to process for EO condition, got {len(args_list)}"
                
                # Verify all participants are included
                processed_ids = {args[1] for args in args_list}
                assert processed_ids == set(test_participants), \
                    f"Not all participants were processed: {processed_ids} vs {set(test_participants)}"
        
        # Reset mock
        mock_pool_instance.reset_mock()
        
        # Test with EC condition only
        with patch('os.walk', return_value=mock_walk_data):
            with patch('os.listdir', return_value=test_participants):
                # Process only EC condition
                clean_pipeline(
                    derivates_dir=derivatives_dir,
                    preprocessed_dir=preprocessed_dir,
                    sfreq=self.sfreq,
                    epochs_length=self.epochs_length,
                    line_noise=self.line_noise,
                    n_processes=1,
                    conditions=['EC'],
                    sessions=['ses-1'],
                    plots=False
                )
                
                # Check that the correct number of files was processed
                args_list = mock_pool_instance.imap.call_args[0][1]
                assert len(args_list) == expected_ec_only_files, \
                    f"Expected {expected_ec_only_files} files to process for EC condition, got {len(args_list)}"
                
                # Verify all participants are included
                processed_ids = {args[1] for args in args_list}
                assert processed_ids == set(test_participants), \
                    f"Not all participants were processed: {processed_ids} vs {set(test_participants)}"
        
        # Reset mock
        mock_pool_instance.reset_mock()
        
        # Test with both conditions
        with patch('os.walk', return_value=mock_walk_data):
            with patch('os.listdir', return_value=test_participants):
                # Process both conditions
                clean_pipeline(
                    derivates_dir=derivatives_dir,
                    preprocessed_dir=preprocessed_dir,
                    sfreq=self.sfreq,
                    epochs_length=self.epochs_length,
                    line_noise=self.line_noise,
                    n_processes=1,
                    conditions=['EO', 'EC'],
                    sessions=['ses-1'],
                    plots=False
                )
                
                # Check that the correct number of files was processed
                args_list = mock_pool_instance.imap.call_args[0][1]
                assert len(args_list) == expected_both_cond_files, \
                    f"Expected {expected_both_cond_files} files to process for both conditions, got {len(args_list)}"
                
                # Verify all participants are included (each participant appears twice, once for each condition)
                processed_ids = {args[1] for args in args_list}
                assert processed_ids == set(test_participants), \
                    f"Not all participants were processed: {processed_ids} vs {set(test_participants)}"
