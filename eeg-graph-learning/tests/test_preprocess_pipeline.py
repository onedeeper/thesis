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
    line_noise = np.array([50, 100, 150])
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