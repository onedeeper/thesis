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
    """Test the process_file function"""
    def __init__(self):
        self.default_filepath = "tests/test_data/test_eeg.csv"
        self.filepath = os.environ.get('EEG_TEST_FILE_PATH', self.default_filepath)
        self.ID = "sub-19694366"
        self.sessID = "ses-1"
        self.cond = "EO"
        self.ID = "sub-19694366"
        self.sessID = "ses-1"
        self.cond = "EO"
        self.epochs_length = 2.0
        self.line_noise = [50, 100, 150]
        self.sfreq = 500

        
    @pytest.fixture
    def setup_test_environment(self):
        """Create a temporary test environment"""
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp()
        self.preprocessed_dir = os.path.join(self.temp_dir, 'cleaned')
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        
        # Use environment variable if available, otherwise use default
        self.filepath = os.environ.get('EEG_TEST_FILE_PATH', self.default_filepath)
        # skip if envrionment variable is not found
        if self.filepath == self.default_filepath:
            pytest.skip("No environment variable found. Skipping test.")
        
        
        yield  # This allows the test to run
        
        # Cleanup after test
        shutil.rmtree(self.temp_dir)
        
        # Only remove the test file if we created it
        if self.filepath == self.default_filepath and os.path.exists(self.default_filepath):
            os.remove(self.default_filepath)

    def test_process_file_basic(self, setup_test_environment):
        """Test basic functionality of process_file"""
        # Set plots to False for simpler testing
        plots = False
        if self.filepath is self.default_filepath:
            pytest.skip("No environment variable found. Skipping test.")
        # Create arguments tuple
        args = (self.filepath, self.ID, self.sessID, self.cond, 
                self.epochs_length, self.line_noise, self.sfreq, 
                plots, self.preprocessed_dir)
        
        # Call function
        result = process_file(args)
        
        # Check output directory was created
        expected_dir = f'{self.preprocessed_dir}/{self.ID}/{self.sessID}/eeg'
        assert os.path.exists(expected_dir)
        
        # Check output file was created
        expected_file = f'{expected_dir}/{self.ID}_{self.sessID}_{self.cond}_preprocessed.npy'
        assert os.path.exists(expected_file)
        
        # Verify return value
        assert result == (self.ID, self.sessID, self.cond)
