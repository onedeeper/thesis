import os
import numpy as np
import pickle
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import pandas as pd

# Import the functions to test
from eeglearn.preprocess.clean import process_file, clean_pipeline

class TestClean:
    """Test class for clean.py functions."""
    
    @pytest.fixture
    def setup_process_file_test(self):
        """Create a temporary test environment for process_file tests."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.preprocessed_dir = os.path.join(self.temp_dir, 'cleaned')
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        
        # Define test parameters
        self.default_filepath = "tests/test_data/test_eeg.csv"
        self.filepath = os.environ.get('EEG_TEST_FILE_PATH', self.default_filepath)
        
        # Skip test if environment variable is not set
        if self.filepath == self.default_filepath:
            pytest.skip("EEG_TEST_FILE_PATH environment variable not found. Skipping test.")
            
        self.ID = "sub-19740274"
        self.sessID = "ses-1"
        self.cond = "EO"
        self.epochs_length = 2.0
        self.line_noise = [50, 100, 150]
        self.sfreq = 500
        self.plots = False
        
        yield  # This allows the test to run
        
        # Cleanup after test
        shutil.rmtree(self.temp_dir)
    
    @pytest.fixture
    def setup_clean_pipeline_test(self):
        """Create a temporary test environment for clean_pipeline tests."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.preprocessed_dir = os.path.join(self.temp_dir, 'cleaned')
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        
        # Define test parameters
        self.default_derivatives_dir = "tests/test_data/TDBRAIN-dataset/derivatives"
        self.derivatives_dir = os.environ.get('EEG_TEST_DERIVATIVES_DIR', self.default_derivatives_dir)
        
        # Skip test if environment variable is not set
        if self.derivatives_dir == self.default_derivatives_dir:
            pytest.skip("EEG_TEST_DERIVATIVES_DIR environment variable not found. Skipping test.")
            
        self.epochs_length = 2.0
        self.line_noise = [50, 100, 150]
        self.sfreq = 500
        self.plots = False
        self.n_processes = 1  # Use single process for testing
        self.conditions = ['EO']
        self.sessions = ['ses-1']
        
        yield  # This allows the test to run
        
        # Cleanup after test
        shutil.rmtree(self.temp_dir)
    
    def test_process_file(self, setup_process_file_test):
        """Test basic functionality of process_file."""
        # Create arguments tuple
        args = (
            self.filepath, self.ID, self.sessID, self.cond,
            self.epochs_length, self.line_noise, self.sfreq,
            self.plots, self.preprocessed_dir
        )
        
        # Call the function
        result = process_file(args)
        
        # Check output directory was created
        expected_dir = Path(self.preprocessed_dir) / self.ID / self.sessID / 'eeg'
        assert os.path.exists(expected_dir)
        
        # Check output file was created
        expected_file = expected_dir / f'{self.ID}_{self.sessID}_{self.cond}_preprocessed.npy'
        assert os.path.exists(expected_file)
        
        # Verify return value
        assert result == (self.ID, self.sessID, self.cond)
        
        # Check if the file can be properly loaded
        with open(expected_file, 'rb') as f:
            preprocessed_data = pickle.load(f)
            
        # Verify basic attributes of preprocessed data
        assert hasattr(preprocessed_data, 'preprocessed_raw')
        assert hasattr(preprocessed_data, 'status')
    
    @patch('eeglearn.preprocess.clean.Pool')
    def test_clean_pipeline_with_mock(self, mock_pool, setup_clean_pipeline_test):
        """Test clean_pipeline with mocked Pool to prevent actual processing."""
        # Setup mock pool
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        
        # Setup a mock list of files to process
        mock_file_list = [
            (f"path/to/file_{i}.csv", f"sub-{i}", "ses-1", "EO", 
             self.epochs_length, self.line_noise, self.sfreq, 
             self.plots, self.preprocessed_dir) 
            for i in range(3)
        ]
        
        # Patch os.walk to return our mock file list
        with patch('os.walk') as mock_walk:
            # Setup mock os.walk to return data that will generate our mock file list
            mock_walk.return_value = [
                ("path/to", [], [f"sub-{i}_ses-1_EO_eeg.csv" for i in range(3)])
            ]
            
            # Patch the os.path.join to return our desired paths
            with patch('os.path.join', side_effect=lambda *args: '/'.join(args)):
                
                # Call the function
                clean_pipeline(
                    derivates_dir=self.derivatives_dir,
                    preprocessed_dir=self.preprocessed_dir,
                    sfreq=self.sfreq,
                    epochs_length=self.epochs_length,
                    line_noise=self.line_noise,
                    n_processes=self.n_processes,
                    conditions=self.conditions,
                    sessions=self.sessions,
                    plots=self.plots
                )
                
                # Verify that Pool was called
                mock_pool.assert_called_once() 