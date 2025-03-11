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
from eeglearn.preprocess.preprocess_pipeline import process_file, preprocess_pipeline

class TestProcessFile:
    
    @pytest.fixture
    def setup_test_environment(self):
        """Create a temporary test environment"""
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp()
        self.preprocessed_dir = os.path.join(self.temp_dir, 'cleaned')
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        
        # Define test parameters with default filepath
        default_filepath = "tests/test_data/test_eeg.csv"
        
        # Use environment variable if available, otherwise use default
        self.filepath = os.environ.get('EEG_TEST_FILE_PATH', default_filepath)
        
        self.ID = "sub-19694366"
        self.sessID = "ses-1"
        self.cond = "EO"
        self.epochs_length = 2.0
        self.line_noise = [50, 100, 150]
        self.sfreq = 500

        # Only create synthetic data if using the default filepath
        if self.filepath == default_filepath:
            n_channels = 33
            n_timepoints = 1000
            
            # Create synthetic data
            data = np.random.randn(n_channels, n_timepoints)
            
            # Save to default filepath
            temp_dir = Path(os.path.dirname(default_filepath))
            temp_dir.mkdir(exist_ok=True, parents=True)
            temp_file = Path(default_filepath)
            
            # Format like your real data
            df = pd.DataFrame(data.T)  # Transpose because your code expects channels as columns
            df.to_csv(temp_file, index=False)
        
        yield  # This allows the test to run
        
        # Cleanup after test
        shutil.rmtree(self.temp_dir)
        
        # Only remove the test file if we created it
        if self.filepath == default_filepath and os.path.exists(default_filepath):
            os.remove(default_filepath)

    def test_process_file_basic(self, setup_test_environment):
        """Test basic functionality of process_file"""
        # Set plots to False for simpler testing
        plots = False
        
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
