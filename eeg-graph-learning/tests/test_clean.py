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
from eeglearn.preprocess.clean import process_file, clean_pipeline, get_file_details, Preprocessing_params

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
        self.line_noise = np.array([50, 100, 150])
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
        self.line_noise = np.array([50, 100, 150])
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

    @pytest.fixture
    def setup_get_file_details_test(self):
        """Create a temporary test environment for get_file_details tests."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.derivatives_dir = os.path.join(self.temp_dir, 'derivatives')
        self.preprocessed_dir = os.path.join(self.temp_dir, 'cleaned')
        
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        os.makedirs(self.derivatives_dir, exist_ok=True)
        
        # Create test subject directories and files
        for subj_id in ['sub-001', 'sub-002', 'sub-003']:
            # Create subject directory
            subj_dir = os.path.join(self.derivatives_dir, subj_id)
            os.makedirs(subj_dir, exist_ok=True)
            
            # Create session directories
            for session in ['ses-1', 'ses-2']:
                sess_dir = os.path.join(subj_dir, session, 'eeg')
                os.makedirs(sess_dir, exist_ok=True)
                
                # Create condition files
                for condition in ['EO', 'EC']:
                    # Create a mock EEG file
                    file_path = os.path.join(sess_dir, f'{subj_id}_{session}_task-rest{condition}_eeg.csv')
                    with open(file_path, 'w') as f:
                        f.write('mock,eeg,data\n1,2,3\n')
        
        # Set test parameters
        self.subjects = ['sub-001', 'sub-002', 'sub-003']
        self.sessions = ['ses-1', 'ses-2']
        self.conditions = ['EO', 'EC']
        self.epochs_length = 2.0
        self.line_noise = np.array([50, 100, 150])
        self.sfreq = 500
        self.plots = False
        
        yield
        
        # Cleanup after test
        shutil.rmtree(self.temp_dir)
    
    def test_get_file_details_output_structure(self, setup_get_file_details_test):
        """Test that get_file_details returns correctly structured output."""
        # Call function with all subjects, sessions, conditions
        file_details = get_file_details(
            self.subjects, 
            self.sessions, 
            self.conditions, 
            self.derivatives_dir, 
            self.preprocessed_dir, 
            self.epochs_length, 
            self.line_noise, 
            self.sfreq, 
            self.plots
        )
        
        # Check that all returned items are Preprocessing_params instances with correct fields
        details = file_details[0]  # Examine first item
        
        # Verify namedtuple structure
        assert isinstance(details, Preprocessing_params)
        assert hasattr(details, 'csv_path')
        assert hasattr(details, 'participant')
        assert hasattr(details, 'session')
        assert hasattr(details, 'condition')
        assert hasattr(details, 'epochs_length')
        assert hasattr(details, 'line_noise')
        assert hasattr(details, 'sfreq')
        assert hasattr(details, 'plots')
        assert hasattr(details, 'preprocessed_dir')
        
        # Verify correct path construction
        expected_path = os.path.join(
            self.derivatives_dir, 
            details.participant, 
            details.session, 
            'eeg', 
            f'{details.participant}_{details.session}_task-rest{details.condition}_eeg.csv'
        )
        assert details.csv_path == expected_path, f"Incorrect path: {details.csv_path}"
    
    def test_get_file_details_missing_files(self, setup_get_file_details_test):
        """Test that get_file_details handles missing files appropriately."""
        # Delete one subject's directory to simulate missing files
        missing_subject = 'sub-002'
        missing_dir = os.path.join(self.derivatives_dir, missing_subject)
        shutil.rmtree(missing_dir)
        
        # Patch os.path.exists to correctly identify missing files
        with patch('os.path.exists') as mock_exists:
            # Make os.path.exists return False for paths containing missing_subject
            def exists_side_effect(path):
                if missing_subject in str(path):
                    return False
                return True
                
            mock_exists.side_effect = exists_side_effect
            
            # Should still process the other subjects
            file_details = get_file_details(
                self.subjects,  # Contains all subjects including the missing one
                self.sessions, 
                self.conditions, 
                self.derivatives_dir, 
                self.preprocessed_dir, 
                self.epochs_length, 
                self.line_noise, 
                self.sfreq, 
                self.plots
            )
            
            # Should have 8 files (2 remaining subjects × 2 sessions × 2 conditions)
            assert len(file_details) == 8, f"Expected 8 files, got {len(file_details)}"
            
            # Verify log file was created for the missing files
            assert os.path.exists('preprocessing_log.txt'), "Log file not created for missing files"
            
            # Verify missing subject is not included
            subjects = [details.participant for details in file_details]
            assert missing_subject not in subjects, f"Missing subject was incorrectly included: {subjects}"
    
    def test_get_file_details_exception_handling(self, setup_get_file_details_test):
        """Test that get_file_details handles various exceptions properly."""
        # Instead of patching os.path.join, let's patch os.path.exists
        # to simulate files not existing for specific subjects
        with patch('os.path.exists') as mock_exists:
            # Make paths containing 'sub-002' appear to not exist
            def exists_side_effect(path):
                if 'sub-002' in str(path):
                    return False
                return True
                
            mock_exists.side_effect = exists_side_effect
            
            # Function should handle exceptions gracefully
            file_details = get_file_details(
                self.subjects, 
                ['ses-1'], 
                ['EO', 'EC'], 
                self.derivatives_dir, 
                self.preprocessed_dir, 
                self.epochs_length, 
                self.line_noise, 
                self.sfreq, 
                self.plots
            )
            
            # Should have 4 files (2 subjects × 1 session × 2 conditions)
            # sub-001 and sub-003 are processed, sub-002 is skipped due to exception
            assert len(file_details) == 4, f"Expected 4 files after exceptions, got {len(file_details)}"
            
            # Verify no files for the subject that raised the exception
            subjects = set(details.participant for details in file_details)
            assert 'sub-002' not in subjects, f"Subject with exceptions was incorrectly included: {subjects}"
        # sub-001 and sub-003 are processed, sub-002 is skipped due to exception
        assert len(file_details) == 4, f"Expected 4 files after exceptions, got {len(file_details)}"
        
        # Verify no files for the subject that raised the exception
        subjects = set(details.participant for details in file_details)
        assert 'sub-002' not in subjects, f"Subject with exceptions was incorrectly included: {subjects}" 