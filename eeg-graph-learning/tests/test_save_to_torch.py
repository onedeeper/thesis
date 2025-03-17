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
import torch


from eeglearn.preprocess.preprocessing import Preproccesing
from eeglearn.preprocess import plotting
from eeglearn.preprocess.preprocess_pipeline import clean_pipeline  
from eeglearn.preprocess.save_to_torch import get_filepaths, process_file, preprocess_and_save_data

def test_get_filepaths(tmp_path):
    """
    Test the get_filepaths function, ensuring it retrieves only the expected EEG file paths
    and correctly skips files that have already been processed.
    """
    # Create temporary directories for EEG data and for already processed (save) files
    eeg_dir = tmp_path / "eeg"
    eeg_dir.mkdir()
    save_dir = tmp_path / "save"
    save_dir.mkdir()
    
    # Create dummy EEG files in eeg_dir.
    # Filenames follow the pattern: <participant>_<session>_<condition>_preprocessed.npy
    filenames = [
       "sub-001_ses-1_EO_preprocessed.npy",   # Should be skipped (processed later in save_dir)
       "sub-002_ses-1_EO_preprocessed.npy",   # Should be returned
       "sub-003_ses-1_EC_preprocessed.npy",   # Condition does not match 'EO' when filtering
       "sub-004_ses-1_EO_BAD_preprocessed.npy",  # Contains 'BAD', should be skipped
       "sub-005_ses-2_EO_preprocessed.npy",   # Session does not match ('ses-1'), should be skipped
    ]
    for fname in filenames:
        (eeg_dir / fname).write_bytes(b"dummy")
    
    # Create a dummy processed file in save_dir for sub-001,
    # simulating that sub-001 has already been processed.
    (save_dir / "sub-001.pt").write_bytes(b"processed")
    
    # Patch the module-level "save_dir" in save_to_torch with our temporary save_dir.
    # This ensures that get_filepaths uses our test save directory.
    from eeglearn.preprocess import save_to_torch
    save_to_torch.save_dir = str(save_dir)
    
    # Retrieve file paths filtering for recording_condition "EO" and session "ses-1"
    filepaths = get_filepaths(str(eeg_dir),str(save_dir), recording_condition=["EO"], session="ses-1")
    
    # Only "sub-002_ses-1_EO_preprocessed.npy" should be returned.
    assert len(filepaths) == 1
    assert "sub-002" in filepaths[0]
    assert "ses-1" in filepaths[0]
    assert "EO" in filepaths[0]

def test_process_file_with_real_data():
    """
    Test process_file function using real EEG data.
    """
    # Path to the real test eeg file
    # Use a relative path to the test data file
    test_data_path = os.environ.get('EEG_CLEANED_TEST_FILE')
    if not test_data_path:
        pytest.skip("Environment variable EEG_CLEANED_TEST_FILE not set")
    
    print(f"Using test EEG file from environment: {test_data_path}")
    
    # Extract subject ID from the file path
    subject_id = os.path.basename(test_data_path).split('_')[0]
    condition = os.path.basename(test_data_path).split('_')[2].split('-')[-1]
    print(f"Extracted subject ID: {subject_id}")
    
    # Create a temporary directory for the output
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Process the real data file
        process_file(test_data_path, temp_dir)
        
        # Expected output file name based on the actual subject ID
        expected_output = temp_dir / f"{subject_id}_{condition}.pt"
        
        # Check if the output file exists
        assert expected_output.exists(), f"Output file {expected_output} was not created"
        
        # Load and verify the saved data
        loaded_data = torch.load(expected_output)
        assert isinstance(loaded_data, np.ndarray), "Loaded data is not a numpy array"
        assert len(loaded_data.shape) == 3, "Data should be 3-dimensional (channels x timepoints x epochs)"

def test_preprocess_and_save_data():
    """
    Test the preprocess_and_save_data function with multiple files.
    """
    test_data_path = os.environ.get('EEG_CLEANED_TEST_FILE')
    if not test_data_path:
        pytest.skip("Environment variable EEG_CLEANED_TEST_FILE not set")
        
    # Extract subject ID from the file path
    subject_id = os.path.basename(test_data_path).split('_')[0]
    condition = os.path.basename(test_data_path).split('_')[2].split('-')[-1]
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create a list with the test file path
        filepaths = [str(test_data_path)]
        
        # Process the files
        preprocess_and_save_data(filepaths, temp_dir, n_processes=1)
        
        # Check if output file exists based on the actual subject ID
        expected_output = temp_dir / f"{subject_id}_{condition}.pt"
        assert expected_output.exists(), f"Output file {expected_output} was not created"
        
        # Verify the data
        loaded_data = torch.load(expected_output)
        assert isinstance(loaded_data, np.ndarray), "Loaded data is not a numpy array"
        assert len(loaded_data.shape) == 3, "Data should be 3-dimensional"

def test_process_file_error_handling():
    """
    Test error handling in process_file function.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Test with non-existent file
        non_existent_file = str(temp_dir / "non_existent.npy")
        # Check that no .pt file was created
        pt_files = list(temp_dir.glob("*.pt"))
        assert len(pt_files) == 0, "No files should be created when processing fails"

def test_epoch_data_shape():
    """
    Test the shape of the epoch data.
    """
    # Path to the real test eeg file
    test_data_path = os.environ.get('EEG_CLEANED_TEST_FILE')
    if not test_data_path:
        pytest.skip("Environment variable EEG_CLEANED_TEST_FILE not set")
    
    # dummy save diretory
    save_dir = Path(tempfile.mkdtemp())

    # process the file
    process_file(test_data_path, save_dir)

    # extract subject id from the file path
    subject_id = os.path.basename(test_data_path).split('_')[0]
    condition = os.path.basename(test_data_path).split('_')[2].split('-')[-1]
    # load the saved file
    saved_file = save_dir / f"{subject_id}_{condition}.pt"

    # load the saved file
    loaded_data = torch.load(saved_file)

    # check the shape of the data
    assert loaded_data.shape == (12, 33, 4975)

# def test_correct_condition_name():
#     """
#     Test the correct condition name is used in the saved file.
#     """
#     # Path to the real test eeg file
#     test_data_path = os.environ.get('EEG_CLEANED_TEST_FILE')
#     if not test_data_path:
#         pytest.skip("Environment variable EEG_CLEANED_TEST_FILE not set")
#     # dummy save diretory
#     save_dir = Path(tempfile.mkdtemp())

#     # process the file
#     process_file(test_data_path, save_dir)

#     # load the saved file
#     saved_file = save_dir / f"{subject_id}.pt"

#     # load the saved file
#     loaded_data = torch.load(saved_file)

#     # check if EC or EO is in the file name
#     assert 'EC' in saved_file.name or 'EO' in saved_file.name

#     # extract subject id from the file path
#     subject_id = os.path.basename(test_data_path).split('_')[0]
if __name__ == '__main__':
    pytest.main([__file__])
