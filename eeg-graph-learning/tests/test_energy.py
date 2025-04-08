import pytest
import numpy as np
import os
from pathlib import Path
from eeglearn.features.energy import Energy
from eeglearn.utils.augmentations import create_base_raw, inject_nans
import pickle
import torch
import tempfile

TEST_FILE : str = "sub-19740274_ses-1_task-restEC_preprocessed.npy"

@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_get_energy_initialization() -> None:
    """Test Energy class initialization with test data"""
    # Get path from environment variable
    test_dir : str = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Initialize Energy with the test directory
    energy : Energy = Energy(cleaned_path=test_dir,
                   select_freq_bands=['delta', 'theta', 'beta', 'gamma'],
                   energy_plots=False,
                   include_bad_channels_psd=True)
    
    # Check that initialization correctly set attributes
    assert energy.cleaned_path == test_dir
    assert len(energy.select_freq_bands) ==4  # Should have 5 frequency bands
    assert energy.include_bad_channels_psd is True
    assert len(energy.participant_npy_files) > 0

    # test with bands = None
    # Initialize Energy with the test directory
    energy : Energy = Energy(cleaned_path=test_dir,
                   select_freq_bands=None,
                   energy_plots=False,  
                   include_bad_channels_psd=True)
    
    assert energy.cleaned_path == test_dir
    assert len(energy.select_freq_bands) == 5  # Should have 5 frequency bands
    assert energy.include_bad_channels_psd is True
    assert len(energy.participant_npy_files) > 0



@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_get_energy_len()-> None:
    """
    Test if the len method returns the number of the files to be proccessed. 
    This wil eventually be the number of energy objects generated. 
    """
    test_dir = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Initialize Energy with the test directory
    energy : Energy = Energy(cleaned_path=test_dir,
                   select_freq_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                   full_time_series=True,
                   energy_plots=False,
                   include_bad_channels_psd=False)
    assert len(energy) > 0, "No files found"


@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_get_energy_item()-> None:
    "Tests if the __getitem__method returns a processed energy object."
    test_dir : str = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Initialize Energy with the test directory
    energy : Energy = Energy(cleaned_path=test_dir,
                   select_freq_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                   full_time_series=True,
                   energy_plots=False,
                   verbose_psd=False,
                   include_bad_channels_psd=False)
    assert energy[0][0].shape[0] == 26
    assert energy[0][0].shape[1] == 5

@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_get_energy_shape()-> None:
    """Test that get_energy returns the correct shape of energy matrix"""
    
    # Get path from environment variable
    test_dir : str = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Initialize Energy with the test directory
    energy = Energy(cleaned_path=test_dir,
                   select_freq_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                   full_time_series=True,
                   energy_plots=False,
                   verbose_psd=False,
                   include_bad_channels_psd=True)
    
    # Skip if no files
    if len(energy.participant_npy_files) == 0:
        pytest.skip("No .npy files found in the test directory")
    
    # Get energy for the first file
    folder_path : Path
    file_name : str 
    folder_path, file_name = energy.folders_and_files[0]
    band_matrix : torch.Tensor = energy.get_energy(folder_path, file_name)
    
    # Check shape: should be (n_channels, n_select_freq_bands)
    assert isinstance(band_matrix, torch.Tensor),\
        "Energy matrix should be a torch.Tensor"
    assert band_matrix.shape[1] == len(energy.select_freq_bands), \
        "Should have 5 frequency bands"
    assert band_matrix.shape[0] > 0, "Should have at least one channel"

    # Test epoched energy
    energy = Energy(cleaned_path=test_dir,
                   select_freq_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                   full_time_series=False,
                   energy_plots=False,
                   verbose_psd=False,
                   include_bad_channels_psd=True)
    band_matrix = energy.get_energy(folder_path, file_name)

    assert isinstance(band_matrix, torch.Tensor), \
        "Energy matrix should be a torch.Tensor"
    assert band_matrix.shape[1] == len(energy.select_freq_bands) * 12, \
        f"Should have {len(energy.select_freq_bands) * 12} frequency bands"
    assert band_matrix.shape[0] > 0, "Should have at least one channel"

@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_get_energy_values()-> None:
    """Test that get_energy returns valid energy values"""
    dir_path = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')

    # test with full time series , bad channels included
    energy = Energy(cleaned_path=dir_path,
                   select_freq_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                   full_time_series= True,
                   energy_plots=False,
                   verbose_psd=False,
                   include_bad_channels_psd=True)
    
    data  = energy.get_energy(folder_path=Path(dir_path) / "sub-19740274" / "ses-1" / "eeg" ,
                               file_name= TEST_FILE)
    
    assert isinstance(data,torch.Tensor), "Should be a torch tensor"
    assert data.shape[0] ==  26
    assert data.shape[1] == 5

def test_parallel_returns() -> None:
    """Test that the parallel method returns the correct number of files."""
    dir_path : str = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    energy : Energy = Energy(cleaned_path=dir_path,
                            select_freq_bands=['delta', 'theta',
                                                'alpha', 'beta', 'gamma'],
                            full_time_series=True,
                            save_to_disk=False)
    files = energy.run_energy_parallel()
    assert len(files) == 1

