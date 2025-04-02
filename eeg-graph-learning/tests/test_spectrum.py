"""Created on Thu Apr 02 2025.

author: Udesh Habaraduwa
description: test the PowerSpectrum class

name: test_spectrum.py

version: 1.0
"""
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from eeglearn.features.spectrum import PowerSpectrum

# Extract participant_id and condition
from eeglearn.utils.utils import get_participant_id_condition_from_string


@pytest.fixture
def setup_and_cleanup_test_dirs():
    """Fixture to set up and clean up test directories before and after tests.
    
    This fixture creates the necessary directories for storing PSD plots and spectra,
    and ensures they are cleaned up after the tests are run.
    """
    # Setup - create directories
    project_root = Path(__file__).resolve().parent.parent
    test_dirs = [
        project_root / 'data' / 'psd' / 'plots',
        project_root / 'data' / 'psd' / 'spectra',
        project_root / 'data' / 'psd' / 'spectra_epoched'
    ]
    
    # Create the directories
    for directory in test_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Teardown - remove test directories
    for directory in test_dirs:
        if directory.exists():
            shutil.rmtree(directory)


@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_power_spectrum_initialization():
    """Test PowerSpectrum class initialization with test data.
    
    This test checks that the PowerSpectrum class initializes correctly with the
    test data directory and that the attributes are set as expected.
    """
    # Get path from environment variable
    test_dir = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Initialize PowerSpectrum with the test directory
    ps = PowerSpectrum(cleaned_path=test_dir, 
                       include_bad_channels=True,
                      full_time_series=True,
                      method='welch',
                      fmin=1, 
                      fmax=40)
    
    # Check that initialization correctly set attributes
    assert ps.cleaned_path == test_dir
    assert ps.fmin == 1
    assert ps.fmax == 40
    assert ps.full_time_series is True
    assert ps.method == 'welch'
    assert len(ps.participant_npy_files) > 0


@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_power_spectrum_computation(setup_and_cleanup_test_dirs):
    """Test computation of power spectrum with test data.
    
    This test checks that the power spectrum is computed correctly for a single file
    in the test data directory. It verifies that the computed spectrum and frequencies
    are of the correct shape and type, and that the minimum and maximum frequencies
    match the specified range.
    """
    # Get path from environment variable
    test_dir = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Initialize PowerSpectrum with the test directory and limit to a small frequency
    #  range
    ps = PowerSpectrum(cleaned_path=test_dir, 
                      include_bad_channels=True,
                      full_time_series=True,
                      method='welch',
                      fmin=8,  # Alpha band start 
                      fmax=13) # Alpha band end
    
    # Get the first npy file for testing
    if len(ps.folders_and_files) == 0:
        pytest.skip("No .npy files found in the test directory")
    
    folder_path, file_name = ps.folders_and_files[0]
    print(folder_path, file_name)
    # Compute spectrum for a single file
    ps.get_spectrum(folder_path, file_name)
    
    # Extract participant_id and condition for verification
    from eeglearn.utils.utils import get_participant_id_condition_from_string
    participant_id, condition = get_participant_id_condition_from_string(file_name)
    
    # Check that files were created
    spectrum_file = ps.spectrum_save_dir / f'psd_{participant_id}_{condition}.pt'
    freqs_file = ps.spectrum_save_dir / f'freqs_{participant_id}_{condition}.pt'
    
    assert spectrum_file.exists(), f"Spectrum file {spectrum_file} not   created"
    assert freqs_file.exists(), f"Frequencies file {freqs_file} not created"
    
    # Load and verify the computed data
    spectra = torch.load(spectrum_file)
    freqs = torch.load(freqs_file)
    
    # Basic validation of the output shapes and values
    assert isinstance(spectra, torch.Tensor), "Spectrum should be a torch.Tensor"
    assert isinstance(freqs, torch.Tensor), "Frequencies should be a torch.Tensor"
    assert len(freqs) > 0, "Frequencies array should not be empty"
    assert freqs.min() >= 8, "Minimum frequency should match fmin"
    assert freqs.max() <= 13, "Maximum frequency should match fmax"


@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_power_spectrum_getitem(setup_and_cleanup_test_dirs):
    """Test __getitem__ method of PowerSpectrum after computing spectra.
    
    This test checks that the __getitem__ method of the PowerSpectrum class returns
    the correct data when accessing an item from the dataset. It verifies that the
    spectrum and frequency data are of the correct shape and type, and that the
    computation has been run.
    """
    # Get path from environment variable
    test_dir = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Initialize PowerSpectrum with the test directory
    ps = PowerSpectrum(cleaned_path=test_dir, 
                      include_bad_channels=True,
                      full_time_series=True,
                      method='welch',
                      fmin=1, 
                      fmax=40)
    
    # Skip if no files
    if len(ps.participant_npy_files) == 0:
        pytest.skip("No .npy files found in the test directory")
    
    # Force computation by accessing an item
    spectra, freqs, label = ps[0]
    
    # Check that computation ran and data was returned
    assert ps.ran_spectrum is True, "Spectrum computation should have run"
    
    # Only validate if the file was found (not None)
    if spectra is not None and freqs is not None:
        assert isinstance(spectra, torch.Tensor), "Spectrum should be a torch.Tensor"
        assert isinstance(freqs, torch.Tensor), "Frequencies should be a torch.Tensor"
        assert spectra.shape[0] > 0,"Spectrum should have data for at least one channel"
        assert len(freqs) > 0, "Frequencies array should not be empty"


@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_power_spectrum_epoched(setup_and_cleanup_test_dirs):
    """Test computing epoched power spectrum.
    
    This test checks that the power spectrum is computed correctly for epoched data.
    It verifies that the computed spectrum and frequencies are of the correct shape
    and type, and that the computation has been run.
    """
    # Get path from environment variable
    test_dir = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Initialize PowerSpectrum with the test directory for epoched data
    ps = PowerSpectrum(cleaned_path=test_dir, 
                      include_bad_channels=True,
                      full_time_series=False,  # Use epoched data
                      method='welch',
                      fmin=1, 
                      fmax=40)
    
    # Skip if no files
    if len(ps.participant_npy_files) == 0:
        pytest.skip("No .npy files found in the test directory")
    
    # Force computation by accessing an item
    spectra, freqs , label = ps[0]
    
    # Check that computation ran
    assert ps.ran_spectrum is True, "Spectrum computation should have run"
    
    # If data was found, verify its structure
    if spectra is not None and freqs is not None:
        # For epoched data, we expect shape (n_epochs, n_channels, n_frequencies)
        assert len(spectra.shape) == 3, "Epoched spectra should have 3 dimensions"
        assert isinstance(freqs, torch.Tensor), "Frequencies should be a torch.Tensor"
        assert len(freqs) > 0, "Frequencies array should not be empty"


@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_include_bad_channels_full_time_series(setup_and_cleanup_test_dirs):
    """Test that include_bad_channels=True properly.
    
    This test checks that the PowerSpectrum class correctly handles bad channels
    when computing the power spectrum for full time series data. It verifies that
    the computed spectrum and frequencies are of the correct shape and type, and
    that the computation has been run.
    """
    # Get path from environment variable
    test_dir = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Initialize PowerSpectrum with include_bad_channels=True
    ps_include_bad = PowerSpectrum(
        cleaned_path=test_dir, 
        include_bad_channels=True,
        full_time_series=True,
        method='welch',
        fmin=1, 
        fmax=40
    )
    
    # Also initialize a version with include_bad_channels=False for comparison
    ps_exclude_bad = PowerSpectrum(
        cleaned_path=test_dir, 
        include_bad_channels=False,
        full_time_series=True,
        method='welch',
        fmin=1, 
        fmax=40
    )
    
    # Skip if no files
    if len(ps_include_bad.participant_npy_files) == 0:
        pytest.skip("No .npy files found in the test directory")
    
    # Get the first file for testing
    folder_path, file_name = ps_include_bad.folders_and_files[0]
    
    # Load data to check if there are any bad channels
    data = np.load(folder_path / file_name, allow_pickle=True)
    if not hasattr(data, 'still_bad_channels') or len(data.still_bad_channels) == 0:
        print("No bad channels found in the data. Adding T7 as a bad channel.")
        data.preprocessed_raw.info['bads'] = ['T7']
        data.preprocessed_epochs.info['bads'] = ['T7']
        data.still_bad_channels = ['T7']
        
    
    # create a temporary folder paths
    temp_folder_path = Path(tempfile.mkdtemp())
    temp_save_dir = temp_folder_path / 'psd'
    temp_save_dir.mkdir(parents=True, exist_ok=True)
    # save the data to the temporary folder path
    pickle.dump(data, open(temp_folder_path / file_name, 'wb', pickle.HIGHEST_PROTOCOL))
    # change the directory path of the PowerSpectrum 
    # object to the temporary  save directory path
    ps_include_bad.spectrum_save_dir = temp_save_dir
    ps_exclude_bad.spectrum_save_dir = temp_save_dir

    participant_id, condition = get_participant_id_condition_from_string(file_name)

    # Process with both PowerSpectrum instances
    ps_include_bad.get_spectrum(temp_folder_path, file_name)
    # Load the saved spectra
    spectra_include = torch.load(ps_include_bad.spectrum_save_dir \
                                 / f'psd_{participant_id}_{condition}.pt')
    # Expected number of EEG channels is 26
    print( " spectra_include.shape", spectra_include.shape)
    assert spectra_include.shape[0] == 26, \
        "When including bad channels, PSD should have 26 channels"

    # Process with both PowerSpectrum instances
    ps_exclude_bad.get_spectrum(temp_folder_path, file_name)
    spectra_exclude = torch.load(ps_exclude_bad.spectrum_save_dir \
                                 / f'psd_{participant_id}_{condition}.pt')
    # When respecting bad channels, the number of channels should be less than 26
    assert spectra_exclude.shape[0] < 26, \
          "When excluding bad channels,  PSD should have fewer than 26 channels"


@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_include_bad_channels_epoched(setup_and_cleanup_test_dirs):
    """Test that include_bad_channels=True.
    
    This test checks that the PowerSpectrum class correctly handles bad channels
    when computing the power spectrum for epoched data. It verifies that the
    computed spectrum and frequencies are of the correct shape and type, and
    that the computation has been run.
    """
    # Get path from environment variable
    test_dir = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Initialize PowerSpectrum with include_bad_channels=True
    ps_include_bad = PowerSpectrum(
        cleaned_path=test_dir, 
        full_time_series=False,  # Use epoched data
        include_bad_channels=True,
        method='welch',
        fmin=1, 
        fmax=40
    )
    
    # Also initialize a version with include_bad_channels=False for comparison
    ps_exclude_bad = PowerSpectrum(
        cleaned_path=test_dir, 
        full_time_series=False,  # Use epoched data
        include_bad_channels=False,
        method='welch',
        fmin=1, 
        fmax=40
    )
    
    # Skip if no files
    if len(ps_include_bad.participant_npy_files) == 0:
        pytest.skip("No .npy files found in the test directory")
    
    # Get the first file for testing
    folder_path, file_name = ps_include_bad.folders_and_files[0]
    
    # Load data to check if there are any bad channels
    data = np.load(folder_path / file_name, allow_pickle=True)
    if not hasattr(data, 'still_bad_channels') or len(data.still_bad_channels) == 0:
        print("No bad channels found in the data. Adding T7 as a bad channel.")
        data.preprocessed_raw.info['bads'] = ['T7']
        data.preprocessed_epochs.info['bads'] = ['T7']
        data.still_bad_channels = ['T7']
    
    # create a temporary folder paths
    temp_folder_path = Path(tempfile.mkdtemp())
    temp_save_dir_epoched = temp_folder_path / 'psd' / 'epoched'
    temp_save_dir_epoched.mkdir(parents=True, exist_ok=True)
    
    # save the data to the temporary folder path
    pickle.dump(data, open(temp_folder_path / file_name, 'wb', pickle.HIGHEST_PROTOCOL))
    
    # change the directory path of the PowerSpectrum object to the
    #  temporary save directory path
    ps_include_bad.spectrum_save_dir_epoched = temp_save_dir_epoched
    ps_exclude_bad.spectrum_save_dir_epoched = temp_save_dir_epoched

    participant_id, condition = get_participant_id_condition_from_string(file_name)

    # Process with include_bad_channels=True PowerSpectrum instance
    ps_include_bad.get_spectrum(temp_folder_path, file_name)
    # Load the saved spectra
    spectra_include = torch.load(ps_include_bad.spectrum_save_dir_epoched \
                                  / f'psd_{participant_id}_{condition}.pt')
    # For epoched data, the shape is (n_epochs, n_channels, n_frequencies)
    # Expected number of EEG channels is 26
    print(" spectra_include.shape", spectra_include.shape)
    assert spectra_include.shape[1] == 26,\
          "When including bad channels, PSD should have 26 channels"

    # Process with include_bad_channels=False PowerSpectrum instance
    ps_exclude_bad.get_spectrum(temp_folder_path, file_name)
    # Load the saved spectra
    spectra_exclude = torch.load(ps_exclude_bad.spectrum_save_dir_epoched \
                                 / f'psd_{participant_id}_{condition}.pt')
    # When excluding bad channels, the number of channels should be less than 26
    assert spectra_exclude.shape[1] < 26, \
          "When excluding bad channels, PSD should have fewer than 26 channels"
