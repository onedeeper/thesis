import os
import pytest
import numpy as np
import torch
from pathlib import Path
import shutil
from eeglearn.features.spectrum import PowerSpectrum


@pytest.fixture
def setup_and_cleanup_test_dirs():
    """Fixture to set up and clean up test directories before and after tests"""
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
    """Test PowerSpectrum class initialization with test data"""
    # Get path from environment variable
    test_dir = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Initialize PowerSpectrum with the test directory
    ps = PowerSpectrum(cleaned_path=test_dir, 
                      full_time_series=True,
                      method='welch',
                      fmin=1, 
                      fmax=40)
    
    # Check that initialization correctly set attributes
    assert ps.data_path == test_dir
    assert ps.fmin == 1
    assert ps.fmax == 40
    assert ps.full_time_series is True
    assert ps.method == 'welch'
    assert len(ps.participant_npy_files) > 0


@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_power_spectrum_computation(setup_and_cleanup_test_dirs):
    """Test computation of power spectrum with test data"""
    # Get path from environment variable
    test_dir = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Initialize PowerSpectrum with the test directory and limit to a small frequency range
    ps = PowerSpectrum(cleaned_path=test_dir, 
                      full_time_series=True,
                      method='welch',
                      fmin=8,  # Alpha band start 
                      fmax=13) # Alpha band end
    
    # Get the first npy file for testing
    if len(ps.folders_and_files) == 0:
        pytest.skip("No .npy files found in the test directory")
    
    folder_path, file_name = ps.folders_and_files[0]
    
    # Compute spectrum for a single file
    ps.get_spectrum(folder_path, file_name)
    
    # Extract participant_id and condition for verification
    from eeglearn.utils.utils import get_participant_id_condition_from_string
    participant_id, condition = get_participant_id_condition_from_string(file_name)
    
    # Check that files were created
    spectrum_file = ps.spectrum_save_dir / f'psd_{participant_id}_{condition}.pt'
    freqs_file = ps.spectrum_save_dir / f'freqs_{participant_id}_{condition}.pt'
    
    assert spectrum_file.exists(), f"Spectrum file {spectrum_file} not created"
    assert freqs_file.exists(), f"Frequencies file {freqs_file} not created"
    
    # Load and verify the computed data
    spectra = torch.load(spectrum_file)
    freqs = torch.load(freqs_file)
    
    # Basic validation of the output shapes and values
    assert isinstance(spectra, np.ndarray), "Spectrum should be a numpy array"
    assert isinstance(freqs, np.ndarray), "Frequencies should be a numpy array"
    assert len(freqs) > 0, "Frequencies array should not be empty"
    assert freqs.min() >= 8, "Minimum frequency should match fmin"
    assert freqs.max() <= 13, "Maximum frequency should match fmax"


@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_power_spectrum_getitem(setup_and_cleanup_test_dirs):
    """Test __getitem__ method of PowerSpectrum after computing spectra"""
    # Get path from environment variable
    test_dir = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Initialize PowerSpectrum with the test directory
    ps = PowerSpectrum(cleaned_path=test_dir, 
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
        assert isinstance(spectra, np.ndarray), "Spectrum should be a numpy array"
        assert isinstance(freqs, np.ndarray), "Frequencies should be a numpy array"
        assert spectra.shape[0] > 0, "Spectrum should have data for at least one channel"
        assert len(freqs) > 0, "Frequencies array should not be empty"


@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_power_spectrum_epoched(setup_and_cleanup_test_dirs):
    """Test computing epoched power spectrum"""
    # Get path from environment variable
    test_dir = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Initialize PowerSpectrum with the test directory for epoched data
    ps = PowerSpectrum(cleaned_path=test_dir, 
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
        assert isinstance(freqs, np.ndarray), "Frequencies should be a numpy array"
        assert len(freqs) > 0, "Frequencies array should not be empty"
