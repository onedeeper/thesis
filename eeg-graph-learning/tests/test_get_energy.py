import pytest
import numpy as np
import os
from pathlib import Path
from eeglearn.features.energy import Energy
from eeglearn.utils.augmentations import create_base_raw, inject_nans
import pickle

@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_get_energy_initialization():
    """Test Energy class initialization with test data"""
    # Get path from environment variable
    test_dir = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Initialize Energy with the test directory
    energy = Energy(cleaned_path=test_dir,
                   freq_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                   get_labels=True,
                   plots=False,
                   verbose=False,
                   picks=['eeg'],
                   include_bad_channels=True)
    
    # Check that initialization correctly set attributes
    assert energy.cleaned_path == test_dir
    assert len(energy.freq_bands) == 5  # Should have 5 frequency bands
    assert energy.picks == ['eeg']
    assert energy.include_bad_channels is True
    assert len(energy.participant_npy_files) > 0

@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_get_energy_shape():
    """Test that get_energy returns the correct shape of energy matrix"""
    # Get path from environment variable
    test_dir = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Initialize Energy with the test directory
    energy = Energy(cleaned_path=test_dir,
                   freq_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                   get_labels=True,
                   plots=False,
                   verbose=False,
                   picks=['eeg'],
                   include_bad_channels=True)
    
    # Skip if no files
    if len(energy.participant_npy_files) == 0:
        pytest.skip("No .npy files found in the test directory")
    
    # Get energy for the first file
    folder_path, file_name = energy.folders_and_files[0]
    band_matrix = energy.get_energy(folder_path, file_name)
    
    # Check shape: should be (n_channels, n_freq_bands)
    assert isinstance(band_matrix, np.ndarray), "Energy matrix should be a numpy array"
    assert band_matrix.shape[1] == 5, "Should have 5 frequency bands"
    assert band_matrix.shape[0] > 0, "Should have at least one channel"

@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_get_energy_values():
    """Test that get_energy returns valid energy values"""
    # Create synthetic EEG data with higher sampling rate to accommodate all frequency bands
    # Using 200 Hz sampling rate (Nyquist frequency = 100 Hz) to safely handle gamma band (31-50 Hz)
    raw = create_base_raw(n_times=1000, sfreq=200.0)
    
    #  cleaned directory path
    test_dir = Path(os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'))
    
    # Create a mock data object that mimics the structure of preprocessed data
    class MockData:
        def __init__(self, raw):
            self.preprocessed_raw = raw
    
    # Create test data with artifacts
    raw_with_nans = inject_nans(raw, nan_ratio=0.1)
    
    # Create mock data objects
    mock_data_nans = MockData(raw_with_nans)
    
    # Mock np.load to return our test data  
    def mock_np_load(*args, **kwargs):
        if 'nans' in str(args[0]):
            return mock_data_nans
        return MockData(raw)
    
    # Patch np.load
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(np, 'load', mock_np_load)
        
        # Initialize Energy with the test directory
        energy = Energy(cleaned_path=test_dir,  # Path doesn't matter since we're mocking
                       freq_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                       get_labels=False,
                       plots=False,
                       verbose=False,
                       picks=['eeg'],
                       include_bad_channels=True)
        
        # Test with clean data
        clean_matrix = energy.get_energy(Path('dummy_path'), 'clean.npy')
        assert np.all(clean_matrix >= 0), "Clean data energy values should be non-negative"
        assert not np.any(np.isnan(clean_matrix)), "Clean data should not contain NaN values"
        assert not np.any(np.isinf(clean_matrix)), "Clean data should not contain infinite values"
        
        # Test with data containing NaNs - should raise ValueError
        with pytest.raises(ValueError, match=f"NaN values found in band_matrix for nans.npy"):
            energy.get_energy(Path('dummy_path'), 'nans.npy')
        
        # Test with data containing negative values - should ensure they're made positive
        neg_matrix = energy.get_energy(Path('dummy_path'), 'neg.npy')
        assert np.all(neg_matrix >= 0), "Negative values should be converted to positive in energy calculation"
        assert not np.any(np.isnan(neg_matrix)), "Data with negative values should not contain NaN values after processing"
        assert not np.any(np.isinf(neg_matrix)), "Data with negative values should not contain infinite values after processing"

@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_get_energy_frequency_bands():
    """Test that get_energy correctly handles different frequency bands"""
    # Get path from environment variable
    test_dir = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Test with different frequency band combinations
    test_bands = [
        ['delta', 'theta'],  # Test with 2 bands
        ['alpha', 'beta', 'gamma'],  # Test with 3 bands
        ['delta', 'theta', 'alpha', 'beta', 'gamma']  # Test with all bands
    ]
    
    for bands in test_bands:
        # Initialize Energy with the test directory
        energy = Energy(cleaned_path=test_dir,
                       freq_bands=bands,
                       get_labels=True,
                       plots=False,
                       verbose=False,
                       picks=['eeg'],
                       include_bad_channels=True)
        
        # Skip if no files
        if len(energy.participant_npy_files) == 0:
            pytest.skip("No .npy files found in the test directory")
        
        # Get energy for the first file
        folder_path, file_name = energy.folders_and_files[0]
        band_matrix = energy.get_energy(folder_path, file_name)
        
        # Check that the number of frequency bands matches
        assert band_matrix.shape[1] == len(bands), f"Should have {len(bands)} frequency bands"

@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_get_energy_channel_selection():
    """Test that get_energy works with different channel selections"""
    # Get path from environment variable
    test_dir = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Test with different channel selections
    test_picks = [
        ['eeg'],  # Test with all EEG channels
        ['eeg'],  # Test with specific EEG channels (if available)
        None  # Test with default channel selection
    ]
    
    for picks in test_picks:
        # Initialize Energy with the test directory
        energy = Energy(cleaned_path=test_dir,
                       freq_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                       get_labels=True,
                       plots=False,
                       verbose=False,
                       picks=picks,
                       include_bad_channels=True)
        
        # Skip if no files
        if len(energy.participant_npy_files) == 0:
            pytest.skip("No .npy files found in the test directory")
        
        # Get energy for the first file
        folder_path, file_name = energy.folders_and_files[0]
        band_matrix = energy.get_energy(folder_path, file_name)
        
        # Check that we got valid data
        assert isinstance(band_matrix, np.ndarray), "Energy matrix should be a numpy array"
        assert band_matrix.shape[0] > 0, "Should have at least one channel"
        assert band_matrix.shape[1] == 5, "Should have 5 frequency bands"
