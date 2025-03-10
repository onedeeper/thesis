import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from eeglearn.preprocess.preprocessing import Preproccesing
import os
import uuid

# Fixtures are reusable test resources
@pytest.fixture
def sample_eeg_data():
    """Create a sample EEG dataset for testing or use existing one from environment variable."""
    # Check if environment variable is set
    env_file_path = os.environ.get('EEG_TEST_FILE_PATH')
    
    if env_file_path and os.path.exists(env_file_path):
        # Use the file path from the environment variable
        print(f"Using test EEG file from environment: {env_file_path}")
        return str(env_file_path)
    else:
        # Create synthetic data as fallback
        print("No environment variable found. Creating synthetic test data...")
        n_channels = 33
        n_timepoints = 1000
        
        # Create synthetic data
        data = np.random.randn(n_channels, n_timepoints)
        
        # Create a temp directory for the test file
        temp_dir = Path("tests/test_data")
        temp_dir.mkdir(exist_ok=True, parents=True)
        temp_file = temp_dir / f"synthetic_test_eeg_{uuid.uuid4().hex[:8]}.csv"
        
        # Format data as a DataFrame
        ch_names = [f'EEG{i:03d}' for i in range(1, n_channels + 1)]
        df = pd.DataFrame(data.T, columns=ch_names)  # Transpose to have channels as columns
        
        # Save to CSV
        df.to_csv(temp_file, index=False)
        
        
        return str(temp_file)

def test_preprocessing_initialization(sample_eeg_data):
    """Test if Preprocessing class initializes correctly."""
    # Test with default parameters
    preprocess = Preproccesing(
        filename=sample_eeg_data,
        epochs_length=0,
        line_noise=[],
        sfreq=500,
        plots=False
    )
    
    # Check if basic attributes are created
    assert hasattr(preprocess, 'prep_data'), "prep_data attribute not created"
    assert hasattr(preprocess, 'preprocessed_raw'), "preprocessed_raw attribute not created"
    assert hasattr(preprocess, 'preprocessed_epochs'), "preprocessed_epochs attribute not created"

def test_bad_channel_detection(sample_eeg_data):
    """Test if bad channel detection works."""
    preprocess = Preproccesing(
        filename=sample_eeg_data,
        epochs_length=0,
        line_noise=[50],  # Test with line noise removal
        sfreq=500,
        plots=False
    )
    
    # Check if bad channel attributes exist
    assert hasattr(preprocess, 'bad_channels_original')
    assert hasattr(preprocess, 'bad_channels_before_interpolation')
    assert hasattr(preprocess, 'bad_channels_after_interpolation')
    assert hasattr(preprocess, 'still_bad_channels')
    
    # Check if bad channel attributes are of correct type
    assert isinstance(preprocess.bad_channels_original, dict)
    assert isinstance(preprocess.still_bad_channels, list)

def test_epoching(sample_eeg_data):
    """Test if epoching works correctly."""
    epoch_length = 2.0  # 2 seconds
    
    preprocess = Preproccesing(
        filename=sample_eeg_data,
        epochs_length=epoch_length,
        line_noise=[],
        sfreq=500,
        plots=False
    )
    
    # Check if epochs were created
    assert preprocess.preprocessed_epochs != 'No epoching applied'
    
    # Check epoch properties
    expected_epoch_samples = int(epoch_length * 500)  # sfreq = 500
    assert preprocess.preprocessed_epochs.get_data().shape[2] == expected_epoch_samples

def test_line_noise_removal(sample_eeg_data):
    """Test if line noise removal is working."""
    preprocess = Preproccesing(
        filename=sample_eeg_data,
        epochs_length=0,
        line_noise=[50, 100],  # Test multiple line noise frequencies
        sfreq=500,
        plots=False
    )
    
    # The raw data should be filtered
    assert preprocess.preprocessed_raw.info['highpass'] == 1.0  # From the code's filter settings
    assert preprocess.preprocessed_raw.info['lowpass'] == 100.0  # From the code's filter settings

# Error handling tests
def test_invalid_file_handling():
    """Test if the class handles invalid files appropriately."""
    with pytest.raises(FileNotFoundError):
        Preproccesing(
            filename="nonexistent_file.csv",
            epochs_length=0,
            line_noise=[],
            sfreq=500,
            plots=False
        )
