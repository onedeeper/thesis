import pytest
import numpy as np
import mne
from eeglearn.utils.augmentations import (
    create_base_raw,
    inject_nans,
    add_gaussian_noise,
    add_baseline_drift,
    add_artifacts,
    create_test_dataset
)

def test_create_base_raw():
    """Test creation of base raw EEG data"""
    raw = create_base_raw(n_times=1000, sfreq=100.0)
    assert raw.n_times == 1000
    assert raw.info['sfreq'] == 100.0
    assert len(raw.ch_names) == 33  # 32 channels total

def test_inject_nans():
    """Test injection of NaN values"""
    raw = create_base_raw()
    raw_with_nans = inject_nans(raw, nan_ratio=0.1)
    data = raw_with_nans.get_data()
    assert np.isnan(data).sum() > 0
    assert np.isnan(data).sum() / data.size == pytest.approx(0.1, rel=0.1)

def test_add_gaussian_noise():
    """Test addition of Gaussian noise"""
    raw = create_base_raw()
    raw_with_noise = add_gaussian_noise(raw, noise_level=0.1)
    data = raw_with_noise.get_data()
    assert data.shape == raw.get_data().shape
    assert not np.array_equal(data, raw.get_data())

def test_add_baseline_drift():
    """Test addition of baseline drift"""
    raw = create_base_raw()
    raw_with_drift = add_baseline_drift(raw, drift_amplitude=0.5)
    data = raw_with_drift.get_data()
    assert data.shape == raw.get_data().shape
    # Check that values increase over time
    assert np.mean(data[:, -1]) > np.mean(data[:, 0])

def test_add_artifacts():
    """Test addition of different types of artifacts"""
    raw = create_base_raw()
    artifact_types = ['blink', 'muscle', 'line_noise']
    
    for artifact_type in artifact_types:
        raw_with_artifacts = add_artifacts(raw, artifact_type=artifact_type)
        data = raw_with_artifacts.get_data()
        assert data.shape == raw.get_data().shape
        assert not np.array_equal(data, raw.get_data())

def test_create_test_dataset():
    """Test creation of test dataset with various augmentations"""
    dataset = create_test_dataset(n_samples=3)
    assert len(dataset) == 3
    assert all(isinstance(raw, mne.io.RawArray) for raw in dataset)
    # Check that at least some samples have artifacts
    has_artifacts = False
    for raw in dataset:
        data = raw.get_data()
        if np.isnan(data).any() or (data < 0).any():
            has_artifacts = True
            break
    assert has_artifacts 