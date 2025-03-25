import numpy as np
import mne
import pandas as pd
from typing import Optional, Tuple, List

def create_base_raw(n_times: int = 1000, sfreq: float = 100.0, init_zeros: bool = False) -> mne.io.RawArray:
    """
    Create a base MNE Raw object with synthetic EEG data using the standard 32-channel setup.
    
    Args:
        n_times: Number of time points
        sfreq: Sampling frequency in Hz
        neg_proportion: Proportion of values that should be negative (0.0-1.0)
        
    Returns:
        mne.io.RawArray: Base MNE Raw object
    """
    # Define channel types and names
    ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eog', 'eog', 'eog', 'eog', 'ecg', 'eog', 'emg']

    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz', 'FC4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3',
                'CPz', 'CP4', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2', 'VPVA', 'VNVB', 'HPHL', 'HNHR', 'Erbs',
                'OrbOcc', 'Mass']

    # Define channel positions
    dict_ch_pos_array = {'Fp1' : np.array([-0.02681, 0.08406, -0.01056]),
            'Fp2' : np.array([0.02941, 0.08374, -0.01004]),
            'F7'  : np.array([-0.06699, 0.04169, -0.01596]),
            'F3'  : np.array([-0.04805, 0.05187, 0.03987]),
            'Fz'  : np.array([0.00090, 0.05701, 0.06636]),
            'F4'  : np.array([0.05038, 0.05184, 0.04133]),
            'F8'  : np.array([0.06871, 0.04116, -0.01531]),
            'FC3' : np.array([-0.05883, 0.02102, 0.05482]),
            'FCz' : np.array([0.00057, 0.02463, 0.08763]),
            'FC4' : np.array([0.06029, 0.02116, 0.05558]), 
            'T7'  : np.array([-0.08336, -0.01652, -0.01265]), 
            'C3'  : np.array([-0.06557, -0.01325, 0.06498]),
            'Cz'  : np.array([0.000023, -0.01128, 0.09981]),
            'C4'  : np.array([0.06650, -0.01280, 0.06511]),
            'T8'  : np.array([0.08444, -0.01665, -0.01179]), 
            'CP3' : np.array([-0.06551, -0.04848, 0.06857]),
            'CPz' : np.array([-0.0042, -0.04877, 0.09837]), 
            'CP4' : np.array([0.06503, -0.04835, 0.06857]), 
            'P7'  : np.array([-0.07146, -0.07517, -0.00370]), 
            'P3'  : np.array([-0.05507, -0.08011, 0.05944]), 
            'Pz'  : np.array([-0.00087, -0.08223, 0.08243]),
            'P4'  : np.array([0.05351, -0.08013, 0.05940]), 
            'P8'  : np.array([0.07110, -0.07517, -0.00369]), 
            'O1'  : np.array([-0.02898, -0.11452, 0.00967]),  
            'Oz'  : np.array([-0.00141, -0.11779, 0.01584]),
            'O2'  : np.array([0.02689, -0.11468, 0.00945])
            }

    # Generate synthetic data
    n_channels = len(ch_names)
    if init_zeros:
        data = np.zeros_like(np.random.randn(n_channels, n_times))
    else:
        data = np.random.randn(n_channels, n_times)
    
    # Create montage
    montage = mne.channels.make_dig_montage(ch_pos=dict_ch_pos_array, coord_frame='head')
    
    # Create info object
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    info.set_montage(montage=montage, on_missing='raise')
    
    # Create Raw object
    raw = mne.io.RawArray(data, info)
    return raw

def inject_nans(raw: mne.io.RawArray, nan_ratio: float = 0.1) -> mne.io.RawArray:
    """
    Inject random NaN values into the EEG data.
    
    Args:
        raw: MNE Raw object
        nan_ratio: Ratio of NaN values to inject (0-1)
        
    Returns:
        mne.io.RawArray: MNE Raw object with NaN values
    """
    data = raw.get_data()
    n_nans = int(data.size * nan_ratio)
    nan_indices = np.random.choice(data.size, n_nans, replace=False)
    data.ravel()[nan_indices] = np.nan
    
    new_raw = mne.io.RawArray(data, raw.info)
    return new_raw

def add_gaussian_noise(raw: mne.io.RawArray, noise_level: float = 0.1) -> mne.io.RawArray:
    """
    Add Gaussian noise to the EEG data.
    
    Args:
        raw: MNE Raw object
        noise_level: Standard deviation of the noise
        
    Returns:
        mne.io.RawArray: MNE Raw object with added noise
    """
    data = raw.get_data()
    noise = np.random.normal(0, noise_level, data.shape)
    data = data + noise
    
    new_raw = mne.io.RawArray(data, raw.info)
    return new_raw

def add_baseline_drift(raw: mne.io.RawArray, drift_amplitude: float = 0.5) -> mne.io.RawArray:
    """
    Add baseline drift to the EEG data.
    
    Args:
        raw: MNE Raw object
        drift_amplitude: Amplitude of the drift
        
    Returns:
        mne.io.RawArray: MNE Raw object with baseline drift
    """
    data = raw.get_data()
    n_times = data.shape[1]
    drift = np.linspace(0, drift_amplitude, n_times)
    drift = drift.reshape(1, -1)
    data = data + drift
    
    new_raw = mne.io.RawArray(data, raw.info)
    return new_raw

def add_artifacts(raw: mne.io.RawArray,
                 artifact_type: str = 'blink',
                 n_artifacts: int = 5,
                 amplitude: float = 2.0) -> mne.io.RawArray:
    """
    Add common EEG artifacts to the data.

    Args:
        raw: MNE Raw object
        artifact_type: Type of artifact ('blink', 'muscle', 'line_noise')
        n_artifacts: Number of artifacts to add
        amplitude: Amplitude of the artifacts

    Returns:
        mne.io.RawArray: MNE Raw object with artifacts
    """
    data = raw.get_data()
    n_times = data.shape[1]

    for _ in range(n_artifacts):
        start_idx = np.random.randint(0, n_times - 100)

        if artifact_type == 'blink':
            # Simulate eye blink artifact (front channels)
            blink = amplitude * np.exp(-np.linspace(0, 5, 100))
            data[:4, start_idx:start_idx+100] += blink[None, :]  # Fix broadcasting

        elif artifact_type == 'muscle':
            # Simulate muscle artifact (temporal channels)
            muscle = amplitude * np.random.randn(100)
            data[8:12, start_idx:start_idx+100] += muscle[None, :]  # Fix broadcasting

        elif artifact_type == 'line_noise':
            # Simulate line noise (all channels)
            t = np.linspace(0, 1, 100)
            line_noise = amplitude * np.sin(2 * np.pi * 50 * t)
            data[:, start_idx:start_idx+100] += line_noise[None, :]  # Fix broadcasting

    return mne.io.RawArray(data, raw.info)

def create_test_dataset(n_samples: int = 5,
                       n_times: int = 1000,
                       sfreq: float = 100.0) -> List[mne.io.RawArray]:
    """
    Create a dataset of test MNE Raw objects with various augmentations.
    
    Args:
        n_samples: Number of samples to generate
        n_times: Number of time points per sample
        sfreq: Sampling frequency in Hz
        
    Returns:
        List[mne.io.RawArray]: List of MNE Raw objects
    """
    dataset = []
    
    for i in range(n_samples):
        # Create base raw object
        raw = create_base_raw(n_times, sfreq)
        
        # Randomly apply augmentations
        if np.random.random() < 0.3:
            raw = inject_nans(raw, nan_ratio=0.05)
        if np.random.random() < 0.3:
            raw = add_gaussian_noise(raw, noise_level=0.1)
        if np.random.random() < 0.3:
            raw = add_baseline_drift(raw, drift_amplitude=0.3)
        if np.random.random() < 0.3:
            artifact_type = np.random.choice(['blink', 'muscle', 'line_noise'])
            raw = add_artifacts(raw, artifact_type=artifact_type, n_artifacts=3)
        
        dataset.append(raw)
    
    return dataset 