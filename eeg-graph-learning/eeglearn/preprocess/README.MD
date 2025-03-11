# EEG Preprocessing Module

Originally written by Tuurr Smolders 
Updated by Udesh Habaraduwa 

## Overview
This module provides a comprehensive pipeline for preprocessing EEG data from the TDBRAIN dataset, implementing various cleaning and artifact removal techniques.

## Key Components

### `preprocessing.py`
- Main preprocessing class (`Preproccesing`) that handles individual EEG file processing
- Implements:
  - PREP pipeline for noise removal
  - ICA for artifact correction (EOG, ECG, EMG)
  - Bandpass filtering (1-100Hz)
  - Optional epoching
  - Bad channel detection and interpolation

### `plotting.py`
- Visualization utilities for EEG data
- Features:
  - Raw data plots
  - Power spectral density plots
  - Time-frequency decomposition
  - ICA component visualization
- Uses matplotlib with Agg backend for non-interactive plotting

### `preprocess_pipeline.py`
- Batch processing pipeline for multiple EEG recordings
- Handles:
  - Directory traversal
  - File organization
  - Parallel processing of multiple subjects/sessions
  - Output saving in standardized format

### `save_to_torch.py`
- Functionality to save preprocessed EEG data in PyTorch format.
- This script processes the preprocessed data and saves it as `.pt` files, which can be easily loaded into PyTorch for further analysis or model training.
- **Usage**:
  - The script retrieves file paths of preprocessed EEG data, checks for existing epochs, and saves them in a specified directory.
  - It requires the following parameters:
    - `eeg_dir`: Directory containing the preprocessed EEG data.
    - `save_dir`: Directory where the `.pt` files will be saved.
  - The script can be run directly, and it will automatically handle the processing of all relevant files in the specified directories.

### `preprocess_and_save.py`
- Entry point for the preprocessing pipeline
- Configures preprocessing parameters:
  - Sampling frequency (500 Hz)
  - Line noise removal
  - Epoch length
  - Session and condition selection

## Usage
The pipeline is designed for the TDBRAIN dataset with specific channel configurations and preprocessing steps. Main parameters can be configured through a dictionary including:
- `derivatives_dir`: Input data directory
- `preprocessed_dir`: Output directory
- `condition`: EEG conditions to process (e.g., ['EO', 'EC'])
- `sessions`: Session numbers to process
- `epochs_length`: Length of epochs in seconds
- `sfreq`: Sampling frequency
- `line_noise`: Line noise frequencies to remove
- `plots`: Enable/disable visualization