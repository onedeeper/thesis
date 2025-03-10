# EEG Preprocessing Module

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