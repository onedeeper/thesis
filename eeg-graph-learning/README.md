# EEG Graph Learning

Graph-based machine learning for EEG data analysis using self-supervised learning approaches.

## Project Overview

This repository contains the implementation of graph-based machine learning models for self-supervised pre-training on EEG data. The goal is to develop robust representations of EEG signals that can be used for downstream tasks such as psychiatric condition classification.

## Installation

### Clone the repository
`git clone https://github.com/yourusername/eeg-graph-learning.git `  \

`cd eeg-graph-learning`

###  Create and activate the conda environment
`conda env create -f environment.yml` \
`conda activate eeg-graph-learning`

Install the package in development mode \
`pip install -e .`


## Data Setup

This project uses the TD-Brain dataset, which requires a Data Usage Agreement (DUA).

1. Apply for access to the TD-Brain dataset : https://www.brainclinics.com/resources/tdbrain-dataset
2. Once approved, download the dataset
3. Place the data in the `data/` directory following the structure below:

![Alt text](data_dir_structure.png)

## Testing

### Test Configuration

The test suite can use either synthetic data (generated automatically) or your own EEG data files for testing. By default, it will create synthetic test data, but you can configure it to use your own EEG data files.

#### Using Custom Test Data

To use your own EEG data file for testing, set the `EEG_TEST_FILE_PATH` environment variable to the path of your test file:

```bash
# Bash/Zsh
export EEG_TEST_FILE_PATH="/path/to/your/eeg/test/file.csv"

# Windows Command Prompt
set EEG_TEST_FILE_PATH=C:\path\to\your\eeg\test\file.csv

# Windows PowerShell
$env:EEG_TEST_FILE_PATH="C:\path\to\your\eeg\test\file.csv"
```

The test file should be a CSV file with the following characteristics:
- Channels as columns
- Time points as rows
- 33 channels (26 EEG + 7 other)
- Sampling frequency of 500 Hz

If `EEG_TEST_FILE_PATH` is not set or the file doesn't exist, the test suite will automatically generate synthetic test data.

#### Running Tests

To run the test suite:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_preprocessing.py

# Run tests with verbose output
pytest -v
```

