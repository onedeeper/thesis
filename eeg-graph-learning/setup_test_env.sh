#!/bin/bash

# Set up test environment variables for EEG Graph Learning project
# Run this script with: source setup_test_env.sh

# Path to test EEG file
export EEG_TEST_FILE_PATH="/Users/udeshhabaraduwa/Library/CloudStorage/GoogleDrive-u.habaraduwakandambige@tilburguniversity.edu/My Drive/Tilburg/Masters CSAI/Semester 4/code/thesis/eeg-graph-learning/tests/test_data/TDBRAIN-dataset/derivatives/sub-88046393/ses-1/eeg/sub-88046393_ses-1_task-restEC_eeg.csv"

# Path to preprocessed test file
export EEG_CLEANED_TEST_FILE="/Users/udeshhabaraduwa/Library/CloudStorage/GoogleDrive-u.habaraduwakandambige@tilburguniversity.edu/My Drive/Tilburg/Masters CSAI/Semester 4/code/thesis/eeg-graph-learning/tests/test_data/data/cleaned/sub-19740274/ses-1/eeg/sub-19740274_ses-1_task-restEC_preprocessed.npy"

# Path to folder containing cleaned test data
export EEG_TEST_CLEANED_FOLDER_PATH="/Users/udeshhabaraduwa/Library/CloudStorage/GoogleDrive-u.habaraduwakandambige@tilburguniversity.edu/My Drive/Tilburg/Masters CSAI/Semester 4/code/thesis/eeg-graph-learning/tests/test_data/data/cleaned"

# Path to the derivatives directory
export EEG_TEST_DERIVATIVES_DIR="/Users/udeshhabaraduwa/Library/CloudStorage/GoogleDrive-u.habaraduwakandambige@tilburguniversity.edu/My Drive/Tilburg/Masters CSAI/Semester 4/code/thesis/eeg-graph-learning/tests/test_data/TDBRAIN-dataset/derivatives"

# Print confirmation message
echo "Test environment variables have been set:"
echo "EEG_TEST_FILE_PATH = $EEG_TEST_FILE_PATH"
echo "EEG_CLEANED_TEST_FILE = $EEG_CLEANED_TEST_FILE"
echo "EEG_TEST_CLEANED_FOLDER_PATH = $EEG_TEST_CLEANED_FOLDER_PATH"
echo "EEG_TEST_DERIVATIVES_DIR = $EEG_TEST_DERIVATIVES_DIR"

# Reminder message
echo ""
echo "NOTE: This script must be run with 'source' for the variables to persist in your current shell:"
echo "      source setup_test_env.sh"
echo ""
echo "If you ran it with './setup_test_env.sh', the variables won't be available in your shell." 