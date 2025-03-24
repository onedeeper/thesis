#!/bin/bash

# Set up test environment variables for EEG Graph Learning project
# Run this script with: source setup_test_env.sh

# Path to test EEG file - REPLACE WITH YOUR OWN PATH
export EEG_TEST_FILE_PATH="/path/to/your/test/data/sub-XXXXXXXX/ses-X/eeg/sub-XXXXXXXX_ses-X_task-restEC_eeg.csv"

# Path to preprocessed test file - REPLACE WITH YOUR OWN PATH
export EEG_CLEANED_TEST_FILE="/path/to/your/data/cleaned/sub-XXXXXXXX/ses-X/eeg/sub-XXXXXXXX_ses-X_task-restEC_preprocessed.npy"

# Path to folder containing cleaned test data - REPLACE WITH YOUR OWN PATH
export EEG_TEST_CLEANED_FOLDER_PATH="/path/to/your/data/cleaned"

# Path to the derivatives directory - REPLACE WITH YOUR OWN PATH
export EEG_TEST_DERIVATIVES_DIR="/path/to/your/test/data/TDBRAIN-dataset/derivatives"

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