#!/bin/bash

# clean.sh - Script to run the EEG preprocessing pipeline
# Author: Udesh Habaraduwa

# Exit on error
set -e

# Remove the command echoing (set -x) to reduce verbosity

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the project root directory
cd "$SCRIPT_DIR"

# Detect operating system
OS="$(uname -s)"
case "${OS}" in
    Linux*)     OS_TYPE=Linux;;
    Darwin*)    OS_TYPE=Mac;;
    MINGW*)     OS_TYPE=Windows;;
    MSYS*)      OS_TYPE=Windows;;
    CYGWIN*)    OS_TYPE=Windows;;
    *)          OS_TYPE="UNKNOWN:${OS}"
esac

echo "Detected operating system: ${OS_TYPE}"

# Set up conda environment based on OS
if [ "$OS_TYPE" = "Windows" ]; then
    # For Windows (Git Bash, MSYS2, Cygwin)
    source "$(conda info --base)/etc/profile.d/conda.sh" > /dev/null 2>&1
else
    # For Linux and macOS
    eval "$(conda shell.bash hook)" > /dev/null 2>&1
fi

# Check if the environment exists, create it if it doesn't
if ! conda env list | grep -q "eeg-graph-learning"; then
    echo "Creating conda environment from environment.yml..."
    conda env create -f environment.yml --quiet
else
    echo "Using existing eeg-graph-learning conda environment"
fi

# Activate the conda environment (suppress output)
echo "Activating conda environment..."
conda activate eeg-graph-learning > /dev/null 2>&1

# Install the IPython kernel for Jupyter
echo "Installing Jupyter kernel for this environment..."
python -m ipykernel install --user --name=eeg-graph-learning --display-name="Python (eeg-graph-learning)"

# Run the preprocessing pipeline
echo "Running preprocessing pipeline..."
if python -m eeglearn.preprocess.preprocess_pipeline; then
    echo "Preprocessing completed successfully!"
else
    echo "Error: Preprocessing failed with exit code $?"
    exit 1
fi