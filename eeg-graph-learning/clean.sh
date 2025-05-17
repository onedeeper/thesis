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
    Linux*)     OS_TYPE=linux;;
    Darwin*)    OS_TYPE=mac;;
    MINGW*)     OS_TYPE=windows;;
    MSYS*)      OS_TYPE=windows;;
    CYGWIN*)    OS_TYPE=windows;;
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

ENV_NAME="eeg-graph-learning-test"
# Check if the environment exists first
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "🐍 Setting up conda environment..."
    bash setup_conda_env.sh
else
    echo "🐍 Using existing $ENV_NAME conda environment"
    # Activate the conda environment (suppress output)
    echo "Activating conda environment..."
fi

conda activate "$ENV_NAME" > /dev/null 2>&1
# Install the IPython kernel for Jupyter
echo "Installing Jupyter kernel for this environment..."
python -m ipykernel install --user --name=$ENV_NAME --display-name="Python ($ENV_NAME)"

# Run the preprocessing pipeline
echo "Running preprocessing pipeline..."
if python -m eeglearn.preprocess.preprocess_pipeline; then
    echo "👍 Done."
else
    echo "Error: Preprocessing failed with exit code $?"
    exit 1
fi