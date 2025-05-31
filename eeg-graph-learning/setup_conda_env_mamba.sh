#!/bin/bash

# clean.sh - Script to run the EEG preprocessing pipeline 
# Author: Udesh Habaraduwa 👨‍💻
# Written with AI 

# Exit on error 
set -e


# Get the directory where this script is located 
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the project root directory 🏠
cd "$SCRIPT_DIR"


OS="$(uname -s)"
case "${OS}" in
    Linux*)     OS_TYPE=linux;;
    Darwin*)    OS_TYPE=mac;;
    MINGW*)     OS_TYPE=windows;;
    MSYS*)      OS_TYPE=windows;;
    CYGWIN*)    OS_TYPE=windows;;
    *)          OS_TYPE="UNKNOWN:${OS}"
esac

echo "Detected operating system: ${OS_TYPE} ✅"


if [ "$OS_TYPE" = "windows" ]; then
    # For Windows (Git Bash, MSYS2, Cygwin)
    source "$(conda info --base)/etc/profile.d/conda.sh" > /dev/null 2>&1
else
    # For Linux and macOS
    eval "$(conda shell.bash hook)" > /dev/null 2>&1
fi


ENV_NAME="eeg-graph-learning-2"
LOCKED_YAML="environment.lock.yml"
FALLBACK_YAML="environment.yml"

# Check if the env exists (name matches file stem) 
if ! conda info --envs | grep -q "$ENV_NAME"; then
  # First try the locked YML file
  if [ -f "$LOCKED_YAML" ]; then
    echo "Creating $ENV_NAME from locked file $LOCKED_YAML ... 📦"
    if mamba env create -f "$LOCKED_YAML" --name "$ENV_NAME" --quiet; then
      echo "Successfully created environment from locked file! ✅"
    else
      echo "Failed to create environment from locked file, falling back to $FALLBACK_YAML ... ⚠️"
      mamba env create -f "$FALLBACK_YAML" --name "$ENV_NAME" --quiet
    fi
  else
    echo "No locked YML file found, creating $ENV_NAME from $FALLBACK_YAML ... 📦"
    mamba env create -f "$FALLBACK_YAML" --name "$ENV_NAME" --quiet
  fi
else
  echo "Using existing $ENV_NAME conda environment 🔄"
fi

echo "Activating $ENV_NAME ... 🚀"
conda activate "$ENV_NAME" > /dev/null 2>&1

echo "Installing Jupyter kernel for this environment... 🎯"
python -m ipykernel install --user --name=eeg-graph-learning --display-name="Python (eeg-graph-learning)"


echo "Installing eeg-graph-learning package in development mode... 🛠️"
pip install -e . --no-deps

echo "Setup complete! 🎉✅"