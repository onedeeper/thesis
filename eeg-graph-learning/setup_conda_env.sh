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
    Linux*)     OS_TYPE=Linux;;
    Darwin*)    OS_TYPE=Mac;;
    MINGW*)     OS_TYPE=Windows;;
    MSYS*)      OS_TYPE=Windows;;
    CYGWIN*)    OS_TYPE=Windows;;
    *)          OS_TYPE="UNKNOWN:${OS}"
esac

echo "Detected operating system: ${OS_TYPE} ✅"


if [ "$OS_TYPE" = "Windows" ]; then
    # For Windows (Git Bash, MSYS2, Cygwin)
    source "$(conda info --base)/etc/profile.d/conda.sh" > /dev/null 2>&1
else
    # For Linux and macOS
    eval "$(conda shell.bash hook)" > /dev/null 2>&1
fi


case "$OS_TYPE" in
  Mac)      YAML="environment.mac.yml"      ;;
  Linux)    YAML="environment.linux.yml"    ;;
  Windows)  YAML="environment.windows.yml"  ;;
  *)           echo "Unsupported OS ❌"; exit 1 ;;
esac


ENV_NAME="eeg-graph-learning-${OS_TYPE}"   # bash lowercase trick

# Check if the env exists (name matches file stem) 
if ! conda info --envs | grep -q "$ENV_NAME"; then
  echo "Creating $ENV_NAME from $YAML ... 📦"
  conda env create -f "$YAML" --name "$ENV_NAME" --quiet
else
  echo "Using existing $ENV_NAME conda environment 🔄"
fi

echo "Activating $ENV_NAME ... 🚀"
conda activate "$ENV_NAME" > /dev/null 2>&1


echo "Installing Jupyter kernel for this environment... 🎯"
python -m ipykernel install --user --name=eeg-graph-learning --display-name="Python (eeg-graph-learning)"


echo "Installing eeg-graph-learning package in development mode... 🛠️"
pip install -e .

echo "Setup complete! 🎉✅"
