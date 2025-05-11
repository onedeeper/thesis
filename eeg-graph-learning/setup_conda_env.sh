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


case "$OS_TYPE" in
  mac)      YAML="environment.mac.yml"      ;;
  linux)    YAML="environment.linux.yml"    ;;
  windows)  YAML="environment.windows.yml"  ;;
  *)           echo "Unsupported OS ❌"; exit 1 ;;
esac


ENV_NAME="eeg-graph-learning-${OS_TYPE}"  

# Check if the env exists (name matches file stem) 
if ! conda info --envs | grep -q "$ENV_NAME"; then
  echo "Creating $ENV_NAME from $YAML ... 📦"
  conda env create -f "$YAML" --name "$ENV_NAME" --quiet
else
  echo "Using existing $ENV_NAME conda environment 🔄"
fi

echo "Activating $ENV_NAME ... 🚀"
conda activate "$ENV_NAME" > /dev/null 2>&1

# Determine CUDA availability and version
CUDA="cpu"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.split('.')[0])")
  if [ "$CUDA_VERSION" = "11" ]; then
    CUDA="cu118"
  elif [ "$CUDA_VERSION" = "12" ]; then
    CUDA="cu121"
  fi
  echo "CUDA is available. Using $CUDA version."
else
  echo "CUDA is not available. Using CPU version."
fi

# Install PyG dependencies with the appropriate CUDA version
echo "Installing pytorch-scatter and pytorch-sparse"
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+${CUDA}.html

echo "Installing Jupyter kernel for this environment... 🎯"
python -m ipykernel install --user --name=eeg-graph-learning --display-name="Python (eeg-graph-learning)"


echo "Installing eeg-graph-learning package in development mode... 🛠️"
pip install -e .

echo "Setup complete! 🎉✅"
