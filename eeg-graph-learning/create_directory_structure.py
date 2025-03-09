import os
import pathlib

def create_directory_structure(base_dir="."):
    """Create the EEG Graph Learning project directory structure."""
    
    # Define the directory structure
    directories = [
        "data",
        "eeglearn",
        "eeglearn/features",
        "eeglearn/dataloader",
        "eeglearn/models",
        "eeglearn/pretext",
        "eeglearn/downstream",
        "eeglearn/utils",
        "scripts",
        "tests",
        "experiments/configs",
        "experiments/results",
        "notebooks/examples"
    ]
    
    # Define files to create with empty content
    empty_files = [
        "README.md",
        "setup.py",
        "requirements.txt",
        "environment.yml",
        "Makefile",
        ".gitignore",
        "data/.gitkeep",
        "data/README.md",
        "eeglearn/__init__.py",
        "eeglearn/config.py",
        "eeglearn/features/__init__.py",
        "eeglearn/features/base.py",
        "eeglearn/features/spectrum.py",
        "eeglearn/dataloader/__init__.py",
        "eeglearn/dataloader/dataset.py",
        "eeglearn/dataloader/generator.py",
        "eeglearn/models/__init__.py",
        "eeglearn/pretext/__init__.py",
        "eeglearn/downstream/__init__.py",
        "eeglearn/utils/__init__.py",
        "eeglearn/utils/io.py",
        "eeglearn/utils/preprocessing.py",
        "scripts/setup_data_paths.py",
        "tests/__init__.py",
        "tests/test_features.py"
    ]
    
    # Create base directory if it doesn't exist
    base_path = pathlib.Path(base_dir)
    if not base_path.exists():
        base_path.mkdir(parents=True)
    
    # Create directories
    for directory in directories:
        dir_path = base_path / directory
        if not dir_path.exists():
            print(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create empty files
    for file in empty_files:
        file_path = base_path / file
        if not file_path.exists():
            print(f"Creating file: {file_path}")
            file_path.touch()
    
    # Add basic content to .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Data directories
/data/*
!/data/.gitkeep
!/data/README.md

# Jupyter Notebooks
.ipynb_checkpoints

# VS Code
.vscode/

# Experiment results
/experiments/results/

# Environment
.env
.venv
env/
venv/
ENV/
"""
    
    with open(base_path / ".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("Project structure created successfully!")

if __name__ == "__main__":
    create_directory_structure()
