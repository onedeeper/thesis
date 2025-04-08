
"""Created on Thu Mar 11 2025.

author: Udesh Habaraduwa
description: utility functions used throughout the project

name: utils.py

version: 1.0

"""

import os
import re
from pathlib import Path

import pandas as pd


def get_participant_id_condition_from_string(file_path: str) -> tuple[str, str]:
    """Extract the participant ID and condition from the file path.
    
    Args:
        file_path: Path to the file containing participant ID and condition
        
    Returns:
        tuple: A tuple containing (participant_id, condition)

    """
    # Extract the participant ID and condition from the file path

    participant_match = re.search(r'(sub-\d+)(?=_)', file_path)
    participant_id = participant_match.group(1) if participant_match else None
    
    # Extract condition (restEC or restEO) from the task- portion
    condition_match = re.search(r'(E[CO])', file_path)
    condition = condition_match.group(1) if condition_match else None

    return participant_id, condition

def get_labels_dict() -> dict[str, str]:
    """Get indications for each participant from the details excel file.

    Args:
        None
    Returns:
        dict: A dictionary mapping participant IDs to their indications

    """
    # find the path to the labels file data
    labels_file = Path(__file__).resolve().parent.parent.parent \
        / 'data' / 'TDBRAIN_participants_V2.xlsx'
    labels_df = pd.read_excel(labels_file)
    participant_ids = labels_df['participants_ID']
    participant_labels = labels_df['indication']
    return dict(zip(participant_ids, participant_labels))

def get_cleaned_data_paths(participant_list : list[str], cleaned_path : str) ->\
      tuple[list[tuple[Path, str]], list[str]]:
    """Load the cleaned data from the disk.
    This function exists mostly for enabling the parallel processing of data , 
    for example when computing the spectrum or energy of the data
    with run_spectrum_parallel() and run_energy_parallel()

    Args:
        participant_list: List of participant IDs to load
        cleaned_path: Path to the cleaned data
        
    Returns:
        tuple: A tuple containing (folders_and_files, participant_npy_files)

    """
    assert os.path.exists(cleaned_path), f"cleaned_path does not exist: {cleaned_path}"
    assert len(participant_list) > 0, "participant_list is empty"

    folders_and_files : list[tuple[Path, str]] = []
    participant_npy_files : list[str] = []
    for participant in participant_list:
        participant_folder = Path(cleaned_path) / participant / 'ses-1' / 'eeg'
        try:
            for file in os.listdir(participant_folder):
                if file.endswith('.npy'):
                    participant_npy_files.append(file)
                    folders_and_files.append((participant_folder, file))
        except FileNotFoundError as e:
            raise RuntimeError(f"participant_folder not found for {participant}") from e
                
    assert len(participant_npy_files) > 0, "No .npy files found in cleaned_path"
    return folders_and_files, participant_npy_files
