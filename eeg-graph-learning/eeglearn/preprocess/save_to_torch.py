"""
Created on Thu Mar 11 2025

author: Udesh Habaraduwa
adapted from: https://github.com/TSmolders/DataScience_Thesis
description: save the preprocessed data to torch format

name: save_to_torch.py

version: 1.0

"""

import os
import torch
import numpy as np
import mne 
import random
from tqdm import tqdm
import importlib
from functools import partial
# prevent extensive logging
mne.set_log_level('WARNING')
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from eeglearn.preprocess.preprocessing import Preproccesing
from eeglearn.preprocess import preprocessing_plotting
from eeglearn.config import Config


def get_filepaths(eeg_dir : str, 
                  save_dir : str, 
                  recording_condition : list[str] = ['EC', 'EO'], 
                  session : str = 'ses-1') -> list[str]:
    """
    Retrieve file paths of EEG data for a specific recording condition.

    This function scans the specified directory for EEG data files that have not been processed yet
    and match the given recording condition and session.

    Args:
        eeg_dir (str): The directory where EEG data is stored.
        recording_condition (list, optional): List of recording conditions to include (e.g., ['EC', 'EO']). Defaults to ['EC', 'EO'].
        session (str, optional): The session identifier to filter files (e.g., 'ses-1'). Defaults to 'ses-1'.

    Returns:
        list: A list of file paths for the EEG data files that meet the specified criteria.
    """
    filepaths = []
    participant_ids = set()
    # Check for already processed participant ids
    processed_participant_ids = set([file.split('.')[0] for file in os.listdir(save_dir) if '.pt' in file])
    
    file_paths = [os.path.join(subdir, file) 
                  for subdir, _, files in os.walk(eeg_dir)
                  for file in files
                  if '.npy' in file 
                  and 'BAD' not in file 
                  and session in file
                  and any(condition in file for condition in recording_condition)
                  and file.split('_')[0] not in processed_participant_ids]
    return file_paths


def process_file(filepath : str, 
                 save_dir : str) -> None:
    try:
        participant_id = filepath.split('/')[-1].split('_')[0]
        condition = filepath.split('/')[-1].split('_')[2].split('-')[-1]
        eeg_data = np.load(filepath, allow_pickle=True)
        epochs = eeg_data.preprocessed_epochs.get_data()
        save_path = save_dir / f"{participant_id}_{condition}.pt"
        torch.save(epochs, save_path)
    except Exception as e:
        print(e)
        print(f"Error with {participant_id}")

# epoch data and save to disk
def preprocess_and_save_data(filepaths : list[str], 
                            save_dir : str, 
                            n_processes : int ) -> None:
    """
    Process and save multiple EEG data files to PyTorch format in parallel.
    
    Args:
        filepaths (list[str]): List of file paths to process
        save_dir (str): Directory where processed data will be saved
        n_processes (int): Number of parallel processes to use
        
    Returns:
        None: Files are saved to disk
    """
    os.makedirs(save_dir, exist_ok=True)
    process_file_with_save_dir = partial(process_file, save_dir=save_dir)
    
    with Pool(processes=n_processes) as pool:
       list(tqdm(pool.imap(process_file_with_save_dir, filepaths), total=len(filepaths), desc='Saving to torch...'))

if __name__ == '__main__':
    # Set seed for reproducibility 
    Config.set_global_seed()
    
    # Get the project root directory (2 levels up from this file)
    project_root = Path(__file__).resolve().parent.parent.parent

    # Define data directories using Path objects
    save_dir = project_root / 'data' / 'data_cleaned_torch'
    eeg_dir = project_root / 'data' / 'cleaned'

    # Ensure the directories exist
    save_dir.mkdir(parents=True, exist_ok=True)
    eeg_dir.mkdir(parents=True, exist_ok=True)

    filepaths = get_filepaths(eeg_dir, save_dir, recording_condition=['EC', 'EO'], session='ses-1')
    n_processes = cpu_count() - 1  # Leave one CPU core free
    preprocess_and_save_data(filepaths, save_dir, n_processes)