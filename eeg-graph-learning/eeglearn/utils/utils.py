
"""Created on Thu Mar 11 2025.

author: Udesh Habaraduwa
description: utility functions used throughout the project

name: utils.py

version: 1.0

"""

import itertools
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import torch

def get_participant_id_condition_from_string(file_path: str) -> tuple[str, str]:
    """Extract the participant ID and condition from the file path.
    
    Args:
    ----
        file_path: Path to the file containing participant ID and condition
        
    Returns:
    -------
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
    ----
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

    This function exists mostly for enabling the parallel processing of data. 
    For example when computing the spectrum or energy of the data
    with run_spectrum_parallel() and run_energy_parallel()

    Args:
    ----
        participant_list: List of participant IDs to load
        cleaned_path: Path to the cleaned data
        
    Returns:
    -------
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

def hamming_set(n_regions : int,
                n_permutations : int,
                selection : str,
                output_file_name : str,
                save_to_disk : bool = True):
    """Generate and save the hamming set.

    A set of permutations for the regions that are maximally different from each 
    other. Each permutation is only added to the collection if it is on average
    furthest away from the permutations in the collection already. 

    Args:
    ----
        n_regions: number of different brain regions to be shuffled
        n_permutations : Number of permutations of all possible to collected
        selection : "max" or "median" hamming distance. Renamed to be median instead
                    of mean from the authors, to make the intention of the selection 
                    clearer.
        output_file_name : File name to save, showing the number of regions and
                           permutations
        save_to_disk : if the results should be saved to disk.
        
    Returns:
    -------
        np.Array : A numpy array of shape (n_permutations x regions), 

    Implementation based on Li et al. (2021) GMSS paper.
    Citation: Li, Y., Chen, J., Li, F., et al. (2021). GMSS: Graph-Based 
    Multi-Task Self-Supervised Learning for EEG Emotion Recognition.

    As no significant changes were made, outside of renaming some functions for clarity,
    testing was not implemented. 

    """
    assert n_regions > 0
    assert  n_permutations > 0
    assert selection == "max" or selection == "median"
    # get permutations of size 10 for the given n_regions.
    # shape: 10! x n_regions
    all_perms : np.array = np.array(list(itertools.permutations(list(range(n_regions)),
                                                     n_regions)))
    n_total_perms : int = all_perms.shape[0]
    j : int # a permutation 
    for i in range(n_permutations):
        if i == 0:
            # for the first sample, pick a random permutation from ~3.6 million
            j : int = np.random.randint(n_total_perms)
            # Add the jth permutation to the collection
            collection : np.array = np.array(all_perms[j]).reshape([1, -1])
        else:
            collection : np.array = np.concatenate([collection, all_perms[j].\
                                                    reshape([1, -1])],
                                        axis=0)
        # remove the selected permutation from the remaining ones
        all_perms = np.delete(all_perms, j, axis=0)
        # Find the remaining permutation that is on average most distant from the
        # collection.
        distances : np.array = cdist(collection, all_perms, metric='hamming')\
            .mean(axis=0).flatten()
        if selection == 'max':
            j = distances.argmax()
        elif selection == 'median':
            # get the middle permutation is at m
            median : int = int(distances.shape[0] / 2)
            # The indices that would sort D
            indices : int = distances.argsort()
            # pick a random permutation half +/- 10 from m 
            j = indices[np.random.randint(median - 10, median + 10)]

    assert collection.shape[0] == n_permutations
    assert collection.shape[1] == n_regions
    if save_to_disk:
        torch.save(f'max_hamming_set_{n_regions}_{n_permutations}.pt', torch.Tensor\
                                                                        (collection))
    return collection