
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
from eeglearn.preprocess.preprocessing import Preproccesing
from eeglearn.config import Config
def get_details_from_file_name(file_path: str) -> tuple[str, str]:
    """Extract the participant specific information from a string.
    
    Args:
    ----
        file_path: Path to the file containing participant ID and condition
        
    Returns:
    -------
        tuple: A tuple containing (participant_id, condition, session)

    """
    participant_match : re.Match = re.search(r'(sub-\d+)(?=_)', file_path)
    participant_id : str = participant_match.group(1) if participant_match else None
    
    condition_match = re.search(r'(E[CO])', file_path)
    condition : str = condition_match.group(1) if condition_match else None

    session_match :re.Match =  re.search(r'(ses-[12])', file_path)
    session : str = session_match.group(1) if session_match else None 
    return participant_id, condition, session

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
    participant_labels = labels_df['indication'].astype(str)
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
    sessions = ['ses-1', 'ses-2']
    for participant in participant_list:
        for session in sessions:
            participant_folder = Path(cleaned_path) / participant / session / 'eeg'
            if not (os.path.exists(participant_folder)):
                continue
            try:
                for file in os.listdir(participant_folder):
                    if file.endswith('.npy'):
                        participant_npy_files.append(file)
                        folders_and_files.append((participant_folder, file))
            except FileNotFoundError as e:
                raise RuntimeError(f"participant_folder not found for {participant}")\
                    from e
                    
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

def load_preprocessed_data(folder_path : Path , file_name : str) -> Preproccesing:
    """Load a preprocesed file.

    Args:
    ----
        folder_path : Full folder information leadning to a Preprocessing object
        file_name : A file name which includes information about the participant,
                    session, and condition.
        
    Returns:
    -------
        Preprocessing : All the details relevant to a given participant file after
                        compeleting the preprocessing pipeline.
    """
    try:
        data : Preproccesing = np.load(folder_path / file_name, allow_pickle=True)
    except FileNotFoundError:
        print(f'File {file_name} not found')
        return None, None, None
    assert isinstance(data, Preproccesing), "data is not a Preproccesing object"
    return data

def get_bad_channels(cleaned_data_path) -> dict[str, list[str]]:
        """Retrieve the bad channels from preprocessed data.
        
        First checks if a cached bad channels file exists. If it does, loads and returns it.
        Otherwise, creates the bad channels dictionary by processing the preprocessed files,
        saves it to disk, and returns it.
        
        Args:
        ----
            None.
        Returns:
        ----
            bad_channels : dict[str,list[str]] : A dictionary keyed by file_id, with
                                            list of strings indicating which channels
                                            are bad in the file.
        """
        bad_channels_path = Path(Config.data_path) / "bad_channels.pt"
        
        # Return cached version if it exists
        if bad_channels_path.exists():
            print("⚠️ Loading cached bad channels from disk...")
            return torch.load(bad_channels_path)
            
        print("⚠️ No cached bad channels found. Computing from scratch...")
        # Otherwise create the bad channels dictionary
        participant_list : list[str] = os.listdir(cleaned_data_path)
        folders_and_files : list[str, str] 
        folders_and_files, _ = \
            get_cleaned_data_paths(participant_list=participant_list,
                                   cleaned_path=cleaned_data_path)
        assert len(folders_and_files) > 0, "At least one cleaned file should exist."
        
        bad_channels : dict[str, list[str]] = {}
        for file in folders_and_files:
            folder_path : str = file[0]
            preprocessed_file_name : str = file[1]
            prep_data = load_preprocessed_data(folder_path=folder_path,
                                   file_name=preprocessed_file_name)
            assert isinstance(prep_data, Preproccesing),"Should be a Preprocessing obj"
            bads = prep_data.bad_channels_after_interpolation['bad_all']
            # some bad channels are returned as np.str_ class
            bad_channels[preprocessed_file_name] = [str(item) for item in bads]
            
        print("💾 Caching bad channels to disk...")
        # Cache the bad channels dictionary
        torch.save(bad_channels, bad_channels_path)
        
        return bad_channels