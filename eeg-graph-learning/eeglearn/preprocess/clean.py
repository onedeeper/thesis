import os
import sys
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import close
from pathlib import Path
from multiprocessing import Pool, cpu_count
from eeglearn.preprocess.preprocessing import Preproccesing  # Fixed import statement
from eeglearn.preprocess.save_to_torch import get_filepaths, preprocess_and_save_data
import random
from collections import namedtuple

# namedtuple to store preprocessing parameters to be sent to process_file function when using multiprocessing
Preprocessing_params = namedtuple('Preprocessing_params', ['csv_path', 'participant', 'session', 'condition', 'epochs_length', 'line_noise', 'sfreq', 'plots', 'preprocessed_dir'])

def process_file(args : tuple) -> tuple[str, str, str]:
    """
    Process a single file in the TDBRAIN dataset.

    This function takes a tuple of arguments, unpacks them, and processes the file
    specified by the filepath. It creates a Preprocessing object, saves the preprocessed
    data to a .npy file, and optionally saves preprocessing plots to a .pdf file.

    Args:
        args (tuple): A tuple of arguments. The elements of the tuple are:
            - filepath (str): The path to the file to be processed.
            - ID (str): The subject ID.
            - sessID (str): The session ID.
            - cond (str): The condition.
            - epochs_length (float): The length of epochs in seconds.
            - line_noise (list): A list of frequencies for line noise removal.
            - sfreq (float): The sampling frequency in Hz.
            - plots (bool): Whether to generate and save preprocessing plots.
            - preprocessed_dir (str): The path to the directory where preprocessed data will be saved.

    Returns:
        tuple: A tuple containing the subject ID, session ID, and condition.

    Raises:
        FileNotFoundError: If the file specified by the filepath does not exist.
        PermissionError: If there are insufficient permissions to read/write directories.
    """
    
    filepath : str = args[0]
    assert os.path.exists(filepath), f"File does not exist: {filepath}"
    Id : str = args[1]
    sessID : str = args[2]
    cond : str = args[3]
    epochs_length : float = args[4]
    line_noise : list[int] = args[5]
    sfreq : float = args[6]
    plots : bool = args[7]
    preprocessed_dir : str = args[8]
    # Create Preprocessing object
    preprocessed_data = Preproccesing(
        filepath,
        epochs_length,
        line_noise,
        sfreq,
        plots
    )
    
    # Define directory and subdirectories for preprocessed data
    save_dir = Path(preprocessed_dir) / Id / sessID / 'eeg'
    save_path_data = save_dir / f'{Id}_{sessID}_{cond}_preprocessed.npy'
    
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(save_path_data), exist_ok=True)
    
    if plots == True:
        save_path_plots = save_dir / f'{Id}_{sessID}_{cond}_preprocessing_plots.pdf'
        # Save plots in pdf file
        figs = preprocessed_data.figs
        with PdfPages(save_path_plots) as pdf:
            for fig in figs:
                pdf.savefig(fig)
        close('all')  # Closes all figures for memory management
        delattr(preprocessed_data, 'figs')  # Delete figs attribute, because cannot be pickled
        assert os.path.exists(save_path_plots), f"Plots file does not exist: {save_path_plots}"
    # Store preprocessed data object as .npy file
    with open(save_path_data, 'wb') as output:
        pickle.dump(preprocessed_data, output, pickle.HIGHEST_PROTOCOL)
    assert os.path.exists(save_path_data), f"Data file does not exist: {save_path_data}"
    
    
    return Id, sessID, cond  # Return identifiers for tracking

def get_file_details(subjects : list[str], sessions : list[str], conditions : list[str], derivates_dir : str, preprocessed_dir : str, epochs_length : float, line_noise : np.ndarray, sfreq : float, plots : bool) -> list[tuple[str, str , str, str, float, np.ndarray, float, bool, str]]:
    """
    Get the file details for the preprocessing function.
    args:
        subjects (list): list of subjects to process
        sessions (list): list of sessions to process
        conditions (list): list of conditions to process
        derivates_dir (str): path to the derivatives directory
        preprocessed_dir (str): path to the preprocessed directory
        epochs_length (float): length of epochs to process
        line_noise (np.ndarray): list of frequencies to process
        sfreq (float): sampling frequency
        plots (bool): whether to plot the data
    Returns:
        list: list of tuples containing the file details
    """
    project_root = Path(__file__).parent.parent.parent
    log_file_path = project_root / 'data' / 'preprocessing_log.txt'
    file_details : list[tuple[str, str , str, str, float, np.ndarray, float, bool, str]] = []
    assert len(subjects) > 0, "No subjects to process"
    #print(subjects)
    for subject in subjects:
        for session in sessions:
            for condition in conditions:
                csv_path  = os.path.join(derivates_dir, 
                                                            subject, 
                                                            session, 
                                                            'eeg', 
                                                            f'{subject}_{session}_task-rest{condition}_eeg.csv')
                if not os.path.exists(csv_path):
                    os.makedirs('data', exist_ok=True)
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(f"File does not exist: {csv_path}\n")
                    continue
                file_details.append(Preprocessing_params(csv_path, subject, session, condition, epochs_length, line_noise, sfreq, plots, preprocessed_dir))

    assert len(file_details) > 0, "No files to process"
    return file_details


def clean_pipeline(derivates_dir : str,preprocessed_dir : str,
                   sfreq : float,
                   epochs_length : float = 0, 
                   line_noise : list[int] = np.array([]),
                   n_processes : int = cpu_count() - 1,
                   num_samples : int = 0,
                   conditions : list[str] = ['EO', 'EC'],
                   sessions : list[str] = ['ses-1'],
                   plots : bool = False,
                   exclude_dirs : list[str] = [],
                   verify_output : bool = True) -> None:
    
    """Preprocesses EEG data from the TDBRAIN dataset

    This pipeline handles batch processing of EEG data files, applying preprocessing steps
    including PREP pipeline, ICA artifact correction, and optional epoching. It supports
    parallel processing for improved performance and can handle interruptions by tracking
    processed subjects.

    Args:
        derivates_dir (str): Path to the directory containing the derivatives folder with raw EEG data.
        preprocessed_dir (str): Path to the directory where preprocessed data will be saved.
        sfreq (float): Sampling frequency in Hz.
        epochs_length (float, optional): Length of epochs in seconds. If 0, no epoching is performed.
            Defaults to 0.
        line_noise (list, optional): List of frequencies for line noise removal.
            Empty list means no line noise removal. Defaults to [].
        n_processes (int, optional): Number of parallel processes to use.
            Defaults to number of CPU cores.
        num_samples (int, optional): Number of random samples to process. If 0, process all files.
            Defaults to 0.
        conditions (list, optional): List of conditions to process (e.g., ['EO', 'EC']).
            Defaults to ['EO', 'EC'].
        sessions (list, optional): List of sessions to process (e.g., ['ses-1']).
            Defaults to ['ses-1'].
        plots (bool, optional): Whether to generate and save preprocessing plots.
            Defaults to False.
        exclude_dirs (list, optional): List of directory names to exclude from processing.
            Defaults to [].
        verify_output (bool, optional): Whether to verify the output of the preprocessing.
            Defaults to True.
    Returns:
        None: Results are saved to disk in the specified preprocessed_dir.
            Each subject's data is saved as a .npy file with the following structure:
            {preprocessed_dir}/{subject_id}/{session}/eeg/{subject_id}_{session}_{condition}_preprocessed.npy
    """
    
    assert os.path.isdir(derivates_dir), f"Derivatives directory does not exist: {derivates_dir}"
    assert os.path.isdir(preprocessed_dir), f"Preprocessed directory does not exist: {preprocessed_dir}"
    assert sfreq > 0, "Sampling frequency must be greater than 0"
    assert epochs_length > 0, "Epochs length must be greater than 0"
    # line_noise must be an empty numpy array or integers     
    assert isinstance(line_noise, np.ndarray), "Line noise must be a numpy array"
    assert np.all(np.isfinite(line_noise)) == True, "Line noise must be a numeric array"
    # check for conditions
    assert isinstance(conditions, list), "Conditions must be a list"
    if len(conditions) > 0:
        # if anything else than EC or EO is in the list, raise an error
        assert len(conditions) <= 2, "There are only 2 conditions in the TDBRAIN dataset: 'EC' and 'EO'"
        assert all(condition in ['EC', 'EO'] for condition in conditions), "Conditions must contain 'EC' or 'EO'"
    # check for sessions
    assert isinstance(sessions, list), "Sessions must be a list"
    assert len(sessions) <= 2, "There are only 2 sessions in the TDBRAIN dataset: 'ses-1' and 'ses-2'"
    assert all(session in ['ses-1', 'ses-2'] for session in sessions), "Sessions must contain 'ses-1' or 'ses-2'"
    # check for exclude_dirs
    assert isinstance(exclude_dirs, list), "Exclude directories must be a list"
    

    # In case of crashes/interruptions, starting from the last subject in the preprocessed directory
    all_subjects : list[str] = os.listdir(derivates_dir)
    already_processed_subjects : list[str] = sorted([s for s in all_subjects if os.path.isdir(os.path.join(preprocessed_dir, s))])

    # Collect all files to process
    # this should just be the files that are not in the already_processed_subjects list
    subjects_to_process : set[str] = set(all_subjects) - set(already_processed_subjects)

    if not subjects_to_process:
        print("No subjects to process. Exiting.")
        return

    N_subjs_left_to_process :int  = len(all_subjects) - len(already_processed_subjects)
    assert N_subjs_left_to_process > 0, "Existing files should have been replaced"

    print(f"Found {N_subjs_left_to_process} subjects to process")
    if  num_samples:
        random_subjects : list[str] = random.sample(list(subjects_to_process), num_samples)
        print(f"Processing {num_samples} random subjects --> {len(random_subjects) * len(sessions) * len(conditions)} files")
        files_to_process = get_file_details(random_subjects, sessions, conditions, derivates_dir, preprocessed_dir, epochs_length, line_noise, sfreq, plots)
        assert len(files_to_process) == len(random_subjects)  * len(sessions) * len(conditions), "Total number of CSVS not as expected"
    else: 
        print(f"Processing all remaining {len(subjects_to_process)} subjects")
        files_to_process = get_file_details(subjects_to_process, sessions, conditions, derivates_dir, preprocessed_dir, epochs_length, line_noise, sfreq, plots)
        assert len(files_to_process) == N_subjs_left_to_process * len(sessions) * len(conditions), "Total number of CSVS not as expected"
    
    with Pool(processes=n_processes) as pool:
        list(tqdm(pool.imap(process_file, files_to_process), 
                           total=len(files_to_process), 
                           desc="Processing files.."))
    if verify_output:
        assert len(os.listdir(preprocessed_dir)) >= len(files_to_process) / len(sessions) / len(conditions), "Not all files have been processed"

if __name__ == '__main__':
     # Get the project root directory (2 levels up from this file)
    project_root = Path(__file__).parent.parent.parent

    # Use Path to join paths correctly
    derivates_dir = str(project_root / 'data' / 'TDBRAIN-dataset' / 'derivatives')
    print(f'Reading data from: {derivates_dir}')

    # if cleaned folder does not exist, create it
    if not os.path.exists(str(project_root / 'data' / 'cleaned')):
        os.makedirs(str(project_root / 'data' / 'cleaned'), exist_ok=True) 

    preprocessed_dir = str(project_root / 'data' / 'cleaned')
    print(f'Writing preprocessed data to: {preprocessed_dir}')

    # Create the output directory if it doesn't exist
    os.makedirs(preprocessed_dir, exist_ok=True)

    # the following parameters can be changed by the user
    conditions = ['EO', 'EC'] # conditions to be preprocessed
    sessions = ['ses-1',] # sessions to be preprocessed
    epochs_length = 9.95 # length of epochs in seconds, comment out for no epoching
    sfreq = 500 # sampling frequency
    line_noise = np.arange(50, sfreq / 2, 50) # 50 Hz line noise removal, comment out for no line noise removal
    plots = True # set to True to create and store plots during preprocessing
    n_processes = 4 # number of processes to use for parallel processing
    num_samples = 4 # number of samples to process, set to 0 for all samples
    clean_pipeline(derivates_dir = derivates_dir,
                        preprocessed_dir = preprocessed_dir,
                        sfreq = sfreq,
                        epochs_length = epochs_length,
                        line_noise = line_noise,
                        conditions = conditions,
                        sessions = sessions,
                        plots = plots,
                        num_samples = 2)
    