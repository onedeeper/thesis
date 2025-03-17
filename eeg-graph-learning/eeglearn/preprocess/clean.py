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

def process_file(args):
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
    
    filepath, ID, sessID, cond, epochs_length, line_noise, sfreq, plots, preprocessed_dir = args
    
    
    # Create Preprocessing object
    preprocessed_data = Preproccesing(
        filepath,
        epochs_length,
        line_noise,
        sfreq,
        plots
    )
    
    # Define directory and subdirectories for preprocessed data
    save_dir = Path(preprocessed_dir) / ID / sessID / 'eeg'
    save_path_data = save_dir / f'{ID}_{sessID}_{cond}_preprocessed.npy'
    
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(save_path_data), exist_ok=True)
    
    if plots == True:
        save_path_plots = save_dir / f'{ID}_{sessID}_{cond}_preprocessing_plots.pdf'
        # Save plots in pdf file
        figs = preprocessed_data.figs
        with PdfPages(save_path_plots) as pdf:
            for fig in figs:
                pdf.savefig(fig)
        close('all')  # Closes all figures for memory management
        delattr(preprocessed_data, 'figs')  # Delete figs attribute, because cannot be pickled
    
    # Store preprocessed data object as .npy file
    with open(save_path_data, 'wb') as output:
        pickle.dump(preprocessed_data, output, pickle.HIGHEST_PROTOCOL)
    
    
    return ID, sessID, cond  # Return identifiers for tracking

def clean_pipeline(
                        derivates_dir,
                        preprocessed_dir,
                        sfreq,
                        epochs_length =  0, 
                        line_noise = [],
                        n_processes = cpu_count() - 1,
                        num_samples = 0,
                        conditions = ['EO', 'EC'],
                        sessions = ['ses-1'],
                        plots = False,
                        exclude_dirs = []):
    
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

    Returns:
        None: Results are saved to disk in the specified preprocessed_dir.
            Each subject's data is saved as a .npy file with the following structure:
            {preprocessed_dir}/{subject_id}/{session}/eeg/{subject_id}_{session}_{condition}_preprocessed.npy

    Raises:
        FileNotFoundError: If derivates_dir does not exist.
        PermissionError: If there are insufficient permissions to read/write directories.

    Example:
        >>> preprocess_pipeline(
        ...     derivates_dir='/path/to/derivatives',
        ...     preprocessed_dir='/path/to/output',
        ...     sfreq=500,
        ...     epochs_length=9.95,
        ...     line_noise=[50, 100, 150],
        ...     n_processes=4,
        ...     conditions=['EC'],
        ...     plots=True
        ... )
    """
    

    # In case of crashes/interruptions, starting from the last subject in the preprocessed directory
    subs = [s for s in os.listdir(preprocessed_dir) if os.path.isdir(os.path.join(preprocessed_dir, s))]
    subs = np.sort(subs)
    sample_ids = subs.tolist()
    print(f'subjects already preprocessed: {len(sample_ids)}')

    # Collect all files to process
    files_to_process = []
    total_files = 0
    
    for subdir, dirs, files in os.walk(derivates_dir):  # Iterate through all files
        dirs[:] = [d for d in dirs if d not in exclude_dirs]  # Exclude directories
        total_files += len(files)
        
        for file in files:
            if not any(sample_id in file for sample_id in sample_ids):  # Filter participants to include
                if '.csv' in file:
                    if any(session in file for session in sessions) and any(condition in file for condition in conditions):
                        filepath = os.path.join(subdir, file)
                        #print(filepath.split('/')[-1])
                        
                        # Split file name to obtain ID, session number, and condition
                        ID = str(file.split('_')[0])
                        sessID = str(file.split('_')[1])
                        cond = str(file.split('_')[2])
                        
                        # Add to list of files to process
                        files_to_process.append((
                            filepath, ID, sessID, cond, 
                            epochs_length, line_noise, sfreq, plots, 
                            preprocessed_dir
                        ))
    
    print(f"Found {len(files_to_process)} files to process")
    
    if not files_to_process:
        print("No files to process. Exiting.")
        return
    
    
    # sample number to process
    if num_samples > 0:
        # random indices to process
        print(f' WARNING: Processing {num_samples} samples. Set num_samples to 0 in preprocess_pipeline.py to process all samples.')
        random_indices = np.random.choice(len(files_to_process), num_samples, replace=False)
        files_to_process = files_to_process[:num_samples]

    # if there are fewer files than the number of processes, set the number of processes to the number of files
    if len(files_to_process) < n_processes:
        n_processes = len(files_to_process)
        print(f" n files < n processes, n_processes set to {n_processes}")
    else:
        print(f"Starting parallel processing with {n_processes} processes")    
    
    with Pool(processes=n_processes) as pool:
        list(tqdm(pool.imap(process_file, files_to_process), 
                           total=len(files_to_process), 
                           desc="Processing files.."))   
