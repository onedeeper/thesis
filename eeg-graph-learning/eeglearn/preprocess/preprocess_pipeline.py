"""
Created on Thu Mar 7 2024

author: T.Smolders
edited by: Udesh Habaraduwa

description: automated pipeline for preprocessing of the TDBRAIN dataset

name: preprocess_pipeline.py

version: 1.1
version: 1.2 - edited by Udesh Habaraduwa , parallel processing added

"""
import os
import sys
import numpy as np
import pandas as pd
import pickle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import close
from pathlib import Path
from multiprocessing import Pool, cpu_count
from eeglearn.preprocess.preprocessing import Preproccesing  # Fixed import statement

# Define a worker function that will be executed in parallel
def process_file(args):
    filepath, ID, sessID, cond, epochs_length, line_noise, sfreq, plots, preprocessed_dir = args
    
    print(f'\n [INFO]: processing subject: {ID}, session: {sessID}, condition: {cond}')
    
    # Create Preprocessing object
    preprocessed_data = Preproccesing(
        filepath,
        epochs_length,
        line_noise,
        sfreq,
        plots
    )
    
    # Define directory and subdirectories for preprocessed data
    save_dir = f'{preprocessed_dir}/{ID}/{sessID}/eeg'
    save_path_data = f'{save_dir}/{ID}_{sessID}_{cond}_preprocessed.npy'
    
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(save_path_data), exist_ok=True)
    
    if plots == True:
        save_path_plots = f'{save_dir}/{ID}_{sessID}_{cond}_preprocessing_plots.pdf'
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
    
    print(f'\n [INFO]: preprocessed data object saved to: {save_path_data} \n')
    
    return ID, sessID, cond  # Return identifiers for tracking

def preprocess_pipeline(
                        derivates_dir,
                        preprocessed_dir,
                        sfreq,
                        epochs_length =  0, 
                        line_noise = [],
                        n_processes = cpu_count(),
                        num_samples = 0,
                        conditions = ['EO', 'EC'],
                        sessions = ['ses-1'],
                        plots = False,
                        exclude_dirs = ['preprocessed', 'results_manuscript', 'adhd_sample']):
    """
    Pipeline to preprocess EEG data from the TDBRAIN dataset.

    Requires params dictionary with the following keys (initialized in __main__):
    - derivatives_dir: path to the directory containing the \derivatives\ folder
    - preprocessed_dir: path to the directory where the preprocessed data will be saved
    - condition: list of conditions to be preprocessed
    - sessions: list of sessions to be preprocessed
    - epochs_length: length of epochs in seconds, 0 = no epoching
    - sfreq: sampling frequency in Hz
    - line_noise: list of frequencies for line noise removal, empty list = no line noise removal

    Output:
    - preprocessed data object saved as .npy file
    - plots for each preprocessing step saved in a .pdf file
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
                    if any(session in file for session in sessions) & any(condition in file for condition in conditions):
                        filepath = os.path.join(subdir, file)
                        
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
    
    # Process files in parallel
    print(f"Starting parallel processing with {n_processes} processes")
    # sample number to process
    if num_samples > 0:
        # random indices to process
        random_indices = np.random.choice(len(files_to_process), num_samples, replace=False)
        files_to_process = [files_to_process[i] for i in random_indices]
    
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_file, files_to_process)
    
    print(f"Completed processing {len(results)} files")
    


if __name__ == '__main__':
    
    # Get the project root directory (2 levels up from this file)
    project_root = Path(__file__).parent.parent.parent

    params = {}
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
    sessions = ['ses-1'] # sessions to be preprocessed
    epochs_length = 9.95 # length of epochs in seconds, comment out for no epoching
    sfreq = 500 # sampling frequency
    line_noise = np.arange(50, sfreq / 2, 50) # 50 Hz line noise removal, comment out for no line noise removal
    plots = True # set to True to create and store plots during preprocessing
    n_processes = 4 # number of processes to use for parallel processing
    num_samples = 10 # number of samples to process, comment out for all samples

    preprocess_pipeline(derivates_dir = derivates_dir,
                        preprocessed_dir = preprocessed_dir,
                        sfreq = sfreq,
                        epochs_length = epochs_length,
                        line_noise = line_noise,
                        n_processes = n_processes,
                        num_samples = 0,
                        conditions = conditions,
                        sessions = sessions,
                        plots = plots)