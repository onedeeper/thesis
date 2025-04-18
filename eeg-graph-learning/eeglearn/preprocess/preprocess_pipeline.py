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
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import close
from pathlib import Path
from multiprocessing import Pool, cpu_count
from eeglearn.preprocess.preprocessing import Preproccesing  
from eeglearn.preprocess.save_to_torch import get_filepaths, preprocess_and_save_data
from eeglearn.preprocess.clean import clean_pipeline
from eeglearn.config import Config


if __name__ == '__main__':
    # Set seed for reproducibility - only verbose in the main process
    Config.set_global_seed(verbose=True)
    
    # Get the project root directory (2 levels up from this file)
    project_root = Path(__file__).parent.parent.parent

    derivates_dir = str(project_root / 'data' / 'TDBRAIN-dataset' / 'derivatives')
    print(f'Reading data from: {derivates_dir}')
    assert os.path.exists(derivates_dir), \
        f"Derivatives directory does not exist: {derivates_dir}"
    
    # if cleaned folder does not exist, create it
    if not os.path.exists(str(project_root / 'data' / 'cleaned')):
        os.makedirs(str(project_root / 'data' / 'cleaned'), exist_ok=True) 
    assert os.path.exists(str(project_root / 'data' / 'cleaned')), \
    f"Preprocessed directory does not exist: {str(project_root / 'data' / 'cleaned')}"
    preprocessed_dir = str(project_root / 'data' / 'cleaned')
    print(f'Writing preprocessed data to: {preprocessed_dir}')

    # Create the output directory if it doesn't exist
    os.makedirs(preprocessed_dir, exist_ok=True)
    assert os.path.exists(preprocessed_dir), \
        f"Preprocessed directory does not exist: {preprocessed_dir}"
    # the following parameters can be changed by the user
    conditions = ['EO', 'EC'] # conditions to be preprocessed
    sessions = ['ses-1'] # sessions to be preprocessed
    epochs_length = 12 # length of epochs in seconds, comment out for no epoching
    sfreq = 500 # sampling frequency
    line_noise = np.arange(50, sfreq / 2, 50) # 50 Hz line noise removal
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
                        num_samples = num_samples)
    
    project_root = Path(__file__).resolve().parent.parent.parent

    # Define data directories using Path objects
    save_dir = project_root / 'data' / 'data_cleaned_torch'
    eeg_dir = project_root / 'data' / 'cleaned'

    # Ensure the directories exist
    save_dir.mkdir(parents=True, exist_ok=True)
    eeg_dir.mkdir(parents=True, exist_ok=True)

    get_filepaths(eeg_dir, save_dir, recording_condition=['EC', 'EO'], session='ses-1')

    # save to torch
    filepaths = get_filepaths(eeg_dir, save_dir, recording_condition=['EC', 'EO'],
                               session='ses-1')
    preprocess_and_save_data(filepaths,save_dir, n_processes) 
    assert len(os.listdir(save_dir)) > 0,  "No files were processed"