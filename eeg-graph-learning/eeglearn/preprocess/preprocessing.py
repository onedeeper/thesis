"""
Created on Thu Mar 7 2024

author: T.Smolders

description: Object for automated preprocessing of the TDBRAIN dataset

name: preprocessing.py

version: 1.1

"""

# Importing necessary packages
import os
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from pyprep.prep_pipeline import PrepPipeline
#from .plotting import get_plots
from eeglearn.preprocess.plotting import get_plots
import warnings
import logging

# Configure logging
logging.getLogger('pyprep').setLevel(logging.ERROR)
mne.set_log_level('ERROR')  # Make MNE logging more quiet

# Suppress specific RuntimeWarnings about head frame origin
warnings.filterwarnings('ignore', message='.*more than 20 mm from head frame origin.*')

class Preproccesing:
    """
    Create an object for preprocessing EEG data from the TDBRAIN dataset.

    This class handles the preprocessing of EEG data, including applying the PREP pipeline,
    ICA artifact correction, bandpass filtering, and optional epoching. It also provides
    functionality to generate plots at different stages of the preprocessing.

    Args:
        filename (str): The name of the file to be preprocessed. Input files can be in .csv or .edf format.
        epochs_length (float, optional): Length of epochs in seconds. If 0, no epoching is applied. Defaults to 0.
        line_noise (list, optional): List of frequencies for line noise removal. An empty list means no line noise removal. Defaults to [].
        sfreq (float, optional): Sampling frequency in Hz. Defaults to 500.
        plots (bool, optional): If True, plots will be generated and saved. Defaults to False.

    Provides:
        - Preprocessing of data according to the following pipeline:
            - PREP (see readme for source and details)
            - ICA for ECG, EOG, and EMG artifact correction
            - Bandpass filtering (1-100Hz)
            - Epoching (optional)
        - Plots of the data at different steps in the preprocessing pipeline.

    Returns:
        Preproccesing: An object containing information about:
            - Bad channels after each step in the PREP pipeline
            - Preprocessed data
            - Preprocessed epochs (if epoching is applied)
            - Plots of the data at different steps in the preprocessing pipeline
    """

    def __init__(
            self,
            filename, # path to the .csv file containing the EEG data
            epochs_length = 0, # length of epochs in seconds, 0 = no epoching
            line_noise = [], # frequencies for line noise removal, empty list = no line noise removal
            sfreq = 500, # sampling frequency in Hz
            plots = False # if True, plots will be made and saved
    ):
        ## Set montage based on channel names and locations provided in Van Dijk et al., (2022) (Copied from Anne van Duijvenbode)
        ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',\
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', \
                'eog', 'eog', 'eog', 'eog', 'ecg', 'eog', 'emg']

        ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz', 'FC4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', \
                    'CPz', 'CP4', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2', 'VPVA', 'VNVB', 'HPHL', 'HNHR', 'Erbs', \
                    'OrbOcc', 'Mass']

        dict_ch_pos_array = {'Fp1' : np.array([-0.02681, 0.08406, -0.01056]),
                    'Fp2' : np.array([0.02941, 0.08374, -0.01004]),
                    'F7'  : np.array([-0.06699, 0.04169, -0.01596]),
                    'F3'  : np.array([-0.04805, 0.05187, 0.03987]),
                    'Fz'  : np.array([0.00090, 0.05701, 0.06636]),
                    'F4'  : np.array([0.05038, 0.05184, 0.04133]),
                    'F8'  : np.array([0.06871, 0.04116, -0.01531]),
                    'FC3' : np.array([-0.05883, 0.02102, 0.05482]),
                    'FCz' : np.array([0.00057, 0.02463, 0.08763]),
                    'FC4' : np.array([0.06029, 0.02116, 0.05558]), 
                    'T7'  : np.array([-0.08336, -0.01652, -0.01265]), 
                    'C3'  : np.array([-0.06557, -0.01325, 0.06498]),
                    'Cz'  : np.array([0.000023, -0.01128, 0.09981]),
                    'C4'  : np.array([0.06650, -0.01280, 0.06511]),
                    'T8'  : np.array([0.08444, -0.01665, -0.01179]), 
                    'CP3' : np.array([-0.06551, -0.04848, 0.06857]),
                    'CPz' : np.array([-0.0042, -0.04877, 0.09837]), 
                    'CP4' : np.array([0.06503, -0.04835, 0.06857]), 
                    'P7'  : np.array([-0.07146, -0.07517, -0.00370]), 
                    'P3'  : np.array([-0.05507, -0.08011, 0.05944]), 
                    'Pz'  : np.array([-0.00087, -0.08223, 0.08243]),
                    'P4'  : np.array([0.05351, -0.08013, 0.05940]), 
                    'P8'  : np.array([0.07110, -0.07517, -0.00369]), 
                    'O1'  : np.array([-0.02898, -0.11452, 0.00967]),  
                    'Oz'  : np.array([-0.00141, -0.11779, 0.01584]),
                    'O2'  : np.array([0.02689, -0.11468, 0.00945])
                    }
        montage = mne.channels.make_dig_montage(ch_pos = dict_ch_pos_array, coord_frame = 'head')

        # create info object for MNE
        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
        info.set_montage(montage=montage, on_missing= 'raise')
        
        # read raw EEG data
        eeg_data = pd.read_csv(filename, sep=',')
        #print(eeg_data.shape)
        eeg_data = eeg_data.transpose().to_numpy() # transpose data because MNE expects one channel per row instead of per column
        raw = mne.io.RawArray(eeg_data, info) # load data as MNE object, with the previously created 'info'
        #print('\n', 'RAW DATA LOADED', '\n')

        if plots == True:
            # plot non-preprocessed data
            non_prep_plot = get_plots(raw,
                                    step='Before preprocessing',
                                    scalings={'eeg': 1e2, 'eog': 'auto', 'emg': 'auto', 'ecg': 'auto'}
                                    )

        # define PREP parameters
        prep_params = {
            "ref_chs": "eeg",  # channels to be used for rereferencing
            "reref_chs": "eeg",  # channels from which reference signal will be subtracted
            "line_freqs": line_noise,  # frequencies for line noise removal
        }
        prep = PrepPipeline(raw, prep_params, montage)  # documentation: https://pyprep.readthedocs.io/en/latest/_modules/pyprep/prep_pipeline.html#PrepPipeline
        #print('\n', "'PREP' OBJECT CREATED", '\n')

        # run/fit PREP, applying the following steps to the raw data:
        #   1. 1Hz high pass filtering
        #   2. removing line noise
        #   3. rereferencing
        #   4. detect and interpolate bad channels
        prep.fit()
        #print('\n', "'PREP' PIPELINE APPLIED", '\n')

        # store preprocessed data & bad channels as attributes
        self.prep_data = prep.raw
        self.bad_channels_original = prep.noisy_channels_original  # (dict) Detailed bad channels in each criteria before robust reference.
        self.bad_channels_before_interpolation = prep.noisy_channels_before_interpolation  # (dict) Detailed bad channels in each criteria just before interpolation.
        self.bad_channels_after_interpolation = prep.noisy_channels_after_interpolation  # (dict) Detailed bad channels in each criteria just after interpolation.
        self.still_bad_channels = prep.still_noisy_channels  # (list) Names of the noisy channels after interpolation.

        raw = prep.raw

        if plots == True:
            # plot data after PREP
            prep_plot = get_plots(raw,
                                step='After PREP preprocessing',
                                scalings={'eeg': 1e2, 'eog': 'auto', 'emg': 'auto', 'ecg': 'auto'})

        ## Repairing EOG, ECG, and EMG artifacts with ICA
        # create a copy of the raw data and apply a low pass filter to remove 
        # slow drifts for better ICA fitting. The fitted ICA can be applied to 
        # the unfiltered signal, because filtering is a linear operation.
        filt_raw = raw.copy().filter(l_freq=1, h_freq=None)

        self.status = []

        # creating & fitting ICA object
        try:
            ica = ICA(n_components=15, max_iter="auto")  # n PCA components
            ica.fit(filt_raw)
            self.status.append("ICA FITTED")
        except Exception as e: # sometimes n_components is too high, so in that case we try again with a lower number
            self.status.append("TOO MANY COMPONENTS FOR ICA FITTING, TRYING AGAIN WITH LOWER NUMBER OF COMPONENTS")
            n_components = 15 - len(self.still_bad_channels)
            ica = ICA(n_components=n_components, max_iter="auto")  # n PCA components
            ica.fit(filt_raw)
            self.status.append("ICA FITTED WITH LOWER NUMBER OF COMPONENTS")


        # automatically detect ICs that best capture EOG signal
        try:
            ica.exclude = []
            eog_indices, eog_scores = ica.find_bads_eog(raw)
            self.status.append("EOG ARTIFACTS DETECTED")
        except:
            self.status.append("SOMETHING WRONG WITH EOG SIGNAL, SO NO EOG ARTIFACTS DETECTED")
            eog_indices = []
            eog_scores = []

        # automatically detect ICs that best capture ECG signal
        try:
            ica.exclude = []
            ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
            self.status.append("ECG ARTIFACTS DETECTED")
        except:
            self.status.append("SOMETHING WRONG WITH ECG SIGNAL, SO NO ECG ARTIFACTS DETECTED")
            ecg_indices = []
            ecg_scores = []

        # automatically detect ICs that best capture muscle artifacts
        try:
            ica.exclude = []
            emg_indices, emg_scores = ica.find_bads_muscle(raw)
            self.status.append("EMG ARTIFACTS DETECTED")
        except:
            self.status.append("SOMETHING WRONG WITH EMG SIGNAL, SO NO EMG ARTIFACTS DETECTED")
            emg_indices = []
            emg_scores = []

        # repair all artifacts with ICA
        ica.apply(raw, exclude = eog_indices + ecg_indices + emg_indices)
        self.status.append("ICA APPLIED")

        if plots == True:
            # plot data after ICA
            ica_prep_plot = get_plots(raw,
                                    step='After PREP & ICA preprocessing',
                                    ica=ica,
                                    plot_ica_overlay=True)

        ## Applying high-pass & low-pass filter
        raw.filter(l_freq=1, h_freq=100)

        if plots == True:
            # plot data after BP filtering
            bp_ica_prep_plot = get_plots(raw,
                                        step='After PREP, ICA & BP filter preprocessing')

        self.preprocessed_raw = raw  # Raw object with PREP preprocessing, ICA applied and BP filtering

        if plots == True:
            # storing plots as attribute
            self.figs = [non_prep_plot, prep_plot, ica_prep_plot, bp_ica_prep_plot]

        ## Epoching the data
        if epochs_length > 0:
            self.preprocessed_epochs = mne.make_fixed_length_epochs(
                raw,
                duration=epochs_length,
                overlap=0
                )  # Epochs object with PREP preprocessing and ICA applied
        else:
            self.preprocessed_epochs = 'No epoching applied'
            self.status.append("NO EPOCHING APPLIED")
        self.status.append("EPOCHING APPLIED")


if __name__ == '__main__':
    preprocess = Preproccesing(
            filename='/Users/udeshhabaraduwa/Library/CloudStorage/GoogleDrive-u.habaraduwakandambige@tilburguniversity.edu/My Drive/Tilburg/Masters CSAI/Semester 4/code/thesis/eeg-graph-learning/tests/test_data/TDBRAIN-dataset/derivatives/sub-19694366/sub-19694366_ses-1_task-restEC_eeg.csv',
            epochs_length=0,
            line_noise=[],
            sfreq=500,
            plots=True
        )
    print(preprocess.status)