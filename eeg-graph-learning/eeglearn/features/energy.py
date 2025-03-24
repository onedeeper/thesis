import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas 
import mne
from eeglearn.utils.utils import get_participant_id_condition_from_string
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from eeglearn.utils.utils import get_labels_dict

class Energy(Dataset):
    """
    Dataset class for computing and loading energy features from EEG data.

    δ (1-3 Hz)

    θ (4-7 Hz)

    α (8-13 Hz)

    β (14-30 Hz)

    γ (31-50 Hz)
    """
    def __init__(self, 
                 cleaned_path : str,
                 freq_bands : list[str] = ['delta', 'theta', 'alpha', 'beta', 'gamma'],
                 get_labels : bool = True,
                 plots : bool = False,
                 verbose : bool = False,
                 include_bad_channels : bool = True,
                 full_time_series : bool = False,
                 ) -> None:
        self.cleaned_path = cleaned_path
        self.participant_list = os.listdir(self.cleaned_path)
        self.get_labels = get_labels
        self.plots = plots
        self.verbose = verbose
        self.include_bad_channels = include_bad_channels
        if get_labels:
            self.labels_dict = get_labels_dict()
        self.freq_bands = freq_bands
        self.full_time_series = full_time_series
        # freq bands
        self.delta_band = [1,3]
        self.theta_band = [4,7]
        self.alpha_band = [8,13]
        self.beta_band = [14,30]
        self.gamma_band = [31,50]
        
        # create the folder to save the plots and the spectra
        # Get the project root directory (2 levels up from this file)
        project_root = Path(__file__).resolve().parent.parent.parent

        # Define data directories using Path objects
        self.plot_save_dir = project_root / 'data' / 'energy' / 'plots'
        self.energy_save_dir = project_root / 'data' / 'energy' / 'energy'
        self.energy_save_dir_epoched = project_root / 'data' / 'energy' / 'energy_epoched'
        # Ensure the directories exist
        self.plot_save_dir.mkdir(parents=True, exist_ok=True)
        if self.full_time_series:
            self.energy_save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.energy_save_dir_epoched.mkdir(parents=True, exist_ok=True)
        # Get the actual numer of numpy files to process
        self.folders_and_files = [] 
        self.participant_npy_files = []
        for participant in self.participant_list:
            participant_folder = Path(self.cleaned_path) / participant / 'ses-1' / 'eeg'
            for file in os.listdir(participant_folder):
                if file.endswith('.npy'):
                    self.participant_npy_files.append(file)
                    self.folders_and_files.append((participant_folder, file))

    def __len__(self):
        """
        Return the number of EEG data files in the dataset.
        
        Returns:
            int: The number of .npy files across all participant folders.
        """
        return len(self.participant_npy_files)

    def __getitem__(self, idx):
        pass
    
    def plot_energy(self,):
        pass

    def get_energy(self, folder_path, file_name):
        participant_id, condition = get_participant_id_condition_from_string(file_name)     
        data = np.load(folder_path / file_name, allow_pickle=True) 
        # delta band energy
        delta_band_energy = data.preprocessed_raw.filter(l_freq=1, h_freq=3)
    def run_energy_parallel(self):
        pass

if __name__ == "__main__":
    cleaned_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'cleaned'
    labels_file = Path(__file__).resolve().parent.parent.parent / 'data' / 'TDBRAIN_participants_V2.xlsx'
    dataset = Energy(cleaned_path=cleaned_path,
                          get_labels=True,
                          plots=True,
                          verbose=False)
    print(len(dataset))
    dataset.get_energy(dataset.folders_and_files[0][0], dataset.folders_and_files[0][1])