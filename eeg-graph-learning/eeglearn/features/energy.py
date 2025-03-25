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
from eeglearn.preprocess.plotting import get_plots
from itertools import permutations
from eeglearn.config import Config

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
                 picks : list[str] = None,
                 include_bad_channels : bool = True,
                 full_time_series : bool = False,
                 ) -> None:
        self.cleaned_path = cleaned_path
        self.participant_list = os.listdir(self.cleaned_path)
        self.get_labels = get_labels
        self.plots = plots
        self.verbose = verbose
        self.include_bad_channels = include_bad_channels
        self.picks = picks
        if get_labels:
            self.labels_dict = get_labels_dict()
        
        # Define all available frequency bands
        self.all_freq_bands = {
            'delta': [1, 3],
            'theta': [4, 7],
            'alpha': [8, 13],
            'beta': [14, 30],
            'gamma': [31, 50]
        }
        
        # Validate and set the requested frequency bands
        if not all(band in self.all_freq_bands for band in freq_bands):
            raise ValueError(f"Invalid frequency band. Available bands are: {list(self.all_freq_bands.keys())}")
        self.freq_bands = [[band, self.all_freq_bands[band]] for band in freq_bands]
        
        self.full_time_series = full_time_series
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

    def get_energy(self, folder_path:  Path, file_name: str) -> np.ndarray:
        participant_id, condition = get_participant_id_condition_from_string(file_name)     
        data = np.load(folder_path / file_name, allow_pickle=True) 
        filtered_data = []
        band_energy = []
        # get only the requested bands
        for i, band in enumerate(self.freq_bands):
            # Bandpass filter for the requested band. Returns the entire raw object 
            # with the filter applied to the selected channels by self.picks
            filtered_data.append(data.preprocessed_raw.filter(l_freq=band[1][0],
                                                                h_freq=band[1][1],
                                                                picks = self.picks))
            # Compute the energy of the filtered data. 26 channels x time points.
            band_energy.append(np.mean(filtered_data[i].get_data(picks = self.picks) ** 2, axis = 1))
        
        band_matrix  = np.array(band_energy).T
        # test for nan values
        if np.isnan(band_matrix).any():
            raise ValueError(f"NaN values found in band_matrix for {file_name}")
        # test for inf values
        if np.isinf(band_matrix).any():
            raise ValueError(f"Inf values found in band_matrix for {file_name}")
        return band_matrix
        
    def get_permutations(self, data : np.ndarray) -> list[str]:
        """
        Get all the permutations of the data.
        """
        permutations_of_bands = list(permutations(range(len(self.freq_bands))))
        # permute the data
        permuted_data = []
        for permutation in permutations_of_bands:
            permuted_data.append(data[:, permutation])
        return permuted_data
    
    def run_energy_parallel(self):
        pass
 
if __name__ == "__main__":
    # Set seed for reproducibility
    Config.set_global_seed()
    
    cleaned_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'cleaned'
    labels_file = Path(__file__).resolve().parent.parent.parent / 'data' / 'TDBRAIN_participants_V2.xlsx'
    dataset = Energy(cleaned_path=cleaned_path,
                          get_labels=True,
                          plots=True,
                          verbose=False,
                          picks = ['eeg'],
                          freq_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma'])
    print(len(dataset))
    print(dataset.get_permutations(dataset.get_energy(dataset.folders_and_files[0][0], dataset.folders_and_files[0][1]))[0].shape)
    #dataset.get_energy(dataset.folders_and_files[0][0], dataset.folders_and_files[0][1])