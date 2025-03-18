import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from eeglearn.preprocess.preprocessing import Preproccesing
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas 
import mne
import re
from eeglearn.utils.io import get_participant_id_condition_from_string
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

class PowerSpectrum(Dataset):
    """
    Dataset class for computing and loading power spectrum features from EEG data.
    
    This class inherits from torch.utils.data.Dataset and provides functionality to 
    compute power spectral density (PSD) features from preprocessed EEG data. It handles
    loading data files and computing spectral features within specified frequency bands.
    
    Attributes:
        data_path (str): Path to directory containing preprocessed EEG data.
        file_list (list): List of participant folders available in the data_path.
        fmin (float): Minimum frequency (Hz) for spectral analysis.
        fmax (float): Maximum frequency (Hz) for spectral analysis.
        full_time_series (bool): Whether to use the full time series or epochs.
    """
    
    def __init__(self, 
                 cleaned_path, 
                 plots=False,
                 full_time_series=False,
                 method='welch',
                 fmin=0.5, 
                 fmax=130, 
                 tmin=None,
                 tmax=None,
                 picks=None,
                 exclude=[],
                 proj=False,
                 verbose=False):
        """
        Initialize the PowerSpectrum dataset.
        Does not accept n_jobs as an argument as this will cause nested multiprocessing.
        
        Args:
            cleaned_path (str): Path to directory containing preprocessed EEG data.
            fmin (float, optional): Minimum frequency (Hz) for spectral analysis. Defaults to 0.5 Hz.
            fmax (float, optional): Maximum frequency (Hz) for spectral analysis. Defaults to 130 Hz.
            full_time_series (bool, optional): Whether to use the full time series instead of epochs. 
                                              Defaults to False.
            other arugments, refer : https://mne.tools/1.8/generated/mne.time_frequency.Spectrum.html
        """
        self.data_path = cleaned_path
        self.participant_list = os.listdir(self.data_path)
        self.fmin = fmin
        self.fmax = fmax
        self.full_time_series = full_time_series
        self.tmin = tmin
        self.tmax = tmax
        self.picks = picks
        self.exclude = exclude
        self.proj = proj
        self.method = method
        self.verbose = verbose
        self.ran_spectrum = False
        self.plots = plots
        # run the spectrum function
        # save the results in a new folder

        # create the folder to save the plots and the spectra
        # Get the project root directory (2 levels up from this file)
        project_root = Path(__file__).resolve().parent.parent.parent

        # Define data directories using Path objects
        self.plot_save_dir = project_root / 'data' / 'psd' / 'plots'
        self.spectrum_save_dir = project_root / 'data' / 'psd' / 'spectra'
        self.spectrum_save_dir_epoched = project_root / 'data' / 'psd' / 'spectra_epoched'
        # Ensure the directories exist
        self.plot_save_dir.mkdir(parents=True, exist_ok=True)
        if not self.full_time_series:
            self.spectrum_save_dir_epoched.mkdir(parents=True, exist_ok=True)
        else:
            self.spectrum_save_dir_epoched.mkdir(parents=True, exist_ok=True)
        # Get the actual numer of numpy files to process
        self.folders_and_files = []
        self.participant_npy_files = []
        for participant in self.participant_list:
            participant_folder = Path(self.data_path) / participant / 'ses-1' / 'eeg'
            for file in os.listdir(participant_folder):
                if file.endswith('.npy'):
                    self.participant_npy_files.append(file)
                    self.folders_and_files.append((participant_folder, file))

    def __len__(self):
        """
        Return the number of participants in the dataset.
        
        Returns:
            int: The number of participant folders in the data path.
        """
        return len(self.participant_npy_files)
    
    def __getitem__(self, idx):
        """
        Get spectral data for a specific participant based on index.
        
        This method computes the spectrum if it hasn't been computed already,
        then returns the data for the participant at the specified index.
        
        Args:
            idx (int): Index of the participant in the file_list.
            
        Returns:
            torch.Tensor: The spectral data for the requested participant.
        """
        # make sure the spectrum is computed first
        if not self.ran_spectrum:
            self.run_spectrum_parallel()
        try:
            participant_id, condition =  get_participant_id_condition_from_string(self.participant_npy_files[idx])
            if self.full_time_series:
                spectra = torch.load(self.spectrum_save_dir / f'psd_{participant_id}_{condition}.pt')
                freqs = torch.load(self.spectrum_save_dir / f'freqs_{participant_id}_{condition}.pt')
            else:
                spectra = torch.load(self.spectrum_save_dir_epoched / f'psd_{participant_id}_{condition}.pt')
                freqs = torch.load(self.spectrum_save_dir_epoched / f'freqs_{participant_id}_{condition}.pt')
            return spectra, freqs
        except IndexError:
            print(f'Spectrum for {self.participant_npy_files[idx]} not found')
            return None, None
        except FileNotFoundError:
            print(f'Spectrum for {self.participant_npy_files[idx]} not found')
            return None, None

    def plot_psd(self, psd_object, xscale='linear'):
        """
        Plot the power spectral density (PSD) of an EEG dataset.
        
        This method uses MNE's plotting functions to visualize the PSD of the EEG data.
        It allows for customization of the x-axis scale (linear or logarithmic) and
        the display of decibel (dB) units.
        """
        with mne.viz.use_browser_backend('matplotlib'):
            #print(type(raw))
            fig = psd_object.plot(picks='eeg', xscale=xscale, dB=True, show=False)
            fig = FigureCanvas(fig)
            fig.draw()
            fig = np.asarray(fig.buffer_rgba())

        return fig
    
    def get_spectrum(self, folder_path, file_name):
        """
        Compute power spectrum density representations for all participants and conditions.
        
        This method iterates through all participant folders and processes each .npy file
        to extract the power spectral density features using MNE's compute_psd method."""   
        participant_id, condition = get_participant_id_condition_from_string(file_name)     
        data = np.load(folder_path / file_name, allow_pickle=True) 
        # from the epoched data, compute the psd
        if self.full_time_series:
            psd = data.preprocessed_raw.compute_psd(method=self.method,
                                                fmin=self.fmin,
                                                fmax=self.fmax, 
                                                tmin=self.tmin,
                                                tmax=self.tmax,
                                                picks=self.picks,
                                                exclude=self.exclude,
                                                proj=self.proj,
                                                verbose=self.verbose)
            spectra, freqs = psd.get_data(return_freqs=True)
            torch.save(spectra, f'{self.spectrum_save_dir}/psd_{participant_id}_{condition}.pt')
            torch.save(freqs, f'{self.spectrum_save_dir}/freqs_{participant_id}_{condition}.pt')
            if self.plots:
                fig = self.plot_psd(psd)
                plt.figure(figsize=(10, 6))
                plt.imshow(fig)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'{self.plot_save_dir}/psd_{participant_id}_{condition}.png', dpi=300)
                plt.close()
        else:
            psd = data.preprocessed_epochs.compute_psd(method=self.method,
                                                fmin=self.fmin,
                                                fmax=self.fmax, 
                                                tmin=self.tmin,
                                                tmax=self.tmax,
                                                picks=self.picks,
                                                exclude=self.exclude)
            spectra, freqs = psd.get_data(return_freqs=True)
            torch.save(spectra, f'{self.spectrum_save_dir_epoched}/psd_{participant_id}_{condition}.pt')
            torch.save(freqs, f'{self.spectrum_save_dir_epoched}/freqs_{participant_id}_{condition}.pt')
            if self.plots:
                fig = self.plot_psd(psd)
                plt.figure(figsize=(10, 6))
                plt.imshow(fig)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'{self.plot_save_dir}/psd_{participant_id}_{condition}.png', dpi=300)
                plt.close()
                
    def run_spectrum_parallel(self):
        """ 
        Compute power spectrum density representations for all participants and conditions.
        
        This method iterates through all participant folders and processes each .npy file
        to extract the power spectral density features using MNE's compute_psd method.
        The PSD data is computed for the frequency range specified by fmin and fmax.
        
        Returns:
            dict: A dictionary containing the computed PSD data for each participant and condition,
                  or None during development.
        """
        self.ran_spectrum = True
        processes = cpu_count() - 1
        print(f'Using {processes} processes for spectrum computation')
        with Pool(processes) as p:
            list(tqdm(p.starmap(self.get_spectrum, self.folders_and_files), 
                     total=len(self.folders_and_files), 
                     desc="Computing spectrums"))

if __name__ == "__main__":
    dataset = PowerSpectrum(cleaned_path="/Users/udeshhabaraduwa/Library/CloudStorage/GoogleDrive-u.habaraduwakandambige@tilburguniversity.edu/My Drive/Tilburg/Masters CSAI/Semester 4/code/thesis/eeg-graph-learning/data/cleaned",
                            full_time_series=False,
                            method='multitaper',
                            plots=True,
                            fmin=0.5,
                            fmax=130,
                            tmin=None,
                            tmax=None,
                            picks=None,
                            proj=False,
                            verbose=False)
    print(len(dataset))
    print(dataset[9][0].shape, dataset[9][1].shape)
    for i in range(len(dataset[:10])):
        print(dataset[i][0].shape, dataset[i][1].shape)
