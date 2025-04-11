"""A clas to compute the band specific energy for each participant.

Created on: April 2025
Author: Udesh Habaraduwa
"""

import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from eeglearn.config import Config
from eeglearn.features.spectrum import PowerSpectrum
from eeglearn.utils.utils import (
    get_cleaned_data_paths,
    get_labels_dict,
    get_participant_id_condition_from_string,
)
from itertools import permutations
import random

class Energy(Dataset):
    """A class to compute the band specific energy for each participant.

    Methods
    -------
    __init__:
        Initialize the Energy class.
    __len__:
        Return the number of participants in the dataset.
    __getitem__:
        Return the energy for a given participant.
    get_energy:
        Compute the energy for a given participant.
    run_energy_parallel:
        Compute the energy for all participants in parallel.    

    """

    def __init__(self, 
                 cleaned_path : str,
                 select_freq_bands : list[str] = None,
                 save_to_disk : bool = True,
                 energy_plots : bool = False,
                 full_time_series : bool = False,
                 method_psd : str = 'welch',
                 fmin_psd : float = 0.5,
                 fmax_psd : float = 130,
                 tmax_psd : float  = None,
                 picks_psd : list[str] = None,
                 include_bad_channels_psd : bool = True,
                 proj_psd : bool = False, 
                 verbose_psd : bool = False,
                 ) -> None:
        """Dataset class for computing and loading energy features from EEG data.

        δ (1-3 Hz)
        θ (4-7 Hz)
        α (8-13 Hz)
        β (14-30 Hz)
        γ (31-50 Hz)

        Args:
        ----
            cleaned_path (str): Path to the cleaned data.
            select_freq_bands (list[str]): List of frequency bands.
            save_to_disk (bool): Whether to save the energy to disk.
            get_labels (bool): Whether to get the labels.
            plots (bool): Whether to plot the energy.
            verbose (bool): Whether to print verbose output.
            full_time_series (bool): Whether to use the full time series.
            method_psd (str): Method to use for PSD computation.
            fmin_psd (float): Minimum frequency for PSD computation.
            fmax_psd (float): Maximum frequency for PSD computation.
            tmax_psd (float): Maximum time for PSD computation.
            picks_psd (list[str]): List of channels to compute the energy for.
            energy_plots (bool): Whether to plot the energy.
            proj_psd (bool): Whether to project the PSD.
            verbose_psd (bool): Whether to print verbose output.
            include_bad_channels_psd (bool): Whether to include bad
              channels in the PSD computation.

        """
        self.channel_names: list[str] =  ['Fp1', 'Fp2', 'F7', 
                               'F3', 'Fz', 'F4', 
                               'F8', 'FC3', 'FCz', 
                               'FC4', 'T7', 'C3', 'Cz', 'C4', 'T8', 
                               'CP3', 'CPz', 'CP4', 'P7', 'P3', 'Pz', 
                               'P4', 'P8', 'O1', 'Oz', 'O2']
        self.n_eeg_channels = 26
        # Define all available frequency bands, and the index of the band
        self.all_freq_bands :dict[tuple[list,int]] = {
            'delta': ([1, 3], 0),
            'theta': ([4, 7], 1),
            'alpha': ([8, 13], 2),
            'beta': ([14, 30], 3),
            'gamma': ([31, 50], 4)
        }
        self.cleaned_path : str = cleaned_path
        assert os.path.exists(self.cleaned_path), "cleaned_path does not exist"
        if select_freq_bands is None:
            # this is guaranteed to be in inserstion order
            select_freq_bands : list[str] = list(self.all_freq_bands.keys())
            self.select_freq_bands = select_freq_bands
        else:
            assert isinstance(select_freq_bands, list), \
                "select_freq_bands must be a list"
            assert all(band in self.all_freq_bands for band in select_freq_bands), \
                "Invalid frequency band."
            self.select_freq_bands = select_freq_bands
        
        # get the position in the order of each band
        ordered_bands = [(band,self.all_freq_bands[band][1]) for\
                          band in self.select_freq_bands]
        # get the band names in this order
        self.select_freq_bands : list[str] = [band[0] for band in sorted(ordered_bands,\
                                                      key = lambda x : x[1])]
        
        assert len(self.select_freq_bands) == len(select_freq_bands),\
            "Requested bands and reorderd bands differ in length"
        
        if include_bad_channels_psd:
            assert isinstance(include_bad_channels_psd, bool), \
                "include_bad_channels_psd must be a bool"
            self.include_bad_channels_psd : bool = include_bad_channels_psd
        self.ran_energy : bool = False  
        self.save_to_disk : bool = save_to_disk
        self.energy_plots : bool = energy_plots
        self.full_time_series : bool = full_time_series
        self.method_psd : str = method_psd
        self.fmin_psd : float = fmin_psd
        self.fmax_psd : float = fmax_psd
        self.tmax_psd : float = tmax_psd
        self.picks_psd : list[str] = picks_psd
        self.include_bad_channels_psd : bool = include_bad_channels_psd
        self.proj_psd : bool = proj_psd
        self.participant_list : list[str] = os.listdir(self.cleaned_path)
        assert len(self.participant_list) > 0, "No participants found in cleaned_path"
        self.labels_dict : dict = get_labels_dict()
        assert self.labels_dict is not None, "labels_dict is None"
        self.verbose_psd : bool = verbose_psd

        project_root : Path = Path(__file__).resolve().parent.parent.parent
        assert project_root.name == \
            'eeg-graph-learning',"project_root is not eeg-graph-learning"
        
        self.plot_save_dir : Path = project_root / 'data' / 'energy' / 'plots'
        self.plot_save_dir.mkdir(parents=True, exist_ok=True)
        assert os.path.exists(self.plot_save_dir), "Plotting directory path invalid"
        self.energy_save_dir : Path = project_root / 'data' / 'energy' / 'energy'
        self.energy_save_dir.mkdir(parents=True, exist_ok=True)
        assert os.path.exists(self.energy_save_dir),\
            " Energy save directory path invalid"
        self.energy_save_dir_epoched : Path = project_root \
            / 'data' / 'energy' / 'energy_epoched'
        self.energy_save_dir_epoched.mkdir(parents=True, exist_ok=True)
        assert os.path.exists(self.energy_save_dir_epoched), \
            "Energy epoched directory path invalid"
        
        
        # Get the actual number of numpy files to process
        self.folders_and_files : list[tuple[Path, str]] = [] 
        self.participant_npy_files : list[str] = []
        self.folders_and_files, self.participant_npy_files = \
            get_cleaned_data_paths(self.participant_list, self.cleaned_path)    

        assert len(self.participant_npy_files) > 0,"No .npy files found in cleaned_path"

    def __len__(self):
        """Return the number of EEG data files in the dataset.
        
        Returns
        -------
            int: The number of .npy files across all participant folders.

        """
        return len(self.participant_npy_files)

    def __getitem__(self, idx : int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Get band energy data for a specific file based on index.
        
        This method computes the band energy data if it hasn't been computed already,
        then returns the data for the file at the specified index.
        
        Args:
        ----
            idx (int): Index of the file in the participant_npy_files list.
            
        Returns:
        -------
            tuple: Contains:
                - torch.Tensor: The band energy data.
                - str: The participant label (if get_labels is True).
                
        Note:
        ----
            If the spectrum file is not found, returns (None, None, None).

        """
        # make sure the spectrum is computed first
        assert self.save_to_disk,\
    "Designed to index by loading. Instantiate energy object with save_to_disk= True"
        if not self.ran_energy:
            self.run_energy_parallel()
        try:
            participant_id, condition =  get_participant_id_condition_from_string\
                (self.participant_npy_files[idx])
            label = self.labels_dict[participant_id]
            if self.full_time_series:
                energy = torch.load(self.energy_save_dir /\
                                      f'energy_{participant_id}_{condition}.pt')
            else:
                energy = torch.load(self.energy_save_dir_epoched /\
                                      f'energy_{participant_id}_{condition}.pt')
            return energy, label
        except IndexError:
            print(f'Energy for {self.participant_npy_files[idx]} not found')
            return None, None, None
        except FileNotFoundError:
            print(f'Energy for {self.participant_npy_files[idx]} not found')
            return None, None, None
    
    def plot_energy(self,):
        """Plot the energy of the EEG data for a given participant."""
        pass

    def get_energy(self, folder_path:  Path, file_name: str) -> torch.Tensor:
        """Compute the energy of the EEG data for one file.

        Args:
        ----
            folder_path (Path): The path to the folder containing the EEG data.
            file_name (str): The name of the file containing the EEG data.

        Returns:
        -------
            np.ndarray: The energy of the EEG data.

        """
        participant_id : str
        condition : str
        participant_id, condition = get_participant_id_condition_from_string(file_name)
        path_to_file : Path = folder_path / file_name
        assert os.path.exists(path_to_file),f"file does not exist: {path_to_file}"
        #filtered_data : list[mne.io.Raw] = []
        # # get the spectrum
        spectra : torch.Tensor
        freqs : torch.Tensor
        n_bad_channels : int 
        n_channels_included : int
        spectrum : PowerSpectrum = PowerSpectrum(
                                                cleaned_path=self.cleaned_path,
                                                full_time_series=self.full_time_series,
                                                fmin= \
                                                    self.all_freq_bands['delta'][0][0],
                                                fmax= \
                                                    self.all_freq_bands['gamma'][0][1],
                                                verbose=self.verbose_psd,
                                                picks=self.picks_psd,
                                                include_bad_channels= \
                                                    self.include_bad_channels_psd,
                                                save_to_disk=False)
        spectra, freqs, _, n_bad_channels = spectrum.get_spectrum(
            folder_path=folder_path,
            file_name=file_name,
            save_to_disk=False)
        n_epochs = spectra.shape[0]
        assert not torch.isnan(spectra).any(), "spectra contains nans"
        assert not torch.isnan(freqs).any(), "freqs contains nans"
        n_freqs : int = len(freqs)
        masks : dict = {
            "delta" : (freqs >= 1) & (freqs <= 3),
            "theta" : (freqs >= 4) & (freqs <= 7),
            "alpha" : (freqs >= 8) & (freqs <= 13),
            "beta" :  (freqs >= 14) & (freqs <= 30),
            "gamma" : (freqs >= 31) & (freqs <= 50)
        }
        if self.full_time_series:
            # n_bads either >= 0 , can be 1,2,  --> n _eeg_channels
            # if you are including them in the analysis, then this should not be
            # removed from the included bands
            # if include_bads --> do not subtract
            # if you are Not including, then remove them. 
            # if (not include_bads) --> subtract 
            n_channels_included = self.n_eeg_channels - \
                (1 - self.include_bad_channels_psd)*(n_bad_channels)
            assert spectra.shape[1] == n_freqs, \
                "spectra and freqs have different number of frequencies"
            band_energies : dict = {
                "delta" : torch.sum(spectra[:,masks['delta']], dim = 1),
                "theta" : torch.sum(spectra[:,masks['theta']], dim = 1),
                "alpha" : torch.sum(spectra[:,masks['alpha']], dim = 1),
                "beta"  : torch.sum(spectra[:,masks['beta']], dim = 1),
                "gamma" : torch.sum(spectra[:,masks['gamma']], dim = 1)
            }
            assert band_energies['delta'].shape == \
            band_energies['theta'].shape == \
                band_energies['alpha'].shape == \
                band_energies['beta'].shape == \
                band_energies['gamma'].shape ==  (n_channels_included,), \
                    "Band energies have different shapes"
            # combine the band energies into a single tensor
            # n_channels x n_bands
            selected_bands : list = [band_energies[band] for \
                                      band in self.select_freq_bands]
            combined_energy : torch.Tensor = torch.cat(selected_bands).\
                reshape(n_channels_included,-1)
            assert combined_energy.shape == (n_channels_included, 
                                             len(self.select_freq_bands)), \
                "combined_energy has wrong shape"
            if self.save_to_disk:
                torch.save(combined_energy, self.energy_save_dir /\
                            f"energy_{participant_id}_{condition}.pt")
            return combined_energy
        else:
            n_channels_included = self.n_eeg_channels - \
                (1 - self.include_bad_channels_psd)*(n_bad_channels)
            assert spectra.shape[2] == n_freqs,\
                "spectra and freqs have different number of frequencies"
            # There are 5 matrices of shape (n_epochs x n_channels x n_freq_bins)
            # we collapse the frequency bins dimension by summing it up
            # then we combine the 5 matrices into a 2d matrix of shape 
            # (n_channels x (n_epochs * n_selected_bands))
            band_energies : dict = {
                "delta" : torch.sum(spectra[:,:,masks['delta']], dim = 2),
                "theta" : torch.sum(spectra[:,:,masks['theta']], dim = 2),
                "alpha" : torch.sum(spectra[:,:,masks['alpha']], dim = 2),
                "beta"  : torch.sum(spectra[:,:,masks['beta']], dim = 2),
                "gamma" : torch.sum(spectra[:,:,masks['gamma']], dim = 2)
            }
            assert band_energies['delta'].shape == \
            band_energies['theta'].shape == \
                band_energies['alpha'].shape == \
                band_energies['beta'].shape == \
                band_energies['gamma'].shape ==  (n_epochs, n_channels_included), \
                    "Band energies have different shapes"
            
            selected_bands : list = [band_energies[band] for \
                                      band in self.select_freq_bands]
            combined_energy : torch.Tensor = torch.cat(selected_bands).T
            
            assert combined_energy.shape ==  (n_channels_included,
                                              len(self.select_freq_bands)*n_epochs), \
                f"combined_energy has wrong shape : {combined_energy.shape}"

            if self.save_to_disk:
                torch.save(combined_energy, self.energy_save_dir_epoched /\
                            f"energy_{participant_id}_{condition}.pt")
            return combined_energy
    
    # TODO: add this back in
    # def energy_topo_plot(self, energy_dict : dict = None) -> None:
    #     """
    #     Plot the energy of the delta, theta, alpha, beta, and gamma bands.
    #     """
    #     if energy_dict is None:
    #         energy_dict = self.get_energy(participant_id, condition)
    #     for participant_id, condition in self.participant_list:
    #         energy = self.get_energy(participant_id, condition)
    #         print(energy['delta_band_energy'].shape)


    def run_energy_parallel(self) -> None:
        """ 
        Compute band energy density for all participants and conditions in parallel.
        
        It uses the get_energy method for individual
        file processing and uses a process pool to distribute the workload.
        
        The method sets ran_energy to True to indicate that energy computation has 
        been performed, to make sure when called by __getitem__ it does not compute
        the energy again.

        """
        self.ran_energy = True
        processes = cpu_count() - 1
        print(f'Using {processes} processes for energy computation')
        with Pool(processes) as p:
            results : list[torch.Tensor] = \
                list(tqdm(p.starmap(self.get_energy, self.folders_and_files), 
                     total=len(self.folders_and_files), 
                     desc="Computing Energy bands"))
            if not self.save_to_disk:
                return results
    #TODO: add the permutations
    def get_permutations(self, data : torch.Tensor) -> list[torch.Tensor,int]:
        """
        Get all the frequency band permutations of the data.
        Takes a n_channels x bands matrix and shuffles the c

        Args:
        ----
            data (torch.Tensor): A n_channels x n_bands tensor
                                (optionally, dimension two can be multipled by epochs).

        Returns:
        -------
            list[torch.Tensor, int]: The permutations of the
            data and the number of permutations.
        """
        assert isinstance(data, torch.Tensor)
        # Assert shape for non-epoched and epoched cases
        assert len(data.shape) >=2 or len(data.shape) <= 3
        band_position : dict = {
        "delta" : 0,
        "theta" : 1,
        "alpha" : 2,
        "beta" :3,
        "gamma": 4,
        }
        possible_perms : dict[int, tuple[str, str, str,str,str]] =  \
            {pseudo_label : perm for pseudo_label, perm \
             in enumerate(permutations(list(self.all_freq_bands.keys())))}
        pseudo_label : int = random.randint(0,119)
        band_ordering : list[int] = [band_position[band]\
                                      for band in possible_perms[pseudo_label]]
        shuffled_columns : torch.Tensor = data[:,band_ordering]
        return (shuffled_columns,pseudo_label) 
    
if __name__ == "__main__":
    # Set seed for reproducibility
    Config.set_global_seed()
    
    cleaned_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'cleaned'
    labels_file = Path(__file__).resolve().parent.parent.parent / 'data' / \
        'TDBRAIN_participants_V2.xlsx'
    dataset = Energy(cleaned_path=cleaned_path,
                     full_time_series=False,
                          energy_plots=True,
                          verbose_psd=False,
                          picks_psd = ['eeg'],
                          include_bad_channels_psd=False,
                          save_to_disk=True,
                          select_freq_bands=['alpha', 'theta', 'gamma'])
    print(len(dataset))
    #files = dataset.run_energy_parallel()
    #print(len(files))
    print(dataset[0][0].shape)
    print(dataset.get_permutations(dataset[0][0]))
         