import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas 
import mne
from eeglearn.utils.utils import get_participant_id_condition_from_string,\
    get_cleaned_data_paths, get_labels_dict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from eeglearn.config import Config
from eeglearn.preprocess.preprocessing import Preproccesing
from eeglearn.utils.plotting import plot_psd
class PowerSpectrum(Dataset):
    """
    Dataset class for computing and loading power spectrum features from EEG data.
    
    This class inherits from torch.utils.data.Dataset and provides functionality to 
    compute power spectral density (PSD) features from preprocessed EEG data. It handles
    loading data files, computing spectral features within specified frequency bands,
    and saving results for future use.
    
    Attributes:
        cleaned_path (str): Path to directory containing preprocessed EEG data.
        include_bad_channels (bool): Whether to include bad channels in PSD
                                    computation. Defaults to False.
        save_to_disk (bool): Whether to save computed spectra and frequencies to 
                             disk. Defaults to True.
        plots (bool): Whether to generate and save PSD plots. Defaults to False.
        full_time_series (bool): Whether to compute PSD on the full time series 
                                 instead of epochs. Defaults to False.
        method (str): Method for PSD computation ('welch' or 'multitaper'). 
                      Defaults to 'welch'.
        fmin (float): Minimum frequency (Hz) for spectral analysis. Defaults to 0.5.
        fmax (float): Maximum frequency (Hz) for spectral analysis. Defaults to 130.
        tmin (float | None): Start time for analysis (in seconds). None uses the 
                           beginning. Defaults to None.
        tmax (float | None): End time for analysis (in seconds). None uses the end.
                           Defaults to None.
        picks (list[str] | str): Channels to include. Defaults to 'data' (all data 
                               channels).
        proj (bool): Whether to apply projection. Defaults to False.
        verbose (bool): Whether to print detailed processing information. 
                      Defaults to False.
        participant_list (list): List of participants found in `cleaned_path`.
        ran_spectrum (bool): Flag indicating if `run_spectrum_parallel` has been 
                             executed. Initialized to False.
        labels_dict (dict): Dictionary mapping participant IDs to labels.
        plot_save_dir (Path): Directory path for saving PSD plots.
        spectrum_save_dir (Path): Directory path for saving full time series spectra.
        spectrum_save_dir_epoched (Path): Directory path for saving epoched spectra.
        folders_and_files (list[tuple[Path, str]]): List of (folder_path, file_name)
                                                    tuples for processing.
        participant_npy_files (list[str]): List of .npy file names to be processed.

    Methods:
        __len__(): Returns the total number of EEG data files to process.
        __getitem__(idx): Retrieves the computed spectrum data for the file at the
                          given index.
        get_spectrum(folder_path, file_name, save_to_disk): Computes PSD for a single
                                                            EEG file.
        run_spectrum_parallel(return_results): Computes PSD for all files in parallel.
    """
    
    def __init__(self, 
                 cleaned_path : str,
                 include_bad_channels : bool = False,
                 save_to_disk : bool = True,    
                 plots : bool = False,
                 full_time_series : bool = False,
                 method : str = 'welch',
                 fmin : float = 0.5, 
                 fmax : float = 130, 
                 tmin : float = None,
                 tmax : float = None,
                 picks : list[str] = None,
                 proj : bool = False,
                 verbose : bool = False,
                 ) -> None:
        """
        Initialize the PowerSpectrum dataset.
        
        Args:
            ignore_bad_channels (bool): Whether to ignore bad channels
            cleaned_path (str): Path to directory containing preprocessed EEG data.
            plots (bool): Whether to generate and save PSD plots.
            full_time_series (bool): Whether to use the full time series instead of 
                                    epochs. 
            method (str): Method to use for PSD computation {'welch', 'multitaper'}
            fmin (float): Minimum frequency (Hz) for spectral analysis.
            fmax (float): Maximum frequency (Hz) for spectral analysis.
            tmin (float): Start time for analysis (in seconds).
            tmax (float): End time for analysis (in seconds).
            picks (list[str]): Channels to include in the analysis. By default, 
                only includes the data channels:
            (https://mne.tools/stable/documentation/glossary.html#term-data-channels)

            exclude (list[str]): Channels to exclude from the analysis.
            proj (bool): Whether to apply projection.
            verbose (bool): Whether to print detailed information.
            
        Note:
            This method does not accept n_jobs as an argument as this will cause
            nested multiprocessing.
            It only returns the PSD for the EEG channels.
            Does not use exclude. 
        """
        self.save_to_disk = save_to_disk
        self.cleaned_path = cleaned_path
        self.participant_list = os.listdir(self.cleaned_path)
        self.fmin = fmin
        self.fmax = fmax
        self.full_time_series = full_time_series
        self.tmin = tmin
        self.tmax = tmax
        self.include_bad_channels = include_bad_channels
        self.proj = proj
        self.method = method
        self.verbose = verbose
        self.ran_spectrum = False
        self.plots = plots
        # load the labels file
        self.labels_dict = get_labels_dict()
        self.picks = ['data'] if picks is None else picks
        # create the folder to save the plots and the spectra
        # Get the project root directory (2 levels up from this file)
        project_root = Path(__file__).resolve().parent.parent.parent

        # Define data directories using Path objects
        self.plot_save_dir = project_root / 'data' / 'psd' / 'plots'
        self.spectrum_save_dir = project_root / 'data' / 'psd' / 'spectra'
        self.spectrum_save_dir_epoched = project_root / 'data' / 'psd' / \
            'spectra_epoched'
        # Ensure the directories exist
        self.plot_save_dir.mkdir(parents=True, exist_ok=True)
        if self.full_time_series:
            try:
                self.spectrum_save_dir.mkdir(parents=True, exist_ok=True)
            except FileExistsError:
                raise RuntimeError("failed to create spectrum_save_dir")
        else:
            try:
                self.spectrum_save_dir_epoched.mkdir(parents=True, exist_ok=True)
            except FileExistsError:
                raise RuntimeError("failed to create spectrum_save_dir_epoched")
        # Get the actual numer of numpy files to process
        self.folders_and_files : list[tuple[Path, str]] = []
        self.participant_npy_files : list[str] = []
        self.folders_and_files, self.participant_npy_files = \
            get_cleaned_data_paths(self.participant_list, self.cleaned_path)

    def __len__(self) -> int:
        """
        Return the number of EEG data files in the dataset.
        
        Returns:
            int: The number of .npy files across all participant folders.
        """
        return len(self.participant_npy_files)
    
    def __getitem__(self, idx : int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get spectral data for a specific file based on index.
        
        This method computes the spectrum if it hasn't been computed already,
        then returns the data for the file at the specified index.
        
        Args:
            idx (int): Index of the file in the participant_npy_files list.
            
        Returns:
            tuple: Contains:
                - torch.Tensor: The spectral data (PSD values).
                - torch.Tensor: The frequency values corresponding to the PSD data.
                - str: The participant label (if get_labels is True).
                
        Note:
            If the spectrum file is not found, returns (None, None, None).
        """
        # make sure the spectrum is computed first
        if not self.ran_spectrum:
            self.run_spectrum_parallel()
        try:
            participant_id, condition =  get_participant_id_condition_from_string\
                (self.participant_npy_files[idx])
            label = self.labels_dict[participant_id]
            if self.full_time_series:
                spectra = torch.load(self.spectrum_save_dir /\
                                      f'psd_{participant_id}_{condition}.pt')
                freqs = torch.load(self.spectrum_save_dir /\
                                      f'freqs_{participant_id}_{condition}.pt')
            else:
                spectra = torch.load(self.spectrum_save_dir_epoched /\
                                      f'psd_{participant_id}_{condition}.pt')
                freqs = torch.load(self.spectrum_save_dir_epoched /\
                                      f'freqs_{participant_id}_{condition}.pt')
            return spectra, freqs, label
        except IndexError:
            print(f'Spectrum for {self.participant_npy_files[idx]} not found')
            return None, None, None
        except FileNotFoundError:
            print(f'Spectrum for {self.participant_npy_files[idx]} not found')
            return None, None, None
    
    def get_spectrum(self, folder_path : str, 
                     file_name : str, save_to_disk : bool = True) \
                          -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Compute power spectrum density for a single EEG data file.
        
        This method loads an EEG data file (.npy containing a Preprocessing object), 
        computes the power spectral density using the specified method (`compute_psd`),
        and optionally saves the results to disk and generates plots.
        
        The specific channels included in the output depend on the `picks` and `exclude`
        parameters used during `compute_psd`. The `ordered_ch_names` list reflects the
        actual order of channels corresponding to the rows in the returned `spectra` 
        tensor.
        
        Args:
            folder_path (str): Path to the folder containing the EEG data file.
            file_name (str): Name of the EEG data file (.npy).
            save_to_disk (bool): If True, save the computed spectra, frequencies, and
                                 channel names to disk. Defaults to True. 
        Returns:
            tuple[torch.Tensor, torch.Tensor, str, int, list[str]]: A tuple containing:
                - spectra (torch.Tensor): The computed power spectral density.
                  Shape: (n_channels, n_freqs) for full time series, or 
                  (n_epochs, n_channels, n_freqs) for epoched data.
                - freqs (torch.Tensor): The frequencies (in Hz) corresponding to the
                  columns of the spectra tensor. Shape: (n_freqs,).
                - participant_id_condition (str): A string combining participant ID and
                  condition (e.g., "sub-001_cond-A").
                - n_bad_channels (int): The number of channels marked as bad in the
                  original MNE object before PSD computation.
                - ordered_ch_names (list[str]): The list of channel names corresponding
                  to the rows (dim 0 or 1) of the `spectra` tensor, reflecting the
                  order determined by `compute_psd`.
        
        Saved Files (if `save_to_disk=True`):
            - Spectra: Saved as `.pt` files (e.g., `psd_sub-001_cond-A.pt`). Contains a
              tuple: `(spectra_tensor, ordered_ch_names)`.
            - Frequencies: Saved as `.pt` files (e.g., `freqs_sub-001_cond-A.pt`).
              Contains the `freqs` tensor.
            - Files are saved in `self.spectrum_save_dir` or
              `self.spectrum_save_dir_epoched` depending on `self.full_time_series`.
        """
        participant_id, condition = get_participant_id_condition_from_string(file_name)
        try:
            data : Preproccesing = np.load(folder_path / file_name, allow_pickle=True)
        except FileNotFoundError:
            print(f'File {file_name} not found')
            return None, None, None
        assert isinstance(data, Preproccesing), "data is not a Preproccesing object"

        condition : str
        participant_id : str
        spectra : np.ndarray 
        freqs : np.ndarray
        n_bad_channels : int
        # list of ch_names that will be the row order in the final data matrix
        ordered_ch_names : list[str]  
        # from the epoched data, compute the psd
        # check shape of the epoched and full time series data
        if self.include_bad_channels:
            exclude = []
        else:
            exclude = 'bads'
        if self.full_time_series:
            #print(dir(data.preprocessed_raw))
            #print(data.preprocessed_raw.ch_names, data.preprocessed_raw.pick)
            n_bad_channels = len(data.preprocessed_raw.info['bads'])
            psd : mne.time_frequency.Spectrum = data.preprocessed_raw.compute_psd(
                                                method=self.method,
                                                fmin=self.fmin,
                                                fmax=self.fmax, 
                                                tmin=self.tmin,
                                                tmax=self.tmax,
                                                picks=self.picks,
                                                proj=self.proj,
                                                exclude=exclude,
                                                verbose=self.verbose)
            assert isinstance(psd, mne.time_frequency.Spectrum), \
                f"psd is not a mne.time_frequency.Spectrum object for {file_name}"
            ordered_ch_names = psd.ch_names
            spectra, freqs = psd.get_data(return_freqs=True, picks=self.picks,
                                          exclude=exclude)
            if self.save_to_disk:
                path_to_psd : Path = self.spectrum_save_dir / \
                    f'psd_{participant_id}_{condition}.pt'
                torch.save((torch.from_numpy(spectra), ordered_ch_names), path_to_psd)
                path_to_freqs : Path = self.spectrum_save_dir / \
                    f'freqs_{participant_id}_{condition}.pt'
                torch.save(torch.from_numpy(freqs), path_to_freqs)
            if self.plots:  
                plot_psd(psd=psd, plot_save_dir=self.plot_save_dir, \
                          participant_id=participant_id, condition=condition)
        else:
            n_bad_channels : int = len(data.preprocessed_epochs.info['bads'])
            psd : mne.time_frequency.Spectrum.EpochsSpectrum = \
                data.preprocessed_epochs.compute_psd(method=self.method,
                                                fmin=self.fmin,
                                                fmax=self.fmax, 
                                                tmin=self.tmin,
                                                tmax=self.tmax,
                                                picks=self.picks,
                                                proj=self.proj,
                                                exclude=exclude,
                                                verbose=self.verbose)
            assert isinstance(psd, mne.time_frequency.spectrum.EpochsSpectrum), \
       f"psd is not a mne.time_frequency.Spectrum.EpochsSpectrum object for {file_name}"
            ordered_ch_names = psd.ch_names
            spectra, freqs = psd.get_data(return_freqs=True, picks=self.picks,
                                          exclude=exclude)
            if self.save_to_disk:
                path_to_psd : Path = self.spectrum_save_dir_epoched / \
                    f'psd_{participant_id}_{condition}.pt'
                torch.save((torch.from_numpy(spectra), ordered_ch_names), path_to_psd)
                path_to_freqs : Path = self.spectrum_save_dir_epoched / \
                    f'freqs_{participant_id}_{condition}.pt'
                torch.save(torch.from_numpy(freqs), path_to_freqs)
            if self.plots:
                plot_psd( psd=psd, plot_save_dir=self.plot_save_dir, \
                          participant_id=participant_id, condition=condition)
        assert isinstance(spectra, np.ndarray), \
            f"spectra is not a numpy array for {file_name}"
        assert isinstance(freqs, np.ndarray), \
            f"freqs is not a numpy array for {file_name}"
        return torch.from_numpy(spectra), torch.from_numpy(freqs), participant_id \
            + '_' + condition, n_bad_channels, ordered_ch_names
                
    def run_spectrum_parallel(self, return_results : bool = False) -> None:
        """ 
        Compute power spectrum density for all participants and conditions in parallel.
        
        This method uses `multiprocessing.Pool` to parallelize the computation of PSD 
        features across all EEG data files specified in `self.folders_and_files`. It 
        'calls `self.get_spectrum` for each file.
        
        If `save_to_disk` is True within `get_spectrum` (which is the default unless
        overridden during `PowerSpectrum` initialization), the results (spectra tensor, 
        channel names, frequency tensor) are saved to disk for each file.
        
        This method sets `self.ran_spectrum = True` after completion.

        Args:
            return_results (bool): If True, the results from all `get_spectrum` calls
                                   are collected and returned in memory. 
                                   Defaults to False.

        Returns:
            list[tuple[torch.Tensor, torch.Tensor, str, int, list[str]]] or None:
                - If `return_results` is True: Returns a list where each element is the 
                  tuple returned by `get_spectrum` for a single file: 
                  `(spectra, freqs, participant_id_condition, n_bad_channels,
                    ordered_ch_names)`.
                - If `return_results` is False: Returns None.
        
        Note:
            See the `get_spectrum` docstring for details on the saved file format.
        """
        self.ran_spectrum = True
        processes = cpu_count() - 1
        print(f'Using {processes} processes for spectrum computation')
        with Pool(processes) as p:
            results : list[tuple[torch.Tensor, torch.Tensor, str]] = \
                list(tqdm(p.starmap(self.get_spectrum, self.folders_and_files), 
                     total=len(self.folders_and_files), 
                     desc="Computing spectrums"))
            
        
if __name__ == "__main__":
    # Set seed for reproducibility - only verbose in the main process
    Config.set_global_seed(verbose=True)
    
    # Find the path to the cleaned data from root directory
    cleaned_path = Path(__file__).resolve().parent.parent.parent \
        / 'data' / 'cleaned'
    # find the path to the labels file data
    labels_file = Path(__file__).resolve().parent.parent.parent\
          / 'data' / 'TDBRAIN_participants_V2.xlsx'
    dataset = PowerSpectrum(cleaned_path=cleaned_path,
                            full_time_series=False,
                            method='welch',
                            plots=True,
                            fmin=0.5,
                            fmax=130,
                            tmin=None,
                            tmax=None,
                            proj=True,
                            verbose=False,
                            include_bad_channels=True,
                           )
    print(len(dataset))
    #dataset.get_spectrum(dataset.folders_and_files[0][0], \
    #                     dataset.folders_and_files[0][1], save_to_disk=False)
    for i in range(len(dataset)):
        print(dataset[i][0][0].shape, dataset[i][1].shape, dataset[i][2]) 
        print(dataset[i][0][1])
        print("------------------------")