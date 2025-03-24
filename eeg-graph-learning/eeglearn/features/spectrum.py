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

class PowerSpectrum(Dataset):
    """
    Dataset class for computing and loading power spectrum features from EEG data.
    
    This class inherits from torch.utils.data.Dataset and provides functionality to 
    compute power spectral density (PSD) features from preprocessed EEG data. It handles
    loading data files, computing spectral features within specified frequency bands,
    and saving results for future use.
    
    Attributes:
        data_path (str): Path to directory containing preprocessed EEG data.
        participant_list (list): List of participants available in the data_path.
        fmin (float): Minimum frequency (Hz) for spectral analysis.
        fmax (float): Maximum frequency (Hz) for spectral analysis.
        full_time_series (bool): Whether to use the full time series or epochs.
        tmin (float): Start time for analysis (in seconds) or None to use beginning of data.
        tmax (float): End time for analysis (in seconds) or None to use end of data.
        picks (list): Channels to include in the analysis.
        exclude (list): Channels to exclude from the analysis.
        proj (bool): Whether to apply projection.
        method (str): Method to use for PSD computation ('welch', 'multitaper').
        verbose (bool): Whether to print detailed information during processing.
        plots (bool): Whether to generate and save PSD plots.
        plot_save_dir (Path): Directory to save PSD plots.
        spectrum_save_dir (Path): Directory to save spectrum data for full time series.
        spectrum_save_dir_epoched (Path): Directory to save spectrum data for epoched data.
        folders_and_files (list): List of tuples with folder paths and file names to process.
        participant_npy_files (list): List of .npy files to process.
        ran_spectrum (bool): Flag indicating whether spectrum calculation has been run.
        ignore_bad_channels (bool): Whether to ignore bad channels during spectrum computation.
    """
    
    def __init__(self, 
                 include_bad_channels : bool,
                 cleaned_path : str,
                 get_labels : bool = True, 
                 plots : bool = False,
                 full_time_series : bool = False,
                 method : str = 'welch',
                 fmin : float = 0.5, 
                 fmax : float = 130, 
                 tmin : float = None,
                 tmax : float = None,
                 picks : list[str] = None,
                 exclude : list[str] = [],
                 proj : bool = False,
                 verbose : bool = False,
                 ) -> None:
        """
        Initialize the PowerSpectrum dataset.
        
        Args:
            ignore_bad_channels (bool): Whether to ignore bad channels during spectrum computation.
            cleaned_path (str): Path to directory containing preprocessed EEG data.
            get_labels (bool, optional): Whether to load participant labels. Defaults to True.
            plots (bool, optional): Whether to generate and save PSD plots. Defaults to False.
            full_time_series (bool, optional): Whether to use the full time series instead of epochs. 
                                              Defaults to False.
            method (str, optional): Method to use for PSD computation. Options include 'welch',
                                  'multitaper', etc. Defaults to 'welch'.
            fmin (float, optional): Minimum frequency (Hz) for spectral analysis. Defaults to 0.5 Hz.
            fmax (float, optional): Maximum frequency (Hz) for spectral analysis. Defaults to 130 Hz.
            tmin (float, optional): Start time for analysis (in seconds). Defaults to None.
            tmax (float, optional): End time for analysis (in seconds). Defaults to None.
            picks (list[str], optional): Channels to include in the analysis. Defaults to None.
            exclude (list[str], optional): Channels to exclude from the analysis. Defaults to [].
            proj (bool, optional): Whether to apply projection. Defaults to False.
            verbose (bool, optional): Whether to print detailed information. Defaults to False.
            
        Note:
            This method does not accept n_jobs as an argument as this will cause nested multiprocessing.
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
        self.include_bad_channels = include_bad_channels

        # create the folder to save the plots and the spectra
        # Get the project root directory (2 levels up from this file)
        project_root = Path(__file__).resolve().parent.parent.parent

        # Define data directories using Path objects
        self.plot_save_dir = project_root / 'data' / 'psd' / 'plots'
        self.spectrum_save_dir = project_root / 'data' / 'psd' / 'spectra'
        self.spectrum_save_dir_epoched = project_root / 'data' / 'psd' / 'spectra_epoched'
        # Ensure the directories exist
        self.plot_save_dir.mkdir(parents=True, exist_ok=True)
        if self.full_time_series:
            self.spectrum_save_dir.mkdir(parents=True, exist_ok=True)
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

        # load the labels file
        if get_labels:
            self.labels_dict = get_labels_dict()

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
            participant_id, condition =  get_participant_id_condition_from_string(self.participant_npy_files[idx])
            label = self.labels_dict[participant_id]
            if self.full_time_series:
                spectra = torch.load(self.spectrum_save_dir / f'psd_{participant_id}_{condition}.pt')
                freqs = torch.load(self.spectrum_save_dir / f'freqs_{participant_id}_{condition}.pt')
            else:
                spectra = torch.load(self.spectrum_save_dir_epoched / f'psd_{participant_id}_{condition}.pt')
                freqs = torch.load(self.spectrum_save_dir_epoched / f'freqs_{participant_id}_{condition}.pt')
            return spectra, freqs, label
        except IndexError:
            print(f'Spectrum for {self.participant_npy_files[idx]} not found')
            return None, None, None
        except FileNotFoundError:
            print(f'Spectrum for {self.participant_npy_files[idx]} not found')
            return None, None, None

    def plot_psd(self, psd_object : mne.time_frequency.Spectrum, xscale : str = 'linear') -> plt.Figure:
        """
        Plot the power spectral density (PSD) of an EEG dataset.
        
        This method uses MNE's plotting functions to visualize the PSD of the EEG data.
        It allows for customization of the x-axis scale and displays PSD in decibel (dB) units.
        
        Args:
            psd_object (mne.time_frequency.Spectrum): The PSD object from MNE.
            xscale (str, optional): Scale for the x-axis ('linear' or 'log'). Defaults to 'linear'.
            
        Returns:
            numpy.ndarray: The figure as a numpy array for saving to a file.
        """
        with mne.viz.use_browser_backend('matplotlib'):
            #print(type(raw))
            fig = psd_object.plot(picks='eeg', xscale=xscale, dB=True, show=False)
            fig = FigureCanvas(fig)
            fig.draw()
            fig = np.asarray(fig.buffer_rgba())

        return fig
    
    def get_spectrum(self, folder_path : str, file_name : str) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Compute power spectrum density for a single EEG data file.
        
        This method loads an EEG data file, computes the power spectral density using 
        the specified method, and saves the results to disk. If plots is True, it also 
        generates and saves a visualization of the PSD.
        
        Args:
            folder_path (str): Path to the folder containing the EEG data file.
            file_name (str): Name of the EEG data file (.npy).
            
        Returns:
            tuple: Contains:
                - torch.Tensor: The spectral data (PSD values).
                - torch.Tensor: The frequency values corresponding to the PSD data.
                - str: The participant ID and condition.
                
        Note:
            This method prints information about data shapes and bad channels for debugging.
        """
        participant_id, condition = get_participant_id_condition_from_string(file_name)     
        data = np.load(folder_path / file_name, allow_pickle=True) 
        # from the epoched data, compute the psd
        # check shape of the epoched and full time series data
        

        if self.full_time_series:
            if self.include_bad_channels:
                data.preprocessed_raw.info['bads'] = []
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
            if self.include_bad_channels:
                data.preprocessed_epochs.info['bads'] = []
            psd = data.preprocessed_epochs.compute_psd(method=self.method,
                                                fmin=self.fmin,
                                                fmax=self.fmax, 
                                                tmin=self.tmin,
                                                tmax=self.tmax,
                                                picks=self.picks,
                                                exclude=self.exclude,
                                                proj=self.proj,
                                                verbose=self.verbose)
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
                
    def run_spectrum_parallel(self) -> None:
        """ 
        Compute power spectrum density for all participants and conditions in parallel.
        
        This method uses multiprocessing to parallelize the computation of PSD features
        across all EEG data files. It leverages the get_spectrum method for individual
        file processing and uses a process pool to distribute the workload.
        
        The method sets ran_spectrum to True to indicate that spectrum computation has been
        performed, preventing redundant calculations when __getitem__ is called.
        """
        self.ran_spectrum = True
        processes = cpu_count() - 1
        print(f'Using {processes} processes for spectrum computation')
        with Pool(processes) as p:
            list(tqdm(p.starmap(self.get_spectrum, self.folders_and_files), 
                     total=len(self.folders_and_files), 
                     desc="Computing spectrums"))

if __name__ == "__main__":
    # Find the path to the cleaned data from root directory
    cleaned_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'cleaned'
    # find the path to the labels file data
    labels_file = Path(__file__).resolve().parent.parent.parent / 'data' / 'TDBRAIN_participants_V2.xlsx'
    dataset = PowerSpectrum(cleaned_path=cleaned_path,
                            get_labels=True,
                            include_bad_channels=True,
                            full_time_series=True,
                            method='multitaper',
                            plots=True,
                            fmin=0.5,
                            fmax=130,
                            tmin=None,
                            tmax=None,
                            picks=['eeg'],
                            proj=False,
                            verbose=False)
    print(len(dataset))
    for i in range(len(dataset)):
        print(dataset[i][0].shape, dataset[i][1].shape, dataset[i][2]) 
