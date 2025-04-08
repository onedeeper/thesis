"""
Created on Sun Apr 6 2025

author: Udesh Habaraduwa

description: plotting functions for the TDBRAIN dataset

"""

import matplotlib.pyplot as plt
import numpy as np
import mne
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import mne
from pathlib import Path

def plot_psd(psd : mne.time_frequency.Spectrum,
                 plot_save_dir : Path,
                 xscale : str = 'linear',
                 participant_id : str = None,
                 condition : str = None) -> plt.Figure:
        """
        Plot the power spectral density (PSD) of an EEG dataset.
        
        This method uses MNE's plotting functions to visualize the PSD of the EEG data.
        It allows for customization of the x-axis scale and displays PSD in decibel
          (dB) units.
        
        Args:
            psd_object (mne.time_frequency.Spectrum): The PSD object from MNE.
            xscale (str): Scale for the x-axis ('linear' or 'log').
            
        Returns:
            None
        """
        with mne.viz.use_browser_backend('matplotlib'):
            # Create the PSD plot
            fig = psd.plot(picks='eeg', xscale=xscale, dB=True, show=False)
            
            # Remove gridlines from the original MNE plot
            for ax in fig.get_axes():
                ax.grid(False)  # Turn off the grid
            
            # Convert to image
            fig = FigureCanvas(fig)
            fig.draw()
            fig = np.asarray(fig.buffer_rgba())
            
            # Display and save the image
            plt.figure(figsize=(10, 6))
            plt.imshow(fig)
            plt.axis('off')  # Turn off the axis
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove margins
            plt.tight_layout(pad=0)  # Remove padding
            plt.savefig(f'{plot_save_dir}/psd_{participant_id}_{condition}.png',
                        dpi=300, 
                        bbox_inches='tight',  # Tight bounding box
                        pad_inches=0)  # No padding
            plt.close()