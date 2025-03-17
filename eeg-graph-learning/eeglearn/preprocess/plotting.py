"""
Created on Mon Mar 11 2024

author: T.Smolders

description: Functions for plotting the raw data, power spectral density and
             time frequency decomposition at different steps in the
             preprocessing pipeline

name: plotting.py

version: 1.0

"""
import numpy as np
import pandas as pd
import mne
from mne.time_frequency import tfr_multitaper
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib

matplotlib.use('Agg')  # use backend so plots dont get displayed while code is running

mne.set_log_level('WARNING')


def get_plots(raw, step, scalings={'eeg': 1.5, 'eog': 'auto', 'emg': 'auto', 'ecg': 'auto'},
              xscale='linear', baseline_correction=False, channel=[6], plot_ica_overlay=False, ica=''):
    """
    Functions to plot raw data, power spectral density, time frequency
    decomposition and optionally the reconstructed signal after ICA with the
    artifactual ICs highlighted.

    Parameters:
    - raw: raw data object
    - step: string, name of the preprocessing step
    - scalings: dictionary, scaling factors for each channel type
    - xscale: string, scale of the x-axis in the power spectral density plot
    - baseline_correction: boolean, whether to apply baseline correction to the time frequency decomposition
    - channel: list index, channel number to plot the time frequency decomposition for
    - plot_ica: boolean, whether to plot the ICA results
    - ica: object, ICA object of the raw data

    Returns:
    - fig: figure object containing the raw data, power spectral density and time frequency decomposition
    """

    def plot_raw(raw, scalings={'eeg': 1.5, 'eog': 'auto', 'emg': 'auto', 'ecg': 'auto'}):
        ## plotting raw data
        with mne.viz.use_browser_backend('matplotlib'):
            fig = raw.plot(n_channels=33, scalings=scalings, title='title', show_scrollbars=False, show=False)
            fig = FigureCanvas(fig)
            fig.draw()
            fig = np.asarray(fig.buffer_rgba())
        return fig

    def plot_psd(raw, xscale='linear'):
        ## plotting power spectral density
        with mne.viz.use_browser_backend('matplotlib'):
            fig = raw.compute_psd(fmin=0.5, fmax=130).plot(picks='eeg', xscale=xscale, dB=True, show=False)
            fig = FigureCanvas(fig)
            fig.draw()
            fig = np.asarray(fig.buffer_rgba())

        return fig

    def plot_tfr(raw, axes, channel=[6], baseline_correction=False):
        ## plotting time frequency decomposition
        # define frequencies of interest
        freqs = np.array([  # 5 steps per frequency band
            0.5, 1.125, 1.75, 2.375, 3,  # delta
            4, 4.75, 5.5, 6.25, 7,  # theta
            8, 9, 10, 11, 12,  # alpha
            13, 17.25, 21.5, 25.75, 30,  # beta
            42, 54, 66, 78, 90  # gamma
            ])
        n_cycles = freqs / 2.0  # different number of cycles per frequency

        # create epochs & tfr object
        epochs = mne.make_fixed_length_epochs(raw, duration=9.95, overlap=0)
        tfr_mt = tfr_multitaper(
                        epochs,
                        freqs=freqs,
                        n_cycles=n_cycles,
                        use_fft=True,
                        return_itc=False,
                        average=True,
                        decim=1,  # decim reduces sampling rate of the tf decomposition by the defined factor
                        )

        if baseline_correction == True:
            tfr_mt.apply_baseline((0, 0.5), mode='mean')  # baseline correction

        # plot time frequency decomposition
        tfr_mt.plot(channel, tmin=1, tmax=8, axes=axes, show=False)  # plot average TFR for the nth channel

        return

    def plot_ica(raw, ica):
        ## plotting ICA components
        with mne.viz.use_browser_backend('matplotlib'):
            fig = ica.plot_overlay(raw, picks='eeg', show=False)
            fig = FigureCanvas(fig)
            fig.draw()
            fig = np.asarray(fig.buffer_rgba())

        return fig

    ## create combined figure
    fig_raw = plot_raw(raw, scalings=scalings)
    fig_psd = plot_psd(raw, xscale=xscale)

    if plot_ica_overlay == True:
        fig_ica = plot_ica(raw, ica)
        fig, axes = plt.subplot_mosaic(
            [['ax_raw', 'ax_raw', 'ax_tfr'],
            ['ax_psd', 'ax_psd', 'ax_psd'],
            ['ax_ica', 'ax_ica', 'ax_ica']],
            figsize=(25, 25)
        )
        axes['ax_raw'].imshow(fig_raw)
        axes['ax_raw'].axis('off')
        axes['ax_ica'].imshow(fig_ica)
        axes['ax_ica'].axis('off')
        axes['ax_psd'].imshow(fig_psd)
        plot_tfr(raw, axes=axes['ax_tfr'], channel=channel)
        axes['ax_tfr'].set_title(f'{channel = }', fontsize=20)
        axes['ax_raw'].set_title(f'{scalings = }', fontsize=20)
        axes['ax_ica'].set_title('before ICA (red) and after ICA (black)', fontsize=20)
        fig.suptitle(step, fontsize=40)

    if plot_ica_overlay == False:
        fig, axes = plt.subplot_mosaic(
            [['ax_raw', 'ax_raw', 'ax_tfr'],
            ['ax_psd', 'ax_psd', 'ax_psd']],
            figsize=(25, 25)
        )
        axes['ax_raw'].imshow(fig_raw)
        axes['ax_raw'].axis('off')
        axes['ax_psd'].imshow(fig_psd)
        plot_tfr(raw, axes=axes['ax_tfr'], channel=channel)
        axes['ax_tfr'].set_title(f'{channel = }', fontsize=20)
        axes['ax_raw'].set_title(f'{scalings = }', fontsize=20)
        fig.suptitle(step, fontsize=40)

    return fig