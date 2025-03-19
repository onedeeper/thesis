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

class DiffEntropy(Dataset):
    """
    Dataset class for computing and loading diff entropy features from EEG data.
    """
    
    def __init__(self, 
                 cleaned_path : str,
                 freq_bands : list[int],
                 get_labels : bool = True,
                 plots : bool = False
                 ):
        self.cleaned_path = cleaned_path
        self.get_labels = get_labels
        self.plots = plots
        self.verbose = verbose
        self.labels_dict = get_labels_dict(labels_file)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
    
    def plot_diff_entropy(self,):
        pass

    def get_diff_entropy(self, folder_path, file_name):
        pass

    def run_diff_entropy_parallel(self):
        pass

if __name__ == "__main__":
    cleaned_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'cleaned'
    labels_file = Path(__file__).resolve().parent.parent.parent / 'data' / 'TDBRAIN_participants_V2.xlsx'
    dataset = DiffEntropy(cleaned_path=cleaned_path,
                          get_labels=True,
                          plots=True,
                          verbose=False)
    print(len(dataset))