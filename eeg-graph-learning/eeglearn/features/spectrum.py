import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path


class PowerSpectrum(Dataset):
    def __init__(self, torch_path, full_time_series=False):
        """
        Initialize the dataset, without loading everything into memory.
        """
        self.data_path = torch_path
        self.file_list = os.listdir(self.data_path)

    def __len__(self):
        return len(os.listdir(self.data_path))
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset as needed to save memory.
        """
        file_path = Path(self.data_path) / self.file_list[idx] / 'ses-1' / 'task-restEO' / 'sub-19694366_ses-1_task-restEO_preprocessed.npy'
        data = torch.load(file_path)
        return data
    
    def get_spectrum(self, data):
        """
        Get the spectrum of the data.
        """
        return torch.fft.fft(data)
    

if __name__ == "__main__":
    dataset = PowerSpectrum(torch_path="/Users/udeshhabaraduwa/Library/CloudStorage/GoogleDrive-u.habaraduwakandambige@tilburguniversity.edu/My Drive/Tilburg/Masters CSAI/Semester 4/code/thesis/eeg-graph-learning/data/cleaned",
                            full_time_series=True)
    print(len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    print(dir(train_loader))