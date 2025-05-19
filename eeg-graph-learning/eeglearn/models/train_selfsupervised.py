"""Self-supervised EEG training pipeline.

Implementation of a self-supervised training approach for EEG data based on 
Li et al. 2023 (https://ieeexplore.ieee.org/abstract/document/9765326).
This module handles data splitting, model training, and metrics tracking for
both frequency and spatial graph representations.

Functions:
    split_data: Split participants into train/test/validation sets
    train: Train the self-supervised model and save metrics
"""

import os
from pathlib import Path

import torch
from torch import nn
from eeglearn.config import Config

from sklearn.model_selection import train_test_split
from AutoWeight import AutomaticWeightedLoss
from eeglearn.models.model import SelfSupervisedTrain
from eeglearn.features.graphs import Graphs
import pandas as pd

batch_size : int = Config.batch_size
epochs : int = Config.epochs
lr : float = Config.lr
weight_decay : float = Config.weight_decay
device : str = Config.device
cleaned_data_path : Path = Config.cleaned_data_path
energy_path : Path = Config.energy_path
model_weights_dir : Path = Config.model_weights_dir
metrics_dir  : Path = Config.metrics_dir

def split_data() -> None:
    """Split participants into train, validation, and test sets.

    Returns:
        dict: Dictionary with keys 'train', 'valid', 'test' containing
             lists of participant IDs for each set
    """

    all_participants = cleaned_data_path
    N = os.listdir(all_participants)
    train, test_valid = train_test_split(N, test_size=0.2, random_state=42)
    test, valid = train_test_split(test_valid, test_size=0.5, random_state=42)

    data_dict = {
        "train" : train,
        "test" : test,
        "valid" : valid
    }

    return data_dict

def train() -> None:
    """Train the self-supervised model on frequency and spatial graphs.
    
    Loads data, trains the model with both frequency and spatial losses,
    tracks performance metrics, and saves model weights at specified epochs.
    """
    print(f"⚠️ Saving weights to {model_weights_dir}")
    if not os.path.exists(model_weights_dir):
        model_weights_dir.mkdir(exist_ok=True)

    print(f"⚠️ Saving metrics to {metrics_dir}")
    if not os.path.exists(metrics_dir):
        metrics_dir.mkdir(exist_ok=True)
    awl = AutomaticWeightedLoss(2)
    net = SelfSupervisedTrain(5, 32, batch_size, HF = 120, HS = 128)
    net = net.to(device)
    print(net)
    # check the device
    print(f"⚠️ Training on device : {device}")
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr,
                                 weight_decay  = weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode = "min",
                                                           factor = 0.1,
                                                           patience= 5,
                                                           threshold = 0.0001,
                                                           threshold_mode = 'rel',
                                                           cooldown = 0,
                                                           min_lr = 0,
                                                           eps = 1e-8)
    split : dict[str,list[str]] = split_data()
    train_participants = split['train']
    validation_participants = split['valid']
    test_participants = split['test']
    print("⚠️ Participants split..")
    print(f"n train : {len(train_participants)}")
    print(f"n valid : {len(validation_participants)}")
    print(f"n test : {len(test_participants)}")

    train_freq_data = [data for participant in train_participants
                       for data in os.listdir(energy_path / "frequency_perms")
                       if participant in data] 
    
    train_spatial_data = [data for participant in train_participants
                       for data in os.listdir(energy_path / "spatial_perms")
                       if participant in data]
    
    spatial_graphs = Graphs(perm_type = "spatial",
                                energy_path= energy_path,
                                distance="ellipsoid",
                                cleaned_data_path=cleaned_data_path,
                                batch_size=batch_size,
                                n_neighbors=3,
                                shuffle=True,
                                drop_last=True,
                                n_workers=7)
    
    frequency_graphs = Graphs(perm_type = "frequency",
                            energy_path= energy_path,
                            distance="ellipsoid",
                            cleaned_data_path=cleaned_data_path,
                            batch_size=batch_size,
                            n_neighbors=3,
                            shuffle=True,
                            drop_last=True,
                            n_workers=7)
    
    metrics : dict[str,list] = {
        'epoch' : [],
        'weighted_loss' : [],
        'freq_loss' : [],
        'spatial_loss' : [],
        'freq_acc' : [],
        'spatial_acc' : []
    }
    spatial_graph_loader = spatial_graphs.get_graphs(files_to_load=train_spatial_data)
    frequency_graph_loader = frequency_graphs.get_graphs(files_to_load=train_freq_data)

    for epoch in range(epochs):
        loader = zip(frequency_graph_loader, spatial_graph_loader)
        epoch_weighted_loss = 0.0
        epoch_loss_freq = 0.0
        epoch_loss_spatial = 0.0
        correct_pred_freq = 0
        correct_pred_spatial = 0

        for ind, batch in enumerate(loader):
            freq_data, spatial_data = batch
            #print(f"batch shape : {freq_data}")
            freq_data, spatial_data = freq_data.to(device), spatial_data.to(device)
            freq, spatial, = net(freq_data,spatial_data)

            y_freq, y_spatial = freq_data.y, spatial_data.y
            _, pred_freq = torch.max(freq, dim = 1)
            _, pred_spatial = torch.max(spatial, dim = 1)

            correct_pred_freq += sum([1 for a,b in zip(pred_freq,y_freq) if a == b])
            correct_pred_spatial += sum([1 for a,b in zip(pred_spatial,y_spatial)\
                                         if a == b])

            loss_frequency = criterion(freq, y_freq)
            loss_spatial = criterion(spatial, y_spatial)

            weighted_loss = awl(loss_frequency,loss_spatial)

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
            epoch_weighted_loss += float(weighted_loss.item())
            epoch_loss_freq += float(loss_frequency.item())
            epoch_loss_spatial += float(loss_spatial.item())

        scheduler.step(epoch_weighted_loss)
        denominator = (ind+1)*batch_size

        if epoch%5==0:
            file_name = f"pretrain_epoch_{epoch}"
            torch.save(net.state_dict(), model_weights_dir / file_name)

        #epoch metrics
        epoch_avg_weighted_loss = epoch_weighted_loss/(ind+1)
        epoch_avg_freq_loss = epoch_loss_freq/(ind+1)
        epoch_avg_spatial_loss = epoch_loss_spatial/(ind+1)
        freq_acc = correct_pred_freq/denominator
        spatial_acc = correct_pred_spatial/denominator
        
    
        # Save metrics
        metrics['epoch'].append(epoch)
        metrics['weighted_loss'].append(epoch_avg_weighted_loss)
        metrics['freq_loss'].append(epoch_avg_freq_loss)
        metrics['spatial_loss'].append(epoch_avg_spatial_loss)
        metrics['freq_acc'].append(freq_acc)
        metrics['spatial_acc'].append(spatial_acc)

        print(f'Epoch [{epoch}/{epochs}] \n')
        print(f'Weighted loss [{epoch_avg_weighted_loss:.4f}]  ')
        print(f'Frequency loss[{epoch_avg_freq_loss:.4f}]')
        print(f'Spatial loss[{epoch_avg_spatial_loss:.4f}]')
        print('ACC@1:')
        print(f'fequency ACC[{correct_pred_freq/denominator:.4f}]')
        print(f'spatial ACC[{correct_pred_spatial/denominator:.4f}]')
        print("----------------------------------------------")

    pd.DataFrame(metrics).to_csv(metrics_dir / "training_metrics.csv", index=False)
if __name__ == "__main__":
    train()