import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from eeglearn.config import Config
from eeglearn.preprocess.preprocessing import Preproccesing
from eeglearn.utils.utils import get_details_from_file_name, get_cleaned_data_paths,\
                            load_preprocessed_data, get_labels_dict
from operator import itemgetter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from multiprocessing import cpu_count

from microstructpy.geometry import Ellipsoid
from pygeodesy.ellipsoidalVincenty import Cartesian
from pygeodesy import Ellipsoid as pyg_Ellipsoid
from pygeodesy.ellipsoidalKarney import LatLon as KLatLon
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from AutoWeight import AutomaticWeightedLoss
from eeglearn.models.model import SelfSupervisedTrain, JointlyTrainModel
from eeglearn.features.graphs import Graphs
import json
import pandas as pd
from eeglearn.config import Config
from sklearn.preprocessing import LabelEncoder
from eeglearn.features.graphs import Graphs

batch_size : int = Config.batch_size
epochs : int = Config.epochs
lr : float = Config.lr
weight_decay : float = Config.weight_decay
device : str = Config.device
cleaned_data_path : Path = Config.cleaned_data_path
energy_path : Path = Config.energy_path
model_weights_dir : Path = Config.model_weights_dir / 'jointly'
metrics_dir  : Path = Config.metrics_dir / 'jointly'
ignore_replication_nans : bool = True
"""Self-supervised  + multi-task learning EEG training pipeline .

Implementation of a self-supervised training approach for EEG data based on Li et al. 2023
(https://ieeexplore.ieee.org/abstract/document/9765326). This module handles data splitting,
model training, and metrics tracking for both frequency and spatial graph representations.

Functions:
    split_data: Split participants into train/test/validation sets
    train: Execute the self-supervised training process and save metrics
"""

def split_data(ignore_replication_nans : bool = False) -> None:
    """Create the graph representations of each epoched recording in a collection.

        Args:
        ----
            None
        Returns:
        ----
            dict : A dictionary of keyed by `train`, `valid` or `test` with lists
                  of participant Ids 

    """

    all_participants = cleaned_data_path
    labels = get_labels_dict()
    participant_files = os.listdir(all_participants)
    N = []
    if ignore_replication_nans:
        print(f"⚠️ Ignoring participants with Nan labelcs or in replication")
        for participant in participant_files:
            if labels[participant] == 'nan' or labels[participant] == 'REPLICATION':
                continue
            N.append(participant)
    else:
        N = participant_files
            
    train, test_valid = train_test_split(N, test_size=0.2, random_state=42)
    test, valid = train_test_split(test_valid, test_size=0.5, random_state=42)

    data_dict = {
        "train" : train,
        "test" : test,
        "valid" : valid
    }

    return data_dict

def get_graphs_original(files_to_load : list, label_encoder : LabelEncoder):
    """Load an energy object for each participant and convert it into a graph
    with the psych label. 

            Args:
            ----
                None
            Returns:
            ----
                None

    """
    epoched_path : Path = energy_path / "energy_epoched"
    energy_files : list = os.listdir(epoched_path)
    energy_file_ids : list = { get_details_from_file_name(file)[0] : file
                              for file in energy_files}
    full_file_names_to_load = [energy_file_ids[file] for file in files_to_load]
    #print(full_file_names_to_load)
    print(len(full_file_names_to_load))
    graphs =  Graphs(
                    perm_type=None,
                    energy_path=energy_path / "energy_epoched",
                    distance="ellipsoid", 
                     cleaned_data_path=cleaned_data_path)
    graphs.get_graphs(full_file_names_to_load, label_encoder)
    # graphs : list = []
    # for file in files_to_load:
    #     if file in energy_file_ids:
    #         #print(file, energy_file_ids[file])
    #         x, channels, subject, condition, y = torch.\
    #                                         load( epoched_path / energy_file_ids[file])
    #         #print(data)

    # #print(energy_objects[0])

def test() -> None:
    """Periodically evaluate the model and save the best performing one.

            Args:
            ----
                None
            Returns:
            ----
                None

    """
    pass
def train()-> None :
    """Train the self-supervised model jointly on the pre-text and the downstream task

        Args:
        ----
            None
        Returns:
        ----
            None

    """
    highest_acc = 0.0
    print(f"⚠️ Saving weights to {model_weights_dir}")
    if not os.path.exists(model_weights_dir):
        model_weights_dir.mkdir(exist_ok=True)

    print(f"⚠️ Saving metrics to {metrics_dir}")
    if not os.path.exists(metrics_dir):
        metrics_dir.mkdir(exist_ok=True)

    psych_labels = get_labels_dict()
    unique_labels = set(psych_labels.values())
    if ignore_replication_nans:
        unique_labels = [label for label in unique_labels
                         if not (label == "REPLICATION" or label == "nan")]
    encoder = LabelEncoder()
    encoder.fit(list(unique_labels))
    n_classes = len(unique_labels)
    awl = AutomaticWeightedLoss(3)
    net = JointlyTrainModel(5, 32, batch_size, HF = 120, HS = 128, HC =  n_classes)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           factor=0.1, 
                                                           patience=4,
                                                           threshold=0.0001,
                                                           threshold_mode='rel',
                                                           cooldown=1, min_lr=0,
                                                           eps=1e-8)
    split : dict[str,list[str]] = split_data(ignore_replication_nans=\
                                             ignore_replication_nans)
    train_participants = split['train']
    validation_participants = split['valid']
    test_participants = split['test']
    print("⚠️ Participants split..")
    print(f"n train : {len(train_participants)}")
    print(f"n valid : {len(validation_participants)}")
    print(f"n test : {len(test_participants)}")

    graphs_original = get_graphs_original(train_participants, encoder)
    # train_freq_data = [data for participant in train_participants
    #                    for data in os.listdir(energy_path / "frequency_perms")
    #                    if participant in data] 
    
    # train_spatial_data = [data for participant in train_participants
    #                    for data in os.listdir(energy_path / "spatial_perms")
    #                    if participant in data]
    
    # spatial_graphs = Graphs(perm_type = "spatial",
    #                             energy_path= energy_path,
    #                             distance="ellipsoid",
    #                             cleaned_data_path=cleaned_data_path,
    #                             batch_size=batch_size,
    #                             n_neighbors=3,
    #                             shuffle=True,
    #                             drop_last=True,
    #                             n_workers=7)
    
    # frequency_graphs = Graphs(perm_type = "frequency",
    #                         energy_path= energy_path,
    #                         distance="ellipsoid",
    #                         cleaned_data_path=cleaned_data_path,
    #                         batch_size=batch_size,
    #                         n_neighbors=3,
    #                         shuffle=True,
    #                         drop_last=True,
    #                         n_workers=7)
    # spatial_graph_loader = spatial_graphs.get_graphs(files_to_load=train_spatial_data)
    # frequency_graph_loader = frequency_graphs.get_graphs(files_to_load=train_freq_data)


    metrics : dict[str,list] = {
            'epoch' : [],
            'weighted_loss' : [],
            'freq_loss' : [],
            'spatial_loss' : [],
            'freq_acc' : [],
            'spatial_acc' : []
        }
if __name__ == "__main__":
    train()