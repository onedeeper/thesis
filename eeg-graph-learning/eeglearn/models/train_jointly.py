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
from eeglearn.models.model import SelfSupervisedTrain, JointlyTrainModel,\
    SelfSupervisedTest
from eeglearn.features.graphs import Graphs
import json
import pandas as pd
from eeglearn.config import Config
from sklearn.preprocessing import LabelEncoder
from eeglearn.features.graphs import Graphs
from torch_geometric.data import Batch

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
num_workers : int = Config.num_workers
random_seed : int = Config.RANDOM_SEED
drop_last : bool = Config.drop_last

from itertools import cycle


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
        print(f"⚠️ Ignoring participants with Nan labels or in replication")
        for participant in participant_files:
            try:
                if labels[participant] in {'nan', 'NaN', np.nan, 'REPLICATION'}:
                    continue
            except KeyError:
                continue 
            N.append(participant)
    else:
        N = participant_files
            
    train, test_valid = train_test_split(N, test_size=0.2, random_state=random_seed)
    test, valid = train_test_split(test_valid, test_size=0.5, random_state=random_seed)

    data_dict = {
        "train" : train,
        "test" : test,
        "valid" : valid
    }

    return data_dict

def get_graphs_original(files_to_load : list, label_encoder : LabelEncoder, 
                        testing : bool = False):
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
    energy_file_ids : dict = { get_details_from_file_name(file)[0] : file
                              for file in energy_files}
    # full_file_names_to_load = [energy_file_ids[file] for file in files_to_load]

    full_file_names_to_load = [energy_file_ids[file] \
                                for file in files_to_load \
                                if file in energy_file_ids]
    graphs =  Graphs(
                    perm_type=None,
                    energy_path=energy_path / "energy_epoched",
                    distance="ellipsoid", 
                     cleaned_data_path=cleaned_data_path,
                     n_workers=num_workers,
                     drop_last=drop_last)
    if testing:
        graphs =  Graphs(
                    perm_type=None,
                    energy_path=energy_path / "energy_epoched",
                    distance="ellipsoid", 
                     cleaned_data_path=cleaned_data_path,
                     n_workers=num_workers,
                     drop_last=False)
    return graphs.get_graphs(files_to_load=full_file_names_to_load, 
                             label_encoder= label_encoder)

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
        
    print(f"⚠️ Training with data loader drop_last : {drop_last}")
    # Create necessary log files
    with open(metrics_dir / "epoch_log.txt", "w") as f:
        f.write("batch_size\tepoch\tlr\tdrop_rate\tacc\n")
    with open(metrics_dir / "update_log.txt", "w") as f:
        f.write("epoch\tlr\tbatch_size\tacc\n")

    psych_labels = get_labels_dict()
    unique_labels = list(set(psych_labels.values()))
    if ignore_replication_nans:
        unique_labels = sorted([label for label in unique_labels
                         if not (label == "REPLICATION" or label == "nan")])
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

    original_graph_loader = get_graphs_original(train_participants, encoder)
    train_freq_data = [fname for participant in train_participants
                       for fname in os.listdir(energy_path / "frequency_perms")
                       if participant in fname] 
    
    train_spatial_data = [fname for participant in train_participants
                       for fname in os.listdir(energy_path / "spatial_perms")
                       if participant in fname]
    
    spatial_graphs = Graphs(perm_type = "spatial",
                                energy_path= energy_path,
                                distance="ellipsoid",
                                cleaned_data_path=cleaned_data_path,
                                batch_size=batch_size,
                                n_neighbors=3,
                                shuffle=True,
                                drop_last=drop_last,
                                n_workers=num_workers)
    
    frequency_graphs = Graphs(perm_type = "frequency",
                            energy_path= energy_path,
                            distance="ellipsoid",
                            cleaned_data_path=cleaned_data_path,
                            batch_size=batch_size,
                            n_neighbors=3,
                            shuffle=True,
                            drop_last=drop_last,
                            n_workers=num_workers)
    spatial_graph_loader = spatial_graphs.get_graphs(files_to_load=train_spatial_data)
    frequency_graph_loader = frequency_graphs.get_graphs(files_to_load=train_freq_data)

    metrics : dict[str,list] = {
            'epoch' : [],
            'weighted_loss' : [],
            'freq_loss' : [],
            'spatial_loss' : [],
            'original_loss' : [],
            'freq_acc' : [],
            'spatial_acc' : [],
            'original_acc': [],
        }
    for epoch in range(2):
        loader = zip(frequency_graph_loader, spatial_graph_loader,
                     cycle(original_graph_loader))
        epoch_weighted_loss = 0.0
        epoch_loss_freq = 0.0
        epoch_loss_spatial = 0.0
        epoch_loss_original = 0.0
        correct_pred_freq = 0
        correct_pred_spatial = 0
        correct_pred_original  = 0

        for ind, batch in enumerate(loader):
            fdata, sdata, gdata = batch
            fdata, sdata, gdata = fdata.to(device),\
                                 sdata.to(device),\
                                 gdata.to(device)
            freq_out, spatial_out, original_out, = net(fdata, sdata, gdata)
            # the true pseudo labels and true connectivity graphs
            y_freq, y_spatial, y_original = fdata.y, sdata.y, gdata.y
            _, pred1 = torch.max(freq_out, dim=1)
            _, pred2 = torch.max(spatial_out, dim=1)
            _, pred3 = torch.max(original_out, dim=1)

            correct_pred_freq += sum([1 for a,b in zip(pred1, y_freq) if a==b])
            correct_pred_spatial += sum([1 for a,b in zip(pred2, y_spatial) if a==b])
            correct_pred_original += sum([1 for a,b in zip(pred3, y_original) if a==b])
            loss_freq = criterion(freq_out, y_freq)
            loss_spatial = criterion(spatial_out, y_spatial)
            loss_original = criterion(original_out, y_original)
            # balanced loss from the multiple tasks. 
            loss = awl(loss_freq, loss_spatial, loss_original)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_weighted_loss += float(loss.item())
            epoch_loss_freq += float(loss_freq.item())
            epoch_loss_spatial += float(loss_spatial.item())
            epoch_loss_original += float(loss_original.item())
        highest_acc, current_acc = validate(
                                        validate_data=validation_participants,
                                        net = net,
                                        label_encoder= encoder, 
                                        highest_acc=highest_acc,
                                        epoch=epoch)
        
        writeEachEpoch(epoch, batch_size, lr, current_acc)
        scheduler.step(epoch_weighted_loss) 
        denominator = (ind+1)*batch_size
        if epoch % 5 == 0:
                print()
                print(f'-----highest_acc {highest_acc:.4f} current_acc {current_acc:.4f}-----')
                print(f'batch {batch_size}, lr {lr}')
                print()

        epoch_avg_weighted_loss = epoch_weighted_loss/(ind+1)
        epoch_avg_freq_loss = epoch_loss_freq/(ind+1)
        epoch_avg_spatial_loss = epoch_loss_spatial/(ind+1)
        epoch_avg_original_loss = epoch_loss_original/(ind+1)
        freq_acc = correct_pred_freq/denominator
        spatial_acc = correct_pred_spatial/denominator
        original_acc = correct_pred_original/denominator
        # Save metrics
        metrics['epoch'].append(epoch)
        metrics['weighted_loss'].append(epoch_avg_weighted_loss)
        metrics['freq_loss'].append(epoch_avg_freq_loss)
        metrics['spatial_loss'].append(epoch_avg_spatial_loss)
        metrics['original_loss'].append(epoch_avg_original_loss)
        metrics['freq_acc'].append(freq_acc)
        metrics['spatial_acc'].append(spatial_acc)
        metrics['original_acc'].append(original_acc)

        print(f'Epoch [{epoch}/{epochs}] \\n')
        print(f'Weighted loss [{epoch_avg_weighted_loss:.4f}]  ')
        print(f'Frequency loss[{epoch_avg_freq_loss:.4f}]')
        print(f'Spatial loss[{epoch_avg_spatial_loss:.4f}]')
        print(f'Original loss[{epoch_avg_original_loss:.4f}]')
        print(f'ACC@1:')
        print(f'fequency ACC[{correct_pred_freq/denominator:.4f}]')
        print(f'spatial ACC[{correct_pred_spatial/denominator:.4f}]')
        print(f'original ACC[{correct_pred_original/denominator:.4f}]')
        print("----------------------------------------------")
            
    pd.DataFrame(metrics).to_csv(metrics_dir / "training_metrics_jointly.csv",
                                 index=False)


def writeEachEpoch(epoch, batchsize, lr, current_acc):
    drop_rate = Config.drop_rate
    log = []
    log.append(f'{batchsize}\t{epoch}\t{lr}\t{drop_rate}\t{current_acc:.4f}\n')
    with open(metrics_dir / "epoch_log.txt", 'a') as f:
        f.writelines(log)


def updatelog(epoch, acc):
    log = []
    log.append(f'{epoch}\t{lr}\t{batch_size}\t{acc:.4f}\n')
    with open(metrics_dir / "update_log.txt", 'a') as f:
        f.writelines(log)


def validate(net, validate_data, label_encoder, highest_acc, epoch):
    criterion = nn.CrossEntropyLoss().to(device)
    gloader = get_graphs_original(validate_data, 
                                  label_encoder=label_encoder,
                                  testing=True)
    net.testmode = True
    net.eval()
    epoch_loss = 0.0
    correct_pred = 0
    total_samples = 0
    for ind, data in enumerate(gloader):
        data = data.to(device)
        current_batch_size = data.y.size(0)
        total_samples += batch_size

        if current_batch_size < batch_size:
            data_list = data.to_data_list()
            needed = batch_size - current_batch_size
            additional_samples = [data_list[i % current_batch_size]\
                                  for i in range(needed)]
            data_list.extend(additional_samples)
            data = Batch.from_data_list(data_list)
        out = net(data)
        y = data.y
        _, pre = torch.max(out, dim=1)

        correct_pred += sum([1 for a, b in zip(pre, y) if a == b])
        loss = criterion(out, y)

        epoch_loss += float(loss.item())

    ACC = correct_pred / total_samples
    if ACC > highest_acc:
        updatelog(epoch = epoch,acc=ACC)
        highest_acc = ACC
        ck = {}
        ck['epoch'] = epoch
        ck['model'] = net.state_dict()
        ck['ACC'] = ACC
        
        torch.save(ck, model_weights_dir / "checkpoint.pkl")

    net.train()
    net.testmode=False
    return highest_acc, ACC

if __name__ == "__main__":
    train()