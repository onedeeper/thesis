"""
Model utilities for the eeglearn package.

This module provides reusable utility functions for model training, data handling,
and validation across different training scripts.

Created on: March 2025
Author: Udesh Habaraduwa
"""

import os
from pathlib import Path
from itertools import cycle

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from eeglearn.config import Config
from eeglearn.utils.utils import get_details_from_file_name, get_labels_dict
from eeglearn.features.graphs import Graphs


def split_data(ignore_replication_nans: bool = False) -> dict:
    """Split participants into train, validation, and test sets using stratified 
    sampling.

    Args:
        ignore_replication_nans: Whether to exclude participants with NaN labels 
                                 or in replication status.

    Returns:
        Dictionary with keys 'train', 'valid', 'test' containing lists of
          participant IDs.
    """
    labels = get_labels_dict()
    participant_files = os.listdir(Config.cleaned_data_path)
    
    if ignore_replication_nans:
        print("⚠️  Ignoring participants with Nan labels or in replication")
        valid_participants = []
        valid_labels = []
        
        for participant in participant_files:
            try:
                label = labels[participant]
                if label in {'nan', 'NaN', np.nan, 'REPLICATION'} \
                    or label not in Config.main_classes:
                    continue
                valid_participants.append(participant)
                valid_labels.append(label)
            except KeyError:
                continue
    else:
        valid_participants = participant_files
        valid_labels = [labels[p] for p in valid_participants]
    
    for participant in valid_participants:
        assert labels[participant] in Config.main_classes

    train, test_valid, train_labels, test_valid_labels = train_test_split(
        valid_participants, valid_labels, 
        test_size=0.2, 
        random_state=Config.RANDOM_SEED,
        stratify=valid_labels
    )

    test, valid, _, _ = train_test_split(
        test_valid, test_valid_labels,
        test_size=0.5,
        random_state=Config.RANDOM_SEED,
        stratify=test_valid_labels
    )

    splits = [("Train", train), ("Valid", valid), ("Test", test)]
    for split_name, split_data in splits:
        split_classes = set(labels[p] for p in split_data)
        assert split_classes == set(Config.main_classes), \
            f"Not all classes present in {split_name.lower()} set"

    print("Class distribution:")
    for split_name, split_data in splits:
        class_counts = {c: sum(1 for p in split_data if labels[p] == c) 
                        for c in Config.main_classes}
        print(f"{split_name} set class counts:", class_counts)

    return {"train": train, "test": test, "valid": valid}


def get_graphs_original(files_to_load: list, label_encoder: LabelEncoder, batch_size: 
                        int, testing: bool = False):
    """Load energy objects for participants and convert them into graphs with labels.

    Args:
        files_to_load: List of participant files to load.
        label_encoder: Label encoder for psychological labels.
        batch_size: Batch size for data loading.
        testing: Whether the graphs are for testing (affects drop_last setting).

    Returns:
        PyTorch geometric graph data loader.
    """
    epoched_path = Config.energy_path / "energy_epoched"
    energy_files = os.listdir(epoched_path)
    energy_file_ids = {get_details_from_file_name(file)[0]: file 
                       for file in energy_files}
    
    full_file_names_to_load = [energy_file_ids[file] 
                              for file in files_to_load 
                              if file in energy_file_ids]
    
    graphs = Graphs(
        perm_type=None,
        energy_path=epoched_path,
        distance="ellipsoid", 
        cleaned_data_path=Config.cleaned_data_path,
        n_workers=Config.num_workers,
        drop_last=not testing,
        batch_size=batch_size
    )
    
    return graphs.get_graphs(
        files_to_load=full_file_names_to_load, 
        label_encoder=label_encoder,
        skip_bads=Config.skip_bads
    )


def create_graph_loaders(participants: list, encoder: LabelEncoder, batch_size: int):
    """Create and cache graph loaders for different permutation types."""
    loaders = {}
    
    loader_path = Config.project_root / 'eeglearn' / 'models' /\
          'original_graph_loader.pt'
    if not loader_path.exists() or Config.optuna:
        loaders['original'] = get_graphs_original(participants, encoder, batch_size)
        torch.save(loaders['original'], loader_path)
    else:
        loaders['original'] = torch.load(loader_path)
    
    perm_types = ['spatial', 'frequency']
    for perm_type in perm_types:
        loader_path = Config.project_root / 'eeglearn' / 'models' / \
            f'{perm_type}_graph_loader.pt'
        
        if not loader_path.exists() or Config.optuna:
            data_files = [fname for participant in participants
                         for fname in os.listdir(Config.energy_path/\
                                                 f"{perm_type}_perms")
                         if participant in fname]
            
            graphs = Graphs(
                perm_type=perm_type,
                energy_path=Config.energy_path,
                distance="ellipsoid",
                cleaned_data_path=Config.cleaned_data_path,
                batch_size=batch_size,
                n_neighbors=3,
                shuffle=True,
                drop_last=Config.drop_last,
                n_workers=Config.num_workers
            )
            
            loaders[perm_type] = graphs.get_graphs(files_to_load=data_files, 
                                                   skip_bads=Config.skip_bads)
            torch.save(loaders[perm_type], loader_path)
        else:
            loaders[perm_type] = torch.load(loader_path)
    
    return loaders


def print_training_params():
    """Pretty print training parameters."""
    params = {
        'Batch Size': Config.batch_size,
        'Epochs': Config.epochs,
        'Learning Rate': Config.lr,
        'Weight Decay': Config.weight_decay,
        'Dropout Rate': Config.drop_rate,
        'GCN Output Size': Config.gcn_out_size,
        'Linear Layer Size': Config.linear_size,
        'Chebyshev Order (K)': Config.K,
        'Early Stopping Patience': Config.stop_at,
        'Optuna Mode': Config.optuna
    }
    
    print("🚀 Training Parameters:")
    for param, value in params.items():
        print(f"   {param}: {value}")
    print()


def setup_directories(model_weights_dir: Path, metrics_dir: Path):
    """Create necessary directories and log files.
    
    Args:
        model_weights_dir: Directory to save model weights
        metrics_dir: Directory to save training metrics
    """
    required_paths = [Config.cleaned_data_path, Config.energy_path, Config.data_path]
    path_descriptions = ["'data/cleaned'", "'data/energy'", "'data'"]
    
    for path, desc in zip(required_paths, path_descriptions):
        assert path.exists(), f"{desc} folder should exist."
    
    for dir_path, name in [(model_weights_dir, "weights"), (metrics_dir, "metrics")]:
        print(f"⚠️  Saving {name} to {dir_path}")
        dir_path.mkdir(exist_ok=True, parents=True)
    
    print(f"⚠️  Training with data loader drop_last: {Config.drop_last}")
    
    log_files = [
        (metrics_dir / "epoch_log.txt", "batch_size\tepoch\tlr\tdrop_rate\tacc\n"),
        (metrics_dir / "update_log.txt", "epoch\tlr\tbatch_size\tacc\n")
    ]
    
    for log_file, header in log_files:
        with open(log_file, "w") as f:
            f.write(header)


def setup_label_encoder(ignore_replication_nans: bool = True):
    """Setup label encoder for psychological labels.
    
    Args:
        ignore_replication_nans: Whether to ignore NaN labels and replications
        
    Returns:
        Tuple of (encoder, n_classes)
    """
    all_psych_labels = get_labels_dict()
    all_unique_labels = list(set(all_psych_labels.values()))
    
    if ignore_replication_nans:
        selected_labels = sorted([
            label for label in all_unique_labels 
            if label not in {'nan', 'NaN', np.nan, 'REPLICATION'} and label in
              Config.main_classes
        ])
    else:
        selected_labels = sorted([ label for label in all_unique_labels 
                                  if label in Config.main_classes])
    
    encoder = LabelEncoder()
    encoder.fit(selected_labels)
    return encoder, len(selected_labels)


def calculate_class_weights(train_participants: list, all_psych_labels: dict, 
                            encoder: LabelEncoder, n_classes: int):
    """Calculate class weights for balanced training.
    
    Args:
        train_participants: List of training participant IDs
        all_psych_labels: Dictionary mapping participant IDs to labels
        encoder: Fitted label encoder
        n_classes: Number of classes
        
    Returns:
        Tensor of class weights for loss function
    """
    train_labels = np.array([all_psych_labels[p] for p in train_participants])
    train_labels_encoded = encoder.transform(train_labels)
    
    class_frequencies = np.bincount(train_labels_encoded, minlength=n_classes).\
        astype(float)
    class_weights = 1.0 / class_frequencies
    rescaled_weights = class_weights * (n_classes / class_weights.sum())
    
    return torch.as_tensor(rescaled_weights, dtype=torch.float32, device=Config.device)


def write_epoch_log(epoch: int, batchsize: int, lr: float, current_acc: float, 
                   metrics_dir: Path):
    """Log epoch information to epoch_log.txt.
    
    Args:
        epoch: Current epoch number
        batchsize: Batch size used
        lr: Learning rate used
        current_acc: Current accuracy
        metrics_dir: Directory to save metrics
    """
    drop_rate = Config.drop_rate
    log = f'{batchsize}\t{epoch}\t{lr}\t{drop_rate}\t{current_acc:.4f}\n'
    with open(metrics_dir / "epoch_log.txt", 'a') as f:
        f.write(log)


def update_log(epoch: int, acc: float, lr: float, batch_size: int, metrics_dir: Path):
    """Update the log file with best model information.
    
    Args:
        epoch: Epoch when best model was found
        acc: Best accuracy achieved
        lr: Learning rate used
        batch_size: Batch size used
        metrics_dir: Directory to save metrics
    """
    log = f'{epoch}\t{lr}\t{batch_size}\t{acc:.4f}\n'
    with open(metrics_dir / "update_log.txt", 'a') as f:
        f.write(log)


def validate_model(net, validate_data: list, label_encoder: LabelEncoder, 
                   highest_acc: float, best_f1_score: float, epoch: int, 
                   batch_size: int, lr: float, model_weights_dir: Path, 
                   metrics_dir: Path, testing_on_sample_data: bool = None):
    """Evaluate model performance on validation data.
    
    Args:
        net: Neural network model
        validate_data: List of validation participant IDs
        label_encoder: Label encoder for predictions
        highest_acc: Current highest accuracy
        best_f1_score: Current best F1 score
        epoch: Current epoch number
        batch_size: Batch size for validation
        lr: Learning rate used
        model_weights_dir: Directory to save model weights
        metrics_dir: Directory to save metrics
        testing_on_sample_data: Whether using sample data for testing
        
    Returns:
        Tuple of (highest_acc, current_acc, epoch_loss, f1_score)
    """
    if testing_on_sample_data is None:
        testing_on_sample_data = Config.testing_on_sample_data
        
    criterion = nn.CrossEntropyLoss().to(Config.device)
    gloader = get_graphs_original(validate_data, label_encoder, batch_size, 
                                  testing=testing_on_sample_data)
    
    net.testmode = True
    net.eval()
    
    epoch_loss = 0.0
    correct_pred = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    for _, data in enumerate(gloader):
        data = data.to(Config.device)
        current_batch_size = data.y.size(0)
        total_samples += current_batch_size

        if testing_on_sample_data and (current_batch_size < batch_size):
            data_list = data.to_data_list()
            needed = batch_size - current_batch_size
            additional_samples = [data_list[i % current_batch_size] for i in 
                                  range(needed)]
            data_list.extend(additional_samples)
            data = Batch.from_data_list(data_list)
            total_samples = total_samples - current_batch_size + batch_size
        
        out = net(data)
        y = data.y
        pre = torch.argmax(out, dim=1)

        if testing_on_sample_data and (current_batch_size < batch_size):
            all_preds.extend(pre[:current_batch_size].cpu().numpy())
            all_labels.extend(y[:current_batch_size].cpu().numpy())
        else:
            all_preds.extend(pre.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        correct_pred += torch.sum(pre == y).item()
        loss = criterion(out, y)
        epoch_loss += loss.item()

    ACC = correct_pred / total_samples
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    if ACC > highest_acc:
        update_log(epoch, ACC, lr, batch_size, metrics_dir)
        highest_acc = ACC
        
        if f1 > best_f1_score:
            checkpoint = {
                'epoch': epoch,
                'model': net.state_dict(),
                'ACC': ACC,
                'F1': f1
            }
            torch.save(checkpoint, model_weights_dir /\
                       f"{ACC:.3f}_{f1:.3f}_checkpoint.pkl")

    net.train()
    net.testmode = False
    return highest_acc, ACC, epoch_loss, f1
