"""
Model utilities for EEG graph learning.

Provides functions for data splitting, graph creation, model training and validation.
"""

import os
from pathlib import Path
from itertools import cycle

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Batch
from torch.utils.data import Sampler
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from torch_geometric.loader import DataLoader
from eeglearn.config import Config
from eeglearn.utils.utils import get_details_from_file_name, get_labels_dict
from eeglearn.features.graphs import Graphs


def split_data(ignore_replication_nans: bool = False) -> dict:
    """Split participants into train/val/test sets using stratified sampling.
    
    Args:
        ignore_replication_nans: Exclude participants with NaN labels or in replication
        
    Returns:
        Dict with 'train', 'valid', 'test' lists of participant IDs
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
    
    # Verify all participants have valid labels
    for participant in valid_participants:
        assert labels[participant] in Config.main_classes, \
            f"Invalid label {labels[participant]} for participant {participant}"
    n_samples = int(len(valid_participants) * Config.sample_proportion_of_data)
    # Sample a proportion of valid participants if specified
    if Config.sample_proportion_of_data < 1.0:
        valid_participants = np.random.choice(valid_participants, 
                                            size=n_samples, 
                                            replace=False)
        valid_labels = [labels[p] for p in valid_participants]
    print(f"⚠️  Using {n_samples} out of {len(valid_participants)} total participants")

    # First split: train vs test+valid
    train, test_valid, train_labels, test_valid_labels = train_test_split(
        valid_participants, valid_labels, 
        test_size=1 - Config.p_train, 
        random_state=Config.RANDOM_SEED,
        stratify=valid_labels if Config.use_stratify else None
    )

    # Second split: test vs valid
    test, valid, test_labels, valid_labels = train_test_split(
        test_valid, test_valid_labels,
        test_size=0.5,  # Split remaining data equally
        random_state=Config.RANDOM_SEED,
        stratify=test_valid_labels if Config.use_stratify else None
    )

    splits = [("Train", train), ("Valid", valid), ("Test", test)]
    
    # Verify class distribution if using stratification
    if Config.use_stratify:        
        for split_name, split_data in splits:
            split_classes = set(labels[p] for p in split_data)
            assert split_classes == set(Config.main_classes), \
                f"Not all classes present in {split_name.lower()} set"

    # Print class distribution statistics
    print("\nClass distribution:")
    for split_name, split_data in splits:
        class_counts = {c: sum(1 for p in split_data if labels[p] == c) 
                       for c in Config.main_classes}
        print(f"{split_name} set class counts:", class_counts)

    return {"train": train, "valid": valid, "test": test}


def get_graphs_original(files_to_load: list, label_encoder: LabelEncoder, batch_size: int, 
                       drop_last: bool = False):
    """Load energy objects and convert to labeled graphs.
    
    Args:
        files_to_load: Participant files to load
        label_encoder: Label encoder for psychological labels
        batch_size: Batch size for data loading
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        PyTorch geometric graph data loader
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
        drop_last=drop_last,
        batch_size=batch_size
    )
    
    return graphs.get_graphs(
        files_to_load=full_file_names_to_load, 
        label_encoder=label_encoder,
        skip_bads=Config.skip_bads
    )


def create_graph_loaders(data_split: str, participants: list, encoder: LabelEncoder, 
                        batch_size: int,
                        perm_types: list[str | None] = [None, "spatial", "frequency"], 
                        drop_last: bool = Config.drop_last):
    """Create DataLoaders for graphs with different permutation types.
    
    Args:
        data_split: Split identifier ('train', 'test', 'valid')
        participants: List of participant IDs
        encoder: Label encoder for psychological labels
        batch_size: Batch size for DataLoader
        perm_types: List of permutation types (None=original, spatial, frequency)
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        Dict mapping graph types to PyTorch DataLoaders
    """
    loaders = {}
    
    graph_lists = create_graph_list(participants=participants, 
                                    encoder=encoder, 
                                    perm_types=perm_types, 
                                    data_split=data_split)
    
    for graph_type, graphs in graph_lists.items():
        if data_split == "train" and graph_type == "original" and \
                            Config.use_sampler_for_data_loading:
            print("⚠️  Using balanced sampler")
            sampler = BalancedGraphSampler(graphs)
            loader = DataLoader(graphs,
                                    batch_size=batch_size,      
                                    sampler=sampler,        
                                    num_workers=Config.num_workers,
                                    drop_last=drop_last)
        else:
            loader = DataLoader(dataset=graphs,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=Config.num_workers,
                                drop_last=drop_last,
                                )
        loaders[graph_type] = loader
    return loaders


def create_graph_list(participants: list, encoder: LabelEncoder, 
                     data_split, perm_types: list[str | None] = 
                     [None, "spatial", "frequency"]):
    """Create graph lists for different permutation types.
    
    Args:
        participants: List of participant IDs
        encoder: Label encoder for psychological labels
        data_split: Split identifier
        perm_types: List of permutation types (None=original)
        
    Returns:
        Dict mapping graph types to lists of graph objects
    """
    graph_lists = {}
    
    for perm_type in perm_types:
        key = 'original' if perm_type is None else perm_type
        cache_path = Config.project_root / 'eeglearn' / 'models' / \
            f'{data_split}_{key}_graph_list.pt'
        
        if not cache_path.exists() or Config.optuna:
            print(f"⚠️  Creating new graph list for {key} type {data_split}")
            if perm_type is None:
                epoched_path = Config.energy_path / "energy_epoched"
                energy_files = os.listdir(epoched_path)
                energy_file_ids = {get_details_from_file_name(file)[0]: file 
                                  for file in energy_files}
                
                data_files = [energy_file_ids[file] 
                              for file in participants 
                              if file in energy_file_ids]
                
                energy_path = epoched_path
            else:
                data_files = [fname for participant in participants
                             for fname in os.listdir(Config.energy_path/\
                                                     f"{perm_type}_perms")
                             if participant in fname]
                energy_path = Config.energy_path
            
            graphs = Graphs(
                perm_type=perm_type,
                energy_path=energy_path,
                distance="ellipsoid",
                cleaned_data_path=Config.cleaned_data_path,
                batch_size=1,  
                n_neighbors=3,
                shuffle=False,  
                drop_last=False, 
                n_workers=Config.num_workers
            )
            
            graph_lists[key] = graphs.get_graphs(
                files_to_load=data_files,
                label_encoder=encoder if perm_type is None else None,
                skip_bads=Config.skip_bads,
                return_data_loader=False
            )
            torch.save(graph_lists[key], cache_path)
        else:
            print(f"⚠️  Loading cached graph list for {key} type {data_split}")
            graph_lists[key] = torch.load(cache_path)
    
    return graph_lists


def print_training_params():
    """Print training configuration parameters."""
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
    """Create directories and log files for model training.
    
    Args:
        model_weights_dir: Directory for model weights
        metrics_dir: Directory for training metrics
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


def validate_model(net, validation_loader: list, label_encoder: LabelEncoder, 
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
    
    net.testmode = True
    net.eval()
    
    epoch_loss = 0.0
    correct_pred = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    for _, data in enumerate(validation_loader['original']):
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
                       f"Acc_{ACC:.3f}_f1_{f1:.3f}_checkpoint.pkl")

    net.train()
    net.testmode = False
    return highest_acc, ACC, epoch_loss, f1



class BalancedGraphSampler(Sampler):
    """A PyTorch Sampler that balances class distribution in graph datasets.
    
    This sampler implements a weighted sampling strategy where the probability
    of selecting a sample is inversely proportional to its class frequency.
    This ensures each class is equally likely to be sampled, helping to address
    class imbalance in the dataset.
    
    Args:
        data_list (list): List of graph data objects, each with a 'y' attribute
                         containing the class label
        replacement (bool, optional): Whether to sample with replacement.
                                    Defaults to True.
    
    Attributes:
        weights (torch.DoubleTensor): Sampling weights for each sample
        num_nodes (int): Total number of samples in the dataset
        replacement (bool): Whether sampling is done with replacement
    
    WRITTEN BY AI
    VERIFIED BY AUTHOR
    """
    def __init__(self, data_list, replacement=True):
        labels         = [int(g.y) for g in data_list]
        freq           = Counter(labels)
        self.weights   = torch.DoubleTensor([1.0 / freq[l] for l in labels])
        self.num_nodes = len(data_list)
        
        self.replacement = replacement
        print("⚠️  Class weights:", {l: 1.0 / freq[l] for l in freq})

    def __iter__(self):
        return iter(torch.multinomial(self.weights,
                                      self.num_nodes,
                                      self.replacement).tolist())

    def __len__(self):
        return self.num_nodes