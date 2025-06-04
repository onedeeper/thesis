"""Self-supervised EEG training pipeline.

Implementation of a self-supervised training approach for EEG data based on 
Li et al. 2023 (https://ieeexplore.ieee.org/abstract/document/9765326).
This module handles data splitting, model training, and metrics tracking for
both frequency and spatial graph representations.

Functions:
    train: Train the self-supervised model and save metrics
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

from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from AutoWeight import AutomaticWeightedLoss

from eeglearn.config import Config
from eeglearn.utils.utils import get_details_from_file_name, get_labels_dict
from eeglearn.models.model import SelfSupervisedTrain
from eeglearn.features.graphs import Graphs
from eeglearn.utils.models import (
    split_data, create_graph_loaders, print_training_params,
    setup_directories, setup_label_encoder, calculate_class_weights,
    write_epoch_log, update_log, get_experiment_filename, validate_self_supervised_model
)

testing_on_sample_data = Config.testing_on_sample_data
device = Config.device
num_workers = Config.num_workers
drop_last = Config.drop_last
skip_bads = Config.skip_bads
project_root = Config.project_root
data_path = Config.data_path
cleaned_data_path = Config.cleaned_data_path
energy_path = Config.energy_path
model_weights_dir = Config.model_weights_dir / 'self_supervised'
metrics_dir = Config.metrics_dir / 'self_supervised'
ignore_replication_nans = True
random_seed = Config.RANDOM_SEED
main_classes = Config.main_classes
optuna = Config.optuna


def train() -> float:
    """Train the self-supervised model on frequency and spatial graphs.
    
    Returns:
        float: Best validation loss achieved during training
    """
    batch_size = Config.batch_size
    epochs = Config.epochs
    lr = Config.lr
    weight_decay = Config.weight_decay
    drop_rate = Config.drop_rate
    gcn_out_size = Config.gcn_out_size
    linear_size = Config.linear_size
    K = Config.K
    stop_at = Config.stop_at
    
    print_training_params()
    setup_directories(model_weights_dir, metrics_dir)
    
    # Check and print device information
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Using CPU for training")
    print(f"📱 Device: {device}")

    # Note: For self-supervised pretext tasks, we don't need label encoder or class weights
    # But we use the same data split as other training approaches
    if Config.load_data_split_from != "":
        print(f"⚠️  Data split loaded from {data_path / Config.load_data_split_from}")
        split = torch.load(data_path / Config.load_data_split_from)
    else:
        split = split_data()
    
    train_participants = split['train']
    validation_participants = split['valid']
    test_participants = split['test']
    
    print("⚠️  Participants split:")
    for split_name, participants in [("train", train_participants), 
                                     ("valid", validation_participants),
                                     ("test", test_participants)]:
        print(f"n {split_name}: {len(participants)}")

    print("🔄  Building graphs.")
    # Create training loaders for frequency and spatial permutations
    train_loaders = create_graph_loaders(
        participants=train_participants,
        encoder=None,  # No encoder needed for pretext tasks
        batch_size=batch_size,
        data_split_type="train",
        perm_types=["frequency", "spatial"],
        drop_last=drop_last
    )
    
    # Create validation loaders
    validation_loaders = create_graph_loaders(
        participants=validation_participants,
        encoder=None,  # No encoder needed for pretext tasks
        batch_size=batch_size,
        data_split_type="validation", 
        perm_types=["frequency", "spatial"],
        drop_last= drop_last
    )

    # Create validation loaders
    test_loaders = create_graph_loaders(
        participants=test_participants,
        encoder=None,  # No encoder needed for pretext tasks
        batch_size=batch_size,
        data_split_type="test", 
        perm_types=["frequency", "spatial"],
        drop_last= drop_last
    )
    
    print("\n📊 Graph Loader Information:")
    print(f"  • Training loaders:")
    for loader_type, loader in train_loaders.items():
        print(f"    - {loader_type}: {len(loader)} batches")
    
    print(f"\n  • Validation loaders:")
    for loader_type, loader in validation_loaders.items():
        print(f"    - {loader_type}: {len(loader)} batches")
    print()

    print(f"\n  • Test loaders:")
    for loader_type, loader in test_loaders.items():
        print(f"    - {loader_type}: {len(loader)} batches")
    print()
    # Initialize metrics dictionary to match train_jointly structure
    metrics = {
        'epoch': [], 'train_weighted_loss': [], 'train_freq_loss': [], 'train_spatial_loss': [],
        'train_freq_acc': [], 'train_spatial_acc': [],
        'validation_weighted_loss': [], 'validation_freq_loss': [], 'validation_spatial_loss': [],
        'validation_freq_acc': [], 'validation_spatial_acc': []
    }

    print(f"⚠️  Training for epochs: {epochs}")
    
    awl = AutomaticWeightedLoss(2)
    net = SelfSupervisedTrain(
        inchannel=5, 
        gcn_out_size=gcn_out_size, 
        batch=batch_size, 
        K=K,
        linear_size=linear_size,
        drop_rate=drop_rate,
        HF=120, 
        HS=128
    ).to(device)
    
    print(net)
    print(f"⚠️ Training on device : {device}")
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=4, threshold=0.0001,
        threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-8
    )
    
    # Setup early stopping based on validation loss
    trainer = Engine(lambda engine, batch: batch)
    early_stopping = EarlyStopping(
        patience=stop_at,
        score_function=lambda engine: -engine.state.metrics['val_loss'],  # Negative because we want to minimize
        trainer=trainer
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, early_stopping)
    
    best_val_loss = float('inf')

    for epoch in range(epochs):
        net.train()
        
        # Training loop
        train_loader = zip(train_loaders['frequency'], train_loaders['spatial'])
        train_epoch_weighted_loss = 0.0
        train_epoch_loss_freq = 0.0
        train_epoch_loss_spatial = 0.0
        train_correct_pred_freq = 0
        train_correct_pred_spatial = 0
        total_train_samples = 0

        for ind, (freq_data, spatial_data) in enumerate(train_loader):
            freq_data, spatial_data = freq_data.to(device), spatial_data.to(device)
            freq_logits, spatial_logits = net(freq_data, spatial_data)

            y_freq, y_spatial = freq_data.y, spatial_data.y
            _, pred_freq = torch.max(freq_logits, dim=1)
            _, pred_spatial = torch.max(spatial_logits, dim=1)

            train_correct_pred_freq += torch.sum(pred_freq == y_freq).item()
            train_correct_pred_spatial += torch.sum(pred_spatial == y_spatial).item()

            train_loss_frequency = criterion(freq_logits, y_freq)
            train_loss_spatial = criterion(spatial_logits, y_spatial)
            train_weighted_loss = awl(train_loss_frequency, train_loss_spatial)

            optimizer.zero_grad()
            train_weighted_loss.backward()
            optimizer.step()
            
            train_epoch_weighted_loss += train_weighted_loss.item()
            train_epoch_loss_freq += train_loss_frequency.item()
            train_epoch_loss_spatial += train_loss_spatial.item()
            total_train_samples += y_freq.size(0)

        # Calculate training averages
        train_avg_weighted_loss = train_epoch_weighted_loss / (ind + 1)
        train_avg_freq_loss = train_epoch_loss_freq / (ind + 1)
        train_avg_spatial_loss = train_epoch_loss_spatial / (ind + 1)
        train_freq_acc = train_correct_pred_freq / total_train_samples
        train_spatial_acc = train_correct_pred_spatial / total_train_samples

        # Validation step
        (best_val_loss, val_weighted_loss, val_freq_acc, val_spatial_acc, 
         val_freq_loss, val_spatial_loss, _) = validate_self_supervised_model(
            net, validation_loaders, epoch, batch_size, lr, 
            model_weights_dir, metrics_dir, best_val_loss
        )

        # Update scheduler with validation loss
        scheduler.step(val_weighted_loss)
        
        # Update early stopping engine
        trainer.state.metrics = {'val_loss': val_weighted_loss}
        trainer.fire_event(Events.EPOCH_COMPLETED)
        
        if trainer.should_terminate:
            print(f"🟢  Early stopping triggered at epoch {epoch}")
            break

        # Log epoch information  
        write_epoch_log(epoch, batch_size, lr, val_freq_acc, metrics_dir)

        # Save metrics
        metrics['epoch'].append(epoch)
        metrics['train_weighted_loss'].append(train_avg_weighted_loss)
        metrics['train_freq_loss'].append(train_avg_freq_loss)
        metrics['train_spatial_loss'].append(train_avg_spatial_loss)
        metrics['train_freq_acc'].append(train_freq_acc)
        metrics['train_spatial_acc'].append(train_spatial_acc)
        metrics['validation_weighted_loss'].append(val_weighted_loss)
        metrics['validation_freq_loss'].append(val_freq_loss)
        metrics['validation_spatial_loss'].append(val_spatial_loss)
        metrics['validation_freq_acc'].append(val_freq_acc)
        metrics['validation_spatial_acc'].append(val_spatial_acc)

        # Print epoch results
        print(f'Epoch [{epoch}/{epochs}]')
        print(f'Training Weighted loss [{train_avg_weighted_loss:.4f}]')
        print(f'Training Frequency loss[{train_avg_freq_loss:.4f}]')
        print(f'Training Spatial loss[{train_avg_spatial_loss:.4f}]')
        print('Training ACC@1:')
        print(f'Training Frequency ACC[{train_freq_acc:.4f}]')
        print(f'Training Spatial ACC[{train_spatial_acc:.4f}]')
        print("----------------------------------------------")
        print(f'Validation Weighted loss [{val_weighted_loss:.4f}]')
        print(f'Validation Frequency loss[{val_freq_loss:.4f}]')
        print(f'Validation Spatial loss[{val_spatial_loss:.4f}]')
        print('Validation ACC@1:')
        print(f'Validation Frequency ACC[{val_freq_acc:.4f}]')
        print(f'Validation Spatial ACC[{val_spatial_acc:.4f}]')
        print(f'Best Validation Loss [{best_val_loss:.4f}]')
        print("==============================================")

    # Save metrics
    metrics_filename = get_experiment_filename("training_metrics_self_supervised", "csv")
    pd.DataFrame(metrics).to_csv(metrics_dir / metrics_filename, index=False)
    
    return best_val_loss


if __name__ == "__main__":
    train()