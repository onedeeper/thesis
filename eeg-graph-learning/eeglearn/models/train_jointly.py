"""Joint self-supervised and multi-task learning for EEG data.

Implementation of a joint training approach combining self-supervised learning with
multi-task learning for EEG data based on Li et al. 2023.
Handles data splitting, model training, and metrics tracking for frequency and spatial
graph representations.

Functions:
    train: Execute the self-supervised training process and save metrics
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
from eeglearn.models.model import JointlyTrainModel
from eeglearn.features.graphs import Graphs
from eeglearn.utils.models import (
    split_data, get_graphs_original, create_graph_loaders, print_training_params,
    setup_directories, setup_label_encoder, calculate_class_weights,
    write_epoch_log, update_log, validate_model
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
model_weights_dir = Config.model_weights_dir / 'jointly'
metrics_dir = Config.metrics_dir / 'jointly'
ignore_replication_nans = True
random_seed = Config.RANDOM_SEED
main_classes = Config.main_classes
optuna = Config.optuna


def train() -> float:
    """Train the joint self-supervised model on pretext and downstream tasks.
    
    Returns:
        float: Best F1 score achieved during training
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
    
    encoder, n_classes = \
        setup_label_encoder(ignore_replication_nans=ignore_replication_nans)
    all_psych_labels = get_labels_dict()
    
    split = split_data(ignore_replication_nans=ignore_replication_nans)
    train_participants = split['train']
    validation_participants = split['valid']
    test_participants = split['test']
    
    print("⚠️  Participants split:")
    for split_name, participants in [("train", train_participants), 
                                     ("valid", validation_participants),
                                       ("test", test_participants)]:
        print(f"n {split_name}: {len(participants)}")
    
    rescaled_class_weights = calculate_class_weights(train_participants,
                                                     all_psych_labels, 
                                                     encoder,
                                                     n_classes)
    
    print("🔄  Building graphs.")
    loaders = create_graph_loaders(participants=train_participants, 
                                   encoder=encoder, 
                                   batch_size=batch_size,
                                   data_split="train")
    
    validation_loader = create_graph_loaders(participants=validation_participants, 
                                   encoder=encoder, 
                                   batch_size=batch_size,
                                   data_split="validation",
                                   original_only=True)
    test_loader  = create_graph_loaders(participants=test_participants, 
                                   encoder=encoder, 
                                   batch_size=batch_size,
                                   data_split="test",
                                   original_only=True)
    metrics = {
        'epoch': [], 'weighted_loss': [], 'freq_loss': [], 'spatial_loss': [], 
        'original_loss': [], 'freq_acc': [], 'spatial_acc': [], 'original_acc': [], 
        'f1_score': []
    }
    
    print(f"⚠️  Training for epochs: {epochs}")
    
    awl = AutomaticWeightedLoss(3)
    net = JointlyTrainModel(
        inchannel=5, gcn_out_size=gcn_out_size, batch=batch_size, K=K,
        linear_size=linear_size, drop_rate=drop_rate, testmode=False,
        HF=120, HS=128, HC=n_classes
    ).to(device)
    
    criterion_original = nn.CrossEntropyLoss(weight=rescaled_class_weights).to(device)
    criterion_permuted = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=4, threshold=0.0001,
        threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-8
    )
    
    trainer = Engine(lambda engine, batch: batch)
    early_stopping = EarlyStopping(
        patience=stop_at,
        score_function=lambda engine: -engine.state.metrics['val_loss'],
        trainer=trainer
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, early_stopping)
    
    highest_acc = 0.0
    best_f1_score = 0.0
    
    for epoch in range(epochs):
        loader = zip(loaders['frequency'], loaders['spatial'], 
                     cycle(loaders['original']))
        
        epoch_losses = {'weighted': 0.0, 'freq': 0.0, 'spatial': 0.0, 'original': 0.0}
        correct_predictions = {'freq': 0, 'spatial': 0, 'original': 0}
        
        for ind, (fdata, sdata, gdata) in enumerate(loader):
            fdata, sdata, gdata = fdata.to(device), sdata.to(device), gdata.to(device)
            
            freq_out, spatial_out, original_out = net(fdata, sdata, gdata)
            y_freq, y_spatial, y_original = fdata.y, sdata.y, gdata.y
            
            predictions = {
                'freq': torch.argmax(freq_out, dim=1),
                'spatial': torch.argmax(spatial_out, dim=1),
                'original': torch.argmax(original_out, dim=1)
            }
            
            for key, pred in predictions.items():
                target = locals()[f'y_{key}']
                correct_predictions[key] += torch.sum(pred == target).item()
            
            loss_freq = criterion_permuted(freq_out, y_freq)
            loss_spatial = criterion_permuted(spatial_out, y_spatial)
            loss_original = criterion_original(original_out, y_original)
            loss = awl(loss_freq, loss_spatial, loss_original)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses['weighted'] += loss.item()
            epoch_losses['freq'] += loss_freq.item()
            epoch_losses['spatial'] += loss_spatial.item()
            epoch_losses['original'] += loss_original.item()
        
        highest_acc, current_acc, epoch_loss, f1 = validate_model(
            net, validation_loader, encoder, highest_acc, best_f1_score,
            epoch, batch_size, lr, model_weights_dir, metrics_dir,
            testing_on_sample_data
        )
        
        if f1 > best_f1_score:
            best_f1_score = f1
        
        trainer.state.metrics = {'val_loss': epoch_loss}
        trainer.fire_event(Events.EPOCH_COMPLETED)
        
        if trainer.should_terminate:
            print(f"🟢  Early stopping triggered at epoch {epoch}")
            break
        
        write_epoch_log(epoch, batch_size, lr, current_acc, metrics_dir)
        scheduler.step(epoch_losses['weighted'])
        
        denominator = (ind + 1) * batch_size
        avg_losses = {k: v / (ind + 1) for k, v in epoch_losses.items()}
        accuracies = {k: v / denominator for k, v in correct_predictions.items()}
        metrics['epoch'].append(epoch)
        for loss_type, loss_val in avg_losses.items():
            if loss_type != 'weighted':
                metrics[f'{loss_type}_loss'].append(loss_val)
            else:
                metrics['weighted_loss'].append(loss_val)
        
        for acc_type, acc_val in accuracies.items():
            metrics[f'{acc_type}_acc'].append(acc_val)
        metrics['f1_score'].append(f1)
        
        if epoch % 5 == 0:
            print(f'\n## highest_acc {highest_acc:.4f} curr_acc {current_acc:.4f} ##')
            print(f'batch {batch_size}, lr {lr}\n')
        
        print(f'Epoch [{epoch}/{epochs}]')
        print(f'Weighted loss [{avg_losses["weighted"]:.4f}]')
        print(f'Frequency loss[{avg_losses["freq"]:.4f}]')
        print(f'Spatial loss[{avg_losses["spatial"]:.4f}]')
        print(f'Original loss[{avg_losses["original"]:.4f}]')
        print('ACC@1:')
        print(f'Frequency ACC[{accuracies["freq"]:.4f}]')
        print(f'Spatial ACC[{accuracies["spatial"]:.4f}]')
        print(f'Original ACC[{accuracies["original"]:.4f}]')
        print(f'F1 Score[{f1:.4f}]')
        print("----------------------------------------------")
    
    pd.DataFrame(metrics).to_csv(metrics_dir / "training_metrics_jointly.csv", 
                                 index=False)
    return best_f1_score


if __name__ == "__main__":
    train()