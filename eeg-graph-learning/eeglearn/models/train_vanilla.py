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

from eeglearn.config import Config
from eeglearn.utils.utils import get_details_from_file_name, get_labels_dict
from eeglearn.models.model import VanillaGraphModel
from eeglearn.features.graphs import Graphs
from eeglearn.utils.models import (
    split_data, get_graphs_original, print_training_params,
    setup_directories, setup_label_encoder, calculate_class_weights,
    write_epoch_log, update_log, validate_model, create_graph_loaders,
    get_experiment_filename
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
model_weights_dir = Config.model_weights_dir / 'vanilla'
metrics_dir = Config.metrics_dir / 'vanilla'
ignore_replication_nans = True
random_seed = Config.RANDOM_SEED
main_classes = Config.main_classes
optuna = Config.optuna


def train() -> float:
    """Train the model on EEG data.
    
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
    
    split = split_data()
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
    train_loader = create_graph_loaders(participants=train_participants,
                                   encoder=encoder,
                                   batch_size=batch_size,
                                   data_split="train",
                                   perm_types=[None],
                                   drop_last=True)
    metrics = {
        'epoch': [], 
        'train_loss': [], 'train_acc': [], 
        'train_f1_weighted': [], 'train_f1_macro': [],
        'validation_loss': [], 'validation_acc': [], 
        'validation_f1_weighted': [], 'validation_f1_macro': []
    }
    
    print(f"⚠️  Training for epochs: {epochs}")
    
    net = VanillaGraphModel(
        inchannel=5, gcn_out_size=gcn_out_size, batch=batch_size, K=K,
        linear_size=linear_size, drop_rate=drop_rate, testmode=False,
        HC=n_classes
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=rescaled_class_weights).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=4, threshold=0.0001,
        threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-8
    )
    
    trainer = Engine(lambda engine, batch: batch)
    early_stopping = EarlyStopping(
        patience=stop_at,
        score_function=lambda engine: engine.state.metrics['val_macro_f1'],
        trainer=trainer
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, early_stopping)
    
    highest_acc = 0.0
    best_f1_score = 0.0
    best_validation_f1_score_macro = 0.0
    best_validation_f1_score_weighted = 0.0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        all_train_preds = []
        all_train_labels = []
        total_train_samples = 0
        
        net.train()
        for ind, data in enumerate(train_loader['original']):
            data = data.to(device)
            
            out = net(data)
            y = data.y
            total_train_samples += y.size(0)
            
            predictions = torch.argmax(out, dim=1)
            correct_predictions += torch.sum(predictions == y).item()
            
            all_train_preds.extend(predictions.cpu().numpy())
            all_train_labels.extend(y.cpu().numpy())
            
            loss = criterion(out, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        validation_participants_loader = create_graph_loaders(
            participants=validation_participants,
            encoder=encoder,
            batch_size=batch_size,
            data_split="validation",
            perm_types=[None],
            drop_last=not Config.testing_on_sample_data)

        highest_acc, current_val_acc, val_epoch_loss, val_f1_weighted, val_f1_macro = validate_model(
            net, validation_participants_loader, encoder, highest_acc, best_f1_score,
            epoch, batch_size, lr, model_weights_dir, metrics_dir,
            testing_on_sample_data
        )
        
        if val_f1_macro > best_validation_f1_score_macro:
            best_validation_f1_score_macro = val_f1_macro
        
        if val_f1_weighted > best_validation_f1_score_weighted:
            best_validation_f1_score_weighted = val_f1_weighted
            best_f1_score = val_f1_weighted

        trainer.state.metrics = {'val_macro_f1': val_f1_macro}
        trainer.fire_event(Events.EPOCH_COMPLETED)
        
        if trainer.should_terminate:
            print(f"🟢  Early stopping triggered at epoch {epoch}")
            break
        
        write_epoch_log(epoch, batch_size, lr, current_val_acc, metrics_dir)
        scheduler.step(val_epoch_loss)
        
        avg_train_loss = epoch_loss / (ind + 1)
        train_accuracy = correct_predictions / total_train_samples
        
        train_f1_weighted = f1_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
        train_f1_macro = f1_score(all_train_labels, all_train_preds, average='macro', zero_division=0)

        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(avg_train_loss)
        metrics['train_acc'].append(train_accuracy)
        metrics['train_f1_weighted'].append(train_f1_weighted)
        metrics['train_f1_macro'].append(train_f1_macro)
        
        metrics['validation_loss'].append(val_epoch_loss)
        metrics['validation_acc'].append(current_val_acc)
        metrics['validation_f1_weighted'].append(val_f1_weighted)
        metrics['validation_f1_macro'].append(val_f1_macro)
        
        if epoch % 1 == 0:
            print(f'\n## Epoch [{epoch}/{epochs}] ##')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Training ACC: {train_accuracy:.4f}')
            print(f'Training F1 Weighted: {train_f1_weighted:.4f}')
            print(f'Training F1 Macro: {train_f1_macro:.4f}')
            print("----------------------------------------------")
            print(f'Validation Loss: {val_epoch_loss:.4f}')
            print(f'Validation ACC: {current_val_acc:.4f}')
            print(f'Validation F1 Weighted: {val_f1_weighted:.4f}')
            print(f'Validation F1 Macro: {val_f1_macro:.4f}')
            print(f'Best Validation ACC: {highest_acc:.4f}')
            print(f'Best Validation F1 Weighted: {best_validation_f1_score_weighted:.4f}')
            print(f'Best Validation F1 Macro: {best_validation_f1_score_macro:.4f}')
            print("==============================================")
    
    metrics_filename = get_experiment_filename("training_metrics_vanilla", "csv")
    pd.DataFrame(metrics).to_csv(metrics_dir / metrics_filename, index=False)
    return best_validation_f1_score_macro


if __name__ == "__main__":
    train()