"""Self-supervised EEG training pipeline.

Implementation of a self-supervised training approach for EEG data based on 
Li et al. 2023 (https://ieeexplore.ieee.org/abstract/document/9765326).
This module handles data splitting, model training, and metrics tracking for
both frequency and spatial graph representations.

Functions:
    train: Train the self-supervised model and save metrics
    train_with_kfold_cv: Train the self-supervised model using k-fold cross-validation
"""

import os
from pathlib import Path
from itertools import cycle

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from AutoWeight import AutomaticWeightedLoss

from eeglearn.config import Config
from eeglearn.utils.utils import get_details_from_file_name, get_labels_dict
from eeglearn.models.models import SelfSupervisedTrain
from eeglearn.features.graphs import Graphs
from eeglearn.utils.models import (
    split_data, create_graph_loaders, print_training_params,
    setup_directories, setup_label_encoder, calculate_class_weights,
    write_epoch_log, update_log, get_experiment_filename, validate_self_supervised_model
)

import json

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
    setup_directories({"weights": model_weights_dir, "metrics": metrics_dir})
    
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
            net, validation_loaders, epoch, criterion, batch_size, lr, 
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


def train_with_kfold_cv(k_folds: int = 5) -> dict:
    """Train the self-supervised model using k-fold cross-validation.
    
    Args:
        k_folds: Number of folds for cross-validation (default: 5)
        
    Returns:
        dict: Cross-validation results containing mean and std of metrics across folds
    """
    print(f"🔄 Starting {k_folds}-fold cross-validation for self-supervised model")
    
    batch_size = Config.batch_size
    epochs = Config.epochs
    lr = Config.lr
    weight_decay = Config.weight_decay
    drop_rate = Config.drop_rate
    gcn_out_size = Config.gcn_out_size
    linear_size = Config.linear_size
    K = Config.K
    stop_at = Config.stop_at
 
    model_config = {
        'model_type': 'SelfSupervisedTrain',
        'input_channels': 5,
        'gcn_out_size': gcn_out_size,
        'batch_size': batch_size,
        'K': K,
        'linear_size': linear_size,
        'drop_rate': drop_rate,
        'HF': 120,
        'HS': 128,
        'training_params': {
            'epochs': epochs,
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'early_stopping_patience': stop_at,
            'scheduler': 'ReduceLROnPlateau',
            'scheduler_params': {
                'mode': 'min',
                'factor': 0.1,
                'patience': 4,
                'threshold': 0.0001,
                'threshold_mode': 'rel',
                'cooldown': 1,
                'min_lr': 0,
                'eps': 1e-8
            },
            'optimizer': 'Adam',
            'loss_function': 'CrossEntropyLoss',
            'automatic_loss_weighting': True
        }
    }
    
    print_training_params()
    
    cv_model_weights_dir = Config.model_weights_dir / 'self_supervised_cv'
    cv_metrics_dir = Config.metrics_dir / 'self_supervised_cv'
    setup_directories({"weights": cv_model_weights_dir, "metrics": cv_metrics_dir})
    
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Using CPU for training")
    print(f"📱 Device: {Config.device}")
    
    # For self-supervised learning, we still need labels for stratified splitting
    # but won't use them for the pretext tasks
    encoder, n_classes = setup_label_encoder(ignore_replication_nans=True)
    all_psych_labels = get_labels_dict()
    
    if Config.load_data_split_from != "":
        print(f"⚠️  Data split loaded from {Config.data_path / Config.load_data_split_from}")
        split = torch.load(Config.data_path / Config.load_data_split_from)
    else:
        split = split_data()
    
    cv_participants = split['train'] + split['valid']
    test_participants = split['test']
    cv_labels = [all_psych_labels[p] for p in cv_participants]
    
    print(f"⚠️  Using {len(cv_participants)} participants for {k_folds}-fold CV")
    print(f"⚠️  Test set: {len(test_participants)} participants (held out)")
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=Config.RANDOM_SEED)
    model_config['cv_config'] = {
        'k_folds': k_folds,
        'stratified': True,
        'random_seed': Config.RANDOM_SEED,
        'n_train_participants': len(cv_participants),
        'n_test_participants': len(test_participants)
    }
    
    fold_results = {
        'fold': [],
        'best_val_loss': [],
        'best_val_freq_acc': [],
        'best_val_spatial_acc': [],
        'final_train_freq_acc': [],
        'final_train_spatial_acc': []
    }
    
    full_training_history = {
        'fold': [],
        'epoch': [],
        'train_freq_acc': [],
        'train_spatial_acc': [],
        'train_freq_loss': [],
        'train_spatial_loss': [],
        'train_weighted_loss': [],
        'val_freq_acc': [],
        'val_spatial_acc': [],
        'val_freq_loss': [],
        'val_spatial_loss': [],
        'val_weighted_loss': [],
        'learning_rate': []
    }
    
    test_loaders = create_graph_loaders(
        participants=test_participants,
        encoder=None,  # No encoder needed for pretext tasks
        batch_size=batch_size,
        data_split_type="test", 
        perm_types=["frequency", "spatial"],
        drop_last= drop_last
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(cv_participants, cv_labels)):
        print(f"\n{'='*50}")
        print(f"🔥 Training Fold {fold + 1}/{k_folds}")
        print(f"{'='*50}")
        
        fold_train_participants = [cv_participants[i] for i in train_idx]
        fold_val_participants = [cv_participants[i] for i in val_idx]
        
        print(f"📊 Fold {fold + 1} split:")
        print(f"   Training: {len(fold_train_participants)} participants")
        print(f"   Validation: {len(fold_val_participants)} participants")
        
        print("🔄  Building graphs for this fold...")
        train_loaders = create_graph_loaders(
            participants=fold_train_participants,
            encoder=None,  # No encoder needed for pretext tasks
            batch_size=batch_size,
            data_split_type=f"train_fold_{fold}",
            perm_types=["frequency", "spatial"],
            drop_last=drop_last
        )
        
        validation_loaders = create_graph_loaders(
            participants=fold_val_participants,
            encoder=None,  # No encoder needed for pretext tasks
            batch_size=batch_size,
            data_split_type=f"validation_fold_{fold}",
            perm_types=["frequency", "spatial"],
            drop_last=drop_last
        )
        
        net = SelfSupervisedTrain(
            inchannel=5, 
            gcn_out_size=gcn_out_size, 
            batch=batch_size, 
            K=K,
            linear_size=linear_size,
            drop_rate=drop_rate,
            HF=120, 
            HS=128
        ).to(Config.device)
        
        awl = AutomaticWeightedLoss(2)
        criterion = nn.CrossEntropyLoss().to(Config.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=4, threshold=0.0001,
            threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-8
        )
        
        trainer = Engine(lambda engine, batch: batch)
        early_stopping = EarlyStopping(
            patience=stop_at,
            score_function=lambda engine: -engine.state.metrics['val_loss'],  # Negative because we want to minimize
            trainer=trainer
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, early_stopping)
        
        fold_best_val_loss = float('inf')
        fold_best_val_freq_acc = 0.0
        fold_best_val_spatial_acc = 0.0
        fold_final_train_freq_acc = 0.0
        fold_final_train_spatial_acc = 0.0
        
        for epoch in range(epochs):
            net.train()
            
            train_loader = zip(train_loaders['frequency'], train_loaders['spatial'])
            train_epoch_weighted_loss = 0.0
            train_epoch_loss_freq = 0.0
            train_epoch_loss_spatial = 0.0
            train_correct_pred_freq = 0
            train_correct_pred_spatial = 0
            total_train_samples = 0
            batch_count = 0

            for ind, (freq_data, spatial_data) in enumerate(train_loader):
                freq_data, spatial_data = (freq_data.to(Config.device),
                                           spatial_data.to(Config.device))
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
                batch_count += 1

            # Calculate training averages
            train_avg_weighted_loss = train_epoch_weighted_loss / batch_count
            train_avg_freq_loss = train_epoch_loss_freq / batch_count
            train_avg_spatial_loss = train_epoch_loss_spatial / batch_count
            train_freq_acc = train_correct_pred_freq / total_train_samples
            train_spatial_acc = train_correct_pred_spatial / total_train_samples

            # Validation step
            (best_val_loss, val_weighted_loss, val_freq_acc, val_spatial_acc, 
             val_freq_loss, val_spatial_loss, _) = validate_self_supervised_model(
                net, validation_loaders, epoch, criterion, batch_size, lr, 
                cv_model_weights_dir, cv_metrics_dir, fold_best_val_loss
            )
            
            current_lr = optimizer.param_groups[0]['lr']
            full_training_history['fold'].append(fold + 1)
            full_training_history['epoch'].append(epoch)
            full_training_history['train_freq_acc'].append(train_freq_acc)
            full_training_history['train_spatial_acc'].append(train_spatial_acc)
            full_training_history['train_freq_loss'].append(train_avg_freq_loss)
            full_training_history['train_spatial_loss'].append(train_avg_spatial_loss)
            full_training_history['train_weighted_loss'].append(train_avg_weighted_loss)
            full_training_history['val_freq_acc'].append(val_freq_acc)
            full_training_history['val_spatial_acc'].append(val_spatial_acc)
            full_training_history['val_freq_loss'].append(val_freq_loss)
            full_training_history['val_spatial_loss'].append(val_spatial_loss)
            full_training_history['val_weighted_loss'].append(val_weighted_loss)
            full_training_history['learning_rate'].append(current_lr)
            
            # Update best metrics
            fold_best_val_loss = min(fold_best_val_loss, val_weighted_loss)
            fold_best_val_freq_acc = max(fold_best_val_freq_acc, val_freq_acc)
            fold_best_val_spatial_acc = max(fold_best_val_spatial_acc, val_spatial_acc)
            
            # Update final metrics (last epoch values)
            fold_final_train_freq_acc = train_freq_acc
            fold_final_train_spatial_acc = train_spatial_acc
            
            # Update scheduler with validation loss
            scheduler.step(val_weighted_loss)
            
            # Update early stopping engine
            trainer.state.metrics = {'val_loss': val_weighted_loss}
            trainer.fire_event(Events.EPOCH_COMPLETED)
            
            if trainer.should_terminate:
                print(f"🟢  Early stopping triggered at epoch {epoch} for fold {fold + 1}")
                break
            
            if epoch % 5 == 0:  # Print progress every 5 epochs
                print(f'Fold {fold + 1}, Epoch [{epoch}/{epochs}] - '
                      f'Train Freq Acc: {train_freq_acc:.4f}, '
                      f'Train Spatial Acc: {train_spatial_acc:.4f}, '
                      f'Val Freq Acc: {val_freq_acc:.4f}, '
                      f'Val Spatial Acc: {val_spatial_acc:.4f}, '
                      f'Val Loss: {val_weighted_loss:.4f}')
        
        # Store fold results
        fold_results['fold'].append(fold + 1)
        fold_results['best_val_loss'].append(fold_best_val_loss)
        fold_results['best_val_freq_acc'].append(fold_best_val_freq_acc)
        fold_results['best_val_spatial_acc'].append(fold_best_val_spatial_acc)
        fold_results['final_train_freq_acc'].append(fold_final_train_freq_acc)
        fold_results['final_train_spatial_acc'].append(fold_final_train_spatial_acc)
        
        print(f"✅ Fold {fold + 1} completed:")
        print(f"   Best Val Loss: {fold_best_val_loss:.4f}")
        print(f"   Best Val Freq Acc: {fold_best_val_freq_acc:.4f}")
        print(f"   Best Val Spatial Acc: {fold_best_val_spatial_acc:.4f}")
    
    # Calculate cross-validation summary statistics
    cv_results = {}
    for metric in ['best_val_loss', 'best_val_freq_acc', 'best_val_spatial_acc',
                   'final_train_freq_acc', 'final_train_spatial_acc']:
        values = fold_results[metric]
        cv_results[f'{metric}_mean'] = np.mean(values)
        cv_results[f'{metric}_std'] = np.std(values)
    
    # Save detailed fold results
    fold_results_df = pd.DataFrame(fold_results)
    cv_results_filename = get_experiment_filename(f"cv_{k_folds}fold_detailed_results_self_supervised", "csv")
    fold_results_df.to_csv(cv_metrics_dir / cv_results_filename, index=False)
    
    # Save model configuration
    model_config_filename = get_experiment_filename(f"cv_{k_folds}fold_model_config_self_supervised", "json")
    with open(cv_metrics_dir / model_config_filename, 'w') as f:
        json.dump(model_config, f, indent=4)
    print(f"📝 Model configuration saved to: {model_config_filename}")
    
    # Save full training history
    training_history_df = pd.DataFrame(full_training_history)
    training_history_filename = get_experiment_filename(f"cv_{k_folds}fold_training_history_self_supervised", "csv")
    training_history_df.to_csv(cv_metrics_dir / training_history_filename, index=False)
    print(f"📊 Full training history saved to: {training_history_filename}")
    
    # Save CV summary
    cv_summary = pd.DataFrame([cv_results])
    cv_summary_filename = get_experiment_filename(f"cv_{k_folds}fold_summary_self_supervised", "csv")
    cv_summary.to_csv(cv_metrics_dir / cv_summary_filename, index=False)
    
    # Print final results
    print(f"\n{'='*60}")
    print(f"🎯 {k_folds}-Fold Cross-Validation Results (Self-Supervised)")
    print(f"{'='*60}")
    print(f"Best Validation Loss:         {cv_results['best_val_loss_mean']:.4f} ± {cv_results['best_val_loss_std']:.4f}")
    print(f"Best Val Frequency Accuracy:  {cv_results['best_val_freq_acc_mean']:.4f} ± {cv_results['best_val_freq_acc_std']:.4f}")
    print(f"Best Val Spatial Accuracy:    {cv_results['best_val_spatial_acc_mean']:.4f} ± {cv_results['best_val_spatial_acc_std']:.4f}")
    print(f"Final Train Freq Accuracy:    {cv_results['final_train_freq_acc_mean']:.4f} ± {cv_results['final_train_freq_acc_std']:.4f}")
    print(f"Final Train Spatial Accuracy: {cv_results['final_train_spatial_acc_mean']:.4f} ± {cv_results['final_train_spatial_acc_std']:.4f}")
    
    return cv_results


if __name__ == "__main__":
    # Example usage:
    # For regular training:
    # train()
    
    # For k-fold cross-validation:
    train_with_kfold_cv(k_folds=Config.k_folds)
    