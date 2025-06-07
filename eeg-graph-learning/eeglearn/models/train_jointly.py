"""Joint self-supervised and multi-task learning for EEG data.

Implementation of a joint training approach combining self-supervised learning with
multi-task learning for EEG data based on Li et al. 2023.
Handles data splitting, model training, and metrics tracking for frequency and spatial
graph representations.

Functions:
    train: Execute the self-supervised training process and save metrics
    train_with_kfold_cv: Execute k-fold cross-validation training
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

import json
from eeglearn.config import Config
from eeglearn.utils.utils import get_details_from_file_name, get_labels_dict
from eeglearn.models.models import JointlyTrainModel
from eeglearn.features.graphs import Graphs
from eeglearn.utils.models import (
    split_data, get_graphs_original, create_graph_loaders, print_training_params,
    setup_directories, setup_label_encoder, calculate_class_weights,
    write_epoch_log, update_log, validate_model, get_experiment_filename
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
    setup_directories({"weights": model_weights_dir, "metrics": metrics_dir})
    
    # Check and print device information
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Using CPU for training")
    print(f"📱 Device: {device}")
    
    encoder, n_classes = \
        setup_label_encoder(ignore_replication_nans=ignore_replication_nans)
    all_psych_labels = get_labels_dict()
    
    if Config.load_data_split_from != "":
        print(f"⚠️  Data split loaded from {data_path / Config.load_data_split_from}")
        split = torch.load(data_path /  Config.load_data_split_from)
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
    
    rescaled_class_weights = calculate_class_weights(train_participants,
                                                     all_psych_labels, 
                                                     encoder,
                                                     n_classes)
    
    print("🔄  Building graphs.")
    train_loaders = create_graph_loaders(participants=train_participants, 
                                   encoder=encoder, 
                                   batch_size=batch_size,
                                   data_split_type="train",
                                   drop_last= True)
    
    validation_loader = create_graph_loaders(participants=validation_participants, 
                                   encoder=encoder, 
                                   batch_size=batch_size,
                                   data_split_type="validation",
                                   perm_types=[None],
                                   drop_last = not Config.testing_on_sample_data)
    test_loader  = create_graph_loaders(participants=test_participants, 
                                   encoder=encoder, 
                                   batch_size=batch_size,
                                   data_split_type="test",
                                   perm_types=[None],
                                   drop_last=drop_last)
     
    print("\n📊 Graph Loader Information:")
    print(f"  • Training loaders:")
    for loader_type, loader in train_loaders.items():
        print(f"    - {loader_type}: {len(loader)} batches")
    
    print(f"\n  • Validation loader:")
    for loader_type, loader in validation_loader.items():
        print(f"    - {loader_type}: {len(loader)} batches")
    
    print(f"\n  • Test loader:")
    for loader_type, loader in test_loader.items():
        print(f"    - {loader_type}: {len(loader)} batches")
    print()
    metrics = {
        'epoch': [], 'train_weighted_loss': [], 'train_freq_loss': [], 'train_spatial_loss': [], 
        'train_original_loss': [], 'train_freq_acc': [], 'train_spatial_acc': [], 
        'train_original_acc': [], 'train_original_f1_weighted': [], 'train_original_f1_macro': [],
        'validation_loss': [], 'validation_acc': [], 
        'validation_f1_weighted': [], 'validation_f1_macro': []
    }
    
    print(f"⚠️  Training for epochs: {epochs}")
    
    awl = AutomaticWeightedLoss(3)
    net = JointlyTrainModel(
        inchannel=5, gcn_out_size=gcn_out_size, batch=batch_size, K=K,
        linear_size=linear_size, drop_rate=drop_rate, testmode=False,
        HF=120, HS=128, HC=n_classes
    ).to(device)
    
    criterion_original = nn.CrossEntropyLoss(weight=rescaled_class_weights).to(device)
    if not Config.use_class_weighting:
        criterion_original = nn.CrossEntropyLoss().to(device)

    criterion_permuted = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=4, threshold=0.0001,
        threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-8
    )
    
    trainer = Engine(lambda engine, batch: batch)
    early_stopping = EarlyStopping(
    patience=stop_at,
    score_function=lambda eng: eng.state.metrics['val_macro_f1'],
    trainer=trainer
)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, early_stopping)
    
    validation_highest_acc = 0.0
    best_validation_f1_score_macro = 0.0
    best_validation_f1_score_weighted = 0.0
    
    for epoch in range(epochs):
        loader = zip(train_loaders['frequency'], train_loaders['spatial'], 
                     cycle(train_loaders['original']))
        
        training_epoch_losses = {'weighted': 0.0, 'freq': 0.0, 'spatial': 0.0, 
                                 'original': 0.0}
        correct_predictions = {'freq': 0, 'spatial': 0, 'original': 0}
        all_train_original_preds = []
        all_train_original_labels = []
        
        total_samples = 0
        for ind, (fdata, sdata, gdata) in enumerate(loader):

            fdata, sdata, gdata = fdata.to(device), sdata.to(device), gdata.to(device)
            freq_logits, spatial_logits, original_logits = net(fdata, sdata, gdata)
            y_freq, y_spatial, y_original = fdata.y, sdata.y, gdata.y
            
            total_samples += len(gdata.y)
            predictions = {
                'freq': torch.argmax(freq_logits, dim=1),
                'spatial': torch.argmax(spatial_logits, dim=1),
                'original': torch.argmax(original_logits, dim=1)
            }
            
            all_train_original_preds.extend(predictions['original'].cpu().numpy())
            all_train_original_labels.extend(y_original.cpu().numpy())
            
            for key, pred in predictions.items():
                target = locals()[f'y_{key}']
                correct_predictions[key] += torch.sum(pred == target).item()
            
            training_loss_freq = criterion_permuted(freq_logits, y_freq)
            training_loss_spatial = criterion_permuted(spatial_logits, y_spatial)
            training_loss_original = criterion_original(original_logits, y_original)
            training_combined_loss = awl(training_loss_freq, training_loss_spatial, 
                                         training_loss_original)
            
            optimizer.zero_grad()
            training_combined_loss.backward()
            optimizer.step()
            
            training_epoch_losses['weighted'] += training_combined_loss.item()
            training_epoch_losses['freq'] += training_loss_freq.item()
            training_epoch_losses['spatial'] += training_loss_spatial.item()
            training_epoch_losses['original'] += training_loss_original.item()
            
        
        validation_highest_acc, validation_current_acc, validation_epoch_loss, \
            validation_f1_weighted, validation_f1_macro = validate_model(
            net, validation_loader, encoder, criterion_original, validation_highest_acc,
            best_validation_f1_score_macro,
            epoch, batch_size, lr, model_weights_dir, metrics_dir,
            testing_on_sample_data
        )
        if validation_f1_macro > best_validation_f1_score_macro:
            best_validation_f1_score_macro = validation_f1_macro
            
        if validation_f1_weighted > best_validation_f1_score_weighted:
            best_validation_f1_score_weighted = validation_f1_weighted
        
        trainer.state.metrics = {'val_macro_f1': validation_f1_macro}
        trainer.fire_event(Events.EPOCH_COMPLETED)
        
        if trainer.should_terminate:
            print(f"🟢  Early stopping triggered at epoch {epoch}")
            break
        
        # Calculate F1 scores for training original task
        train_f1_original_weighted = f1_score(all_train_original_labels, all_train_original_preds, average='weighted', zero_division=0)
        train_f1_original_macro = f1_score(all_train_original_labels, all_train_original_preds, average='macro', zero_division=0)

        write_epoch_log(epoch, batch_size, lr, validation_current_acc, metrics_dir)
        scheduler.step(training_epoch_losses['weighted'])
    
        avg_losses = {k: v / (ind + 1) for k, v in training_epoch_losses.items()}
        accuracies = {k: v / total_samples for k, v in correct_predictions.items()}
        metrics['epoch'].append(epoch)
        for loss_type, loss_val in avg_losses.items():
            if loss_type != 'weighted':
                metrics[f'train_{loss_type}_loss'].append(loss_val)
            else:
                metrics['train_weighted_loss'].append(loss_val)
        
        metrics['validation_loss'].append(validation_epoch_loss)
        metrics['validation_acc'].append(validation_current_acc)
        metrics['validation_f1_weighted'].append(validation_f1_weighted)
        metrics['validation_f1_macro'].append(validation_f1_macro)
        
        for acc_type, acc_val in accuracies.items():
            metrics[f'train_{acc_type}_acc'].append(acc_val)
        metrics['train_original_f1_weighted'].append(train_f1_original_weighted)
        metrics['train_original_f1_macro'].append(train_f1_original_macro)
        
        
        print(f'Epoch [{epoch}/{epochs}]')
        print(f'Training Weighted loss [{avg_losses["weighted"]:.4f}]')
        print(f'Training Frequency loss[{avg_losses["freq"]:.4f}]')
        print(f'Training Spatial loss[{avg_losses["spatial"]:.4f}]')
        print(f'Training Original loss[{avg_losses["original"]:.4f}]')
        print('Training ACC@1:')
        print(f'Training Frequency ACC[{accuracies["freq"]:.4f}]')
        print(f'Training Spatial ACC[{accuracies["spatial"]:.4f}]')
        print(f'Training Original ACC[{accuracies["original"]:.4f}]')
        print(f'Training Original F1 Weighted [{train_f1_original_weighted:.4f}]')
        print(f'Training Original F1 Macro [{train_f1_original_macro:.4f}]')
        print("----------------------------------------------")
        print(f'Validation Loss [{validation_epoch_loss:.4f}]')
        print(f'Validation ACC [{validation_current_acc:.4f}]')
        print(f'Validation Weighted F1 Score [{validation_f1_weighted:.4f}]')
        print(f'Validation Macro F1 Score [{validation_f1_macro:.4f}]')
        print(f'Best Validation ACC [{validation_highest_acc:.4f}]')
        print(f'Best Validation Weighted F1 Score [{best_validation_f1_score_weighted:.4f}]')
        print(f'Best Validation Macro F1 Score [{best_validation_f1_score_macro:.4f}]')
        print("==============================================")
    
    metrics_filename = get_experiment_filename("training_metrics_jointly", "csv")
    pd.DataFrame(metrics).to_csv(metrics_dir / metrics_filename, index=False)
    return best_validation_f1_score_macro


def train_with_kfold_cv(k_folds: int = 5) -> dict:
    """Train the joint self-supervised model using k-fold cross-validation.
    
    Args:
        k_folds: Number of folds for cross-validation (default: 5)
        
    Returns:
        dict: Cross-validation results containing mean and std of metrics across folds
    """
    print(f"🔄 Starting {k_folds}-fold cross-validation for jointly trained model")
    
    batch_size = Config.batch_size
    epochs = Config.epochs
    lr = Config.lr
    weight_decay = Config.weight_decay
    drop_rate = Config.drop_rate
    gcn_out_size = Config.gcn_out_size
    linear_size = Config.linear_size
    K = Config.K
    stop_at = Config.stop_at
    
    # Store model architecture and hyperparameters
    model_config = {
        'model_type': 'JointlyTrainModel',
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
            'use_class_weighting': Config.use_class_weighting,
            'automatic_loss_weighting': True
        }
    }
    
    print_training_params()
    
    # Setup directories with CV suffix
    cv_model_weights_dir = Config.model_weights_dir / 'jointly_cv'
    cv_metrics_dir = Config.metrics_dir / 'jointly_cv'
    setup_directories({"weights": cv_model_weights_dir, "metrics": cv_metrics_dir})
    
    # Check device
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Using CPU for training")
    print(f"📱 Device: {Config.device}")
    
    encoder, n_classes = setup_label_encoder(ignore_replication_nans=True)
    model_config['n_classes'] = n_classes
    
    all_psych_labels = get_labels_dict()
    
    # Get data split - we'll use train+valid for CV, keep test separate
    if Config.load_data_split_from != "":
        print(f"⚠️  Data split loaded from {Config.data_path / Config.load_data_split_from}")
        split = torch.load(Config.data_path / Config.load_data_split_from)
    else:
        split = split_data()
    
    # Combine train and validation for CV, keep test separate
    cv_participants = split['train'] + split['valid']
    test_participants = split['test']
    cv_labels = [all_psych_labels[p] for p in cv_participants]
    
    print(f"⚠️  Using {len(cv_participants)} participants for {k_folds}-fold CV")
    print(f"⚠️  Test set: {len(test_participants)} participants (held out)")
    
    # Setup stratified k-fold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=Config.RANDOM_SEED)
    model_config['cv_config'] = {
        'k_folds': k_folds,
        'stratified': True,
        'random_seed': Config.RANDOM_SEED,
        'n_train_participants': len(cv_participants),
        'n_test_participants': len(test_participants)
    }
    
    # Store results for each fold
    fold_results = {
        'fold': [],
        'best_val_acc': [],
        'best_val_f1_weighted': [],
        'best_val_f1_macro': [],
        'final_train_acc': [],
        'final_train_f1_weighted': [],
        'final_train_f1_macro': []
    }
    
    # Store full training history for plotting learning curves
    full_training_history = {
        'fold': [],
        'epoch': [],
        'train_acc_original': [],
        'train_f1_weighted_original': [],
        'train_f1_macro_original': [],
        'train_loss_original': [],
        'train_loss_freq': [],
        'train_loss_spatial': [],
        'train_loss_weighted': [],
        'val_acc': [],
        'val_f1_weighted': [],
        'val_f1_macro': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(cv_participants, cv_labels)):
        print(f"\n{'='*50}")
        print(f"🔥 Training Fold {fold + 1}/{k_folds}")
        print(f"{'='*50}")
        
        # Split participants for this fold
        fold_train_participants = [cv_participants[i] for i in train_idx]
        fold_val_participants = [cv_participants[i] for i in val_idx]
        
        print(f"📊 Fold {fold + 1} split:")
        print(f"   Training: {len(fold_train_participants)} participants")
        print(f"   Validation: {len(fold_val_participants)} participants")
        
        # Calculate class weights for this fold
        rescaled_class_weights = calculate_class_weights(
            fold_train_participants, all_psych_labels, encoder, n_classes
        )
        
        # Build graphs for this fold
        print("🔄  Building graphs for this fold...")
        train_loaders = create_graph_loaders(
            participants=fold_train_participants,
            encoder=encoder,
            batch_size=batch_size,
            data_split_type=f"train_fold_{fold}",
            drop_last=True
        )
        
        validation_loader = create_graph_loaders(
            participants=fold_val_participants,
            encoder=encoder,
            batch_size=batch_size,
            data_split_type=f"validation_fold_{fold}",
            perm_types=[None],
            drop_last=not Config.testing_on_sample_data
        )
        
        # Initialize model for this fold
        net = JointlyTrainModel(
            inchannel=5, gcn_out_size=gcn_out_size, batch=batch_size, K=K,
            linear_size=linear_size, drop_rate=drop_rate, testmode=False,
            HF=120, HS=128, HC=n_classes
        ).to(Config.device)
        
        # Setup training components
        awl = AutomaticWeightedLoss(3)
        criterion_original = nn.CrossEntropyLoss(weight=rescaled_class_weights).to(Config.device)
        if not Config.use_class_weighting:
            criterion_original = nn.CrossEntropyLoss().to(Config.device)
        
        criterion_permuted = nn.CrossEntropyLoss().to(Config.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=4, threshold=0.0001,
            threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-8
        )
        
        # Early stopping for this fold
        trainer = Engine(lambda engine, batch: batch)
        early_stopping = EarlyStopping(
            patience=stop_at,
            score_function=lambda eng: eng.state.metrics['val_macro_f1'],
            trainer=trainer
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, early_stopping)
        
        # Training tracking for this fold
        fold_best_val_acc = 0.0
        fold_best_val_f1_macro = 0.0
        fold_best_val_f1_weighted = 0.0
        fold_final_train_acc = 0.0
        fold_final_train_f1_weighted = 0.0
        fold_final_train_f1_macro = 0.0
        
        # Training loop for this fold
        for epoch in range(epochs):
            net.train()
            loader = zip(train_loaders['frequency'], train_loaders['spatial'], 
                        cycle(train_loaders['original']))
            
            training_epoch_losses = {'weighted': 0.0, 'freq': 0.0, 'spatial': 0.0, 'original': 0.0}
            correct_predictions = {'freq': 0, 'spatial': 0, 'original': 0}
            all_train_original_preds = []
            all_train_original_labels = []
            total_samples = 0
            batch_count = 0
            
            for ind, (fdata, sdata, gdata) in enumerate(loader):
                fdata, sdata, gdata = fdata.to(Config.device), sdata.to(Config.device), gdata.to(Config.device)
                freq_logits, spatial_logits, original_logits = net(fdata, sdata, gdata)
                y_freq, y_spatial, y_original = fdata.y, sdata.y, gdata.y
                
                total_samples += len(gdata.y)
                batch_count += 1
                predictions = {
                    'freq': torch.argmax(freq_logits, dim=1),
                    'spatial': torch.argmax(spatial_logits, dim=1),
                    'original': torch.argmax(original_logits, dim=1)
                }
                
                all_train_original_preds.extend(predictions['original'].cpu().numpy())
                all_train_original_labels.extend(y_original.cpu().numpy())
                
                for key, pred in predictions.items():
                    target = locals()[f'y_{key}']
                    correct_predictions[key] += torch.sum(pred == target).item()
                
                training_loss_freq = criterion_permuted(freq_logits, y_freq)
                training_loss_spatial = criterion_permuted(spatial_logits, y_spatial)
                training_loss_original = criterion_original(original_logits, y_original)
                training_combined_loss = awl(training_loss_freq, training_loss_spatial, 
                                             training_loss_original)
                
                optimizer.zero_grad()
                training_combined_loss.backward()
                optimizer.step()
                
                training_epoch_losses['weighted'] += training_combined_loss.item()
                training_epoch_losses['freq'] += training_loss_freq.item()
                training_epoch_losses['spatial'] += training_loss_spatial.item()
                training_epoch_losses['original'] += training_loss_original.item()
            
            # Calculate training metrics for this epoch
            train_acc_original = correct_predictions['original'] / total_samples
            train_f1_weighted_original = f1_score(all_train_original_labels, 
                                                 all_train_original_preds, 
                                                 average='weighted', zero_division=0)
            train_f1_macro_original = f1_score(all_train_original_labels,
                                              all_train_original_preds, 
                                              average='macro', zero_division=0)
            
            # Average losses over batches
            for key in training_epoch_losses:
                training_epoch_losses[key] /= batch_count
            
            # Validation for this fold
            net.eval()
            validation_highest_acc, validation_current_acc, validation_epoch_loss, \
                validation_f1_weighted, validation_f1_macro = validate_model(
                net, validation_loader, encoder, criterion_original, fold_best_val_acc,
                fold_best_val_f1_macro, epoch, batch_size, lr, cv_model_weights_dir, cv_metrics_dir,
                Config.testing_on_sample_data
            )
            
            # Store full training history for this epoch
            current_lr = optimizer.param_groups[0]['lr']
            full_training_history['fold'].append(fold + 1)
            full_training_history['epoch'].append(epoch)
            full_training_history['train_acc_original'].append(train_acc_original)
            full_training_history['train_f1_weighted_original'].append(train_f1_weighted_original)
            full_training_history['train_f1_macro_original'].append(train_f1_macro_original)
            full_training_history['train_loss_original'].append(training_epoch_losses['original'])
            full_training_history['train_loss_freq'].append(training_epoch_losses['freq'])
            full_training_history['train_loss_spatial'].append(training_epoch_losses['spatial'])
            full_training_history['train_loss_weighted'].append(training_epoch_losses['weighted'])
            full_training_history['val_acc'].append(validation_current_acc)
            full_training_history['val_f1_weighted'].append(validation_f1_weighted)
            full_training_history['val_f1_macro'].append(validation_f1_macro)
            full_training_history['val_loss'].append(validation_epoch_loss)
            full_training_history['learning_rate'].append(current_lr)
            
            # Update fold bests
            fold_best_val_acc = max(fold_best_val_acc, validation_current_acc)
            fold_best_val_f1_macro = max(fold_best_val_f1_macro, validation_f1_macro)
            fold_best_val_f1_weighted = max(fold_best_val_f1_weighted, validation_f1_weighted)
            
            # Update final training metrics (these will be the last epoch's values)
            fold_final_train_acc = train_acc_original
            fold_final_train_f1_weighted = train_f1_weighted_original
            fold_final_train_f1_macro = train_f1_macro_original
            
            # Early stopping check
            trainer.state.metrics = {'val_macro_f1': validation_f1_macro}
            trainer.fire_event(Events.EPOCH_COMPLETED)
            
            if trainer.should_terminate:
                print(f"🟢  Early stopping triggered at epoch {epoch} for fold {fold + 1}")
                break
            
            scheduler.step(training_epoch_losses['weighted'])
            
            if epoch % 5 == 0:  # Print every 5 epochs to reduce clutter
                print(f'Fold {fold + 1}, Epoch [{epoch}/{epochs}] - '
                      f'Train Acc: {train_acc_original:.4f}, '
                      f'Val Acc: {validation_current_acc:.4f}, '
                      f'Train F1: {train_f1_macro_original:.4f}, '
                      f'Val F1: {validation_f1_macro:.4f}')
        
        # Store results for this fold
        fold_results['fold'].append(fold + 1)
        fold_results['best_val_acc'].append(fold_best_val_acc)
        fold_results['best_val_f1_weighted'].append(fold_best_val_f1_weighted)
        fold_results['best_val_f1_macro'].append(fold_best_val_f1_macro)
        fold_results['final_train_acc'].append(fold_final_train_acc)
        fold_results['final_train_f1_weighted'].append(fold_final_train_f1_weighted)
        fold_results['final_train_f1_macro'].append(fold_final_train_f1_macro)
        
        print(f"✅ Fold {fold + 1} completed:")
        print(f"   Best Val Acc: {fold_best_val_acc:.4f}")
        print(f"   Best Val F1 Macro: {fold_best_val_f1_macro:.4f}")
        print(f"   Best Val F1 Weighted: {fold_best_val_f1_weighted:.4f}")
    
    # Calculate cross-validation statistics
    cv_results = {}
    for metric in ['best_val_acc', 'best_val_f1_weighted', 'best_val_f1_macro',
                   'final_train_acc', 'final_train_f1_weighted', 'final_train_f1_macro']:
        values = fold_results[metric]
        cv_results[f'{metric}_mean'] = np.mean(values)
        cv_results[f'{metric}_std'] = np.std(values)
    
    # Save detailed results
    fold_results_df = pd.DataFrame(fold_results)
    cv_results_filename = get_experiment_filename(f"cv_{k_folds}fold_detailed_results", "csv")
    fold_results_df.to_csv(cv_metrics_dir / cv_results_filename, index=False)
    
    # Save model configuration
    model_config_filename = get_experiment_filename(f"cv_{k_folds}fold_model_config", "json")
    with open(cv_metrics_dir / model_config_filename, 'w') as f:
        json.dump(model_config, f, indent=4)
    print(f"📝 Model configuration saved to: {model_config_filename}")
    
    # Save full training history for learning curve plotting
    training_history_df = pd.DataFrame(full_training_history)
    training_history_filename = get_experiment_filename(f"cv_{k_folds}fold_training_history", "csv")
    training_history_df.to_csv(cv_metrics_dir / training_history_filename, index=False)
    print(f"📊 Full training history saved to: {training_history_filename}")
    
    # Save summary results
    cv_summary = pd.DataFrame([cv_results])
    cv_summary_filename = get_experiment_filename(f"cv_{k_folds}fold_summary", "csv")
    cv_summary.to_csv(cv_metrics_dir / cv_summary_filename, index=False)
    
    # Print final results
    print(f"\n{'='*60}")
    print(f"🎯 {k_folds}-Fold Cross-Validation Results")
    print(f"{'='*60}")
    print(f"Validation Accuracy:     {cv_results['best_val_acc_mean']:.4f} ± {cv_results['best_val_acc_std']:.4f}")
    print(f"Validation F1 Macro:     {cv_results['best_val_f1_macro_mean']:.4f} ± {cv_results['best_val_f1_macro_std']:.4f}")
    print(f"Validation F1 Weighted:  {cv_results['best_val_f1_weighted_mean']:.4f} ± {cv_results['best_val_f1_weighted_std']:.4f}")
    print(f"Training Accuracy:       {cv_results['final_train_acc_mean']:.4f} ± {cv_results['final_train_acc_std']:.4f}")
    print(f"Training F1 Macro:       {cv_results['final_train_f1_macro_mean']:.4f} ± {cv_results['final_train_f1_macro_std']:.4f}")
    print(f"Training F1 Weighted:    {cv_results['final_train_f1_weighted_mean']:.4f} ± {cv_results['final_train_f1_weighted_std']:.4f}")
    
    return cv_results


if __name__ == "__main__":
    # Example usage:
    # For regular training:
    # train()
    
    # For k-fold cross-validation:
    train_with_kfold_cv(k_folds=2)
    
    # Default: run regular training
    #train()