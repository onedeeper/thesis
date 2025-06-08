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
import optuna

from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping

from eeglearn.config import Config
from eeglearn.utils.utils import get_details_from_file_name, get_labels_dict
from eeglearn.models.models import VanillaGraphModel, EEGNet
from eeglearn.features.graphs import Graphs
from eeglearn.utils.models import (
    split_data, get_graphs_original, print_training_params,
    setup_directories, setup_label_encoder, calculate_class_weights,
    write_epoch_log, update_log, validate_model, create_graph_loaders,
    get_experiment_filename, create_time_series_data_dataloader,
    validate_EEGNet_model
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
model_weights_dir = Config.model_weights_dir / 'baseline'
metrics_dir = Config.metrics_dir / 'baseline'
eeg_net_data_folder = Config.data_path / "eegnet"
ignore_replication_nans = True
random_seed = Config.RANDOM_SEED
main_classes = Config.main_classes
is_optuna_enabled = Config.optuna


def train(trial: optuna.Trial = None) -> float:
    """Train the EEGNet model with detailed metrics tracking.
    
    Parameters:
        trial (optuna.Trial, optional): Optuna trial for hyperparameter optimization
        
    Returns:
        float: Best validation F1 macro score achieved during training
    """
    batch_size = Config.batch_size
    epochs = Config.epochs
    lr = Config.lr
    stop_at = Config.stop_at
    weight_decay = Config.weight_decay
    n_channels = Config.n_eeg_channels
    n_timepoints = Config.eeg_net_n_time_steps
    drop_rate = Config.drop_rate
    kernel_length = Config.kernel_length

    # Model configuration for saving
    model_config = {
        'model_type': 'EEGNet',
        'n_channels': n_channels,
        'n_timepoints': n_timepoints,
        'kernel_length': kernel_length,
        'dropout_rate': drop_rate,
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
            'use_class_weighting': Config.use_class_weighting
        }
    }

    print_training_params()
    setup_directories({"weights" : model_weights_dir, 
                       "metrics" : metrics_dir,
                       "eeg_net_data_dir" : eeg_net_data_folder})
    # Check and print device information
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"📱 Device: {device}")
    encoder, n_classes = \
        setup_label_encoder(ignore_replication_nans=ignore_replication_nans)
    
    # Add n_classes to model config
    model_config['n_classes'] = n_classes
    
    all_psych_labels = get_labels_dict()
    if Config.load_data_split_from != "":
        print(f"⚠️  Data split loaded from {data_path / Config.load_data_split_from}")
        split = torch.load(data_path /  Config.load_data_split_from)
    else:
        split = split_data()

    train_participants = split['train']
    validation_participants = split['valid']
    test_participants = split['test']
    

    model_config['data_split'] = {
        'n_train_participants': len(train_participants),
        'n_validation_participants': len(validation_participants),
        'n_test_participants': len(test_participants),
        'load_data_split_from': Config.load_data_split_from if Config.load_data_split_from != "" else None
    }

    for split_name, participants in [("train", train_participants), 
                                     ("valid", validation_participants),
                                       ("test", test_participants)]:
        print(f"n {split_name}: {len(participants)}")
    
    rescaled_class_weights = calculate_class_weights(train_participants,
                                                     all_psych_labels, 
                                                     encoder,
                                                     n_classes)
    print("⏳ Loading preprocessed EEG time series data...")
    train_loader = create_time_series_data_dataloader(data_split_type="train",
                                                eegnet_data_path=eeg_net_data_folder,
                                                      participants=train_participants,
                                                      label_encoder=encoder,
                                                      batch_size=batch_size,
                                                      drop_last=drop_last,
                                                      num_workers = num_workers)

    validation_loader = create_time_series_data_dataloader(data_split_type="valid",
                                                eegnet_data_path=eeg_net_data_folder,
                                                participants=validation_participants,
                                                      label_encoder=encoder,
                                                      batch_size=batch_size,
                                                      drop_last=drop_last,
                                                      num_workers = num_workers)
    
    test_loader = create_time_series_data_dataloader(data_split_type="test",
                                                      participants=test_participants,
                                                eegnet_data_path=eeg_net_data_folder,
                                                      label_encoder=encoder,
                                                      batch_size=batch_size,
                                                      drop_last=drop_last,
                                                      num_workers=num_workers)
    print("\n📊 Data Loader Information:")
    print(f"\n  • Training loader: {len(train_loader)} batches")
    print(f"\n  • Validation loader: {len(validation_loader)} batches") 
    print(f"\n  • Test loader: {len(test_loader)} batches")
    print()

    metrics = {
        'epoch': [], 
        'train_loss': [], 
        'train_acc': [], 
        'train_f1_weighted': [], 
        'train_f1_macro': [],
        'val_loss': [], 
        'val_acc': [], 
        'val_f1_weighted': [], 
        'val_f1_macro': [],
        'learning_rate': []
    }

    print(f"⚠️  Training for epochs: {epochs}")

    net = EEGNet(
        n_channels=n_channels,  
        n_timepoints=n_timepoints,  
        n_classes=n_classes, 
        kernel_length=kernel_length,
        dropout_rate=drop_rate
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=rescaled_class_weights).to(device)
    if not Config.use_class_weighting:
        criterion = nn.CrossEntropyLoss().to(device)
    
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

    validation_highest_acc = 0.0
    best_validation_f1_score_macro = 0.0
    best_validation_f1_score_weighted = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        all_train_preds = []
        all_train_labels = []
        total_train_samples = 0
        batch_count = 0

        net.train()
        for ind, data in enumerate(train_loader):
            X = data[0].float().to(device)  
            y = data[1].squeeze().long().to(device) 
            
            training_logits = net(X)
            total_train_samples += y.size(0)
            batch_count += 1
            
            predictions = torch.argmax(training_logits, dim=1)
            correct_predictions += torch.sum(predictions == y).item()
            
            all_train_preds.extend(predictions.cpu().numpy())
            all_train_labels.extend(y.cpu().numpy())
            
            loss = criterion(training_logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / batch_count
        train_accuracy = correct_predictions / total_train_samples
        train_f1_weighted = f1_score(all_train_labels, all_train_preds, 
                                     average='weighted', zero_division=0)
        train_f1_macro = f1_score(all_train_labels, all_train_preds, 
                                  average='macro', zero_division=0)
        
        (validation_highest_acc, 
         validation_current_acc,
         validation_epoch_loss,
         validation_f1_weighted, 
         validation_f1_macro) = validate_EEGNet_model(
            net=net,
            validation_loader=validation_loader,
            criterion=criterion,
            label_encoder=encoder,
            highest_acc=validation_highest_acc,
            best_macro_f1=best_validation_f1_score_macro,
            epoch=epoch,
            batch_size=batch_size,
            lr=lr,
            model_weights_dir=model_weights_dir,
            metrics_dir=metrics_dir
        )
        if validation_f1_macro > best_validation_f1_score_macro:
            best_validation_f1_score_macro = validation_f1_macro
        
        if validation_f1_weighted > best_validation_f1_score_weighted:
            best_validation_f1_score_weighted = validation_f1_weighted

        trainer.state.metrics = {'val_macro_f1': validation_f1_macro}
        trainer.fire_event(Events.EPOCH_COMPLETED)

        if trial:
            trial.report(validation_f1_macro, epoch)
            if trial.should_prune():
                metrics_df = pd.DataFrame(metrics)
                if not metrics_df.empty:
                    metrics_filename_pruned = \
get_experiment_filename(f"training_metrics_EEGNet_pruned_trial_{trial.number}", "csv")
                    metrics_df.to_csv(metrics_dir/metrics_filename_pruned, index=False)
                raise optuna.TrialPruned()

        if trainer.should_terminate:
            print(f"🟢 Ignite Early stopping triggered at epoch {epoch}")
            break

        write_epoch_log(epoch, batch_size, lr, validation_current_acc, metrics_dir)
        scheduler.step(validation_epoch_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store all metrics with consistent naming
        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(avg_train_loss)
        metrics['train_acc'].append(train_accuracy)
        metrics['train_f1_weighted'].append(train_f1_weighted)
        metrics['train_f1_macro'].append(train_f1_macro)
        metrics['val_loss'].append(validation_epoch_loss)
        metrics['val_acc'].append(validation_current_acc)
        metrics['val_f1_weighted'].append(validation_f1_weighted)
        metrics['val_f1_macro'].append(validation_f1_macro)
        metrics['learning_rate'].append(current_lr)
        
        if epoch % 1 == 0:
            print(f'\n## Epoch [{epoch}/{epochs}] ##')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Training ACC: {train_accuracy:.4f}')
            print(f'Training F1 Weighted: {train_f1_weighted:.4f}')
            print(f'Training F1 Macro: {train_f1_macro:.4f}')
            print("----------------------------------------------")
            print(f'Validation Loss: {validation_epoch_loss:.4f}')
            print(f'Validation ACC: {validation_current_acc:.4f}')
            print(f'Validation F1 Weighted: {validation_f1_weighted:.4f}')
            print(f'Validation F1 Macro: {validation_f1_macro:.4f}')
            print(f'Best Validation ACC: {validation_highest_acc:.4f}')
            print(f'Best Validation F1 Weighted: {best_validation_f1_score_weighted:.4f}')
            print(f'Best Validation F1 Macro: {best_validation_f1_score_macro:.4f}')
            print(f'Learning Rate: {current_lr:.6f}')
            print("==============================================")
    
    # Save detailed training metrics
    metrics_filename = get_experiment_filename("training_metrics_EEGNet", "csv")
    pd.DataFrame(metrics).to_csv(metrics_dir / metrics_filename, index=False)
    print(f"📊 Training metrics saved to: {metrics_filename}")
    
    # Save model configuration
    model_config_filename = get_experiment_filename("model_config_EEGNet", "json")
    with open(metrics_dir / model_config_filename, 'w') as f:
        json.dump(model_config, f, indent=4)
    print(f"📝 Model configuration saved to: {model_config_filename}")
    
    return best_validation_f1_score_macro


def train_with_kfold_cv(k_folds: int = 5) -> dict:
    """Train the EEGNet model using k-fold cross-validation.
    
    Parameters:
        k_folds (int): Number of folds for cross-validation (default: 5)
        
    Returns:
        dict: Cross-validation results containing mean and std of metrics across folds
    """
    print(f"🔄 Starting {k_folds}-fold cross-validation for EEGNet model")
    
    batch_size = Config.batch_size
    epochs = Config.epochs
    lr = Config.lr
    weight_decay = Config.weight_decay
    n_channels = Config.n_eeg_channels
    n_timepoints = Config.eeg_net_n_time_steps
    drop_rate = Config.drop_rate
    kernel_length = Config.kernel_length
    stop_at = Config.stop_at
 
    model_config = {
        'model_type': 'EEGNet',
        'n_channels': n_channels,
        'n_timepoints': n_timepoints,
        'kernel_length': kernel_length,
        'dropout_rate': drop_rate,
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
            'use_class_weighting': Config.use_class_weighting
        }
    }
    
    print_training_params()
    
    cv_model_weights_dir = Config.model_weights_dir / 'baseline_cv'
    cv_metrics_dir = Config.metrics_dir / 'baseline_cv'
    cv_eeg_net_data_folder = Config.data_path / "eegnet_cv"
    setup_directories({"weights": cv_model_weights_dir, 
                       "metrics": cv_metrics_dir,
                       "eeg_net_data_dir": cv_eeg_net_data_folder})
    
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Using CPU for training")
    print(f"📱 Device: {device}")
    
    encoder, n_classes = setup_label_encoder(ignore_replication_nans=True)
    model_config['n_classes'] = n_classes
    
    all_psych_labels = get_labels_dict()
    
    if Config.load_data_split_from != "":
        print(f"⚠️  Data split loaded from {data_path / Config.load_data_split_from}")
        split = torch.load(data_path / Config.load_data_split_from)
    else:
        split = split_data()
    
    cv_participants = split['train'] + split['valid']
    test_participants = split['test']
    cv_labels = [all_psych_labels[p] for p in cv_participants]
    
    print(f"⚠️  Using {len(cv_participants)} participants for {k_folds}-fold CV")
    print(f"⚠️  Test set: {len(test_participants)} participants (held out)")
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    model_config['cv_config'] = {
        'k_folds': k_folds,
        'stratified': True,
        'random_seed': random_seed,
        'n_train_participants': len(cv_participants),
        'n_test_participants': len(test_participants)
    }
    
    fold_results = {
        'fold': [],
        'best_val_acc': [],
        'best_val_f1_weighted': [],
        'best_val_f1_macro': [],
        'final_train_acc': [],
        'final_train_f1_weighted': [],
        'final_train_f1_macro': []
    }
    
    full_training_history = {
        'fold': [],
        'epoch': [],
        'train_acc': [],
        'train_f1_weighted': [],
        'train_f1_macro': [],
        'train_loss': [],
        'val_acc': [],
        'val_f1_weighted': [],
        'val_f1_macro': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    test_loader = create_time_series_data_dataloader(data_split_type="test",
                                                      participants=test_participants,
                                                eegnet_data_path=cv_eeg_net_data_folder,
                                                      label_encoder=encoder,
                                                      batch_size=batch_size,
                                                      drop_last=drop_last,
                                                      num_workers=num_workers)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(cv_participants, cv_labels)):
        print(f"\n{'='*50}")
        print(f"🔥 Training Fold {fold + 1}/{k_folds}")
        print(f"{'='*50}")
        
        fold_train_participants = [cv_participants[i] for i in train_idx]
        fold_val_participants = [cv_participants[i] for i in val_idx]
        
        print(f"📊 Fold {fold + 1} split:")
        print(f"   Training: {len(fold_train_participants)} participants")
        print(f"   Validation: {len(fold_val_participants)} participants")
        
        rescaled_class_weights = calculate_class_weights(
            fold_train_participants, all_psych_labels, encoder, n_classes
        )
        
        print("⏳ Loading preprocessed EEG time series data for this fold...")
        train_loader = create_time_series_data_dataloader(
            data_split_type=f"train_fold_{fold}",
            eegnet_data_path=cv_eeg_net_data_folder,
            participants=fold_train_participants,
            label_encoder=encoder,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=num_workers
        )
        
        validation_loader = create_time_series_data_dataloader(
            data_split_type=f"valid_fold_{fold}",
            eegnet_data_path=cv_eeg_net_data_folder,
            participants=fold_val_participants,
            label_encoder=encoder,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=num_workers
        )
        
        net = EEGNet(
            n_channels=n_channels,  
            n_timepoints=n_timepoints,  
            n_classes=n_classes, 
            kernel_length=kernel_length,
            dropout_rate=drop_rate
        ).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=rescaled_class_weights).to(device)
        if not Config.use_class_weighting:
            criterion = nn.CrossEntropyLoss().to(device)
        
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
        
        fold_best_val_acc = 0.0
        fold_best_val_f1_macro = 0.0
        fold_best_val_f1_weighted = 0.0
        fold_final_train_acc = 0.0
        fold_final_train_f1_weighted = 0.0
        fold_final_train_f1_macro = 0.0
        
        for epoch in range(epochs):
            net.train()
            
            epoch_loss = 0.0
            correct_predictions = 0
            all_train_preds = []
            all_train_labels = []
            total_train_samples = 0
            batch_count = 0
            
            for ind, data in enumerate(train_loader):
                X = data[0].float().to(device)  
                y = data[1].squeeze().long().to(device) 
                
                training_logits = net(X)
                total_train_samples += y.size(0)
                batch_count += 1
                
                predictions = torch.argmax(training_logits, dim=1)
                correct_predictions += torch.sum(predictions == y).item()
                
                all_train_preds.extend(predictions.cpu().numpy())
                all_train_labels.extend(y.cpu().numpy())
                
                loss = criterion(training_logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            train_accuracy = correct_predictions / total_train_samples
            train_f1_weighted = f1_score(all_train_labels, all_train_preds, 
                                         average='weighted', zero_division=0)
            train_f1_macro = f1_score(all_train_labels, all_train_preds, 
                                      average='macro', zero_division=0)
            avg_train_loss = epoch_loss / batch_count
            
            # Validate model and get metrics
            (validation_highest_acc, 
             validation_current_acc,
             validation_epoch_loss,
             validation_f1_weighted, 
             validation_f1_macro) = validate_EEGNet_model(
                net=net,
                validation_loader=validation_loader,
                criterion=criterion,
                label_encoder=encoder,
                highest_acc=fold_best_val_acc,
                best_macro_f1=fold_best_val_f1_macro,
                epoch=epoch,
                batch_size=batch_size,
                lr=lr,
                model_weights_dir=cv_model_weights_dir,
                metrics_dir=cv_metrics_dir
            )
            
            current_lr = optimizer.param_groups[0]['lr']
            full_training_history['fold'].append(fold + 1)
            full_training_history['epoch'].append(epoch)
            full_training_history['train_acc'].append(train_accuracy)
            full_training_history['train_f1_weighted'].append(train_f1_weighted)
            full_training_history['train_f1_macro'].append(train_f1_macro)
            full_training_history['train_loss'].append(avg_train_loss)
            full_training_history['val_acc'].append(validation_current_acc)
            full_training_history['val_f1_weighted'].append(validation_f1_weighted)
            full_training_history['val_f1_macro'].append(validation_f1_macro)
            full_training_history['val_loss'].append(validation_epoch_loss)
            full_training_history['learning_rate'].append(current_lr)
            
            fold_best_val_acc = max(fold_best_val_acc, validation_current_acc)
            fold_best_val_f1_macro = max(fold_best_val_f1_macro, validation_f1_macro)
            fold_best_val_f1_weighted = max(fold_best_val_f1_weighted, validation_f1_weighted)
            
            fold_final_train_acc = train_accuracy
            fold_final_train_f1_weighted = train_f1_weighted
            fold_final_train_f1_macro = train_f1_macro
            
            trainer.state.metrics = {'val_macro_f1': validation_f1_macro}
            trainer.fire_event(Events.EPOCH_COMPLETED)
            
            if trainer.should_terminate:
                print(f"🟢 Ignite Early stopping triggered at epoch {epoch} for fold {fold + 1}")
                break
            
            scheduler.step(validation_epoch_loss)
            
            if epoch % 5 == 0:  
                print(f'Fold {fold + 1}, Epoch [{epoch}/{epochs}] - '
                      f'Train Acc: {train_accuracy:.4f}, '
                      f'Val Acc: {validation_current_acc:.4f}, '
                      f'Train F1: {train_f1_macro:.4f}, '
                      f'Val F1: {validation_f1_macro:.4f}')
        
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
    cv_results_filename = get_experiment_filename(f"cv_{k_folds}fold_detailed_results_EEGNet", "csv")
    fold_results_df.to_csv(cv_metrics_dir / cv_results_filename, index=False)
    
    # Save model configuration
    model_config_filename = get_experiment_filename(f"cv_{k_folds}fold_model_config_EEGNet", "json")
    with open(cv_metrics_dir / model_config_filename, 'w') as f:
        json.dump(model_config, f, indent=4)
    print(f"📝 Model configuration saved to: {model_config_filename}")
    
    # Save full training history
    training_history_df = pd.DataFrame(full_training_history)
    training_history_filename = get_experiment_filename(f"cv_{k_folds}fold_training_history_EEGNet", "csv")
    training_history_df.to_csv(cv_metrics_dir / training_history_filename, index=False)
    print(f"📊 Full training history saved to: {training_history_filename}")
    
    # Save summary statistics
    cv_summary = pd.DataFrame([cv_results])
    cv_summary_filename = get_experiment_filename(f"cv_{k_folds}fold_summary_EEGNet", "csv")
    cv_summary.to_csv(cv_metrics_dir / cv_summary_filename, index=False)
    
    print(f"\n{'='*60}")
    print(f"🎯 {k_folds}-Fold Cross-Validation Results (EEGNet Model)")
    print(f"{'='*60}")
    print(f"Validation Accuracy:     {cv_results['best_val_acc_mean']:.4f} ± {cv_results['best_val_acc_std']:.4f}")
    print(f"Validation F1 Macro:     {cv_results['best_val_f1_macro_mean']:.4f} ± {cv_results['best_val_f1_macro_std']:.4f}")
    print(f"Validation F1 Weighted:  {cv_results['best_val_f1_weighted_mean']:.4f} ± {cv_results['best_val_f1_weighted_std']:.4f}")
    print(f"Training Accuracy:       {cv_results['final_train_acc_mean']:.4f} ± {cv_results['final_train_acc_std']:.4f}")
    print(f"Training F1 Macro:       {cv_results['final_train_f1_macro_mean']:.4f} ± {cv_results['final_train_f1_macro_std']:.4f}")
    print(f"Training F1 Weighted:    {cv_results['final_train_f1_weighted_mean']:.4f} ± {cv_results['final_train_f1_weighted_std']:.4f}")
    
    return cv_results


if __name__ == "__main__":
    #train_with_kfold_cv(k_folds=Config.k_folds)
    train()