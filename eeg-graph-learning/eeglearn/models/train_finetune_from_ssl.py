"""Module for fine-tuning classification head using pre-trained self-supervised weights.

This module provides functionality to:
1. Load pre-trained self-supervised weights into a joint model
2. Freeze the encoder layers while training only the classification head
3. Train and validate the model on EEG graph data
4. Save the best performing model based on validation accuracy

The module implements a transfer learning approach where a model pre-trained
using self-supervised learning is fine-tuned for a specific classification task.

WRITTEN BY AI
CHECKED AND VERIFIED BY AUTHOR
"""

import torch
import torch.nn as nn
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import json

from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping

from eeglearn.config import Config
from eeglearn.models.models import JointlyTrainModel
from eeglearn.utils.models import( setup_label_encoder, split_data, create_graph_loaders
                                ,calculate_class_weights, print_training_params,
                                setup_directories,get_labels_dict, write_epoch_log,
                                validate_model, get_experiment_filename)


data_path = Config.data_path
project_root = Config.project_root
testing_on_sample_data = Config.testing_on_sample_data
device = Config.device
num_workers = Config.num_workers
drop_last = Config.drop_last
skip_bads = Config.skip_bads
project_root = Config.project_root
data_path = Config.data_path
cleaned_data_path = Config.cleaned_data_path
energy_path = Config.energy_path
ignore_replication_nans = True
random_seed = Config.RANDOM_SEED
main_classes = Config.main_classes
optuna = Config.optuna

def load_ssl_weights(model, ssl_weights_path):
    """Load self-supervised weights into the joint model."""
    print(f"Loading weights from: {ssl_weights_path}")
    
    # Load self-supervised weights
    ssl_state = torch.load(ssl_weights_path, map_location=Config.device)
    model_state = model.state_dict()
    
    # Transfer matching weights (conv1, HF, HS)
    for key in ssl_state:
        if key in model_state:
            model_state[key] = ssl_state[key]
            print(f"Loaded: {key}")
    
    model.load_state_dict(model_state)
    
    # Freeze encoder parts, only train classification head (HC)
    for name, param in model.named_parameters():
        if not name.startswith('HC'):
            param.requires_grad = False
    
    print("Encoder frozen, only training classification head")
    return model


def train():
    """Train classification head using pre-trained self-supervised weights."""
    # Config

    device = Config.device
    batch_size = Config.batch_size # <-- fine tuned
    epochs = Config.epochs
    stop_at = Config.stop_at
     
    lr = Config.lr  # <-- fine tuned
    weight_decay = Config.weight_decay # <-- fine tuned
    hc_drop_rate = Config.drop_rate    # <-- fine tuned    
    hc_linear_size = Config.linear_size # <-- fine tuned
    pretrained_weights_path = Config.pretrained_weights_path
    model_weights_dir = Config.model_weights_dir / "fine_tune"
    model_metrics_dir = Config.metrics_dir / "fine_tune"

    print_training_params()
    setup_directories({"weights": model_weights_dir, "metrics": model_metrics_dir})

    # Check and print device information
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Using CPU for training")
    print(f"📱 Device: {device}")


    # Setup data
    encoder, n_classes = setup_label_encoder(ignore_replication_nans=True)
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
    train_loader = create_graph_loaders(
        participants=train_participants, encoder=encoder, batch_size=batch_size,
        data_split_type="train", perm_types=[None], drop_last=drop_last
    )
    
    validation_loader = create_graph_loaders(
        participants=validation_participants, encoder=encoder, batch_size=batch_size,
        data_split_type="validation", perm_types=[None], 
        drop_last= not testing_on_sample_data 
    )
    
    test_loader = create_graph_loaders(
        participants=test_participants, encoder=encoder, batch_size=batch_size,
        data_split_type="test", perm_types=[None], drop_last=drop_last
    )
    
    print("\n📊 Graph Loader Information:")
    print(f"  • Training loaders:")
    for loader_type, loader in train_loader.items():
        print(f"    - {loader_type}: {len(loader)} batches")
    
    print(f"\n  • Validation loader:")
    for loader_type, loader in validation_loader.items():
        print(f"    - {loader_type}: {len(loader)} batches")
    
    print(f"\n  • Test loader:")
    for loader_type, loader in test_loader.items():
        print(f"    - {loader_type}: {len(loader)} batches")
    print()

    # Initialize metrics tracking
    metrics = {
        'epoch': [], 
        'train_loss': [], 'train_acc': [], 
        'train_f1_weighted': [], 'train_f1_macro': [],
        'validation_loss': [], 'validation_acc': [], 
        'validation_f1_weighted': [], 'validation_f1_macro': []
    }
    
    print(f"⚠️  Training for epochs: {epochs}")

    net = JointlyTrainModel(
        inchannel=5, 
        gcn_out_size=Config.pretrained_gcn_out_size, # Fixed from pre-trained SSL model
        batch=batch_size, 
        K=Config.pretrained_k,                     # Fixed from pre-trained SSL model
        linear_size=Config.pretrained_linear_size, # For HF/HS, fixed from pre-trained SSL model
        drop_rate=Config.pretrained_drop_rate,     # For HF/HS, fixed from pre-trained SSL model
        linear_size_hc=hc_linear_size,             # For HC head
        drop_rate_hc=hc_drop_rate,                 # For HC head
        testmode=False, HF=120, HS=128, HC=n_classes
    ).to(device)
    
    assert os.path.exists(pretrained_weights_path), "No self-supervised weights found"
    net = load_ssl_weights(net, pretrained_weights_path)

    # Training setup
    criterion = nn.CrossEntropyLoss(weight =rescaled_class_weights).to(device)
    if not Config.use_class_weighting:
        criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=4, threshold=0.0001,
        threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-8
    )
    
    # Setup early stopping with ignite
    trainer = Engine(lambda engine, batch: batch)
    early_stopping = EarlyStopping(
        patience=stop_at,
        score_function=lambda engine: engine.state.metrics['val_macro_f1'],
        trainer=trainer
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, early_stopping)
    
    validation_highest_acc = 0.0
    best_val_f1_macro = 0.0
    best_val_f1_weighted = 0.0
    
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        net.train()
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []
        
        for ind, data in enumerate(train_loader['original']):
            data = data.to(device)
            
            # Forward pass (only get classification output)
            net.testmode = True
            output = net(data)
            net.testmode = False
            
            loss = criterion(output, data.y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            train_correct += (pred == data.y).sum().item()
            train_total += data.y.size(0)
            
            all_train_preds.extend(pred.cpu().numpy())
            all_train_labels.extend(data.y.cpu().numpy())
        
        avg_train_loss = epoch_loss / (ind + 1)
        train_acc = train_correct / train_total
        train_f1_weighted = f1_score(all_train_labels, all_train_preds, 
                                     average='weighted', zero_division=0)
        train_f1_macro = f1_score(all_train_labels, all_train_preds,
                                   average='macro', zero_division=0)
        
        # Validation using the validate_model function to match vanilla training
        validation_highest_acc, validation_current_acc, validation_epoch_loss, validation_f1_weighted, validation_f1_macro = validate_model(
            net, validation_loader, encoder, criterion, validation_highest_acc, 
            best_val_f1_macro,
            epoch, batch_size, lr, model_weights_dir, model_metrics_dir,
            testing_on_sample_data
        )
        
        # Update best scores
        if validation_f1_macro > best_val_f1_macro:
            best_val_f1_macro = validation_f1_macro
        
        if validation_f1_weighted > best_val_f1_weighted:
            best_val_f1_weighted = validation_f1_weighted
        
        # Early stopping check
        trainer.state.metrics = {'val_macro_f1': validation_f1_macro}
        trainer.fire_event(Events.EPOCH_COMPLETED)
        
        if trainer.should_terminate:
            print(f"🟢  Early stopping triggered at epoch {epoch}")
            break
        
        # Logging and scheduling
        write_epoch_log(epoch, batch_size, lr, validation_current_acc, model_metrics_dir)
        scheduler.step(validation_epoch_loss)
        
        # Store metrics
        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(avg_train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['train_f1_weighted'].append(train_f1_weighted)
        metrics['train_f1_macro'].append(train_f1_macro)
        
        metrics['validation_loss'].append(validation_epoch_loss)
        metrics['validation_acc'].append(validation_current_acc)
        metrics['validation_f1_weighted'].append(validation_f1_weighted)
        metrics['validation_f1_macro'].append(validation_f1_macro)
        
        # Print epoch results
        if epoch % 1 == 0:
            print(f'\n## Epoch [{epoch}/{epochs}] ##')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Training ACC: {train_acc:.4f}')
            print(f'Training F1 Weighted: {train_f1_weighted:.4f}')
            print(f'Training F1 Macro: {train_f1_macro:.4f}')
            print("----------------------------------------------")
            print(f'Validation Loss: {validation_epoch_loss:.4f}')
            print(f'Validation ACC: {validation_current_acc:.4f}')
            print(f'Validation F1 Weighted: {validation_f1_weighted:.4f}')
            print(f'Validation F1 Macro: {validation_f1_macro:.4f}')
            print(f'Best Validation ACC: {validation_highest_acc:.4f}')
            print(f'Best Validation F1 Weighted: {best_val_f1_weighted:.4f}')
            print(f'Best Validation F1 Macro: {best_val_f1_macro:.4f}')
            print("==============================================")
    
    # Save metrics to CSV
    metrics_filename = get_experiment_filename("training_metrics_fine_tune", "csv")
    pd.DataFrame(metrics).to_csv(model_metrics_dir / metrics_filename, index=False)
    
    return best_val_f1_macro


def train_with_kfold_cv(k_folds: int = 5) -> dict:
    """Train the fine-tuned model using k-fold cross-validation.
    
    Args:
        k_folds (int): Number of folds for cross-validation (default: 5)
        
    Returns:
        dict: Cross-validation results containing mean and std of metrics across folds
    """
    print(f"🔄 Starting {k_folds}-fold cross-validation for fine-tuned model")
    
    # Config
    device = Config.device
    batch_size = Config.batch_size
    epochs = Config.epochs
    stop_at = Config.stop_at
    lr = Config.lr
    weight_decay = Config.weight_decay
    hc_drop_rate = Config.drop_rate
    hc_linear_size = Config.linear_size
    pretrained_weights_path = Config.pretrained_weights_path
    
    model_config = {
        'model_type': 'JointlyTrainModel_FineTuned',
        'input_channels': 5,
        'gcn_out_size': Config.pretrained_gcn_out_size,  # Fixed from pre-trained
        'batch_size': batch_size,
        'K': Config.pretrained_k,  # Fixed from pre-trained
        'linear_size': Config.pretrained_linear_size,  # For HF/HS, fixed from pre-trained
        'drop_rate': Config.pretrained_drop_rate,  # For HF/HS, fixed from pre-trained
        'linear_size_hc': hc_linear_size,  # For HC head
        'drop_rate_hc': hc_drop_rate,  # For HC head
        'pretrained_weights_path': str(pretrained_weights_path),
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
    
    cv_model_weights_dir = Config.model_weights_dir / 'fine_tune_cv'
    cv_metrics_dir = Config.metrics_dir / 'fine_tune_cv'
    setup_directories({"weights": cv_model_weights_dir, "metrics": cv_metrics_dir})
    
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Using CPU for training")
    print(f"📱 Device: {device}")
    
    # Verify SSL weights exist
    assert os.path.exists(pretrained_weights_path), f"No self-supervised weights found at {pretrained_weights_path}"
    
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
    
    test_loader = create_graph_loaders(
        participants=test_participants,
        encoder=encoder,
        batch_size=batch_size,
        data_split_type="test",
        perm_types=[None],
        drop_last=drop_last
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
        
        rescaled_class_weights = calculate_class_weights(
            fold_train_participants, all_psych_labels, encoder, n_classes
        )
        
        print("🔄  Building graphs for this fold...")
        train_loader = create_graph_loaders(
            participants=fold_train_participants,
            encoder=encoder,
            batch_size=batch_size,
            data_split_type=f"train_fold_{fold}",
            perm_types=[None],
            drop_last=True
        )
        
        validation_loader = create_graph_loaders(
            participants=fold_val_participants,
            encoder=encoder,
            batch_size=batch_size,
            data_split_type=f"validation_fold_{fold}",
            perm_types=[None],
            drop_last=not testing_on_sample_data
        )
        
        # Initialize model with fine-tuning configuration
        net = JointlyTrainModel(
            inchannel=5,
            gcn_out_size=Config.pretrained_gcn_out_size,  # Fixed from pre-trained SSL model
            batch=batch_size,
            K=Config.pretrained_k,  # Fixed from pre-trained SSL model
            linear_size=Config.pretrained_linear_size,  # For HF/HS, fixed from pre-trained SSL model
            drop_rate=Config.pretrained_drop_rate,  # For HF/HS, fixed from pre-trained SSL model
            linear_size_hc=hc_linear_size,  # For HC head
            drop_rate_hc=hc_drop_rate,  # For HC head
            testmode=False, HF=120, HS=128, HC=n_classes
        ).to(device)
        
        # Load SSL weights and freeze encoder
        net = load_ssl_weights(net, pretrained_weights_path)
        
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
            
            for ind, data in enumerate(train_loader['original']):
                data = data.to(device)
                
                # Forward pass (only get classification output)
                net.testmode = True
                output = net(data)
                net.testmode = False
                
                y = data.y
                total_train_samples += y.size(0)
                batch_count += 1
                
                predictions = torch.argmax(output, dim=1)
                correct_predictions += torch.sum(predictions == y).item()
                
                all_train_preds.extend(predictions.cpu().numpy())
                all_train_labels.extend(y.cpu().numpy())
                
                loss = criterion(output, y)
                
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
            
            net.eval()
            validation_highest_acc, validation_current_acc, validation_epoch_loss, \
                validation_f1_weighted, validation_f1_macro = validate_model(
                net, validation_loader, encoder, criterion, fold_best_val_acc,
                fold_best_val_f1_macro, epoch, batch_size, lr, cv_model_weights_dir,
                cv_metrics_dir,
                testing_on_sample_data
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
                print(f"🟢  Early stopping triggered at epoch {epoch} for fold {fold + 1}")
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
    
    # Calculate cross-validation results
    cv_results = {}
    for metric in ['best_val_acc', 'best_val_f1_weighted', 'best_val_f1_macro',
                   'final_train_acc', 'final_train_f1_weighted', 'final_train_f1_macro']:
        values = fold_results[metric]
        cv_results[f'{metric}_mean'] = np.mean(values)
        cv_results[f'{metric}_std'] = np.std(values)
    
    # Save detailed fold results
    fold_results_df = pd.DataFrame(fold_results)
    cv_results_filename = get_experiment_filename(f"cv_{k_folds}fold_detailed_results_fine_tune", "csv")
    fold_results_df.to_csv(cv_metrics_dir / cv_results_filename, index=False)
    
    # Save model configuration
    model_config_filename = get_experiment_filename(f"cv_{k_folds}fold_model_config_fine_tune", "json")
    with open(cv_metrics_dir / model_config_filename, 'w') as f:
        json.dump(model_config, f, indent=4)
    print(f"📝 Model configuration saved to: {model_config_filename}")
    
    # Save full training history
    training_history_df = pd.DataFrame(full_training_history)
    training_history_filename = get_experiment_filename(f"cv_{k_folds}fold_training_history_fine_tune", "csv")
    training_history_df.to_csv(cv_metrics_dir / training_history_filename, index=False)
    print(f"📊 Full training history saved to: {training_history_filename}")
    
    # Save CV summary
    cv_summary = pd.DataFrame([cv_results])
    cv_summary_filename = get_experiment_filename(f"cv_{k_folds}fold_summary_fine_tune", "csv")
    cv_summary.to_csv(cv_metrics_dir / cv_summary_filename, index=False)
    
    print(f"\n{'='*60}")
    print(f"🎯 {k_folds}-Fold Cross-Validation Results (Fine-Tuned Model)")
    print(f"{'='*60}")
    print(f"Validation Accuracy:     {cv_results['best_val_acc_mean']:.4f} ± {cv_results['best_val_acc_std']:.4f}")
    print(f"Validation F1 Macro:     {cv_results['best_val_f1_macro_mean']:.4f} ± {cv_results['best_val_f1_macro_std']:.4f}")
    print(f"Validation F1 Weighted:  {cv_results['best_val_f1_weighted_mean']:.4f} ± {cv_results['best_val_f1_weighted_std']:.4f}")
    print(f"Training Accuracy:       {cv_results['final_train_acc_mean']:.4f} ± {cv_results['final_train_acc_std']:.4f}")
    print(f"Training F1 Macro:       {cv_results['final_train_f1_macro_mean']:.4f} ± {cv_results['final_train_f1_macro_std']:.4f}")
    print(f"Training F1 Weighted:    {cv_results['final_train_f1_weighted_mean']:.4f} ± {cv_results['final_train_f1_weighted_std']:.4f}")
    
    return cv_results


if __name__ == "__main__":
    train_with_kfold_cv(k_folds=Config.k_folds)