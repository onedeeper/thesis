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
    get_experiment_filename, get_raw_eeg_data, create_time_series_data_dataloader,
    validate_EEGNet_model
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
model_weights_dir = Config.model_weights_dir / 'baseline'
metrics_dir = Config.metrics_dir / 'baseline'
eeg_net_data_folder = Config.data_path / "eegnet"
ignore_replication_nans = True
random_seed = Config.RANDOM_SEED
main_classes = Config.main_classes
is_optuna_enabled = Config.optuna


def train(trial: optuna.Trial = None) -> float:
    batch_size = Config.batch_size
    epochs = Config.epochs
    lr = Config.lr
    stop_at = Config.stop_at
    weight_decay = Config.weight_decay
    n_channels =Config.n_eeg_channels
    n_timepoints=Config.eeg_net_n_time_steps
    drop_rate= Config.drop_rate
    kernel_length = Config.kernel_length

    print_training_params()
    setup_directories({"weights_dir" : model_weights_dir, 
                       "metrics_dir" : metrics_dir,
                       "eeg_net_data_dir" : eeg_net_data_folder})
    # Check and print device information
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
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
                                                      participants=train_participants,
                                                      encoder=encoder,
                                                      batch_size=batch_size,
                                                      drop_last=drop_last)

    validation_loader = create_time_series_data_dataloader(data_split_type="valid",
                                                participants=validation_participants,
                                                      encoder=encoder,
                                                      batch_size=batch_size,
                                                      drop_last=drop_last)
    
    test_loader = create_time_series_data_dataloader(data_split_type="test",
                                                      participants=test_participants,
                                                      encoder=encoder,
                                                      batch_size=batch_size,
                                                      drop_last=drop_last)
    print("\n📊 Data Loader Information:")
    print(f"\n  • Training loader: {len(train_loader)} batches")
    print(f"\n  • Validation loader: {len(validation_loader)} batches") 
    print(f"\n  • Test loader: {len(test_loader)} batches")
    print()

    metrics = {
        'epoch': [], 
        'train_loss': [], 'train_acc': [], 
        'train_f1_weighted': [], 'train_f1_macro': [],
        'validation_loss': [], 'validation_acc': [], 
        'validation_f1_weighted': [], 'validation_f1_macro': []
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

        net.train()
        for ind, data in enumerate(train_loader):
            X = data[0].float().to(device)  
            y = data[1].squeeze().long().to(device) 
            
            training_logits = net(X)
            total_train_samples += y.size(0)
            
            predictions = torch.argmax(training_logits, dim=1)
            correct_predictions += torch.sum(predictions == y).item()
            
            all_train_preds.extend(predictions.cpu().numpy())
            all_train_labels.extend(y.cpu().numpy())
            
            loss = criterion(training_logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # Validate model and get metrics
        (validation_highest_acc, 
         validation_current_acc,
         validation_epoch_loss,
         validation_f1_weighted, 
         validation_f1_macro) = validate_EEGNet_model(
            net=net,
            validation_loader=validation_loader,
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
        
        avg_train_loss = epoch_loss / (ind + 1)
        train_accuracy = correct_predictions / total_train_samples
        
        train_f1_weighted = f1_score(all_train_labels, 
                                     all_train_preds, 
                                     average='weighted',
                                     zero_division=0)
        train_f1_macro = f1_score(all_train_labels,
                                  all_train_preds,
                                  average='macro',
                                  zero_division=0)

        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(avg_train_loss)
        metrics['train_acc'].append(train_accuracy)
        metrics['train_f1_weighted'].append(train_f1_weighted)
        metrics['train_f1_macro'].append(train_f1_macro)
        
        metrics['validation_loss'].append(validation_epoch_loss)
        metrics['validation_acc'].append(validation_current_acc)
        metrics['validation_f1_weighted'].append(validation_f1_weighted)
        metrics['validation_f1_macro'].append(validation_f1_macro)
        
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
            print("==============================================")
    
    metrics_filename = get_experiment_filename("training_metrics_EEGNet", "csv")
    pd.DataFrame(metrics).to_csv(metrics_dir / metrics_filename, index=False)
    return best_validation_f1_score_macro

if __name__ == "__main__":
    train()