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
from eeglearn.models.models import VanillaGraphModel, EEGNet
from eeglearn.features.graphs import Graphs
from eeglearn.utils.models import (
    split_data, get_graphs_original, print_training_params,
    setup_directories, setup_label_encoder, calculate_class_weights,
    write_epoch_log, update_log, validate_model, create_graph_loaders,
    get_experiment_filename, get_raw_eeg_data, create_time_series_data_dataloader
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
ignore_replication_nans = True
random_seed = Config.RANDOM_SEED
main_classes = Config.main_classes
optuna = Config.optuna


def train() -> float:
    batch_size = Config.batch_size
    epochs = Config.epochs
    lr = Config.lr
    stop_at = Config.stop_at
    print_training_params()
    setup_directories(model_weights_dir, metrics_dir)
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

    for split_name, participants in [("train", train_participants), 
                                     ("valid", validation_participants),
                                       ("test", test_participants)]:
        print(f"n {split_name}: {len(participants)}")
    
    rescaled_class_weights = calculate_class_weights(train_participants,
                                                     all_psych_labels, 
                                                     encoder,
                                                     n_classes)
    print("⏳ Loading preprocessed EEG time series data...")
    if os.path.exists(data_path / "raw_train_loader.pt"):
        train_loader = torch.load(data_path / "raw_train_loader.pt")
    train_loader = create_time_series_data_dataloader(data_split_type="train",
                                                      participants=train_participants,
                                                      encoder=encoder,
                                                      batch_size=batch_size,
                                                      drop_last=drop_last)

    validation_loader = create_time_series_data_dataloader(data_split_type="validation",
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
        n_channels=26,  # Number of EEG channels
        n_timepoints=Config.eeg_net_n_time_steps,  # Number of time points
        n_classes=n_classes,  # Number of output classes
    ).to(device)

    print(net)
if __name__ == "__main__":
    train()