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
    print("⏳ Loading data.")
    train_loader = create_time_series_data_dataloader(participants=train_participants,
                                                      encoder=encoder,
                                                      batch_size=batch_size,
                                                      drop_last=drop_last)

    validation_loader = create_time_series_data_dataloader(
                                                participants=validation_participants,
                                                      encoder=encoder,
                                                      batch_size=batch_size,
                                                      drop_last=drop_last)
    
    test_loader = create_time_series_data_dataloader(participants=test_participants,
                                                      encoder=encoder,
                                                      batch_size=batch_size,
                                                      drop_last=drop_last)