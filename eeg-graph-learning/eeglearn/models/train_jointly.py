"""Joint self-supervised and multi-task learning for EEG data.

Implementation of a joint training approach combining self-supervised learning with
multi-task learning for EEG data based on Li et al. 2023.
Handles data splitting, model training, and metrics tracking for frequency and spatial
graph representations.

Functions:
    split_data: Split participants into train/test/validation sets
    train: Execute the self-supervised training process and save metrics
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from eeglearn.config import Config
from eeglearn.utils.utils import get_details_from_file_name, get_labels_dict

from sklearn.model_selection import train_test_split
from AutoWeight import AutomaticWeightedLoss
from eeglearn.models.model import JointlyTrainModel
from eeglearn.features.graphs import Graphs
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Batch

from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from itertools import cycle
from sklearn.metrics import f1_score


# development settings
testing_on_sample_data = Config.testing_on_sample_data

# Training hyperparameters
batch_size : int = Config.batch_size
epochs : int = Config.epochs
lr : float = Config.lr
weight_decay : float = Config.weight_decay
stop_at : int = Config.stop_at

# Hardware and processing settings
device : str = Config.device
num_workers : int = Config.num_workers
drop_last : bool = Config.drop_last
skip_bads : bool = Config.skip_bads

# Path configurations
project_root : Path = Config.project_root
data_path : Path = Config.data_path
cleaned_data_path : Path = Config.cleaned_data_path
energy_path : Path = Config.energy_path
model_weights_dir : Path = Config.model_weights_dir / 'jointly'
metrics_dir : Path = Config.metrics_dir / 'jointly'

# Data processing settings
ignore_replication_nans : bool = True
random_seed : int = Config.RANDOM_SEED
main_classes : list[str] = Config.main_classes
optuna : bool = Config.optuna

# Model architecture parameters
drop_rate: float = Config.drop_rate
linear_size: int = Config.linear_size
gcn_out_size: int = Config.gcn_out_size
K: int = Config.K

def split_data(ignore_replication_nans : bool = False) -> dict:
    """Split participants into train, validation, and test sets using stratified sampling.

    Args:
        ignore_replication_nans: Whether to exclude participants with NaN labels 
                                 or in replication status.

    Returns:
        Dictionary with keys 'train', 'valid', 'test' containing lists of participant 
                                IDs.
    """

    all_participants = cleaned_data_path
    labels = get_labels_dict()
    participant_files = os.listdir(all_participants)
    N = []
    participant_labels = []
    
    if ignore_replication_nans:
        print("⚠️  Ignoring participants with Nan labels or in replication")
        for participant in participant_files:
            try:
                if labels[participant] in {'nan', 'NaN', np.nan, 'REPLICATION'} \
                    or labels[participant] not in main_classes:
                    continue
            except KeyError:
                continue 
            N.append(participant)
            participant_labels.append(labels[participant])
    else:
        N = participant_files
        participant_labels = [labels[p] for p in N]
    
    for participant in N:
        assert labels[participant] in main_classes

    train, test_valid, train_labels, test_valid_labels = train_test_split(
        N, participant_labels, 
        test_size=0.2, 
        random_state=random_seed,
        stratify=participant_labels
    )

    # Second split: Split the 20% into equal parts for test and validation
    test, valid, _, _ = train_test_split(
        test_valid,
        test_valid_labels,
        test_size=0.5,
        random_state=random_seed,
        stratify=test_valid_labels
    )

    # Verify all classes are present in each split
    train_classes = set(labels[p] for p in train)
    valid_classes = set(labels[p] for p in valid)
    test_classes = set(labels[p] for p in test)
    
    assert train_classes == set(main_classes), "Not all classes present in training set"
    assert valid_classes == set(main_classes), "Not all classes present in validation set"
    assert test_classes == set(main_classes), "Not all classes present in test set"

    print(f"Class distribution:")
    for split_name, split_data in [("Train", train), ("Valid", valid), ("Test", test)]:
        class_counts = {c: sum(1 for p in split_data if labels[p] == c) \
                        for c in main_classes}
        print(f"{split_name} set class counts:", class_counts)

    data_dict = {
        "train": train,
        "test": test,
        "valid": valid
    }

    return data_dict

def get_graphs_original(files_to_load : list, label_encoder : LabelEncoder, 
                        testing : bool = False):
    """Load energy objects for participants and convert them into graphs with labels.

    Args:
        files_to_load: List of participant files to load.
        label_encoder: Label encoder for psychological labels.
        testing: Whether the graphs are for testing (affects drop_last setting).

    Returns:
        PyTorch geometric graph data loader.
    """
    epoched_path : Path = energy_path / "energy_epoched"
    energy_files : list = os.listdir(epoched_path)
    energy_file_ids : dict = { get_details_from_file_name(file)[0] : file
                              for file in energy_files}
    # full_file_names_to_load = [energy_file_ids[file] for file in files_to_load]

    full_file_names_to_load = [energy_file_ids[file] \
                                for file in files_to_load \
                                if file in energy_file_ids]
    graphs =  Graphs(
                    perm_type=None,
                    energy_path=energy_path / "energy_epoched",
                    distance="ellipsoid", 
                     cleaned_data_path=cleaned_data_path,
                     n_workers=num_workers,
                     drop_last=drop_last,
                     batch_size = batch_size)
    if testing:
        graphs =  Graphs(
                    perm_type=None,
                    energy_path=energy_path / "energy_epoched",
                    distance="ellipsoid", 
                     cleaned_data_path=cleaned_data_path,
                     n_workers=num_workers,
                     drop_last=False,
                     batch_size = batch_size)
    return graphs.get_graphs(files_to_load=full_file_names_to_load, 
                             label_encoder= label_encoder,
                             skip_bads=skip_bads)

def train() -> float:
    """Train the joint self-supervised model on pretext and downstream tasks.
    
    Loads data, builds graph representations, trains the model using frequency,
    spatial and original graph data, and saves metrics and model weights.
    
    Returns:
        float: Best F1 score achieved during training
    """
    assert os.path.exists(cleaned_data_path),\
        "'data/cleaned' folder should exist and contain preprocessed data."
    assert os.path.exists(energy_path),\
            "'data/energy' folder should exist and contain derived energy features."
    assert os.path.exists(data_path),\
        "'data' folder should be in root directory for saving metrics and weights."
    print(f"⚠️  Saving weights to {model_weights_dir}")
    if not os.path.exists(model_weights_dir):
        model_weights_dir.mkdir(exist_ok=True, parents=True)

    print(f"⚠️  Saving metrics to {metrics_dir}")
    if not os.path.exists(metrics_dir):
        metrics_dir.mkdir(exist_ok=True, parents=True)
        
    print(f"⚠️  Training with data loader drop_last : {drop_last}")
    # Create necessary log files
    with open(metrics_dir / "epoch_log.txt", "w") as f:
        f.write("batch_size\tepoch\tlr\tdrop_rate\tacc\n")
    with open(metrics_dir / "update_log.txt", "w") as f:
        f.write("epoch\tlr\tbatch_size\tacc\n")

    all_psych_labels = get_labels_dict()
    all_unique_psych_labels = list(set(all_psych_labels.values()))
    if ignore_replication_nans:
        selected_unique_psych_labels = sorted([label for label in all_unique_psych_labels 
                              if label not in {'nan', 'NaN', np.nan, 'REPLICATION'} 
                              and label in main_classes])
    encoder = LabelEncoder()
    encoder.fit(list(selected_unique_psych_labels))
    n_classes = len(selected_unique_psych_labels)

    
    
    # Create a dummy engine for early stopping
    def dummy_update_fn(engine, batch):
        return batch
    
    trainer = Engine(dummy_update_fn)
    
    # Add early stopping handler
    early_stopping = EarlyStopping(
        patience=stop_at,  # Number of epochs to wait before stopping
        score_function=lambda engine: -engine.state.metrics['val_loss'],
        trainer=trainer
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, early_stopping)

    split : dict[str,list[str]] = split_data(ignore_replication_nans=\
                                             ignore_replication_nans)
    train_participants = split['train']
    validation_participants = split['valid']
    test_participants = split['test']
    print("⚠️  Participants split..")
    print(f"n train : {len(train_participants)}")
    print(f"n valid : {len(validation_participants)}")
    print(f"n test : {len(test_participants)}")

    # class weights for rescaling the loss during training.
    train_labels = np.array([all_psych_labels[p]             
                             for p in train_participants])
    train_labels_encoded = encoder.transform(train_labels)     

    class_frequencies = np.bincount(train_labels_encoded,
                         minlength=n_classes).astype(float)   
    class_weights = 1.0 / class_frequencies
    rescaled_class_weights = class_weights * (n_classes / class_weights.sum()) 
    rescaled_class_weights = torch.as_tensor(rescaled_class_weights,
                                    dtype=torch.float32,
                                    device=device)

    print("🔄  Building graphs.")
    train_freq_data = [fname for participant in train_participants
                       for fname in os.listdir(energy_path / "frequency_perms")
                       if participant in fname] 
    
    train_spatial_data = [fname for participant in train_participants
                       for fname in os.listdir(energy_path / "spatial_perms")
                       if participant in fname]
    
    spatial_graphs = Graphs(perm_type = "spatial",
                                energy_path= energy_path,
                                distance="ellipsoid",
                                cleaned_data_path=cleaned_data_path,
                                batch_size=batch_size,
                                n_neighbors=3,
                                shuffle=True,
                                drop_last=drop_last,
                                n_workers=num_workers)
    
    frequency_graphs = Graphs(perm_type = "frequency",
                            energy_path= energy_path,
                            distance="ellipsoid",
                            cleaned_data_path=cleaned_data_path,
                            batch_size=batch_size,
                            n_neighbors=3,
                            shuffle=True,
                            drop_last=drop_last,
                            n_workers=num_workers)
    
    loader_save_path = project_root / 'eeglearn' / 'models' / 'original_graph_loader.pt'

    if not os.path.exists(loader_save_path) or optuna:
        original_graph_loader = get_graphs_original(train_participants, encoder) 
        torch.save(original_graph_loader, loader_save_path)
    else:
        original_graph_loader = torch.load(loader_save_path)

    loader_save_path = project_root / 'eeglearn' /'models'/ 'spatial_graph_loader.pt'
    if not os.path.exists(loader_save_path) or optuna:
        spatial_graph_loader = spatial_graphs.get_graphs(files_to_load=
                                                         train_spatial_data,
                                                         skip_bads=skip_bads)
        torch.save(spatial_graph_loader, loader_save_path)
    else:
        spatial_graph_loader = torch.load(loader_save_path)

    loader_save_path = project_root / 'eeglearn'/ 'models' / 'frequency_graph_loader.pt'
    if not os.path.exists(loader_save_path) or optuna:
        frequency_graph_loader = frequency_graphs.get_graphs(files_to_load=
                                                             train_freq_data,
                                                             skip_bads=skip_bads)
        torch.save(frequency_graph_loader, loader_save_path)
    else:
        frequency_graph_loader = torch.load(loader_save_path)

    metrics : dict[str,list] = {
            'epoch' : [],
            'weighted_loss' : [],
            'freq_loss' : [],
            'spatial_loss' : [],
            'original_loss' : [],
            'freq_acc' : [],
            'spatial_acc' : [],
            'original_acc': [],
            'f1_score': []
        }
    print(f"⚠️  Training for epochs : {epochs}")

    awl = AutomaticWeightedLoss(3)
    net = JointlyTrainModel(
        inchannel=5, 
        gcn_out_size=gcn_out_size, 
        batch=batch_size, 
        K=K,
        linear_size=linear_size,
        drop_rate=drop_rate,
        testmode=False,
        HF=120, 
        HS=128, 
        HC=n_classes
    ).to(device)
    criterion_original = nn.CrossEntropyLoss(weight = rescaled_class_weights).to(device)
    criterion_permuted = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           factor=0.1, 
                                                           patience=4,
                                                           threshold=0.0001,
                                                           threshold_mode='rel',
                                                           cooldown=1, min_lr=0,
                                                           eps=1e-8)
    highest_acc = 0.0
    best_f1_score = 0.0
    for epoch in range(epochs):
        loader = zip(frequency_graph_loader, spatial_graph_loader,
                     cycle(original_graph_loader))
        epoch_weighted_loss = 0.0
        epoch_loss_freq = 0.0
        epoch_loss_spatial = 0.0
        epoch_loss_original = 0.0
        correct_pred_freq = 0
        correct_pred_spatial = 0
        correct_pred_original  = 0

        for ind, batch in enumerate(loader):
            fdata, sdata, gdata = batch
            fdata, sdata, gdata = fdata.to(device),\
                                 sdata.to(device),\
                                 gdata.to(device)
            freq_out, spatial_out, original_out, = net(fdata, sdata, gdata)
            # the true pseudo labels and true connectivity graphs
            y_freq, y_spatial, y_original = fdata.y, sdata.y, gdata.y
            _, pred1 = torch.max(freq_out, dim=1)
            _, pred2 = torch.max(spatial_out, dim=1)
            _, pred3 = torch.max(original_out, dim=1)

            correct_pred_freq += sum([1 for a,b in zip(pred1, y_freq) if a==b])
            correct_pred_spatial += sum([1 for a,b in zip(pred2, y_spatial) if a==b])
            correct_pred_original += sum([1 for a,b in zip(pred3, y_original) if a==b])
            loss_freq = criterion_permuted(freq_out, y_freq)
            loss_spatial = criterion_permuted(spatial_out, y_spatial)
            loss_original = criterion_original(original_out, y_original)
            # balanced loss from the multiple tasks. 
            loss = awl(loss_freq, loss_spatial, loss_original)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_weighted_loss += float(loss.item())
            epoch_loss_freq += float(loss_freq.item())
            epoch_loss_spatial += float(loss_spatial.item())
            epoch_loss_original += float(loss_original.item())
        highest_acc, current_acc, epoch_loss, f1 = validate(
                                        validate_data=validation_participants,
                                        net = net,
                                        label_encoder= encoder, 
                                        highest_acc=highest_acc,
                                        best_f1_score = best_f1_score,
                                        epoch=epoch)
        if f1 > best_f1_score:
            best_f1_score = f1
        # Update engine metrics and check early stopping
        trainer.state.metrics = {'val_loss': epoch_loss}
        trainer.fire_event(Events.EPOCH_COMPLETED)
        
        # Check if early stopping criteria met
        if trainer.should_terminate:
            print(f" 🟢  Early stopping triggered at epoch {epoch}")
            break
        writeEachEpoch(epoch, batch_size, lr, current_acc)
        scheduler.step(epoch_weighted_loss) 
        denominator = (ind+1)*batch_size
        if epoch % 5 == 0:
                print()
                print(f'## highest_acc {highest_acc:.4f} curr_acc {current_acc:.4f}##')
                print(f'batch {batch_size}, lr {lr}')
                print()

        epoch_avg_weighted_loss = epoch_weighted_loss/(ind+1)
        epoch_avg_freq_loss = epoch_loss_freq/(ind+1)
        epoch_avg_spatial_loss = epoch_loss_spatial/(ind+1)
        epoch_avg_original_loss = epoch_loss_original/(ind+1)
        freq_acc = correct_pred_freq/denominator
        spatial_acc = correct_pred_spatial/denominator
        original_acc = correct_pred_original/denominator
        # Save metrics
        metrics['epoch'].append(epoch)
        metrics['weighted_loss'].append(epoch_avg_weighted_loss)
        metrics['freq_loss'].append(epoch_avg_freq_loss)
        metrics['spatial_loss'].append(epoch_avg_spatial_loss)
        metrics['original_loss'].append(epoch_avg_original_loss)
        metrics['freq_acc'].append(freq_acc)
        metrics['spatial_acc'].append(spatial_acc)
        metrics['original_acc'].append(original_acc)
        metrics['f1_score'].append(f1)

        print(f'Epoch [{epoch}/{epochs}]')
        print(f'Weighted loss [{epoch_avg_weighted_loss:.4f}]  ')
        print(f'Frequency loss[{epoch_avg_freq_loss:.4f}]')
        print(f'Spatial loss[{epoch_avg_spatial_loss:.4f}]')
        print(f'Original loss[{epoch_avg_original_loss:.4f}]')
        print('ACC@1:')
        print(f'fequency ACC[{correct_pred_freq/denominator:.4f}]')
        print(f'spatial ACC[{correct_pred_spatial/denominator:.4f}]')
        print(f'original ACC[{correct_pred_original/denominator:.4f}]')
        print(f'F1 Score[{f1:.4f}]')
        print("----------------------------------------------")
        
    pd.DataFrame(metrics).to_csv(metrics_dir / "training_metrics_jointly.csv",
                                 index=False)
    return best_f1_score

def writeEachEpoch(epoch, batchsize, lr, current_acc):
    """Log epoch information to epoch_log.txt.
    
    Args:
        epoch: Current epoch number.
        batchsize: Batch size used for training.
        lr: Learning rate used for training.
        current_acc: Current validation accuracy.
    """
    drop_rate = Config.drop_rate
    log = []
    log.append(f'{batchsize}\t{epoch}\t{lr}\t{drop_rate}\t{current_acc:.4f}\n')
    with open(metrics_dir / "epoch_log.txt", 'a') as f:
        f.writelines(log)


def updatelog(epoch, acc):
    """Update the log file with best model information.
    
    Args:
        epoch: Current epoch number.
        acc: Current best accuracy.
    """
    log = []
    log.append(f'{epoch}\t{lr}\t{batch_size}\t{acc:.4f}\n')
    with open(metrics_dir / "update_log.txt", 'a') as f:
        f.writelines(log)


def validate(net, validate_data, label_encoder, highest_acc, best_f1_score,epoch):
    """Evaluate model performance on validation data.
    
    Args:
        net: Model to evaluate.
        validate_data: List of validation participant IDs.
        label_encoder: Label encoder for psychological labels.
        highest_acc: Current highest accuracy seen so far.
        epoch: Current epoch number.
        
    Returns:
        Tuple of (highest_acc, current_acc, epoch_loss, f1).
    """
    criterion = nn.CrossEntropyLoss().to(device)
    gloader = get_graphs_original(validate_data, 
                                  label_encoder=label_encoder,
                                  testing=testing_on_sample_data)
    net.testmode = True
    net.eval()
    epoch_loss = 0.0
    correct_pred = 0
    total_samples = 0
    
    # Lists to store all predictions and true labels for f1 calculation
    all_preds = []
    all_labels = []
    
    for _, data in enumerate(gloader):
        data = data.to(device)
        current_batch_size = data.y.size(0)
        total_samples += current_batch_size

        if testing_on_sample_data and (current_batch_size < batch_size):
            data_list = data.to_data_list()
            needed = batch_size - current_batch_size
            additional_samples = [data_list[i % current_batch_size]\
                                  for i in range(needed)]
            data_list.extend(additional_samples)
            data = Batch.from_data_list(data_list)
            # Update total_samples to account for padding
            total_samples = total_samples - current_batch_size + batch_size
            
        out = net(data)
        y = data.y
        _, pre = torch.max(out, dim=1)

        # Only store non-padded predictions and labels for f1 calculation
        if testing_on_sample_data and (current_batch_size < batch_size):
            # Only take the original samples, not the padding
            all_preds.extend(pre[:current_batch_size].cpu().numpy())
            all_labels.extend(y[:current_batch_size].cpu().numpy())
        else:
            all_preds.extend(pre.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        correct_pred += sum([1 for a, b in zip(pre, y) if a == b])
        loss = criterion(out, y)

        epoch_loss += float(loss.item())

    ACC = correct_pred / total_samples
    
    # Calculate F1 score on non-padded samples only
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    if ACC > highest_acc:
        updatelog(epoch = epoch, acc=ACC)
        highest_acc = ACC
        ck = {}
        ck['epoch'] = epoch
        ck['model'] = net.state_dict()
        ck['ACC'] = ACC
        ck['F1'] = f1
        if f1 > best_f1_score:
            torch.save(ck, model_weights_dir / f"{ACC:.3f}_{f1:.3f}_checkpoint.pkl")

    net.train()
    net.testmode=False
    return highest_acc, ACC, epoch_loss, f1

if __name__ == "__main__":
    train()