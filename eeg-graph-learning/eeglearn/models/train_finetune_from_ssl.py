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


def evaluate_test_set(model, test_loader, device):
    """Evaluate the model on the test set."""
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    print("\nEvaluating on test set...")
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            model.testmode = True
            output = model(data)
            model.testmode = False
            
            pred = torch.argmax(output, dim=1)
            test_correct += (pred == data.y).sum().item()
            test_total += data.y.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    
    test_acc = test_correct / test_total
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 (Weighted): {f1_weighted:.4f}")
    print(f"Test F1 (Macro): {f1_macro:.4f}")
    
    return test_acc, f1_weighted, f1_macro


def train_classification():
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
            net, validation_loader, encoder, validation_highest_acc, 
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


if __name__ == "__main__":
    train_classification() 