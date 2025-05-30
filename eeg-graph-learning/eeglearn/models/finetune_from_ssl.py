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
from pathlib import Path
from sklearn.metrics import f1_score
from eeglearn.config import Config
from eeglearn.models.model import JointlyTrainModel
from eeglearn.utils.models import setup_label_encoder, split_data, create_graph_loaders


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
    batch_size = Config.batch_size
    epochs = Config.epochs
    lr = Config.lr
    
    # Setup data
    encoder, n_classes = setup_label_encoder(ignore_replication_nans=True)
    split = split_data(ignore_replication_nans=True)
    
    train_loader = create_graph_loaders(
        participants=split['train'], encoder=encoder, batch_size=batch_size,
        data_split="train", perm_types=[None], drop_last=True
    )['original']
    
    val_loader = create_graph_loaders(
        participants=split['valid'], encoder=encoder, batch_size=batch_size,
        data_split="validation", perm_types=[None], drop_last=False
    )['original']
    
    test_loader = create_graph_loaders(
        participants=split['test'], encoder=encoder, batch_size=batch_size,
        data_split="test", perm_types=[None], drop_last=False
    )['original']
    
    # Initialize model
    model = JointlyTrainModel(
        inchannel=5, gcn_out_size=Config.gcn_out_size, batch=batch_size, 
        K=Config.K, linear_size=Config.linear_size, drop_rate=Config.drop_rate,
        testmode=False, HF=120, HS=128, HC=n_classes
    ).to(device)
    
    # Load self-supervised weights
    ssl_weights_dir = Config.model_weights_dir / 'self_supervised'
    ssl_weights = list(ssl_weights_dir.glob("best_model_epoch_*"))
    
    if ssl_weights:
        model = load_ssl_weights(model, ssl_weights[0])
    else:
        print("No self-supervised weights found, training from scratch")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0
    
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data in train_loader:
            data = data.to(device)
            
            # Forward pass (only get classification output)
            model.testmode = True
            output = model(data)
            model.testmode = False
            
            loss = criterion(output, data.y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            train_correct += (pred == data.y).sum().item()
            train_total += data.y.size(0)
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                model.testmode = True
                output = model(data)
                model.testmode = False
                
                pred = torch.argmax(output, dim=1)
                val_correct += (pred == data.y).sum().item()
                val_total += data.y.size(0)
        
        val_acc = val_correct / val_total
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_classification_model.pth')
            print(f"Epoch {epoch}: New best validation accuracy: {val_acc:.4f}")
        
        print(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    print(f"Training completed! Best validation accuracy: {best_acc:.4f}")
    
    # Load best model and evaluate on test set
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load('best_classification_model.pth'))
    test_acc, test_f1_weighted, test_f1_macro = evaluate_test_set(model, test_loader, device)
    
    print(f"\nFinal Results:")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 (Weighted): {test_f1_weighted:.4f}")
    print(f"Test F1 (Macro): {test_f1_macro:.4f}")
    
    return model


if __name__ == "__main__":
    train_classification() 