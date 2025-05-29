import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from eeglearn.config import Config
from eeglearn.utils.utils import get_labels_dict
from eeglearn.utils.models import (
    split_data,
    setup_label_encoder,
    get_graphs_original
)
# Assuming JointlyTrainModel and other potential models are in eeglearn.models.model
# This import will need to be valid in your environment.
from eeglearn.models.model import JointlyTrainModel


def evaluate(args):
    """Load a trained model and evaluate it on the test set."""
    print_evaluation_params(args)
    Config.set_global_seed(args.data_split_seed)
    device = Config.device

    encoder, n_classes = setup_label_encoder(ignore_replication_nans=True)
    
    # Model specific parameters (defaults from Config or common values)
    gcn_out_size = Config.gcn_out_size
    linear_size = Config.linear_size
    K = Config.K
    batch_size = args.batch_size

    # Instantiate model
    if args.model_type.lower() == "jointly":
        model = JointlyTrainModel(
            inchannel=5,  # Number of frequency bands, assumed from train_jointly.py
            gcn_out_size=gcn_out_size,
            batch=batch_size, # Model might use this, as seen in train_jointly.py
            K=K,
            linear_size=linear_size,
            drop_rate=0.0,  # No dropout during evaluation
            testmode=True,  # Critical for evaluation behavior
            HF=120, HS=128, HC=n_classes # Output heads sizes, from train_jointly.py
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    # Load model weights
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location=device)
    
    expected_keys = ['model', 'model_state_dict', 'state_dict'] # Common keys
    loaded_state_dict = None
    for key in expected_keys:
        if key in checkpoint:
            loaded_state_dict = checkpoint[key]
            break
    if loaded_state_dict is None: # Assume the checkpoint is the state_dict itself
        loaded_state_dict = checkpoint

    try:
        model.load_state_dict(loaded_state_dict)
    except RuntimeError as e:
        print(f"RuntimeError loading state_dict: {e}")
        print("Attempting to load with strict=False...")
        try:
            model.load_state_dict(loaded_state_dict, strict=False)
            print("Successfully loaded with strict=False. Some keys might have been missing/unexpected.")
        except Exception as e_strict_false:
            print(f"Failed to load state_dict even with strict=False: {e_strict_false}")
            raise
                           
    print(f"Successfully loaded model weights from {args.model_path}")
    if 'epoch' in checkpoint: print(f"Model trained for {checkpoint['epoch']} epochs.")
    if 'ACC' in checkpoint: print(f"Checkpoint accuracy (validation): {checkpoint['ACC']:.4f}")
    if 'F1' in checkpoint: print(f"Checkpoint F1 (validation): {checkpoint['F1']:.4f}")
    
    model.eval()

    # Prepare test data
    splits = split_data(ignore_replication_nans=True) 
    test_participants = splits['test']
    
    if not test_participants:
        print("No test participants found. Exiting.")
        return

    print(f"Evaluating on {len(test_participants)} test participants.")
    
    test_loader = get_graphs_original(
        files_to_load=test_participants,
        label_encoder=encoder,
        batch_size=batch_size,
        testing=True # This ensures drop_last=False
    )

    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss().to(device) 

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            
            # Forward pass: Assumes model in testmode takes a single graph object (original data)
            # The JointlyTrainModel's forward pass in testmode should handle this.
            out = model(data) 
            
            loss = criterion(out, data.y)
            total_loss += loss.item()
            
            preds = torch.argmax(out, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
    if not all_labels or not all_preds:
        print("No predictions or labels collected. Cannot calculate metrics.")
        return

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print("\n--- Test Set Evaluation Results ---")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")

    class_names = encoder.classes_
    print("\nClassification Report:")
    
    # Ensure labels are within the valid range for confusion_matrix and classification_report
    unique_labels_in_data = np.unique(np.concatenate((all_labels, all_preds))).astype(int)
    valid_labels_for_report = [l for l in unique_labels_in_data if l < len(class_names)]
    
    # If no valid labels, report will be problematic.
    if not valid_labels_for_report:
         print("Warning: No valid labels present in test predictions/labels for report generation.")
         report_str = "Classification report cannot be generated due to missing valid labels."
         cm_df = pd.DataFrame() # Empty dataframe for CM
    else:
        # Use np.arange(len(class_names)) for labels in metrics to ensure all classes are considered.
        # zero_division=0 handles cases where a class has no predictions or no true samples.
        report_str = classification_report(all_labels, all_preds, target_names=class_names, labels=np.arange(len(class_names)), zero_division=0)
        cm = confusion_matrix(all_labels, all_preds, labels=np.arange(len(class_names)))
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    print(report_str)
    print("\nConfusion Matrix:")
    print(cm_df)
    
    # Save results
    model_checkpoint_name = Path(args.model_path).stem
    results_dir = Config.metrics_dir / args.model_type / 'evaluation_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame({
        'metric': ['loss', 'accuracy', 'f1_weighted', 'f1_macro'],
        'value': [avg_loss, accuracy, f1_weighted, f1_macro]
    })
    results_df.to_csv(results_dir / f'test_summary_{model_checkpoint_name}.csv', index=False)
    
    if not cm_df.empty:
        cm_df.to_csv(results_dir / f'confusion_matrix_{model_checkpoint_name}.csv')
        # Plot and save confusion matrix
        plt.figure(figsize=(max(6, len(class_names) * 0.8), max(5, len(class_names) * 0.6)))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_checkpoint_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(results_dir / f'confusion_matrix_{model_checkpoint_name}.png')
        plt.close()


    with open(results_dir / f'classification_report_{model_checkpoint_name}.txt', 'w') as f:
        f.write(report_str)

    print(f"\nResults saved to {results_dir}")


def print_evaluation_params(args):
    """Prints the evaluation parameters."""
    print("📊 Evaluation Parameters:")
    print(f"   Model Type: {args.model_type}")
    print(f"   Model Path: {args.model_path}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Data Split Seed: {args.data_split_seed}")
    print(f"   Device: {Config.device}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained EEG classification model.")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["jointly"],  # Expand with other supported model types
        help="Type of the model to evaluate (e.g., 'jointly')."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model checkpoint file (.pkl or .pt)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=Config.batch_size, 
        help=f"Batch size for the test data loader (default: {Config.batch_size})."
    )
    parser.add_argument(
        "--data_split_seed",
        type=int,
        default=Config.RANDOM_SEED, 
        help=f"Seed for reproducing train/test data splits (default: {Config.RANDOM_SEED})."
    )

    parsed_args = parser.parse_args()
    
    if not os.path.exists(parsed_args.model_path):
        print(f"Error: Model file not found at {parsed_args.model_path}")
        exit(1)

    evaluate(parsed_args) 