import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
from eeglearn.config import Config
from eeglearn.models.models import EEGNet
from eeglearn.utils.models import setup_label_encoder
from eeglearn.utils.models import create_time_series_data_dataloader, split_data

def evaluate():
    """
    Load a trained EEGNet model, evaluate it on the test set, and generate comprehensive performance metrics and visualizations.
    
    Returns:
        None: Prints metrics and saves visualization plots.
    """
    device = Config.device
    encoder, n_classes = setup_label_encoder(ignore_replication_nans=True)
    batch_size = Config.batch_size
    epochs = Config.epochs
    lr = Config.lr
    weight_decay = Config.weight_decay
    drop_rate = Config.drop_rate
    n_channels = Config.n_eeg_channels
    n_timepoints = Config.eeg_net_n_time_steps
    kernel_length = Config.kernel_length
    stop_at = Config.stop_at
    
    # Model parameters
    model = EEGNet(
        n_channels=n_channels,  
        n_timepoints=n_timepoints,  
        n_classes=n_classes, 
        kernel_length=kernel_length,
        dropout_rate=drop_rate
    ).to(device)

    # Load model weights
    model_path = "/Users/udeshhabaraduwa/thesis _local/thesis/eeg-graph-learning/data/weights/baseline/test_Acc_0.333_weighted_f1_0.338_macro_f1_0.341_checkpoint.pkl"
    checkpoint = torch.load(model_path, map_location=device)
    
    # Try to load state dict
    for key in ['model', 'model_state_dict', 'state_dict']:
        if key in checkpoint:
            model.load_state_dict(checkpoint[key])
            break
    else:  # If no keys found, assume checkpoint is the state dict
        model.load_state_dict(checkpoint)
                           
    print(f"Successfully loaded model weights from {model_path}")
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    model.eval()

    # Load test data
    eeg_net_data_folder = Config.data_path / "eegnet"
    
    # Get data split
    if Config.load_data_split_from != "":
        print(f"⚠️  Data split loaded from {Config.data_path / Config.load_data_split_from}")
        split = torch.load(Config.data_path / Config.load_data_split_from)
    else:
        split = split_data()
    
    test_participants = split['test']
    
    test_loader = create_time_series_data_dataloader(
        data_split_type="test",
        eegnet_data_path=eeg_net_data_folder,
        participants=test_participants,
        label_encoder=encoder,
        batch_size=Config.batch_size,
        drop_last=Config.drop_last,
        num_workers=Config.num_workers
    )
    print(f"Loaded test data from {eeg_net_data_folder}")

    # Check if all classes are present in test set
    test_labels = []
    for data in test_loader:
        y = data[1].squeeze().long()
        test_labels.extend(encoder.inverse_transform(y))
    unique_test_labels = set(test_labels)
    all_classes = set(encoder.classes_)
    missing_classes = all_classes - unique_test_labels
    
    if missing_classes:
        print("\nWARNING: The following classes are missing from the test set:")
        for cls in missing_classes:
            print(f"- {cls}")
        print("\nThis may affect the reliability of the evaluation metrics.")
    else:
        print("\nAll classes are present in the test set.")

    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss().to(device) 

    with torch.no_grad():
        for data in test_loader:
            X = data[0].float().to(device)
            y = data[1].squeeze().long().to(device)
            
            labels = encoder.inverse_transform(y.cpu())
            label_counts = {}
            for label in labels:
                label_counts[str(label)] = label_counts.get(label, 0) + 1
            # print(label_counts)
            
            out = model(X) 
            loss = criterion(out, y)
            total_loss += loss.item()
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    
    # Calculate precision, recall, and F1-score for each class
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Decode labels back to original classes for reporting
    class_names = list(encoder.classes_)
    decoded_true = encoder.inverse_transform(all_labels)
    decoded_pred = encoder.inverse_transform(all_preds)
    
    # Print comprehensive metrics
    print("\n" + "="*60)
    print("EEGNET MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print(f"Micro F1-Score: {f1_micro:.4f}")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")
    
    # Calculate macro and weighted precision/recall
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    
    print("\nPer-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 80)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1_per_class[i]:<10.4f} {support[i]:<10}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print("-" * 80)
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    # Set up seaborn style - white background, no grid
    sns.set_style("white")
    sns.despine()
    
    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    sns.despine()
    plt.tight_layout()
    plt.savefig('eegnet_confusion_matrix.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    # 2. Normalized Confusion Matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    sns.despine()
    plt.tight_layout()
    plt.savefig('eegnet_confusion_matrix_normalized.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    # 3. Per-Class Performance Metrics - Small Multiples
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Precision
    sns.barplot(x=list(range(len(class_names))), y=precision, ax=axes[0], color='steelblue')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Precision')
    axes[0].set_xticks(range(len(class_names)))
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].set_ylim(0, 1)
    sns.despine(ax=axes[0])
    
    # Recall
    sns.barplot(x=list(range(len(class_names))), y=recall, ax=axes[1], color='steelblue')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Recall')
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_ylim(0, 1)
    sns.despine(ax=axes[1])
    
    # F1-Score
    sns.barplot(x=list(range(len(class_names))), y=f1_per_class, ax=axes[2], color='steelblue')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_xticks(range(len(class_names)))
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].set_ylim(0, 1)
    sns.despine(ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('eegnet_per_class_metrics.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    # 4. Class Distribution in Test Set
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    class_distribution = [counts[i] for i in range(len(class_names))]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=class_names, y=class_distribution, ax=ax, color='steelblue')
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Samples')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, count in enumerate(class_distribution):
        ax.text(i, count + max(class_distribution) * 0.01, str(count), 
                ha='center', va='bottom')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig('eegnet_test_set_distribution.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    # 5. Model Performance Summary
    metrics_summary = {
        'Accuracy': accuracy,
        'Macro F1': f1_macro,
        'Weighted F1': f1_weighted,
        'Macro Precision': macro_precision,
        'Macro Recall': macro_recall
    }
    
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics_names = list(metrics_summary.keys())
    metrics_values = list(metrics_summary.values())
    
    sns.barplot(x=metrics_names, y=metrics_values, ax=ax, color='steelblue')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, value in enumerate(metrics_values):
        ax.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig('eegnet_model_performance_summary.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"\nVisualization files saved:")
    print("- eegnet_confusion_matrix.png")
    print("- eegnet_confusion_matrix_normalized.png") 
    print("- eegnet_per_class_metrics.png")
    print("- eegnet_test_set_distribution.png")
    print("- eegnet_model_performance_summary.png")

if __name__ == "__main__":
    evaluate() 