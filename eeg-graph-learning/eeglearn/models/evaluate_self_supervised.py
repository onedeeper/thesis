import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (f1_score, confusion_matrix, classification_report, 
                             precision_recall_fscore_support)
from eeglearn.config import Config
from eeglearn.models.models import SelfSupervisedTrain
from eeglearn.utils.models import (setup_label_encoder, create_graph_loaders, 
                                   calculate_class_weights) 
import itertools
def evaluate():
    """
    Load a trained model, evaluate it on the test set, and generate comprehensive performance metrics and visualizations.
    
    Returns:
        None: Prints metrics and saves visualization plots.

    WRITTEN WITH AI
    INSPECTED AND VERIFIED BY AUTHOR
    """
    device = Config.device
    # Note: For self-supervised pretext tasks, we don't need label encoder
    # But we use the same data split as other training approaches
    batch_size = Config.batch_size
    epochs = Config.epochs
    lr = Config.lr
    weight_decay = Config.weight_decay
    drop_rate = Config.drop_rate
    gcn_out_size = Config.gcn_out_size
    linear_size = Config.linear_size
    K = Config.K
    stop_at = Config.stop_at
    
    # Model parameters - updated to match training script
    model = SelfSupervisedTrain(
        inchannel=5, 
        gcn_out_size=gcn_out_size, 
        batch=batch_size, 
        K=K,
        linear_size=linear_size,
        drop_rate=drop_rate,
        HF=120, 
        HS=128
    ).to(device)

    # Load model weights
    model_path = "/Users/udeshhabaraduwa/thesis _local/thesis/eeg-graph-learning/data/weights/best_models/all_data/all_data_best_model_self_supervised.pt"
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

    # Load test data - updated to match training script approach
    test_graphs_spatial_path = "/Users/udeshhabaraduwa/thesis _local/thesis/eeg-graph-learning/data/graph_list/all_data_self_supervised_test_spatial_graph_list.pt"
    test_graphs_frequency_path  = "/Users/udeshhabaraduwa/thesis _local/thesis/eeg-graph-learning/data/graph_list/all_data_self_supervised_test_frequency_graph_list.pt"
    test_graphs_spatial = torch.load(test_graphs_spatial_path)
    test_graphs_frequency = torch.load(test_graphs_frequency_path)
    
    # Create test loaders for both frequency and spatial - matching training script
    test_loaders = create_graph_loaders(
        data_split_type="test",
        batch_size=Config.batch_size,
        graph_lists={"spatial" : test_graphs_spatial,
                     "frequency": test_graphs_frequency},
        encoder=None,  # No encoder needed for pretext tasks
        perm_types=["frequency", "spatial"]
    )
    print(f"Loaded test data from:")
    print(f"  • Spatial: {test_graphs_spatial_path}")
    print(f"  • Frequency: {test_graphs_frequency_path}")

    
    all_preds_freq = []
    all_labels_freq = []
    all_preds_spatial = []
    all_labels_spatial = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss().to(device) 

    with torch.no_grad():
        test_loader = zip(test_loaders['frequency'], test_loaders['spatial'])
        for freq_data, spatial_data in test_loader:
            freq_data, spatial_data = freq_data.to(device), spatial_data.to(device)
            freq_logits, spatial_logits = model(freq_data, spatial_data)
            
            y_freq, y_spatial = freq_data.y, spatial_data.y
            
            # Calculate losses
            loss_freq = criterion(freq_logits, y_freq)
            loss_spatial = criterion(spatial_logits, y_spatial)
            total_loss += (loss_freq.item() + loss_spatial.item()) / 2
            
            # Get predictions
            _, pred_freq = torch.max(freq_logits, dim=1)
            _, pred_spatial = torch.max(spatial_logits, dim=1)
            
            all_preds_freq.extend(pred_freq.cpu().numpy())
            all_labels_freq.extend(y_freq.cpu().numpy())
            all_preds_spatial.extend(pred_spatial.cpu().numpy())
            all_labels_spatial.extend(y_spatial.cpu().numpy())

    # Calculate metrics for frequency task
    avg_loss = total_loss / len(list(zip(test_loaders['frequency'], test_loaders['spatial'])))
    
    # Frequency task metrics
    accuracy_freq = np.mean(np.array(all_preds_freq) == np.array(all_labels_freq))
    f1_weighted_freq = f1_score(all_labels_freq, all_preds_freq, average='weighted', zero_division=0)
    f1_macro_freq = f1_score(all_labels_freq, all_preds_freq, average='macro', zero_division=0)
    f1_micro_freq = f1_score(all_labels_freq, all_preds_freq, average='micro', zero_division=0)
    
    # Spatial task metrics  
    accuracy_spatial = np.mean(np.array(all_preds_spatial) == np.array(all_labels_spatial))
    f1_weighted_spatial = f1_score(all_labels_spatial, all_preds_spatial, average='weighted', zero_division=0)
    f1_macro_spatial = f1_score(all_labels_spatial, all_preds_spatial, average='macro', zero_division=0)
    f1_micro_spatial = f1_score(all_labels_spatial, all_preds_spatial, average='micro', zero_division=0)
    
    # Calculate precision, recall, and F1-score for each class - frequency task
    precision_freq, recall_freq, f1_per_class_freq, support_freq = precision_recall_fscore_support(
        all_labels_freq, all_preds_freq, average=None, zero_division=0
    )
    
    # Calculate precision, recall, and F1-score for each class - spatial task
    precision_spatial, recall_spatial, f1_per_class_spatial, support_spatial = precision_recall_fscore_support(
        all_labels_spatial, all_preds_spatial, average=None, zero_division=0
    )
    
    # Get class names (assuming they are 0, 1 for the pretext tasks)
    n_classes_freq = len(np.unique(all_labels_freq))
    n_classes_spatial = len(np.unique(all_labels_spatial))
    class_names_freq = [f"Class {i}" for i in range(n_classes_freq)]
    class_names_spatial = [f"Class {i}" for i in range(n_classes_spatial)]
    
    # Print comprehensive metrics
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS - SELF-SUPERVISED PRETEXT TASKS")
    print("="*60)
    print(f"Average Test Loss: {avg_loss:.4f}")
    
    print("\nFREQUENCY TASK METRICS:")
    print("-" * 40)
    print(f"Accuracy: {accuracy_freq:.4f} ({accuracy_freq*100:.2f}%)")
    print(f"Macro F1-Score: {f1_macro_freq:.4f}")
    print(f"Micro F1-Score: {f1_micro_freq:.4f}")
    print(f"Weighted F1-Score: {f1_weighted_freq:.4f}")
    
    print("\nSPATIAL TASK METRICS:")
    print("-" * 40)
    print(f"Accuracy: {accuracy_spatial:.4f} ({accuracy_spatial*100:.2f}%)")
    print(f"Macro F1-Score: {f1_macro_spatial:.4f}")
    print(f"Micro F1-Score: {f1_micro_spatial:.4f}")
    print(f"Weighted F1-Score: {f1_weighted_spatial:.4f}")
    
    # Calculate macro and weighted precision/recall for frequency task
    macro_precision_freq = np.mean(precision_freq)
    macro_recall_freq = np.mean(recall_freq)
    weighted_precision_freq = np.average(precision_freq, weights=support_freq)
    weighted_recall_freq = np.average(recall_freq, weights=support_freq)
    
    # Calculate macro and weighted precision/recall for spatial task
    macro_precision_spatial = np.mean(precision_spatial)
    macro_recall_spatial = np.mean(recall_spatial)
    weighted_precision_spatial = np.average(precision_spatial, weights=support_spatial)
    weighted_recall_spatial = np.average(recall_spatial, weights=support_spatial)
    
    print(f"\nFrequency Task - Macro Precision: {macro_precision_freq:.4f}")
    print(f"Frequency Task - Macro Recall: {macro_recall_freq:.4f}")
    print(f"Frequency Task - Weighted Precision: {weighted_precision_freq:.4f}")
    print(f"Frequency Task - Weighted Recall: {weighted_recall_freq:.4f}")
    
    print(f"\nSpatial Task - Macro Precision: {macro_precision_spatial:.4f}")
    print(f"Spatial Task - Macro Recall: {macro_recall_spatial:.4f}")
    print(f"Spatial Task - Weighted Precision: {weighted_precision_spatial:.4f}")
    print(f"Spatial Task - Weighted Recall: {weighted_recall_spatial:.4f}")
    
    print("\nPer-Class Metrics - Frequency Task:")
    print("-" * 80)
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 80)
    for i, class_name in enumerate(class_names_freq):
        print(f"{class_name:<15} {precision_freq[i]:<10.4f} {recall_freq[i]:<10.4f} {f1_per_class_freq[i]:<10.4f} {support_freq[i]:<10}")
    
    print("\nPer-Class Metrics - Spatial Task:")
    print("-" * 80)
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 80)
    for i, class_name in enumerate(class_names_spatial):
        print(f"{class_name:<15} {precision_spatial[i]:<10.4f} {recall_spatial[i]:<10.4f} {f1_per_class_spatial[i]:<10.4f} {support_spatial[i]:<10}")
    
    # Detailed classification reports
    print("\nDetailed Classification Report - Frequency Task:")
    print("-" * 80)
    print(classification_report(all_labels_freq, all_preds_freq, target_names=class_names_freq, zero_division=0))
    
    print("\nDetailed Classification Report - Spatial Task:")
    print("-" * 80)
    print(classification_report(all_labels_spatial, all_preds_spatial, target_names=class_names_spatial, zero_division=0))
    
    # Set up seaborn style - white background, no grid
    sns.set_style("white")
    sns.despine()
    
    
    # 5. Per-Class Performance Metrics Comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Frequency task metrics
    sns.barplot(x=list(range(len(class_names_freq))), y=precision_freq, ax=axes[0,0], color='steelblue')
    axes[0,0].set_xlabel('Class')
    axes[0,0].set_ylabel('Precision')
    axes[0,0].set_title('Frequency Task')
    axes[0,0].set_xticks(range(len(class_names_freq)))
    axes[0,0].set_xticklabels(class_names_freq, rotation=45, ha='right')
    axes[0,0].set_ylim(0, 1)
    sns.despine(ax=axes[0,0])
    
    sns.barplot(x=list(range(len(class_names_freq))), y=recall_freq, ax=axes[0,1], color='steelblue')
    axes[0,1].set_xlabel('Class')
    axes[0,1].set_ylabel('Recall')
    axes[0,1].set_title('Frequency Task')
    axes[0,1].set_xticks(range(len(class_names_freq)))
    axes[0,1].set_xticklabels(class_names_freq, rotation=45, ha='right')
    axes[0,1].set_ylim(0, 1)
    sns.despine(ax=axes[0,1])
    
    sns.barplot(x=list(range(len(class_names_freq))), y=f1_per_class_freq, ax=axes[0,2], color='steelblue')
    axes[0,2].set_xlabel('Class')
    axes[0,2].set_ylabel('F1-Score')
    axes[0,2].set_title('Frequency Task')
    axes[0,2].set_xticks(range(len(class_names_freq)))
    axes[0,2].set_xticklabels(class_names_freq, rotation=45, ha='right')
    axes[0,2].set_ylim(0, 1)
    sns.despine(ax=axes[0,2])
    
    # Spatial task metrics
    sns.barplot(x=list(range(len(class_names_spatial))), y=precision_spatial, ax=axes[1,0], color='steelblue')
    axes[1,0].set_xlabel('Class')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].set_title('Spatial Task')
    axes[1,0].set_xticks(range(len(class_names_spatial)))
    axes[1,0].set_xticklabels(class_names_spatial, rotation=45, ha='right')
    axes[1,0].set_ylim(0, 1)
    sns.despine(ax=axes[1,0])
    
    sns.barplot(x=list(range(len(class_names_spatial))), y=recall_spatial, ax=axes[1,1], color='steelblue')
    axes[1,1].set_xlabel('Class')
    axes[1,1].set_ylabel('Recall')
    axes[1,1].set_title('Spatial Task')
    axes[1,1].set_xticks(range(len(class_names_spatial)))
    axes[1,1].set_xticklabels(class_names_spatial, rotation=45, ha='right')
    axes[1,1].set_ylim(0, 1)
    sns.despine(ax=axes[1,1])
    
    sns.barplot(x=list(range(len(class_names_spatial))), y=f1_per_class_spatial, ax=axes[1,2], color='steelblue')
    axes[1,2].set_xlabel('Class')
    axes[1,2].set_ylabel('F1-Score')
    axes[1,2].set_title('Spatial Task')
    axes[1,2].set_xticks(range(len(class_names_spatial)))
    axes[1,2].set_xticklabels(class_names_spatial, rotation=45, ha='right')
    axes[1,2].set_ylim(0, 1)
    sns.despine(ax=axes[1,2])
    
    plt.tight_layout()
    plt.savefig('per_class_metrics_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    def _idx_to_perm(idx: int, n_bands: int = 5):
        return list(itertools.permutations(range(n_bands)))[idx]

    def _build_band_matrix(true_idx, pred_idx, n_bands: int = 5):
        M = np.zeros((n_bands, n_bands), dtype=int)
        for t_i, p_i in zip(true_idx, pred_idx):
            t_perm = _idx_to_perm(int(t_i), n_bands)
            p_perm = _idx_to_perm(int(p_i), n_bands)
            for orig_pos, band in enumerate(t_perm):
                pred_pos = p_perm.index(band)
                M[band, pred_pos] += 1
        return (M.T / M.sum(1, keepdims=True)).T

    # Frequency task
    band_acc_freq = _build_band_matrix(all_labels_freq, all_preds_freq)
    plt.figure(figsize=(5, 4))
    sns.heatmap(band_acc_freq, annot=True, fmt=".2f", cmap="Blues",
                cbar_kws={"shrink": 0.8})
    plt.xlabel("Predicted position"); plt.ylabel("True band")
    plt.title("Band relocation accuracy – Frequency")
    plt.tight_layout()
    plt.savefig("band_accuracy_frequency.png", dpi=300)
    plt.show()

    # Spatial task
    band_acc_spatial = _build_band_matrix(all_labels_spatial, all_preds_spatial)
    plt.figure(figsize=(5, 4))
    sns.heatmap(band_acc_spatial, annot=True, fmt=".2f", cmap="Blues",
                cbar_kws={"shrink": 0.8})
    plt.xlabel("Predicted position"); plt.ylabel("True band")
    plt.title("Band relocation accuracy – Spatial")
    plt.tight_layout()
    plt.savefig("band_accuracy_spatial.png", dpi=300)
    plt.show()
    # =========================================================

    print("\nVisualization files saved:")
    print("- per_class_metrics_comparison.png")
    print("- model_performance_summary.png")
    print("- band_accuracy_frequency.png")
    print("- band_accuracy_spatial.png")

    # 6. Model Performance Summary Comparison
    metrics_summary = {
        'Frequency Accuracy': accuracy_freq,
        'Frequency Macro F1': f1_macro_freq,
        'Frequency Weighted F1': f1_weighted_freq,
        'Spatial Accuracy': accuracy_spatial,
        'Spatial Macro F1': f1_macro_spatial,
        'Spatial Weighted F1': f1_weighted_spatial
    }
    
    fig, ax = plt.subplots(figsize=(10, 4))
    metrics_names = list(metrics_summary.keys())
    metrics_values = list(metrics_summary.values())
    
    colors = ['steelblue' if 'Frequency' in name else 'orange' for name in metrics_names]
    sns.barplot(x=metrics_names, y=metrics_values, ax=ax, palette=colors)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, value in enumerate(metrics_values):
        ax.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig('model_performance_summary.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"\nVisualization files saved:")
    print("- confusion_matrix_frequency.png")
    print("- confusion_matrix_spatial.png")
    print("- confusion_matrix_normalized_frequency.png") 
    print("- confusion_matrix_normalized_spatial.png")
    print("- per_class_metrics_comparison.png")
    print("- model_performance_summary.png")

if __name__ == "__main__":
    evaluate() 