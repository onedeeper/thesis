import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
from eeglearn.config import Config
from eeglearn.models.models import VanillaGraphModel
from eeglearn.utils.models import setup_label_encoder
from eeglearn.utils.models import create_graph_loaders

def evaluate():
    """
    Load a trained model, evaluate it on the test set, and generate comprehensive performance metrics and visualizations.
    
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
    gcn_out_size = Config.gcn_out_size
    linear_size = Config.linear_size
    K = Config.K
    stop_at = Config.stop_at
    
    # Model parameters
    model = VanillaGraphModel(
        inchannel=5, gcn_out_size=gcn_out_size, batch=batch_size, K=K,
        linear_size=linear_size, drop_rate=drop_rate, testmode=True,
        HF=120, HS=128, HC=n_classes
    ).to(device)

    # Load model weights
    model_path = "/Users/udeshhabaraduwa/thesis _local/thesis/eeg-graph-learning/data/weights/best_models/all_data/all_data_best_model_jointly.pkl"
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
    test_graphs_path = "/Users/udeshhabaraduwa/thesis _local/thesis/eeg-graph-learning/data/graph_list/all_data_class_weighting_jointly_test_original_graph_list.pt"
    test_graphs = torch.load(test_graphs_path)
    test_loader = create_graph_loaders(data_split_type="test",
                                       batch_size=Config.batch_size,
                                       graph_lists={"original" : test_graphs},
                                       encoder=encoder,
                                       perm_types=[None])
    print(f"Loaded test data from {test_graphs_path}")

    # Check if all classes are present in test set
    test_labels = []
    for data in test_loader['original']:
        test_labels.extend(encoder.inverse_transform(data.y))
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
        for data in test_loader['original']:
            #print(data.y)
            labels = encoder.inverse_transform(data.y)
            label_counts = {}
            for label in labels:
                label_counts[str(label)] = label_counts.get(label, 0) + 1
            # print(label_counts)
            data = data.to(device)
            out = model(data) 
            loss = criterion(out, data.y)
            total_loss += loss.item()
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    avg_loss = total_loss / len(test_loader['original'])
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
    print("MODEL EVALUATION RESULTS")
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
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight', 
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
    plt.savefig('confusion_matrix_normalized.png', dpi=300, bbox_inches='tight',
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
    plt.savefig('per_class_metrics.png', dpi=300, bbox_inches='tight',
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
    plt.savefig('test_set_distribution.png', dpi=300, bbox_inches='tight',
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
    plt.savefig('model_performance_summary.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"\nVisualization files saved:")
    print("- confusion_matrix.png")
    print("- confusion_matrix_normalized.png") 
    print("- per_class_metrics.png")
    print("- test_set_distribution.png")
    print("- model_performance_summary.png")

if __name__ == "__main__":
    evaluate() 