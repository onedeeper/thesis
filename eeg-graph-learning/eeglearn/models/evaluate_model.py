import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
from eeglearn.config import Config
from eeglearn.models.models import JointlyTrainModel
from eeglearn.utils.models import setup_label_encoder
from eeglearn.utils.models import create_graph_loaders
def evaluate():
    """Load a trained model and evaluate it on the test set."""
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
    model = JointlyTrainModel(
        inchannel=5, gcn_out_size=gcn_out_size, batch=batch_size, K=K,
        linear_size=linear_size, drop_rate=drop_rate, testmode=True,
        HF=120, HS=128, HC=n_classes
    ).to(device)

    # Load model weights
    model_path = "/Users/udeshhabaraduwa/thesis _local/thesis/eeg-graph-learning/data/weights/jointly/Acc_0.219_f1_0.385_checkpoint.pkl"
    checkpoint = torch.load(model_path, map_location=device)
    
    # Try to load state dict
    for key in ['model', 'model_state_dict', 'state_dict']:
        if key in checkpoint:
            model.load_state_dict(checkpoint[key])
            break
    else:  # If no keys found, assume checkpoint is the state dict
        model.load_state_dict(checkpoint)
                           
    print(f"Successfully loaded model weights from {model_path}")
    model.eval()

    # Load test data
    test_graphs_path = "/Users/udeshhabaraduwa/thesis _local/thesis/eeg-graph-learning/eeglearn/models/test_original_graph_list.pt"
    test_graphs = torch.load(test_graphs_path)
    test_loader = create_graph_loaders(data_split_type="test",
                                       batch_size=Config.batch_size,
                                       graph_lists={"original" : test_graphs},
                                       encoder=encoder,
                                       perm_types=[None])
    print(f"Loaded test data from {test_graphs_path}")

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
            print(label_counts)
            data = data.to(device)
            out = model(data) 
            loss = criterion(out, data.y)
            total_loss += loss.item()
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Decode labels back to original classes
    decoded_true = encoder.inverse_transform(all_labels)
    decoded_pred = encoder.inverse_transform(all_preds)
    
    # Create confusion matrix
    cm = confusion_matrix(decoded_true, decoded_pred)
    plt.figure(figsize=(10, 8))
    class_names = list(encoder.classes_)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("\n--- Test Set Evaluation Results ---")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print("Confusion matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    evaluate() 