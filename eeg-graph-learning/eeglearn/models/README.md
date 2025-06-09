# Models Module

This module contains implementations of graph neural network models for EEG classification and self-supervised learning. The models are designed to work with graph representations of EEG data for psychiatric condition classification.

## Available Models

- **Vanilla Model**: Standard graph convolution model for supervised classification
- **Jointly Trained Model**: Multi-task learning with frequency, spatial, and classification heads
- **Self-Supervised Model**: Pre-training model using frequency and spatial permutation tasks
- **EEGNet Baseline**: CNN-based baseline model for comparison

## Quick Start: Training the Vanilla Model

This guide walks you through training and evaluating the vanilla model using the provided example configuration.

### 1. Environment Setup

First, ensure your environment is properly set up:

```bash
# Activate the conda environment
conda activate eeg-graph-learning

# Navigate to the project root
cd eeg-graph-learning
```

### 2. Configuration Setup

The model behavior is controlled through the `Config` class in `eeglearn/config.py`. Here are the key parameters for the vanilla model:

#### Essential Configuration Parameters

```python
# Training hyperparameters
epochs = 100                    # Number of training epochs
batch_size = 5                  # Batch size for training
lr = 0.0016898579982266685     # Learning rate
weight_decay = 0.0005931173538055033  # L2 regularization
drop_rate = 0.1360323026782416  # Dropout rate
stop_at = 10                    # Early stopping patience

# Model architecture
gcn_out_size = 64              # GCN output features
linear_size = 512              # Linear layer size
K = 1                          # Chebyshev polynomial order

# Data settings
testing_on_sample_data = True  # Use small sample for testing
use_class_weighting = False    # Enable class balancing
main_classes = ["ADHD", "HEALTHY", "MDD", "OCD", "SMC"]  # Target classes
```

#### Customizing Configuration

To modify the configuration for your experiment:

1. **Edit `eeglearn/config.py`** directly, or
2. **Set experiment name** to track different runs:
   ```python
   Config.experiment_name = "my_vanilla_experiment"
   ```

3. **Adjust for your dataset size**:
   ```python
   # For full dataset
   Config.testing_on_sample_data = False
   Config.use_stratify = True
   
   # For quick testing
   Config.testing_on_sample_data = True
   Config.use_stratify = False
   ```

### 3. Running Training

#### Option A: Direct Python Execution (Recommended)

```bash
# Navigate to the models directory
cd eeglearn/models

# Run vanilla training
python train_vanilla.py
```

#### Option B: Import and Run Programmatically

```python
from eeglearn.models.train_vanilla import train_with_kfold_cv
from eeglearn.config import Config

# Optional: modify config before training
Config.epochs = 50
Config.experiment_name = "vanilla_test"

# Run 5-fold cross-validation training
results = train_with_kfold_cv(k_folds=5)
print(f"Average validation F1: {results['best_val_f1_macro_mean']:.4f}")
```

### 4. Understanding Training Output

During training, you'll see output like this:

```
🚀 Using GPU: NVIDIA GeForce RTX 4090
📱 Device: cuda
⚠️  Participants split:
n train: 120
n valid: 30
n test: 30

🔄  Building graphs.
📊 Graph Loader Information:
  • Training loaders:
    - original: 24 batches
  • Validation loader:
    - original: 6 batches
  • Test loader:
    - original: 6 batches

⚠️  Training for epochs: 100
Fold 1, Epoch [0/100] - Train Acc: 0.2500, Val Acc: 0.3333, Train F1: 0.2000, Val F1: 0.2500
```

### 5. Training Outputs

The training process creates several output files in `data/weights/vanilla/` and `data/metrics/vanilla/`:

#### Model Weights
- `{experiment_name}_vanilla_best_model_*.pt` - Best model checkpoints
- Model weights are saved when validation performance improves

#### Metrics Files
- `cv_5fold_summary_vanilla_{timestamp}.csv` - Cross-validation summary
- `cv_5fold_detailed_results_vanilla_{timestamp}.csv` - Per-fold detailed results  
- `cv_5fold_training_history_vanilla_{timestamp}.csv` - Complete training history
- `cv_5fold_model_config_vanilla_{timestamp}.json` - Model configuration used

### 6. Evaluating Model Performance

#### Option A: Evaluate Saved Model

```bash
# Run evaluation on test set
python evaluate_vanilla_model.py
```

#### Option B: Evaluate Programmatically

```python
from eeglearn.models.evaluate_vanilla_model import evaluate

# This will load the best saved model and evaluate on test set
evaluate()
```

#### Evaluation Outputs

The evaluation generates:

1. **Console Output**: Comprehensive metrics including:
   ```
   ====================================================
   MODEL EVALUATION RESULTS
   ====================================================
   Test Loss: 1.4523
   Accuracy: 0.7333 (73.33%)
   Macro F1-Score: 0.7104
   Micro F1-Score: 0.7333
   Weighted F1-Score: 0.7284
   ```

2. **Visualizations**:
   - `confusion_matrix.png` - Confusion matrix
   - `confusion_matrix_normalized.png` - Normalized confusion matrix
   - `per_class_metrics.png` - Per-class performance charts

3. **Per-Class Metrics**:
   ```
   Class           Precision  Recall     F1-Score   Support   
   ADHD            0.8000     0.6667     0.7273     15        
   HEALTHY         0.7500     0.8571     0.8000     14        
   MDD             0.6667     0.7500     0.7059     12        
   ```

### 7. Model Architecture Details

The Vanilla model (`VanillaGraphModel`) consists of:

1. **Graph Convolution Layer**: ChebConv with configurable K parameter
2. **Fully Connected Layers**: 
   - Linear(gcn_out_size * 26, linear_size)
   - Linear(linear_size, linear_size // 2)  
   - Linear(linear_size // 2, n_classes)
3. **Regularization**: Batch normalization, ReLU activation, dropout

#### Input Requirements
- **Node features**: 5 features per EEG channel (26 channels total)
- **Graph structure**: Adjacency matrix based on EEG electrode positions
- **Batch size**: Configurable (default: 5)

### 8. Troubleshooting

#### Common Issues

**CUDA Out of Memory**:
```python
Config.batch_size = 3  # Reduce batch size
Config.num_workers = 4  # Reduce data loader workers
```

**Poor Performance**:
- Check class distribution in your data
- Enable class weighting: `Config.use_class_weighting = True`
- Adjust learning rate: `Config.lr = 0.001`
- Increase model capacity: `Config.linear_size = 1024`

**Data Loading Errors**:
- Ensure data preprocessing is complete
- Check data paths in `Config.data_path`
- Verify participant metadata file exists

#### Debug Mode

For detailed debugging, modify the config:

```python
Config.testing_on_sample_data = True  # Use smaller dataset
Config.epochs = 5                     # Fewer epochs for testing
Config.batch_size = 2                 # Smaller batches
```

### 9. Hyperparameter Tuning

For automated hyperparameter optimization:

```bash
# Run Optuna tuning for vanilla model
python optuna_tuning_vanilla.py
```

This will automatically search for optimal:
- Learning rate
- Weight decay  
- Dropout rate
- Model architecture parameters

### 10. Advanced Usage

#### Custom Data Splits

```python
# Save a custom data split for reproducibility
from eeglearn.utils.models import split_data
import torch

custom_split = split_data()
torch.save(custom_split, "data/my_custom_split.pt")

# Use the custom split
Config.load_data_split_from = "my_custom_split.pt"
```

#### Multi-GPU Training

```python
# The model automatically detects and uses available GPUs
# For specific GPU selection:
import torch
torch.cuda.set_device(0)  # Use GPU 0
```

## Other Models

### Jointly Trained Model
```bash
python train_jointly.py       # Training
python evaluate_jointly_model.py  # Evaluation
```

### Self-Supervised Model  
```bash
python train_selfsupervised.py    # Pre-training
python evaluate_self_supervised.py # Evaluation
```

### EEGNet Baseline
```bash
python train_eegnet_baseline.py     # Training
python evaluate_eegnet_baseline.py  # Evaluation
```

## Model Comparison

| Model | Purpose | Input | Output |
|-------|---------|-------|--------|
| Vanilla | Standard classification | Original graphs | Class predictions |
| Jointly | Multi-task learning | Freq + Spatial + Original | 3 prediction heads |
| Self-Supervised | Pre-training | Freq + Spatial graphs | Permutation predictions |
| EEGNet | CNN baseline | Raw EEG timeseries | Class predictions |

For more details about the data preprocessing pipeline and graph construction, see the main project README. 