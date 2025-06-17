# Models Module

This module contains implementations of graph neural network models for EEG classification and self-supervised learning. The models are designed to work with graph representations of EEG data for psychiatric condition classification.

## Available Models

- **Vanilla Model**: Standard graph convolution model for supervised classification
- **Jointly Trained Model**: Multi-task learning with frequency, spatial, and classification heads
- **Self-Supervised Model**: Pre-training model using frequency and spatial permutation tasks
- **Fine-tuned Model**: Fine-tuning pre-trained models on downstream tasks
- **EEGNet Baseline**: CNN-based baseline model for comparison

## Training Models

### Quick Start

The fastest way to train a model is to use one of the pre-configured setups:

```bash
# 1. Navigate to the configs directory
cd configs/config_files/

# 2. Copy the desired configuration
all_data_vanilla_experiment_config.py ../../config.py

Replace the eeglearn/config.py with the desired model config.

# 3. Run training
cd ../../
python train_vanilla.py

# 4. Evaluate results
python evaluate_vanilla_model.py
```

### Available Pre-configured Experiments

#### Full Dataset Configurations
- `all_data_vanilla_experiment_config.py` - Vanilla graph model
- `all_data_self_supervised_experiment_config.py` - Self-supervised pre-training
- `all_data_jointly_experiment_config.py` - Joint training with multiple tasks
- `all_data_fine_tune_experiment_config.py` - Fine-tuning from pre-trained model
- `all_data_baseline_experiment_config.py` - EEGNet baseline

#### Tuur's Dataset Configurations
- `tuur_data_vanilla_experiment_config.py` - Vanilla model on subset
- `tuur_data_self_supervised_experiment_config.py` - Self-supervised on subset
- `tuur_data_jointly_experiment_config.py` - Joint training on subset
- `tuur_data_fine_tune_experiment_config.py` - Fine-tuning on subset
- `tuur_data_baseline_experiment_config.py` - EEGNet baseline on subset

### Training Commands

After copying the appropriate configuration file:

```bash
# For vanilla model
python train_vanilla.py

# For self-supervised pre-training
python train_selfsupervised.py

# For jointly trained model
python train_jointly.py

# For fine-tuning from pre-trained model
python train_finetune_from_ssl.py

# For EEGNet baseline
python train_eegnet_baseline.py
```

### Configuration File Structure

Each configuration file contains optimized hyperparameters and settings:

- **Experiment settings**: Name, reproducibility, data splits
- **Training hyperparameters**: Learning rate, batch size, epochs, weight decay
- **Model architecture**: Layer sizes, dropout rates, Chebyshev order
- **Data selection**: Classes to include, preprocessing options
- **Hardware settings**: Device selection, number of workers

### Cross-Validation Training

Most models support k-fold cross-validation:

```python
from eeglearn.models.train_vanilla import train_with_kfold_cv

# Run 5-fold cross-validation
results = train_with_kfold_cv(k_folds=5)
print(f"Average F1: {results['best_val_f1_macro_mean']:.4f}")
```

### Understanding Training Output

During training, you'll see progress information:

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

### Training Outputs

Training generates several output files:

#### Model Weights (`data/weights/`)
- `{experiment_name}_best_model_*.pt` - Best model checkpoints
- Organized by model type and cross-validation folds

#### Metrics (`data/metrics/`)
- `cv_5fold_summary_{model}_{timestamp}.csv` - Cross-validation summary
- `cv_5fold_detailed_results_{model}_{timestamp}.csv` - Per-fold results
- `cv_5fold_training_history_{model}_{timestamp}.csv` - Training history
- `cv_5fold_model_config_{model}_{timestamp}.json` - Model configuration

### Evaluating Models

After training, evaluate performance:

```bash
# For vanilla model
python evaluate_vanilla_model.py

# For self-supervised model
python evaluate_self_supervised.py

# For jointly trained model
python evaluate_jointly_model.py

# For fine-tuned model
python evaluate_fine_tuning.py

# For EEGNet baseline
python evaluate_eegnet_baseline.py
```

### Evaluation Outputs

Evaluation generates comprehensive results:

1. **Console Output**:
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

## Model Architecture Details

### Vanilla Graph Model

The vanilla model consists of:
1. **Graph Convolution Layer**: ChebConv with configurable K parameter  
2. **Fully Connected Layers**:
   - Linear(gcn_out_size * 26, linear_size)
   - Linear(linear_size, linear_size // 2)
   - Linear(linear_size // 2, n_classes)
3. **Regularization**: Batch normalization, ReLU activation, dropout

### Self-Supervised Model

Pre-training tasks:
- **Frequency permutation**: Predicting frequency band order
- **Spatial permutation**: Predicting electrode permutation

### Jointly Trained Model

Multi-task learning with:
- **Classification head**: Main psychiatric condition prediction
- **Frequency head**: Frequency permutation prediction  
- **Spatial head**: Spatial permutation prediction

### EEGNet Baseline

CNN architecture specifically designed for EEG:
- **Temporal convolution**: Captures frequency information
- **Spatial convolution**: Models spatial relationships
- **Separable convolution**: Reduces parameters

## Input Requirements

All graph models expect:
- **Node features**: 5 features per EEG channel (26 channels total)
- **Graph structure**: Adjacency matrix based on electrode positions
- **Batch processing**: Configurable batch sizes

EEGNet expects:
- **Raw EEG**: Time series data (channels × timepoints)
- **Preprocessing**: Filtered and epoched data

## Troubleshooting

### Common Issues

**CUDA Out of Memory**:
- Reduce batch size in configuration file
- Reduce number of data loader workers

**Poor Performance**:
- Try different configuration files
- Check class distribution in data
- Consider using class weighting configurations

**Data Loading Errors**:
- Ensure data preprocessing is complete
- Check data paths in configuration
- Verify all required files exist

### Hyperparameter Tuning

For automated optimization:

```bash
# Run Optuna tuning
python optuna_tuning_vanilla.py      # Vanilla model
python optuna_tuning_jointly.py      # Joint model
python optuna_tuning_self_supervised.py  # Self-supervised
python optuna_tuning_eegnet.py       # EEGNet baseline
python optuna_tuning_fine_tuning.py  # Fine-tuning
```

## Model Comparison

| Model | Purpose | Training Script | Evaluation Script |
|-------|---------|----------------|------------------|
| Vanilla | Standard classification | `train_vanilla.py` | `evaluate_vanilla_model.py` |
| Self-Supervised | Pre-training | `train_selfsupervised.py` | `evaluate_self_supervised.py` |
| Jointly | Multi-task learning | `train_jointly.py` | `evaluate_jointly_model.py` |
| Fine-tuned | Transfer learning | `train_finetune_from_ssl.py` | `evaluate_fine_tuning.py` |
| EEGNet | CNN baseline | `train_eegnet_baseline.py` | `evaluate_eegnet_baseline.py` |

For more details about data preprocessing and graph construction, see the main project documentation. 