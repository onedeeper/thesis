"""Hyperparameter optimization for fine-tuning classification head using Optuna.

This module provides automated hyperparameter tuning for the classification head
after loading pre-trained self-supervised weights. It optimizes hyperparameters
that affect only the classification head training, while keeping the encoder frozen.

The script optimizes the following hyperparameters:
    - batch_size: Training batch size [128, 256]
    - lr: Learning rate (log-uniform distribution: 1e-4 to 1e-2)
    - weight_decay: L2 regularization strength (log-uniform: 1e-6 to 1e-3)
    - drop_rate: Dropout rate (uniform: 0.1 to 0.4)
    - linear_size: Linear layer size [256, 512, 1024]

Usage:
    Run the script directly to start hyperparameter optimization:
    ```
    python optuna_tuning_fine_tuning.py
    ```

Author: Udesh Habaraduwa
Created: 2025

WRITTEN WITH AI
REVIEWED AND VERIFIED BY AUTHOR
"""

# optuna_runner.py
import optuna, importlib
import torch
import os
from pathlib import Path
from eeglearn.config import Config
from datetime import datetime

# Make sure torch / cudnn behaves the same
Config.set_global_seed(verbose=False)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pid = os.getpid()

def objective(trial):
    """Objective function for Optuna hyperparameter optimization of classification head.
    
    Args:
        trial (optuna.Trial): Optuna trial object for sampling hyperparameters
        
    Returns:
        float: Validation macro F1 score to maximize
    """
    # Development settings
    Config.experiment_name = f"tuur_optuna_finetune_{timestamp}_pid{pid}"
    Config.optuna = True
    Config.load_data_split_from = "turr_all_data_train_test_valid_split.pt"

    # Reproducibility settings
    Config.RANDOM_SEED = 42
    Config.DETERMINISTIC = True

    # Training hyperparameters
    Config.epochs = 15  # Cap epochs for faster hyperparameter search
    Config.stop_at = 5

    # Data selection
    Config.use_class_weighting = False
    Config.p_train = 0.8
    Config.sample_proportion_of_data = 1.0
    Config.use_tuur_smolder_data = True
    Config.drop_last = True
    Config.skip_bads = True
    Config.main_classes = ["ADHD", "HEALTHY", "MDD", "OCD", "SMC"]
    Config.use_stratify = True
    Config.testing_on_sample_data = True
    
    # Fixed architecture parameters for the pre-trained SSL model parts (GCN, HF, HS)
    Config.pretrained_weights_path = "/mnt/disk2/thesis/eeg-graph-learning/data/weights/self_supervised/tuur_data/tuur_data_self_supervised_best_model_val_loss_1.6647_epoch_25.pt"
    Config.pretrained_gcn_out_size = 64   # Example: Must match your actual SSL model
    Config.pretrained_k = 1              # Example: Must match your actual SSL model
    Config.pretrained_linear_size = 512  # Example: Must match your actual SSL model's HF/HS linear_size
    Config.pretrained_drop_rate =   0.20306923446428116  # Example: Must match your actual SSL model's HF/HS drop_rate
    
    # Tunable parameters for classification head (HC)
    Config.linear_size = trial.suggest_categorical("hc_linear_size", [256, 512, 1024]) # Renamed for clarity
    Config.drop_rate = trial.suggest_float("hc_drop_rate", 0.1, 0.4) # Renamed for clarity
    
    # Tunable training parameters for fine-tuning
    Config.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    Config.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    Config.batch_size = trial.suggest_categorical("batch_size", [128, 256])
    
    trainer = importlib.import_module("eeglearn.models.train_finetune_from_ssl")
    val_metric = trainer.train_classification()   
    return val_metric


if __name__ == "__main__":
    results_filename = f"optuna_finetune_results_{timestamp}_pid{pid}.csv"
    
    # Create pruner to terminate unpromising trials early
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    
    # Create study for maximizing validation performance
    study = optuna.create_study(direction="maximize", pruner=pruner)
    
    # Run optimization with 30 trials or 24 hour timeout
    study.optimize(objective, n_trials=30, timeout=60*60*24)

    print("Best trial:", study.best_trial.params)
    print(f"Best value: {study.best_trial.value:.4f}")
    
    # Save detailed results
    wanted = (
        'number', 'state', 'values',
        'params', 'user_attrs',
        'intermediate_values', 'duration',
        'datetime_start', 'datetime_complete'
    )

    df = study.trials_dataframe(attrs=wanted, multi_index=True)
    df.to_csv(results_filename, index=False)
    print(f"Results saved to '{results_filename}'") 
