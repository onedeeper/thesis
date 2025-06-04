"""
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
    """Objective function for Optuna hyperparameter optimization of EEGNet.
    
    Args:
        trial (optuna.Trial): Optuna trial object for sampling hyperparameters
        
    Returns:
        float: Validation metric (e.g., macro F1 score) to maximize
    """
    # Experiment settings
    Config.experiment_name = f"all_data_eegnet_optuna_tuning_{timestamp}_pid{pid}"
    Config.optuna = True
    Config.load_data_split_from = "all_data_split_0.8.train_test_valid_split.pt" 

    # Reproducibility settings
    Config.RANDOM_SEED = 42
    Config.DETERMINISTIC = True

    Config.epochs = 30  
    Config.stop_at = 5

    # Data selection 
    Config.use_class_weighting = True
    Config.p_train = 0.8
    Config.sample_proportion_of_data = 1.0
    Config.use_tuur_smolder_data = False 
    Config.drop_last = True # For dataloader
    Config.skip_bads = True # For data preprocessing
    Config.main_classes = ["ADHD", "HEALTHY", "MDD", "OCD", "SMC"] 
    Config.use_stratify = True # For data splitting
    Config.testing_on_sample_data = True # For quick tests, set to False for full runs
    
    # Tunable EEGNet parameters
    Config.kernel_length = trial.suggest_categorical("eegnet_kernel_length", [32, 64])
    Config.drop_rate = trial.suggest_categorical("eegnet_dropout_rate", [0.25, 0.5]) 
    # Tunable training parameters
    Config.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    Config.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True) 
    Config.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128]) 
    

    trainer = importlib.import_module("eeglearn.models.train_eegnet_baseline")
    val_metric = trainer.train(trial=trial)
    return val_metric

 
if __name__ == "__main__":
    results_filename = f"all_data_optuna_eegnet_results_{timestamp}_pid{pid}.csv"
    
    # Create pruner to terminate unpromising trials early
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, n_min_trials=3)
    
    # Create study for maximizing validation performance
    study = optuna.create_study(direction="maximize", pruner=pruner)
    
    # Run optimization (adjust n_trials and timeout as needed)
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
