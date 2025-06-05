"""
Configuration settings for the eeglearn package.

This module provides a central configuration system for the entire package,
ensuring consistent settings across all components.

Created on: March 2025
Author: Udesh Habaraduwa
"""
import torch
from pathlib import Path
class Config:
    """
    Central configuration class for eeglearn.
    
    This class serves as a single source of truth for configuration settings,
    particularly focused on reproducibility settings for scientific research.
    """

    # Development settings
    experiment_name = "test"
    optuna :bool = False
    load_data_split_from  = ""
    # Reproducibility settings
    RANDOM_SEED = 42
    DETERMINISTIC = True

    # Training hyperparameters
    epochs = 100
    batch_size = 5
    lr = 0.0016898579982266685
    weight_decay = 0.0005931173538055033
    drop_rate = 0.1360323026782416
    stop_at = 10

    # Data selection
    testing_on_sample_data = True
    use_class_weighting = False
    p_train = 0.4
    sample_proportion_of_data = 1.0
    use_tuur_smolder_data = False
    drop_last = True
    skip_bads = True
    main_classes : list[str] = ["ADHD", "HEALTHY", "MDD", "OCD", "SMC"]
    use_stratify = True
    if testing_on_sample_data:
        use_stratify = False
        main_classes : list[str] = ["ADHD","MDD", "SMC", "OCD"]

    # GCN / fully connected Model architecture parameters
    gcn_out_size = 64
    linear_size =  512
    K = 1

    ## EEG net 
    eeg_net_n_time_steps = 45253 # smallest length raw eeg found in the dataset.
    n_eeg_channels = 26
    kernel_length = 64
    
    # for fine tuning.
    pretrained_weights_path = "/mnt/disk2/thesis/eeg-graph-learning/data/weights/self_supervised/tuur_data/tuur_data_self_supervised_best_model_val_loss_1.6647_epoch_25.pt"
    pretrained_gcn_out_size = 64  # Example: Must match your actual SSL model
    pretrained_k = 1              # Example: Must match your actual SSL model
    pretrained_linear_size = 512  # Example: Must match your actual SSL model's HF/HS linear_size
    pretrained_drop_rate = 0.20306923446428116    # Example: Must match your actual SSL model's HF/HS drop_rate
    
    num_workers = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps'\
                           if torch.backends.mps.is_available() else 'cpu')
    

    # Path configurations
    project_root : Path = Path(__file__).resolve().parent.parent
    data_path : Path = project_root / 'data'
    cleaned_data_path : Path = data_path / 'cleaned'
    energy_path : Path = data_path / 'energy'
    model_weights_dir : Path = data_path / 'weights'
    metrics_dir : Path = data_path / 'metrics'    

    # classes from :
    # https://www.frontiersin.org/journals/aging-neuroscience/articles/10.3389/fnagi.2022.1019869/full#supplementary-material
    # main_classes : list[str] = [
    #     "MDD", "UNKNOWN", "ADHD", "SMC", "OCD", "HEALTHY", "INSOMNIA",
    #     "TINNITUS", "PARKINSON", "Dyslexia", "CHRONIC PAIN", "BURNOUT"
    # ]
    @classmethod
    def set_global_seed(cls, verbose=False):
        """
        Set random seed across all libraries from a single source of truth.
        
        This method centralizes the seed setting process to ensure consistent
        reproducibility across the entire codebase.
        
        Args:
            verbose (bool): Whether to print a message when setting the seed.
                            Set to True only in the main process, not in worker processes.
                            Default: False
        
        Returns:
            int: The random seed that was set
        """
        from eeglearn.utils.seed import set_seed
        set_seed(cls.RANDOM_SEED, cls.DETERMINISTIC, verbose=verbose)
        return cls.RANDOM_SEED 