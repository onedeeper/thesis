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
    experiment_name = "all_data_split"
    optuna :bool = False
    load_data_split_from  = ""
    # Reproducibility settings
    RANDOM_SEED = 42
    DETERMINISTIC = True

    # Training hyperparameters
    epochs = 1
    batch_size = 32 
    lr = 0.001  # original : 0.01
    weight_decay = 8e-5
    drop_rate = 0.25 
    stop_at = 10

    # Data selection
    testing_on_sample_data = False
    use_sampler_for_data_loading = True
    p_train = 0.8
    sample_proportion_of_data = 1.0
    use_tuur_smolder_data = False
    drop_last = True
    skip_bads = True
    main_classes : list[str] = ["ADHD", "HEALTHY", "MDD", "OCD", "SMC"]
    use_stratify = True
    if testing_on_sample_data:
        use_stratify = False
        main_classes : list[str] = ["ADHD","MDD", "SMC", "OCD"]
    # Model architecture parameters
    gcn_out_size = 32
    linear_size = 512
    K = 2  # Order of Chebyshev polynomials

    # Hardware and processing settings
    num_workers = 6
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