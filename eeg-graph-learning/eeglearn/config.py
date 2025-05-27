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
    # Random seed settings
    RANDOM_SEED = 42
    DETERMINISTIC = True
    
    # Other global configuration settings can be added here
    epochs = 100
    batch_size = 256 
    lr = 0.0001 # original : 0.01
    weight_decay = 8e-5
    drop_rate = 0.25 
    num_workers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps'\
                           if torch.backends.mps.is_available() else 'cpu')
    num_jigsaw = 4
    K = 2
    project_root : Path = Path(__file__).resolve().parent.parent
    cleaned_data_path : Path = project_root / 'data' / 'cleaned'
    energy_path : Path = project_root / 'data' / 'energy'
    model_weights_dir : Path = project_root / 'data' / 'weights'
    metrics_dir : Path = project_root / 'data' / 'metrics'
    data_path : Path  = project_root / 'data'
    drop_last : bool = True
    stop_at : int = 10
    skip_bads : bool = False # If examples with bad channels should be skipped.
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