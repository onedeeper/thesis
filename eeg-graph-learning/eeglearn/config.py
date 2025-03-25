"""
Configuration settings for the eeglearn package.

This module provides a central configuration system for the entire package,
ensuring consistent settings across all components.

Created on: March 2025
Author: Udesh Habaraduwa
"""

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
    
    @classmethod
    def set_global_seed(cls):
        """
        Set random seed across all libraries from a single source of truth.
        
        This method centralizes the seed setting process to ensure consistent
        reproducibility across the entire codebase.
        
        Returns:
            int: The random seed that was set
        """
        from eeglearn.utils.seed import set_seed
        set_seed(cls.RANDOM_SEED, cls.DETERMINISTIC)
        return cls.RANDOM_SEED 