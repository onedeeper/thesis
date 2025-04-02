"""
Main package initialization for eeglearn.

This module initializes the eeglearn package and sets up global configuration.
"""

from eeglearn.config import Config

# Set the global random seed for reproducibility
Config.set_global_seed()
