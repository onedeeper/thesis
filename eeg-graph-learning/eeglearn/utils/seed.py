import random
import numpy as np
import torch
import os
import sklearn
import optuna

def set_seed(seed=42, deterministic=True):
    """
    Set random seeds for reproducibility across multiple libraries
    
    Parameters:
    -----------
    seed : int
        The random seed to set (default: 42)
    deterministic : bool
        Whether to set PyTorch to use deterministic algorithms
        May impact performance, but ensures reproducibility
    """
    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Make PyTorch operations deterministic
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for any other libraries that check it
    # (like TensorFlow, though it's not used in your project)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For scikit-learn (if using random operations)
    try:
        import sklearn
        sklearn.utils.check_random_state(seed)
    except ImportError:
        pass
    
    # For Optuna (will only affect new studies)
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        pass
        
    print(f"✅ Random seed set to {seed}") 