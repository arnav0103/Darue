"""
Reproducibility utilities for setting random seeds.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (may be slower)
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    if deterministic:
        # --- FIX START: Set CuBLAS workspace for deterministic behavior ---
        # This is required for torch.use_deterministic_algorithms(True) with CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # --- FIX END ---

        # Make PyTorch operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # For PyTorch >= 1.8, use torch.use_deterministic_algorithms
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            # Older PyTorch version
            pass


__all__ = ['set_seed']