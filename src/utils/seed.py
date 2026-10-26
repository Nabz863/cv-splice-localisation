"""Seed every random number generator we use, so runs are repeatable."""
import os, random
import numpy as np

def set_seed(seed: int = 0):
    random.seed(seed)                                  # Python's built-in RNG
    np.random.seed(seed)                               # NumPy's global RNG
    os.environ["PYTHONHASHSEED"] = str(seed)           # fixes hash order (e.g. set iteration)
    try:
        import torch
        torch.manual_seed(seed)                        # PyTorch CPU RNG (weight init, dropout)
        torch.cuda.manual_seed_all(seed)               # PyTorch RNG on every GPU
        torch.backends.cudnn.deterministic = True      # use deterministic GPU kernels
        torch.backends.cudnn.benchmark = False         # don't auto-pick kernels (can be non-deterministic)
    except ImportError:                                # CPU-only scripts may not have torch installed
        pass
