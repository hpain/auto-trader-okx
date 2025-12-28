import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import logging

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    # Fallback for doc/linting if torch not installed yet
    torch = None
    Dataset = object
    DataLoader = object

class LazyTimeSeriesDataset(Dataset):
    """
    Memory-Efficient PyTorch Dataset.
    Does NOT pre-calculate windows. Slices on-the-fly.
    RAM Usage: 1x Raw Data (vs 60x for naive windowing).
    """
    def __init__(self, data: np.ndarray, target: np.ndarray, window_size: int = 60):
        # Store only raw data
        # Ensure data is simple float32 numpy array
        self.data = torch.tensor(data, dtype=torch.float32) if torch else data
        self.target = torch.tensor(target, dtype=torch.float32) if torch else target
        self.window_size = window_size
        
        # Calculate valid length
        self.length = len(data) - window_size

    def __len__(self):
        # Helper to avoid negative length
        return max(0, self.length)

    def __getitem__(self, idx):
        # Slice window on-the-fly
        # Input: [idx : idx + window_size]
        # Target: [idx + window_size] (Next step prediction)
        X_window = self.data[idx : idx + self.window_size]
        y_val = self.target[idx + self.window_size]
        
        return X_window, y_val

def create_lazy_loader(data: pd.DataFrame, target: pd.Series, window_size: int = 60, batch_size: int = 32, shuffle: bool = True):
    """
    Creates a DataLoader using the Lazy Dataset.
    """
    if not torch: return None
    
    # Convert to pure numpy/tensor once
    X = data.values
    y = target.values
    
    dataset = LazyTimeSeriesDataset(X, y, window_size)
    
    # Check if dataset is valid
    if len(dataset) <= 0:
        return None
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
