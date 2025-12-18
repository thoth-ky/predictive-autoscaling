"""
Data Loaders for Time Series Training
PyTorch Dataset and DataLoader for multi-horizon time series prediction.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional, Tuple


class TimeSeriesDataset(Dataset):
    """
    Dataset for multi-horizon time series prediction.

    Handles input sequences (X) and multiple target horizons (y_dict).
    """

    def __init__(self, X: np.ndarray, y_dict: Dict[int, np.ndarray],
                 metadata: Optional[np.ndarray] = None):
        """
        Initialize dataset.

        Args:
            X: Input sequences of shape (n_samples, window_size, n_features)
            y_dict: Dictionary mapping horizon to target arrays
                   {20: (n_samples, 20), 60: (n_samples, 60), 120: (n_samples, 120)}
            metadata: Optional metadata for each sample
        """
        self.X = torch.FloatTensor(X)
        self.y_dict = {h: torch.FloatTensor(y) for h, y in y_dict.items()}
        self.metadata = metadata
        self.horizons = sorted(y_dict.keys())

        # Validate shapes
        n_samples = len(X)
        for horizon, y in self.y_dict.items():
            assert len(y) == n_samples, f"Mismatch in samples for horizon {horizon}"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            Tuple of (X, y_dict) where y_dict has targets for all horizons
        """
        x = self.X[idx]
        y = {h: self.y_dict[h][idx] for h in self.horizons}

        return x, y


class SingleHorizonDataset(Dataset):
    """
    Simplified dataset for single-horizon prediction.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.

        Args:
            X: Input sequences of shape (n_samples, window_size, n_features)
            y: Target sequences of shape (n_samples, horizon)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_data_loaders(X_train: np.ndarray, y_train_dict: Dict[int, np.ndarray],
                        X_val: np.ndarray, y_val_dict: Dict[int, np.ndarray],
                        batch_size: int = 32, num_workers: int = 4,
                        shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.

    Args:
        X_train: Training input sequences
        y_train_dict: Training targets per horizon
        X_val: Validation input sequences
        y_val_dict: Validation targets per horizon
        batch_size: Batch size for training
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = TimeSeriesDataset(X_train, y_train_dict)
    val_dataset = TimeSeriesDataset(X_val, y_val_dict)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == '__main__':
    # Test data loaders
    print("Time Series Data Loaders")
    print("=" * 60)

    # Create dummy data
    n_samples = 1000
    window_size = 240
    n_features = 8
    horizons = [20, 60, 120]

    X = np.random.randn(n_samples, window_size, n_features)
    y_dict = {h: np.random.randn(n_samples, h) for h in horizons}

    print(f"Creating dataset with:")
    print(f"  Samples: {n_samples}")
    print(f"  Window size: {window_size}")
    print(f"  Features: {n_features}")
    print(f"  Horizons: {horizons}")

    # Create dataset
    dataset = TimeSeriesDataset(X, y_dict)

    print(f"\nDataset length: {len(dataset)}")

    # Test getting a sample
    x, y = dataset[0]
    print(f"\nSample 0:")
    print(f"  X shape: {x.shape}")
    for h in horizons:
        print(f"  y[{h}] shape: {y[h].shape}")

    # Create data loader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"\nData loader:")
    print(f"  Batches: {len(loader)}")

    # Test one batch
    for batch_x, batch_y in loader:
        print(f"\nFirst batch:")
        print(f"  X shape: {batch_x.shape}")
        for h in horizons:
            print(f"  y[{h}] shape: {batch_y[h].shape}")
        break

    print("\nData loaders working correctly!")
