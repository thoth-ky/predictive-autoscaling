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
    Supports container IDs for multi-container training with embeddings.
    """

    def __init__(
        self,
        X: np.ndarray,
        y_dict: Dict[int, np.ndarray],
        container_ids: Optional[np.ndarray] = None,
        metadata: Optional[np.ndarray] = None,
    ):
        """
        Initialize dataset.

        Args:
            X: Input sequences of shape (n_samples, window_size, n_features)
            y_dict: Dictionary mapping horizon to target arrays
                   {20: (n_samples, 20), 60: (n_samples, 60), 120: (n_samples, 120)}
            container_ids: Optional container IDs for each sample (n_samples,)
                          Used for multi-container training with embeddings
            metadata: Optional metadata for each sample
        """
        self.X = torch.FloatTensor(X)
        self.y_dict = {h: torch.FloatTensor(y) for h, y in y_dict.items()}
        self.container_ids = (
            torch.LongTensor(container_ids) if container_ids is not None else None
        )
        self.metadata = metadata
        self.horizons = sorted(y_dict.keys())

        # Validate shapes
        n_samples = len(X)
        for horizon, y in self.y_dict.items():
            assert len(y) == n_samples, f"Mismatch in samples for horizon {horizon}"

        if self.container_ids is not None:
            assert (
                len(self.container_ids) == n_samples
            ), "container_ids must match number of samples"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            For single-container mode: (X, y_dict)
            For multi-container mode: (X, y_dict, container_id)
        """
        x = self.X[idx]
        y = {h: self.y_dict[h][idx] for h in self.horizons}

        if self.container_ids is not None:
            return x, y, self.container_ids[idx]

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


def create_data_loaders(
    X_train: np.ndarray,
    y_train_dict: Dict[int, np.ndarray],
    X_val: np.ndarray,
    y_val_dict: Dict[int, np.ndarray],
    container_ids_train: Optional[np.ndarray] = None,
    container_ids_val: Optional[np.ndarray] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.

    Args:
        X_train: Training input sequences
        y_train_dict: Training targets per horizon
        X_val: Validation input sequences
        y_val_dict: Validation targets per horizon
        container_ids_train: Optional container IDs for training (for multi-container)
        container_ids_val: Optional container IDs for validation (for multi-container)
        batch_size: Batch size for training
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = TimeSeriesDataset(X_train, y_train_dict, container_ids_train)
    val_dataset = TimeSeriesDataset(X_val, y_val_dict, container_ids_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
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

    print("Creating dataset with:")
    print(f"  Samples: {n_samples}")
    print(f"  Window size: {window_size}")
    print(f"  Features: {n_features}")
    print(f"  Horizons: {horizons}")

    # Create dataset
    dataset = TimeSeriesDataset(X, y_dict)

    print(f"\nDataset length: {len(dataset)}")

    # Test getting a sample
    x, y = dataset[0]
    print("\nSample 0:")
    print(f"  X shape: {x.shape}")
    for h in horizons:
        print(f"  y[{h}] shape: {y[h].shape}")

    # Create data loader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("\nData loader:")
    print(f"  Batches: {len(loader)}")

    # Test one batch
    for batch_x, batch_y in loader:
        print("\nFirst batch:")
        print(f"  X shape: {batch_x.shape}")
        for h in horizons:
            print(f"  y[{h}] shape: {batch_y[h].shape}")
        break

    print("\nData loaders working correctly!")
