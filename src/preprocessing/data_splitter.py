"""
Temporal Data Splitter
Handle train/validation/test splitting for time series data.

CRITICAL: No shuffling - maintain temporal order!
"""

import numpy as np
from typing import Tuple, Dict, Optional, Union


class TemporalSplitter:
    """
    Split time series data into train/val/test sets while preserving temporal order.

    IMPORTANT: Unlike random splitting, temporal splitting ensures:
    - Train data comes BEFORE validation data
    - Validation data comes BEFORE test data
    - No data leakage from future to past
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        """
        Initialize temporal splitter.

        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing

        Raises:
            ValueError: If ratios don't sum to 1.0
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
            )

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(
        self,
        X: np.ndarray,
        y: Union[np.ndarray, Dict[int, np.ndarray]],
        metadata: Optional[np.ndarray] = None,
    ) -> Tuple:
        """
        Split data into train/val/test sets.

        Args:
            X: Input sequences of shape (n_samples, window_size, n_features)
            y: Targets - either:
               - Single array (n_samples, horizon)
               - Dict mapping horizon to arrays {20: (n_samples, 20), ...}
            metadata: Optional metadata for each sample

        Returns:
            If y is array: (X_train, X_val, X_test, y_train, y_val, y_test)
            If y is dict: (X_train, X_val, X_test, y_train_dict, y_val_dict, y_test_dict)
            If metadata provided, also returns (meta_train, meta_val, meta_test)
        """
        n_samples = len(X)

        # Calculate split indices
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))

        # Split X
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]

        # Split y (handle both single array and dict)
        if isinstance(y, dict):
            y_train_dict = {h: y[h][:train_end] for h in y.keys()}
            y_val_dict = {h: y[h][train_end:val_end] for h in y.keys()}
            y_test_dict = {h: y[h][val_end:] for h in y.keys()}

            if metadata is not None:
                meta_train = metadata[:train_end]
                meta_val = metadata[train_end:val_end]
                meta_test = metadata[val_end:]
                return (
                    X_train,
                    X_val,
                    X_test,
                    y_train_dict,
                    y_val_dict,
                    y_test_dict,
                    meta_train,
                    meta_val,
                    meta_test,
                )
            else:
                return X_train, X_val, X_test, y_train_dict, y_val_dict, y_test_dict

        else:
            y_train = y[:train_end]
            y_val = y[train_end:val_end]
            y_test = y[val_end:]

            if metadata is not None:
                meta_train = metadata[:train_end]
                meta_val = metadata[train_end:val_end]
                meta_test = metadata[val_end:]
                return (
                    X_train,
                    X_val,
                    X_test,
                    y_train,
                    y_val,
                    y_test,
                    meta_train,
                    meta_val,
                    meta_test,
                )
            else:
                return X_train, X_val, X_test, y_train, y_val, y_test

    def get_split_info(self, n_samples: int) -> Dict:
        """
        Get information about how data will be split.

        Args:
            n_samples: Total number of samples

        Returns:
            Dictionary with split sizes and indices
        """
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))

        return {
            "total_samples": n_samples,
            "train_samples": train_end,
            "val_samples": val_end - train_end,
            "test_samples": n_samples - val_end,
            "train_indices": (0, train_end),
            "val_indices": (train_end, val_end),
            "test_indices": (val_end, n_samples),
        }


def split_temporal_data(
    X: np.ndarray,
    y: Union[np.ndarray, Dict[int, np.ndarray]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    metadata: Optional[np.ndarray] = None,
) -> Tuple:
    """
    Convenience function to split time series data temporally.

    Args:
        X: Input sequences
        y: Target sequences (array or dict)
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        metadata: Optional metadata

    Returns:
        Split datasets (train, val, test)

    Example:
        >>> X_train, X_val, X_test, y_train, y_val, y_test = split_temporal_data(X, y)
    """
    splitter = TemporalSplitter(train_ratio, val_ratio, test_ratio)
    return splitter.split(X, y, metadata)


if __name__ == "__main__":
    # Test temporal splitter
    print("Temporal Data Splitter")
    print("=" * 60)

    # Create dummy data
    n_samples = 1000
    window_size = 240
    n_features = 8
    horizons = [20, 60, 120]

    X = np.random.randn(n_samples, window_size, n_features)
    y_dict = {h: np.random.randn(n_samples, h) for h in horizons}

    print(f"Total samples: {n_samples}")
    print(f"Input shape: {X.shape}")
    print(f"Target horizons: {horizons}")

    # Test splitter
    splitter = TemporalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    # Get split info
    split_info = splitter.get_split_info(n_samples)
    print("\nSplit Information:")
    for key, value in split_info.items():
        print(f"  {key}: {value}")

    # Perform split
    X_train, X_val, X_test, y_train_dict, y_val_dict, y_test_dict = splitter.split(
        X, y_dict
    )

    print("\nAfter Split:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")

    for h in horizons:
        print(f"\n  Horizon {h}:")
        print(f"    y_train: {y_train_dict[h].shape}")
        print(f"    y_val: {y_val_dict[h].shape}")
        print(f"    y_test: {y_test_dict[h].shape}")

    # Verify temporal order preserved
    print("\nTemporal Order Verification:")
    print(f"  Train samples: indices 0 to {split_info['train_samples'] - 1}")
    print(
        f"  Val samples: indices {split_info['train_samples']} "
        f"to {split_info['val_indices'][1] - 1}"
    )
    print(f"  Test samples: indices {split_info['val_indices'][1]} to {n_samples - 1}")
    print("  ✅ No shuffling - temporal order preserved!")
