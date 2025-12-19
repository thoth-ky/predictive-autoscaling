"""
Data Normalizers for Time Series
Various normalization strategies for time series data.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import pickle
from typing import Optional, Tuple


class TimeSeriesNormalizer:
    """
    Wrapper for normalizing time series data with multiple strategies.

    Supports:
    - MinMax scaling (0-1)
    - Standard scaling (z-score)
    - Robust scaling (median/IQR)
    """

    def __init__(self, method: str = "minmax"):
        """
        Initialize normalizer.

        Args:
            method: Normalization method ('minmax', 'standard', 'robust')
        """
        self.method = method

        if method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "standard":
            self.scaler = StandardScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        self.is_fitted = False

    def fit(self, data: np.ndarray) -> "TimeSeriesNormalizer":
        """
        Fit the normalizer on training data.

        Args:
            data: Training data of shape (n_samples, n_features)
                  or (n_samples, timesteps, n_features)

        Returns:
            self
        """
        # Handle 3D data (sequences)
        if data.ndim == 3:
            n_samples, n_timesteps, n_features = data.shape
            data_2d = data.reshape(-1, n_features)
        else:
            data_2d = data

        self.scaler.fit(data_2d)
        self.is_fitted = True

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted normalizer.

        Args:
            data: Data to normalize

        Returns:
            Normalized data (same shape as input)
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")

        original_shape = data.shape

        # Handle 3D data
        if data.ndim == 3:
            n_samples, n_timesteps, n_features = data.shape
            data_2d = data.reshape(-1, n_features)
            normalized_2d = self.scaler.transform(data_2d)
            return normalized_2d.reshape(original_shape)
        else:
            return self.scaler.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform (denormalize) data.

        Args:
            data: Normalized data

        Returns:
            Original scale data
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")

        original_shape = data.shape

        # Handle 3D data
        if data.ndim == 3:
            n_samples, n_timesteps, n_features = data.shape
            data_2d = data.reshape(-1, n_features)
            denormalized_2d = self.scaler.inverse_transform(data_2d)
            return denormalized_2d.reshape(original_shape)
        else:
            return self.scaler.inverse_transform(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            data: Data to fit and normalize

        Returns:
            Normalized data
        """
        self.fit(data)
        return self.transform(data)

    def save(self, path: str):
        """Save normalizer to file."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "method": self.method,
                    "scaler": self.scaler,
                    "is_fitted": self.is_fitted,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "TimeSeriesNormalizer":
        """Load normalizer from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        normalizer = cls(method=data["method"])
        normalizer.scaler = data["scaler"]
        normalizer.is_fitted = data["is_fitted"]

        return normalizer


def normalize_data(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    method: str = "minmax",
) -> Tuple:
    """
    Normalize train/val/test sets using the same fitted scaler.

    IMPORTANT: Scaler is fit ONLY on training data to prevent data leakage.

    Args:
        X_train, X_val, X_test: Input data
        y_train, y_val, y_test: Target data (optional)
        method: Normalization method

    Returns:
        Tuple of (normalized_data, X_scaler, y_scaler)
        normalized_data is dict with 'X_train', 'X_val', etc.
    """
    # Fit X scaler on training data only
    X_scaler = TimeSeriesNormalizer(method=method)
    X_train_norm = X_scaler.fit_transform(X_train)

    result = {"X_train": X_train_norm, "X_scaler": X_scaler}

    # Transform validation and test if provided
    if X_val is not None:
        result["X_val"] = X_scaler.transform(X_val)
    if X_test is not None:
        result["X_test"] = X_scaler.transform(X_test)

    # Normalize targets if provided
    y_scaler = None
    if y_train is not None:
        y_scaler = TimeSeriesNormalizer(method=method)
        result["y_train"] = y_scaler.fit_transform(y_train)

        if y_val is not None:
            result["y_val"] = y_scaler.transform(y_val)
        if y_test is not None:
            result["y_test"] = y_scaler.transform(y_test)

        result["y_scaler"] = y_scaler

    return result


if __name__ == "__main__":
    # Test normalizers
    print("Time Series Normalizers")
    print("=" * 60)

    # Create dummy data
    X_train = np.random.randn(100, 240, 8) * 50 + 100  # Mean 100, std 50
    X_val = np.random.randn(20, 240, 8) * 50 + 100
    y_train = np.random.randn(100, 60) * 20 + 50

    print("Before normalization:")
    print(f"  X_train range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"  X_train mean: {X_train.mean():.2f}")
    print(f"  y_train range: [{y_train.min():.2f}, {y_train.max():.2f}]")

    # Test each normalization method
    for method in ["minmax", "standard", "robust"]:
        print(f"\n{method.upper()} Normalization:")

        result = normalize_data(X_train, X_val, y_train=y_train, method=method)

        X_train_norm = result["X_train"]
        X_val_norm = result["X_val"]
        y_train_norm = result["y_train"]

        print(
            f"  X_train_norm range: [{X_train_norm.min():.2f}, {X_train_norm.max():.2f}]"
        )
        print(f"  X_train_norm mean: {X_train_norm.mean():.2f}")
        print(f"  X_val_norm range: [{X_val_norm.min():.2f}, {X_val_norm.max():.2f}]")
        print(
            f"  y_train_norm range: [{y_train_norm.min():.2f}, {y_train_norm.max():.2f}]"
        )

        # Test inverse transform
        X_train_restored = result["X_scaler"].inverse_transform(X_train_norm)
        diff = np.abs(X_train - X_train_restored).mean()
        print(f"  Inverse transform error: {diff:.6f}")

    print("\nNormalizers working correctly!")
