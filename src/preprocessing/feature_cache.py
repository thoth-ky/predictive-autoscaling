"""
Feature Caching System

Provides efficient caching of processed features to disk, avoiding redundant
preprocessing and saving significant time on subsequent training runs.
"""

import hashlib
import json
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


class FeatureCache:
    """
    Manages caching of processed features to disk.

    Uses content-based hashing to ensure cache validity and avoid stale data.
    """

    def __init__(self, cache_dir: str = "data/cache/features"):
        """
        Initialize feature cache.

        Args:
            cache_dir: Directory to store cached features
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _compute_cache_key(
        self,
        data_file: str,
        metric_name: str,
        container_names: List[str],
        window_size_minutes: int,
        prediction_horizons_minutes: List[int],
        additional_params: Optional[Dict] = None,
    ) -> str:
        """
        Compute a unique cache key based on data and parameters.

        Args:
            data_file: Path to raw data file
            metric_name: Metric being processed
            container_names: List of containers
            window_size_minutes: Window size
            prediction_horizons_minutes: Prediction horizons
            additional_params: Any additional parameters affecting processing

        Returns:
            Hexadecimal cache key string
        """
        # Get file modification time and size for data file fingerprint
        if os.path.exists(data_file):
            file_stat = os.stat(data_file)
            file_fingerprint = f"{file_stat.st_mtime}_{file_stat.st_size}"
        else:
            file_fingerprint = "unknown"

        # Build parameter dictionary
        params = {
            "data_file": data_file,
            "file_fingerprint": file_fingerprint,
            "metric_name": metric_name,
            "container_names": sorted(container_names),  # Sort for consistency
            "window_size_minutes": window_size_minutes,
            "prediction_horizons_minutes": sorted(prediction_horizons_minutes),
        }

        if additional_params:
            params.update(additional_params)

        # Create deterministic string representation
        param_str = json.dumps(params, sort_keys=True)

        # Hash to create cache key
        cache_key = hashlib.sha256(param_str.encode()).hexdigest()[:16]

        return cache_key

    def _get_cache_path(self, cache_key: str) -> str:
        """Get the full path for a cache file."""
        return os.path.join(self.cache_dir, f"features_{cache_key}.pkl")

    def _get_metadata_path(self, cache_key: str) -> str:
        """Get the full path for a cache metadata file."""
        return os.path.join(self.cache_dir, f"metadata_{cache_key}.json")

    def exists(
        self,
        data_file: str,
        metric_name: str,
        container_names: List[str],
        window_size_minutes: int,
        prediction_horizons_minutes: List[int],
        additional_params: Optional[Dict] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if cached features exist for the given parameters.

        Returns:
            Tuple of (exists: bool, cache_key: str or None)
        """
        cache_key = self._compute_cache_key(
            data_file,
            metric_name,
            container_names,
            window_size_minutes,
            prediction_horizons_minutes,
            additional_params,
        )

        cache_path = self._get_cache_path(cache_key)
        exists = os.path.exists(cache_path)

        return exists, cache_key if exists else None

    def save(
        self,
        data_file: str,
        metric_name: str,
        container_names: List[str],
        window_size_minutes: int,
        prediction_horizons_minutes: List[int],
        X_train: np.ndarray,
        y_train_dict: Dict[int, np.ndarray],
        X_val: np.ndarray,
        y_val_dict: Dict[int, np.ndarray],
        X_test: np.ndarray,
        y_test_dict: Dict[int, np.ndarray],
        container_ids_train: Optional[np.ndarray] = None,
        container_ids_val: Optional[np.ndarray] = None,
        container_ids_test: Optional[np.ndarray] = None,
        vocab: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        additional_params: Optional[Dict] = None,
    ) -> str:
        """
        Save processed features to cache.

        Args:
            data_file: Path to raw data file
            metric_name: Metric name
            container_names: List of containers
            window_size_minutes: Window size
            prediction_horizons_minutes: Prediction horizons
            X_train, y_train_dict: Training data
            X_val, y_val_dict: Validation data
            X_test, y_test_dict: Test data
            container_ids_train/val/test: Container IDs for splits
            vocab: Container vocabulary
            feature_names: Names of features
            metadata: Additional metadata
            additional_params: Additional parameters

        Returns:
            Cache key
        """
        cache_key = self._compute_cache_key(
            data_file,
            metric_name,
            container_names,
            window_size_minutes,
            prediction_horizons_minutes,
            additional_params,
        )

        # Prepare data bundle
        data_bundle = {
            "X_train": X_train,
            "y_train_dict": y_train_dict,
            "X_val": X_val,
            "y_val_dict": y_val_dict,
            "X_test": X_test,
            "y_test_dict": y_test_dict,
            "container_ids_train": container_ids_train,
            "container_ids_val": container_ids_val,
            "container_ids_test": container_ids_test,
            "vocab": vocab,
            "feature_names": feature_names,
            "metadata": metadata,
        }

        # Save data bundle
        cache_path = self._get_cache_path(cache_key)
        with open(cache_path, "wb") as f:
            pickle.dump(data_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save metadata
        metadata_info = {
            "cache_key": cache_key,
            "created_at": datetime.now().isoformat(),
            "data_file": data_file,
            "metric_name": metric_name,
            "container_names": container_names,
            "window_size_minutes": window_size_minutes,
            "prediction_horizons_minutes": prediction_horizons_minutes,
            "shapes": {
                "X_train": X_train.shape,
                "X_val": X_val.shape,
                "X_test": X_test.shape,
            },
            "additional_params": additional_params,
        }

        metadata_path = self._get_metadata_path(cache_key)
        with open(metadata_path, "w") as f:
            json.dump(metadata_info, f, indent=2, default=str)

        # Calculate cache file size
        cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)

        print(f"\nFeatures cached successfully!")
        print(f"  Cache key: {cache_key}")
        print(f"  Cache size: {cache_size_mb:.2f} MB")
        print(f"  Location: {cache_path}")

        return cache_key

    def load(
        self, cache_key: str
    ) -> Tuple[
        np.ndarray,
        Dict[int, np.ndarray],
        np.ndarray,
        Dict[int, np.ndarray],
        np.ndarray,
        Dict[int, np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[Any],
        Optional[List[str]],
        Optional[Dict],
    ]:
        """
        Load cached features.

        Args:
            cache_key: Cache key to load

        Returns:
            Tuple of (X_train, y_train_dict, X_val, y_val_dict, X_test, y_test_dict,
                     container_ids_train, container_ids_val, container_ids_test,
                     vocab, feature_names, metadata)
        """
        cache_path = self._get_cache_path(cache_key)

        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        print(f"\nLoading cached features...")
        print(f"  Cache key: {cache_key}")

        with open(cache_path, "rb") as f:
            data_bundle = pickle.load(f)

        # Load metadata for info
        metadata_path = self._get_metadata_path(cache_key)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata_info = json.load(f)
                print(f"  Created: {metadata_info.get('created_at', 'unknown')}")
                print(f"  Metric: {metadata_info.get('metric_name', 'unknown')}")
                container_names = metadata_info.get('container_names', [])
                print(f"  Containers: {len(container_names)}")

        print(f"  Train samples: {len(data_bundle['X_train']):,}")
        print(f"  Val samples: {len(data_bundle['X_val']):,}")
        print(f"  Test samples: {len(data_bundle['X_test']):,}")

        return (
            data_bundle["X_train"],
            data_bundle["y_train_dict"],
            data_bundle["X_val"],
            data_bundle["y_val_dict"],
            data_bundle["X_test"],
            data_bundle["y_test_dict"],
            data_bundle.get("container_ids_train"),
            data_bundle.get("container_ids_val"),
            data_bundle.get("container_ids_test"),
            data_bundle.get("vocab"),
            data_bundle.get("feature_names"),
            data_bundle.get("metadata"),
        )

    def list_cached_items(self) -> List[Dict[str, Any]]:
        """
        List all cached items with their metadata.

        Returns:
            List of metadata dictionaries
        """
        cached_items = []

        for filename in os.listdir(self.cache_dir):
            if filename.startswith("metadata_") and filename.endswith(".json"):
                metadata_path = os.path.join(self.cache_dir, filename)
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        cached_items.append(metadata)
                except Exception as e:
                    print(f"Warning: Could not read {filename}: {e}")

        # Sort by creation time (newest first)
        cached_items.sort(
            key=lambda x: x.get("created_at", ""), reverse=True
        )

        return cached_items

    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear cached items.

        Args:
            older_than_days: Only clear items older than this many days.
                           If None, clears all items.
        """
        import time

        cleared_count = 0
        current_time = time.time()

        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)

            if older_than_days is not None:
                file_age_days = (current_time - os.path.getmtime(filepath)) / 86400
                if file_age_days < older_than_days:
                    continue

            try:
                os.remove(filepath)
                cleared_count += 1
            except Exception as e:
                print(f"Warning: Could not remove {filename}: {e}")

        print(f"Cleared {cleared_count} cache files from {self.cache_dir}")

    def print_cache_info(self):
        """Print information about cached items."""
        cached_items = self.list_cached_items()

        if not cached_items:
            print(f"\nNo cached features found in {self.cache_dir}")
            return

        print(f"\n{'='*80}")
        print(f"CACHED FEATURES ({len(cached_items)} items)")
        print(f"{'='*80}")

        for idx, item in enumerate(cached_items, 1):
            print(f"\n{idx}. {item.get('metric_name', 'unknown')}")
            print(f"   Cache key: {item.get('cache_key', 'unknown')}")
            print(f"   Created: {item.get('created_at', 'unknown')}")
            container_names = item.get('container_names', [])
            print(f"   Containers: {len(container_names)}")
            print(
                f"   Window size: {item.get('window_size_minutes', 0)} minutes"
            )
            if "shapes" in item:
                shapes = item["shapes"]
                total_samples = (
                    shapes.get("X_train", [0])[0]
                    + shapes.get("X_val", [0])[0]
                    + shapes.get("X_test", [0])[0]
                )
                print(f"   Total samples: {total_samples:,}")

        print(f"\n{'='*80}")
