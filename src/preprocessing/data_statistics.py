"""
Data Statistics and Analysis Module

Provides comprehensive statistics collection and exploratory data analysis
for training data, including distribution metrics, temporal patterns, and
window visualizations.
"""

import json
import os
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


class DataStatistics:
    """Collects and computes comprehensive statistics about training data."""

    def __init__(self):
        self.stats = {}

    def compute_raw_data_stats(
        self, df: pd.DataFrame, metric_name: str
    ) -> Dict[str, Any]:
        """
        Compute statistics on raw data before preprocessing.

        Args:
            df: Raw DataFrame with container_labels and metric values
            metric_name: Name of the metric being analyzed

        Returns:
            Dictionary of raw data statistics
        """
        stats = {
            "metric_name": metric_name,
            "total_records": len(df),
            "columns": list(df.columns),
            "date_range": {},
            "value_statistics": {},
            "missing_data": {},
        }

        # Date range analysis
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            stats["date_range"] = {
                "start": str(df["timestamp"].min()),
                "end": str(df["timestamp"].max()),
                "duration_hours": (
                    df["timestamp"].max() - df["timestamp"].min()
                ).total_seconds()
                / 3600,
                "total_datapoints": len(df),
            }

            # Time gaps analysis
            time_diffs = df["timestamp"].diff().dt.total_seconds()
            stats["date_range"]["median_interval_seconds"] = float(
                time_diffs.median()
            )
            stats["date_range"]["max_gap_seconds"] = float(time_diffs.max())

        # Value statistics
        if "value" in df.columns:
            values = df["value"].dropna()
            stats["value_statistics"] = {
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "percentile_25": float(values.quantile(0.25)),
                "percentile_75": float(values.quantile(0.75)),
                "percentile_95": float(values.quantile(0.95)),
                "percentile_99": float(values.quantile(0.99)),
                "zero_values": int((values == 0).sum()),
                "zero_percentage": float((values == 0).sum() / len(values) * 100),
            }

        # Missing data analysis
        stats["missing_data"] = {
            col: {
                "count": int(df[col].isna().sum()),
                "percentage": float(df[col].isna().sum() / len(df) * 100),
            }
            for col in df.columns
        }

        return stats

    def compute_processed_data_stats(
        self,
        processed: pd.DataFrame,
        container_names: Optional[List[str]] = None,
        is_multi_container: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute statistics on processed data after feature engineering.

        Args:
            processed: Processed DataFrame with features
            container_names: List of container names (for multi-container)
            is_multi_container: Whether this is multi-container training

        Returns:
            Dictionary of processed data statistics
        """
        stats = {
            "total_records": len(processed),
            "feature_columns": [
                col
                for col in processed.columns
                if col
                not in ["timestamp", "value", "container_name", "container_id", "metric_name"]
            ],
            "containers": {},
        }

        # Container distribution
        if is_multi_container and "container_name" in processed.columns:
            container_counts = processed["container_name"].value_counts()
            stats["containers"] = {
                "unique_count": len(container_counts),
                "names": list(container_counts.index),
                "distribution": {
                    name: {
                        "count": int(count),
                        "percentage": float(count / len(processed) * 100),
                    }
                    for name, count in container_counts.items()
                },
            }
        elif not is_multi_container and "container_name" in processed.columns:
            stats["containers"] = {
                "name": processed["container_name"].iloc[0]
                if len(processed) > 0
                else "unknown",
                "count": len(processed),
            }

        # Feature statistics
        feature_stats = {}
        for col in stats["feature_columns"]:
            if col in processed.columns and pd.api.types.is_numeric_dtype(
                processed[col]
            ):
                values = processed[col].dropna()
                if len(values) > 0:
                    feature_stats[col] = {
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "missing_count": int(processed[col].isna().sum()),
                    }

        stats["feature_statistics"] = feature_stats

        return stats

    def compute_window_stats(
        self,
        X: np.ndarray,
        y_dict: Dict[int, np.ndarray],
        container_ids: Optional[np.ndarray] = None,
        vocab: Optional[Any] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Compute statistics on windowed data (sequences).

        Args:
            X: Input windows (n_samples, window_size, n_features)
            y_dict: Target values for each horizon {horizon: (n_samples,)}
            container_ids: Container IDs for each window (multi-container)
            vocab: Container vocabulary (multi-container)
            metadata: Additional metadata about windows

        Returns:
            Dictionary of window statistics
        """
        stats = {
            "total_windows": len(X),
            "window_shape": X.shape,
            "n_features": X.shape[2] if len(X.shape) > 2 else X.shape[1],
            "window_size": X.shape[1] if len(X.shape) > 2 else 1,
            "horizons": {},
        }

        # Horizon statistics
        for horizon, y in y_dict.items():
            horizon_minutes = horizon // 4  # Convert timesteps to minutes
            stats["horizons"][f"{horizon_minutes}min"] = {
                "horizon_timesteps": horizon,
                "shape": y.shape,
                "mean": float(np.mean(y)),
                "std": float(np.std(y)),
                "min": float(np.min(y)),
                "max": float(np.max(y)),
            }

        # Input statistics
        stats["input_statistics"] = {
            "mean": float(np.mean(X)),
            "std": float(np.std(X)),
            "min": float(np.min(X)),
            "max": float(np.max(X)),
            "nan_count": int(np.isnan(X).sum()),
            "inf_count": int(np.isinf(X).sum()),
        }

        # Container distribution in windows (if multi-container)
        if container_ids is not None and vocab is not None:
            container_counts = Counter(container_ids)
            stats["container_distribution"] = {
                vocab.get_name(cid): {
                    "window_count": count,
                    "percentage": float(count / len(X) * 100),
                }
                for cid, count in sorted(container_counts.items())
            }

        # Add metadata if available
        if metadata:
            stats["metadata"] = metadata

        return stats

    def compute_split_stats(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train_dict: Dict[int, np.ndarray],
        y_val_dict: Dict[int, np.ndarray],
        y_test_dict: Dict[int, np.ndarray],
        container_ids_train: Optional[np.ndarray] = None,
        container_ids_val: Optional[np.ndarray] = None,
        container_ids_test: Optional[np.ndarray] = None,
        vocab: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Compute statistics on train/val/test splits.

        Args:
            X_train, X_val, X_test: Split input data
            y_train_dict, y_val_dict, y_test_dict: Split target data
            container_ids_train/val/test: Container IDs for each split
            vocab: Container vocabulary

        Returns:
            Dictionary of split statistics
        """
        total_samples = len(X_train) + len(X_val) + len(X_test)

        stats = {
            "total_samples": total_samples,
            "splits": {
                "train": {
                    "samples": len(X_train),
                    "percentage": float(len(X_train) / total_samples * 100),
                },
                "validation": {
                    "samples": len(X_val),
                    "percentage": float(len(X_val) / total_samples * 100),
                },
                "test": {
                    "samples": len(X_test),
                    "percentage": float(len(X_test) / total_samples * 100),
                },
            },
        }

        # Per-split horizon statistics
        for split_name, y_dict in [
            ("train", y_train_dict),
            ("validation", y_val_dict),
            ("test", y_test_dict),
        ]:
            horizon_stats = {}
            for horizon, y in y_dict.items():
                horizon_minutes = horizon // 4
                horizon_stats[f"{horizon_minutes}min"] = {
                    "mean": float(np.mean(y)),
                    "std": float(np.std(y)),
                    "min": float(np.min(y)),
                    "max": float(np.max(y)),
                }
            stats["splits"][split_name]["horizon_statistics"] = horizon_stats

        # Container distribution per split (if multi-container)
        if all(
            ids is not None
            for ids in [container_ids_train, container_ids_val, container_ids_test]
        ):
            if vocab is not None:
                for split_name, container_ids in [
                    ("train", container_ids_train),
                    ("validation", container_ids_val),
                    ("test", container_ids_test),
                ]:
                    container_counts = Counter(container_ids)
                    stats["splits"][split_name]["container_distribution"] = {
                        vocab.get_name(cid): {
                            "count": count,
                            "percentage": float(count / len(container_ids) * 100),
                        }
                        for cid, count in sorted(container_counts.items())
                    }

        return stats

    def generate_summary_report(self) -> str:
        """
        Generate a human-readable summary report.

        Returns:
            Formatted string report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("DATA STATISTICS SUMMARY")
        lines.append("=" * 80)

        if "raw_data" in self.stats:
            raw = self.stats["raw_data"]
            lines.append("\n### RAW DATA ###")
            lines.append(f"Metric: {raw.get('metric_name', 'N/A')}")
            lines.append(f"Total records: {raw.get('total_records', 0):,}")
            lines.append(f"Columns: {', '.join(raw.get('columns', []))}")

            if "date_range" in raw:
                dr = raw["date_range"]
                lines.append(f"\nTime Range:")
                lines.append(f"  Start: {dr.get('start', 'N/A')}")
                lines.append(f"  End: {dr.get('end', 'N/A')}")
                lines.append(f"  Duration: {dr.get('duration_hours', 0):.2f} hours")
                lines.append(
                    f"  Median interval: {dr.get('median_interval_seconds', 0):.1f}s"
                )

            if "value_statistics" in raw:
                vs = raw["value_statistics"]
                lines.append(f"\nValue Statistics:")
                lines.append(f"  Mean: {vs.get('mean', 0):.4f}")
                lines.append(f"  Std: {vs.get('std', 0):.4f}")
                lines.append(
                    f"  Range: [{vs.get('min', 0):.4f}, {vs.get('max', 0):.4f}]"
                )
                lines.append(
                    f"  P50/P95/P99: {vs.get('median', 0):.4f} / {vs.get('percentile_95', 0):.4f} / {vs.get('percentile_99', 0):.4f}"
                )
                lines.append(
                    f"  Zero values: {vs.get('zero_values', 0)} ({vs.get('zero_percentage', 0):.2f}%)"
                )

        if "processed_data" in self.stats:
            proc = self.stats["processed_data"]
            lines.append("\n### PROCESSED DATA ###")
            lines.append(f"Total records: {proc.get('total_records', 0):,}")
            lines.append(
                f"Features: {len(proc.get('feature_columns', []))} - {', '.join(proc.get('feature_columns', []))}"
            )

            if "containers" in proc:
                cont = proc["containers"]
                if "unique_count" in cont:
                    lines.append(f"\nContainers: {cont['unique_count']}")
                    if "distribution" in cont:
                        for name, info in cont["distribution"].items():
                            lines.append(
                                f"  {name}: {info['count']:,} ({info['percentage']:.1f}%)"
                            )

        if "windows" in self.stats:
            win = self.stats["windows"]
            lines.append("\n### WINDOWED DATA ###")
            lines.append(f"Total windows: {win.get('total_windows', 0):,}")
            lines.append(f"Window shape: {win.get('window_shape', 'N/A')}")
            lines.append(
                f"Features per timestep: {win.get('n_features', 0)}"
            )
            lines.append(f"Window size: {win.get('window_size', 0)} timesteps")

            if "horizons" in win:
                lines.append(f"\nPrediction Horizons:")
                for horizon_name, hstats in win["horizons"].items():
                    lines.append(
                        f"  {horizon_name}: mean={hstats.get('mean', 0):.4f}, std={hstats.get('std', 0):.4f}"
                    )

            if "container_distribution" in win:
                lines.append(f"\nContainer Distribution in Windows:")
                for name, info in win["container_distribution"].items():
                    lines.append(
                        f"  {name}: {info['window_count']:,} ({info['percentage']:.1f}%)"
                    )

        if "splits" in self.stats:
            split = self.stats["splits"]
            lines.append("\n### TRAIN/VAL/TEST SPLITS ###")
            lines.append(f"Total samples: {split.get('total_samples', 0):,}")
            for split_name in ["train", "validation", "test"]:
                if split_name in split["splits"]:
                    s = split["splits"][split_name]
                    lines.append(
                        f"  {split_name.capitalize()}: {s.get('samples', 0):,} ({s.get('percentage', 0):.1f}%)"
                    )

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def save_to_file(self, filepath: str):
        """Save statistics to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.stats, f, indent=2, default=str)
        print(f"Statistics saved to: {filepath}")

    def add_stats(self, key: str, stats: Dict[str, Any]):
        """Add a statistics dictionary with a given key."""
        self.stats[key] = stats


def generate_window_examples(
    X: np.ndarray,
    y_dict: Dict[int, np.ndarray],
    container_ids: Optional[np.ndarray] = None,
    vocab: Optional[Any] = None,
    n_examples: int = 3,
    feature_names: Optional[List[str]] = None,
) -> str:
    """
    Generate human-readable examples of windows for visualization.

    Args:
        X: Input windows
        y_dict: Target values
        container_ids: Container IDs (optional)
        vocab: Container vocabulary (optional)
        n_examples: Number of examples to show
        feature_names: Names of features

    Returns:
        Formatted string with window examples
    """
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("WINDOW EXAMPLES")
    lines.append("=" * 80)

    n_examples = min(n_examples, len(X))

    # Show examples from beginning, middle, and end
    indices = []
    if n_examples >= 1:
        indices.append(0)
    if n_examples >= 2:
        indices.append(len(X) // 2)
    if n_examples >= 3:
        indices.append(len(X) - 1)

    for idx_num, idx in enumerate(indices[:n_examples], 1):
        lines.append(f"\n### Example {idx_num} (Window index: {idx}) ###")

        # Container info
        if container_ids is not None and vocab is not None:
            container_name = vocab.get_name(container_ids[idx])
            lines.append(f"Container: {container_name}")

        # Window shape
        window = X[idx]
        lines.append(f"Shape: {window.shape}")

        # Show first and last few timesteps
        if len(window.shape) == 2:
            n_timesteps, n_features = window.shape
            lines.append(f"\nInput sequence ({n_timesteps} timesteps x {n_features} features):")

            # First 3 timesteps
            lines.append("  First 3 timesteps:")
            for t in range(min(3, n_timesteps)):
                if feature_names and len(feature_names) == n_features:
                    feature_str = ", ".join(
                        f"{name}={window[t, i]:.4f}"
                        for i, name in enumerate(feature_names[:5])
                    )
                    if n_features > 5:
                        feature_str += f", ... ({n_features} total)"
                else:
                    feature_str = ", ".join(f"{window[t, i]:.4f}" for i in range(min(5, n_features)))
                    if n_features > 5:
                        feature_str += f", ... ({n_features} total)"
                lines.append(f"    t={t}: {feature_str}")

            # Last 3 timesteps
            if n_timesteps > 6:
                lines.append("  ...")
            if n_timesteps > 3:
                lines.append(f"  Last 3 timesteps:")
                for t in range(max(3, n_timesteps - 3), n_timesteps):
                    if feature_names and len(feature_names) == n_features:
                        feature_str = ", ".join(
                            f"{name}={window[t, i]:.4f}"
                            for i, name in enumerate(feature_names[:5])
                        )
                        if n_features > 5:
                            feature_str += f", ... ({n_features} total)"
                    else:
                        feature_str = ", ".join(f"{window[t, i]:.4f}" for i in range(min(5, n_features)))
                        if n_features > 5:
                            feature_str += f", ... ({n_features} total)"
                    lines.append(f"    t={t}: {feature_str}")

        # Target values for each horizon
        lines.append(f"\nTarget values:")
        for horizon, y in sorted(y_dict.items()):
            horizon_minutes = horizon // 4
            target = y[idx]
            if hasattr(target, '__len__') and len(target) > 1:
                # It's a sequence - show summary
                lines.append(f"  {horizon_minutes}min horizon: mean={target.mean():.4f}, min={target.min():.4f}, max={target.max():.4f}")
            else:
                lines.append(f"  {horizon_minutes}min horizon: {float(target):.4f}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)
