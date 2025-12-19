"""
Preprocessing Module
Data loading, feature engineering, and temporal splitting.
"""

from src.preprocessing.sliding_windows import (
    TimeSeriesWindowGenerator,
    prepare_container_metrics,
    add_temporal_features,
    add_lag_features,
    add_rolling_features,
    create_features_and_windows,
    create_multi_horizon_features_and_windows,
)
from src.preprocessing.data_splitter import (
    split_temporal_data,
)
from src.preprocessing.metric_specific import (
    prepare_metric_data,
)

__all__ = [
    # Window generation
    "TimeSeriesWindowGenerator",
    "prepare_container_metrics",
    "add_temporal_features",
    "add_lag_features",
    "add_rolling_features",
    "create_features_and_windows",
    "create_multi_horizon_features_and_windows",
    # Data splitting
    "split_temporal_data",
    # Metric-specific preprocessing
    "prepare_metric_data",
]
