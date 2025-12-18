"""
Feature Engineering: Sliding Window Creation for Time Series Prediction

This module creates sliding windows from container metrics for ML model training.
It prepares features (X) and targets (y) for predicting future resource usage.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from datetime import timedelta


class TimeSeriesWindowGenerator:
    """
    Generate sliding windows for time series prediction.
    
    Creates sequences of past observations to predict future values.
    Example: Use 60 minutes of CPU history to predict next 15 minutes.
    """
    
    def __init__(self, 
                 window_size: int = 240,  # 240 * 15s = 1 hour
                 prediction_horizon: int = 60,  # 60 * 15s = 15 minutes
                 stride: int = 1):
        """
        Initialize window generator.
        
        Args:
            window_size: Number of timesteps to look back (input sequence length)
            prediction_horizon: Number of timesteps to predict ahead (target)
            stride: Step size between consecutive windows (1 = overlapping windows)
        """
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        
    def create_sequences(self, 
                        data: np.ndarray, 
                        timestamps: Optional[pd.DatetimeIndex] = None
                        ) -> Tuple[np.ndarray, np.ndarray, Optional[List]]:
        """
        Create sliding window sequences from time series data.
        
        Args:
            data: Time series data (1D or 2D array)
                  Shape: (n_timesteps,) or (n_timesteps, n_features)
            timestamps: Optional timestamps for each data point
            
        Returns:
            X: Input sequences (past observations)
               Shape: (n_samples, window_size, n_features)
            y: Target values (future observations to predict)
               Shape: (n_samples, prediction_horizon, n_features)
            window_timestamps: List of (start_time, end_time, target_time) for each window
        """
        # Ensure 2D array
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_timesteps, n_features = data.shape
        
        # Calculate number of windows we can create
        max_index = n_timesteps - self.window_size - self.prediction_horizon
        n_windows = (max_index // self.stride) + 1
        
        if n_windows <= 0:
            raise ValueError(
                f"Not enough data to create windows. Need at least "
                f"{self.window_size + self.prediction_horizon} timesteps, "
                f"got {n_timesteps}"
            )
        
        # Preallocate arrays
        X = np.zeros((n_windows, self.window_size, n_features))
        y = np.zeros((n_windows, self.prediction_horizon, n_features))
        
        window_timestamps = [] if timestamps is not None else None
        
        # Create windows
        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            target_start = end_idx
            target_end = target_start + self.prediction_horizon
            
            X[i] = data[start_idx:end_idx]
            y[i] = data[target_start:target_end]
            
            if timestamps is not None:
                window_timestamps.append({
                    'window_start': timestamps[start_idx],
                    'window_end': timestamps[end_idx - 1],
                    'target_start': timestamps[target_start],
                    'target_end': timestamps[target_end - 1]
                })
        
        return X, y, window_timestamps
    
    def create_multivariate_sequences(self,
                                     df: pd.DataFrame,
                                     feature_columns: List[str],
                                     target_column: str,
                                     timestamp_column: str = 'timestamp'
                                     ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Create sequences from a DataFrame with multiple features.
        
        Args:
            df: DataFrame with time series data
            feature_columns: List of column names to use as input features
            target_column: Column name to predict
            timestamp_column: Name of timestamp column
            
        Returns:
            X: Input sequences (n_samples, window_size, n_features)
            y: Target sequences (n_samples, prediction_horizon)
            metadata: DataFrame with window information
        """
        # Sort by timestamp
        df = df.sort_values(timestamp_column).reset_index(drop=True)
        
        # Extract features and target
        X_data = df[feature_columns].values
        y_data = df[target_column].values.reshape(-1, 1)
        timestamps = pd.DatetimeIndex(df[timestamp_column])
        
        # Create sequences
        X, y_seq, window_times = self.create_sequences(X_data, timestamps)
        
        # For target, we typically just want the values, not the full sequence shape
        # Reshape y to (n_samples, prediction_horizon)
        y = y_seq[:, :, 0]  # Remove feature dimension if target is single column
        
        # Create metadata DataFrame
        if window_times:
            metadata = pd.DataFrame(window_times)
        else:
            metadata = pd.DataFrame()
        
        return X, y, metadata


def prepare_container_metrics(df: pd.DataFrame,
                              container_name: str,
                              metric_name: str = 'container_cpu',
                              resample_freq: str = '15S') -> pd.DataFrame:
    """
    Prepare container metrics for windowing.
    
    Args:
        df: Raw metrics DataFrame
        container_name: Name of container to extract (e.g., 'metrics-webapp')
        metric_name: Metric to extract (e.g., 'container_cpu')
        resample_freq: Resampling frequency (default: 15S for 15 seconds)
        
    Returns:
        DataFrame with resampled time series
    """
    # Filter for specific container and metric
    mask = (df['metric_name'] == metric_name)
    
    if 'label_name' in df.columns:
        mask &= (df['label_name'] == container_name)
    
    filtered = df[mask].copy()
    
    if len(filtered) == 0:
        raise ValueError(f"No data found for {container_name} / {metric_name}")
    
    # Set timestamp as index
    filtered['timestamp'] = pd.to_datetime(filtered['timestamp'])
    filtered = filtered.set_index('timestamp').sort_index()
    
    # Resample to ensure regular intervals
    resampled = filtered['value'].resample(resample_freq).mean()
    
    # Forward fill missing values (up to 3 intervals)
    resampled = resampled.fillna(method='ffill', limit=3)
    
    # Backward fill any remaining NaNs
    resampled = resampled.fillna(method='bfill', limit=3)
    
    # Create DataFrame
    result = pd.DataFrame({
        'timestamp': resampled.index,
        'value': resampled.values
    })
    
    return result


def add_temporal_features(df: pd.DataFrame, 
                         timestamp_column: str = 'timestamp') -> pd.DataFrame:
    """
    Add temporal features (hour, day of week, etc.) to help model learn patterns.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_column: Name of timestamp column
        
    Returns:
        DataFrame with additional temporal features
    """
    df = df.copy()
    
    # Ensure datetime
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    # Extract temporal features
    df['hour'] = df[timestamp_column].dt.hour
    df['day_of_week'] = df[timestamp_column].dt.dayofweek
    df['minute'] = df[timestamp_column].dt.minute
    
    # Cyclical encoding (important for models to understand time is circular)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df


def add_lag_features(df: pd.DataFrame,
                    value_column: str = 'value',
                    lags: List[int] = [1, 4, 12, 24]) -> pd.DataFrame:
    """
    Add lagged features (previous values at different time offsets).
    
    Args:
        df: DataFrame with time series data
        value_column: Column to create lags from
        lags: List of lag periods (in terms of dataframe rows)
              Default: [1, 4, 12, 24] = [15s, 1min, 3min, 6min] ago
        
    Returns:
        DataFrame with lag features
    """
    df = df.copy()
    
    for lag in lags:
        df[f'{value_column}_lag_{lag}'] = df[value_column].shift(lag)
    
    return df


def add_rolling_features(df: pd.DataFrame,
                        value_column: str = 'value',
                        windows: List[int] = [4, 12, 24, 60]) -> pd.DataFrame:
    """
    Add rolling statistics (moving averages, std dev, etc.).
    
    Args:
        df: DataFrame with time series data
        value_column: Column to calculate rolling stats from
        windows: List of window sizes
                 Default: [4, 12, 24, 60] = [1min, 3min, 6min, 15min]
        
    Returns:
        DataFrame with rolling features
    """
    df = df.copy()
    
    for window in windows:
        df[f'{value_column}_roll_mean_{window}'] = (
            df[value_column].rolling(window=window, min_periods=1).mean()
        )
        df[f'{value_column}_roll_std_{window}'] = (
            df[value_column].rolling(window=window, min_periods=1).std()
        )
        df[f'{value_column}_roll_min_{window}'] = (
            df[value_column].rolling(window=window, min_periods=1).min()
        )
        df[f'{value_column}_roll_max_{window}'] = (
            df[value_column].rolling(window=window, min_periods=1).max()
        )
    
    return df


def create_features_and_windows(df: pd.DataFrame,
                               container_name: str = 'metrics-webapp',
                               metric_name: str = 'container_cpu',
                               window_size_minutes: int = 60,
                               prediction_horizon_minutes: int = 15,
                               include_temporal: bool = True,
                               include_lags: bool = True,
                               include_rolling: bool = True) -> Tuple:
    """
    Complete pipeline: prepare data, add features, create windows.
    
    Args:
        df: Raw metrics DataFrame
        container_name: Container to analyze
        metric_name: Metric to predict
        window_size_minutes: How many minutes of history to use
        prediction_horizon_minutes: How many minutes ahead to predict
        include_temporal: Add temporal features
        include_lags: Add lag features
        include_rolling: Add rolling statistics
        
    Returns:
        X: Input sequences
        y: Target sequences
        feature_names: List of feature names
        metadata: Window metadata
    """
    # Step 1: Prepare container metrics
    prepared = prepare_container_metrics(df, container_name, metric_name)
    
    # Step 2: Add features
    if include_temporal:
        prepared = add_temporal_features(prepared)
    
    if include_lags:
        prepared = add_lag_features(prepared)
    
    if include_rolling:
        prepared = add_rolling_features(prepared)
    
    # Drop NaN rows created by lag/rolling features
    prepared = prepared.dropna().reset_index(drop=True)
    
    # Step 3: Create windows
    # Convert minutes to number of 15-second intervals
    window_size = window_size_minutes * 4  # 4 intervals per minute
    prediction_horizon = prediction_horizon_minutes * 4
    
    generator = TimeSeriesWindowGenerator(
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        stride=4  # Create window every minute
    )
    
    # Get feature columns (exclude timestamp and original value)
    feature_columns = [col for col in prepared.columns 
                      if col not in ['timestamp', 'value']]
    
    if not feature_columns:
        # If no features, just use the raw value
        feature_columns = ['value']
    
    X, y, metadata = generator.create_multivariate_sequences(
        df=prepared,
        feature_columns=feature_columns,
        target_column='value',
        timestamp_column='timestamp'
    )
    
    print(f"✅ Created {len(X)} windows")
    print(f"   Input shape: {X.shape} (samples, timesteps, features)")
    print(f"   Target shape: {y.shape} (samples, prediction_horizon)")
    print(f"   Features: {feature_columns}")
    
    return X, y, feature_columns, metadata


class MultiHorizonWindowGenerator:
    """
    Generate sliding windows with multiple prediction horizons.

    Creates sequences for predicting at multiple time horizons simultaneously
    (e.g., 5min, 15min, 30min ahead).
    """

    def __init__(self,
                 window_size: int = 240,
                 prediction_horizons: List[int] = None,
                 stride: int = 4):
        """
        Initialize multi-horizon window generator.

        Args:
            window_size: Number of timesteps to look back
            prediction_horizons: List of horizons to predict (e.g., [20, 60, 120])
            stride: Step size between windows
        """
        self.window_size = window_size
        self.prediction_horizons = prediction_horizons or [20, 60, 120]
        self.stride = stride

    def create_multi_horizon_sequences(self,
                                      data: np.ndarray,
                                      timestamps: Optional[pd.DatetimeIndex] = None
                                      ) -> Tuple:
        """
        Create windows with multiple prediction horizons.

        Args:
            data: Time series data (1D or 2D array)
            timestamps: Optional timestamps

        Returns:
            X: Input sequences (n_samples, window_size, n_features)
            y_dict: Dictionary mapping horizon to targets
                   {20: (n_samples, 20), 60: (n_samples, 60), 120: (n_samples, 120)}
            window_timestamps: List of timestamp metadata
        """
        # Ensure 2D array
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_timesteps, n_features = data.shape

        # Calculate max horizon to ensure we have enough data
        max_horizon = max(self.prediction_horizons)
        max_index = n_timesteps - self.window_size - max_horizon
        n_windows = (max_index // self.stride) + 1

        if n_windows <= 0:
            raise ValueError(
                f"Not enough data to create windows. Need at least "
                f"{self.window_size + max_horizon} timesteps, got {n_timesteps}"
            )

        # Preallocate input array
        X = np.zeros((n_windows, self.window_size, n_features))

        # Preallocate target arrays for each horizon
        y_dict = {}
        for horizon in self.prediction_horizons:
            y_dict[horizon] = np.zeros((n_windows, horizon, n_features))

        window_timestamps = [] if timestamps is not None else None

        # Create windows
        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size

            # Input window
            X[i] = data[start_idx:end_idx]

            # Target windows for each horizon
            for horizon in self.prediction_horizons:
                target_start = end_idx
                target_end = target_start + horizon
                y_dict[horizon][i] = data[target_start:target_end]

            if timestamps is not None:
                timestamp_info = {
                    'window_start': timestamps[start_idx],
                    'window_end': timestamps[end_idx - 1],
                }
                # Add target ranges for each horizon
                for horizon in self.prediction_horizons:
                    target_start = end_idx
                    target_end = target_start + horizon
                    timestamp_info[f'target_start_{horizon}'] = timestamps[target_start]
                    timestamp_info[f'target_end_{horizon}'] = timestamps[target_end - 1]

                window_timestamps.append(timestamp_info)

        # For univariate prediction, squeeze the feature dimension
        if n_features == 1:
            for horizon in self.prediction_horizons:
                y_dict[horizon] = y_dict[horizon].squeeze(-1)

        return X, y_dict, window_timestamps


def create_multi_horizon_features_and_windows(df: pd.DataFrame,
                                             container_name: str = 'metrics-webapp',
                                             metric_name: str = 'container_cpu',
                                             window_size_minutes: int = 60,
                                             prediction_horizon_minutes: List[int] = None,
                                             include_temporal: bool = True,
                                             include_lags: bool = True,
                                             include_rolling: bool = True) -> Tuple:
    """
    Complete pipeline for multi-horizon prediction: prepare data, add features, create windows.

    Args:
        df: Raw metrics DataFrame
        container_name: Container to analyze
        metric_name: Metric to predict
        window_size_minutes: Minutes of history to use
        prediction_horizon_minutes: List of minutes ahead to predict (e.g., [5, 15, 30])
        include_temporal: Add temporal features
        include_lags: Add lag features
        include_rolling: Add rolling statistics

    Returns:
        X: Input sequences
        y_dict: Target sequences per horizon
        feature_names: List of feature names
        metadata: Window metadata
    """
    if prediction_horizon_minutes is None:
        prediction_horizon_minutes = [5, 15, 30]

    # Step 1: Prepare container metrics
    prepared = prepare_container_metrics(df, container_name, metric_name)

    # Step 2: Add features
    if include_temporal:
        prepared = add_temporal_features(prepared)

    if include_lags:
        prepared = add_lag_features(prepared)

    if include_rolling:
        prepared = add_rolling_features(prepared)

    # Drop NaN rows created by lag/rolling features
    prepared = prepared.dropna().reset_index(drop=True)

    # Step 3: Create windows
    # Convert minutes to number of 15-second intervals
    window_size = window_size_minutes * 4  # 4 intervals per minute
    prediction_horizons = [h * 4 for h in prediction_horizon_minutes]  # Convert to timesteps

    generator = MultiHorizonWindowGenerator(
        window_size=window_size,
        prediction_horizons=prediction_horizons,
        stride=4  # Create window every minute
    )

    # Get feature columns (exclude timestamp and original value)
    feature_columns = [col for col in prepared.columns
                      if col not in ['timestamp', 'value']]

    if not feature_columns:
        # If no features, just use the raw value
        feature_columns = ['value']

    # Prepare data arrays
    data_values = prepared[feature_columns].values
    timestamps = pd.DatetimeIndex(prepared['timestamp'])

    X, y_dict, metadata = generator.create_multi_horizon_sequences(
        data=data_values,
        timestamps=timestamps
    )

    print(f"✅ Created {len(X)} windows with multiple horizons")
    print(f"   Input shape: {X.shape} (samples, timesteps, features)")
    print(f"   Target horizons:")
    for h, y in y_dict.items():
        horizon_minutes = h // 4
        print(f"     {horizon_minutes} min ({h} steps): {y.shape}")
    print(f"   Features: {feature_columns}")

    return X, y_dict, feature_columns, metadata


if __name__ == '__main__':
    # Example usage
    print("Sliding Window Feature Engineering Module")
    print("=" * 60)
    print("\nExample: Create 60-minute windows to predict at multiple horizons")

    # Load example data
    import glob
    data_files = glob.glob('../data/raw/metrics_*.csv')

    if data_files:
        latest_file = max(data_files, key=lambda x: x)
        print(f"\nLoading: {latest_file}")

        df = pd.read_csv(latest_file)

        # Test single horizon (original functionality)
        print("\n--- Single Horizon (15 min) ---")
        X, y, features, meta = create_features_and_windows(
            df,
            container_name='metrics-webapp',
            window_size_minutes=60,
            prediction_horizon_minutes=15
        )

        # Test multi-horizon
        print("\n--- Multi-Horizon (5, 15, 30 min) ---")
        X_multi, y_multi_dict, features_multi, meta_multi = create_multi_horizon_features_and_windows(
            df,
            container_name='metrics-webapp',
            window_size_minutes=60,
            prediction_horizon_minutes=[5, 15, 30]
        )
    else:
        print("\n⚠️  No data files found. Run export script first.")
