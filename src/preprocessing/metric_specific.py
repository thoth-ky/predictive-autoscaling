"""
Metric-Specific Preprocessing
Handle preprocessing for different container metrics (CPU, memory, disk, network).
"""

from enum import Enum
from typing import Optional
import pandas as pd


class MetricType(Enum):
    """Enumeration of supported container metrics."""

    CPU = "container_cpu_rate"
    MEMORY = "container_memory_usage"
    DISK_READS = "container_fs_reads_rate"
    DISK_WRITES = "container_fs_writes_rate"
    NETWORK_RX = "container_network_receive_rate"
    NETWORK_TX = "container_network_transmit_rate"

    @classmethod
    def from_string(cls, metric_name: str):
        """Get MetricType from string name."""
        mapping = {
            "cpu": cls.CPU,
            "memory": cls.MEMORY,
            "disk_reads": cls.DISK_READS,
            "disk_writes": cls.DISK_WRITES,
            "network_rx": cls.NETWORK_RX,
            "network_tx": cls.NETWORK_TX,
        }
        return mapping.get(metric_name.lower(), cls.CPU)


class MetricPreprocessor:
    """
    Preprocess container metrics with metric-specific transformations.

    Handles:
    - Metric-specific unit conversions
    - Outlier detection and clipping
    - Resampling to regular intervals
    - Missing value handling
    """

    def __init__(self, metric_type: MetricType, config: Optional[dict] = None):
        """
        Initialize preprocessor.

        Args:
            metric_type: Type of metric to preprocess
            config: Configuration with outlier_threshold, etc.
        """
        self.metric_type = metric_type
        self.config = config or {}
        self.outlier_threshold = self.config.get("outlier_threshold", 3.0)
        self.resample_freq = self.config.get("resample_freq", "15s")

    def process(
        self,
        df: pd.DataFrame,
        container_name: str = "webapp",
        service_col: str = "service",
    ) -> pd.DataFrame:
        """
        Main preprocessing pipeline.

        Args:
            df: Raw metrics DataFrame
            container_name: Name or service identifier of container
            service_col: Column name for service/container identifier

        Returns:
            Preprocessed DataFrame with timestamp and value columns
        """
        # 1. Filter for specific metric and container
        filtered = self._filter_data(df, container_name, service_col)

        if len(filtered) == 0:
            raise ValueError(
                f"No data found for metric={self.metric_type.value}, "
                f"container={container_name}"
            )

        # 2. Apply metric-specific transformations
        transformed = self._metric_specific_transform(filtered.copy())

        # 3. Resample to ensure regular intervals
        resampled = self._resample_data(transformed)

        # 4. Handle outliers
        cleaned = self._handle_outliers(resampled)

        return cleaned

    def _filter_data(
        self, df: pd.DataFrame, container_name: str, service_col: str
    ) -> pd.DataFrame:
        """Filter DataFrame for specific container using container_labels."""
        if "container_labels" not in df.columns:
            raise ValueError("DataFrame must have 'container_labels' column")

        # If no specific container requested, return all data
        if not container_name or container_name == "all":
            return df

        # Filter by container name in the labels
        mask = df["container_labels"].str.contains(
            f"container={container_name}", case=False, na=False
        ) | df["container_labels"].str.contains(
            f"pod={container_name}", case=False, na=False
        )

        return df[mask]

    def _metric_specific_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply metric-specific transformations and unit conversions.

        - CPU: Convert to percentage (0-100)
        - Memory: Convert bytes to MB
        - Disk: Ensure non-negative rates
        - Network: Convert bytes/sec to Mbps
        """
        if self.metric_type == MetricType.CPU:
            # CPU is already a rate, ensure it's in percentage
            # If it's a fraction (0-1), convert to percentage
            if df["value"].max() <= 1.0:
                df["value"] = df["value"] * 100
            # Clip to valid range
            df["value"] = df["value"].clip(lower=0, upper=100)

        elif self.metric_type in [MetricType.MEMORY]:
            # Convert bytes to MB for easier interpretation
            df["value"] = df["value"] / (1024**2)
            df["value"] = df["value"].clip(lower=0)

        elif self.metric_type in [MetricType.DISK_READS, MetricType.DISK_WRITES]:
            # Disk I/O already a rate, ensure non-negative
            df["value"] = df["value"].clip(lower=0)
            # Optionally convert to ops/sec if needed
            # (already in that unit from Prometheus rate())

        elif self.metric_type in [MetricType.NETWORK_RX, MetricType.NETWORK_TX]:
            # Convert bytes/sec to Mbps for easier interpretation
            df["value"] = (df["value"] * 8) / (1024**2)
            df["value"] = df["value"].clip(lower=0)

        return df

    def _resample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample to regular intervals and handle missing values.

        Args:
            df: DataFrame with timestamp and value columns

        Returns:
            Resampled DataFrame with regular intervals
        """
        # Ensure timestamp column is datetime
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Set timestamp as index
        df_indexed = df.set_index("timestamp").sort_index()

        # Resample to regular intervals (default 15 seconds)
        resampled = df_indexed["value"].resample(self.resample_freq).mean()

        # Forward fill missing values (up to 3 intervals)
        resampled = resampled.ffill(limit=3)

        # Backward fill any remaining NaNs
        resampled = resampled.bfill(limit=3)

        # Create clean DataFrame
        result = pd.DataFrame(
            {"timestamp": resampled.index, "value": resampled.values}
        ).reset_index(drop=True)

        # Drop any remaining NaNs
        result = result.dropna()

        return result

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers using statistical threshold.

        Uses z-score method: values beyond threshold * std are capped.
        This preserves temporal continuity (doesn't remove data points).
        """
        if self.outlier_threshold is None:
            return df

        mean = float(df["value"].mean())
        std = float(df["value"].std())

        if std == 0:
            return df  # No variation, no outliers

        # Calculate bounds
        upper_bound = mean + (self.outlier_threshold * std)
        lower_bound = max(0, mean - (self.outlier_threshold * std))

        # Clip outliers instead of removing them
        df["value"] = df["value"].clip(lower=lower_bound, upper=upper_bound)

        return df


def prepare_metric_data(
    df: pd.DataFrame,
    metric_name: str,
    container_name: str = "webapp",
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Convenience function to preprocess metric data.

    Args:
        df: Raw metrics DataFrame from CSV export
        metric_name: Name of metric (cpu, memory, disk_reads, etc.)
        container_name: Target container/service name
        config: Optional configuration dict

    Returns:
        Preprocessed DataFrame ready for feature engineering

    Example:
        >>> df = pd.read_csv('data/raw/metrics_latest.csv')
        >>> cpu_data = prepare_metric_data(df, 'cpu', 'webapp')
    """
    metric_type = MetricType.from_string(metric_name)
    preprocessor = MetricPreprocessor(metric_type, config)

    return preprocessor.process(df, container_name)


if __name__ == "__main__":
    # Example usage
    import glob

    print("Metric-Specific Preprocessing Module")
    print("=" * 60)

    # Find latest data files
    data_files = glob.glob(
        "/home/thoth/dpn/predictive-autoscaling/data/raw/metrics/container_*.csv"
    )

    if data_files:
        latest_file = max(data_files)
        print(f"\nLoading: {latest_file}")

        df = pd.read_csv(latest_file)
        print(f"Raw data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Test preprocessing for each metric
        metrics = [
            "cpu",
            "memory",
            "disk_reads",
            "disk_writes",
            "network_rx",
            "network_tx",
        ]

        for metric in metrics:
            try:
                processed = prepare_metric_data(df, metric, "webapp")
                print(f"\n{metric.upper()}:")
                print(f"  Processed shape: {processed.shape}")
                print(
                    f"  Value range: [{processed['value'].min():.2f}, "
                    f"{processed['value'].max():.2f}]"
                )
                print(
                    f"  Time range: {processed['timestamp'].min()} to "
                    f"{processed['timestamp'].max()}"
                )
            except ValueError as e:
                print(f"\n{metric.upper()}: {str(e)}")
    else:
        print("\nNo data files found. Run metrics export script first.")
        print("Example: python scripts/exporters/export_metrics_targeted.py")
