"""
Metric-Specific Preprocessing
Handle preprocessing for different container metrics (CPU, memory, disk, network).
"""

from enum import Enum
from typing import Optional, List
import pandas as pd
import re
from src.preprocessing.container_vocabulary import ContainerVocabulary


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


def extract_container_name(container_labels: str) -> str:
    """
    Extract container name from Prometheus compressed labels string.

    Parses labels like 'container=webapp,pod=test-pod,namespace=default'
    and extracts the container name.

    Args:
        container_labels: Comma-separated key=value pairs from Prometheus

    Returns:
        Container name (e.g., 'webapp')

    Raises:
        ValueError: If container label not found

    Example:
        >>> extract_container_name('container=webapp,pod=test,namespace=default')
        'webapp'
    """
    if pd.isna(container_labels) or not container_labels:
        raise ValueError("Container labels cannot be empty")

    # Try to match container= pattern
    container_match = re.search(r"container=([^,]+)", container_labels, re.IGNORECASE)
    if container_match:
        return container_match.group(1).strip()

    # Fallback: try pod= pattern
    pod_match = re.search(r"pod=([^,]+)", container_labels, re.IGNORECASE)
    if pod_match:
        # Extract container name from pod name (e.g., 'webapp-abc123' -> 'webapp')
        pod_name = pod_match.group(1).strip()
        # Remove trailing hash/ID if present
        base_name = re.sub(r"-[a-f0-9]+$", "", pod_name)
        return base_name

    raise ValueError(
        f"Could not extract container name from labels: {container_labels}"
    )


def build_container_vocabulary(df: pd.DataFrame) -> ContainerVocabulary:
    """
    Build container vocabulary from DataFrame.

    Extracts all unique container names and creates a vocabulary mapping
    container names to numeric IDs for embedding layers.

    Args:
        df: DataFrame with either 'container_labels' or 'container_name' column

    Returns:
        ContainerVocabulary with all unique containers

    Example:
        >>> df = pd.read_csv('metrics.csv')
        >>> vocab = build_container_vocabulary(df)
        >>> vocab.num_containers
        5
    """
    # Try to use already-extracted container_name column first
    if "container_name" in df.columns:
        unique_containers = sorted(df["container_name"].unique())
    elif "container_labels" in df.columns:
        # Fall back to extracting from container_labels
        container_names = df["container_labels"].apply(extract_container_name)
        unique_containers = sorted(container_names.unique())
    else:
        raise ValueError(
            "DataFrame must have either 'container_name' or 'container_labels' column"
        )

    # Build vocabulary
    vocab = ContainerVocabulary()
    for container in unique_containers:
        vocab.add_container(container)

    return vocab


def add_container_ids(
    df: pd.DataFrame, vocab: ContainerVocabulary
) -> pd.DataFrame:
    """
    Add numeric container_id column using vocabulary.

    Maps container names to numeric IDs for embedding layers.
    Modifies DataFrame in place by adding 'container_id' column.

    Args:
        df: DataFrame with 'container_name' column
        vocab: ContainerVocabulary for mapping

    Returns:
        DataFrame with added 'container_id' column

    Raises:
        ValueError: If DataFrame missing 'container_name' column
        KeyError: If container name not in vocabulary

    Example:
        >>> vocab = build_container_vocabulary(df)
        >>> df = add_container_ids(df, vocab)
        >>> df['container_id'].unique()
        array([0, 1, 2])
    """
    if "container_name" not in df.columns:
        raise ValueError("DataFrame must have 'container_name' column")

    # Map container names to IDs
    df["container_id"] = df["container_name"].apply(vocab.get_id)

    return df


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
            Preprocessed DataFrame with timestamp, value, container_name, and metric_name columns
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
        resampled = self._resample_data(transformed, container_name)

        # 4. Handle outliers
        cleaned = self._handle_outliers(resampled)

        return cleaned

    def _filter_data(
        self, df: pd.DataFrame, container_name: str, service_col: str
    ) -> pd.DataFrame:
        """
        Filter DataFrame for specific container(s) using container_labels.

        Now supports multi-container mode:
        - If container_name is "all", returns all containers
        - If container_name is a list, filters to those containers
        - If container_name is a single string, filters to that container

        Always extracts and adds 'container_name' column from labels.
        """
        if "container_labels" not in df.columns:
            raise ValueError("DataFrame must have 'container_labels' column")

        # Extract container name from labels and add as column
        df = df.copy()
        df["container_name"] = df["container_labels"].apply(extract_container_name)

        # If no specific container requested, return all data
        if not container_name or container_name == "all":
            return df

        # If list of containers, filter to those containers
        if isinstance(container_name, list):
            mask = df["container_name"].isin(container_name)
            filtered = df[mask]
            if len(filtered) == 0:
                raise ValueError(
                    f"No data found for containers {container_name}. "
                    f"Available: {df['container_name'].unique().tolist()}"
                )
            return filtered

        # Single container - filter by container name
        mask = df["container_name"] == container_name
        filtered = df[mask]

        # If exact match failed, try case-insensitive partial match
        if len(filtered) == 0:
            mask = df["container_name"].str.contains(
                container_name, case=False, na=False
            )
            filtered = df[mask]

        return filtered

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

    def _resample_data(self, df: pd.DataFrame, container_name: str) -> pd.DataFrame:
        """
        Resample to regular intervals and handle missing values.

        For multi-container data, resamples each container separately to
        maintain temporal continuity within each container.

        Args:
            df: DataFrame with timestamp and value columns
            container_name: Name of container(s) for metadata

        Returns:
            Resampled DataFrame with regular intervals
        """
        # Ensure timestamp column is datetime
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Check if this is multi-container data
        if "container_name" in df.columns and df["container_name"].nunique() > 1:
            # Multi-container mode: resample each container separately
            resampled_parts = []

            for container in df["container_name"].unique():
                container_df = df[df["container_name"] == container].copy()

                # Resample this container's data
                container_df = container_df.set_index("timestamp").sort_index()
                resampled_values = container_df["value"].resample(
                    self.resample_freq
                ).mean()

                # Forward fill missing values (up to 3 intervals)
                resampled_values = resampled_values.ffill(limit=3)

                # Backward fill any remaining NaNs
                resampled_values = resampled_values.bfill(limit=3)

                # Create DataFrame for this container
                container_result = pd.DataFrame(
                    {
                        "timestamp": resampled_values.index,
                        "value": resampled_values.values,
                        "container_name": container,
                        "metric_name": self.metric_type.value,
                    }
                )

                resampled_parts.append(container_result)

            # Combine all containers
            result = pd.concat(resampled_parts, ignore_index=True)
            result = result.sort_values(["container_name", "timestamp"])
            result = result.reset_index(drop=True)

        else:
            # Single container mode: original logic
            df_indexed = df.set_index("timestamp").sort_index()

            # Resample to regular intervals (default 15 seconds)
            resampled = df_indexed["value"].resample(self.resample_freq).mean()

            # Forward fill missing values (up to 3 intervals)
            resampled = resampled.ffill(limit=3)

            # Backward fill any remaining NaNs
            resampled = resampled.bfill(limit=3)

            # Get container name from data if available
            actual_container_name = (
                df["container_name"].iloc[0]
                if "container_name" in df.columns
                else container_name
            )

            # Create clean DataFrame with container and metric information
            result = pd.DataFrame(
                {
                    "timestamp": resampled.index,
                    "value": resampled.values,
                    "container_name": actual_container_name,
                    "metric_name": self.metric_type.value,
                }
            ).reset_index(drop=True)

        # Drop any remaining NaNs
        result = result.dropna()

        return result

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers using statistical threshold.

        Uses z-score method: values beyond threshold * std are capped.
        This preserves temporal continuity (doesn't remove data points).

        For multi-container data, calculates outliers per-container to avoid
        cross-contamination.
        """
        if self.outlier_threshold is None:
            return df

        # Check if multi-container data
        if "container_name" in df.columns and df["container_name"].nunique() > 1:
            # Handle outliers per container
            processed_parts = []

            for container in df["container_name"].unique():
                container_df = df[df["container_name"] == container].copy()

                mean = float(container_df["value"].mean())
                std = float(container_df["value"].std())

                if std > 0:  # Only clip if there's variation
                    upper_bound = mean + (self.outlier_threshold * std)
                    lower_bound = max(0, mean - (self.outlier_threshold * std))
                    container_df["value"] = container_df["value"].clip(
                        lower=lower_bound, upper=upper_bound
                    )

                processed_parts.append(container_df)

            # Combine all containers
            df = pd.concat(processed_parts, ignore_index=True)
        else:
            # Single container mode: original logic
            mean = float(df["value"].mean())
            std = float(df["value"].std())

            if std > 0:  # Only clip if there's variation
                upper_bound = mean + (self.outlier_threshold * std)
                lower_bound = max(0, mean - (self.outlier_threshold * std))
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
