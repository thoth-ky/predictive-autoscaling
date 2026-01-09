#!/usr/bin/env python3
"""
Local Training Script
Train time series models for container metrics locally.

Usage:
    python scripts/train_local.py --metric cpu --model-type lstm
    python scripts/train_local.py --metric memory --model-type arima
    python scripts/train_local.py --metric disk_reads --model-type prophet
"""

import argparse
import sys
import os
import glob
from typing import Optional
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.base_config import load_config, create_default_config  # noqa: E402
from src.preprocessing.metric_specific import (
    prepare_metric_data,
    MetricType,
    extract_container_name,
    build_container_vocabulary,
    add_container_ids,
)  # noqa: E402
from src.preprocessing.sliding_windows import (  # noqa: E402
    create_multi_horizon_features_and_windows,
    MultiHorizonWindowGenerator,
)
from src.preprocessing.data_splitter import split_temporal_data  # noqa: E402
from src.preprocessing.data_statistics import (  # noqa: E402
    DataStatistics,
    generate_window_examples,
)
from src.preprocessing.feature_cache import FeatureCache  # noqa: E402
from src.preprocessing.exploratory_analysis import ExploratoryAnalyzer  # noqa: E402
from src.training.metric_trainer import MetricTrainer  # noqa: E402


def find_latest_data_file(data_dir: str = "data/raw/metrics") -> str:
    """
    Find the most recent metrics CSV file based on timestamp in filename.

    The function looks for CSV files with timestamps in their names (format: YYYYMMDD_HHMMSS)
    and returns the one with the most recent timestamp.

    Args:
        data_dir: Directory to search for metrics files

    Returns:
        Path to the most recent metrics file

    Raises:
        FileNotFoundError: If no metrics files are found
    """
    data_pattern = os.path.join(data_dir, "*.csv")
    data_files = glob.glob(data_pattern)

    if not data_files:
        raise FileNotFoundError(
            f"No metrics data found in {data_dir}. "
            f"Please run a metrics export script first:\n"
            f"  python scripts/exporters/export_metrics_targeted.py"
        )

    # Try to extract timestamps from filenames to find the most recent
    import re
    from datetime import datetime

    file_timestamps = []
    for filepath in data_files:
        filename = os.path.basename(filepath)
        # Look for timestamp pattern like 20251224_001604
        match = re.search(r'(\d{8})_(\d{6})', filename)
        if match:
            timestamp_str = match.group(1) + match.group(2)
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
                file_timestamps.append((filepath, timestamp))
            except ValueError:
                # If parsing fails, use file modification time as fallback
                file_timestamps.append((filepath, datetime.fromtimestamp(os.path.getmtime(filepath))))
        else:
            # No timestamp in filename, use file modification time
            file_timestamps.append((filepath, datetime.fromtimestamp(os.path.getmtime(filepath))))

    # Sort by timestamp (most recent first) and return the first one
    file_timestamps.sort(key=lambda x: x[1], reverse=True)
    latest_file = file_timestamps[0][0]
    
    print(f"  Auto-selected most recent file: {os.path.basename(latest_file)}")
    print(f"  Timestamp: {file_timestamps[0][1]}")
    
    return latest_file


def discover_containers(df: pd.DataFrame) -> list:
    """
    Auto-discover unique container names from CSV.

    Args:
        df: Raw metrics DataFrame with 'container_labels' column

    Returns:
        Sorted list of unique container names
    """
    df_temp = df.copy()
    df_temp["container_name"] = df_temp["container_labels"].apply(
        extract_container_name
    )
    containers = sorted(df_temp["container_name"].unique().tolist())
    print(f"📦 Discovered {len(containers)} containers: {', '.join(containers)}")
    return containers


def parse_container_selection(
    container_arg: str, containers_list: Optional[list], df: pd.DataFrame
) -> list:
    """
    Parse container argument and return list of containers to train on.

    Args:
        container_arg: Value from --container argument
        containers_list: Value from --containers argument (takes precedence)
        df: Raw DataFrame for auto-discovery

    Returns:
        List of container names to train on
    """
    if containers_list:  # Explicit --containers arg takes precedence
        return containers_list
    elif container_arg == "all":
        return discover_containers(df)
    elif "," in container_arg:  # Comma-separated list
        return [c.strip() for c in container_arg.split(",")]
    else:
        return [container_arg]  # Single container


def prepare_training_data(
    metric_name: str,
    container_names: list,
    data_file: Optional[str] = None,
    window_size_minutes: int = 60,
    prediction_horizons_minutes: Optional[list] = None,
    use_cache: bool = True,
    save_cache: bool = True,
    collect_statistics: bool = True,
    show_eda: bool = False,
):
    """
    Load and prepare data for training (supports multi-container).

    Args:
        metric_name: Name of metric to train on
        container_names: List of container names to analyze
        data_file: Path to CSV data file
        window_size_minutes: Lookback window in minutes
        prediction_horizons_minutes: List of prediction horizons in minutes
        use_cache: Whether to use cached features if available
        save_cache: Whether to save processed features to cache
        collect_statistics: Whether to collect and display data statistics
        show_eda: Whether to show exploratory data analysis

    Returns:
        Tuple of (X_train, y_train_dict, X_val, y_val_dict, X_test, y_test_dict,
                  container_ids_train, container_ids_val, container_ids_test,
                  vocab, scalers, metadata, statistics, feature_names)
    """
    if prediction_horizons_minutes is None:
        prediction_horizons_minutes = [5, 15, 30]

    # Initialize statistics collector
    stats_collector = DataStatistics() if collect_statistics else None
    eda_analyzer = ExploratoryAnalyzer() if show_eda else None

    # Initialize cache
    cache = FeatureCache() if (use_cache or save_cache) else None

    # Check cache first
    if use_cache and cache and data_file:
        cache_exists, cache_key = cache.exists(
            data_file,
            metric_name,
            container_names,
            window_size_minutes,
            prediction_horizons_minutes,
        )

        if cache_exists:
            print(f"\n{'='*70}")
            print("LOADING FROM CACHE")
            print(f"{'='*70}")

            (
                X_train,
                y_train_dict,
                X_val,
                y_val_dict,
                X_test,
                y_test_dict,
                container_ids_train,
                container_ids_val,
                container_ids_test,
                vocab,
                feature_names,
                metadata,
            ) = cache.load(cache_key)

            # Collect statistics on cached data if requested
            if collect_statistics:
                print("\nCollecting statistics from cached data...")
                stats_collector.add_stats(
                    "splits",
                    stats_collector.compute_split_stats(
                        X_train,
                        X_val,
                        X_test,
                        y_train_dict,
                        y_val_dict,
                        y_test_dict,
                        container_ids_train,
                        container_ids_val,
                        container_ids_test,
                        vocab,
                    ),
                )

            return (
                X_train,
                y_train_dict,
                X_val,
                y_val_dict,
                X_test,
                y_test_dict,
                container_ids_train,
                container_ids_val,
                container_ids_test,
                vocab,
                None,
                metadata,
                stats_collector,
                feature_names,
            )

    # Load data
    if data_file is None:
        # Automatic file detection
        data_file = find_latest_data_file()

    print(f"\n{'='*70}")
    print("LOADING AND PROCESSING DATA")
    print(f"{'='*70}")
    print(f"\nLoading data from: {data_file}")
    df = pd.read_csv(data_file)
    print(f"  Total records: {len(df):,}")

    # Collect raw data statistics
    if collect_statistics:
        print("\nCollecting raw data statistics...")
        raw_stats = stats_collector.compute_raw_data_stats(df, metric_name)
        stats_collector.add_stats("raw_data", raw_stats)

    # Perform EDA on raw data
    if show_eda and eda_analyzer:
        print("\nPerforming exploratory data analysis on raw data...")
        # Store raw df for EDA report generation later
        raw_df_for_eda = df.copy()

    # Determine if multi-container mode
    is_multi_container = len(container_names) > 1
    container_arg = container_names if is_multi_container else container_names[0]

    # Preprocess metric-specific data
    print(f"\nPreprocessing {metric_name} data...")
    processed = prepare_metric_data(df, metric_name, container_arg)
    print(f"  Processed records: {len(processed):,}")
    print(
        f"  Time range: {processed['timestamp'].min()} to {processed['timestamp'].max()}"
    )

    # Collect processed data statistics
    if collect_statistics:
        print("\nCollecting processed data statistics...")
        processed_stats = stats_collector.compute_processed_data_stats(
            processed, container_names, is_multi_container
        )
        stats_collector.add_stats("processed_data", processed_stats)

    # Build container vocabulary for multi-container mode
    vocab = None
    container_ids_array = None
    if is_multi_container:
        print(
            f"\nBuilding container vocabulary for {len(container_names)} containers..."
        )
        vocab = build_container_vocabulary(processed)
        print(f"  Vocabulary size: {vocab.num_containers}")
        print(f"  Containers: {', '.join(vocab.get_all_names())}")

        # Add container IDs
        processed = add_container_ids(processed, vocab)
        container_ids_array = processed["container_id"].values

    # Create features and windows
    print("\nCreating features and windows...")

    # Convert prediction horizons from minutes to timesteps (15s intervals)
    prediction_horizons_timesteps = [h * 4 for h in prediction_horizons_minutes]
    window_size_timesteps = window_size_minutes * 4

    if is_multi_container:
        # Add features before windowing (matching single-container behavior)
        from src.preprocessing.sliding_windows import (
            add_temporal_features,
            add_lag_features,
            add_rolling_features,
        )

        print("  Adding temporal, lag, and rolling features...")
        processed = add_temporal_features(processed)
        processed = add_lag_features(processed)
        processed = add_rolling_features(processed)

        # Drop NaN rows created by lag/rolling features
        processed = processed.dropna().reset_index(drop=True)

        # Get feature columns (exclude timestamp, original value, and metadata)
        # Also exclude 'metric_name' as it's a string column
        feature_columns = [
            col
            for col in processed.columns
            if col
            not in [
                "timestamp",
                "value",
                "container_name",
                "container_id",
                "metric_name",
            ]
        ]

        if not feature_columns:
            # If no features, just use the raw value
            feature_columns = ["value"]

        print(f"  Using {len(feature_columns)} features: {feature_columns}")

        # Prepare data arrays
        data_values = processed[feature_columns].values

        # Create container ID to name mapping for better error messages
        container_id_to_name = (
            processed.groupby("container_id")["container_name"].first().to_dict()
        )

        # Use multi-container windowing
        generator = MultiHorizonWindowGenerator(
            window_size=window_size_timesteps,
            prediction_horizons=prediction_horizons_timesteps,
            stride=4,  # 1 minute between windows
        )

        X, y_dict, window_container_ids, metadata = (
            generator.create_multi_container_sequences(
                data_values,
                processed["container_id"].values,
                processed["timestamp"],
                container_id_to_name=container_id_to_name,
            )
        )

        print(f"  Created {len(X)} windows across {len(container_names)} containers")
        print(f"  Window shape: {X.shape}")
        for horizon in prediction_horizons_timesteps:
            print(
                f"    Horizon {horizon//4}min ({horizon} steps): {y_dict[horizon].shape}"
            )

        # Store feature names for consistency with single-container path
        feature_names = feature_columns

    else:
        # Single-container mode (original behavior)
        X, y_dict, feature_names, metadata = create_multi_horizon_features_and_windows(
            processed,
            container_name=container_names[0],
            metric_name=MetricType.from_string(metric_name).value,
            window_size_minutes=window_size_minutes,
            prediction_horizon_minutes=prediction_horizons_minutes,
        )
        window_container_ids = None

    # Split data temporally
    print("\nSplitting data (70% train, 15% val, 15% test)...")

    if is_multi_container:
        # Split with container IDs
        split_results = split_temporal_data(
            X, y_dict, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        X_train, X_val, X_test, y_train_dict, y_val_dict, y_test_dict = split_results

        # Split container IDs using same indices
        n_train = len(X_train)
        n_val = len(X_val)

        container_ids_train = window_container_ids[:n_train]
        container_ids_val = window_container_ids[n_train : n_train + n_val]
        container_ids_test = window_container_ids[n_train + n_val :]

    else:
        X_train, X_val, X_test, y_train_dict, y_val_dict, y_test_dict = (
            split_temporal_data(
                X, y_dict, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
            )
        )
        container_ids_train = None
        container_ids_val = None
        container_ids_test = None

    print("\nData preparation complete!")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Test samples: {len(X_test):,}")

    # Collect window and split statistics
    if collect_statistics:
        print("\nCollecting window statistics...")
        window_stats = stats_collector.compute_window_stats(
            X, y_dict, window_container_ids, vocab, metadata
        )
        stats_collector.add_stats("windows", window_stats)

        print("Collecting split statistics...")
        split_stats = stats_collector.compute_split_stats(
            X_train,
            X_val,
            X_test,
            y_train_dict,
            y_val_dict,
            y_test_dict,
            container_ids_train,
            container_ids_val,
            container_ids_test,
            vocab,
        )
        stats_collector.add_stats("splits", split_stats)

        # Print summary report
        print(stats_collector.generate_summary_report())

    # Show window examples
    if collect_statistics:
        print(
            generate_window_examples(
                X, y_dict, window_container_ids, vocab, n_examples=2, feature_names=feature_names
            )
        )

    # Generate and show EDA report
    if show_eda and eda_analyzer:
        print("\nGenerating exploratory data analysis report...")
        eda_report = eda_analyzer.generate_eda_report(
            raw_df=raw_df_for_eda if 'raw_df_for_eda' in locals() else None,
            processed_df=processed,
            X=X,
            y_dict=y_dict,
            feature_names=feature_names,
            metric_name=metric_name,
        )
        print(eda_report)

    if is_multi_container and not collect_statistics:
        # Only print if we haven't already shown in statistics
        print("\n  Container distribution in training set:")
        from collections import Counter

        train_counts = Counter(container_ids_train)
        for cid, count in sorted(train_counts.items()):
            container_name = vocab.get_name(cid)
            print(
                f"    {container_name}: {count} samples ({100*count/len(X_train):.1f}%)"
            )

    # Save to cache if requested
    if save_cache and cache and data_file:
        print(f"\n{'='*70}")
        print("SAVING TO CACHE")
        print(f"{'='*70}")
        cache.save(
            data_file,
            metric_name,
            container_names,
            window_size_minutes,
            prediction_horizons_minutes,
            X_train,
            y_train_dict,
            X_val,
            y_val_dict,
            X_test,
            y_test_dict,
            container_ids_train,
            container_ids_val,
            container_ids_test,
            vocab,
            feature_names,
            metadata,
        )

    return (
        X_train,
        y_train_dict,
        X_val,
        y_val_dict,
        X_test,
        y_test_dict,
        container_ids_train,
        container_ids_val,
        container_ids_test,
        vocab,
        None,
        metadata,
        stats_collector,
        feature_names,
    )


def main():  # noqa: C901
    parser = argparse.ArgumentParser(
        description="Train time series models for container metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train LSTM model for CPU (single container)
  python scripts/train_local.py --metric cpu --model-type lstm \\
      --data-file data/raw/metrics/container_cpu_rate_20251224_001604.csv

  # Train on all containers (auto-discovery)
  python scripts/train_local.py --metric cpu --container all \\
      --data-file data/raw/metrics/container_cpu_rate_20251224_001604.csv

  # Train on specific containers (comma-separated)
  python scripts/train_local.py --metric cpu --container "webapp,db,redis" \\
      --data-file data/raw/metrics/container_cpu_rate_20251224_001604.csv

  # Train on specific containers (space-separated)
  python scripts/train_local.py --metric cpu --containers webapp db redis \\
      --data-file data/raw/metrics/container_cpu_rate_20251224_001604.csv

  # Train ARIMA for memory
  python scripts/train_local.py --metric memory --model-type arima \\
      --data-file data/raw/metrics/container_memory_usage_20251224.csv

  # Train Prophet for network with custom config
  python scripts/train_local.py --metric network_rx --model-type prophet \\
      --config custom.yaml --data-file data/raw/metrics/container_network_20251224.csv

  # Train with full statistics and exploratory analysis
  python scripts/train_local.py --metric cpu --show-eda --show-stats \\
      --data-file data/raw/metrics/container_cpu_rate_20251224_001604.csv

  # Use cached features for faster subsequent runs
  python scripts/train_local.py --metric cpu --use-cache \\
      --data-file data/raw/metrics/container_cpu_rate_20251224_001604.csv

  # List cached features
  python scripts/train_local.py --list-cache

  # Clear old cache
  python scripts/train_local.py --clear-cache
        """,
    )

    parser.add_argument(
        "--metric",
        required=True,
        choices=[
            "cpu",
            "memory",
            "disk_reads",
            "disk_writes",
            "network_rx",
            "network_tx",
        ],
        help="Metric to train model for",
    )

    parser.add_argument(
        "--model-type",
        default="lstm",
        choices=["lstm", "arima", "prophet"],
        help="Type of model to train",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom YAML config (optional, uses defaults if not specified)",
    )

    parser.add_argument(
        "--data-file",
        type=str,
        help="Path to metrics CSV file",
    )

    # Container selection (mutually exclusive)
    container_group = parser.add_mutually_exclusive_group()
    container_group.add_argument(
        "--container",
        type=str,
        default="webapp",
        help="Container name, 'all' for auto-discovery, or comma-separated list (e.g., 'webapp,db,redis')",
    )

    container_group.add_argument(
        "--containers",
        type=str,
        nargs="+",
        help="Explicit list of containers (space-separated), e.g., webapp db redis",
    )

    parser.add_argument(
        "--no-mlflow", action="store_true", help="Disable MLflow experiment tracking"
    )

    parser.add_argument(
        "--register-model",
        action="store_true",
        default=True,
        help="Register model in MLflow Model Registry (default: True)",
    )

    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Don't register model in MLflow Model Registry",
    )

    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs (overrides config)"
    )

    # Caching and analysis options
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Use cached features if available (default: True)",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use or save cached features",
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached features and exit",
    )

    parser.add_argument(
        "--list-cache",
        action="store_true",
        help="List all cached features and exit",
    )

    parser.add_argument(
        "--show-stats",
        action="store_true",
        default=True,
        help="Show detailed data statistics (default: True)",
    )

    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Don't collect or show data statistics",
    )

    parser.add_argument(
        "--show-eda",
        action="store_true",
        help="Show exploratory data analysis with visualizations",
    )

    parser.add_argument(
        "--save-stats",
        type=str,
        help="Save statistics to JSON file (e.g., experiments/stats/cpu_stats.json)",
    )

    parser.add_argument(
        "--save-eda-report",
        type=str,
        help="Save EDA report to text file (e.g., experiments/reports/cpu_eda.txt)",
    )

    args = parser.parse_args()

    # Handle cache management commands
    if args.clear_cache or args.list_cache:
        from src.preprocessing.feature_cache import FeatureCache

        cache = FeatureCache()

        if args.list_cache:
            cache.print_cache_info()

        if args.clear_cache:
            print("\nClearing feature cache...")
            cache.clear_cache()
            print("Cache cleared successfully!")

        sys.exit(0)

    print("=" * 70)
    print(f"Training {args.model_type.upper()} model for {args.metric}")
    print("=" * 70)

    # Load raw data first to discover containers
    if not args.data_file:
        print("\nError: --data-file argument is required")
        print("Please specify a metrics CSV file:")
        print(
            "  python scripts/train_local.py --metric cpu --data-file data/raw/metrics/container_cpu_rate_*.csv"
        )
        sys.exit(1)

    print(f"\nLoading data to discover containers: {args.data_file}")
    df_raw = pd.read_csv(args.data_file)

    # Parse container selection
    selected_containers = parse_container_selection(
        args.container, args.containers, df_raw
    )

    print(
        f"\n🎯 Training on {len(selected_containers)} container(s): {', '.join(selected_containers)}"
    )

    # Load or create configuration
    if args.config:
        print(f"\nLoading config from: {args.config}")
        config = load_config(args.config)
    else:
        config_path = f"src/config/model_configs/{args.metric}_config.yaml"
        if os.path.exists(config_path):
            print(f"\nLoading default config: {config_path}")
            config = load_config(config_path)
        else:
            print(f"\nCreating default config for {args.metric}")
            config = create_default_config(args.metric, args.model_type)

    # Override model type
    config.model.model_type = args.model_type

    # Configure multi-container settings
    if len(selected_containers) > 1:
        print(f"\nConfiguring multi-container training:")
        config.model.use_container_embeddings = True
        config.model.num_containers = len(selected_containers)
        config.container_name = "multi"  # Special marker for multi-container
        print(f"  - Using container embeddings: True")
        print(f"  - Number of containers: {config.model.num_containers}")
        print(f"  - Embedding dimension: {config.model.container_embedding_dim}")
    else:
        config.container_name = selected_containers[0]

    # Override epochs if specified
    if args.epochs:
        config.training.epochs = args.epochs

    # Determine cache and stats settings
    use_cache = args.use_cache and not args.no_cache
    save_cache = use_cache  # Save to cache if we're using cache
    collect_stats = args.show_stats and not args.no_stats
    show_eda = args.show_eda

    # Prepare data with container selection
    # Convert horizons from timesteps to minutes (4 timesteps = 1 minute at 15s intervals)
    prediction_horizons_minutes = [h // 4 for h in config.data.prediction_horizons]

    try:
        (
            X_train,
            y_train_dict,
            X_val,
            y_val_dict,
            X_test,
            y_test_dict,
            container_ids_train,
            container_ids_val,
            container_ids_test,
            vocab,
            scalers,
            metadata,
            stats_collector,
            feature_names,
        ) = prepare_training_data(
            metric_name=args.metric,
            container_names=selected_containers,
            data_file=args.data_file,
            window_size_minutes=60,
            prediction_horizons_minutes=prediction_horizons_minutes,
            use_cache=use_cache,
            save_cache=save_cache,
            collect_statistics=collect_stats,
            show_eda=show_eda,
        )
    except Exception as e:
        print(f"\nError preparing data: {e}")
        print("\nPlease ensure you have metrics data available.")
        print("Run one of the export scripts first:")
        print("  python scripts/exporters/export_metrics_targeted.py")
        raise e
        sys.exit(1)

    # Determine model registration setting
    register_model = args.register_model and not args.no_register

    # Create trainer
    trainer = MetricTrainer(
        config, use_mlflow=not args.no_mlflow, register_model=register_model
    )

    # Store vocabulary for saving with model
    if vocab is not None:
        trainer.container_vocab = vocab

    # Prepare normalized data with container IDs
    trainer.prepare_data(
        X_train,
        y_train_dict,
        X_val,
        y_val_dict,
        X_test,
        y_test_dict,
        container_ids_train=container_ids_train,
        container_ids_val=container_ids_val,
        container_ids_test=container_ids_test,
    )

    # Train model
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)

    # Evaluate
    print("\nEvaluating on test set...")
    results = trainer.evaluate()

    # Finalize training (close MLflow run if active)
    trainer.finalize()

    # Save results summary
    results_dir = f"experiments/results/{args.metric}"
    os.makedirs(results_dir, exist_ok=True)

    import json
    from datetime import datetime

    results_file = os.path.join(
        results_dir,
        f'{args.model_type}_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
    )

    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    with open(results_file, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Save statistics if requested
    if args.save_stats and stats_collector:
        stats_file = args.save_stats
        stats_collector.save_to_file(stats_file)

    # Save EDA report if requested
    if args.save_eda_report and show_eda:
        from src.preprocessing.exploratory_analysis import ExploratoryAnalyzer

        eda_analyzer = ExploratoryAnalyzer()
        # We don't have raw_df here, but we can generate report from what we have
        eda_report = eda_analyzer.generate_eda_report(
            X=X_train,
            y_dict=y_train_dict,
            feature_names=feature_names,
            metric_name=args.metric,
        )
        eda_analyzer.save_report(eda_report, args.save_eda_report)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
