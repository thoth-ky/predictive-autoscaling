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
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.base_config import load_config, create_default_config
from src.preprocessing.metric_specific import prepare_metric_data
from src.preprocessing.sliding_windows import create_multi_horizon_features_and_windows
from src.preprocessing.data_splitter import split_temporal_data
from src.training.metric_trainer import MetricTrainer


def find_latest_data_file(data_dir: str = 'data/raw') -> str:
    """Find the most recent metrics CSV file."""
    data_pattern = os.path.join(data_dir, 'metrics_*.csv')
    data_files = glob.glob(data_pattern)

    if not data_files:
        raise FileNotFoundError(
            f"No metrics data found in {data_dir}. "
            f"Please run a metrics export script first:\n"
            f"  cd scripts && python export_metrics_simple.py --seconds 900"
        )

    latest_file = max(data_files)
    return latest_file


def prepare_training_data(metric_name: str, container_name: str = 'webapp',
                         data_file: Optional[str] = None,
                         window_size_minutes: int = 60,
                         prediction_horizons_minutes: list = None):
    """
    Load and prepare data for training.

    Args:
        metric_name: Name of metric to train on
        container_name: Container to analyze
        data_file: Path to CSV data file (auto-detects if None)
        window_size_minutes: Lookback window in minutes
        prediction_horizons_minutes: List of prediction horizons in minutes

    Returns:
        Tuple of (X_train, y_train_dict, X_val, y_val_dict, X_test, y_test_dict)
    """
    if prediction_horizons_minutes is None:
        prediction_horizons_minutes = [5, 15, 30]

    # Load data
    if data_file is None:
        data_file = find_latest_data_file()

    print(f"\nLoading data from: {data_file}")
    df = pd.read_csv(data_file)
    print(f"  Total records: {len(df):,}")

    # Preprocess metric-specific data
    print(f"\nPreprocessing {metric_name} data...")
    processed = prepare_metric_data(df, metric_name, container_name)
    print(f"  Processed records: {len(processed):,}")
    print(f"  Time range: {processed['timestamp'].min()} to {processed['timestamp'].max()}")

    # Create features and windows
    print(f"\nCreating features and windows...")
    X, y_dict, feature_names, metadata = create_multi_horizon_features_and_windows(
        processed,
        container_name=container_name,
        metric_name=f'container_{metric_name}' if not metric_name.startswith('container_') else metric_name,
        window_size_minutes=window_size_minutes,
        prediction_horizon_minutes=prediction_horizons_minutes
    )

    # Split data temporally
    print(f"\nSplitting data (70% train, 15% val, 15% test)...")
    X_train, X_val, X_test, y_train_dict, y_val_dict, y_test_dict = split_temporal_data(
        X, y_dict,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    print(f"\nData preparation complete!")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Test samples: {len(X_test):,}")

    return X_train, y_train_dict, X_val, y_val_dict, X_test, y_test_dict


def main():
    parser = argparse.ArgumentParser(
        description='Train time series models for container metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train LSTM model for CPU
  python scripts/train_local.py --metric cpu --model-type lstm

  # Train ARIMA for memory
  python scripts/train_local.py --metric memory --model-type arima

  # Train Prophet for network with custom config
  python scripts/train_local.py --metric network_rx --model-type prophet --config custom.yaml

  # Train with custom data file
  python scripts/train_local.py --metric cpu --data-file data/raw/metrics_20251210.csv
        """
    )

    parser.add_argument(
        '--metric',
        required=True,
        choices=['cpu', 'memory', 'disk_reads', 'disk_writes', 'network_rx', 'network_tx'],
        help='Metric to train model for'
    )

    parser.add_argument(
        '--model-type',
        default='lstm',
        choices=['lstm', 'arima', 'prophet'],
        help='Type of model to train'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom YAML config (optional, uses defaults if not specified)'
    )

    parser.add_argument(
        '--data-file',
        type=str,
        help='Path to metrics CSV file (auto-detects latest if not specified)'
    )

    parser.add_argument(
        '--container',
        type=str,
        default='webapp',
        help='Container name to analyze (default: webapp)'
    )

    parser.add_argument(
        '--no-mlflow',
        action='store_true',
        help='Disable MLflow experiment tracking'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides config)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print(f"Training {args.model_type.upper()} model for {args.metric}")
    print("=" * 70)

    # Load or create configuration
    if args.config:
        print(f"\nLoading config from: {args.config}")
        config = load_config(args.config)
    else:
        config_path = f'src/config/model_configs/{args.metric}_config.yaml'
        if os.path.exists(config_path):
            print(f"\nLoading default config: {config_path}")
            config = load_config(config_path)
        else:
            print(f"\nCreating default config for {args.metric}")
            config = create_default_config(args.metric, args.model_type)

    # Override model type
    config.model.model_type = args.model_type

    # Override epochs if specified
    if args.epochs:
        config.training.epochs = args.epochs

    # Prepare data
    try:
        X_train, y_train_dict, X_val, y_val_dict, X_test, y_test_dict = prepare_training_data(
            metric_name=args.metric,
            container_name=args.container,
            data_file=args.data_file,
            window_size_minutes=60,
            prediction_horizons_minutes=[5, 15, 30]
        )
    except Exception as e:
        print(f"\nError preparing data: {e}")
        print("\nPlease ensure you have metrics data available.")
        print("Run one of the export scripts first:")
        print("  cd scripts && python export_metrics_simple.py --seconds 900")
        sys.exit(1)

    # Create trainer
    trainer = MetricTrainer(config, use_mlflow=not args.no_mlflow)

    # Prepare normalized data
    trainer.prepare_data(
        X_train, y_train_dict,
        X_val, y_val_dict,
        X_test, y_test_dict
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

    # Save results summary
    results_dir = f'experiments/results/{args.metric}'
    os.makedirs(results_dir, exist_ok=True)

    import json
    from datetime import datetime

    results_file = os.path.join(
        results_dir,
        f'{args.model_type}_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
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

    with open(results_file, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\nResults saved to: {results_file}")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
