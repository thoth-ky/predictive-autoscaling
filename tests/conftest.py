"""
Shared Test Fixtures
Pytest fixtures available to all test modules.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing.container_vocabulary import ContainerVocabulary
import mlflow


@pytest.fixture
def synthetic_multi_container_data():
    """
    Generate synthetic time series for 3 containers.

    Returns:
        DataFrame with columns: timestamp, container_labels, value, container_id
    """
    containers = ["webapp", "database", "redis"]
    timestamps = pd.date_range("2024-01-01", periods=2000, freq="15s")

    data = []
    for container in containers:
        # Different patterns per container to ensure they're distinguishable
        if container == "webapp":
            # Higher frequency oscillation
            values = 50 + 20 * np.sin(np.linspace(0, 10 * np.pi, 2000))
        elif container == "database":
            # Lower frequency, lower amplitude
            values = 30 + 10 * np.sin(np.linspace(0, 5 * np.pi, 2000))
        else:  # redis
            # Different phase and frequency
            values = 70 + 15 * np.cos(np.linspace(0, 8 * np.pi, 2000))

        for ts, val in zip(timestamps, values):
            data.append(
                {
                    "timestamp": ts,
                    "container_labels": f"container={container},pod=test-{container},namespace=default",
                    "value": val + np.random.normal(0, 1),  # Add noise
                    "container_id": f"/kubepods/test-{container}",
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_single_container_data():
    """
    Generate synthetic time series for a single container.

    Returns:
        DataFrame with columns: timestamp, container_labels, value, container_id
    """
    container = "webapp"
    timestamps = pd.date_range("2024-01-01", periods=2000, freq="15s")

    data = []
    # Sinusoidal pattern with trend and noise
    values = (
        50 + 20 * np.sin(np.linspace(0, 10 * np.pi, 2000)) + np.linspace(0, 10, 2000)
    )

    for ts, val in zip(timestamps, values):
        data.append(
            {
                "timestamp": ts,
                "container_labels": f"container={container},pod=test-{container},namespace=default",
                "value": val + np.random.normal(0, 1),
                "container_id": f"/kubepods/test-{container}",
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def container_vocab():
    """
    Create a sample container vocabulary.

    Returns:
        ContainerVocabulary with 3 containers mapped
    """
    vocab = ContainerVocabulary()
    vocab.add_container("webapp")
    vocab.add_container("database")
    vocab.add_container("redis")
    return vocab


@pytest.fixture
def small_container_vocab():
    """
    Create a small container vocabulary for quick tests.

    Returns:
        ContainerVocabulary with 2 containers
    """
    vocab = ContainerVocabulary()
    vocab.add_container("webapp")
    vocab.add_container("database")
    return vocab


@pytest.fixture
def sample_config():
    """
    Create a sample experiment configuration.

    Returns:
        ExperimentConfig for testing
    """
    from src.config.base_config import (
        ExperimentConfig,
        DataConfig,
        ModelConfig,
        TrainingConfig,
    )

    config = ExperimentConfig(
        metric_name="cpu",
        container_name="webapp",
        data=DataConfig(window_size=240, prediction_horizons=[20, 60, 120], stride=4),
        model=ModelConfig(
            model_type="lstm",
            input_size=8,
            hidden_size=32,  # Smaller for tests
            num_layers=2,
            dropout=0.2,
        ),
        training=TrainingConfig(
            epochs=5, batch_size=16, learning_rate=0.001  # Fewer epochs for tests
        ),
    )

    return config


@pytest.fixture
def multi_container_config():
    """
    Create a configuration for multi-container training.

    Returns:
        ExperimentConfig configured for multi-container mode
    """
    from src.config.base_config import (
        ExperimentConfig,
        DataConfig,
        ModelConfig,
        TrainingConfig,
    )

    config = ExperimentConfig(
        metric_name="cpu",
        container_name="multi",
        data=DataConfig(window_size=240, prediction_horizons=[20, 60, 120], stride=4),
        model=ModelConfig(
            model_type="lstm",
            input_size=8,
            hidden_size=32,
            num_layers=2,
            dropout=0.2,
            use_container_embeddings=True,
            num_containers=3,
            container_embedding_dim=8,
        ),
        training=TrainingConfig(epochs=5, batch_size=16, learning_rate=0.001),
    )

    return config


@pytest.fixture
def realistic_feature_count():
    """
    Return typical feature count after full feature engineering.

    This matches the feature count from production after applying:
    - Temporal features: hour_sin, hour_cos, day_sin, day_cos (4)
    - Lag features: lag_1, lag_4, lag_16 (3)
    - Rolling features: rolling_4, rolling_16, rolling_60 (3)
    - Original value (1)

    Total depends on config, but typically ~11-27 features.
    """
    return 27  # Conservative estimate for typical config


@pytest.fixture
def multi_container_training_data(realistic_feature_count):
    """
    Create realistic multi-container training data matching production shapes.

    Returns:
        Tuple of (X, y_dict, container_ids) with realistic dimensions
    """
    num_containers = 3
    samples_per_container = 500
    total_samples = num_containers * samples_per_container
    window_size = 240

    # Create features matching production feature engineering
    X = np.random.randn(total_samples, window_size, realistic_feature_count)

    # Create container IDs - repeat pattern for each container
    container_ids = np.repeat(range(num_containers), samples_per_container)

    # Create multi-horizon targets
    y_dict = {
        20: np.random.randn(total_samples, 20),
        60: np.random.randn(total_samples, 60),
        120: np.random.randn(total_samples, 120),
    }

    return X, y_dict, container_ids


@pytest.fixture
def single_container_training_data(realistic_feature_count):
    """
    Create realistic single-container training data.

    Returns:
        Tuple of (X, y_dict) with realistic dimensions
    """
    num_samples = 1000
    window_size = 240

    X = np.random.randn(num_samples, window_size, realistic_feature_count)

    y_dict = {
        20: np.random.randn(num_samples, 20),
        60: np.random.randn(num_samples, 60),
        120: np.random.randn(num_samples, 120),
    }

    return X, y_dict


@pytest.fixture
def engineered_feature_dataframe():
    """
    Create a DataFrame with engineered features matching train_local.py behavior.

    Returns:
        DataFrame with temporal, lag, and rolling features added
    """
    from src.preprocessing.sliding_windows import (
        add_temporal_features,
        add_lag_features,
        add_rolling_features,
    )

    # Create base data
    timestamps = pd.date_range("2024-01-01", periods=1000, freq="15s")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": 50
            + 20 * np.sin(np.linspace(0, 10 * np.pi, 1000))
            + np.random.randn(1000),
            "container_name": ["webapp"] * 1000,
        }
    )

    # Apply feature engineering (matching train_local.py)
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = df.dropna().reset_index(drop=True)

    return df


@pytest.fixture(autouse=True)
def cleanup_mlflow():
    """
    Automatically clean up MLflow runs after each test.

    This fixture runs before and after every test to ensure
    no MLflow runs are left active.
    """
    # Before test: ensure clean state
    active_run = mlflow.active_run()
    if active_run:
        mlflow.end_run()

    yield  # Run the test

    # After test: clean up any runs started during test
    active_run = mlflow.active_run()
    if active_run:
        try:
            mlflow.end_run()
        except Exception:
            # Ignore errors if the tracking directory was already cleaned up
            pass
