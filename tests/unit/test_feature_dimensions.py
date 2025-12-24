"""
Unit Tests for Feature Dimension Validation

These tests validate that model input dimensions match the actual feature count
after feature engineering, preventing dimension mismatch errors.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from src.models.lstm.lstm_model import LSTMPredictor
from src.config.base_config import ExperimentConfig, ModelConfig
from src.preprocessing.sliding_windows import (
    add_temporal_features,
    add_lag_features,
    add_rolling_features,
)


@pytest.mark.unit
def test_engineered_features_match_model_input_size(engineered_feature_dataframe):
    """Validate that feature engineering creates expected dimensions."""
    df = engineered_feature_dataframe

    # Get feature columns (excluding metadata)
    feature_cols = [
        col for col in df.columns if col not in ["timestamp", "value", "container_name"]
    ]

    expected_features = len(feature_cols)

    # Build model with correct input size
    config = ExperimentConfig.from_yaml("src/config/model_configs/cpu_config.yaml")
    config.model.input_size = expected_features

    model = LSTMPredictor(config.model)

    # Validate the dimension consistency
    assert model.input_size == expected_features, (
        f"Model input_size ({model.input_size}) should match "
        f"feature count ({expected_features})"
    )

    # Test forward pass with realistic shape
    batch_size = 4
    window_size = 240
    X = torch.randn(batch_size, window_size, expected_features)
    predictions = model.predict_all_horizons(X)

    assert predictions is not None
    assert len(predictions) == 3  # Three horizons
    assert all(isinstance(v, torch.Tensor) for v in predictions.values())


@pytest.mark.unit
def test_temporal_features_count():
    """Validate temporal feature engineering adds expected number of features."""
    # Create base data
    timestamps = pd.date_range("2024-01-01", periods=100, freq="15s")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": np.random.randn(100),
            "container_name": ["webapp"] * 100,
        }
    )

    original_cols = set(df.columns)

    # Add temporal features
    df = add_temporal_features(df)

    new_cols = set(df.columns) - original_cols

    # Verify common temporal features exist
    assert "hour_sin" in new_cols
    assert "hour_cos" in new_cols
    assert any("dow" in col for col in new_cols), "Expected day-of-week features"
    assert (
        len(new_cols) >= 4
    ), f"Expected at least 4 temporal features, got {len(new_cols)}"


@pytest.mark.unit
def test_lag_features_count():
    """Validate lag feature engineering adds expected number of features."""
    timestamps = pd.date_range("2024-01-01", periods=100, freq="15s")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": np.random.randn(100),
            "container_name": ["webapp"] * 100,
        }
    )

    original_cols = set(df.columns)

    # Add lag features
    df = add_lag_features(df)
    df = df.dropna()

    new_cols = set(df.columns) - original_cols

    # Expected lag features: value_lag_1, value_lag_4, etc.
    assert any(
        "lag_1" in col for col in new_cols
    ), f"Expected lag_1 feature, got {new_cols}"
    assert any(
        "lag_4" in col for col in new_cols
    ), f"Expected lag_4 feature, got {new_cols}"
    assert len(new_cols) >= 2, f"Expected at least 2 lag features, got {len(new_cols)}"
    assert len(df) < 100  # Some rows dropped due to lag


@pytest.mark.unit
def test_rolling_features_count():
    """Validate rolling feature engineering adds expected number of features."""
    timestamps = pd.date_range("2024-01-01", periods=100, freq="15s")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": np.random.randn(100),
            "container_name": ["webapp"] * 100,
        }
    )

    original_cols = set(df.columns)

    # Add rolling features
    df = add_rolling_features(df)
    df = df.dropna()

    new_cols = set(df.columns) - original_cols

    # Expected rolling features: value_roll_mean_4, value_roll_std_4, etc.
    assert any(
        "roll" in col and "4" in col for col in new_cols
    ), f"Expected rolling window features, got {new_cols}"
    assert (
        len(new_cols) >= 3
    ), f"Expected at least 3 rolling features, got {len(new_cols)}"
    assert len(df) < 100  # Some rows dropped due to rolling window


@pytest.mark.unit
def test_multi_container_embeddings_dimension_calculation(realistic_feature_count):
    """Validate LSTM input size calculation with container embeddings."""
    num_features = realistic_feature_count  # 27
    embedding_dim = 8

    config = ExperimentConfig.from_yaml("src/config/model_configs/cpu_config.yaml")
    config.model.input_size = num_features
    config.model.use_container_embeddings = True
    config.model.container_embedding_dim = embedding_dim
    config.model.num_containers = 3

    model = LSTMPredictor(config.model)

    # The LSTM should expect features + embeddings
    expected_lstm_input = num_features + embedding_dim
    assert model.lstm.input_size == expected_lstm_input, (
        f"LSTM input size should be features ({num_features}) + "
        f"embeddings ({embedding_dim}) = {expected_lstm_input}, "
        f"got {model.lstm.input_size}"
    )

    # Test with actual data
    batch_size = 4
    window_size = 240
    X = torch.randn(batch_size, window_size, num_features)
    container_ids = torch.LongTensor([0, 1, 2, 0])

    # After concatenating embeddings, input should be correct size
    predictions = model.predict_all_horizons(X, container_ids)
    assert predictions is not None
    assert 20 in predictions
    assert 60 in predictions
    assert 120 in predictions


@pytest.mark.unit
def test_model_fails_with_wrong_dimensions(realistic_feature_count):
    """Validate that model raises error with incorrect input dimensions."""
    num_features = realistic_feature_count
    embedding_dim = 8

    config = ExperimentConfig.from_yaml("src/config/model_configs/cpu_config.yaml")
    config.model.input_size = num_features
    config.model.use_container_embeddings = True
    config.model.container_embedding_dim = embedding_dim
    config.model.num_containers = 3

    model = LSTMPredictor(config.model)

    # Create data with WRONG number of features
    wrong_features = (
        num_features  # Should be num_features, not num_features + embedding
    )
    X = torch.randn(4, 240, wrong_features)

    # This should FAIL when container_ids are not provided
    # because embeddings won't be concatenated
    with pytest.raises(RuntimeError, match="input.size"):
        model.predict_all_horizons(X)  # Missing container_ids!


@pytest.mark.unit
def test_single_container_no_embeddings(realistic_feature_count):
    """Validate single-container model works without embeddings."""
    num_features = realistic_feature_count

    config = ExperimentConfig.from_yaml("src/config/model_configs/cpu_config.yaml")
    config.model.input_size = num_features
    config.model.use_container_embeddings = False

    model = LSTMPredictor(config.model)

    # LSTM should expect exactly num_features (no embeddings)
    assert model.lstm.input_size == num_features

    # Test with data
    X = torch.randn(4, 240, num_features)
    predictions = model.predict_all_horizons(X)

    assert predictions is not None
    assert len(predictions) == 3


@pytest.mark.unit
def test_feature_count_consistency_across_preprocessing():
    """
    Validate that feature count remains consistent through the preprocessing pipeline.
    This mimics the train_local.py workflow.
    """
    # Create raw data
    timestamps = pd.date_range("2024-01-01", periods=1000, freq="15s")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": 50 + 20 * np.sin(np.linspace(0, 10 * np.pi, 1000)),
            "container_name": ["webapp"] * 1000,
        }
    )

    # Apply full feature engineering pipeline (matching train_local.py)
    df = add_temporal_features(df)
    feature_count_after_temporal = len(
        [c for c in df.columns if c not in ["timestamp", "value", "container_name"]]
    )

    df = add_lag_features(df)
    df = df.dropna()
    feature_count_after_lag = len(
        [c for c in df.columns if c not in ["timestamp", "value", "container_name"]]
    )

    df = add_rolling_features(df)
    df = df.dropna().reset_index(drop=True)
    final_feature_count = len(
        [c for c in df.columns if c not in ["timestamp", "value", "container_name"]]
    )

    # Verify features accumulate correctly
    assert feature_count_after_temporal >= 4  # At least temporal features
    assert feature_count_after_lag > feature_count_after_temporal  # Lags added
    assert final_feature_count > feature_count_after_lag  # Rolling added

    # Build model with final feature count
    config = ExperimentConfig.from_yaml("src/config/model_configs/cpu_config.yaml")
    config.model.input_size = final_feature_count

    model = LSTMPredictor(config.model)

    # Model should accept this number of features
    assert model.input_size == final_feature_count

    # Test forward pass
    X = torch.randn(2, 240, final_feature_count)
    predictions = model.predict_all_horizons(X)
    assert predictions is not None


@pytest.mark.unit
def test_container_embeddings_require_container_ids(realistic_feature_count):
    """
    Validate that models with container embeddings fail gracefully
    when container_ids are not provided.
    """
    num_features = realistic_feature_count

    config = ExperimentConfig.from_yaml("src/config/model_configs/cpu_config.yaml")
    config.model.input_size = num_features
    config.model.use_container_embeddings = True
    config.model.num_containers = 3
    config.model.container_embedding_dim = 8

    model = LSTMPredictor(config.model)

    X = torch.randn(4, 240, num_features)

    # Should fail with clear error when container_ids missing
    with pytest.raises(RuntimeError):
        model.predict_all_horizons(X)  # No container_ids provided

    # Should work when container_ids provided
    container_ids = torch.LongTensor([0, 1, 2, 0])
    predictions = model.predict_all_horizons(X, container_ids)
    assert predictions is not None
