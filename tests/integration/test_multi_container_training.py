"""
Integration Tests: Multi-Container Training
End-to-end tests for multi-container model training.
"""

import pytest
import torch
import numpy as np
from src.models.lstm.lstm_model import LSTMPredictor
from src.training.data_loaders import create_data_loaders
from src.preprocessing.sliding_windows import MultiHorizonWindowGenerator
from src.preprocessing.metric_specific import (
    extract_container_name,
    build_container_vocabulary,
    add_container_ids,
)


@pytest.mark.integration
def test_lstm_with_container_embeddings(multi_container_config):
    """Test LSTM forward pass with container embeddings."""
    config = multi_container_config

    # Create model
    model = LSTMPredictor(config.model)

    # Verify embeddings are configured
    assert model.use_container_embeddings is True
    assert hasattr(model, "container_embedding")

    # Test forward pass
    batch_size = 8
    seq_len = 240
    features = 8

    X = torch.randn(batch_size, seq_len, features)
    container_ids = torch.randint(0, config.model.num_containers, (batch_size,))

    # Forward pass
    features_out = model.forward(X, container_ids)
    assert features_out.shape[0] == batch_size

    # Test predictions
    predictions = model.predict_all_horizons(X, container_ids)

    assert len(predictions) == len(config.data.prediction_horizons)
    for horizon in config.data.prediction_horizons:
        assert horizon in predictions
        assert predictions[horizon].shape == (batch_size, horizon)


@pytest.mark.integration
def test_data_loader_with_container_ids(
    synthetic_multi_container_data, container_vocab
):
    """Test DataLoader returns correct format with container IDs."""
    df = synthetic_multi_container_data.copy()

    # Extract and add container IDs
    df["container_name"] = df["container_labels"].apply(extract_container_name)
    df = add_container_ids(df, container_vocab)

    # Create windows
    generator = MultiHorizonWindowGenerator(
        window_size=100, prediction_horizons=[20, 60], stride=10
    )

    X, y_dict, window_container_ids, metadata = (
        generator.create_multi_container_sequences(
            df["value"].values, df["container_id"].values, df["timestamp"]
        )
    )

    # Add dummy features dimension
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split data
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))

    X_train = X[:n_train]
    X_val = X[n_train : n_train + n_val]

    y_train_dict = {h: y[:n_train] for h, y in y_dict.items()}
    y_val_dict = {h: y[n_train : n_train + n_val] for h, y in y_dict.items()}

    container_ids_train = window_container_ids[:n_train]
    container_ids_val = window_container_ids[n_train : n_train + n_val]

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train,
        y_train_dict,
        X_val,
        y_val_dict,
        container_ids_train=container_ids_train,
        container_ids_val=container_ids_val,
        batch_size=16,
        num_workers=0,  # Avoid multiprocessing in tests
    )

    # Test iteration
    for batch_data in train_loader:
        assert len(batch_data) == 3, "Expected 3-tuple (X, y_dict, container_ids)"

        batch_X, batch_y_dict, batch_container_ids = batch_data

        assert isinstance(batch_X, torch.Tensor)
        assert isinstance(batch_y_dict, dict)
        assert isinstance(batch_container_ids, torch.Tensor)

        assert batch_X.shape[0] == batch_container_ids.shape[0]
        assert len(batch_y_dict) == 2  # Two horizons

        break  # Only test first batch


@pytest.mark.integration
@pytest.mark.slow
def test_end_to_end_multi_container_training(
    synthetic_multi_container_data, multi_container_config
):
    """
    End-to-end test: preprocess data, create windows, train model.

    This is a minimal training test (few epochs) to verify the pipeline works.
    """
    df = synthetic_multi_container_data.copy()
    config = multi_container_config

    # Preprocess
    df["container_name"] = df["container_labels"].apply(extract_container_name)
    vocab = build_container_vocabulary(df)
    df = add_container_ids(df, vocab)

    # Create windows
    generator = MultiHorizonWindowGenerator(
        window_size=config.data.window_size,
        prediction_horizons=config.data.prediction_horizons,
        stride=config.data.stride,
    )

    X, y_dict, window_container_ids, metadata = (
        generator.create_multi_container_sequences(
            df["value"].values, df["container_id"].values, df["timestamp"]
        )
    )

    # Add features dimension
    X = X.reshape(X.shape[0], X.shape[1], 1)
    config.model.input_size = 1

    # Split
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))

    X_train = X[:n_train]
    X_val = X[n_train : n_train + n_val]

    y_train_dict = {h: y[:n_train] for h, y in y_dict.items()}
    y_val_dict = {h: y[n_train : n_train + n_val] for h, y in y_dict.items()}

    container_ids_train = window_container_ids[:n_train]
    container_ids_val = window_container_ids[n_train : n_train + n_val]

    # Create loaders
    train_loader, val_loader = create_data_loaders(
        X_train,
        y_train_dict,
        X_val,
        y_val_dict,
        container_ids_train=container_ids_train,
        container_ids_val=container_ids_val,
        batch_size=config.training.batch_size,
        num_workers=0,
    )

    # Create model
    model = LSTMPredictor(config.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    criterion = torch.nn.MSELoss()

    # Train for a few epochs
    model.train()
    initial_loss = None
    final_loss = None

    for epoch in range(3):  # Just 3 epochs for testing
        epoch_loss = 0
        batch_count = 0

        for batch_data in train_loader:
            batch_X, batch_y_dict, batch_container_ids = batch_data

            optimizer.zero_grad()

            # Forward pass
            predictions = model.predict_all_horizons(batch_X, batch_container_ids)

            # Compute loss (simplified - just use one horizon)
            loss = criterion(predictions[20], batch_y_dict[20])

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count

        if epoch == 0:
            initial_loss = avg_loss
        if epoch == 2:
            final_loss = avg_loss

        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

    # Verify loss decreased
    assert (
        final_loss < initial_loss
    ), f"Training did not improve: initial={initial_loss:.4f}, final={final_loss:.4f}"

    print(f"✓ Training improved: {initial_loss:.4f} -> {final_loss:.4f}")


@pytest.mark.integration
def test_single_container_backward_compatibility(
    synthetic_single_container_data, sample_config
):
    """Test that single-container mode still works."""
    from src.preprocessing.sliding_windows import (
        create_multi_horizon_features_and_windows,
    )

    df = synthetic_single_container_data.copy()
    df["container_name"] = "webapp"
    df["metric_name"] = "container_cpu_rate"

    # Create windows (old way)
    X, y_dict, feature_names, metadata = create_multi_horizon_features_and_windows(
        df,
        container_name="webapp",
        metric_name="container_cpu_rate",
        window_size_minutes=10,
        prediction_horizon_minutes=[5, 15],
    )

    # Split
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))

    X_train = X[:n_train]
    X_val = X[n_train : n_train + n_val]

    y_train_dict = {h: y[:n_train] for h, y in y_dict.items()}
    y_val_dict = {h: y[n_train : n_train + n_val] for h, y in y_dict.items()}

    # Create loaders WITHOUT container IDs
    train_loader, val_loader = create_data_loaders(
        X_train, y_train_dict, X_val, y_val_dict, batch_size=16, num_workers=0
    )

    # Test iteration
    for batch_data in train_loader:
        assert len(batch_data) == 2, "Expected 2-tuple (X, y_dict) for single-container"

        batch_X, batch_y_dict = batch_data

        assert isinstance(batch_X, torch.Tensor)
        assert isinstance(batch_y_dict, dict)

        break

    print("✓ Single-container mode works correctly")


@pytest.mark.integration
@pytest.mark.slow
def test_multi_container_training_with_realistic_features(
    multi_container_training_data, realistic_feature_count, multi_container_config
):
    """
    Test multi-container training with realistic feature counts.

    This test uses realistic data dimensions matching production (27 features)
    instead of simplified data (1 feature). This would have caught the MLflow
    dimension mismatch bug.
    """
    X, y_dict, container_ids = multi_container_training_data
    config = multi_container_config

    # Update config to match realistic feature count
    config.model.input_size = realistic_feature_count
    config.model.num_containers = 3

    # Split data
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))

    X_train = X[:n_train]
    X_val = X[n_train : n_train + n_val]

    y_train_dict = {h: y[:n_train] for h, y in y_dict.items()}
    y_val_dict = {h: y[n_train : n_train + n_val] for h, y in y_dict.items()}

    container_ids_train = container_ids[:n_train]
    container_ids_val = container_ids[n_train : n_train + n_val]

    # Create loaders
    train_loader, val_loader = create_data_loaders(
        X_train,
        y_train_dict,
        X_val,
        y_val_dict,
        container_ids_train=container_ids_train,
        container_ids_val=container_ids_val,
        batch_size=config.training.batch_size,
        num_workers=0,
    )

    # Create model with realistic dimensions
    model = LSTMPredictor(config.model)

    # Verify LSTM input size includes embeddings
    expected_lstm_input = realistic_feature_count + config.model.container_embedding_dim
    assert model.lstm.input_size == expected_lstm_input, (
        f"LSTM should expect {expected_lstm_input} features "
        f"({realistic_feature_count} + {config.model.container_embedding_dim} embedding), "
        f"got {model.lstm.input_size}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    criterion = torch.nn.MSELoss()

    # Train for a few epochs
    model.train()
    initial_loss = None
    final_loss = None

    for epoch in range(3):
        epoch_loss = 0
        batch_count = 0

        for batch_data in train_loader:
            batch_X, batch_y_dict, batch_container_ids = batch_data

            optimizer.zero_grad()

            # Forward pass with container IDs
            predictions = model.predict_all_horizons(batch_X, batch_container_ids)

            # Compute loss across all horizons
            loss = sum(
                criterion(predictions[h], batch_y_dict[h]) for h in y_dict.keys()
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count

        if epoch == 0:
            initial_loss = avg_loss
        if epoch == 2:
            final_loss = avg_loss

    # Verify training improved
    assert (
        final_loss < initial_loss
    ), f"Training did not improve: initial={initial_loss:.4f}, final={final_loss:.4f}"

    # Test that we can create MLflow examples without dimension errors
    # This is what was failing before the bug fix
    with torch.no_grad():
        model.eval()
        X_example = torch.FloatTensor(X_val[0:1])
        container_id_example = torch.LongTensor(container_ids_val[0:1])

        # This should NOT raise a dimension error
        output_example = model.predict_all_horizons(X_example, container_id_example)

        # Verify output structure
        assert isinstance(output_example, dict)
        assert len(output_example) == len(y_dict)
        for horizon, pred in output_example.items():
            assert pred.shape == (1, horizon)

    print(
        f"✓ Multi-container training with realistic features: {initial_loss:.4f} -> {final_loss:.4f}"
    )
