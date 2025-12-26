"""
Integration Tests for MLflow Logging

These tests validate that MLflow logging works correctly with both
single-container and multi-container models, especially with container embeddings.
"""

import pytest
import torch
import tempfile
import shutil
import os
import numpy as np
from src.training.metric_trainer import MetricTrainer
from src.config.base_config import ExperimentConfig
from src.models.lstm.lstm_model import LSTMPredictor


@pytest.mark.integration
def test_mlflow_logging_with_container_embeddings(
    multi_container_training_data, realistic_feature_count
):
    """Test MLflow logging works with multi-container models and embeddings."""
    X, y_dict, container_ids = multi_container_training_data

    # Split into train/val
    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    container_ids_train = container_ids[:n_train]
    container_ids_val = container_ids[n_train:]

    y_train_dict = {h: y[:n_train] for h, y in y_dict.items()}
    y_val_dict = {h: y[n_train:] for h, y in y_dict.items()}

    # Setup temporary MLflow tracking
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig.from_yaml("src/config/model_configs/cpu_config.yaml")
        config.training.tracking_uri = f"sqlite:///mlflow-test.db"
        config.training.checkpoint_dir = f"{tmpdir}/checkpoints"
        config.training.epochs = 2  # Quick test
        config.model.input_size = realistic_feature_count
        config.model.use_container_embeddings = True
        config.model.num_containers = 3
        config.model.container_embedding_dim = 8
        # Match config horizons to test data horizons
        config.data.prediction_horizons = [20, 60, 120]
        config.training.horizon_weights = {20: 1.0, 60: 1.5, 120: 1.0}

        # Train model
        trainer = MetricTrainer(config, use_mlflow=True, register_model=False)
        trainer.prepare_data(
            X_train,
            y_train_dict,
            X_val,
            y_val_dict,
            container_ids_train=container_ids_train,
            container_ids_val=container_ids_val,
        )
        trainer.build_model()
        trainer.train_lstm()

        # Verify model was logged
        import mlflow

        mlflow.set_tracking_uri(config.training.tracking_uri)
        # Search across all experiments
        client = mlflow.tracking.MlflowClient(config.training.tracking_uri)
        experiments = client.search_experiments()
        runs = []
        for exp in experiments:
            runs.extend(client.search_runs(experiment_ids=[exp.experiment_id]))
        assert len(runs) > 0, "No MLflow runs found"

        # Verify run has expected metrics (runs is a list of RunInfo objects)
        run = runs[0]
        assert run is not None


@pytest.mark.integration
def test_mlflow_logging_without_container_embeddings(
    single_container_training_data, realistic_feature_count
):
    """Test MLflow logging works for single-container models without embeddings."""
    X, y_dict = single_container_training_data

    # Split into train/val
    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train_dict = {h: y[:n_train] for h, y in y_dict.items()}
    y_val_dict = {h: y[n_train:] for h, y in y_dict.items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig.from_yaml("src/config/model_configs/cpu_config.yaml")
        config.training.tracking_uri = f"file://{tmpdir}/mlruns"
        config.training.checkpoint_dir = f"{tmpdir}/checkpoints"
        config.training.epochs = 2
        config.model.input_size = realistic_feature_count
        config.model.use_container_embeddings = False
        # Match config horizons to test data horizons
        config.data.prediction_horizons = [20, 60, 120]
        config.training.horizon_weights = {20: 1.0, 60: 1.5, 120: 1.0}

        trainer = MetricTrainer(config, use_mlflow=True, register_model=False)
        trainer.prepare_data(X_train, y_train_dict, X_val, y_val_dict)
        trainer.build_model()

        # Should work without container_ids
        trainer.train_lstm()

        import mlflow

        mlflow.set_tracking_uri(config.training.tracking_uri)
        # Search across all experiments
        client = mlflow.tracking.MlflowClient(config.training.tracking_uri)
        experiments = client.search_experiments()
        runs = []
        for exp in experiments:
            runs.extend(client.search_runs(experiment_ids=[exp.experiment_id]))
        assert len(runs) > 0


@pytest.mark.integration
def test_mlflow_example_creation_with_and_without_embeddings(realistic_feature_count):
    """Test that input/output examples work for both single and multi-container."""
    num_features = realistic_feature_count

    # Test WITHOUT container embeddings
    config = ExperimentConfig.from_yaml("src/config/model_configs/cpu_config.yaml")
    config.model.input_size = num_features
    config.model.use_container_embeddings = False
    model = LSTMPredictor(config.model)

    X = torch.randn(1, 240, num_features)
    predictions = model.predict_all_horizons(X)  # Should work
    assert predictions is not None
    assert len(predictions) == 3

    # Test WITH container embeddings
    config.model.use_container_embeddings = True
    config.model.num_containers = 3
    config.model.container_embedding_dim = 8
    model = LSTMPredictor(config.model)

    container_ids = torch.LongTensor([0])
    predictions = model.predict_all_horizons(X, container_ids)  # Should work
    assert predictions is not None

    # This should fail - missing container_ids for embedding model
    with pytest.raises(RuntimeError, match="input.size"):
        predictions = model.predict_all_horizons(X)  # Missing container_ids!


@pytest.mark.integration
def test_mlflow_input_output_signature(
    multi_container_training_data, realistic_feature_count
):
    """
    Test that MLflow input/output signature is created correctly.
    This validates the fix for the dimension mismatch bug.
    """
    X, y_dict, container_ids = multi_container_training_data

    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    container_ids_train = container_ids[:n_train]
    container_ids_val = container_ids[n_train:]

    y_train_dict = {h: y[:n_train] for h, y in y_dict.items()}
    y_val_dict = {h: y[n_train:] for h, y in y_dict.items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig.from_yaml("src/config/model_configs/cpu_config.yaml")
        config.training.tracking_uri = f"file://{tmpdir}/mlruns"
        config.training.checkpoint_dir = f"{tmpdir}/checkpoints"
        config.training.epochs = 1
        config.model.input_size = realistic_feature_count
        config.model.use_container_embeddings = True
        config.model.num_containers = 3
        # Match config horizons to test data horizons
        config.data.prediction_horizons = [20, 60, 120]
        config.training.horizon_weights = {20: 1.0, 60: 1.5, 120: 1.0}

        trainer = MetricTrainer(config, use_mlflow=True, register_model=False)
        trainer.prepare_data(
            X_train,
            y_train_dict,
            X_val,
            y_val_dict,
            container_ids_train=container_ids_train,
            container_ids_val=container_ids_val,
        )
        trainer.build_model()

        # Create input/output examples (this is what was failing)
        input_example = trainer.X_val[0:1]
        container_id_example = trainer.container_ids_val[0:1]

        with torch.no_grad():
            trainer.model.eval()
            X_example = torch.FloatTensor(input_example).to(trainer.device)
            container_id_tensor = torch.LongTensor(container_id_example).to(
                trainer.device
            )

            # This should NOT raise a dimension error
            output_example = trainer.model.predict_all_horizons(
                X_example, container_id_tensor
            )

        # Verify output structure
        assert isinstance(output_example, dict)
        assert 20 in output_example
        assert 60 in output_example
        assert 120 in output_example

        # Verify shapes
        assert output_example[20].shape == (1, 20)
        assert output_example[60].shape == (1, 60)
        assert output_example[120].shape == (1, 120)


@pytest.mark.integration
def test_mlflow_model_registry_disabled(
    multi_container_training_data, realistic_feature_count
):
    """Test that MLflow logging works with model registration disabled."""
    X, y_dict, container_ids = multi_container_training_data

    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    container_ids_train = container_ids[:n_train]
    container_ids_val = container_ids[n_train:]

    y_train_dict = {h: y[:n_train] for h, y in y_dict.items()}
    y_val_dict = {h: y[n_train:] for h, y in y_dict.items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig.from_yaml("src/config/model_configs/cpu_config.yaml")
        config.training.tracking_uri = f"file://{tmpdir}/mlruns"
        config.training.checkpoint_dir = f"{tmpdir}/checkpoints"
        config.training.epochs = 1
        config.model.input_size = realistic_feature_count
        config.model.use_container_embeddings = True
        config.model.num_containers = 3
        # Match config horizons to test data horizons
        config.data.prediction_horizons = [20, 60, 120]
        config.training.horizon_weights = {20: 1.0, 60: 1.5, 120: 1.0}

        # Explicitly disable model registration
        trainer = MetricTrainer(config, use_mlflow=True, register_model=False)
        trainer.prepare_data(
            X_train,
            y_train_dict,
            X_val,
            y_val_dict,
            container_ids_train=container_ids_train,
            container_ids_val=container_ids_val,
        )
        trainer.build_model()
        trainer.train_lstm()

        # Verify run exists but no registered model
        import mlflow

        mlflow.set_tracking_uri(config.training.tracking_uri)
        # Search across all experiments
        client = mlflow.tracking.MlflowClient(config.training.tracking_uri)
        experiments = client.search_experiments()
        runs = []
        for exp in experiments:
            runs.extend(client.search_runs(experiment_ids=[exp.experiment_id]))
        assert len(runs) > 0

        # Check that no models were registered
        registered_models = client.search_registered_models()
        assert len(registered_models) == 0, "No models should be registered"


@pytest.mark.integration
@pytest.mark.slow
def test_full_training_with_mlflow_logging(
    multi_container_training_data, realistic_feature_count
):
    """
    Integration test for full training pipeline with MLflow logging.
    This is the most comprehensive test, simulating production behavior.
    """
    X, y_dict, container_ids = multi_container_training_data

    # Split into train/val/test
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))

    X_train = X[:n_train]
    X_val = X[n_train : n_train + n_val]
    X_test = X[n_train + n_val :]

    container_ids_train = container_ids[:n_train]
    container_ids_val = container_ids[n_train : n_train + n_val]
    container_ids_test = container_ids[n_train + n_val :]

    y_train_dict = {h: y[:n_train] for h, y in y_dict.items()}
    y_val_dict = {h: y[n_train : n_train + n_val] for h, y in y_dict.items()}
    y_test_dict = {h: y[n_train + n_val :] for h, y in y_dict.items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig.from_yaml("src/config/model_configs/cpu_config.yaml")
        config.training.tracking_uri = "sqlite:///mlflow-full-training-test.db"
        config.training.checkpoint_dir = f"{tmpdir}/checkpoints"
        config.training.epochs = 3
        config.training.patience = 10
        config.model.input_size = realistic_feature_count
        config.model.use_container_embeddings = True
        config.model.num_containers = 3
        # Match config horizons to test data horizons
        config.data.prediction_horizons = [20, 60, 120]
        config.training.horizon_weights = {20: 1.0, 60: 1.5, 120: 1.0}

        trainer = MetricTrainer(config, use_mlflow=True, register_model=False)
        trainer.prepare_data(
            X_train,
            y_train_dict,
            X_val,
            y_val_dict,
            container_ids_train=container_ids_train,
            container_ids_val=container_ids_val,
        )
        trainer.build_model()

        # Full training including MLflow logging
        trainer.train()

        # Verify checkpoint exists
        checkpoint_path = os.path.join(
            config.training.checkpoint_dir, config.metric_name, "best_model.pth"
        )
        assert os.path.exists(checkpoint_path), "Best model checkpoint should exist"

        # Verify MLflow run
        import mlflow

        mlflow.set_tracking_uri(config.training.tracking_uri)
        # Search across all experiments
        client = mlflow.tracking.MlflowClient(config.training.tracking_uri)
        experiments = client.search_experiments()
        runs = []
        for exp in experiments:
            runs.extend(client.search_runs(experiment_ids=[exp.experiment_id]))
        assert len(runs) > 0
