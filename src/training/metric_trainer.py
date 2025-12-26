"""
Metric Trainer
Unified training pipeline for all metric models (LSTM, ARIMA, Prophet).
"""

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Optional
import os

# Local imports
from src.models.lstm.lstm_model import LSTMPredictor
from src.models.statistical.arima_model import ARIMAPredictor
from src.models.statistical.prophet_model import ProphetPredictor
from src.models.utils.losses import MultiHorizonLoss
from src.models.utils.metrics import ModelEvaluator
from src.models.utils.normalizers import TimeSeriesNormalizer
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.data_loaders import create_data_loaders
from src.config.base_config import ExperimentConfig
from src.training.mlflow_utils import (
    MLflowModelLogger,
    create_model_name,
    get_model_tags,
    get_model_description,
)


class MetricTrainer:
    """
    Train time series models for container metrics.

    Supports:
    - LSTM with multi-horizon prediction
    - ARIMA statistical baseline
    - Prophet statistical baseline
    - Early stopping, checkpointing
    - MLflow experiment tracking
    """

    def __init__(
        self,
        config: ExperimentConfig,
        use_mlflow: bool = True,
        register_model: bool = True,
    ):
        """
        Initialize trainer.

        Args:
            config: Experiment configuration
            use_mlflow: Whether to use MLflow tracking
            register_model: Whether to register model in MLflow Model Registry
        """
        self.config = config
        self.metric_name = config.metric_name
        self.model_type = config.model.model_type
        self.register_model = register_model

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # MLflow setup
        self.use_mlflow = use_mlflow
        self.mlflow_logger = None
        if use_mlflow:
            try:
                import mlflow

                mlflow.set_tracking_uri(config.training.tracking_uri)
                mlflow.set_experiment(
                    f"{config.training.experiment_name}-{self.metric_name}"
                )
                self.mlflow = mlflow

                # Initialize MLflow model logger
                self.mlflow_logger = MLflowModelLogger(
                    mlflow_module=mlflow,
                    metric_name=self.metric_name,
                    model_type=self.model_type,
                )
            except ImportError:
                print("MLflow not available, skipping tracking")
                self.use_mlflow = False

        # Initialize components
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.X_scaler = None
        self.y_scaler = None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
        }

    def prepare_data(
        self,
        X_train: np.ndarray,
        y_train_dict: Dict[int, np.ndarray],
        X_val: np.ndarray,
        y_val_dict: Dict[int, np.ndarray],
        X_test: Optional[np.ndarray] = None,
        y_test_dict: Optional[Dict[int, np.ndarray]] = None,
        container_ids_train: Optional[np.ndarray] = None,
        container_ids_val: Optional[np.ndarray] = None,
        container_ids_test: Optional[np.ndarray] = None,
    ):
        """
        Prepare and normalize data for training.

        Args:
            X_train, X_val, X_test: Input sequences
            y_train_dict, y_val_dict, y_test_dict: Target dict per horizon
            container_ids_train, container_ids_val, container_ids_test: Container IDs (for multi-container)
        """
        print(f"\nPreparing data for {self.metric_name}...")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  Horizons: {sorted(y_train_dict.keys())}")
        if container_ids_train is not None:
            print(
                f"  Multi-container mode: {len(np.unique(container_ids_train))} containers"
            )

        # Store container IDs for later use
        self.container_ids_train = container_ids_train
        self.container_ids_val = container_ids_val
        self.container_ids_test = container_ids_test

        # Normalize inputs
        self.X_scaler = TimeSeriesNormalizer(method=self.config.data.normalization)
        X_train_norm = self.X_scaler.fit_transform(X_train)
        X_val_norm = self.X_scaler.transform(X_val)

        # Normalize targets (separately for each horizon)
        self.y_scaler = {}
        y_train_norm_dict = {}
        y_val_norm_dict = {}

        for horizon in y_train_dict.keys():
            self.y_scaler[horizon] = TimeSeriesNormalizer(
                method=self.config.data.normalization
            )
            y_train_norm_dict[horizon] = (
                self.y_scaler[horizon]
                .fit_transform(y_train_dict[horizon].reshape(-1, 1))
                .reshape(y_train_dict[horizon].shape)
            )
            y_val_norm_dict[horizon] = (
                self.y_scaler[horizon]
                .transform(y_val_dict[horizon].reshape(-1, 1))
                .reshape(y_val_dict[horizon].shape)
            )

        # Store for later use
        self.X_train = X_train_norm
        self.y_train_dict = y_train_norm_dict
        self.X_val = X_val_norm
        self.y_val_dict = y_val_norm_dict

        if X_test is not None:
            self.X_test = self.X_scaler.transform(X_test)
            self.y_test_dict = {}
            for horizon in y_test_dict.keys():
                self.y_test_dict[horizon] = (
                    self.y_scaler[horizon]
                    .transform(y_test_dict[horizon].reshape(-1, 1))
                    .reshape(y_test_dict[horizon].shape)
                )

        print("  Data normalization complete!")

    def build_model(self):
        """Build model based on configuration."""
        if self.model_type == "lstm":
            # Update config with data dimensions
            self.config.model.input_size = self.X_train.shape[2]  # Number of features
            self.config.model.prediction_horizons = self.config.data.prediction_horizons

            # Create LSTM model
            self.model = LSTMPredictor(self.config.model)
            self.model.to(self.device)

            # Setup optimizer
            if self.config.training.optimizer == "adam":
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.config.training.learning_rate,
                    weight_decay=self.config.training.weight_decay,
                )
            elif self.config.training.optimizer == "sgd":
                self.optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=self.config.training.learning_rate,
                    weight_decay=self.config.training.weight_decay,
                    momentum=0.9,
                )

            # Setup loss function
            self.loss_fn = MultiHorizonLoss(
                horizon_weights=self.config.training.horizon_weights,
                base_loss=self.config.training.loss_function,
            )

            print("\nModel: LSTM")
            print(
                f"  Parameters: {self.model.get_model_info()['trainable_parameters']:,}"
            )

        elif self.model_type in ["arima", "prophet"]:
            # Statistical models - different training approach
            if self.model_type == "arima":
                self.model = ARIMAPredictor(self.config.model.__dict__)
            else:  # prophet
                self.model = ProphetPredictor(self.config.model.__dict__)

            print(f"\nModel: {self.model_type.upper()}")

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train_epoch(self, train_loader) -> float:
        """
        Train for one epoch (LSTM only).

        Args:
            train_loader: DataLoader for training data

        Returns:
            Average training loss
        """
        if self.model_type != "lstm" or self.model is None:
            raise ValueError("train_epoch is only supported for LSTM models")

        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_data in train_loader:
            # Handle both single and multi-container cases
            if len(batch_data) == 3:  # Multi-container
                batch_X, batch_y_dict, batch_container_ids = batch_data
                batch_container_ids = batch_container_ids.to(self.device)
            else:  # Single-container (backward compat)
                batch_X, batch_y_dict = batch_data
                batch_container_ids = None

            batch_X = batch_X.to(self.device)
            batch_y_dict = {h: y.to(self.device) for h, y in batch_y_dict.items()}

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model.predict_all_horizons(batch_X, batch_container_ids)

            # Calculate loss
            loss = self.loss_fn(predictions, batch_y_dict)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def validate(self, val_loader) -> float:
        """
        Validate model (LSTM only).

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                # Handle both single and multi-container cases
                if len(batch_data) == 3:  # Multi-container
                    batch_X, batch_y_dict, batch_container_ids = batch_data
                    batch_container_ids = batch_container_ids.to(self.device)
                else:  # Single-container (backward compat)
                    batch_X, batch_y_dict = batch_data
                    batch_container_ids = None

                batch_X = batch_X.to(self.device)
                batch_y_dict = {h: y.to(self.device) for h, y in batch_y_dict.items()}

                # Forward pass
                predictions = self.model.predict_all_horizons(
                    batch_X, batch_container_ids
                )

                # Calculate loss
                loss = self.loss_fn(predictions, batch_y_dict)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def train_lstm(self):
        """Train LSTM model with early stopping and checkpointing."""
        print(f"\n{'=' * 60}")
        print(f"Training LSTM for {self.metric_name}")
        print(f"{'=' * 60}")

        # Create data loaders (with container IDs if available)
        train_loader, val_loader = create_data_loaders(
            self.X_train,
            self.y_train_dict,
            self.X_val,
            self.y_val_dict,
            container_ids_train=getattr(self, "container_ids_train", None),
            container_ids_val=getattr(self, "container_ids_val", None),
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
        )

        # Setup callbacks
        early_stop = EarlyStopping(
            patience=self.config.training.patience,
            min_delta=self.config.training.min_delta,
            mode="min",
        )

        checkpoint = ModelCheckpoint(
            checkpoint_dir=os.path.join(
                self.config.training.checkpoint_dir, self.metric_name
            ),
            metric_name="val_loss",
            mode="min",
            save_every=self.config.training.save_every_n_epochs,
        )

        # Start MLflow run
        if self.use_mlflow:
            self.mlflow.start_run()
            self.mlflow.log_params(self.config.to_dict())

        # Training loop
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss = self.validate(val_loader)

            # Record history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)

            # Log to MLflow
            if self.use_mlflow:
                self.mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    step=epoch,
                )

            # Print progress
            if epoch % self.config.training.log_interval == 0 or epoch == 0:
                print(f"\nEpoch {epoch + 1}/{self.config.training.epochs}")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss: {val_loss:.6f}")

            # Checkpoint
            checkpoint(
                self.model,
                self.optimizer,
                epoch,
                val_loss,
                additional_info={
                    "X_scaler": self.X_scaler,
                    "y_scaler": self.y_scaler,
                    "config": self.config,
                },
            )

            # Early stopping
            if early_stop(val_loss, epoch):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        # Load best model
        best_checkpoint = checkpoint.load_best_model(self.model, self.optimizer)
        self.best_val_loss = best_checkpoint["best_score"]

        # Log model to MLflow
        if self.use_mlflow and self.mlflow_logger:
            print("\nLogging model to MLflow...")

            # Create input/output examples from validation data
            input_example = self.X_val[0:1]  # First validation sample
            container_id_example = None
            if self.container_ids_val is not None:
                container_id_example = self.container_ids_val[0:1]

            with torch.no_grad():
                self.model.eval()
                X_example = torch.FloatTensor(input_example).to(self.device)

                # Pass container IDs if available (for multi-container models)
                if container_id_example is not None:
                    container_id_tensor = torch.LongTensor(container_id_example).to(
                        self.device
                    )
                    output_example = self.model.predict_all_horizons(
                        X_example, container_id_tensor
                    )
                else:
                    output_example = self.model.predict_all_horizons(X_example)

                output_example = {
                    h: pred.cpu().numpy() for h, pred in output_example.items()
                }

            # Log the model
            checkpoint_path = os.path.join(
                self.config.training.checkpoint_dir, self.metric_name, "best_model.pth"
            )

            registered_model_name = None
            if self.register_model:
                registered_model_name = create_model_name(
                    metric_name=self.metric_name,
                    model_type=self.model_type,
                    container_name=self.config.container_name,
                )

            model_uri = self.mlflow_logger.log_pytorch_model(
                model=self.model,
                artifact_path="model",
                config=self.config,
                X_scaler=self.X_scaler,
                y_scaler=self.y_scaler,
                input_example=input_example,
                output_example=output_example,
                registered_model_name=registered_model_name,
                checkpoint_path=checkpoint_path,
            )

            print(f"  Model logged to: {model_uri}")

            # Add tags and description if registered
            if self.register_model and registered_model_name:
                print(f"  Model registered as: {registered_model_name}")

                # Get latest version
                client = self.mlflow.tracking.MlflowClient()
                versions = client.search_model_versions(
                    f"name='{registered_model_name}'"
                )
                if versions:
                    latest_version = max(versions, key=lambda v: int(v.version))

                    # Set tags
                    tags = get_model_tags(self.config, self.best_val_loss)
                    for key, value in tags.items():
                        client.set_model_version_tag(
                            name=registered_model_name,
                            version=latest_version.version,
                            key=key,
                            value=value,
                        )

                    print(f"  Model version: {latest_version.version}")

        # End MLflow run
        if self.use_mlflow:
            self.mlflow.end_run()

        print(f"\n{'=' * 60}")
        print("Training Complete!")
        print(f"{'=' * 60}")

    def train_statistical(self):
        """Train statistical model (ARIMA or Prophet)."""
        print(f"\n{'=' * 60}")
        print(f"Training {self.model_type.upper()} for {self.metric_name}")
        print(f"{'=' * 60}")

        # Start MLflow run
        if self.use_mlflow:
            self.mlflow.start_run()
            # Log parameters
            params = {
                "model_type": self.model_type,
                "metric_name": self.metric_name,
                "container_name": self.config.container_name,
                "train_samples": len(self.X_train),
                "prediction_horizons": self.config.data.prediction_horizons,
            }
            # Add model-specific params
            if self.model_type == "arima":
                params["arima_order"] = str(self.config.model.arima_order)
                params["arima_seasonal_order"] = str(
                    self.config.model.arima_seasonal_order
                )
            elif self.model_type == "prophet":
                params["changepoint_prior_scale"] = (
                    self.config.model.prophet_changepoint_prior_scale
                )
                params["yearly_seasonality"] = (
                    self.config.model.prophet_yearly_seasonality
                )
                params["weekly_seasonality"] = (
                    self.config.model.prophet_weekly_seasonality
                )
                params["daily_seasonality"] = (
                    self.config.model.prophet_daily_seasonality
                )

            self.mlflow.log_params(params)

        # For statistical models, use the raw sequence data
        # We'll use only the first feature if multivariate
        if self.X_train.ndim == 3:
            # Extract last value from each window as the current state
            y_sequence = self.X_train[:, -1, 0]  # Use last timestep, first feature
        else:
            y_sequence = self.X_train[:, 0]

        # Fit model
        print(f"\nFitting {self.model_type} model...")
        if self.model_type == "prophet":
            # Prophet needs timestamps
            timestamps = pd.date_range(
                start="2024-01-01", periods=len(y_sequence), freq="15S"
            )
            self.model.fit(y_sequence, timestamps=timestamps)
        else:  # ARIMA
            self.model.fit(y_sequence)

        print("Model fitted successfully!")

        # Save model
        save_dir = os.path.join(self.config.training.checkpoint_dir, self.metric_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.model_type}_model.pkl")
        self.model.save_model(save_path)
        print(f"Model saved to {save_path}")

        # Store path for later MLflow logging
        self.statistical_model_path = save_path

        # Note: Model will be logged to MLflow after evaluation is complete
        # to include evaluation metrics in the model metadata

    def train(self):
        """Main training entry point."""
        self.build_model()

        if self.model_type == "lstm":
            self.train_lstm()
        else:
            self.train_statistical()

    def evaluate(
        self, X_test: Optional[np.ndarray] = None, y_test_dict: Optional[Dict] = None
    ) -> Dict:
        """
        Evaluate trained model.

        Args:
            X_test: Test data (optional, uses stored if None)
            y_test_dict: Test targets (optional)

        Returns:
            Evaluation results
        """
        if X_test is None:
            X_test = self.X_test
            y_test_dict = self.y_test_dict

        print(f"\n{'=' * 60}")
        print(f"Evaluating {self.model_type.upper()} on {self.metric_name}")
        print(f"{'=' * 60}")

        if self.model_type == "lstm":
            # LSTM evaluation
            self.model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(self.device)

                # Pass container_ids if model uses embeddings
                if hasattr(self.model, 'use_container_embeddings') and self.model.use_container_embeddings:
                    if self.container_ids_test is not None:
                        container_ids_tensor = torch.LongTensor(self.container_ids_test).to(self.device)
                        predictions = self.model.predict_all_horizons(X_test_tensor, container_ids_tensor)
                    else:
                        raise ValueError("Model requires container_ids but container_ids_test is None")
                else:
                    predictions = self.model.predict_all_horizons(X_test_tensor)

                y_pred_dict = {h: pred.cpu().numpy() for h, pred in predictions.items()}

            # Denormalize predictions
            for horizon in y_pred_dict.keys():
                y_pred_dict[horizon] = (
                    self.y_scaler[horizon]
                    .inverse_transform(y_pred_dict[horizon].reshape(-1, 1))
                    .reshape(y_pred_dict[horizon].shape)
                )
                y_test_dict[horizon] = (
                    self.y_scaler[horizon]
                    .inverse_transform(y_test_dict[horizon].reshape(-1, 1))
                    .reshape(y_test_dict[horizon].shape)
                )

        else:
            # Statistical model evaluation
            # ARIMA/Prophet make one continuous forecast, so we compare against
            # the first test sample's targets (treating it as continuation)
            horizons = sorted(self.config.data.prediction_horizons)
            y_pred_dict_raw = self.model.predict_multi_horizon(horizons)

            # Denormalize predictions for statistical models
            y_pred_dict = {}
            for horizon in horizons:
                # Reshape to (1, horizon) for consistency, then denormalize
                pred_reshaped = y_pred_dict_raw[horizon].reshape(1, -1)
                y_pred_dict[horizon] = (
                    self.y_scaler[horizon].inverse_transform(pred_reshaped).reshape(-1)
                )

            # For statistical models, evaluate only on first test window
            # since they produce a continuous forecast, not per-window predictions
            y_test_dict_eval = {}
            for horizon in horizons:
                # Denormalize test data
                test_denorm = (
                    self.y_scaler[horizon]
                    .inverse_transform(y_test_dict[horizon].reshape(-1, 1))
                    .reshape(y_test_dict[horizon].shape)
                )
                # Use only first test window for comparison
                y_test_dict_eval[horizon] = test_denorm[0]

            # Override for evaluation
            y_test_dict = y_test_dict_eval

        # Compute metrics
        evaluator = ModelEvaluator(metric_name=self.metric_name)
        results = evaluator.horizon_analysis(y_test_dict, y_pred_dict)
        evaluator.print_evaluation_report(y_test_dict, y_pred_dict)

        # Log evaluation metrics to MLflow
        if self.use_mlflow and self.model_type in ["arima", "prophet"]:
            # Log metrics for each horizon
            for horizon_name, metrics in results.get("by_horizon", {}).items():
                for metric_name, value in metrics.items():
                    self.mlflow.log_metric(f"{horizon_name}_{metric_name}", value)

            # Log overall metrics
            if "overall" in results:
                for metric_name, value in results["overall"].items():
                    self.mlflow.log_metric(f"overall_{metric_name}", value)

            # Log statistical model to MLflow after evaluation
            if self.mlflow_logger and hasattr(self, "statistical_model_path"):
                print("\nLogging model to MLflow...")

                # Create input/output examples
                input_example = self.X_val[0:1]
                horizons = sorted(self.config.data.prediction_horizons)
                output_example_raw = self.model.predict_multi_horizon(horizons)

                # Denormalize for proper example
                output_example = {}
                for horizon in horizons:
                    pred_reshaped = output_example_raw[horizon].reshape(1, -1)
                    output_example[horizon] = (
                        self.y_scaler[horizon]
                        .inverse_transform(pred_reshaped)
                        .reshape(-1)
                    )

                # Determine if we should register
                registered_model_name = None
                if self.register_model:
                    registered_model_name = create_model_name(
                        metric_name=self.metric_name,
                        model_type=self.model_type,
                        container_name=self.config.container_name,
                    )

                # Log the model
                model_uri = self.mlflow_logger.log_statistical_model(
                    model=self.model,
                    artifact_path="model",
                    config=self.config,
                    X_scaler=self.X_scaler,
                    y_scaler=self.y_scaler,
                    model_path=self.statistical_model_path,
                    input_example=input_example,
                    output_example=output_example,
                    registered_model_name=registered_model_name,
                )

                print(f"  Model logged to: {model_uri}")

                # Add tags and description if registered
                if self.register_model and registered_model_name:
                    print(f"  Model registered as: {registered_model_name}")

                    # Get latest version
                    client = self.mlflow.tracking.MlflowClient()
                    versions = client.search_model_versions(
                        f"name='{registered_model_name}'"
                    )
                    if versions:
                        latest_version = max(versions, key=lambda v: int(v.version))

                        # Set tags
                        tags = get_model_tags(self.config)
                        # Add evaluation metrics to tags
                        if "overall" in results:
                            for metric_name, value in results["overall"].items():
                                tags[f"eval_{metric_name}"] = f"{value:.4f}"

                        for key, value in tags.items():
                            client.set_model_version_tag(
                                name=registered_model_name,
                                version=latest_version.version,
                                key=key,
                                value=value,
                            )

                        # Set description
                        description = get_model_description(self.config, results)
                        client.update_model_version(
                            name=registered_model_name,
                            version=latest_version.version,
                            description=description,
                        )

                        print(f"  Model version: {latest_version.version}")

        return results

    def finalize(self):
        """Finalize training and close MLflow run if active."""
        if self.use_mlflow:
            active_run = self.mlflow.active_run()
            if active_run:
                self.mlflow.end_run()
                print("\nMLflow run closed.")


if __name__ == "__main__":
    # Example usage
    from src.config.base_config import create_default_config

    print("Metric Trainer")
    print("=" * 60)

    # Create dummy data for testing
    n_samples = 1000
    window_size = 240
    n_features = 8
    horizons = [20, 60, 120]

    X_train = np.random.randn(700, window_size, n_features)
    X_val = np.random.randn(150, window_size, n_features)
    X_test = np.random.randn(150, window_size, n_features)

    y_train_dict = {h: np.random.randn(700, h) for h in horizons}
    y_val_dict = {h: np.random.randn(150, h) for h in horizons}
    y_test_dict = {h: np.random.randn(150, h) for h in horizons}

    # Create config
    config = create_default_config("cpu", model_type="lstm")
    config.training.epochs = 5  # Quick test
    config.training.batch_size = 32

    # Create trainer
    trainer = MetricTrainer(config, use_mlflow=False)

    # Prepare data
    trainer.prepare_data(X_train, y_train_dict, X_val, y_val_dict, X_test, y_test_dict)

    # Train
    trainer.train()

    # Evaluate
    results = trainer.evaluate()

    print("\nTrainer test complete!")
