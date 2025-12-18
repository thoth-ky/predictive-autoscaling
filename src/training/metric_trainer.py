"""
Metric Trainer
Unified training pipeline for all metric models (LSTM, ARIMA, Prophet).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import os
from tqdm import tqdm

# Local imports
from src.models.lstm.lstm_model import LSTMPredictor
from src.models.statistical.arima_model import ARIMAPredictor
from src.models.statistical.prophet_model import ProphetPredictor
from src.models.utils.losses import MultiHorizonLoss, get_loss_function
from src.models.utils.metrics import ModelEvaluator
from src.models.utils.normalizers import TimeSeriesNormalizer
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.data_loaders import create_data_loaders
from src.config.base_config import ExperimentConfig


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

    def __init__(self, config: ExperimentConfig, use_mlflow: bool = True):
        """
        Initialize trainer.

        Args:
            config: Experiment configuration
            use_mlflow: Whether to use MLflow tracking
        """
        self.config = config
        self.metric_name = config.metric_name
        self.model_type = config.model.model_type

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # MLflow setup
        self.use_mlflow = use_mlflow
        if use_mlflow:
            try:
                import mlflow
                mlflow.set_tracking_uri(config.training.tracking_uri)
                mlflow.set_experiment(f"{config.training.experiment_name}-{self.metric_name}")
                self.mlflow = mlflow
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
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
        }

    def prepare_data(self, X_train: np.ndarray, y_train_dict: Dict[int, np.ndarray],
                    X_val: np.ndarray, y_val_dict: Dict[int, np.ndarray],
                    X_test: Optional[np.ndarray] = None,
                    y_test_dict: Optional[Dict[int, np.ndarray]] = None):
        """
        Prepare and normalize data for training.

        Args:
            X_train, X_val, X_test: Input sequences
            y_train_dict, y_val_dict, y_test_dict: Target dict per horizon
        """
        print(f"\nPreparing data for {self.metric_name}...")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  Horizons: {sorted(y_train_dict.keys())}")

        # Normalize inputs
        self.X_scaler = TimeSeriesNormalizer(method=self.config.data.normalization)
        X_train_norm = self.X_scaler.fit_transform(X_train)
        X_val_norm = self.X_scaler.transform(X_val)

        # Normalize targets (separately for each horizon)
        self.y_scaler = {}
        y_train_norm_dict = {}
        y_val_norm_dict = {}

        for horizon in y_train_dict.keys():
            self.y_scaler[horizon] = TimeSeriesNormalizer(method=self.config.data.normalization)
            y_train_norm_dict[horizon] = self.y_scaler[horizon].fit_transform(
                y_train_dict[horizon].reshape(-1, 1)).reshape(y_train_dict[horizon].shape)
            y_val_norm_dict[horizon] = self.y_scaler[horizon].transform(
                y_val_dict[horizon].reshape(-1, 1)).reshape(y_val_dict[horizon].shape)

        # Store for later use
        self.X_train = X_train_norm
        self.y_train_dict = y_train_norm_dict
        self.X_val = X_val_norm
        self.y_val_dict = y_val_norm_dict

        if X_test is not None:
            self.X_test = self.X_scaler.transform(X_test)
            self.y_test_dict = {}
            for horizon in y_test_dict.keys():
                self.y_test_dict[horizon] = self.y_scaler[horizon].transform(
                    y_test_dict[horizon].reshape(-1, 1)).reshape(y_test_dict[horizon].shape)

        print("  Data normalization complete!")

    def build_model(self):
        """Build model based on configuration."""
        if self.model_type == 'lstm':
            # Update config with data dimensions
            self.config.model.input_size = self.X_train.shape[2]  # Number of features
            self.config.model.prediction_horizons = self.config.data.prediction_horizons

            # Create LSTM model
            self.model = LSTMPredictor(self.config.model)
            self.model.to(self.device)

            # Setup optimizer
            if self.config.training.optimizer == 'adam':
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.config.training.learning_rate,
                    weight_decay=self.config.training.weight_decay
                )
            elif self.config.training.optimizer == 'sgd':
                self.optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=self.config.training.learning_rate,
                    weight_decay=self.config.training.weight_decay,
                    momentum=0.9
                )

            # Setup loss function
            self.loss_fn = MultiHorizonLoss(
                horizon_weights=self.config.training.horizon_weights,
                base_loss=self.config.training.loss_function
            )

            print(f"\nModel: LSTM")
            print(f"  Parameters: {self.model.get_model_info()['trainable_parameters']:,}")

        elif self.model_type in ['arima', 'prophet']:
            # Statistical models - different training approach
            if self.model_type == 'arima':
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
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_X, batch_y_dict in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y_dict = {h: y.to(self.device) for h, y in batch_y_dict.items()}

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model.predict_all_horizons(batch_X)

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
            for batch_X, batch_y_dict in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y_dict = {h: y.to(self.device) for h, y in batch_y_dict.items()}

                # Forward pass
                predictions = self.model.predict_all_horizons(batch_X)

                # Calculate loss
                loss = self.loss_fn(predictions, batch_y_dict)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def train_lstm(self):
        """Train LSTM model with early stopping and checkpointing."""
        print(f"\n{'='*60}")
        print(f"Training LSTM for {self.metric_name}")
        print(f"{'='*60}")

        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            self.X_train, self.y_train_dict,
            self.X_val, self.y_val_dict,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers
        )

        # Setup callbacks
        early_stop = EarlyStopping(
            patience=self.config.training.patience,
            min_delta=self.config.training.min_delta,
            mode='min'
        )

        checkpoint = ModelCheckpoint(
            checkpoint_dir=os.path.join(
                self.config.training.checkpoint_dir,
                self.metric_name
            ),
            metric_name='val_loss',
            mode='min',
            save_every=self.config.training.save_every_n_epochs
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
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)

            # Log to MLflow
            if self.use_mlflow:
                self.mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, step=epoch)

            # Print progress
            if epoch % self.config.training.log_interval == 0 or epoch == 0:
                print(f"\nEpoch {epoch+1}/{self.config.training.epochs}")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss: {val_loss:.6f}")

            # Checkpoint
            checkpoint(self.model, self.optimizer, epoch, val_loss, additional_info={
                'X_scaler': self.X_scaler,
                'y_scaler': self.y_scaler,
                'config': self.config,
            })

            # Early stopping
            if early_stop(val_loss, epoch):
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Load best model
        checkpoint.load_best_model(self.model, self.optimizer)

        # End MLflow run
        if self.use_mlflow:
            self.mlflow.end_run()

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")

    def train_statistical(self):
        """Train statistical model (ARIMA or Prophet)."""
        print(f"\n{'='*60}")
        print(f"Training {self.model_type.upper()} for {self.metric_name}")
        print(f"{'='*60}")

        # For statistical models, use the raw sequence data
        # We'll use only the first feature if multivariate
        if self.X_train.ndim == 3:
            # Extract last value from each window as the current state
            y_sequence = self.X_train[:, -1, 0]  # Use last timestep, first feature
        else:
            y_sequence = self.X_train[:, 0]

        # Fit model
        print(f"\nFitting {self.model_type} model...")
        if self.model_type == 'prophet':
            # Prophet needs timestamps
            timestamps = pd.date_range(
                start='2024-01-01',
                periods=len(y_sequence),
                freq='15S'
            )
            self.model.fit(y_sequence, timestamps=timestamps)
        else:  # ARIMA
            self.model.fit(y_sequence)

        print("Model fitted successfully!")

        # Save model
        save_dir = os.path.join(self.config.training.checkpoint_dir, self.metric_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{self.model_type}_model.pkl')
        self.model.save_model(save_path)
        print(f"Model saved to {save_path}")

    def train(self):
        """Main training entry point."""
        self.build_model()

        if self.model_type == 'lstm':
            self.train_lstm()
        else:
            self.train_statistical()

    def evaluate(self, X_test: Optional[np.ndarray] = None,
                y_test_dict: Optional[Dict] = None) -> Dict:
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

        print(f"\n{'='*60}")
        print(f"Evaluating {self.model_type.upper()} on {self.metric_name}")
        print(f"{'='*60}")

        if self.model_type == 'lstm':
            # LSTM evaluation
            self.model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(self.device)
                predictions = self.model.predict_all_horizons(X_test_tensor)
                y_pred_dict = {h: pred.cpu().numpy() for h, pred in predictions.items()}

            # Denormalize predictions
            for horizon in y_pred_dict.keys():
                y_pred_dict[horizon] = self.y_scaler[horizon].inverse_transform(
                    y_pred_dict[horizon].reshape(-1, 1)).reshape(y_pred_dict[horizon].shape)
                y_test_dict[horizon] = self.y_scaler[horizon].inverse_transform(
                    y_test_dict[horizon].reshape(-1, 1)).reshape(y_test_dict[horizon].shape)

        else:
            # Statistical model evaluation
            horizons = sorted(self.config.data.prediction_horizons)
            y_pred_dict = self.model.predict_multi_horizon(horizons)

        # Compute metrics
        evaluator = ModelEvaluator(metric_name=self.metric_name)
        results = evaluator.horizon_analysis(y_test_dict, y_pred_dict)
        evaluator.print_evaluation_report(y_test_dict, y_pred_dict)

        return results


if __name__ == '__main__':
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
    config = create_default_config('cpu', model_type='lstm')
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
