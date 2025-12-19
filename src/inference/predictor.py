"""
Multi-Metric Predictor
Load and run inference with trained models for all metrics.
"""

import torch
import numpy as np
from typing import Dict
import os

from src.models.lstm.lstm_model import LSTMPredictor
from src.models.statistical.arima_model import ARIMAPredictor
from src.models.statistical.prophet_model import ProphetPredictor


class SingleMetricPredictor:
    """
    Predictor for a single metric with model loading and inference.
    """

    def __init__(
        self,
        metric_name: str,
        model_type: str = "lstm",
        checkpoint_dir: str = "experiments/checkpoints",
    ):
        """
        Initialize predictor.

        Args:
            metric_name: Name of metric (cpu, memory, etc.)
            model_type: Type of model (lstm, arima, prophet)
            checkpoint_dir: Directory containing model checkpoints
        """
        self.metric_name = metric_name
        self.model_type = model_type
        self.checkpoint_dir = os.path.join(checkpoint_dir, metric_name)

        self.model = None
        self.X_scaler = None
        self.y_scaler = {}
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_model()

    def _load_model(self):
        """Load trained model from checkpoint."""
        if self.model_type == "lstm":
            self._load_lstm()
        elif self.model_type in ["arima", "prophet"]:
            self._load_statistical()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _load_lstm(self):
        """Load LSTM model from checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        print(f"Loading LSTM model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load config
        self.config = checkpoint.get("config")

        # Initialize model
        self.model = LSTMPredictor(self.config.model)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load scalers
        self.X_scaler = checkpoint.get("X_scaler")
        self.y_scaler = checkpoint.get("y_scaler", {})

        print("  Model loaded successfully")
        print(f"  Best validation loss: {checkpoint.get('val_loss', 'N/A')}")

    def _load_statistical(self):
        """Load statistical model (ARIMA or Prophet)."""
        model_path = os.path.join(self.checkpoint_dir, f"{self.model_type}_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}")

        print(f"Loading {self.model_type.upper()} model from {model_path}")

        if self.model_type == "arima":
            self.model = ARIMAPredictor.load_model(model_path)
        else:  # prophet
            self.model = ProphetPredictor.load_model(model_path)

        print("  Model loaded successfully")

    def predict(self, X: np.ndarray, horizon: int = 60) -> np.ndarray:
        """
        Make predictions for a specific horizon.

        Args:
            X: Input sequences of shape (n_samples, window_size, n_features)
            horizon: Prediction horizon in timesteps

        Returns:
            Predictions of shape (n_samples, horizon)
        """
        if self.model_type == "lstm":
            return self._predict_lstm(X, horizon)
        else:
            return self._predict_statistical(X, horizon)

    def _predict_lstm(self, X: np.ndarray, horizon: int) -> np.ndarray:
        """Make predictions with LSTM model."""
        # Normalize input
        if self.X_scaler:
            X_norm = self.X_scaler.transform(X)
        else:
            X_norm = X

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_norm).to(self.device)

        # Predict
        with torch.no_grad():
            predictions = self.model.predict(X_tensor, horizon)
            predictions_np = predictions.cpu().numpy()

        # Denormalize
        if horizon in self.y_scaler:
            predictions_np = (
                self.y_scaler[horizon]
                .inverse_transform(predictions_np.reshape(-1, 1))
                .reshape(predictions_np.shape)
            )

        return predictions_np

    def _predict_statistical(self, X: np.ndarray, horizon: int) -> np.ndarray:
        """Make predictions with statistical model."""
        # For statistical models, we predict from the current state
        # Multiple predictions for multiple samples
        n_samples = X.shape[0]
        predictions = []

        for i in range(n_samples):
            # Predict
            pred = self.model.predict(horizon)

            predictions.append(pred)

        return np.array(predictions)

    def predict_all_horizons(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Make predictions for all configured horizons.

        Args:
            X: Input sequences

        Returns:
            Dict mapping horizon to predictions
        """
        if self.model_type == "lstm":
            # LSTM can predict all horizons at once
            # Normalize
            X_norm = self.X_scaler.transform(X) if self.X_scaler else X
            X_tensor = torch.FloatTensor(X_norm).to(self.device)

            with torch.no_grad():
                predictions_dict = self.model.predict_all_horizons(X_tensor)
                predictions_np = {
                    h: pred.cpu().numpy() for h, pred in predictions_dict.items()
                }

            # Denormalize
            for h in predictions_np.keys():
                if h in self.y_scaler:
                    predictions_np[h] = (
                        self.y_scaler[h]
                        .inverse_transform(predictions_np[h].reshape(-1, 1))
                        .reshape(predictions_np[h].shape)
                    )

            return predictions_np
        else:
            # Statistical models predict sequentially
            default_horizons = [20, 60, 120]
            return {h: self.predict(X, h) for h in default_horizons}


class MultiMetricPredictor:
    """
    Manage predictors for multiple metrics.

    Allows predicting all container metrics at once.
    """

    def __init__(self, checkpoint_dir: str = "experiments/checkpoints"):
        """
        Initialize multi-metric predictor.

        Args:
            checkpoint_dir: Base directory for model checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.predictors = {}

    def load_metric(self, metric_name: str, model_type: str = "lstm"):
        """
        Load a specific metric's model.

        Args:
            metric_name: Metric to load
            model_type: Model type for this metric
        """
        try:
            predictor = SingleMetricPredictor(
                metric_name=metric_name,
                model_type=model_type,
                checkpoint_dir=self.checkpoint_dir,
            )
            self.predictors[metric_name] = predictor
            print(f"Loaded {metric_name} predictor")
        except Exception as e:
            print(f"Failed to load {metric_name}: {e}")

    def load_all_metrics(self, model_type: str = "lstm"):
        """
        Load models for all available metrics.

        Args:
            model_type: Model type to use for all metrics
        """
        metrics = [
            "cpu",
            "memory",
            "disk_reads",
            "disk_writes",
            "network_rx",
            "network_tx",
        ]

        for metric in metrics:
            self.load_metric(metric, model_type)

    def predict(self, metric_name: str, X: np.ndarray, horizon: int = 60) -> np.ndarray:
        """
        Predict for a specific metric.

        Args:
            metric_name: Metric to predict
            X: Input sequences
            horizon: Prediction horizon

        Returns:
            Predictions
        """
        if metric_name not in self.predictors:
            raise ValueError(f"Metric {metric_name} not loaded")

        return self.predictors[metric_name].predict(X, horizon)

    def predict_all_metrics(
        self, X_dict: Dict[str, np.ndarray], horizon: int = 60
    ) -> Dict[str, np.ndarray]:
        """
        Predict for all loaded metrics.

        Args:
            X_dict: Dict mapping metric name to input sequences
            horizon: Prediction horizon

        Returns:
            Dict mapping metric name to predictions
        """
        predictions = {}

        for metric_name in self.predictors.keys():
            if metric_name in X_dict:
                predictions[metric_name] = self.predict(
                    metric_name, X_dict[metric_name], horizon
                )

        return predictions


if __name__ == "__main__":
    # Test predictor
    print("Multi-Metric Predictor")
    print("=" * 60)

    # Note: This test assumes models have been trained
    # For actual testing, train models first using train_local.py

    # Create multi-metric predictor
    predictor = MultiMetricPredictor()

    # Check what models are available
    checkpoint_dir = "experiments/checkpoints"
    if os.path.exists(checkpoint_dir):
        available_metrics = [
            d
            for d in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, d))
        ]
        print(f"\nAvailable trained models: {available_metrics}")

        if available_metrics:
            # Load first available metric
            metric = available_metrics[0]
            print(f"\nTesting with {metric} model...")

            try:
                predictor.load_metric(metric, model_type="lstm")

                # Create dummy input
                window_size = 240
                n_features = 8
                X_test = np.random.randn(5, window_size, n_features)

                # Predict
                predictions = predictor.predict(metric, X_test, horizon=60)
                print(f"\nPredictions shape: {predictions.shape}")
                print(f"Sample prediction: {predictions[0, :5]}")

                print("\nPredictor test successful!")
            except Exception as e:
                print(f"\nError testing predictor: {e}")
                print("Make sure models are trained first using train_local.py")
        else:
            print("\nNo trained models found.")
            print("Train models first using:")
            print("  python scripts/train_local.py --metric cpu --model-type lstm")
    else:
        print(f"\nCheckpoint directory not found: {checkpoint_dir}")
        print("Train models first!")
