"""
LSTM Time Series Model
LSTM architecture with multi-horizon prediction heads for container metrics.
"""

import torch
import torch.nn as nn
from typing import Optional
from src.models.base_model import BaseTimeSeriesModel


class LSTMPredictor(BaseTimeSeriesModel):
    """
    LSTM model for time series prediction with multiple prediction horizons.

    Supports:
    - Bidirectional LSTM
    - Multiple prediction horizons (5min, 15min, 30min)
    - Dropout regularization
    - Flexible input features
    """

    def __init__(self, config):
        """
        Initialize LSTM model.

        Args:
            config: ModelConfig object with architecture parameters
        """
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.bidirectional = config.bidirectional
        self.prediction_horizons = getattr(config, "prediction_horizons", [20, 60, 120])

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        # Calculate LSTM output size
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)

        # Dropout layer after LSTM
        self.post_lstm_dropout = nn.Dropout(self.dropout)

        # Multi-horizon prediction heads
        # Each horizon gets its own linear layer for prediction
        self.prediction_heads = nn.ModuleDict(
            {
                f"horizon_{h}": nn.Sequential(
                    nn.Linear(lstm_output_size, lstm_output_size // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(lstm_output_size // 2, h),
                )
                for h in self.prediction_horizons
            }
        )

    def forward(self, x):
        """
        Forward pass through LSTM.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            LSTM features from last timestep (batch_size, hidden_size * directions)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use the output from the last timestep
        # lstm_out shape: (batch_size, seq_len, hidden_size * directions)
        features = lstm_out[:, -1, :]  # (batch_size, hidden_size * directions)

        # Apply dropout
        features = self.post_lstm_dropout(features)

        return features

    def predict(self, x, horizon: int):
        """
        Predict for a specific time horizon.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            horizon: Number of timesteps to predict (must be in prediction_horizons)

        Returns:
            Predictions of shape (batch_size, horizon)
        """
        if horizon not in self.prediction_horizons:
            raise ValueError(
                f"Horizon {horizon} not in configured horizons: "
                f"{self.prediction_horizons}"
            )

        # Get LSTM features
        features = self.forward(x)

        # Apply horizon-specific prediction head
        predictions = self.prediction_heads[f"horizon_{horizon}"](features)

        return predictions

    def predict_all_horizons(self, x):
        """
        Predict for all configured horizons.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Dictionary mapping horizon to predictions
            {20: (batch, 20), 60: (batch, 60), 120: (batch, 120)}
        """
        # Get LSTM features once
        features = self.forward(x)

        # Apply each prediction head
        predictions = {}
        for horizon in self.prediction_horizons:
            predictions[horizon] = self.prediction_heads[f"horizon_{horizon}"](features)

        return predictions


class SimpleLSTM(BaseTimeSeriesModel):
    """
    Simplified LSTM for single-horizon prediction.
    Useful for baseline comparisons or when only one horizon is needed.
    """

    def __init__(self, config, prediction_horizon: int = 60):
        """
        Initialize simple LSTM.

        Args:
            config: ModelConfig object
            prediction_horizon: Single horizon to predict
        """
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.bidirectional = config.bidirectional
        self.prediction_horizon = prediction_horizon

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        # Prediction head
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, self.prediction_horizon)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, sequence_length, input_size)

        Returns:
            Predictions (batch_size, prediction_horizon)
        """
        lstm_out, _ = self.lstm(x)
        features = lstm_out[:, -1, :]  # Last timestep
        predictions = self.fc(features)
        return predictions

    def predict(self, x, horizon: Optional[int] = None):
        """
        Make predictions.

        Args:
            x: Input tensor
            horizon: Ignored (model has fixed horizon)

        Returns:
            Predictions for the fixed horizon
        """
        return self.forward(x)


def create_lstm_model(config) -> BaseTimeSeriesModel:
    """
    Factory function to create LSTM model based on configuration.

    Args:
        config: Configuration object

    Returns:
        LSTM model instance
    """
    # Check if multi-horizon prediction is needed
    if hasattr(config, "prediction_horizons") and len(config.prediction_horizons) > 1:
        return LSTMPredictor(config)
    else:
        horizon = getattr(config, "prediction_horizon", 60)
        return SimpleLSTM(config, prediction_horizon=horizon)


if __name__ == "__main__":
    # Example usage and testing
    from src.config.base_config import ModelConfig, DataConfig

    print("LSTM Time Series Model")
    print("=" * 60)

    # Create config
    model_config = ModelConfig(
        model_type="lstm",
        input_size=8,  # e.g., value + 7 features
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
    )

    data_config = DataConfig(prediction_horizons=[20, 60, 120])  # 5, 15, 30 minutes

    # Add horizons to model config
    model_config.prediction_horizons = data_config.prediction_horizons

    # Create model
    model = LSTMPredictor(model_config)

    print("\nModel Info:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test forward pass
    batch_size = 16
    sequence_length = 240  # 60 minutes
    input_size = 8

    dummy_input = torch.randn(batch_size, sequence_length, input_size)

    print("\nTest Forward Pass:")
    print(f"  Input shape: {dummy_input.shape}")

    # Test single horizon prediction
    horizon = 60
    predictions = model.predict(dummy_input, horizon)
    print(f"  Predictions for horizon {horizon}: {predictions.shape}")

    # Test all horizons
    all_predictions = model.predict_all_horizons(dummy_input)
    print("\n  All Horizons:")
    for h, pred in all_predictions.items():
        print(f"    Horizon {h}: {pred.shape}")

    print("\nModel created successfully!")
